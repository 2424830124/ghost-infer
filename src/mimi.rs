//! Kyutai's Mimi audio decoder on the CPU: CSM's frames (32 codes per 80 ms) to 24 kHz audio, a few frames at
//! a time. A frame's codes are summed into one 512-wide vector (mimi.gguf keeps every codebook's rows already
//! through its quantizer's output projection), upsampled to 25 steps a second, run through an 8-layer
//! transformer, and SEANet's convolutions upsample that by 8 · 6 · 5 · 4 to samples. Every layer is causal, so
//! decoding frame by frame with a little state (`MimiState`) gives what decoding them all at once does.

use rayon::prelude::*;

use crate::gguf::{Gguf, Value};
use crate::quant::{dot_f32, Matrix};

/// Output values computed per rayon task.
const GRAIN: usize = 64;

fn floats(g: &Gguf, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let t = g.tensor(name).ok_or_else(|| format!("mimi.gguf: missing tensor {name}"))?;
    let n = t.n_elements() as usize;
    let mut v = vec![0f32; n];
    Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, rows: 1, cols: n }.row_f32(0, &mut v);
    // torch order (outermost first)
    Ok((v, t.dims.iter().rev().map(|&d| d as usize).collect()))
}

/// out[t][o] = Σ_i w[o][i] · x[t][i] (+ b[o]) for `x` of rows of `cols`.
fn linear(w: &[f32], b: Option<&[f32]>, cols: usize, x: &[f32]) -> Vec<f32> {
    let rows = w.len() / cols;
    let mut out = vec![0f32; x.len() / cols * rows];
    out.par_chunks_mut(GRAIN).enumerate().for_each(|(c, chunk)| {
        for (j, v) in chunk.iter_mut().enumerate() {
            let idx = c * GRAIN + j;
            let (t, o) = (idx / rows, idx % rows);
            *v = dot_f32(&w[o * cols..(o + 1) * cols], &x[t * cols..(t + 1) * cols]) + b.map_or(0.0, |b| b[o]);
        }
    });
    out
}

fn elu(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = v.exp_m1();
        }
    }
}

/// erf to about 1.5e-7 (Abramowitz and Stegun 7.1.26), for the exact GELU.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t * (-x * x).exp();
    y.copysign(x)
}

/// A causal convolution, stride 1: weights as [tap][out][in], so each tap is a matrix over the input channels.
struct Conv {
    cin: usize,
    cout: usize,
    k: usize,
    dil: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Conv {
    fn load(g: &Gguf, prefix: &str, dil: usize) -> Result<Conv, String> {
        let (w, d) = floats(g, &format!("{prefix}.conv.weight"))?;
        let [cout, cin, k] = d[..] else { return Err(format!("{prefix}: not a 1-D convolution")) };
        let mut taps = vec![0f32; w.len()];
        for o in 0..cout {
            for i in 0..cin {
                for j in 0..k {
                    taps[(j * cout + o) * cin + i] = w[(o * cin + i) * k + j];
                }
            }
        }
        let b = match g.tensor(&format!("{prefix}.conv.bias")) {
            Some(_) => floats(g, &format!("{prefix}.conv.bias"))?.0,
            None => vec![0.0; cout],
        };
        Ok(Conv { cin, cout, k, dil, w: taps, b })
    }

    /// Input rows it keeps from one call to the next.
    fn history(&self) -> usize {
        (self.k - 1) * self.dil
    }

    /// `x`: rows of `cin`; `state`: the `history()` input rows before them (zeros at the start), moved on.
    fn run(&self, state: &mut Vec<f32>, x: &[f32]) -> Vec<f32> {
        let (cin, cout) = (self.cin, self.cout);
        let mut buf = std::mem::take(state);
        buf.extend_from_slice(x);
        let mut out = vec![0f32; x.len() / cin * cout];
        out.par_chunks_mut(GRAIN).enumerate().for_each(|(c, chunk)| {
            for (j, v) in chunk.iter_mut().enumerate() {
                let idx = c * GRAIN + j;
                let (t, o) = (idx / cout, idx % cout);
                let mut acc = self.b[o];
                for tap in 0..self.k {
                    let r = t + tap * self.dil;
                    acc += dot_f32(&self.w[(tap * cout + o) * cin..][..cin], &buf[r * cin..(r + 1) * cin]);
                }
                *v = acc;
            }
        });
        *state = buf.split_off(buf.len() - self.history() * cin);
        out
    }
}

/// A causal transposed convolution (stride `s`), its right overhang trimmed: [tap][out][in] weights.
struct ConvTr {
    cin: usize,
    cout: usize,
    k: usize,
    s: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl ConvTr {
    fn load(g: &Gguf, prefix: &str, s: usize) -> Result<ConvTr, String> {
        let (w, d) = floats(g, &format!("{prefix}.conv.weight"))?;
        let [cin, cout, k] = d[..] else { return Err(format!("{prefix}: not a 1-D transposed convolution")) };
        let mut taps = vec![0f32; w.len()];
        for i in 0..cin {
            for o in 0..cout {
                for j in 0..k {
                    taps[(j * cout + o) * cin + i] = w[(i * cout + o) * k + j];
                }
            }
        }
        Ok(ConvTr { cin, cout, k, s, w: taps, b: floats(g, &format!("{prefix}.conv.bias"))?.0 })
    }

    /// `x`: rows of `cin`, giving `s` output rows each; `tail`: what earlier rows added to the rows after
    /// theirs (`k - s` rows, bias not yet added), moved on.
    fn run(&self, tail: &mut Vec<f32>, x: &[f32]) -> Vec<f32> {
        let (cin, cout, k, s) = (self.cin, self.cout, self.k, self.s);
        let t = x.len() / cin;
        let n = t * s + k - s;
        let mut y = vec![0f32; n * cout];
        y.par_chunks_mut(GRAIN).enumerate().for_each(|(c, chunk)| {
            for (j, v) in chunk.iter_mut().enumerate() {
                let idx = c * GRAIN + j;
                let (r, o) = (idx / cout, idx % cout);
                let mut acc = 0.0;
                // input row ti reaches output rows ti * s + tap
                let mut tap = r % s;
                while tap < k {
                    if r >= tap && (r - tap) / s < t {
                        let ti = (r - tap) / s;
                        acc += dot_f32(&self.w[(tap * cout + o) * cin..][..cin], &x[ti * cin..(ti + 1) * cin]);
                    }
                    tap += s;
                }
                *v = acc;
            }
        });
        for (a, b) in y.iter_mut().zip(tail.iter()) {
            *a += b;
        }
        *tail = y.split_off(t * s * cout);
        for row in y.chunks_exact_mut(cout) {
            for (v, b) in row.iter_mut().zip(&self.b) {
                *v += b;
            }
        }
        y
    }
}

/// The transposed convolution from 12.5 to 25 steps a second: one 4-tap kernel per channel, stride 2, no bias.
struct Upsample {
    ch: usize,
    w: Vec<f32>, // [channel][tap]
}

impl Upsample {
    const K: usize = 4;
    const S: usize = 2;

    fn run(&self, tail: &mut Vec<f32>, x: &[f32]) -> Vec<f32> {
        let ch = self.ch;
        let t = x.len() / ch;
        let mut y = vec![0f32; (t * Self::S + Self::K - Self::S) * ch];
        for ti in 0..t {
            for tap in 0..Self::K {
                let r = ti * Self::S + tap;
                for c in 0..ch {
                    y[r * ch + c] += self.w[c * Self::K + tap] * x[ti * ch + c];
                }
            }
        }
        for (a, b) in y.iter_mut().zip(tail.iter()) {
            *a += b;
        }
        *tail = y.split_off(t * Self::S * ch);
        y
    }
}

struct Layer {
    norm1: (Vec<f32>, Vec<f32>),
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    scale1: Vec<f32>,
    norm2: (Vec<f32>, Vec<f32>),
    fc1: Vec<f32>,
    fc2: Vec<f32>,
    scale2: Vec<f32>,
}

fn layer_norm(x: &[f32], (w, b): &(Vec<f32>, Vec<f32>), eps: f32) -> Vec<f32> {
    let d = w.len();
    let mut out = vec![0f32; x.len()];
    for (row, o) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            o[i] = (row[i] - mean) * inv * w[i] + b[i];
        }
    }
    out
}

/// One SEANet upsampling stage: ELU, the transposed convolution, then a residual block
/// (ELU, k-tap convolution to half the channels, ELU, 1-tap convolution back, added to its input).
struct Stage {
    up: ConvTr,
    res_a: Conv,
    res_b: Conv,
}

pub struct Mimi {
    pub sample_rate: usize,
    dim: usize,
    heads: usize,
    window: usize,
    rope_base: f32,
    eps: f32,
    codebooks: Vec<f32>, // (codebook, code) rows of `dim`
    n_codebooks: usize,
    codes: usize,
    upsample: Upsample,
    layers: Vec<Layer>,
    first: Conv,
    stages: Vec<Stage>,
    last: Conv,
}

/// What decoding the next frames needs from the earlier ones.
pub struct MimiState {
    up: Vec<f32>,
    /// per layer: the cached keys and values (RoPE applied) of the positions from `base` on
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    base: usize,
    pos: usize,
    first: Vec<f32>,
    stages: Vec<(Vec<f32>, Vec<f32>)>, // transposed convolution tail, residual convolution history
    last: Vec<f32>,
}

impl Mimi {
    pub fn load(g: &Gguf) -> Result<Mimi, String> {
        let u = |k: &str| g.get(&format!("mimi.{k}")).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| format!("mimi.gguf: missing mimi.{k}"));
        let f = |k: &str| g.get(&format!("mimi.{k}")).and_then(Value::as_f32).ok_or_else(|| format!("mimi.gguf: missing mimi.{k}"));
        let ratios: Vec<usize> = g.get("mimi.upsampling_ratios").and_then(Value::as_array).ok_or("mimi.gguf: missing mimi.upsampling_ratios")?.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect();
        let (codebooks, d) = floats(g, "quantizer.codebooks.weight")?;
        let dim = d[1];
        let codes = 2048;
        let (up, _) = floats(g, "upsample.conv.weight")?;
        let mut layers = Vec::new();
        for i in 0..u("block_count")? {
            let p = |s: &str| floats(g, &format!("decoder_transformer.layers.{i}.{s}")).map(|v| v.0);
            layers.push(Layer {
                norm1: (p("input_layernorm.weight")?, p("input_layernorm.bias")?),
                q: p("self_attn.q_proj.weight")?,
                k: p("self_attn.k_proj.weight")?,
                v: p("self_attn.v_proj.weight")?,
                o: p("self_attn.o_proj.weight")?,
                scale1: p("self_attn_layer_scale.scale")?,
                norm2: (p("post_attention_layernorm.weight")?, p("post_attention_layernorm.bias")?),
                fc1: p("mlp.fc1.weight")?,
                fc2: p("mlp.fc2.weight")?,
                scale2: p("mlp_layer_scale.scale")?,
            });
        }
        // SEANet's layer list: a convolution, then per ratio [ELU, transposed convolution, residual block], ELU,
        // a convolution
        let mut stages = Vec::new();
        for (s, &r) in ratios.iter().enumerate() {
            let at = 1 + 3 * s;
            stages.push(Stage {
                up: ConvTr::load(g, &format!("decoder.layers.{}", at + 1), r)?,
                res_a: Conv::load(g, &format!("decoder.layers.{}.block.1", at + 2), 1)?,
                res_b: Conv::load(g, &format!("decoder.layers.{}.block.3", at + 2), 1)?,
            });
        }
        Ok(Mimi {
            sample_rate: u("sample_rate")?,
            dim,
            heads: u("head_count")?,
            window: u("sliding_window")?,
            rope_base: f("rope.freq_base")?,
            eps: f("norm_eps")?,
            n_codebooks: codebooks.len() / dim / codes,
            codebooks,
            codes,
            upsample: Upsample { ch: dim, w: up },
            layers,
            first: Conv::load(g, "decoder.layers.0", 1)?,
            last: Conv::load(g, &format!("decoder.layers.{}", 2 + 3 * ratios.len()), 1)?,
            stages,
        })
    }

    pub fn state(&self) -> MimiState {
        MimiState {
            up: vec![0.0; (Upsample::K - Upsample::S) * self.dim],
            kv: self.layers.iter().map(|_| (Vec::new(), Vec::new())).collect(),
            base: 0,
            pos: 0,
            first: vec![0.0; self.first.history() * self.first.cin],
            stages: self.stages.iter().map(|s| (vec![0.0; (s.up.k - s.up.s) * s.up.cout], vec![0.0; s.res_a.history() * s.res_a.cin])).collect(),
            last: vec![0.0; self.last.history() * self.last.cin],
        }
    }

    /// The audio of `frames` (each `n_codebooks` codes), following the frames `st` has seen.
    pub fn decode(&self, st: &mut MimiState, frames: &[Vec<u32>]) -> Vec<f32> {
        let d = self.dim;
        let mut x = vec![0f32; frames.len() * d];
        for (f, row) in frames.iter().zip(x.chunks_exact_mut(d)) {
            for (cb, &code) in f.iter().enumerate().take(self.n_codebooks) {
                let c = (code as usize).min(self.codes - 1); // CSM's 2048..2050 are not audio
                let e = &self.codebooks[(cb * self.codes + c) * d..][..d];
                row.iter_mut().zip(e).for_each(|(a, b)| *a += b);
            }
        }
        let mut x = self.upsample.run(&mut st.up, &x);
        self.transformer(st, &mut x);
        let mut x = self.first.run(&mut st.first, &x);
        for (stage, (tail, hist)) in self.stages.iter().zip(st.stages.iter_mut()) {
            elu(&mut x);
            x = stage.up.run(tail, &x);
            let mut h = x.clone();
            elu(&mut h);
            let mut h = stage.res_a.run(hist, &h);
            elu(&mut h);
            let h = stage.res_b.run(&mut Vec::new(), &h);
            x.iter_mut().zip(&h).for_each(|(a, b)| *a += b);
        }
        elu(&mut x);
        self.last.run(&mut st.last, &x)
    }

    fn transformer(&self, st: &mut MimiState, x: &mut [f32]) {
        let d = self.dim;
        let (heads, hd) = (self.heads, d / self.heads);
        let half = hd / 2;
        let t = x.len() / d;
        let inv: Vec<f32> = (0..half).map(|i| self.rope_base.powf(-(2.0 * i as f32) / hd as f32)).collect();
        // rotate-half RoPE, as the HF weights are trained
        let rope = |v: &mut [f32], pos: usize| {
            for h in 0..heads {
                let r = &mut v[h * hd..(h + 1) * hd];
                for i in 0..half {
                    let (s, c) = (pos as f32 * inv[i]).sin_cos();
                    let (a, b) = (r[i], r[i + half]);
                    r[i] = a * c - b * s;
                    r[i + half] = a * s + b * c;
                }
            }
        };
        let scale = 1.0 / (hd as f32).sqrt();
        for (layer, (kc, vc)) in self.layers.iter().zip(st.kv.iter_mut()) {
            let h = layer_norm(x, &layer.norm1, self.eps);
            let mut q = linear(&layer.q, None, d, &h);
            let mut k = linear(&layer.k, None, d, &h);
            let v = linear(&layer.v, None, d, &h);
            for ti in 0..t {
                rope(&mut q[ti * d..(ti + 1) * d], st.pos + ti);
                rope(&mut k[ti * d..(ti + 1) * d], st.pos + ti);
            }
            kc.extend_from_slice(&k);
            vc.extend_from_slice(&v);
            let mut att = vec![0f32; t * d];
            att.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
                let (ti, hh) = (idx / heads, idx % heads);
                let at = st.pos + ti - st.base; // this row in the cache
                let from = (at + 1).saturating_sub(self.window);
                let qv = &q[ti * d + hh * hd..][..hd];
                let scores: Vec<f32> = (from..=at).map(|p| dot_f32(qv, &kc[p * d + hh * hd..][..hd]) * scale).collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f32 = e.iter().sum();
                for (p, w) in (from..=at).zip(&e) {
                    let vv = &vc[p * d + hh * hd..][..hd];
                    out.iter_mut().zip(vv).for_each(|(o, v)| *o += w / total * v);
                }
            });
            let a = linear(&layer.o, None, d, &att);
            for (xr, (ar, s)) in x.iter_mut().zip(a.iter().zip(layer.scale1.iter().cycle())) {
                *xr += ar * s;
            }
            let h = layer_norm(x, &layer.norm2, self.eps);
            let mut m = linear(&layer.fc1, None, d, &h);
            for v in m.iter_mut() {
                *v = 0.5 * *v * (1.0 + erf(*v * std::f32::consts::FRAC_1_SQRT_2));
            }
            let m = linear(&layer.fc2, None, layer.fc1.len() / d, &m);
            for (xr, (mr, s)) in x.iter_mut().zip(m.iter().zip(layer.scale2.iter().cycle())) {
                *xr += mr * s;
            }
        }
        st.pos += t;
        // keep the window's positions only
        let keep = self.window.saturating_sub(1);
        let held = st.pos - st.base;
        if held > 2 * self.window {
            let drop = held - keep;
            for (kc, vc) in &mut st.kv {
                kc.drain(..drop * d);
                vc.drain(..drop * d);
            }
            st.base += drop;
        }
    }
}
