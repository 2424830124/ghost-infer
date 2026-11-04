//! The SNAC 24 kHz decoder on the CPU: the codes maya1 (and Orpheus) speak in to audio. A frame is 7 codes:
//! one of the coarsest level, two of the middle, four of the finest, which make four latent steps (each code
//! held for 4, 2 and 1 steps, its row of snac.gguf's tables added). A 1x1 and a depthwise convolution widen
//! the latent to 1024 channels; four blocks upsample it by 8 · 8 · 4 · 2 (a transposed convolution, a little
//! noise, three residual units each), and a last convolution makes the samples: 2048 a frame (85 ms).
//! The convolutions look both ways, so a stream (`SnacStream`) decodes each stretch with a few frames of
//! context on either side and keeps the middle. scripts/snac-convert.py writes snac.gguf.

use rayon::prelude::*;

use crate::gguf::{Gguf, Value};
use crate::quant::Matrix;

pub const CODES_PER_FRAME: usize = 7;

fn floats(g: &Gguf, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let t = g.tensor(name).ok_or_else(|| format!("snac.gguf: missing tensor {name}"))?;
    let n = t.n_elements() as usize;
    let mut v = vec![0f32; n];
    Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, rows: 1, cols: n }.row_f32(0, &mut v);
    // torch order (outermost first)
    Ok((v, t.dims.iter().rev().map(|&d| d as usize).collect()))
}

/// Activations are channels of `t` samples each: x[c * t + i].
fn snake(x: &mut [f32], alpha: &[f32], t: usize) {
    x.par_chunks_mut(t).zip(alpha.par_iter()).for_each(|(row, &a)| {
        let inv = 1.0 / (a + 1e-9);
        for v in row {
            let s = (a * *v).sin();
            *v += inv * s * s;
        }
    });
}

/// A 1x1 convolution: out[o] = b[o] + Σ_i w[o][i] · x[i], over `t` steps, in tiles of steps that stay in cache.
fn pointwise(w: &[f32], b: Option<&[f32]>, cin: usize, x: &[f32], t: usize) -> Vec<f32> {
    const TILE: usize = 512;
    let cout = w.len() / cin;
    let mut out = vec![0f32; cout * t];
    out.par_chunks_mut(t).enumerate().for_each(|(o, row)| {
        let wo = &w[o * cin..(o + 1) * cin];
        let bias = b.map_or(0.0, |b| b[o]);
        for (k, tile) in row.chunks_mut(TILE).enumerate() {
            let at = k * TILE;
            tile.fill(bias);
            for (i, &wi) in wo.iter().enumerate() {
                let xs = &x[i * t + at..i * t + at + tile.len()];
                for (y, &xv) in tile.iter_mut().zip(xs) {
                    *y += wi * xv;
                }
            }
        }
    });
    out
}

/// A depthwise convolution of 7 taps, `dil` apart, centred ("same" padding).
fn depthwise(w: &[f32], b: &[f32], dil: usize, x: &[f32], t: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    out.par_chunks_mut(t).enumerate().for_each(|(c, row)| {
        let (xs, wc) = (&x[c * t..(c + 1) * t], &w[c * 7..c * 7 + 7]);
        row.fill(b[c]);
        for (j, &wj) in wc.iter().enumerate() {
            let off = (j as isize - 3) * dil as isize;
            let (from, to) = ((-off).max(0) as usize, (t as isize - off.max(0)).max(0) as usize);
            for i in from..to.max(from) {
                row[i] += wj * xs[(i as isize + off) as usize];
            }
        }
    });
    out
}

/// A transposed convolution of stride s, 2s taps and padding s/2 (s even): `t` steps in, t·s out. Output
/// step u = t0·s + j - s/2 takes tap j from input t0 and tap j + s from input t0 - 1.
fn upsample(w: &[f32], b: &[f32], cin: usize, cout: usize, s: usize, x: &[f32], t: usize) -> Vec<f32> {
    let (k, p) = (2 * s, s / 2);
    let mut out = vec![0f32; cout * t * s];
    out.par_chunks_mut(t * s).enumerate().for_each(|(o, row)| {
        // acc[j][t0] for t0 in 0..=t
        let mut acc = vec![0f32; s * (t + 1)];
        for i in 0..cin {
            let wi = &w[(i * cout + o) * k..(i * cout + o) * k + k];
            let xs = &x[i * t..(i + 1) * t];
            for j in 0..s {
                let (a, b2) = (wi[j], wi[j + s]);
                let row = &mut acc[j * (t + 1)..(j + 1) * (t + 1)];
                for (t0, r) in row[..t].iter_mut().enumerate() {
                    *r += a * xs[t0];
                }
                for (t0, r) in row[1..].iter_mut().enumerate() {
                    *r += b2 * xs[t0];
                }
            }
        }
        for j in 0..s {
            for t0 in 0..=t {
                let u = (t0 * s + j) as isize - p as isize;
                if u >= 0 && (u as usize) < t * s {
                    row[u as usize] = b[o] + acc[j * (t + 1) + t0];
                }
            }
        }
    });
    out
}

/// The same normal draw for the same step of the same block every time (SplitMix64 and Box-Muller), so a
/// stretch decoded twice, or in pieces, hears the same noise.
fn noise(block: usize, step: u64) -> f32 {
    let mix = |mut z: u64| {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    let h = mix(step.wrapping_mul(0x9e3779b97f4a7c15) ^ ((block as u64 + 1) << 56));
    let u1 = ((h >> 40) as f32 + 0.5) / (1u64 << 24) as f32;
    let u2 = ((mix(h) >> 40) as f32) / (1u64 << 24) as f32;
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

struct Residual {
    snake1: Vec<f32>,
    dw: (Vec<f32>, Vec<f32>),
    snake2: Vec<f32>,
    pw: (Vec<f32>, Vec<f32>),
    dil: usize,
}

struct Block {
    snake: Vec<f32>,
    up: (Vec<f32>, Vec<f32>),
    noise: Vec<f32>,
    res: Vec<Residual>,
    cin: usize,
    cout: usize,
    stride: usize,
}

pub struct Snac {
    pub sample_rate: usize,
    latent: usize,
    tables: Vec<Vec<f32>>,
    in_dw: (Vec<f32>, Vec<f32>),
    in_pw: (Vec<f32>, Vec<f32>),
    blocks: Vec<Block>,
    out_snake: Vec<f32>,
    out: (Vec<f32>, f32),
    /// the noise blocks (on, as SNAC decodes; GHOST_INFER_SNAC_NOISE=0 leaves them out, as snac-check does)
    pub noise: bool,
}

/// A frame's codes, each already the code (0..4096) of its level: [l1, l2a, l3a, l3b, l2b, l3c, l3d], the
/// order maya1 writes them in.
pub type Frame = [u32; CODES_PER_FRAME];

/// The samples a frame decodes to.
pub const FRAME_SAMPLES: usize = 2048;

impl Snac {
    pub fn load(g: &Gguf) -> Result<Snac, String> {
        let u = |k: &str| g.get(k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| format!("snac.gguf: missing {k}"));
        let pair = |n: &str| -> Result<(Vec<f32>, Vec<f32>), String> { Ok((floats(g, &format!("{n}.weight"))?.0, floats(g, &format!("{n}.bias"))?.0)) };
        let latent = u("snac.latent_dim")?;
        let tables = (0..3).map(|i| floats(g, &format!("q{i}.table")).map(|t| t.0)).collect::<Result<Vec<_>, _>>()?;
        let mut blocks = Vec::new();
        for b in 0.. {
            if g.tensor(&format!("b{b}.snake")).is_none() {
                break;
            }
            let (up_w, dims) = floats(g, &format!("b{b}.up.weight"))?;
            let (cin, cout, k) = (dims[0], dims[1], dims[2]);
            let res = (0..3)
                .map(|r| {
                    Ok(Residual {
                        snake1: floats(g, &format!("b{b}.r{r}.snake1"))?.0,
                        dw: pair(&format!("b{b}.r{r}.dw"))?,
                        snake2: floats(g, &format!("b{b}.r{r}.snake2"))?.0,
                        pw: pair(&format!("b{b}.r{r}.pw"))?,
                        dil: [1, 3, 9][r],
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            blocks.push(Block {
                snake: floats(g, &format!("b{b}.snake"))?.0,
                up: (up_w, floats(g, &format!("b{b}.up.bias"))?.0),
                noise: floats(g, &format!("b{b}.noise.weight"))?.0,
                res,
                cin,
                cout,
                stride: k / 2,
            });
        }
        if blocks.iter().any(|b| b.stride % 2 == 1) {
            return Err("snac.gguf: an odd upsampling stride is not supported".into());
        }
        let (ow, ob) = pair("out")?;
        Ok(Snac {
            sample_rate: u("snac.sample_rate")?,
            latent,
            tables,
            in_dw: pair("in.dw")?,
            in_pw: pair("in.pw")?,
            blocks,
            out_snake: floats(g, "out.snake")?.0,
            out: (ow, ob[0]),
            noise: std::env::var("GHOST_INFER_SNAC_NOISE").map_or(true, |v| v != "0"),
        })
    }

    /// The audio of `frames`, decoded as one stretch; `first` is the first frame's place in the whole
    /// utterance (for the noise).
    pub fn decode(&self, frames: &[Frame], first: u64) -> Vec<f32> {
        let t = frames.len() * 4;
        // the latent: each level's rows held for 4, 2 and 1 steps
        let mut x = vec![0f32; self.latent * t];
        for (f, codes) in frames.iter().enumerate() {
            let rows = [(0, codes[0], 4, 0), (1, codes[1], 2, 0), (1, codes[4], 2, 2), (2, codes[2], 1, 0), (2, codes[3], 1, 1), (2, codes[5], 1, 2), (2, codes[6], 1, 3)];
            for (level, code, hold, at) in rows {
                let row = &self.tables[level][code as usize * self.latent..(code as usize + 1) * self.latent];
                for (c, &v) in row.iter().enumerate() {
                    for k in 0..hold {
                        x[c * t + f * 4 + at + k] += v;
                    }
                }
            }
        }
        let mut x = depthwise(&self.in_dw.0, &self.in_dw.1, 1, &x, t);
        x = pointwise(&self.in_pw.0, Some(&self.in_pw.1), self.latent, &x, t);
        let (mut t, mut step0) = (t, first * 4);
        for (bi, b) in self.blocks.iter().enumerate() {
            snake(&mut x, &b.snake, t);
            x = upsample(&b.up.0, &b.up.1, b.cin, b.cout, b.stride, &x, t);
            t *= b.stride;
            step0 *= b.stride as u64;
            if self.noise {
                let h = pointwise(&b.noise, None, b.cout, &x, t);
                let n: Vec<f32> = (0..t).map(|i| noise(bi, step0 + i as u64)).collect();
                x.par_chunks_mut(t).zip(h.par_chunks(t)).for_each(|(row, hr)| row.iter_mut().zip(hr).zip(&n).for_each(|((v, h), n)| *v += n * h));
            }
            for r in &b.res {
                let mut y = x.clone();
                snake(&mut y, &r.snake1, t);
                y = depthwise(&r.dw.0, &r.dw.1, r.dil, &y, t);
                snake(&mut y, &r.snake2, t);
                y = pointwise(&r.pw.0, Some(&r.pw.1), b.cout, &y, t);
                x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            }
        }
        snake(&mut x, &self.out_snake, t);
        let cin = self.out_snake.len();
        (0..t)
            .into_par_iter()
            .map(|i| {
                let mut v = self.out.1;
                for c in 0..cin {
                    for j in 0..7 {
                        let at = i as isize + j as isize - 3;
                        if at >= 0 && (at as usize) < t {
                            v += self.out.0[c * 7 + j] * x[c * t + at as usize];
                        }
                    }
                }
                v.tanh()
            })
            .collect()
    }
}

/// Frames of context decoded before and after a stretch (the decoder sees about ±2.5 frames), and the frames
/// a stretch holds: a stream's audio trails its frames by AFTER + STRETCH frames at most (~600 ms).
const BEFORE: usize = 3;
const AFTER: usize = 3;
const STRETCH: usize = 4;

/// An utterance decoded as its frames come: each stretch with context on both sides, so the pieces join
/// as if decoded at once.
pub struct SnacStream<'a> {
    snac: &'a Snac,
    frames: Vec<Frame>,
    done: usize,
}

impl<'a> SnacStream<'a> {
    pub fn new(snac: &'a Snac) -> SnacStream<'a> {
        SnacStream { snac, frames: Vec::new(), done: 0 }
    }

    /// Adds a frame; the audio now ready (empty until a stretch is).
    pub fn push(&mut self, f: Frame) -> Vec<f32> {
        self.frames.push(f);
        if self.frames.len() >= self.done + STRETCH + AFTER {
            self.stretch(self.done + STRETCH)
        } else {
            Vec::new()
        }
    }

    /// The rest of the audio, at the end of the utterance.
    pub fn finish(&mut self) -> Vec<f32> {
        if self.done < self.frames.len() {
            self.stretch(self.frames.len())
        } else {
            Vec::new()
        }
    }

    /// Frames done..until, decoded with up to BEFORE frames before them and AFTER after.
    fn stretch(&mut self, until: usize) -> Vec<f32> {
        let from = self.done.saturating_sub(BEFORE);
        let to = (until + AFTER).min(self.frames.len());
        let audio = self.snac.decode(&self.frames[from..to], from as u64);
        let keep = &audio[(self.done - from) * FRAME_SAMPLES..(until - from) * FRAME_SAMPLES];
        self.done = until;
        keep.to_vec()
    }
}

/// maya1's audio tokens to frames: a token is 128266 + 4096 · (its place in the frame) + code.
pub const MAYA1_CODE_BASE: u32 = 128266;

pub fn frame_of(tokens: &[u32]) -> Option<Frame> {
    if tokens.len() != CODES_PER_FRAME {
        return None;
    }
    let mut f = [0u32; CODES_PER_FRAME];
    for (k, &t) in tokens.iter().enumerate() {
        let c = t.checked_sub(MAYA1_CODE_BASE + 4096 * k as u32)?;
        if c >= 4096 {
            return None;
        }
        f[k] = c;
    }
    Some(f)
}

/// Decodes REF.json's codes (from scripts/snac-convert.py --ref, noise off) at once and as a stream, and
/// compares both with the reference decoder's audio.
pub fn check(path: &std::path::Path, reference: &std::path::Path) -> Result<(), String> {
    use crate::json::Json;
    let mut snac = Snac::load(&Gguf::open(path)?)?;
    snac.noise = false;
    let j = Json::parse(&std::fs::read_to_string(reference).map_err(|e| e.to_string())?)?;
    let level = |i: usize| -> Vec<u32> { j.get("codes").and_then(|c| c.as_array()).and_then(|c| c.get(i)).and_then(|l| l.as_array()).map(|l| l.iter().filter_map(Json::as_f64).map(|v| v as u32).collect()).unwrap_or_default() };
    let (l1, l2, l3) = (level(0), level(1), level(2));
    let want: Vec<f32> = j.get("audio").and_then(|a| a.as_array()).map(|a| a.iter().filter_map(Json::as_f64).map(|v| v as f32).collect()).unwrap_or_default();
    let frames: Vec<Frame> = (0..l1.len()).map(|i| [l1[i], l2[2 * i], l3[4 * i], l3[4 * i + 1], l2[2 * i + 1], l3[4 * i + 2], l3[4 * i + 3]]).collect();
    let t = std::time::Instant::now();
    let whole = snac.decode(&frames, 0);
    let secs = t.elapsed().as_secs_f64();
    let mut st = SnacStream::new(&snac);
    let mut streamed: Vec<f32> = frames.iter().flat_map(|&f| st.push(f)).collect();
    streamed.extend(st.finish());
    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len().max(1) as f32).sqrt();
    println!("{} frames, {} samples (reference {}), rms {rms:.4}", frames.len(), whole.len(), want.len());
    println!("whole:  max difference {:.2e}, decoded in {:.0} ms ({:.1}x real time)", diff(&whole, &want), secs * 1000.0, whole.len() as f64 / snac.sample_rate as f64 / secs);
    println!("stream: max difference {:.2e} from the reference, {:.2e} from whole", diff(&streamed, &want), diff(&streamed, &whole));
    if whole.len() != want.len() || diff(&whole, &want) > 1e-3 {
        return Err("the decoder does not match the reference".into());
    }
    Ok(())
}
