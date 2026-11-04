//! Sesame CSM speech: text (and the conversation before it) to 24 kHz audio. The model is a folder of three
//! GGUFs from scripts/csm-convert.py. The backbone, a 16-layer Llama, reads the conversation as frames (a
//! text token, or one Mimi frame's 32 codes as the sum of their rows) and picks each new frame's first code;
//! the depth decoder, a 4-layer Llama, picks the other 31 one after another from the backbone's hidden state;
//! Mimi (mimi.rs) turns the frames into audio. A frame runs on the GPU as one CUDA graph (cuda::CsmFrames);
//! the prompt's rows are looked up here.

use std::path::Path;

use crate::cuda::{CsmFrames, CsmGraph, GpuModel, GpuSession};
use crate::gguf::{Gguf, TensorInfo, Value};
use crate::mimi::Mimi;
use crate::model::Config;
use crate::quant::Matrix;
use crate::tokenizer::Tokenizer;

/// One turn spoken before the next: who said it, the words, and its frames (how it sounded).
#[derive(Clone)]
pub struct Turn {
    pub speaker: u32,
    pub text: String,
    pub frames: Vec<Vec<u32>>,
}

/// How codes are drawn: temperature (0: greedy) and top-k, as CSM was tuned (0.9, 50).
#[derive(Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub seed: u64,
}

pub struct Csm {
    bg: Gguf,
    dg: Gguf,
    backbone: GpuModel,
    decoder: GpuModel,
    frames: CsmFrames,
    text_embd: TensorInfo,
    audio_embd: TensorInfo,
    proj_embd: TensorInfo,
    pub tok: Tokenizer,
    pub codebooks: usize,
    /// codes per codebook in the transformers (Mimi's 2048 and a few specials)
    pub vocab: usize,
    eos: u32,
    pub mimi: Mimi,
}

/// A conversation's state on the GPU: the backbone's cache, the decoder's, and the captured frame.
pub struct CsmSession {
    bb: GpuSession,
    dec: GpuSession,
    graph: Option<CsmGraph>,
}

fn info(g: &Gguf, name: &str) -> Result<TensorInfo, String> {
    g.tensor(name).cloned().ok_or_else(|| format!("missing tensor {name}"))
}

fn table<'a>(g: &'a Gguf, t: &TensorInfo) -> Matrix<'a> {
    Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, cols: t.dims[0] as usize, rows: t.dims[1] as usize }
}

impl Csm {
    pub fn load(dir: &Path) -> Result<Csm, String> {
        let bg = Gguf::open(&dir.join("backbone.gguf"))?;
        let dg = Gguf::open(&dir.join("decoder.gguf"))?;
        let mg = Gguf::open(&dir.join("mimi.gguf"))?;
        let u = |k: &str| bg.get(k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| format!("backbone.gguf: missing {k}"));
        let (codebooks, vocab, eos) = (u("csm.codebooks")?, u("csm.audio_vocab")?, u("csm.codebook_eos")? as u32);
        // on a non-blocking stream of its own: it runs on its own thread, beside the language model
        let backbone = GpuModel::load_with(&bg, Config::from_gguf(&bg)?, true)?;
        let mut decoder = GpuModel::load(&dg, Config::from_gguf(&dg)?)?;
        decoder.share_kernels(&backbone)?;
        // the decoder's 31 passes a frame are most of its time: int8 activations there (about a third faster,
        // logits within ~0.5 of fp32 on a spread of ~20); GHOST_INFER_CSM_EXACT=1 keeps them f32
        decoder.int8_q8 = std::env::var_os("GHOST_INFER_CSM_EXACT").is_none();
        Ok(Csm {
            frames: CsmFrames::load(&bg, &dg, codebooks, vocab)?,
            text_embd: info(&bg, "csm.text_embd.weight")?,
            audio_embd: info(&bg, "csm.audio_embd.weight")?,
            proj_embd: info(&dg, "csm.proj_embd.weight")?,
            tok: Tokenizer::from_gguf(&bg)?,
            codebooks,
            vocab,
            eos,
            mimi: Mimi::load(&mg)?,
            backbone,
            decoder,
            bg,
            dg,
        })
    }

    pub fn session(&self) -> Result<CsmSession, String> {
        Ok(CsmSession {
            bb: GpuSession::new(&self.backbone.cfg, self.backbone.cfg.max_ctx)?,
            dec: GpuSession::new(&self.decoder.cfg, self.decoder.cfg.max_ctx)?,
            graph: None,
        })
    }

    /// The most frames the backbone's context holds.
    pub fn max_ctx(&self) -> usize {
        self.backbone.cfg.max_ctx
    }

    /// A turn's text frames: `<|begin_of_text|>[speaker]text<|end_of_text|>`.
    fn text_tokens(&self, speaker: u32, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = self.tok.bos_id.into_iter().collect();
        ids.extend(self.tok.encode(&format!("[{speaker}]{text}")));
        ids.extend(self.tok.eos_id);
        ids
    }

    fn push_text(&self, rows: &mut Vec<f32>, ids: &[u32]) {
        let m = table(&self.bg, &self.text_embd);
        for &id in ids {
            let at = rows.len();
            rows.resize(at + m.cols, 0.0);
            m.row_f32(id as usize, &mut rows[at..]);
        }
    }

    /// One audio frame: the sum of its codebooks' rows.
    fn push_frame(&self, rows: &mut Vec<f32>, codes: &[u32]) {
        let m = table(&self.bg, &self.audio_embd);
        let at = rows.len();
        rows.resize(at + m.cols, 0.0);
        let mut row = vec![0f32; m.cols];
        for (cb, &c) in codes.iter().enumerate() {
            m.row_f32(cb * self.vocab + c as usize, &mut row);
            rows[at..].iter_mut().zip(&row).for_each(|(a, b)| *a += b);
        }
    }

    /// The conversation so far and the turn to say, as backbone rows: each earlier turn's text, its frames
    /// and an end-of-audio frame (all codes 0), then the new turn's text. The oldest turns are left out
    /// while it would not leave `room` frames to speak in.
    fn prompt(&self, context: &[Turn], speaker: u32, text: &str, room: usize) -> Result<Vec<f32>, String> {
        let say = self.text_tokens(speaker, text);
        if say.len() + room > self.max_ctx() {
            return Err(format!("the text is too long to speak ({} tokens; at most {})", say.len(), self.max_ctx().saturating_sub(room)));
        }
        let len = |t: &Turn| self.text_tokens(t.speaker, &t.text).len() + t.frames.len() + 1;
        let mut from = 0;
        while from < context.len() && context[from..].iter().map(len).sum::<usize>() + say.len() + room > self.max_ctx() {
            from += 1;
        }
        let mut rows = Vec::new();
        for t in &context[from..] {
            self.push_text(&mut rows, &self.text_tokens(t.speaker, &t.text));
            for f in &t.frames {
                self.push_frame(&mut rows, f);
            }
            self.push_frame(&mut rows, &vec![self.eos; self.codebooks]);
        }
        self.push_text(&mut rows, &say);
        Ok(rows)
    }

    /// Speaks `text` as `speaker` after `context`: up to `max_frames` frames (80 ms each), handed to
    /// `on_frame` as they come (false stops). Returns them all.
    pub fn speak(&self, s: &mut CsmSession, context: &[Turn], speaker: u32, text: &str, max_frames: usize, sampling: Sampling, mut on_frame: impl FnMut(&[u32]) -> bool) -> Result<Vec<Vec<u32>>, String> {
        let rows = self.prompt(context, speaker, text, max_frames)?;
        let dim = self.backbone.cfg.dim;
        s.bb.reset()?;
        let chunks: Vec<&[f32]> = rows.chunks(1024 * dim).collect();
        for (i, c) in chunks.iter().enumerate() {
            self.backbone.forward_vectors(c, &mut s.bb, i + 1 == chunks.len())?;
        }
        self.frames.set_sampling(&self.backbone, sampling.temperature, sampling.top_k, sampling.seed)?;
        // each frame is handed on one frame late: the one right before the end is not speech but a short hum (the
        // Maya fine-tune closes every clip with the same frame), and it is kept out of the audio. It stays among
        // the frames returned, as the conversation's context.
        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut ended = false;
        while frames.len() < max_frames && s.bb.pos < s.bb.ctx {
            let codes = self.frames.step(&self.backbone, &self.decoder, &mut s.bb, &mut s.dec, &mut s.graph)?;
            // the end: every codebook but the last says end-of-audio
            if codes[..self.codebooks - 1].iter().all(|&c| c == self.eos) {
                ended = true;
                break;
            }
            let go = frames.last().is_none_or(|held| on_frame(held));
            frames.push(codes);
            if !go {
                return Ok(frames);
            }
        }
        // stopped by the length limit, not the end: the held frame is speech
        if !ended {
            if let Some(held) = frames.last() {
                on_frame(held);
            }
        }
        Ok(frames)
    }
}

/// 16-bit PCM mono WAV bytes.
pub fn wav(samples: &[f32], rate: usize) -> Vec<u8> {
    let data = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&(rate as u32).to_le_bytes());
    out.extend_from_slice(&(rate as u32 * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data as u32).to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

/// Compares the engine with numbers from scripts/csm-reference.py (the fp32 transformers model): the prompt's
/// tokens, the first frame's logits for every codebook (each from the reference's codes before it), greedy
/// frames, and Mimi's audio for the reference's codes.
pub fn check(dir: &Path, reference: &Path) -> Result<(), String> {
    use crate::json::Json;
    let r = Json::parse(&std::fs::read_to_string(reference).map_err(|e| format!("cannot read {}: {e}", reference.display()))?)?;
    let nums = |j: &Json| -> Vec<f64> { j.as_array().unwrap_or(&[]).iter().filter_map(Json::as_f64).collect() };
    let rows = |key: &str| -> Vec<Vec<f64>> { r.get(key).and_then(Json::as_array).unwrap_or(&[]).iter().map(nums).collect() };
    let csm = Csm::load(dir)?;
    let mut s = csm.session()?;

    let text = r.get("text").and_then(Json::as_str).ok_or("reference: no text")?;
    let speaker = r.get("speaker").and_then(Json::as_f64).unwrap_or(0.0) as u32;
    let want: Vec<u32> = nums(r.get("tokens").ok_or("reference: no tokens")?).iter().map(|&v| v as u32).collect();
    let got = csm.text_tokens(speaker, text);
    println!("tokens     {}", if got == want { "match".to_string() } else { format!("DIFFER: {got:?} vs {want:?}") });

    // the first frame, teacher-forced with the reference's codes
    let ref_codes: Vec<u32> = nums(r.get("frame0_codes").ok_or("reference: no frame0_codes")?).iter().map(|&v| v as u32).collect();
    let ref_logits = rows("frame0_logits");
    let mut rows_in = Vec::new();
    csm.push_text(&mut rows_in, &want);
    s.bb.reset()?;
    csm.backbone.forward_vectors(&rows_in, &mut s.bb, true)?;
    let report = |name: String, got: &[f32], want: &[f64]| {
        let max_diff = got.iter().zip(want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
        let spread = want.iter().copied().fold(f64::NEG_INFINITY, f64::max) - want.iter().copied().fold(f64::INFINITY, f64::min);
        let argmax = |v: &mut dyn Iterator<Item = f64>| v.enumerate().fold((0, f64::NEG_INFINITY), |b, (i, x)| if x > b.1 { (i, x) } else { b }).0;
        let (ga, wa) = (argmax(&mut got.iter().map(|&v| v as f64)), argmax(&mut want.iter().copied()));
        println!("{name:<10} max diff {max_diff:.4} (logits span {spread:.1}), argmax {ga} vs {wa}{}", if ga == wa { "" } else { "  <-- differs" });
    };
    report("codebook 0".into(), &s.bb.logits[..csm.vocab], &ref_logits[0]);
    let mut hidden = vec![0f32; csm.backbone.cfg.dim];
    csm.backbone.hidden(&s.bb, &mut hidden)?;
    let ref_hidden = nums(r.get("hidden").ok_or("reference: no hidden")?);
    let hd = hidden.iter().zip(&ref_hidden).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
    println!("hidden     max diff {hd:.4}");
    csm.backbone.head_logits(csm.frames.proj(), &mut s.bb)?;
    let dd = csm.decoder.cfg.dim;
    let embd = table(&csm.dg, &csm.proj_embd);
    let mut x = s.bb.logits[..dd].to_vec();
    x.resize(2 * dd, 0.0);
    embd.row_f32(ref_codes[0] as usize, &mut x[dd..]);
    s.dec.reset()?;
    csm.decoder.forward_vectors(&x, &mut s.dec, false)?;
    x.truncate(dd);
    for i in 1..csm.codebooks {
        csm.decoder.head_logits(csm.frames.head(i), &mut s.dec)?;
        if i <= 3 || i == csm.codebooks - 1 {
            report(format!("codebook {i}"), &s.dec.logits[..csm.vocab], &ref_logits[i]);
        }
        if i + 1 < csm.codebooks {
            embd.row_f32(i * csm.vocab + ref_codes[i] as usize, &mut x);
            csm.decoder.forward_vectors(&x, &mut s.dec, false)?;
        }
    }

    // greedy frames from the start, on the GPU graph
    let greedy = rows("greedy_frames");
    let frames = csm.speak(&mut s, &[], speaker, text, greedy.len(), Sampling { temperature: 0.0, top_k: 0, seed: 1 }, |_| true)?;
    let same = |a: &[u32], b: &[f64]| a.iter().zip(b).filter(|(x, y)| **x as f64 == **y).count();
    let codes_same: Vec<usize> = frames.iter().zip(&greedy).map(|(a, b)| same(a, b)).collect();
    let first0: Vec<u32> = frames.iter().map(|f| f[0]).collect();
    println!("greedy     codes equal per frame (of {}): {codes_same:?}; first codes {first0:?}", csm.codebooks);

    // Mimi on the reference's codes, all at once and frame by frame
    let codes: Vec<Vec<u32>> = rows("mimi_codes").iter().map(|f| f.iter().map(|&v| v as u32).collect()).collect();
    let ref_audio = nums(r.get("mimi_audio").ok_or("reference: no mimi_audio")?);
    let t0 = std::time::Instant::now();
    let mut st = csm.mimi.state();
    let whole = csm.mimi.decode(&mut st, &codes);
    let secs = t0.elapsed().as_secs_f64();
    let mut st = csm.mimi.state();
    let stream: Vec<f32> = codes.iter().flat_map(|f| csm.mimi.decode(&mut st, std::slice::from_ref(f))).collect();
    let snr = |a: &[f32]| {
        let (mut sig, mut err) = (0.0, 0.0);
        for (x, y) in a.iter().zip(&ref_audio) {
            sig += y * y;
            err += (*x as f64 - y) * (*x as f64 - y);
        }
        10.0 * (sig / err.max(1e-30)).log10()
    };
    println!(
        "mimi       {} samples (reference {}), SNR {:.1} dB all at once, {:.1} dB frame by frame; {:.0} ms for {:.2} s of audio",
        whole.len(),
        ref_audio.len(),
        snr(&whole),
        snr(&stream),
        secs * 1000.0,
        whole.len() as f64 / csm.mimi.sample_rate as f64
    );
    let out = reference.with_extension("mimi.wav");
    std::fs::write(&out, wav(&whole, csm.mimi.sample_rate)).map_err(|e| e.to_string())?;
    println!("wrote {}", out.display());
    Ok(())
}
