//! Whisper speech-to-text: 16 kHz audio to text, for voice calls. The model is one GGUF from
//! scripts/whisper-convert.py; it runs on the GPU (cuda::GpuWhisper), and the log-mel frames, the tokens'
//! embeddings and the choice of each next token are worked out here.
//!
//! The encoder reads only as many positions as the clip needs (50 a second) rather than Whisper's fixed 30 s,
//! which is what the ACFT fine-tune of large-v3-turbo is trained for: a few seconds of speech encode several
//! times faster.

use rayon::prelude::*;

use crate::cuda::GpuWhisper;
use crate::gguf::{Gguf, TensorInfo, Value};
use crate::quant::Matrix;
use crate::tokenizer::Tokenizer;

pub const RATE: usize = 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;

/// What a transcription found.
pub struct Heard {
    pub text: String,
    /// the language it was spoken in ("en", "ms", ...)
    pub language: String,
    /// how likely the clip held no speech (Whisper's <|nospeech|> after the first token)
    pub no_speech: f32,
}

pub struct Whisper {
    g: Gguf,
    gpu: GpuWhisper,
    filters: Vec<f32>, // n_mels rows of BINS
    n_mels: usize,
    dim: usize,
    embd: TensorInfo,
    pos: Vec<f32>,
    tok: Tokenizer,
    sot: u32,
    eot: u32,
    transcribe: u32,
    notimestamps: u32,
    nospeech: u32,
    timestamp_begin: u32,
    suppress: Vec<u32>,
    begin_suppress: Vec<u32>,
    languages: Vec<(String, u32)>,
    window: Vec<f32>,
    cos: Vec<f32>, // BINS x N_FFT
    sin: Vec<f32>,
}

impl Whisper {
    pub fn load(path: &std::path::Path) -> Result<Whisper, String> {
        let g = Gguf::open(path)?;
        let u = |k: &str| g.get(&format!("whisper.{k}")).and_then(Value::as_u64).map(|v| v as u32).ok_or_else(|| format!("{}: missing whisper.{k}", path.display()));
        let list = |k: &str| -> Vec<u32> { g.get(&format!("whisper.{k}")).and_then(Value::as_array).unwrap_or(&[]).iter().filter_map(|v| v.as_u64().map(|v| v as u32)).collect() };
        let names: Vec<String> = g.get("whisper.languages").and_then(Value::as_array).unwrap_or(&[]).iter().filter_map(|v| v.as_str().map(String::from)).collect();
        let floats = |name: &str| -> Result<Vec<f32>, String> {
            let t = g.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
            let n = t.n_elements() as usize;
            let mut v = vec![0f32; n];
            Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, rows: 1, cols: n }.row_f32(0, &mut v);
            Ok(v)
        };
        // the periodic Hann window and the DFT's rotations, worked out once
        let window: Vec<f32> = (0..N_FFT).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos() as f32).collect();
        let angle = |b: usize, i: usize| 2.0 * std::f64::consts::PI * ((b * i) % N_FFT) as f64 / N_FFT as f64;
        let cos = (0..BINS * N_FFT).map(|j| angle(j / N_FFT, j % N_FFT).cos() as f32).collect();
        let sin = (0..BINS * N_FFT).map(|j| angle(j / N_FFT, j % N_FFT).sin() as f32).collect();
        Ok(Whisper {
            gpu: GpuWhisper::load(&g)?,
            filters: floats("mel_filters")?,
            n_mels: u("n_mels")? as usize,
            dim: u("dim")? as usize,
            embd: g.tensor("dec.embd").cloned().ok_or("missing tensor dec.embd")?,
            pos: floats("dec.pos")?,
            tok: Tokenizer::from_gguf(&g)?,
            sot: u("token.sot")?,
            eot: u("token.eot")?,
            transcribe: u("token.transcribe")?,
            notimestamps: u("token.notimestamps")?,
            nospeech: u("token.nospeech")?,
            timestamp_begin: u("token.timestamp_begin")?,
            suppress: list("suppress"),
            begin_suppress: list("begin_suppress"),
            languages: names.into_iter().zip(list("language_tokens")).collect(),
            window,
            cos,
            sin,
            g,
        })
    }

    /// Whisper's log-mel frames of `audio` (16 kHz): a 400-point FFT every 10 ms with the clip reflected at
    /// its ends, power through the mel filters, log10, floored 8 below the loudest, scaled to about -1..1.
    /// Rows of n_mels, one per 10 ms.
    fn mel(&self, audio: &[f32]) -> Vec<f32> {
        let n = audio.len();
        let pad = N_FFT / 2;
        let at = |i: isize| -> f32 {
            // reflection at both ends (numpy's "reflect")
            let mut i = i;
            if n < 2 {
                return 0.0;
            }
            while i < 0 || i >= n as isize {
                i = if i < 0 { -i } else { 2 * (n as isize - 1) - i };
            }
            audio[i as usize]
        };
        let frames = n / HOP; // the last of 1 + n / HOP frames is dropped, as Whisper does
        let (window, cos, sin, filters) = (&self.window, &self.cos, &self.sin, &self.filters);
        let mut mel = vec![0f32; frames * self.n_mels];
        mel.par_chunks_mut(self.n_mels).enumerate().for_each(|(f, row)| {
            let x: Vec<f32> = (0..N_FFT).map(|i| at((f * HOP + i) as isize - pad as isize) * window[i]).collect();
            let power: Vec<f32> = (0..BINS)
                .map(|b| {
                    let re = crate::quant::dot_f32(&x, &cos[b * N_FFT..(b + 1) * N_FFT]);
                    let im = crate::quant::dot_f32(&x, &sin[b * N_FFT..(b + 1) * N_FFT]);
                    re * re + im * im
                })
                .collect();
            for (m, out) in row.iter_mut().enumerate() {
                *out = crate::quant::dot_f32(&filters[m * BINS..(m + 1) * BINS], &power).max(1e-10).log10();
            }
        });
        let top = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in &mut mel {
            *v = (v.max(top - 8.0) + 4.0) / 4.0;
        }
        mel
    }

    /// Decoder input rows for `tokens` at positions `from..`: each token's embedding plus its position's.
    fn rows(&self, tokens: &[u32], from: usize) -> Vec<f32> {
        let m = Matrix { data: &self.g.mmap[self.embd.offset..self.embd.offset + self.embd.n_bytes], ty: self.embd.ty, cols: self.dim, rows: self.embd.dims[1] as usize };
        let mut rows = vec![0f32; tokens.len() * self.dim];
        for (i, (&t, row)) in tokens.iter().zip(rows.chunks_exact_mut(self.dim)).enumerate() {
            m.row_f32(t as usize, row);
            let p = &self.pos[(from + i) * self.dim..(from + i + 1) * self.dim];
            row.iter_mut().zip(p).for_each(|(a, b)| *a += b);
        }
        rows
    }

    /// Transcribes `audio` (16 kHz mono, at most 30 s). `language`: a code such as "en", or None to let the
    /// model tell which it hears.
    pub fn transcribe(&mut self, audio: &[f32], language: Option<&str>) -> Result<Heard, String> {
        let audio = &audio[..audio.len().min(30 * RATE)];
        // a little quiet after the words: the last sounds are not cut off at the encoder's edge
        let mut clip = audio.to_vec();
        clip.resize(audio.len() + RATE / 5, 0.0);
        let mel = self.mel(&clip);
        self.gpu.encode(&mel)?;
        let mut logits = Vec::new();
        // the first step, after <|startoftranscript|>: which language, and whether there is speech at all
        self.gpu.decode(&self.rows(&[self.sot], 0), &mut logits)?;
        let no_speech = softmax_at(&logits, self.nospeech as usize);
        let lang = match language.and_then(|l| self.languages.iter().find(|(n, _)| n == l)) {
            Some(&(_, t)) => t,
            None => self.languages.iter().map(|&(_, t)| t).max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize])).ok_or("this Whisper file lists no languages")?,
        };
        let language = self.languages.iter().find(|(_, t)| *t == lang).map(|(n, _)| n.clone()).unwrap_or_default();
        let prompt = [lang, self.transcribe, self.notimestamps];
        self.gpu.decode(&self.rows(&prompt, 1), &mut logits)?;
        // about six tokens a second of speech at most, then it is going round in circles
        let most = (audio.len() as f32 / RATE as f32 * 6.0) as usize + 12;
        let mut out: Vec<u32> = Vec::new();
        while out.len() < most && self.gpu.pos < self.gpu.max_text {
            for &t in &self.suppress {
                logits[t as usize] = f32::NEG_INFINITY;
            }
            if out.is_empty() {
                for &t in &self.begin_suppress {
                    logits[t as usize] = f32::NEG_INFINITY;
                }
            }
            // no timestamps, no other special tokens: text, or the end
            for t in self.timestamp_begin as usize..logits.len() {
                logits[t] = f32::NEG_INFINITY;
            }
            for t in self.eot as usize + 1..self.timestamp_begin as usize {
                logits[t] = f32::NEG_INFINITY;
            }
            let next = logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
            if next == self.eot || repeating(&out, next) {
                break;
            }
            out.push(next);
            let pos = self.gpu.pos;
            self.gpu.decode(&self.rows(&[next], pos), &mut logits)?;
        }
        let text = self.tok.decode(&out, false).trim().to_string();
        Ok(Heard { text, language, no_speech })
    }
}

fn softmax_at(logits: &[f32], i: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    (logits[i] - max).exp() / total
}

/// The same few tokens over and over: a run of 1 to 8 tokens about to come a fourth time in a row.
fn repeating(out: &[u32], next: u32) -> bool {
    (1..=8).any(|len| {
        let n = out.len() + 1;
        if n < 4 * len {
            return false;
        }
        let at = |i: usize| if i == out.len() { next } else { out[i] };
        (0..len).all(|j| (1..4).all(|r| at(n - 1 - j) == at(n - 1 - j - r * len)))
    })
}

/// 16 kHz mono samples from a WAV file's bytes (16-bit PCM or 32-bit float, any rate or channel count).
pub fn read_wav(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let (mut fmt, mut data) = (None, None);
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
        let body = &bytes[at + 8..(at + 8 + len).min(bytes.len())];
        match id {
            b"fmt " if body.len() >= 16 => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        at += 8 + len + (len & 1);
    }
    let (fmt, data) = (fmt.ok_or("the WAV file has no fmt chunk")?, data.ok_or("the WAV file has no data")?);
    let u16le = |i: usize| u16::from_le_bytes([fmt[i], fmt[i + 1]]);
    let (format, channels, rate, bits) = (u16le(0), u16le(2) as usize, u32::from_le_bytes(fmt[4..8].try_into().unwrap()) as usize, u16le(14));
    let samples: Vec<f32> = match (format, bits) {
        (1, 16) => data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect(),
        (3, 32) => data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
        (0xFFFE, 16) => data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect(),
        _ => return Err(format!("WAV format {format} with {bits}-bit samples is not supported (16-bit PCM or 32-bit float)")),
    };
    let mono: Vec<f32> = samples.chunks(channels.max(1)).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect();
    Ok(resample(&mono, rate, RATE))
}

/// Linear resampling: enough for speech recognition.
pub fn resample(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = x.len() * to / from;
    (0..n)
        .map(|i| {
            let p = i as f64 * from as f64 / to as f64;
            let j = p as usize;
            let f = (p - j as f64) as f32;
            let a = x[j.min(x.len() - 1)];
            let b = x[(j + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}
