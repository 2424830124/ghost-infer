//! maya1 speech: a 3B Llama that, given a voice description and the words, writes SNAC codes (7 tokens a
//! frame, 85 ms); snac.rs turns them into 24 kHz audio as they come. Its emotion tags (<laugh>, <sigh>,
//! <whisper>...) go in the words. The model is a folder: maya1's GGUF (the voice fine-tune merged in) and
//! snac.gguf from scripts/snac-convert.py; voice.txt, if there, holds the description it was trained on.

use std::path::Path;

use crate::cuda::{GpuModel, GpuSession};
use crate::gguf::Gguf;
use crate::model::Config;
use crate::sampler::Sampler;
use crate::snac::{frame_of, Frame, Snac, CODES_PER_FRAME, MAYA1_CODE_BASE};
use crate::tokenizer::Tokenizer;

/// The description her voice was trained with (the LoRA learned it as hers).
const VOICE: &str = "Realistic female voice in her late 20s with an American accent. Warm, playful and expressive, natural conversational pacing.";
/// The end of speech (<custom_token_2>), and the last audio token.
const CODE_END: u32 = 128258;
const CODE_LAST: u32 = MAYA1_CODE_BASE + 7 * 4096 - 1;
/// Positions a session holds: a prompt (~60 tokens) and up to ~17 s of speech (a reply is at most three
/// sentences; each piece between pauses is spoken on its own).
const CTX: usize = 1536;
/// The first token the output layer scores (<custom_token_0>): every audio token and CODE_END come after it.
const AUDIO_FROM: u32 = 128256;

/// How maya1 is sampled, as its authors run it.
#[derive(Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub repetition: f32,
    pub seed: u64,
}

pub struct Maya1 {
    _g: Gguf,
    model: GpuModel,
    tok: Tokenizer,
    pub snac: Snac,
    voice: String,
    /// the token the first logit is (AUDIO_FROM, or 0 for a GGUF with its own full output layer)
    from: u32,
}

pub struct Maya1Session(GpuSession);

impl Maya1 {
    /// Whether `dir` is a maya1 folder (it has snac.gguf).
    pub fn is_folder(dir: &Path) -> bool {
        dir.join("snac.gguf").is_file()
    }

    pub fn load(dir: &Path) -> Result<Maya1, String> {
        let path = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")) && p.file_name().is_some_and(|n| n != "snac.gguf"))
            .ok_or_else(|| format!("{}: no maya1 .gguf beside snac.gguf", dir.display()))?;
        let mut g = Gguf::open(&path)?;
        let mut cfg = Config::from_gguf(&g)?;
        cfg.max_ctx = cfg.max_ctx.min(CTX);
        // It only ever writes audio tokens (and CODE_END, among them), so its output layer is just those rows of
        // the embedding it is tied to (a 28.7k-row matrix where the GGUF has a 157k-row one), and the embedding
        // stays in RAM (only the prompt's rows are read): ~300 MB of GPU memory left to the language model.
        // Logit i is then token AUDIO_FROM + i.
        let mut from = 0;
        if g.tensor("output.weight").is_none() {
            let e = g.tensor("token_embd.weight").cloned().ok_or("maya1: no token_embd.weight")?;
            let rows = e.dims[1] as usize;
            let row_bytes = e.n_bytes / rows;
            let head = crate::gguf::TensorInfo { name: "output.weight".into(), dims: vec![e.dims[0], (rows - AUDIO_FROM as usize) as u64], ty: e.ty, offset: e.offset + AUDIO_FROM as usize * row_bytes, n_bytes: (rows - AUDIO_FROM as usize) * row_bytes };
            g.tensors.push(head);
            g.metadata.insert("ghost.embd_on_host".into(), crate::gguf::Value::Bool(true));
            cfg.vocab = rows - AUDIO_FROM as usize;
            from = AUDIO_FROM;
        }
        // on a non-blocking stream of its own: it runs on its own thread, beside the language model
        let model = GpuModel::load_with(&g, cfg, true)?;
        let voice = std::fs::read_to_string(dir.join("voice.txt")).map(|v| v.trim().to_string()).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| VOICE.into());
        Ok(Maya1 { tok: Tokenizer::from_gguf(&g)?, snac: Snac::load(&Gguf::open(&dir.join("snac.gguf"))?)?, model, voice, from, _g: g })
    }

    pub fn session(&self) -> Result<Maya1Session, String> {
        Ok(Maya1Session(GpuSession::new(&self.model.cfg, self.model.cfg.max_ctx)?))
    }

    /// Speaks `text` (tags and all): up to `max_frames` frames, handed to `on_frame` as they come (false
    /// stops); SnacStream makes them audio (on another thread, so the GPU is not kept waiting). Returns the
    /// frames spoken.
    pub fn speak(&self, s: &mut Maya1Session, text: &str, max_frames: usize, sampling: Sampling, mut on_frame: impl FnMut(Frame) -> bool) -> Result<usize, String> {
        let s = &mut s.0;
        // the prompt as its authors write it: <SOH><BOS><description="..."> text<EOT><EOH><SOA><CODE_START>
        let prompt = self.tok.encode(&format!("<custom_token_3><|begin_of_text|><description=\"{}\"> {text}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>", self.voice));
        let room = s.ctx.saturating_sub(prompt.len() + 1) / CODES_PER_FRAME;
        if room == 0 {
            return Err(format!("the text is too long to speak ({} tokens)", prompt.len()));
        }
        let max_frames = max_frames.min(room);
        s.reset()?;
        self.model.forward_batch(&prompt, s, true)?;
        // top 256 only to spare sorting all 156k tokens each step: top-p 0.9 at 0.6 never reaches that far
        // (it picks among the logits, so among audio rows: the prompt's tokens in that range are what it penalizes)
        let mut sampler = Sampler::new(sampling.temperature, 256, sampling.top_p, sampling.seed).with_repetition(sampling.repetition, prompt.iter().filter_map(|&t| t.checked_sub(self.from)));
        let (mut codes, mut frames, mut dropped) = (Vec::with_capacity(CODES_PER_FRAME), 0usize, 0usize);
        while frames < max_frames {
            let id = self.from + sampler.sample(&s.logits);
            if id == CODE_END {
                break;
            }
            if (MAYA1_CODE_BASE..=CODE_LAST).contains(&id) {
                codes.push(id);
                if codes.len() == CODES_PER_FRAME {
                    // a frame with a code in the wrong place is left out (the model rarely writes one)
                    match frame_of(&codes) {
                        Some(f) => {
                            frames += 1;
                            if !on_frame(f) {
                                return Ok(frames);
                            }
                        }
                        None => dropped += 1,
                    }
                    codes.clear();
                }
            }
            self.model.forward_batch(&[id], s, true)?;
        }
        if dropped > 0 {
            eprintln!("[speech: {dropped} malformed frames left out]");
        }
        Ok(frames)
    }
}
