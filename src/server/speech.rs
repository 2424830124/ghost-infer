//! Speech for the server (POST /v1/audio/speech): a CSM model (csm.rs) or maya1 (maya1.rs) on a thread of its
//! own, beside the language model. It is loaded before the language model, so the GPU memory a mixture of
//! experts' cache takes is what is left after it. With CSM a conversation id keeps its last turns as context,
//! so a voice stays the same from one reply to the next; maya1 holds its voice (a fine-tune) on its own.

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

use crate::csm::{Csm, Sampling, Turn};

/// Earlier turns a conversation keeps, and the most frames of them (the backbone holds 2048 positions). Few: the
/// voice copies its context, and with only its own last turns there (six) it drifted from her a little more
/// each reply; the anchor (ANCHOR) holds it.
const KEEP_TURNS: usize = 2;
const KEEP_FRAMES: usize = 900;

/// The anchor: a line the voice says first, with nothing before it (where it is most itself), made once per
/// speaker and put first in every conversation's context, so each reply hears her as she sounds at her best.
/// Over 24 lines without it her pitch crept from 189 to 227 Hz (up to 286); with it, 194 to 205.
const ANCHOR: &str = "Hey, it's really good to hear from you. I was just thinking about you, actually. So, how's your day been going?";
const ANCHOR_SEED: u64 = 7;

/// An anchor as anchor.json holds it: {"text", "frames": [[codes of a frame], ...]}.
fn anchor_json(t: &Turn) -> crate::json::Json {
    use crate::json::{num, obj, str, Json};
    let frames = t.frames.iter().map(|f| Json::Arr(f.iter().map(|&c| num(c as f64)).collect())).collect();
    obj(vec![("text", str(&t.text)), ("frames", Json::Arr(frames))])
}

/// An anchor from anchor.json, as said by `speaker`.
fn anchor_from(j: &crate::json::Json, speaker: u32) -> Option<Turn> {
    use crate::json::Json;
    let text = j.get("text")?.as_str()?.to_string();
    let frames: Vec<Vec<u32>> = j.get("frames")?.as_array()?.iter().map(|f| f.as_array().unwrap_or(&[]).iter().filter_map(Json::as_f64).map(|c| c as u32).collect()).collect();
    (!frames.is_empty()).then_some(Turn { speaker, text, frames })
}

pub struct SpeechJob {
    pub text: String,
    pub speaker: u32,
    pub sampling: Sampling,
    pub max_frames: usize,
    /// whose turns to speak after (and to add this one to)
    pub conversation: Option<String>,
    pub events: mpsc::Sender<SpeechEvent>,
}

pub enum SpeechEvent {
    /// samples at `Speech::sample_rate`, as they are decoded (a frame, 80 ms, at a time)
    Audio(Vec<f32>),
    Done,
    Error(String),
}

pub struct Speech {
    pub id: String,
    pub sample_rate: usize,
    /// the sampling a request gets unless it says otherwise: CSM's (0.9, top 50) or maya1's (0.6)
    pub temperature: f32,
    pub top_k: usize,
    jobs: mpsc::Sender<SpeechJob>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Speech {
    /// Loads the speech folder `dir` (CSM's three GGUFs, or maya1's GGUF and snac.gguf) on a new thread;
    /// returns once it has loaded (or failed to).
    pub fn start(dir: &Path) -> Result<Speech, String> {
        if crate::maya1::Maya1::is_folder(dir) {
            return Speech::start_maya1(dir);
        }
        let id = dir.file_name().and_then(|n| n.to_str()).unwrap_or("speech").to_string();
        let (jobs, rx) = mpsc::channel::<SpeechJob>();
        let (ready, loaded) = mpsc::channel();
        let dir = dir.to_path_buf();
        let thread = std::thread::spawn(move || {
            let t = Instant::now();
            let model = crate::cuda::bind_thread().and_then(|()| Csm::load(&dir)).and_then(|m| m.session().map(|s| (m, s)));
            let (model, mut session) = match model {
                Ok(m) => m,
                Err(e) => {
                    let _ = ready.send(Err(e));
                    return;
                }
            };
            let _ = ready.send(Ok((model.mimi.sample_rate, t.elapsed().as_secs_f64())));
            let mut conversations: HashMap<String, Vec<Turn>> = HashMap::new();
            let mut anchors: HashMap<u32, Turn> = HashMap::new();
            for job in rx {
                let mut context = Vec::new();
                if job.conversation.is_some() {
                    if let std::collections::hash_map::Entry::Vacant(e) = anchors.entry(job.speaker) {
                        // the voice folder's anchor.json if it has one (another voice's line, to steer this one
                        // towards it: the frames are Mimi codes, the same for every CSM), else its own line
                        let given = std::fs::read_to_string(dir.join("anchor.json")).ok().and_then(|t| crate::json::Json::parse(&t).ok()).and_then(|j| anchor_from(&j, job.speaker));
                        let made = || {
                            let sampling = Sampling { temperature: 0.9, top_k: 50, seed: ANCHOR_SEED };
                            model.speak(&mut session, &[], job.speaker, ANCHOR, 250, sampling, |_| true).map(|frames| Turn { speaker: job.speaker, text: ANCHOR.into(), frames })
                        };
                        match given.map(Ok).unwrap_or_else(made) {
                            Ok(turn) => {
                                // GHOST_INFER_SAVE_ANCHOR=<file>: the anchor written out, to give another voice
                                if let Some(path) = std::env::var_os("GHOST_INFER_SAVE_ANCHOR") {
                                    let _ = std::fs::write(path, anchor_json(&turn).to_string());
                                }
                                e.insert(turn);
                            }
                            Err(err) => eprintln!("[speech: no anchor: {err}]"),
                        }
                    }
                    context.extend(anchors.get(&job.speaker).cloned());
                }
                context.extend(job.conversation.as_ref().and_then(|c| conversations.get(c)).cloned().unwrap_or_default());
                match speak(&model, &mut session, &context, &job) {
                    Ok(spoken) => {
                        if let Some(c) = job.conversation.clone() {
                            let turns = conversations.entry(c).or_default();
                            turns.extend(spoken);
                            while turns.len() > KEEP_TURNS || (turns.len() > 1 && turns.iter().map(|t| t.frames.len()).sum::<usize>() > KEEP_FRAMES) {
                                turns.remove(0);
                            }
                        }
                        let _ = job.events.send(SpeechEvent::Done);
                    }
                    Err(e) => {
                        let _ = job.events.send(SpeechEvent::Error(e));
                    }
                }
            }
        });
        let (sample_rate, secs) = loaded.recv().map_err(|_| "the speech thread stopped while loading".to_string())??;
        eprintln!("[speech: loaded {id} in {secs:.1}s]");
        Ok(Speech { id, sample_rate, temperature: 0.9, top_k: 50, jobs, thread: Some(thread) })
    }

    /// maya1 on its own thread: each job's pieces spoken one after another, with no context between them (the
    /// fine-tune is her voice every time).
    fn start_maya1(dir: &Path) -> Result<Speech, String> {
        let id = dir.file_name().and_then(|n| n.to_str()).unwrap_or("speech").to_string();
        let (jobs, rx) = mpsc::channel::<SpeechJob>();
        let (ready, loaded) = mpsc::channel();
        let dir = dir.to_path_buf();
        let thread = std::thread::spawn(move || {
            let t = Instant::now();
            let model = crate::cuda::bind_thread().and_then(|()| crate::maya1::Maya1::load(&dir)).and_then(|m| m.session().map(|s| (m, s)));
            let (model, mut session) = match model {
                Ok(m) => m,
                Err(e) => {
                    let _ = ready.send(Err(e));
                    return;
                }
            };
            let _ = ready.send(Ok((model.snac.sample_rate, t.elapsed().as_secs_f64())));
            for job in rx {
                let done = speak_maya1(&model, &mut session, &job);
                let _ = job.events.send(match done {
                    Ok(()) => SpeechEvent::Done,
                    Err(e) => SpeechEvent::Error(e),
                });
            }
        });
        let (sample_rate, secs) = loaded.recv().map_err(|_| "the speech thread stopped while loading".to_string())??;
        eprintln!("[speech: loaded {id} (maya1) in {secs:.1}s]");
        Ok(Speech { id, sample_rate, temperature: 0.6, top_k: 0, jobs, thread: Some(thread) })
    }

    pub fn send(&self, job: SpeechJob) -> Result<(), String> {
        self.jobs.send(job).map_err(|_| "the speech thread stopped".into())
    }

    /// Ends the thread once its jobs are done; the model's GPU memory is free when this returns.
    pub fn stop(mut self) {
        drop(self.jobs);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        eprintln!("[speech: unloaded {}]", self.id);
    }
}

/// Speech-to-text for the server (POST /v1/audio/transcriptions): a Whisper file (whisper.rs) on a thread of
/// its own, loaded before the language model as the speech model is.
pub struct Listener {
    pub id: String,
    jobs: mpsc::Sender<(Vec<f32>, mpsc::Sender<Result<crate::whisper::Heard, String>>)>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Listener {
    pub fn start(path: &Path) -> Result<Listener, String> {
        let id = path.file_stem().and_then(|n| n.to_str()).unwrap_or("whisper").to_string();
        let (jobs, rx) = mpsc::channel::<(Vec<f32>, mpsc::Sender<Result<crate::whisper::Heard, String>>)>();
        let (ready, loaded) = mpsc::channel();
        let path = path.to_path_buf();
        let thread = std::thread::spawn(move || {
            let t = Instant::now();
            let mut w = match crate::cuda::bind_thread().and_then(|()| crate::whisper::Whisper::load(&path)) {
                Ok(w) => w,
                Err(e) => {
                    let _ = ready.send(Err(e));
                    return;
                }
            };
            let _ = ready.send(Ok(t.elapsed().as_secs_f64()));
            for (audio, reply) in rx {
                let t = Instant::now();
                let heard = w.transcribe(&audio, None);
                if let Ok(h) = &heard {
                    eprintln!("[listen: {:.1}s of audio in {:.2}s, {}: {:?}]", audio.len() as f64 / crate::whisper::RATE as f64, t.elapsed().as_secs_f64(), h.language, crate::tools::clip(&h.text, 80));
                }
                let _ = reply.send(heard);
            }
        });
        let secs = loaded.recv().map_err(|_| "the listening thread stopped while loading".to_string())??;
        eprintln!("[listen: loaded {id} in {secs:.1}s]");
        Ok(Listener { id, jobs, thread: Some(thread) })
    }

    /// The text of `audio` (16 kHz mono).
    pub fn transcribe(&self, audio: Vec<f32>) -> Result<crate::whisper::Heard, String> {
        let (tx, rx) = mpsc::channel();
        self.jobs.send((audio, tx)).map_err(|_| "the listening thread stopped".to_string())?;
        rx.recv().map_err(|_| "the listening thread stopped".to_string())?
    }

    /// Ends the thread once its jobs are done; the model's GPU memory is free when this returns.
    pub fn stop(mut self) {
        drop(self.jobs);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        eprintln!("[listen: unloaded {}]", self.id);
    }
}

/// The voice models (--tts, --stt): always available, but loaded only while asked for (a voice call, spoken
/// replies), each time together with the language model: their GPU memory comes out of what that model would
/// have (a mixture of experts' cache halves its speed without it), so they load first and unload before it
/// loads again.
#[derive(Default)]
pub struct Voice {
    pub tts: Option<std::path::PathBuf>,
    pub stt: Option<std::path::PathBuf>,
    pub speech: Option<std::sync::Arc<Speech>>,
    pub listener: Option<std::sync::Arc<Listener>>,
}

impl Voice {
    /// What it can load: the speech and speech-to-text models' names.
    pub fn names(&self) -> (Option<String>, Option<String>) {
        let tts = self.tts.as_ref().map(|d| d.file_name().and_then(|n| n.to_str()).unwrap_or("speech").to_string());
        let stt = self.stt.as_ref().map(|p| p.file_stem().and_then(|n| n.to_str()).unwrap_or("whisper").to_string());
        (tts, stt)
    }

    pub fn loaded(&self) -> bool {
        self.speech.is_some() || self.listener.is_some()
    }

    /// About the GPU memory loading it takes: its files (the speech folder's GGUFs and the speech-to-text file),
    /// their working memory and the reserve kept beside a call.
    pub fn bytes(&self) -> usize {
        let size = |p: &std::path::Path| std::fs::metadata(p).map(|m| m.len() as usize).unwrap_or(0);
        let tts: usize = self.tts.iter().flat_map(|d| std::fs::read_dir(d).into_iter().flatten().flatten()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf"))).map(|p| size(&p)).sum();
        let stt = self.stt.as_deref().map_or(0, size);
        if tts + stt == 0 { 0 } else { tts + stt + VOICE_WORK + VOICE_RESERVE }
    }
}

/// The voice's working memory beside its files (the speech model's cache and the codec's and Whisper's
/// buffers): maya1-her and Whisper turbo (2.9 GB of files) took about 3.5 GB loaded.
const VOICE_WORK: usize = 600 << 20;

/// GPU memory kept free beside a call's model, on top of the usual reserve: the voice's own working memory is
/// small (~35 MB), this is room for the desktop (the app's window, a browser, a screen recorder) to grow during
/// a call. Small: when the context is sized, loading still holds ~300 MB it gives back after, so the memory
/// really free during a call is that much more.
const VOICE_RESERVE: usize = 128 << 20;

/// Loads or unloads the voice models (call it with no language model loaded). Unloading waits for requests
/// still using them.
pub fn set_voice(voice: &std::sync::Mutex<Voice>, on: bool) {
    // while the voice is loaded the model beside it keeps more GPU memory free (cuda::set_extra_reserve)
    crate::cuda::set_extra_reserve(if on { VOICE_RESERVE } else { 0 });
    let mut v = voice.lock().unwrap();
    if on {
        if v.speech.is_none() {
            if let Some(d) = v.tts.clone() {
                match Speech::start(&d) {
                    Ok(s) => v.speech = Some(std::sync::Arc::new(s)),
                    Err(e) => eprintln!("[speech: could not load {}: {e}]", d.display()),
                }
            }
        }
        if v.listener.is_none() {
            if let Some(p) = v.stt.clone() {
                match Listener::start(&p) {
                    Ok(l) => v.listener = Some(std::sync::Arc::new(l)),
                    Err(e) => eprintln!("[listen: could not load {}: {e}]", p.display()),
                }
            }
        }
        return;
    }
    let (speech, listener) = (v.speech.take(), v.listener.take());
    drop(v);
    if let Some(s) = speech {
        Speech::stop(last(s));
    }
    if let Some(l) = listener {
        Listener::stop(last(l));
    }
}

/// What `a` holds, once no one else does (a request still running keeps its own reference until it is done).
fn last<T>(mut a: std::sync::Arc<T>) -> T {
    loop {
        match std::sync::Arc::try_unwrap(a) {
            Ok(x) => return x,
            Err(back) => {
                a = back;
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }
}

/// A piece of what a job says: words, or a pause its text asks for (seconds).
#[derive(Debug, PartialEq)]
enum Piece {
    Words(String),
    Quiet(f32),
}

/// Directions the voice can make a sound of (by how they start), and the sound: the spellings a listener picked
/// from takes of each in Maya's voice (design/voice-samples). "Heh", "haah", "pfft", "ahem", "hnn", "hngh" and
/// both yawns did not sound as they should. A laugh is "hahahah": the "hah" picked in a short line was no laugh
/// in a call.
const SOUNDS: [(&[&str], &str); 11] = [
    (&["laugh", "chuckl", "giggl"], "hahahah,"),
    (&["sigh"], "hhh..."),
    (&["hmm", "think", "ponder"], "hmm..."),
    (&["gasp"], "oh!"),
    (&["scoff", "snort"], "psh,"),
    (&["groan"], "ugh,"),
    (&["shock"], "what?!"),
    (&["whimper"], "mmm-nn..."),
    (&["grunt"], "hmph."),
    (&["moan"], "mmmh..."),
    (&["aha"], "aha!"),
];

/// The directions maya1 has a tag of its own for: it makes the sound itself. The rest fall back to SOUNDS'
/// spellings ("hmm...", "ugh,", "what?!").
const MAYA1_SOUNDS: [(&[&str], &str); 10] = [
    (&["laugh"], "<laugh>"),
    (&["chuckl"], "<chuckle>"),
    (&["giggl"], "<giggle>"),
    (&["sigh"], "<sigh>"),
    (&["gasp"], "<gasp>"),
    (&["whisper"], "<whisper>"),
    (&["scoff", "snort"], "<snort>"),
    (&["exhale"], "<exhale>"),
    (&["gulp"], "<gulp>"),
    (&["cry", "cries", "sob"], "<cry>"),
];

/// An interjection written out that the voice gets wrong, and what it says well instead ("" leaves it out):
/// a laugh (haha, hahaha, hehe, heh) as "hahahah", pfft as "psh", hngh / nngh as "hmph", hnn / nnn as "mmm-nn", ahem left out,
/// and a sound drawn out ("Ohhh", "Ughhh", "Ahhhh", "Mmmm") said once ("Oh", "Ugh", "Ah", "Mm"): drawn out, the
/// voice mumbled it.
fn respell(word: &str) -> Option<String> {
    let w = word.to_lowercase();
    let only = |set: &str| w.chars().all(|c| set.contains(c));
    let fixed = match () {
        _ if w.len() >= 4 && w.starts_with("ha") && only("ha") => "hahahah",
        _ if w.len() >= 3 && w.starts_with("he") && only("he") => "hahahah",
        _ if w.starts_with("pf") && only("pft") => "psh",
        _ if w.len() >= 3 && w.contains("nn") && only("hn") => "mmm-nn",
        _ if w.len() >= 4 && w.contains("ng") && only("hng") => "hmph",
        _ if w == "ahem" => "",
        _ => return drawn_out(&w),
    };
    Some(fixed.to_string())
}

/// A sound drawn out by repeating a letter three times or more, said once (m twice: "mm", "hmm"). The sounds
/// picked by ear as they are ("hhh", "mmmh") stay.
fn drawn_out(w: &str) -> Option<String> {
    if ["hhh", "mmmh"].contains(&w) {
        return None;
    }
    let chars: Vec<char> = w.chars().collect();
    if !chars.iter().all(|c| c.is_ascii_alphabetic()) || !chars.windows(3).any(|t| t[0] == t[1] && t[1] == t[2]) {
        return None;
    }
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let run = chars[i..].iter().take_while(|&&d| d == c).count();
        let keep = if run >= 3 { if c == 'm' { 2 } else { 1 } } else { run };
        out.extend(std::iter::repeat_n(c, keep));
        i += run;
    }
    // only short sounds and words ("Ohhh", "sooo"); a longer word stretched for effect is left as written
    (out.len() <= 3).then_some(out)
}

/// A job's text as words and pauses, from bracketed directions as ElevenLabs reads them: [short pause], [pause]
/// and [long pause] are silences of their own; [laughs], [sighs] and the like become the sound the voice makes
/// of them (SOUNDS); any other direction ([cough], [whispers]...) is left out rather than read aloud, as the
/// voice cannot do it. For maya1 (`tags`) the sounds it has a tag for are that tag, and a laugh written out
/// (haha) is its <laugh>.
fn pieces(text: &str, tags: bool) -> Vec<Piece> {
    let mut out = Vec::new();
    let mut words = String::new();
    let flush = |words: &mut String, out: &mut Vec<Piece>| {
        let mut w = speakable(&words.split_whitespace().collect::<Vec<_>>().join(" "));
        if tags {
            w = maya1_tags(&w);
        }
        if w.chars().any(char::is_alphanumeric) {
            out.push(Piece::Words(w));
        }
        words.clear();
    };
    let text = no_special(text);
    let mut rest = text.as_str();
    while let Some(open) = rest.find('[') {
        let Some(len) = rest[open..].find(']') else { break };
        words.push_str(&rest[..open]);
        let tag = rest[open + 1..open + len].trim().to_lowercase();
        rest = &rest[open + len + 1..];
        let quiet = match tag.as_str() {
            // a pause asked for, heard as one beside the voice's own gaps (kept up to 0.56 s)
            "short pause" | "beat" | "breath" | "breathes" | "inhales" | "exhales" => Some(0.45),
            "pause" | "pauses" => Some(0.8),
            "long pause" => Some(1.4),
            _ => None,
        };
        // (a pause is a clip break with silence: kept inside one clip as "...", the voice ended early more often,
        // losing the reply's last words in 3 to 6 of 17 replies instead of 1)
        if let Some(q) = quiet {
            flush(&mut words, &mut out);
            out.push(Piece::Quiet(q));
        } else if let Some((_, sound)) = MAYA1_SOUNDS.iter().filter(|_| tags).chain(SOUNDS.iter()).find(|(starts, _)| starts.iter().any(|s| tag.starts_with(s))) {
            words.push_str(&format!(" {sound} "));
        }
    }
    words.push_str(rest);
    flush(&mut words, &mut out);
    out
}

/// maya1's tags as it knows them once speakable() is done (it capitalizes a tag that starts a sentence), and
/// a laugh written out (respelled "hahahah") as its <laugh>.
fn maya1_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    // a tag capitalized as a sentence's start hands the capital on to the word after it
    let (mut inside, mut first, mut carry) = (false, false, false);
    for c in text.chars() {
        match c {
            '<' => (inside, first) = (true, true),
            '>' => inside = false,
            _ if inside && first => {
                carry = c.is_uppercase();
                first = false;
            }
            _ if carry && !inside && c.is_alphabetic() => {
                out.extend(c.to_uppercase());
                carry = false;
                continue;
            }
            _ => {}
        }
        out.extend(if inside { c.to_lowercase().collect::<Vec<_>>() } else { vec![c] });
    }
    out.split(' ')
        .map(|tok| {
            let core = tok.trim_matches(|c: char| !c.is_alphanumeric());
            if core.eq_ignore_ascii_case("hahahah") { "<laugh>".to_string() } else { tok.to_string() }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A reply without the chat format's own tokens a model let slip into its text, and the role name right after
/// one: a Llama 3 roleplay model answered "<|python_tag|>user<|python_tag|>(\"Hello...\")", and maya1, whose
/// tokenizer has the same specials, read the tokens as tokens and mumbled.
fn no_special(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find("<|") {
        let Some(len) = rest[open..].find("|>") else { break };
        out.push_str(&rest[..open]);
        rest = &rest[open + len + 2..];
        let word = rest.trim_start().split(|c: char| !c.is_alphabetic()).next().unwrap_or("");
        if matches!(word, "user" | "assistant" | "system") {
            rest = &rest.trim_start()[word.len()..];
        }
        out.push(' ');
    }
    out.push_str(rest);
    out
}

/// Words as the voice reads them well: straight quotes and apostrophes, three dots for an ellipsis, no Markdown
/// emphasis, and capitals for "I" and each sentence's first letter. The voice learned from ordinary written
/// text: given "i’m not a robot, but i *am* a bit of a machine" it said "you are not a robot, but Ira, Simeon",
/// and the same sentence written normally came out word for word.
fn speakable(text: &str) -> String {
    // interjections it says badly respelled, keeping the punctuation around them
    let text = text
        .split(' ')
        .filter_map(|tok| {
            let core = tok.trim_matches(|c: char| !c.is_alphanumeric());
            let at = tok.find(core).unwrap_or(0);
            match respell(core).filter(|_| !core.is_empty()) {
                Some(r) if r.is_empty() => None,
                Some(r) => {
                    // a word in capitals keeps its capital ("Ohhh" -> "Oh")
                    let r = if core.starts_with(char::is_uppercase) { r[..1].to_uppercase() + &r[1..] } else { r };
                    Some(format!("{}{r}{}", &tok[..at], &tok[at + core.len()..]))
                }
                None => Some(tok.to_string()),
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    let chars: Vec<char> = text
        .replace('…', "...")
        .chars()
        .filter_map(|c| match c {
            '‘' | '’' | 'ʼ' | '`' => Some('\''),
            // quotation marks left out: around a line ("Did I ask?") the voice stumbled on it
            '“' | '”' | '"' | '*' | '_' | '~' => None,
            c => Some(c),
        })
        .collect();
    let mut out = String::with_capacity(text.len());
    let mut start = true;
    for (k, &c) in chars.iter().enumerate() {
        let prev = if k > 0 { chars[k - 1] } else { ' ' };
        let next = chars.get(k + 1).copied().unwrap_or(' ');
        if c.is_alphabetic() {
            // "i" on its own or before an apostrophe (i'm, i'd)
            let lone_i = c == 'i' && !prev.is_alphanumeric() && prev != '\'' && !next.is_alphanumeric();
            if start || lone_i {
                out.extend(c.to_uppercase());
            } else {
                out.push(c);
            }
            start = false;
        } else {
            out.push(c);
            // a sentence ends at . ! ? before a space (not 3.5)
            if matches!(c, '.' | '!' | '?') && (next.is_whitespace() || k + 1 == chars.len()) {
                start = true;
            }
        }
    }
    out
}

/// A frame (80 ms) quieter than this is silence, and at most this many in a row of the voice's own silence are
/// kept (560 ms): only its overlong gaps (up to 0.9 s) are shortened. Tighter (320 ms, and 0.008, which caught
/// soft speech too) it sounded choppy.
const QUIET_RMS: f32 = 0.003;
const QUIET_FRAMES_KEPT: usize = 7;

/// What the decoding thread is handed: a frame to decode, or this many samples of silence.
enum Out {
    Frame(Vec<u32>),
    Quiet(usize),
}

/// One job: its words spoken a piece at a time (each with the pieces before it as context, so the voice runs on)
/// and its pauses as silence, decoded by Mimi on a second thread as the frames come. The turns it adds.
fn speak(model: &Csm, s: &mut crate::csm::CsmSession, context: &[Turn], job: &SpeechJob) -> Result<Vec<Turn>, String> {
    let t = Instant::now();
    let (tx, rx) = mpsc::channel::<Out>();
    let mimi = &model.mimi;
    let mut st = model.mimi.state();
    let events = job.events.clone();
    let rate = mimi.sample_rate;
    let turns = std::thread::scope(|sc| {
        let worker = sc.spawn(move || {
            // the voice's own silences cut short (it leaves 0.5 to 0.9 s between sentences and at dashes, which
            // sounded like odd pauses and drowned the pauses asked for); those asked for ([pause]) stay whole
            let mut quiet = QUIET_FRAMES_KEPT - 1; // a clip may start with one quiet frame
            for o in rx {
                let audio = match o {
                    Out::Frame(f) => {
                        let a = mimi.decode(&mut st, &[f]);
                        let rms = (a.iter().map(|v| v * v).sum::<f32>() / a.len().max(1) as f32).sqrt();
                        quiet = if rms < QUIET_RMS { quiet + 1 } else { 0 };
                        if quiet > QUIET_FRAMES_KEPT {
                            continue;
                        }
                        a
                    }
                    Out::Quiet(n) => {
                        quiet = QUIET_FRAMES_KEPT; // the voice's silence after a pause asked for adds nothing
                        vec![0.0; n]
                    }
                };
                if events.send(SpeechEvent::Audio(audio)).is_err() {
                    break; // the client went away
                }
            }
        });
        let mut ctx = context.to_vec();
        let mut turns = Vec::new();
        for p in pieces(&job.text, false) {
            match p {
                Piece::Quiet(secs) => {
                    if tx.send(Out::Quiet((secs * rate as f32) as usize)).is_err() {
                        break;
                    }
                }
                Piece::Words(w) => {
                    let most = job.max_frames.min(w.chars().count() * 2 + 50);
                    // stops when the client is gone (the decoder thread then drops its end)
                    let frames = model.speak(s, &ctx, job.speaker, &w, most, job.sampling, |f| tx.send(Out::Frame(f.to_vec())).is_ok());
                    let turn = Turn { speaker: job.speaker, text: w, frames: frames? };
                    ctx.push(turn.clone());
                    turns.push(turn);
                }
            }
        }
        drop(tx);
        let _ = worker.join();
        Ok::<_, String>(turns)
    })?;
    let frames: usize = turns.iter().map(|t| t.frames.len()).sum();
    eprintln!("[speech: {:.1}s of speech ({} frames in {} pieces, {} earlier turns) in {:.2}s]", frames as f64 * 0.08, frames, turns.len(), context.len(), t.elapsed().as_secs_f64());
    Ok(turns)
}

/// What maya1's decoding thread is handed: a frame, the end of a piece (decode the rest), or silence.
enum Maya1Out {
    Frame(crate::snac::Frame),
    End,
    Quiet(usize),
}

/// One job with maya1: each piece of words spoken on its own (a stream of frames decoded by SNAC on a second
/// thread as they come), its pauses as silence.
fn speak_maya1(model: &crate::maya1::Maya1, s: &mut crate::maya1::Maya1Session, job: &SpeechJob) -> Result<(), String> {
    let t = Instant::now();
    let (tx, rx) = mpsc::channel::<Maya1Out>();
    let snac = &model.snac;
    let events = job.events.clone();
    let rate = snac.sample_rate;
    let sampling = crate::maya1::Sampling { temperature: job.sampling.temperature, top_p: 0.9, repetition: 1.1, seed: job.sampling.seed };
    let (frames, pieces_spoken) = std::thread::scope(|sc| {
        let worker = sc.spawn(move || {
            let mut stream = crate::snac::SnacStream::new(snac);
            for o in rx {
                let audio = match o {
                    Maya1Out::Frame(f) => stream.push(f),
                    Maya1Out::End => std::mem::replace(&mut stream, crate::snac::SnacStream::new(snac)).finish(),
                    Maya1Out::Quiet(n) => vec![0.0; n],
                };
                if !audio.is_empty() && events.send(SpeechEvent::Audio(audio)).is_err() {
                    break; // the client went away
                }
            }
        });
        // each piece said <excited>: her calls are calm, and without it the voice came out breathier and softer
        // than she sounds (harmonic share 54% against her 63%; 60% with it). GHOST_INFER_MAYA1_TONE changes the
        // tag, and empty leaves it out
        let tone = std::env::var("GHOST_INFER_MAYA1_TONE").unwrap_or_else(|_| "excited".into());
        let (mut frames, mut spoken) = (0, 0);
        for (i, p) in pieces(&job.text, true).into_iter().enumerate() {
            match p {
                Piece::Quiet(secs) => {
                    if tx.send(Maya1Out::Quiet((secs * rate as f32) as usize)).is_err() {
                        break;
                    }
                }
                Piece::Words(w) => {
                    let most = job.max_frames.min(w.chars().count() * 2 + 50);
                    let w = if tone.is_empty() || w.starts_with('<') { w } else { format!("<{tone}> {w}") };
                    let piece = crate::maya1::Sampling { seed: sampling.seed.wrapping_add(i as u64), ..sampling };
                    // stops when the client is gone (the decoder thread then drops its end)
                    frames += model.speak(s, &w, most, piece, |f| tx.send(Maya1Out::Frame(f)).is_ok())?;
                    spoken += 1;
                    if tx.send(Maya1Out::End).is_err() {
                        break;
                    }
                }
            }
        }
        drop(tx);
        let _ = worker.join();
        Ok::<_, String>((frames, spoken))
    })?;
    let secs = frames as f64 * crate::snac::FRAME_SAMPLES as f64 / rate as f64;
    eprintln!("[speech: {secs:.1}s of speech ({frames} frames in {pieces_spoken} pieces) in {:.2}s]", t.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_become_pauses_and_laughs() {
        let p = pieces("Oh wow. [short pause] That's huge! [laughs] Okay [cough] so... [long pause] tell me more.", false);
        assert_eq!(
            p,
            vec![
                Piece::Words("Oh wow.".into()),
                Piece::Quiet(0.45),
                Piece::Words("That's huge! Hahahah, Okay so...".into()),
                Piece::Quiet(1.4),
                Piece::Words("Tell me more.".into()),
            ]
        );
        // the sounds picked by ear, from the tags a model writes
        assert_eq!(pieces("[sighs] yeah. [hmm] maybe. [gasps] really? [scoffs] come on. [groans] fine.", false), vec![Piece::Words("Hhh... Yeah. Hmm... Maybe. Oh! Really? Psh, come on. Ugh, fine.".into())]);
    }

    #[test]
    fn badly_said_interjections_are_respelled() {
        assert_eq!(speakable("Hahaha, okay. heh, sure. Pfft. Ahem, as I was saying"), "Hahahah, okay. Hahahah, sure. Psh. As I was saying");
        assert_eq!(speakable("He said hello, hehe"), "He said hello, hahahah");
        assert_eq!(speakable("Hngh, fine. nnn... okay, hnn."), "Hmph, fine. Mmm-nn... Okay, mmm-nn.");
        // real words that look close are left alone
        assert_eq!(speakable("Hang on, he knew."), "Hang on, he knew.");
        // drawn out, said once
        assert_eq!(speakable("Ohhh no. ughhh, Ahhhh... hmmmm, sooo good. Mmmm."), "Oh no. Ugh, Ah... Hmm, so good. Mm.");
        assert_eq!(speakable("all good, keep cool"), "All good, keep cool");
    }

    #[test]
    fn the_newer_sounds() {
        assert_eq!(pieces("[shocked] you won? [whimpers] don't go. [grunts] fine. [moans] not again. [aha] got it. [yawns] night.", false), vec![Piece::Words("What?! You won? Mmm-nn... Don't go. Hmph. Fine. Mmmh... Not again. Aha! Got it. Night.".into())]);
    }

    #[test]
    fn words_are_written_as_the_voice_reads_them() {
        assert_eq!(speakable("i’m not a robot, but i *am* a bit of a machine. and i’d say “hi”…"), "I'm not a robot, but I am a bit of a machine. And I'd say hi...");
        assert_eq!(speakable("it took 3.5 seconds. wild, isn't it? yes"), "It took 3.5 seconds. Wild, isn't it? Yes");
        assert_eq!(speakable("in iceland it's icy, i think"), "In iceland it's icy, I think");
    }

    #[test]
    fn maya1_makes_the_sounds_it_has_tags_for() {
        assert_eq!(
            pieces("[laughs] stop it. [pause] I know [sighs] right? [whispers] come here. [groans] fine, haha.", true),
            vec![Piece::Words("<laugh> Stop it.".into()), Piece::Quiet(0.8), Piece::Words("I know <sigh> right? <whisper> Come here. Ugh, fine, <laugh>".into())]
        );
        assert_eq!(pieces("[giggles] okay [gasps] what [chuckles]", true), vec![Piece::Words("<giggle> Okay <gasp> what <chuckle>".into())]);
    }

    #[test]
    fn chat_format_tokens_are_not_read() {
        assert_eq!(pieces("<|python_tag|>user<|python_tag|>(\"Hello there. How are you?\")", true), vec![Piece::Words("(Hello there. How are you?)".into())]);
        assert_eq!(pieces("<|python_tag|>user", true), vec![]);
        assert_eq!(pieces("Fine, thanks.<|eot_id|>", false), vec![Piece::Words("Fine, thanks.".into())]);
    }

    #[test]
    fn plain_text_is_one_piece() {
        assert_eq!(pieces("Just words here.", false), vec![Piece::Words("Just words here.".into())]);
        assert_eq!(pieces("[pause]", false), vec![Piece::Quiet(0.8)]);
    }
}
