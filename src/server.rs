//! OpenAI-compatible HTTP server: /v1/chat/completions, /v1/completions (both with streaming over
//! server-sent events), /v1/models, and a chat page at /. A small HTTP/1.1 implementation on std::net, one connection thread
//! each; generation runs on the main thread, one request at a time. The cache keeps the tokens of the
//! last conversation, so a request that repeats it (a chat client resending the history) only reads
//! the new part. /api/models lists the GGUF files next to the model and /api/models/load switches to
//! one of them: the model thread drops the running model (freeing its GPU memory) and loads the other.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::json::{Json, num, obj, str};
use crate::sampler::Sampler;
use crate::tokenizer::Utf8Stream;
use crate::{ChatFormat, Engine, Opts};

mod agent;
mod download;
mod maya;
mod mcp;
mod schedule;
mod skills;
mod speech;

const MAX_BODY: usize = 32 << 20;

enum Kind {
    Chat,
    Completion,
}

/// Work for the model thread.
enum Work {
    Job(Job),
    /// switch to these models; the reply comes once a model is running again (Err: it stayed on the old one)
    Load(Plan, mpsc::Sender<Result<(), String>>),
}

/// The models asked for: `model`, with the voice models beside it (Some(true)), unloaded (Some(false)) or as they
/// are; or, with `call`, `model` for chats and `call` for voice calls, both kept loaded with the voice when they
/// all fit, else only the one `for_call` says comes first (the call model with the voice, or `model` alone).
#[derive(Clone, PartialEq)]
struct Plan {
    model: PathBuf,
    voice: Option<bool>,
    call: Option<PathBuf>,
    for_call: bool,
}

impl Plan {
    fn one(model: PathBuf, voice: Option<bool>) -> Plan {
        Plan { model, voice, call: None, for_call: false }
    }
}

/// The call model's context when it is kept beside the chat model (a call's prompt and its turns), the least it
/// may get, and the least the chat model keeps beside them both (the agent's prompt and a working chat; a chat
/// that outgrows it is compacted). A 3B with the voice on an 8 GB card gets about 13K: kept together then, so a
/// call starts at once.
const CALL_CTX: usize = 8192;
const MIN_CALL_CTX: usize = 4096;
const MIN_CHAT_CTX: usize = 8192;
/// A model's GPU memory beside its file and its session (whose size counts the activations): the kernels and the
/// conversion scratch. Its file is a little more than its GPU weights (a big embedding table stays in RAM).
const MODEL_WORK: usize = 64 << 20;

/// Whether `model`, `call` (when another file) and the voice (when not loaded yet) look like they fit in the GPU
/// memory free now with their least contexts: a quick check before trying, from the files' sizes.
fn pair_fits(model: &Path, call: &Path, voice: &speech::Voice) -> bool {
    let Ok(free) = crate::cuda::gpu_free() else { return false };
    let one = |p: &Path, ctx: usize| -> Option<usize> {
        let g = crate::gguf::Gguf::open(p).ok()?;
        let cfg = crate::model::Config::from_gguf(&g).ok()?;
        let size = std::fs::metadata(p).ok()?.len() as usize;
        Some(size + MODEL_WORK + crate::cuda::session_bytes(&cfg, ctx.min(cfg.max_ctx)))
    };
    let models = if call == model { one(model, MIN_CHAT_CTX) } else { one(model, MIN_CHAT_CTX).zip(one(call, MIN_CALL_CTX)).map(|(a, b)| a + b) };
    let voice = if voice.loaded() { 0 } else { voice.bytes() };
    // only plainly too much is not tried: the files overstate the weights (a big embedding table stays in RAM),
    // and loading them is the real test (each context is checked after)
    let Some(need) = models.map(|m| m + voice + crate::cuda::kept_free()) else { return false };
    let fits = need <= free + free / 8;
    if !fits {
        eprintln!("[calls: {} needs about {} MB beside the voice, {} MB is free: one model at a time]", if call == model { model_id(model) } else { format!("{} with {}", model_id(model), model_id(call)) }, need >> 20, free >> 20);
    }
    fits
}

/// How a reply is held in (0: not at all): penalties against repeating itself, as the Sampler has them
/// (`presence` on the words of the model's recent replies and of this one, `dry` on runs of tokens it said
/// before), `sentences`, the most it may say before it is stopped, and `spoken`: a reply that is voiced, whose
/// voice tags are kept to one the voice does well (VoiceTags).
#[derive(Clone, Copy, Default)]
struct Style {
    presence: f32,
    dry: f32,
    sentences: usize,
    spoken: bool,
    /// a repetition penalty (above 1; Sampler::with_repetition) over the prompt and the reply
    repetition: f32,
}

/// The voice tags a spoken reply keeps, as it is written: one pause and one sound at most, and only ones the
/// voice does well; any other ([smiles], [softly], a second [laughs]) is left out, with the space after it, and
/// so is a pause before the first word. A small model told "at most one" wrote 44 in 17 replies, making up its own,
/// and each pause tag is a cut in the voice; left in the chat, they taught it to write more.
#[derive(Default)]
struct VoiceTags {
    open: String, // a tag being written
    paused: bool, // a pause kept
    sounded: bool, // a sound kept
    spoke: bool, // a word was written
    skip_space: bool,
}

impl VoiceTags {
    const PAUSES: [&'static str; 2] = ["pause", "short pause"];
    /// the sounds picked by ear (speech.rs SOUNDS makes them)
    const SOUNDS: [&'static str; 22] = [
        "laughs", "laugh", "chuckles", "sighs", "sigh", "hmm", "gasps", "gasp", "shocked", "aha", "scoffs", "scoff", "grunts", "grunt",
        "groans", "groan", "moans", "moan", "whimpers", "whimper", "giggles", "thinks",
    ];

    /// The text to send of what the model just wrote.
    fn push(&mut self, s: &str) -> String {
        let mut out = String::new();
        for c in s.chars() {
            if !self.open.is_empty() {
                self.open.push(c);
                if c == ']' {
                    let tag = self.open[1..self.open.len() - 1].trim().to_lowercase();
                    let (pause, sound) = (Self::PAUSES.contains(&tag.as_str()), Self::SOUNDS.contains(&tag.as_str()));
                    // Maya ending the call: for the app, never spoken (the voice leaves it out)
                    if tag == "hang up" {
                        out.push_str(&self.open);
                        self.open.clear();
                        continue;
                    }
                    // a pause before the first word is only a later start; a sound there ([laughs] when asked to laugh) is kept
                    if (pause && self.spoke && !self.paused) || (sound && !self.sounded) {
                        out.push_str(&self.open);
                        self.paused |= pause;
                        self.sounded |= sound;
                    } else {
                        self.skip_space = true;
                    }
                    self.open.clear();
                } else if c == '\n' || self.open.len() > 40 {
                    // not a tag after all
                    out.push_str(&std::mem::take(&mut self.open));
                }
                continue;
            }
            if c == '[' {
                self.open.push(c);
                continue;
            }
            if std::mem::take(&mut self.skip_space) && c == ' ' {
                continue;
            }
            self.spoke |= c.is_alphanumeric();
            out.push(c);
        }
        out
    }
}

/// A generation request handed from a connection thread to the model thread.
struct Job {
    model: String, // the model the prompt was written for: if another one runs by then, the job is refused
    kind: Kind,
    prompt: String, // the full text the model reads (chat: already in the chat template)
    max_tokens: Option<usize>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    style: Style,
    stop: Vec<String>,
    /// sent before the reply: text the prompt already holds but the reply is read with (GLM's
    /// prompt ends with <think>, so a thinking reply starts with it too)
    lead: &'static str,
    events: mpsc::Sender<Event>,
}

enum Event {
    Text(String),
    Done { finish: &'static str, prompt_tokens: usize, completion_tokens: usize },
    Error(u16, String),
}

/// The running model, as the connection threads see it.
#[derive(Clone)]
struct Loaded {
    id: String,
    ctx: usize,
    chat_ok: bool,
    default_system: Option<String>,
    thinking: bool, // the chat template has a thinking switch
    glm: bool,      // GLM-4's chat format instead of ChatML
    mistral: bool,  // Mistral's chat format ([INST] ... [/INST]) instead of ChatML
    llama3: bool,   // Llama 3's chat format (<|start_header_id|> ... <|eot_id|>) instead of ChatML
    coder: bool,   // Qwen3-Coder's tool format (XML) instead of Qwen3's (JSON)
    backend: String, // where it runs, e.g. "GPU NVIDIA GeForce RTX 2060 SUPER (sm_75, ...)"
    about: Json,     // what the model is, from its file (about_model)
    tok: Arc<crate::tokenizer::Tokenizer>, // to count a prompt's tokens before sending it (the agent fits chats to the context)
}

#[derive(Default)]
struct Status {
    current: Option<Loaded>,
    /// a second model loaded beside it for voice calls (Plan::call)
    call: Option<Loaded>,
    /// the chat model, the call model and the voice all loaded together (Plan::call, when they fit)
    together: bool,
    loading: Option<String>, // the model being loaded; requests get 503 meanwhile
}

impl Status {
    /// The model a voice call talks to: the call model beside the chat model, else the chat model.
    fn for_call(&self) -> Option<Loaded> {
        self.call.clone().or_else(|| self.current.clone())
    }
}

/// What the connection threads share: the model's status, and where the models are.
struct Shared {
    status: Mutex<Status>,
    dirs: Mutex<Vec<PathBuf>>, // the model's folder, then any --models folders, and those added since (/api/models/folders)
    download_dir: Option<PathBuf>, // the first --models folder: where a downloaded model goes
    defaults: (f32, usize, f32, usize),
    think_default: bool,
    harness: Mutex<crate::tools::Settings>,
    /// the tool groups of the last agent request: after a model loads, the agent prompt's opening with these is
    /// read in ahead of time
    agent_groups: Mutex<Vec<String>>,
    /// made at startup and written into the chat page: the agent tools answer only requests carrying it
    token: String,
    /// the speech and speech-to-text models (--tts, --stt), for /v1/audio/speech and /v1/audio/transcriptions:
    /// loaded only while asked for (see speech::Voice)
    voice: Mutex<speech::Voice>,
}

impl Shared {
    /// The models folders now.
    fn dirs(&self) -> Vec<PathBuf> {
        self.dirs.lock().unwrap().clone()
    }

    /// The models folders, for /api/models/folders.
    fn folders_json(&self) -> Json {
        obj(vec![("folders", Json::Arr(self.dirs().iter().map(|d| str(&d.display().to_string())).collect()))])
    }
}

/// What a model is, from its file: name, size label, parameters (all of them, and those one token uses
/// in a mixture of experts) and quantization, for the chat page to tell the model about itself.
fn about_model(g: &crate::gguf::Gguf, id: &str) -> Json {
    let text = |k: &str| g.get(k).and_then(|v| v.as_str()).map(str);
    let total: u64 = g.tensors.iter().map(|t| t.n_elements()).sum();
    let experts: u64 = g.tensors.iter().filter(|t| t.name.contains("_exps.")).map(|t| t.n_elements()).sum();
    let mut fields = vec![
        ("name", text("general.name").unwrap_or_else(|| str(id))),
        ("parameters", num(total as f64)),
    ];
    if let Some(label) = text("general.size_label") {
        fields.push(("size_label", label));
    }
    if let Ok(c) = crate::model::Config::from_gguf(g) {
        if c.n_experts > 0 {
            let active = total - experts + experts * c.n_experts_used as u64 / c.n_experts as u64;
            fields.push(("active_parameters", num(active as f64)));
            fields.push(("experts", num(c.n_experts as f64)));
            fields.push(("experts_per_token", num(c.n_experts_used as f64)));
        }
    }
    // the quantization is the file name's last part by convention (…-Q3_K_M)
    if let Some(q) = id.rsplit(['-', '.']).next().filter(|q| q.starts_with(['Q', 'q', 'I', 'F', 'B'])) {
        fields.push(("quantization", str(&q.to_uppercase())));
    }
    obj(fields)
}

fn model_id(path: &Path) -> String {
    path.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string()
}

/// The GGUF files in the models folders, by name (a name found in an earlier folder wins).
fn list_models(dirs: &[PathBuf]) -> Vec<(String, PathBuf, u64)> {
    let mut found: Vec<_> = dirs
        .iter()
        .flat_map(|d| std::fs::read_dir(d).into_iter().flatten().flatten())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")))
        .map(|p| (model_id(&p), std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0), p))
        .map(|(id, size, p)| (id, p, size))
        .collect();
    let mut seen = std::collections::HashSet::new();
    found.retain(|m: &(String, PathBuf, u64)| seen.insert(m.0.clone()));
    found.sort_by_key(|m| m.0.to_lowercase());
    found
}

pub fn serve(o: &Opts, host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port)).map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?;
    let _ = PORT.set(port);
    // without a model the server starts with none loaded and waits for one to be picked (chat page, app,
    // or POST /api/models/load) from the --models folders
    let start_idle = o.model.as_os_str().is_empty();
    let own = (!start_idle).then(|| o.model.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new(".")).to_path_buf());
    let mut dirs: Vec<PathBuf> = own.into_iter().chain(o.models.iter().cloned()).collect();
    if dirs.is_empty() {
        dirs.push(PathBuf::from("."));
    }
    let shared = Arc::new(Shared {
        // a model given at startup shows as loading from the first request on
        status: Mutex::new(Status { loading: (!start_idle).then(|| model_id(&o.model)), ..Default::default() }),
        dirs: Mutex::new(dirs),
        download_dir: o.models.first().cloned(),
        defaults: (o.temp, o.top_k, o.top_p, o.max_new),
        think_default: !o.no_think,
        harness: Mutex::new(crate::tools::Settings::load()),
        agent_groups: Mutex::new(vec!["files".to_string()]),
        token: new_token(),
        voice: Mutex::new(speech::Voice { tts: o.tts.clone(), stt: o.stt.clone(), ..Default::default() }),
    });
    std::thread::spawn(mcp::start_all); // the MCP servers in mcp.json, while the model loads
    let (work, rx) = mpsc::channel::<Work>();
    schedule::start(work.clone(), shared.clone()); // scheduled tasks, once a model runs
    let s = shared.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let (work, s) = (work.clone(), s.clone());
            std::thread::spawn(move || {
                if let Err(e) = handle(conn, &work, &s) {
                    eprintln!("[connection: {e}]");
                }
            });
        }
    });

    // the model thread: the engines (GPU state included) never leave it. Each turn of this loop loads the models
    // of a Plan and serves them until another is asked for; they and their files are dropped before the next.
    // Started without one, it first waits for a model to be picked, and waits again if that one fails to load.
    let mut plan = Plan::one(o.model.clone(), None);
    let mut idle = start_idle;
    let mut last_good: Option<Plan> = None;
    let mut waiting: Option<(mpsc::Sender<Result<(), String>>, Option<String>)> = None; // who asked for this switch, and an error to report
    // (chat model, call model) pairs found not to fit beside each other and the voice: not tried again
    let mut no_pair: Vec<(PathBuf, PathBuf)> = Vec::new();
    if start_idle {
        eprintln!("[no model loaded: pick one in the chat page or the app]");
        eprintln!("chat in your browser: http://{host}:{port}/");
    }
    loop {
        if idle {
            shared.status.lock().unwrap().loading = None;
            let (next, reply) = loop {
                match rx.recv() {
                    Err(_) => return Ok(()),
                    Ok(Work::Job(job)) => {
                        let _ = job.events.send(Event::Error(503, "no model is loaded; pick one first".into()));
                    }
                    Ok(Work::Load(p, reply)) => break (p, reply),
                }
            };
            plan = next;
            waiting = Some((reply, None));
            idle = false;
        }
        // what this plan loads: the call model and the voice beside the model when they all fit, else one of them
        let pair = plan.call.clone().filter(|k| o.gpu && !no_pair.contains(&(plan.model.clone(), k.clone())) && pair_fits(&plan.model, k, &shared.voice.lock().unwrap()));
        let (path, voice, second) = match (&plan.call, &pair) {
            (Some(_), Some(k)) => (plan.model.clone(), Some(true), (*k != plan.model).then(|| k.clone())),
            (Some(k), None) if plan.for_call => (k.clone(), Some(true), None),
            (Some(_), None) => (plan.model.clone(), Some(false), None),
            (None, _) => (plan.model.clone(), plan.voice, None),
        };
        // the voice models first (the last models are gone by now): what they take, the next ones go without
        if let Some(on) = voice {
            if on != shared.voice.lock().unwrap().loaded() {
                shared.status.lock().unwrap().loading = Some(model_id(&path));
                speech::set_voice(&shared.voice, on);
                // their threads may have made the GPU context: the language model loads on this one
                let _ = crate::cuda::bind_thread();
            }
        }
        // the call model next, with the context a call needs: the chat model has the rest
        let call_id = second.as_deref().map(model_id);
        if let Some(c) = &call_id {
            shared.status.lock().unwrap().loading = Some(c.clone());
        }
        let t = Instant::now();
        let g2 = match second.as_deref().map(crate::gguf::Gguf::open).transpose() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[could not open the call model {}: {e}; one model at a time]", call_id.as_deref().unwrap_or(""));
                no_pair.push((plan.model.clone(), second.clone().unwrap_or_default()));
                continue;
            }
        };
        let mut engine2 = match g2.as_ref().map(|g| crate::load_with_draft(g, o.gpu, CALL_CTX, None, 0)).transpose() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[could not load the call model {}: {e}; one model at a time]", call_id.as_deref().unwrap_or(""));
                no_pair.push((plan.model.clone(), second.clone().unwrap_or_default()));
                continue;
            }
        };
        if let (Some(e), Some(c)) = (&engine2, &call_id) {
            eprintln!("[loaded {c} for calls on {} in {:.2}s, {}-token context]", e.backend.describe(), t.elapsed().as_secs_f64(), e.backend.ctx());
        }
        let id = model_id(&path);
        shared.status.lock().unwrap().loading = Some(id.clone());
        let t = Instant::now();
        // a failed switch goes back to the models that were running; failing at startup (or on the way back) ends the server
        let mut fall_back = |e: String, plan: &mut Plan| -> Result<(), String> {
            if start_idle && last_good.is_none() {
                // nothing ran yet: report it and wait for another pick
                eprintln!("[could not load {}: {e}]", model_id(&path));
                if let Some((w, _)) = waiting.take() {
                    let _ = w.send(Err(format!("could not load {}: {e}", model_id(&path))));
                }
                idle = true;
                return Ok(());
            }
            match (&mut waiting, &last_good) {
                (Some(w), Some(prev)) if prev != plan => {
                    eprintln!("[could not load {}: {e}; going back to {}]", model_id(&path), model_id(&prev.model));
                    w.1 = Some(format!("could not load {}: {e}", model_id(&path)));
                    *plan = prev.clone();
                    Ok(())
                }
                _ => Err(e),
            }
        };
        let g = match crate::gguf::Gguf::open(&path) {
            Ok(g) => g,
            Err(e) => {
                fall_back(e, &mut plan)?;
                continue;
            }
        };
        // the draft model only fits the model it was given with
        let draft = if path == o.model { o.draft.as_ref() } else { None };
        let mut engine = match crate::load_with_draft(&g, o.gpu, o.ctx.unwrap_or(0), draft, o.draft_k) {
            Ok(e) => e,
            Err(e) => {
                fall_back(e, &mut plan)?;
                continue;
            }
        };
        eprintln!(
            "[loaded {} on {} in {:.2}s, {}-token context{}]",
            id,
            engine.backend.describe(),
            t.elapsed().as_secs_f64(),
            engine.backend.ctx(),
            if engine.draft.is_some() { ", with a draft model for greedy requests" } else { "" }
        );
        // kept together only when each has room to work: else one at a time from now on
        if let Some(k) = &pair {
            let least = crate::model::Config::from_gguf(&g).map_or(MIN_CHAT_CTX, |c| c.max_ctx.min(MIN_CHAT_CTX)).min(o.ctx.unwrap_or(usize::MAX));
            if engine.backend.ctx() < least || engine2.as_ref().is_some_and(|e| e.backend.ctx() < MIN_CALL_CTX) {
                eprintln!("[{id}, the call model and the voice leave too little room for their contexts: one model at a time]");
                no_pair.push((plan.model.clone(), k.clone()));
                continue;
            }
        }
        if last_good.is_none() && !start_idle {
            eprintln!("chat in your browser: http://{host}:{port}/");
            eprintln!("OpenAI-compatible API: http://{host}:{port}/v1 (/v1/chat/completions, /v1/completions, /v1/models)");
        }
        if ChatFormat::detect(&g).is_err() {
            eprintln!("[note: no chat template this server knows (ChatML, GLM, Mistral or Llama 3) in this model, so /v1/chat/completions is off]");
        }
        *shared.status.lock().unwrap() = Status {
            current: Some(loaded(&id, &g, &engine)),
            call: g2.as_ref().zip(engine2.as_ref()).zip(call_id.as_deref()).map(|((g, e), c)| loaded(c, g, e)),
            together: pair.is_some(),
            loading: None,
        };
        if let Some((w, err)) = waiting.take() {
            let _ = w.send(err.map_or(Ok(()), Err));
        }
        last_good = Some(plan.clone());
        // a few short replies first (the weights into memory), then the agent's opening read in, unless a
        // request comes in meanwhile: it goes first
        let warm = shared.harness.lock().unwrap().warm_up;
        WARMING.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut early = if warm { warm_up(&mut engine, &id, &rx) } else { None };
        if early.is_none() {
            warm_agent(&mut engine, &shared);
        }
        WARMING.store(false, std::sync::atomic::Ordering::Relaxed);

        let next = loop {
            let work = match early.take() {
                Some(w) => Ok(w),
                None => rx.recv(),
            };
            match work {
                Err(_) => return Ok(()),
                Ok(Work::Job(job)) => match (engine2.as_mut(), call_id.as_deref()) {
                    (Some(e), Some(c)) if job.model == c => run(e, c, job),
                    _ => run(&mut engine, &id, job),
                },
                // already running: the same models (with both loaded, which comes first no longer matters), or
                // this one model with the voice as asked
                Ok(Work::Load(p, reply)) if p == plan || (pair.is_some() && p.model == plan.model && p.call == plan.call) => {
                    plan = p;
                    let _ = reply.send(Ok(()));
                }
                Ok(Work::Load(p, reply)) if p.call.is_none() && p.model == path && p.voice.is_none_or(|v| v == shared.voice.lock().unwrap().loaded()) => {
                    let _ = reply.send(Ok(()));
                }
                // the model alone, with a call model already found not to fit beside it: as it would load
                Ok(Work::Load(p, reply))
                    if !p.for_call && p.model == path && second.is_none() && !shared.voice.lock().unwrap().loaded() && p.call.as_ref().is_some_and(|k| no_pair.contains(&(p.model.clone(), k.clone()))) =>
                {
                    plan = p;
                    let _ = reply.send(Ok(()));
                }
                Ok(Work::Load(p, reply)) => break (p, reply),
            }
        };
        let (next, reply) = next;
        let also = match (&next.call, next.voice) {
            (Some(k), _) if *k != next.model => format!(", with {} for calls when they fit", model_id(k)),
            (Some(_), _) => ", with the voice when it fits".into(),
            (None, Some(true)) => ", with the voice".into(),
            (None, Some(false)) => ", without the voice".into(),
            (None, None) => String::new(),
        };
        eprintln!("[switching from {id} to {}{also}]", model_id(&next.model));
        plan = next;
        waiting = Some((reply, None));
    }
}

/// What a loaded model is, for the connection threads.
fn loaded(id: &str, g: &crate::gguf::Gguf, engine: &Engine) -> Loaded {
    let fmt = ChatFormat::detect(g).ok();
    Loaded {
        id: id.to_string(),
        ctx: engine.backend.ctx(),
        chat_ok: fmt.is_some(),
        thinking: fmt.as_ref().is_some_and(|f| f.thinking),
        glm: fmt.as_ref().is_some_and(|f| f.glm),
        mistral: fmt.as_ref().is_some_and(|f| f.mistral),
        llama3: fmt.as_ref().is_some_and(|f| f.llama3),
        coder: fmt.as_ref().is_some_and(|f| f.coder),
        backend: engine.backend.describe(),
        about: about_model(g, id),
        tok: engine.tok.clone(),
        default_system: fmt.and_then(|f| f.default_system),
    }
}

/// The model is warming up after it loaded (warm_up, then warm_agent): /api/models says so, for the app to show.
static WARMING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A few short replies right after a model loads (the harness's warm_up setting): a mixture of experts then
/// has the experts they used in RAM and in the GPU's cache before the first chat, which otherwise runs slower
/// while it brings them in. Stops as soon as a request comes in, and hands it back to be served first.
fn warm_up(engine: &mut Engine, id: &str, rx: &mpsc::Receiver<Work>) -> Option<Work> {
    const PROMPTS: [&str; 3] = [
        "def parse_iso_date(text):\n    \"\"\"Parse an ISO 8601 date string into (year, month, day).\"\"\"\n",
        "The three habits that make technical documentation easy to read are",
        "Q: How does a hash map deal with two keys that land in the same bucket?\nA:",
    ];
    let t = Instant::now();
    let mut tokens = 0;
    for prompt in PROMPTS {
        match rx.try_recv() {
            Ok(w) => return Some(w),
            Err(mpsc::TryRecvError::Disconnected) => return None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
        let (tx, events) = mpsc::channel();
        let job = Job { model: id.to_string(), kind: Kind::Completion, prompt: prompt.into(), max_tokens: Some(48), temperature: 0.0, top_k: 1, top_p: 1.0, seed: 1, style: Style::default(), stop: Vec::new(), lead: "", events: tx };
        run(engine, id, job);
        tokens += events.try_iter().map(|e| if let Event::Done { completion_tokens, .. } = e { completion_tokens } else { 0 }).sum::<usize>();
    }
    eprintln!("[warmed up: {tokens} tokens in {:.2}s]", t.elapsed().as_secs_f64());
    None
}

/// Reads the agent prompt's opening (instructions, memory, tool descriptions: a few thousand tokens) into the
/// cache right after a model loads, while the user types: a chat's first reply then reads only its message.
fn warm_agent(engine: &mut Engine, shared: &Shared) {
    let Some(loaded) = shared.status.lock().unwrap().current.clone() else { return };
    let groups = shared.agent_groups.lock().unwrap().clone();
    let Some(text) = agent::prompt_opening(&groups, shared, &loaded) else { return };
    let t = Instant::now();
    let ids = engine.tok.encode(&text);
    let keep = engine.hist.iter().zip(&ids).take_while(|(a, b)| a == b).count();
    let keep = engine.truncate(keep);
    match engine.read_in(&ids[keep..]) {
        Ok(()) => eprintln!("[read the agent's opening in ahead: {} tokens in {:.2}s]", ids.len() - keep, t.elapsed().as_secs_f64()),
        Err(e) => eprintln!("[could not read the agent's opening in: {e}]"),
    }
}

fn run(engine: &mut Engine, model: &str, job: Job) {
    if job.model != model {
        let _ = job.events.send(Event::Error(409, format!("the model changed to {model} while this request waited; send it again")));
        return;
    }
    // GHOST_INFER_WIZARD=<folder>: a voice call's replies written by hand instead of by the model (to tell a
    // weak model from a weak voice): the prompt goes to <folder>\ask.txt, the reply is taken from reply.txt
    if let (true, Some(dir)) = (job.style.spoken, std::env::var_os("GHOST_INFER_WIZARD")) {
        let dir = PathBuf::from(dir);
        let (ask, reply) = (dir.join("ask.txt"), dir.join("reply.txt"));
        let _ = std::fs::remove_file(&reply);
        let _ = std::fs::write(&ask, &job.prompt);
        let t = Instant::now();
        let text = loop {
            if let Ok(r) = std::fs::read_to_string(&reply) {
                if r.ends_with('\n') {
                    let _ = std::fs::remove_file(&reply);
                    break r.trim().to_string();
                }
            }
            if t.elapsed().as_secs() > 180 {
                break "[sighs] Sorry, I lost my train of thought.".to_string();
            }
            std::thread::sleep(std::time::Duration::from_millis(150));
        };
        let _ = std::fs::remove_file(&ask);
        let _ = job.events.send(Event::Text(text));
        let _ = job.events.send(Event::Done { finish: "stop", prompt_tokens: 0, completion_tokens: 0 });
        return;
    }
    let t0 = Instant::now();
    let ids = engine.tok.encode(&job.prompt);
    let ctx = engine.backend.ctx();
    if ids.is_empty() || ids.len() >= ctx {
        let _ = job.events.send(Event::Error(400, format!("the prompt has {} tokens; the context holds {ctx}", ids.len())));
        return;
    }
    // Give the caches to the conversation this request continues, then keep the part of the prompt they
    // already hold (at least one token is read anew, which gives the logits for the first reply token).
    // The card holds one context at a time and a client keeps coming back to the same conversation, so the
    // conversation holding the caches now is copied out to host memory whenever a different one asks for
    // them, and copied back in when its own client returns.
    let keep = engine.adopt(&ids).min(ids.len() - 1);
    let keep = engine.truncate(keep);
    let reused = keep;
    // Every conversation in a session opens with the same head: everything up to the end of the system
    // block. That head is what a subagent shares with the conversation that spawned it and nothing else, and
    // with a tool list it can be tens of thousands of tokens, so it is worth reading once per session rather
    // than once per conversation. A note is left at its end and held apart from the ring, which would have
    // dropped it first for being the oldest; the head is read from there and only what follows it is read
    // again. If the prompt does not start with a system block, or the head does not line up with what the
    // caches hold, the plain prefix is kept instead.
    let keep = match job.prompt.find("<|im_end|>") {
        Some(at) => {
            let head = engine.tok.encode(&job.prompt[..at + "<|im_end|>\n".len()]);
            let h = head.len();
            if h > keep && h < ids.len() && ids[..h] == head[..] && engine.read_head(&ids[keep..h]).is_ok() { h } else { keep }
        }
        None => keep,
    };
    // a recurrent (DeltaNet) model goes back only to a snapshot, and one is taken where a read ends: so the
    // prompt is read in two, up to the reply's own header and then the header. The next prompt may write this
    // turn's header differently (without its empty thinking once a message follows it, as the compaction's
    // question or a nudge does): it then goes back to the header, not to the start.
    let keep = match REPLY_HEADS.iter().filter_map(|h| job.prompt.rfind(h)).max() {
        Some(at) => {
            let head = engine.tok.encode(&job.prompt[..at]);
            let b = head.len();
            if b > keep && b < ids.len() && ids[..b] == head[..] && engine.read_in(&ids[keep..b]).is_ok() { b } else { keep }
        }
        None => keep,
    };
    let max_new = job.max_tokens.unwrap_or(usize::MAX).min(ctx - ids.len()).max(1);
    // GHOST_INFER_SHOW_PROMPT=1: each prompt as the model reads it, to the log
    if std::env::var_os("GHOST_INFER_SHOW_PROMPT").is_some() {
        eprintln!("[prompt, {} tokens, {} kept from the cache]\n{}\n[end of prompt]", ids.len(), keep, job.prompt);
    }
    let mut sampler = Sampler::new(job.temperature, job.top_k, job.top_p, job.seed);
    // a Llama 3 reply never writes a turn's header: a small model unsure how to begin can start one ("assistant",
    // then <|end_header_id|>) and then write nothing but headers
    if job.prompt.ends_with(LLAMA3_REPLY_HEAD) {
        sampler = sampler.with_ban(["<|start_header_id|>", "<|end_header_id|>", "assistant"].iter().filter_map(|t| engine.tok.token_id(t)));
    }
    if job.style.presence > 0.0 {
        let said: Vec<u32> = recent_replies(&job.prompt, 4).iter().flat_map(|r| engine.tok.encode(&without_tags(r))).collect();
        let tok = engine.tok.clone();
        sampler = sampler.with_presence(job.style.presence, said, move |id| is_word(&tok.token_bytes(id, false)));
    }
    if job.style.repetition > 1.0 {
        sampler = sampler.with_repetition(job.style.repetition, ids.iter().copied());
    }
    if job.style.dry > 0.0 {
        // its own recent replies, a line break after each, then what it says: only repeating itself is held back.
        // (Over the whole context, the system prompt's ordinary phrases were held back too, and the model fled to
        // rare spellings: all lowercase, "i’m" with a curly apostrophe, which the voice read as gibberish.)
        let tok = engine.tok.clone();
        let line = engine.tok.encode("\n");
        let history: Vec<u32> = recent_replies(&job.prompt, DRY_REPLIES).iter().flat_map(|r| engine.tok.encode(r).into_iter().chain(line.iter().copied())).collect();
        sampler = sampler.with_dry(job.style.dry, history, move |id| {
            let b = tok.token_bytes(id, false);
            b.is_empty() || b.contains(&b'\n')
        });
    }

    let tok = engine.tok.clone();
    let mut utf8 = Utf8Stream::default();
    // text not sent yet: it could be the start of a stop string (the lead goes out with the first token)
    let mut held = job.lead.to_string();
    let mut hit_stop = false;
    let events = job.events.clone();
    let stops = job.stop.clone();
    let limit = job.style.sentences;
    let mut all = String::new(); // the reply so far, for its sentences
    let mut tags = job.style.spoken.then(VoiceTags::default);
    let result = engine.generate_with(&ids[keep..], max_new, &mut sampler, &mut |id| {
        let piece = utf8.push(&tok.token_bytes(id, true));
        let piece = match tags.as_mut() {
            Some(t) => t.push(&piece),
            None => piece,
        };
        held.push_str(&piece);
        if limit > 0 {
            all.push_str(&piece);
            if let Some(end) = nth_sentence_end(&all, limit) {
                // what follows the last sentence is this token's, not sent yet
                let tail = all.len() - end;
                held.truncate(held.len().saturating_sub(tail));
                hit_stop = true;
                let _ = events.send(Event::Text(std::mem::take(&mut held)));
                return Ok(false);
            }
        }
        if let Some(at) = stops.iter().filter_map(|s| held.find(s.as_str())).min() {
            held.truncate(at);
            hit_stop = true;
            let _ = events.send(Event::Text(std::mem::take(&mut held)));
            return Ok(false);
        }
        let keep_back = stops.iter().map(|s| partial_suffix(&held, s)).max().unwrap_or(0);
        let ready = held.len() - keep_back;
        if ready > 0 {
            let rest = held.split_off(ready);
            let text = std::mem::replace(&mut held, rest);
            if events.send(Event::Text(text)).is_err() {
                return Ok(false); // the client went away
            }
        }
        Ok(true)
    });
    match result {
        Ok(st) => {
            if !held.is_empty() {
                let _ = events.send(Event::Text(held));
            }
            let finish = if hit_stop || st.stopped_on.is_some() { "stop" } else { "length" };
            let _ = job.events.send(Event::Done { finish, prompt_tokens: ids.len(), completion_tokens: st.gen_tokens });
            eprintln!(
                "[{}: prompt {} tokens ({} new), {} generated in {:.2}s ({:.1} tok/s), finish {finish}{}]",
                match job.kind {
                    Kind::Chat => "chat",
                    Kind::Completion => "completion",
                },
                ids.len(),
                ids.len() - reused,
                st.gen_tokens,
                t0.elapsed().as_secs_f64(),
                st.gen_tokens as f64 / st.gen_secs.max(1e-9),
                if st.rounds > 0 { format!(", lookup kept {} of {} guessed tokens in {} checks", st.accepted, st.proposed, st.rounds) } else { String::new() },
            );
            if let Some(s) = crate::cuda::moe_stats_take(st.gen_tokens) {
                eprintln!("{s}");
            }
        }
        Err(e) => {
            // the cache may be half-written: start the next request from scratch
            engine.truncate(0);

            let _ = job.events.send(Event::Error(500, e));
        }
    }
}

/// Length of the longest end of `text` that is the start of `stop` (and could still grow into it).
fn partial_suffix(text: &str, stop: &str) -> usize {
    (1..stop.len().min(text.len() + 1))
        .rev()
        .find(|&k| stop.is_char_boundary(k) && text.is_char_boundary(text.len() - k) && text.ends_with(&stop[..k]))
        .unwrap_or(0)
}

/// The model's replies DRY looks back over.
const DRY_REPLIES: usize = 8;

/// Where the `n`th sentence of `text` ends (after its . ! or ?, once a space follows), if it does yet. An
/// ellipsis ("more... human") and anything in a voice tag ([short pause]) do not end one, and a sentence of fewer
/// than four words ("Oh!", "Hmm.") runs on into the next: the limit cut a reply off at "N... like tiger... oh!".
fn nth_sentence_end(text: &str, n: usize) -> Option<usize> {
    let mut count = 0;
    let mut depth = 0;
    let mut words = 0;
    let mut prev = ' ';
    let mut it = text.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let next = it.peek().map(|&(_, c)| c);
        match c {
            '[' => depth += 1,
            ']' if depth > 0 => depth -= 1,
            '.' | '!' | '?' if depth == 0 && next.is_some_and(char::is_whitespace) && !(c == '.' && prev == '.') && words >= 4 => {
                count += 1;
                words = 0;
                if count == n {
                    return Some(i + 1);
                }
            }
            _ if depth == 0 && c.is_alphanumeric() && !prev.is_alphanumeric() && prev != '\'' => words += 1,
            _ => {}
        }
        prev = c;
    }
    None
}

/// Where the model's reply starts in a chat prompt: ChatML, GLM, Mistral and Llama 3.
const REPLY_HEADS: [&str; 4] = ["<|im_start|>assistant", "<|assistant|>", "[/INST]", "<|start_header_id|>assistant<|end_header_id|>"];

/// The model's last `n` finished replies in a chat prompt (any of REPLY_HEADS' formats), for the presence penalty.
fn recent_replies(prompt: &str, n: usize) -> Vec<&str> {
    let mut out = Vec::new();
    for head in REPLY_HEADS {
        let mut rest = prompt;
        while let Some(at) = rest.find(head) {
            rest = &rest[at + head.len()..];
            // a finished reply ends at the next special marker; the open one at the end (the reply to come) has none
            if let Some(end) = ["<|", "</s>"].iter().filter_map(|m| rest.find(m)).min() {
                out.push(rest[..end].trim());
            }
        }
    }
    out.retain(|r| !r.is_empty());
    let skip = out.len().saturating_sub(n);
    out.split_off(skip)
}

/// A reply without its voice tags ([pause], [laughs]): saying one does not make the next unwelcome.
fn without_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0;
    for c in text.chars() {
        match c {
            '[' => depth += 1,
            ']' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// A token the presence penalty counts: a word of at least four letters (not "the", "you", punctuation or
/// markup, which every reply needs; never a special token such as <|im_end|>, or a reply could not end).
fn is_word(bytes: &[u8]) -> bool {
    let t = String::from_utf8_lossy(bytes);
    !t.contains("<|") && t.chars().filter(|c| c.is_alphabetic()).count() >= 4
}

#[cfg(test)]
mod penalty_tests {
    use super::*;

    #[test]
    fn recent_replies_are_the_finished_ones() {
        let p = "<|im_start|>system\nBe kind.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n[short pause] Hey there.<|im_end|>\n<|im_start|>user\nhow are you<|im_end|>\n<|im_start|>assistant\nGood, you?<|im_end|>\n<|im_start|>user\nfine<|im_end|>\n<|im_start|>assistant\n";
        assert_eq!(recent_replies(p, 4), vec!["[short pause] Hey there.", "Good, you?"]);
        assert_eq!(recent_replies(p, 1), vec!["Good, you?"]);
        assert_eq!(without_tags("[short pause] Hey [laughs] there."), " Hey  there.");
        assert!(is_word(b" figure") && !is_word(b" the") && !is_word(b"<|im_end|>"));
    }

    /// A pause and a sound the voice does well are kept, after the first word; the rest go, written whole or in pieces.
    #[test]
    fn a_spoken_reply_keeps_one_good_tag() {
        let reply = "[pauses] Oh! [smiles] A bright laugh [laughs] there. [pause] Did that [softly] feel like you? [laughs] [pause]";
        let mut t = VoiceTags::default();
        assert_eq!(t.push(reply), "Oh! A bright laugh [laughs] there. [pause] Did that feel like you? ");
        let mut t = VoiceTags::default();
        let streamed: String = ["Hi", " [la", "ughs", "] ok", " [smi", "les] ", "fine"].iter().map(|p| t.push(p)).collect();
        assert_eq!(streamed, "Hi [laughs] ok fine");
        // asked to laugh, it laughs first
        let mut t = VoiceTags::default();
        assert_eq!(t.push("[laughs] Okay, that one got me. [pause] Again?"), "[laughs] Okay, that one got me. [pause] Again?");
        let mut t = VoiceTags::default();
        assert_eq!(t.push("a [note in brackets that runs on far too long to be a tag] b"), "a [note in brackets that runs on far too long to be a tag] b");
    }

    #[test]
    fn sentences_end_at_a_stop_and_a_space() {
        // "Hey!" is too short to be one: it runs on into the next
        let t = "Hey! I was going to be more... human. [short pause] Right? Okay, that settles it then. And one more thing here.";
        assert_eq!(&t[..nth_sentence_end(t, 1).unwrap()], "Hey! I was going to be more... human.");
        assert_eq!(&t[..nth_sentence_end(t, 2).unwrap()], "Hey! I was going to be more... human. [short pause] Right? Okay, that settles it then.");
        assert_eq!(nth_sentence_end(t, 3), None); // the last stop has no space after it yet
        assert_eq!(nth_sentence_end("It took 3.5 seconds to get here, wow", 1), None);
        let r = "Hmm. N... like tiger... oh! Wait, I've got it now. Next one.";
        assert_eq!(&r[..nth_sentence_end(r, 1).unwrap()], "Hmm. N... like tiger... oh!");
        assert_eq!(&r[..nth_sentence_end(r, 2).unwrap()], "Hmm. N... like tiger... oh! Wait, I've got it now.");
    }
}

// ------------------------------------------------------------------ HTTP

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
    host: String,   // the Host header
    origin: String, // the Origin header (empty if none)
    token: String,  // the X-Ghost-Token header (empty if none)
}

fn read_request(conn: &mut TcpStream) -> Result<Request, String> {
    let mut reader = BufReader::new(conn.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("empty request")?.to_string();
    let path = parts.next().ok_or("no path")?.split('?').next().unwrap_or("").to_string();
    let (mut len, mut expect_continue) = (0usize, false);
    let (mut host, mut origin, mut token) = (String::new(), String::new(), String::new());
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).map_err(|e| e.to_string())? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => len = v.trim().parse().map_err(|_| "bad Content-Length")?,
                "expect" => expect_continue = v.trim().eq_ignore_ascii_case("100-continue"),
                "host" => host = v.trim().to_string(),
                "origin" => origin = v.trim().to_string(),
                "x-ghost-token" => token = v.trim().to_string(),
                _ => {}
            }
        }
    }
    if len > MAX_BODY {
        return Err(format!("request body of {len} bytes is too large"));
    }
    if expect_continue {
        conn.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").map_err(|e| e.to_string())?;
    }
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok(Request { method, path, body, host, origin, token })
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

fn status_text(code: u16) -> &'static str {
    match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        403 => "Forbidden",
        405 => "Method Not Allowed",
        _ => "Internal Server Error",
    }
}

fn respond(conn: &mut TcpStream, code: u16, body: &Json) -> Result<(), String> {
    let text = body.to_string();
    let head = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\n{CORS}Connection: close\r\n\r\n",
        status_text(code),
        text.len()
    );
    conn.write_all(head.as_bytes()).and_then(|_| conn.write_all(text.as_bytes())).map_err(|e| e.to_string())
}

/// The built-in chat page (src/chat.html, compiled into the program), with this run's tool token written
/// in. It is sent without CORS headers, so another website cannot read the token out of it.
/// POST /v1/audio/speech, as OpenAI's: {"input": text, "voice": a speaker id (a number, or "maya": 4, the
/// speaker the Maya fine-tune learned), "response_format": "wav" (the default, sent when done) or "pcm" (16-bit
/// mono samples at the X-Sample-Rate header's rate, sent as they are made)}, and also "conversation" (an id:
/// the turns spoken under it are this one's context, which keeps the voice), "temperature", "top_k", "seed".
fn speech_request(conn: &mut TcpStream, req: &Request, shared: &Shared) -> Result<(), String> {
    if req.method == "OPTIONS" {
        let head = format!("HTTP/1.1 204 No Content\r\n{CORS}Content-Length: 0\r\nConnection: close\r\n\r\n");
        return conn.write_all(head.as_bytes()).map_err(|e| e.to_string());
    }
    if req.method != "POST" {
        return respond(conn, 405, &error_body("method not allowed"));
    }
    let (sp, available) = {
        let v = shared.voice.lock().unwrap();
        (v.speech.clone(), v.tts.is_some())
    };
    let Some(sp) = sp else {
        let why = if available { "the voice is off: it loads with a model, POST /api/models/load {\"id\": ..., \"voice\": true}" } else { "no speech model: start the server with --tts <CSM folder>" };
        return respond(conn, if available { 409 } else { 404 }, &error_body(why));
    };
    let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
        Ok(b) => b,
        Err(e) => return respond(conn, 400, &error_body(&format!("request body: {e}"))),
    };
    let text = body.get("input").and_then(Json::as_str).unwrap_or("").trim().to_string();
    if text.is_empty() {
        return respond(conn, 400, &error_body("send the text to speak as \"input\""));
    }
    let speaker = match body.get("voice") {
        Some(Json::Str(v)) if v.eq_ignore_ascii_case("maya") => 4,
        Some(Json::Str(v)) => match v.parse() {
            Ok(n) => n,
            Err(_) => return respond(conn, 400, &error_body(&format!("voice \"{v}\": give a speaker number, or \"maya\""))),
        },
        Some(v) => v.as_f64().unwrap_or(4.0) as u32,
        None => 4,
    };
    let pcm = match body.get("response_format").and_then(Json::as_str).unwrap_or("wav") {
        "wav" => false,
        "pcm" => true,
        f => return respond(conn, 400, &error_body(&format!("response_format \"{f}\": wav or pcm"))),
    };
    let f = |k: &str| body.get(k).and_then(Json::as_f64);
    let seed = f("seed").map_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1), |s| s as u64);
    let sampling = crate::csm::Sampling { temperature: f("temperature").map_or(sp.temperature, |v| v as f32), top_k: f("top_k").map_or(sp.top_k, |v| v as usize), seed };
    // room for slow speech: about two frames (160 ms) a character, at most a minute
    let max_frames = (text.chars().count() * 2 + 50).min(750);
    let (tx, events) = mpsc::channel();
    let conversation = body.get("conversation").and_then(Json::as_str).map(String::from);
    sp.send(speech::SpeechJob { text, speaker, sampling, max_frames, conversation, events: tx })?;
    let s16 = |samples: &[f32]| -> Vec<u8> { samples.iter().flat_map(|&v| ((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()).collect() };
    if pcm {
        // each 80 ms of audio goes out on its own: small writes must not wait for the last one's ACK
        let _ = conn.set_nodelay(true);
        let head = format!("HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nX-Sample-Rate: {}\r\nAccess-Control-Expose-Headers: X-Sample-Rate\r\nCache-Control: no-store\r\nTransfer-Encoding: chunked\r\n{CORS}Connection: close\r\n\r\n", sp.sample_rate);
        conn.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
        for ev in events {
            match ev {
                // a failed write: the client is gone, and dropping `events` stops the speech
                speech::SpeechEvent::Audio(a) => write_chunk(conn, &s16(&a))?,
                speech::SpeechEvent::Done => break,
                speech::SpeechEvent::Error(e) => {
                    eprintln!("[speech: {e}]");
                    break;
                }
            }
        }
        return conn.write_all(b"0\r\n\r\n").map_err(|e| e.to_string());
    }
    let mut audio = Vec::new();
    for ev in events {
        match ev {
            speech::SpeechEvent::Audio(a) => audio.extend(a),
            speech::SpeechEvent::Done => break,
            speech::SpeechEvent::Error(e) => return respond(conn, 500, &error_body(&e)),
        }
    }
    let wav = crate::csm::wav(&audio, sp.sample_rate);
    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: {}\r\nCache-Control: no-store\r\n{CORS}Connection: close\r\n\r\n", wav.len());
    conn.write_all(head.as_bytes()).and_then(|()| conn.write_all(&wav)).map_err(|e| e.to_string())
}

/// POST /v1/audio/transcriptions: a WAV file as the body (or as the file of an OpenAI-style multipart form);
/// {"text", "language", "no_speech" (how likely it held no speech, 0..1), "duration"}.
fn transcription_request(conn: &mut TcpStream, req: &Request, shared: &Shared) -> Result<(), String> {
    if req.method == "OPTIONS" {
        let head = format!("HTTP/1.1 204 No Content\r\n{CORS}Content-Length: 0\r\nConnection: close\r\n\r\n");
        return conn.write_all(head.as_bytes()).map_err(|e| e.to_string());
    }
    if req.method != "POST" {
        return respond(conn, 405, &error_body("method not allowed"));
    }
    let (l, available) = {
        let v = shared.voice.lock().unwrap();
        (v.listener.clone(), v.stt.is_some())
    };
    let Some(l) = l else {
        let why = if available { "speech-to-text is off: it loads with a model, POST /api/models/load {\"id\": ..., \"voice\": true}" } else { "no speech-to-text model: start the server with --stt <whisper.gguf>" };
        return respond(conn, if available { 409 } else { 404 }, &error_body(why));
    };
    // the WAV wherever it starts (a multipart form's boundary and headers come before it)
    let Some(at) = req.body.windows(4).position(|w| w == b"RIFF") else {
        return respond(conn, 400, &error_body("send a WAV file"));
    };
    let audio = match crate::whisper::read_wav(&req.body[at..]) {
        Ok(a) => a,
        Err(e) => return respond(conn, 400, &error_body(&e)),
    };
    let duration = audio.len() as f64 / crate::whisper::RATE as f64;
    match l.transcribe(audio) {
        Ok(h) => respond(conn, 200, &obj(vec![("text", str(&h.text)), ("language", str(&h.language)), ("no_speech", num(h.no_speech)), ("duration", num(duration))])),
        Err(e) => respond(conn, 500, &error_body(&e)),
    }
}

/// One piece of a chunked response.
fn write_chunk(conn: &mut TcpStream, bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }
    conn.write_all(format!("{:x}\r\n", bytes.len()).as_bytes()).and_then(|()| conn.write_all(bytes)).and_then(|()| conn.write_all(b"\r\n")).map_err(|e| e.to_string())
}

fn respond_page(conn: &mut TcpStream, token: &str) -> Result<(), String> {
    let page = include_str!("chat.html").replace("__GHOST_TOKEN__", token);
    respond_file(conn, page.as_bytes(), "text/html; charset=utf-8", "no-store")
}

/// A random hex token (the standard library's hasher keys come from the operating system's randomness).
fn new_token() -> String {
    use std::hash::{BuildHasher, Hasher};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    (0..2u64)
        .map(|i| {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u64(i);
            h.write_u128(nanos);
            h.write_u32(std::process::id());
            format!("{:016x}", h.finish())
        })
        .collect()
}

/// A request the agent tools may answer: from this computer, addressed to localhost (so a website that
/// points its own name at 127.0.0.1 cannot reach them), from the chat page or no page, carrying the token.
fn tools_allowed(req: &Request, conn: &TcpStream, token: &str) -> Result<(), &'static str> {
    if !conn.peer_addr().is_ok_and(|a| a.ip().is_loopback()) {
        return Err("the agent tools only answer requests from this computer");
    }
    let host = req.host.rsplit_once(':').map_or(req.host.as_str(), |(h, _)| h);
    if !matches!(host, "127.0.0.1" | "localhost" | "[::1]") {
        return Err("the agent tools only answer requests addressed to localhost or 127.0.0.1");
    }
    if !req.origin.is_empty() && !req.origin.ends_with(&format!("//{}", req.host)) {
        return Err("the agent tools do not answer other websites");
    }
    if req.token != token {
        return Err("missing or wrong tool token: reload the chat page");
    }
    Ok(())
}

/// The folders of web pages the agent wrote (agent::show_page), served at /preview/<n>/... so a browser tool,
/// which keeps file:// shut, can open them over http. Nothing else on disk is served.
static PREVIEWS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
static PORT: OnceLock<u16> = OnceLock::new();

/// http://127.0.0.1:<port>/preview/<n>/<name> for a page the agent wrote; its folder becomes servable.
pub(crate) fn preview_url(page: &Path) -> Option<String> {
    let dir = page.parent()?.to_path_buf();
    let name = page.file_name()?.to_str()?;
    let mut list = PREVIEWS.lock().unwrap();
    let n = list.iter().position(|d| *d == dir).unwrap_or_else(|| {
        list.push(dir);
        list.len() - 1
    });
    let name: String = name.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect();
    Some(format!("http://127.0.0.1:{}/preview/{n}/{name}", PORT.get()?))
}

/// A file from a preview folder (see PREVIEWS): only to this computer, addressed to localhost (a website
/// pointing its own name at 127.0.0.1 cannot read it), and only from inside that folder.
fn respond_preview(conn: &mut TcpStream, req: &Request, rest: &str) -> Result<(), String> {
    let host = req.host.rsplit_once(':').map_or(req.host.as_str(), |(h, _)| h);
    if !conn.peer_addr().is_ok_and(|a| a.ip().is_loopback()) || !matches!(host, "127.0.0.1" | "localhost" | "[::1]") {
        return respond(conn, 403, &error_body("previews are for this computer only"));
    }
    let rest = rest.split(['?', '#']).next().unwrap_or("");
    let (n, file) = rest.split_once('/').unwrap_or((rest, ""));
    let file = if file.is_empty() { "index.html".to_string() } else { percent_decode(file) };
    let dir = n.parse::<usize>().ok().and_then(|n| PREVIEWS.lock().unwrap().get(n).cloned());
    let found = dir.and_then(|d| {
        let (d, f) = (std::fs::canonicalize(&d).ok()?, std::fs::canonicalize(d.join(&file)).ok()?);
        f.starts_with(&d).then_some(f)
    });
    let Some(path) = found.filter(|p| p.is_file()) else { return respond(conn, 404, &error_body("not found")) };
    let body = std::fs::read(&path).map_err(|e| e.to_string())?;
    let kind = match path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "txt" | "md" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    };
    respond_file(conn, &body, kind, "no-store")
}

/// "a%20b" -> "a b" (a URL path's escapes).
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        match (b[i], b.get(i + 1).copied().and_then(hex), b.get(i + 2).copied().and_then(hex)) {
            (b'%', Some(h), Some(l)) => {
                out.push((h * 16 + l) as u8);
                i += 3;
            }
            (c, _, _) => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `respond` without CORS headers: for the agent tools, which other websites must not reach.
fn respond_private(conn: &mut TcpStream, code: u16, body: &Json) -> Result<(), String> {
    let text = body.to_string();
    let head = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        status_text(code),
        text.len()
    );
    conn.write_all(head.as_bytes()).and_then(|_| conn.write_all(text.as_bytes())).map_err(|e| e.to_string())
}

/// The chat page's 3D library (three.js, MIT) and typeface (Instrument Sans, OFL), from src/vendor, compiled in
/// so the page works offline. The paths name the version, so browsers may keep them for good.
const VENDOR: [(&str, &[u8], &str); 3] = [
    ("/vendor/three-0.160.0.module.min.js", include_bytes!("vendor/three.module.min.js"), "text/javascript; charset=utf-8"),
    ("/vendor/instrument-sans-5.3.0.woff2", include_bytes!("vendor/instrument-sans.woff2"), "font/woff2"),
    ("/vendor/instrument-sans-italic-5.3.0.woff2", include_bytes!("vendor/instrument-sans-italic.woff2"), "font/woff2"),
];

fn respond_file(conn: &mut TcpStream, body: &[u8], content_type: &str, cache: &str) -> Result<(), String> {
    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: {cache}\r\nConnection: close\r\n\r\n", body.len());
    conn.write_all(head.as_bytes()).and_then(|_| conn.write_all(body)).map_err(|e| e.to_string())
}

fn error_body(message: &str) -> Json {
    obj(vec![("error", obj(vec![("message", str(message)), ("type", str("invalid_request_error"))]))])
}

/// One server-sent event as one HTTP chunk.
fn sse(conn: &mut TcpStream, data: &str) -> std::io::Result<()> {
    let event = format!("data: {data}\n\n");
    write!(conn, "{:x}\r\n{event}\r\n", event.len())?;
    conn.flush()
}

fn handle(mut conn: TcpStream, work: &mpsc::Sender<Work>, shared: &Shared) -> Result<(), String> {
    let req = read_request(&mut conn)?;
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (current, for_call, together, loading) = {
        let s = shared.status.lock().unwrap();
        (s.current.clone(), s.for_call(), s.together, s.loading.clone())
    };
    let (defaults, think_default) = (shared.defaults, shared.think_default);
    // the agent tools and their settings: private to the chat page on this computer
    let path = req.path.trim_end_matches('/');
    if req.method == "GET" && req.path.starts_with("/preview/") {
        return respond_preview(&mut conn, &req, &req.path["/preview/".len()..]);
    }
    // speech has a model of its own: it answers whether or not a language model runs
    if path == "/v1/audio/speech" {
        return speech_request(&mut conn, &req, shared);
    }
    if path == "/v1/audio/transcriptions" {
        return transcription_request(&mut conn, &req, shared);
    }
    if ["/api/tools", "/api/harness", "/api/chats", "/api/agent", "/api/memory", "/api/maya", "/api/quit", "/api/skills", "/api/mcp", "/api/instructions", "/api/background", "/api/schedules", "/api/downloads", "/api/models/folders"].iter().any(|p| path.starts_with(p)) {
        if let Err(e) = tools_allowed(&req, &conn, &shared.token) {
            return respond_private(&mut conn, 403, &error_body(e));
        }
        let body = || std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse);
        return match (req.method.as_str(), path) {
            ("GET", "/api/harness") => respond_private(&mut conn, 200, &crate::tools::status(&shared.harness.lock().unwrap())),
            ("POST", "/api/harness") => match body().and_then(|b| {
                let old = shared.harness.lock().unwrap().clone(); // released before the new settings go in
                crate::tools::Settings::from_json(&b, &old)
            }) {
                Ok(s) => {
                    s.save()?;
                    let reply = crate::tools::status(&s);
                    *shared.harness.lock().unwrap() = s;
                    respond_private(&mut conn, 200, &reply)
                }
                Err(e) => respond_private(&mut conn, 400, &error_body(&e)),
            },
            // {"name": ..., "arguments": {...} or "<json>"}: {"result": "..."} or {"error": "..."} for the model
            ("POST", "/api/tools/run") => {
                let b = match body() {
                    Ok(b) => b,
                    Err(e) => return respond_private(&mut conn, 400, &error_body(&e)),
                };
                let name = b.get("name").and_then(Json::as_str).unwrap_or("");
                let args = match b.get("arguments") {
                    Some(Json::Str(a)) if a.trim().is_empty() => Json::Obj(Vec::new()),
                    Some(Json::Str(a)) => Json::parse(a).unwrap_or(Json::Null),
                    Some(a) => a.clone(),
                    None => Json::Obj(Vec::new()),
                };
                let checkpoint = b.get("checkpoint").and_then(Json::as_str).unwrap_or("");
                let settings = shared.harness.lock().unwrap();
                let reply = match (args, crate::tools::run) {
                    (Json::Null, _) => {
                        let given = b.get("arguments").and_then(Json::as_str).unwrap_or("");
                        obj(vec![("error", str(&format!("invalid arguments for {name}: not valid JSON. Arguments given: {}. Fix them and call the tool again.", crate::tools::clip(given, 300))))])
                    }
                    (args, run) => match run(name, &args, &settings, checkpoint) {
                        Ok(r) => obj(vec![("result", str(&r))]),
                        Err(e) => obj(vec![("error", str(&e))]),
                    },
                };
                respond_private(&mut conn, 200, &reply)
            }
            // the saved chats, shared by every client
            ("GET", "/api/chats") => match agent::list_chats() {
                Ok(list) => respond_private(&mut conn, 200, &obj(vec![("chats", list)])),
                Err(e) => respond_private(&mut conn, 500, &error_body(&e)),
            },
            ("GET", p) if p.starts_with("/api/chats/") => match agent::load_chat(&p["/api/chats/".len()..]) {
                Ok(chat) => respond_private(&mut conn, 200, &chat),
                Err(e) => respond_private(&mut conn, 404, &error_body(&e)),
            },
            ("DELETE", p) if p.starts_with("/api/chats/") => match agent::delete_chat(&p["/api/chats/".len()..]) {
                Ok(()) => respond_private(&mut conn, 200, &obj(vec![("deleted", Json::Bool(true))])),
                Err(e) => respond_private(&mut conn, 404, &error_body(&e)),
            },
            // {"id"}: the user has seen what a scheduled run wrote there
            ("POST", "/api/chats/read") => match agent::mark_read(body().ok().and_then(|b| b.get("id").and_then(Json::as_str).map(String::from)).as_deref().unwrap_or("")) {
                Ok(()) => respond_private(&mut conn, 200, &obj(vec![("read", Json::Bool(true))])),
                Err(e) => respond_private(&mut conn, 404, &error_body(&e)),
            },
            // {"id"}: the chat's old tool outputs taken out now (all but the last 3)
            ("POST", "/api/chats/trim") => match agent::trim_chat(body().ok().and_then(|b| b.get("id").and_then(Json::as_str).map(String::from)).as_deref().unwrap_or("")) {
                Ok(n) => respond_private(&mut conn, 200, &obj(vec![("trimmed", num(n as f64))])),
                Err(e) => respond_private(&mut conn, 409, &error_body(&e)),
            },
            // what runs on its own (schedule.rs): the scheduled tasks, with a number that changes whenever they
            // or a chat they wrote to do; a task run now or cancelled {"id"}
            ("GET", "/api/background") => respond_private(&mut conn, 200, &schedule::status_json()),
            // {"id"}: what a background process has printed lately
            ("POST", "/api/background/output") => {
                let id = body().ok().and_then(|b| b.get("id").and_then(Json::as_f64)).unwrap_or(0.0) as usize;
                match crate::tools::process_output(id) {
                    Some(out) => respond_private(&mut conn, 200, &obj(vec![("output", str(&out))])),
                    None => respond_private(&mut conn, 404, &error_body(&format!("no process {id}"))),
                }
            }
            ("POST", "/api/schedules/run") | ("POST", "/api/schedules/cancel") => {
                let id = body().ok().and_then(|b| b.get("id").and_then(Json::as_str).map(String::from)).unwrap_or_default();
                match if path.ends_with("/run") { schedule::run_now(&id) } else { schedule::cancel(&id) } {
                    Ok(j) => respond_private(&mut conn, 200, &j),
                    Err(e) => respond_private(&mut conn, 404, &error_body(&e)),
                }
            }
            // one reply of the agent, streamed as events (see agent.rs)
            // (Maya talks with the call model)
            ("POST", "/api/agent") => {
                let b = match body() {
                    Ok(b) => b,
                    Err(e) => return respond_private(&mut conn, 400, &error_body(&e)),
                };
                let maya = b.get("persona").and_then(Json::as_str) == Some("maya");
                let Some(l) = (if maya { for_call.clone() } else { current.clone() }).filter(|_| loading.is_none()) else {
                    return respond_private(&mut conn, 503, &error_body("no model is running yet; try again in a moment"));
                };
                agent::run(&mut conn, &b, work, shared, &l)
            }
            // Maya, the voice call's companion (maya.rs): what she remembers; after a call {"chat", "seconds"?}, she
            // thinks it over; reset: strangers again
            // the folders the model picker lists GGUF files from: {"add": "<folder>"} or {"remove": "<folder>"}
            // (for this run; the app keeps its own list and gives it again when it starts)
            ("GET", "/api/models/folders") => respond_private(&mut conn, 200, &shared.folders_json()),
            ("POST", "/api/models/folders") => {
                let b = body().unwrap_or(Json::Null);
                let path = |k: &str| b.get(k).and_then(Json::as_str).map(|p| PathBuf::from(p.trim()));
                if let Some(add) = path("add") {
                    if !add.is_dir() {
                        return respond_private(&mut conn, 400, &error_body(&format!("{} is not a folder", add.display())));
                    }
                    let mut dirs = shared.dirs.lock().unwrap();
                    if !dirs.contains(&add) {
                        dirs.push(add);
                    }
                }
                if let Some(remove) = path("remove") {
                    shared.dirs.lock().unwrap().retain(|d| *d != remove);
                }
                respond_private(&mut conn, 200, &shared.folders_json())
            }
            ("GET", "/api/maya") => respond_private(&mut conn, 200, &maya::status()),
            ("POST", "/api/maya/reflect") => {
                let Some(l) = for_call.clone().filter(|_| loading.is_none()) else {
                    return respond_private(&mut conn, 503, &error_body("no model is running yet; try again in a moment"));
                };
                let b = body().unwrap_or(Json::Null);
                let chat = b.get("chat").and_then(Json::as_str).unwrap_or("");
                match maya::reflect(chat, b.get("seconds").and_then(Json::as_f64), work, shared, &l) {
                    Ok(j) => respond_private(&mut conn, 200, &j),
                    Err(e) => respond_private(&mut conn, 400, &error_body(&e)),
                }
            }
            ("POST", "/api/maya/reset") => match maya::reset() {
                Ok(()) => respond_private(&mut conn, 200, &maya::status()),
                Err(e) => respond_private(&mut conn, 500, &error_body(&e)),
            },
            ("POST", "/api/maya/edit") => match maya::edit(&body().unwrap_or(Json::Null)) {
                Ok(j) => respond_private(&mut conn, 200, &j),
                Err(e) => respond_private(&mut conn, 400, &error_body(&e)),
            },
            // {"id": ..., "decision": "once" | "always" | "deny"}: the user's answer to an approval event
            ("POST", "/api/agent/approve") => {
                let b = body().unwrap_or(Json::Null);
                let (id, decision) = (b.get("id").and_then(Json::as_str).unwrap_or(""), b.get("decision").and_then(Json::as_str).unwrap_or("deny"));
                let ok = agent::answer(id, decision);
                respond_private(&mut conn, if ok { 200 } else { 404 }, &obj(vec![("answered", Json::Bool(ok))]))
            }
            // the app's "Quit and stop the model" (the launcher runs the server without a window): answers, then
            // ends the server, which frees the GPU
            ("POST", "/api/quit") => {
                respond_private(&mut conn, 200, &obj(vec![("quitting", Json::Bool(true))]))?;
                mcp::stop_all();
                std::process::exit(0)
            }
            // a model for the models folder (the app's first run): {"url": a Hugging Face file, "file": its name
            // here, "size": its bytes}; GET: how it is going; cancel: stop it (it resumes when started again)
            ("GET", "/api/downloads") => respond_private(&mut conn, 200, &download::status()),
            ("POST", "/api/downloads") => {
                let b = body().unwrap_or(Json::Null);
                let field = |k: &str| b.get(k).and_then(Json::as_str).unwrap_or("").to_string();
                let size = b.get("size").and_then(Json::as_f64).unwrap_or(0.0) as u64;
                let Some(dir) = &shared.download_dir else {
                    return respond_private(&mut conn, 400, &error_body("this server has no models folder to download into (start it with --models <folder>)"));
                };
                match download::start(&field("url"), &field("file"), size, dir) {
                    Ok(()) => respond_private(&mut conn, 200, &download::status()),
                    Err(e) => respond_private(&mut conn, 400, &error_body(&e)),
                }
            }
            ("POST", "/api/downloads/cancel") => {
                download::cancel();
                respond_private(&mut conn, 200, &download::status())
            }
            // a fact the user adds: {"fact": "..."}
            ("POST", "/api/memory/add") => match agent::add_fact(body().ok().and_then(|b| b.get("fact").and_then(Json::as_str).map(String::from)).as_deref().unwrap_or("")) {
                Ok(facts) => respond_private(&mut conn, 200, &obj(vec![("facts", Json::Arr(facts.iter().map(|f| str(f)).collect()))])),
                Err(e) => respond_private(&mut conn, 400, &error_body(&e)),
            },
            // skills: the installed ones, install from a folder, archive or link {"path", "skills": which ones
            // when it holds several}, remove {"name"}
            ("GET", "/api/skills") => respond_private(&mut conn, 200, &obj(vec![("skills", skills::list_json())])),
            ("POST", "/api/skills/install") => {
                let b = body().unwrap_or(Json::Null);
                let path = b.get("path").and_then(Json::as_str).unwrap_or("").to_string();
                let pick: Vec<String> = b.get("skills").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect();
                match skills::install(&path, &pick) {
                    Ok(skills::Installed::Done(names)) => respond_private(&mut conn, 200, &obj(vec![("installed", str(&names.join(", "))), ("skills", skills::list_json())])),
                    Ok(skills::Installed::Choose(choices)) => {
                        let list = choices.iter().map(|c| obj(vec![("path", str(&c.path)), ("name", str(&c.name)), ("description", str(&c.description))])).collect();
                        respond_private(&mut conn, 200, &obj(vec![("source", str(&path)), ("choose", Json::Arr(list)), ("skills", skills::list_json())]))
                    }
                    Err(e) => respond_private(&mut conn, 400, &error_body(&e)),
                }
            }
            ("POST", "/api/skills/remove") => match skills::remove(body().ok().and_then(|b| b.get("name").and_then(Json::as_str).map(String::from)).as_deref().unwrap_or("")) {
                Ok(()) => respond_private(&mut conn, 200, &obj(vec![("skills", skills::list_json())])),
                Err(e) => respond_private(&mut conn, 404, &error_body(&e)),
            },
            // MCP servers: their state; reload after editing mcp.json; import Claude Desktop's
            ("GET", "/api/mcp") => {
                let config = mcp::ensure_config().unwrap_or_default();
                respond_private(&mut conn, 200, &obj(vec![("servers", mcp::status()), ("config", str(&config))]))
            }
            ("POST", "/api/mcp/reload") => {
                std::thread::spawn(mcp::start_all);
                respond_private(&mut conn, 200, &obj(vec![("reloading", Json::Bool(true))]))
            }
            ("POST", "/api/mcp/import-claude") => match mcp::import_claude() {
                Ok(added) => {
                    std::thread::spawn(mcp::start_all);
                    respond_private(&mut conn, 200, &obj(vec![("added", Json::Arr(added.iter().map(|n| str(n)).collect()))]))
                }
                Err(e) => respond_private(&mut conn, 404, &error_body(&e)),
            },
            // the GHOST.md files: the global one, and one per folder (whether each exists)
            ("GET", "/api/instructions") => {
                let folders = shared.harness.lock().unwrap().folders.clone();
                respond_private(&mut conn, 200, &obj(vec![("files", agent::instructions_json(&folders))]))
            }
            // what the model remembers, and the user's remove button: {"index": n}
            ("GET", "/api/memory") => respond_private(&mut conn, 200, &obj(vec![("facts", Json::Arr(agent::memory().iter().map(|f| str(f)).collect()))])),
            ("POST", "/api/memory/forget") => {
                let index = body().ok().and_then(|b| b.get("index").and_then(Json::as_f64)).unwrap_or(-1.0);
                match agent::forget_at(if index < 0.0 { usize::MAX } else { index as usize }) {
                    Ok(facts) => respond_private(&mut conn, 200, &obj(vec![("facts", Json::Arr(facts.iter().map(|f| str(f)).collect()))])),
                    Err(e) => respond_private(&mut conn, 404, &error_body(&e)),
                }
            }
            // {"checkpoint": ...}: every file change made under it, put back
            ("POST", "/api/tools/undo") => {
                let cp = body().ok().and_then(|b| b.get("checkpoint").and_then(Json::as_str).map(String::from)).unwrap_or_default();
                match crate::tools::undo(&cp) {
                    Ok(done) => respond_private(&mut conn, 200, &obj(vec![("undone", Json::Arr(done.iter().map(|d| str(d)).collect()))])),
                    Err(e) => respond_private(&mut conn, 500, &error_body(&e)),
                }
            }
            _ => respond_private(&mut conn, 404, &error_body("not found")),
        };
    }
    match (req.method.as_str(), req.path.trim_end_matches('/')) {
        // the models folder, for the chat page's model picker
        ("GET", "/api/models") => {
            let (voice_names, voice_on) = {
                let v = shared.voice.lock().unwrap();
                (v.names(), v.loaded())
            };
            let models = list_models(&shared.dirs())
                .into_iter()
                .map(|(id, _, size)| obj(vec![("id", str(&id)), ("size", num(size as f64))]))
                .collect();
            let opt = |s: Option<&str>| s.map(str).unwrap_or(Json::Null);
            let body = obj(vec![
                ("current", opt(current.as_ref().map(|c| c.id.as_str()))),
                ("loading", opt(loading.as_deref())),
                // loaded and warming up: a request now goes first, but runs slower than once it is done
                ("warming", Json::Bool(current.is_some() && WARMING.load(std::sync::atomic::Ordering::Relaxed))),
                ("context_length", current.as_ref().map(|c| num(c.ctx as f64)).unwrap_or(Json::Null)),
                ("thinking", Json::Bool(current.as_ref().is_some_and(|c| c.thinking))),
                // Qwen3 also reads "/no_think" in a message as switching thinking off; GLM does not
                ("no_think_tag", Json::Bool(current.as_ref().is_some_and(|c| c.thinking && !c.glm))),
                ("about", current.as_ref().map(|c| c.about.clone()).unwrap_or(Json::Null)),
                // the speech model (--tts), whose voice the chat page can read replies in
                ("speech", voice_names.0.as_deref().map_or(Json::Null, str)),
                // and the speech-to-text model (--stt): with both, the app offers voice calls
                ("listen", voice_names.1.as_deref().map_or(Json::Null, str)),
                // whether they are loaded now (they load with a model, when asked: /api/models/load "voice")
                ("voice", Json::Bool(voice_on)),
                // the model a call talks to now (loaded, with the voice): a call model kept beside the chat
                // model, or the chat model itself
                ("call_model", opt(voice_on.then(|| for_call.as_ref().map(|c| c.id.as_str())).flatten())),
                ("call_context_length", for_call.as_ref().filter(|_| voice_on).map(|c| num(c.ctx as f64)).unwrap_or(Json::Null)),
                // the chat model and the call model (with the voice) both loaded: switching between them is free
                ("together", Json::Bool(together)),
                ("models", Json::Arr(models)),
            ]);
            return respond(&mut conn, 200, &body);
        }
        // the prompt being read: tokens done of the total (0: none), and whether it is waiting on the disk
        // this computer and what runs on it, for the chat page's get_system_info tool
        ("GET", "/api/system") => {
            let mut first = vec![("inference_engine", str(concat!("ghost-infer ", env!("CARGO_PKG_VERSION"), ", running locally on this computer")))];
            if let Some(c) = &current {
                first.push(("model_loaded", str(&c.id)));
                first.push(("model_runs_on", str(&c.backend)));
                first.push(("context_length", num(c.ctx as f64)));
            }
            return respond(&mut conn, 200, &crate::sysinfo::report(first));
        }
        ("GET", "/api/progress") => {
            let (done, total, disk_ms, disk) = crate::cuda::PROGRESS.get();
            let body = obj(vec![("done", num(done as f64)), ("total", num(total as f64)), ("disk_ms", num(disk_ms as f64)), ("disk", Json::Bool(disk))]);
            return respond(&mut conn, 200, &body);
        }
        // {"id": "<file name without .gguf>", "voice": true / false (the voice models loaded beside it, or not; left
        // as they are without it)}; or {"id", "call": "<model for voice calls, may be the same>", "for_call": bool}:
        // both kept loaded with the voice when they all fit, else the call model with the voice (for_call) or the
        // model alone. Answers once the switch is done (it can take a minute)
        ("POST", "/api/models/load") => {
            let b = std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse).ok();
            let field = |k: &str| b.as_ref().and_then(|b| b.get(k)).cloned();
            let Some(id) = field("id").and_then(|v| v.as_str().map(String::from)) else { return respond(&mut conn, 400, &error_body("send {\"id\": \"<model>\"}")) };
            let models = list_models(&shared.dirs());
            let find = |id: &str| models.iter().find(|m| m.0 == id).map(|m| m.1.clone());
            let missing = |id: &str| {
                let dirs: Vec<String> = shared.dirs().iter().map(|d| d.display().to_string()).collect();
                error_body(&format!("no {id}.gguf in {}", dirs.join(" or ")))
            };
            let Some(path) = find(&id) else { return respond(&mut conn, 404, &missing(&id)) };
            let mut plan = Plan::one(path, field("voice").and_then(|v| v.as_bool()));
            if let Some(call) = field("call").and_then(|v| v.as_str().map(String::from)) {
                let Some(p) = find(&call) else { return respond(&mut conn, 404, &missing(&call)) };
                plan.call = Some(p);
                plan.for_call = field("for_call").and_then(|v| v.as_bool()).unwrap_or(false);
            }
            if let Some(l) = &loading {
                return respond(&mut conn, 409, &error_body(&format!("{l} is loading; wait for it")));
            }
            let (tx, done) = mpsc::channel();
            work.send(Work::Load(plan, tx)).map_err(|_| "the model thread stopped")?;
            return match done.recv() {
                Ok(Ok(())) => respond(&mut conn, 200, &obj(vec![("current", str(&id))])),
                Ok(Err(e)) => respond(&mut conn, 500, &error_body(&e)),
                Err(_) => respond(&mut conn, 500, &error_body("the model thread stopped")),
            };
        }
        _ => {}
    }
    // everything else needs a running model
    let Some(Loaded { id: model_id, ctx, chat_ok, default_system, thinking, glm, mistral, llama3, coder, .. }) = current.filter(|_| loading.is_none()) else {
        if matches!(req.path.trim_end_matches('/'), "/v1/chat/completions" | "/v1/completions" | "/v1/models") {
            let msg = match loading.as_deref() {
                Some(what) => format!("loading {what}; try again in a moment"),
                None => "no model is loaded: pick one in the chat page, or POST {\"id\": \"<model>\"} to /api/models/load".into(),
            };
            return respond(&mut conn, 503, &error_body(&msg));
        }
        if let ("GET", "" | "/index.html") = (req.method.as_str(), req.path.trim_end_matches('/')) {
            return respond_page(&mut conn, &shared.token);
        }
        if let Some((_, body, content_type)) = VENDOR.iter().find(|v| v.0 == req.path) {
            return respond_file(&mut conn, body, content_type, "public, max-age=31536000, immutable");
        }
        return respond(&mut conn, 503, &error_body("loading"));
    };
    let (model_id, default_system) = (model_id.as_str(), default_system.as_deref());
    match (req.method.as_str(), req.path.trim_end_matches('/')) {
        ("OPTIONS", _) => {
            let head = format!("HTTP/1.1 204 No Content\r\n{CORS}Content-Length: 0\r\nConnection: close\r\n\r\n");
            return conn.write_all(head.as_bytes()).map_err(|e| e.to_string());
        }
        ("GET", "" | "/index.html") => return respond_page(&mut conn, &shared.token),
        ("GET", p) if VENDOR.iter().any(|v| v.0 == p) => {
            let (_, body, content_type) = VENDOR.iter().find(|v| v.0 == p).unwrap();
            return respond_file(&mut conn, body, content_type, "public, max-age=31536000, immutable");
        }
        ("GET", "/health") => return respond(&mut conn, 200, &obj(vec![("status", str("ok"))])),
        ("GET", "/v1/models") => {
            let m = obj(vec![("id", str(model_id)), ("object", str("model")), ("created", num(created as f64)), ("owned_by", str("ghost-infer")), ("context_length", num(ctx as f64))]);
            return respond(&mut conn, 200, &obj(vec![("object", str("list")), ("data", Json::Arr(vec![m]))]));
        }
        ("POST", "/v1/chat/completions") if chat_ok => {}
        ("POST", "/v1/chat/completions") => return respond(&mut conn, 400, &error_body("this model has no chat template; use /v1/completions")),
        ("POST", "/v1/completions") => {}
        (_, "/v1/chat/completions" | "/v1/completions" | "/v1/models") => return respond(&mut conn, 405, &error_body("method not allowed")),
        _ => return respond(&mut conn, 404, &error_body("not found")),
    }
    let chat = req.path.trim_end_matches('/') == "/v1/chat/completions";
    let _going = agent::Going::new(); // another client using the model: scheduled runs wait for quiet
    let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
        Ok(b) => b,
        Err(e) => return respond(&mut conn, 400, &error_body(&format!("request body: {e}"))),
    };
    let prompt = match (chat, glm, mistral) {
        (true, true, _) => chat_prompt_glm(&body, think_default, thinking),
        (true, _, true) => chat_prompt_mistral(&body),
        (true, ..) if llama3 => chat_prompt_llama3(&body),
        (true, false, false) => chat_prompt(&body, default_system, think_default, thinking, coder),
        (false, ..) => completion_prompt(&body),
    };
    let prompt = match prompt {
        Ok(p) => p,
        Err(e) => return respond(&mut conn, 400, &error_body(&e)),
    };
    let lead = if chat { reply_lead(&prompt) } else { "" };
    let f = |k: &str| body.get(k).and_then(Json::as_f64);
    let stop = match body.get("stop") {
        Some(Json::Str(s)) => vec![s.clone()],
        Some(Json::Arr(a)) => a.iter().filter_map(|s| s.as_str().map(String::from)).collect(),
        _ => Vec::new(),
    };
    let stream = body.get("stream").and_then(Json::as_bool).unwrap_or(false);
    let include_usage = body.get("stream_options").and_then(|s| s.get("include_usage")).and_then(Json::as_bool).unwrap_or(false);
    let (tx, events) = mpsc::channel();
    let job = Job {
        model: model_id.to_string(),
        kind: if chat { Kind::Chat } else { Kind::Completion },
        prompt,
        max_tokens: f("max_completion_tokens").or(f("max_tokens")).map(|n| n.max(1.0) as usize).or(Some(defaults.3)),
        temperature: f("temperature").map(|t| t as f32).unwrap_or(defaults.0),
        top_k: f("top_k").map(|k| k as usize).unwrap_or(defaults.1),
        top_p: f("top_p").map(|p| p as f32).unwrap_or(defaults.2),
        seed: f("seed").map(|s| s as u64).unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)),
        style: Style {
            presence: f("presence_penalty").unwrap_or(0.0) as f32,
            dry: f("dry_multiplier").unwrap_or(0.0) as f32,
            sentences: f("max_sentences").unwrap_or(0.0) as usize,
            spoken: false,
            repetition: f("repetition_penalty").unwrap_or(1.0) as f32,
        },
        stop,
        lead,
        events: tx,
    };
    work.send(Work::Job(job)).map_err(|_| "the model thread stopped")?;

    let id = format!("{}-{created:x}{:08x}", if chat { "chatcmpl" } else { "cmpl" }, rand_suffix());
    let object = if chat { "chat.completion" } else { "text_completion" };
    let choice = |text: Option<&str>, finish: Option<&str>, first: bool| -> Json {
        let finish = finish.map(str).unwrap_or(Json::Null);
        if chat {
            let mut delta = Vec::new();
            if first {
                delta.push(("role", str("assistant")));
            }
            if let Some(t) = text {
                delta.push(("content", str(t)));
            }
            obj(vec![("index", num(0)), ("delta", obj(delta)), ("finish_reason", finish)])
        } else {
            obj(vec![("index", num(0)), ("text", str(text.unwrap_or(""))), ("logprobs", Json::Null), ("finish_reason", finish)])
        }
    };
    // tool calls (chat requests with tools): split out of the text, sent as OpenAI's `tool_calls`
    let mut calls = (chat && !request_tools(&body).is_empty()).then(ToolCalls::default);
    let call_json = |index: usize, name: &str, arguments: &str| -> Json {
        obj(vec![
            ("index", num(index as f64)),
            ("id", str(&format!("call_{:08x}{index}", rand_suffix()))),
            ("type", str("function")),
            ("function", obj(vec![("name", str(name)), ("arguments", str(arguments))])),
        ])
    };
    let usage = |p: usize, c: usize| obj(vec![("prompt_tokens", num(p as f64)), ("completion_tokens", num(c as f64)), ("total_tokens", num((p + c) as f64))]);
    let chunk = |choices: Vec<Json>, extra: Option<(&str, Json)>| -> String {
        let mut fields = vec![
            ("id", str(&id)),
            ("object", str(if chat { "chat.completion.chunk" } else { "text_completion" })),
            ("created", num(created as f64)),
            ("model", str(model_id)),
            ("choices", Json::Arr(choices)),
        ];
        if let Some(e) = extra {
            fields.push(e);
        }
        obj(fields).to_string()
    };

    if !stream {
        let mut text = String::new();
        for ev in events {
            match ev {
                Event::Text(t) => text.push_str(&t),
                Event::Error(code, e) => return respond(&mut conn, code, &error_body(&e)),
                Event::Done { finish, prompt_tokens, completion_tokens } => {
                    let c = if chat {
                        let mut message = vec![("role", str("assistant")), ("content", str(&text))];
                        let mut finish = finish;
                        if let Some(tc) = calls.as_mut() {
                            let (mut content, mut found) = (String::new(), Vec::new());
                            for part in tc.push(&text).into_iter().chain(tc.finish()) {
                                match part {
                                    Part::Text(t) | Part::Bad(t) => content.push_str(&t),
                                    Part::Call { name, arguments } => found.push(call_json(found.len(), &name, &arguments)),
                                }
                            }
                            if !found.is_empty() {
                                let content = content.trim_end();
                                message[1].1 = if content.is_empty() { Json::Null } else { str(content) };
                                message.push(("tool_calls", Json::Arr(found)));
                                finish = "tool_calls";
                            }
                        }
                        obj(vec![("index", num(0)), ("message", obj(message)), ("finish_reason", str(finish))])
                    } else {
                        obj(vec![("index", num(0)), ("text", str(&text)), ("logprobs", Json::Null), ("finish_reason", str(finish))])
                    };
                    let resp = obj(vec![
                        ("id", str(&id)),
                        ("object", str(object)),
                        ("created", num(created as f64)),
                        ("model", str(model_id)),
                        ("choices", Json::Arr(vec![c])),
                        ("usage", usage(prompt_tokens, completion_tokens)),
                    ]);
                    return respond(&mut conn, 200, &resp);
                }
            }
        }
        return respond(&mut conn, 500, &error_body("generation ended without a result"));
    }

    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n{CORS}Connection: close\r\n\r\n");
    conn.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    let mut first = true;
    if chat {
        // the role first, as OpenAI does
        sse(&mut conn, &chunk(vec![choice(Some(""), None, true)], None)).map_err(|e| e.to_string())?;
        first = false;
    }
    let mut n_calls = 0;
    // a failed write means the client is gone: dropping `events` stops the generation
    let mut send_parts = |conn: &mut TcpStream, parts: Vec<Part>, first: &mut bool| -> Result<(), String> {
        for part in parts {
            let c = match part {
                Part::Text(t) | Part::Bad(t) => choice(Some(&t), None, *first),
                Part::Call { name, arguments } => {
                    n_calls += 1;
                    obj(vec![("index", num(0)), ("delta", obj(vec![("tool_calls", Json::Arr(vec![call_json(n_calls - 1, &name, &arguments)]))])), ("finish_reason", Json::Null)])
                }
            };
            sse(conn, &chunk(vec![c], None)).map_err(|e| e.to_string())?;
            *first = false;
        }
        Ok(())
    };
    for ev in events {
        match ev {
            Event::Text(t) if t.is_empty() => {}
            Event::Text(t) => {
                let parts = match calls.as_mut() {
                    Some(tc) => tc.push(&t),
                    None => vec![Part::Text(t)],
                };
                send_parts(&mut conn, parts, &mut first)?;
            }
            Event::Error(_, e) => {
                sse(&mut conn, &error_body(&e).to_string()).map_err(|e| e.to_string())?;
                break;
            }
            Event::Done { finish, prompt_tokens, completion_tokens } => {
                if let Some(tc) = calls.as_mut() {
                    send_parts(&mut conn, tc.finish(), &mut first)?;
                }
                let finish = if calls.as_ref().is_some_and(|tc| tc.calls > 0) { "tool_calls" } else { finish };
                let last = if chat { choice(None, Some(finish), false) } else { choice(Some(""), Some(finish), false) };
                sse(&mut conn, &chunk(vec![last], None)).map_err(|e| e.to_string())?;
                if include_usage {
                    sse(&mut conn, &chunk(Vec::new(), Some(("usage", usage(prompt_tokens, completion_tokens))))).map_err(|e| e.to_string())?;
                }
                break;
            }
        }
    }
    sse(&mut conn, "[DONE]").map_err(|e| e.to_string())?;
    conn.write_all(b"0\r\n\r\n").map_err(|e| e.to_string())
}

fn rand_suffix() -> u32 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0));
    h.finish() as u32
}

/// The text of a message's content: a string, or OpenAI's list of parts (text parts only).
fn content_text(c: Option<&Json>) -> Result<String, String> {
    match c {
        None | Some(Json::Null) => Ok(String::new()),
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(Json::Arr(parts)) => Ok(parts.iter().filter(|p| p.get("type").and_then(Json::as_str) == Some("text")).filter_map(|p| p.get("text").and_then(Json::as_str)).collect::<Vec<_>>().join("")),
        Some(_) => Err("message content must be a string or a list of parts".into()),
    }
}

/// The tools the model may call in this request (none when `tool_choice` is "none").
fn request_tools(body: &Json) -> &[Json] {
    if body.get("tool_choice").and_then(Json::as_str) == Some("none") {
        return &[];
    }
    body.get("tools").and_then(Json::as_array).unwrap_or(&[])
}

/// A user message that is not a new question, for the templates: the harness's own (a nudge, its "aside":
/// the question it asks to name or compact a chat, put after it). The templates leave out the reasoning of
/// the turns before the latest question; with one of these counted, the reply's rounds before it lost theirs
/// and the prompt changed near its start, so all of it was read again (95 seconds for a compaction's
/// question, 40 for a title's). A compaction's summary is the new start, so a question.
fn aside(m: &Json) -> bool {
    let flag = |k: &str| m.get(k).and_then(Json::as_bool).unwrap_or(false);
    flag("aside") || (flag("auto") && !flag("compact"))
}

/// The messages in the ChatML template, ending with an open assistant turn. Tools, tool calls and
/// tool results are written as Qwen's template writes them (`<tools>`, `<tool_call>`,
/// `<tool_response>`), and the reasoning of turns before the latest user question is left out.
/// `switch`: the model's template has a thinking switch; without one, thinking settings are ignored.
/// `coder`: Qwen3-Coder's template, which lists tools, calls and results in its own XML form instead.
fn chat_prompt(body: &Json, default_system: Option<&str>, think_default: bool, switch: bool, coder: bool) -> Result<String, String> {
    let messages = body.get("messages").and_then(Json::as_array).ok_or("messages must be a list")?;
    if messages.is_empty() {
        return Err("messages is empty".into());
    }
    let mut roles = Vec::with_capacity(messages.len());
    let mut contents = Vec::with_capacity(messages.len());
    for m in messages {
        roles.push(match m.get("role").and_then(Json::as_str) {
            Some("developer") => "system",
            Some(r @ ("system" | "user" | "assistant" | "tool")) => r,
            Some(other) => return Err(format!("unknown message role {other}")),
            None => return Err("a message has no role".into()),
        });
        contents.push(content_text(m.get("content"))?);
    }
    let tools = request_tools(body);
    // Qwen3.5 / 3.6: Qwen3-Coder's calls in a thinking model's template, the tools listed as JSON under
    // Qwen3-Coder's instructions, earlier replies of this turn with their reasoning, the reply's opened
    let qwen35 = coder && switch;

    let mut s = String::new();
    let system = if roles[0] == "system" { Some(contents[0].as_str()) } else if roles.contains(&"system") { None } else { default_system };
    if !tools.is_empty() && qwen35 {
        s.push_str("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for t in tools {
            s.push('\n');
            s.push_str(&t.to_python_string());
        }
        s.push_str(CODER_CALLS);
        if let Some(sys) = system.map(str::trim).filter(|t| !t.is_empty()) {
            s.push_str("\n\n");
            s.push_str(sys);
        }
        s.push_str("<|im_end|>\n");
    } else if !tools.is_empty() && coder {
        s.push_str("<|im_start|>system\n");
        s.push_str(system.unwrap_or("You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks."));
        s.push_str("\n\nYou have access to the following functions:\n\n<tools>");
        for t in tools {
            coder_tool(&mut s, t);
        }
        s.push_str(CODER_CALLS);
        s.push_str("<|im_end|>\n");
    } else if !tools.is_empty() {
        s.push_str("<|im_start|>system\n");
        if let Some(sys) = system {
            s.push_str(sys);
            s.push_str("\n\n");
        }
        s.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for t in tools {
            s.push('\n');
            s.push_str(&t.to_python_string());
        }
        s.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
    } else if let Some(sys) = system {
        s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
    }

    // the latest real question (a user turn that is not a tool result, nor an aside: see aside)
    let is_tool_result = |c: &str| c.starts_with("<tool_response>") && c.ends_with("</tool_response>");
    let last_query = (0..messages.len()).rev().find(|&i| roles[i] == "user" && !is_tool_result(&contents[i]) && !aside(&messages[i])).unwrap_or(messages.len() - 1);
    for (i, m) in messages.iter().enumerate() {
        let content = contents[i].as_str();
        match roles[i] {
            "system" if i == 0 => {}
            r @ ("system" | "user") => s.push_str(&format!("<|im_start|>{r}\n{content}<|im_end|>\n")),
            "assistant" if coder && !qwen35 => {
                // Qwen3-Coder does not think; a call's arguments go one <parameter=...> each, values that are
                // not text as JSON (the template writes Python's form; JSON is what coder_call reads back)
                let content = content.rsplit("</think>").next().unwrap_or(content).trim_start_matches('\n');
                let calls = m.get("tool_calls").and_then(Json::as_array).unwrap_or(&[]);
                if calls.is_empty() {
                    s.push_str(&format!("<|im_start|>assistant\n{content}<|im_end|>\n"));
                } else {
                    s.push_str("<|im_start|>assistant");
                    if !content.trim().is_empty() {
                        s.push_str(&format!("\n{}\n", content.trim()));
                    }
                    for call in calls {
                        let f = call.get("function").unwrap_or(call);
                        s.push_str(&format!("\n<tool_call>\n<function={}>\n", f.get("name").and_then(Json::as_str).unwrap_or("")));
                        let args = match f.get("arguments") {
                            Some(Json::Str(a)) => Json::parse(a).unwrap_or(Json::Null),
                            Some(a) => a.clone(),
                            None => Json::Null,
                        };
                        if let Json::Obj(fields) = args {
                            for (k, v) in fields {
                                let v = match v {
                                    Json::Str(t) => t,
                                    other => other.to_python_string(),
                                };
                                s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
                            }
                        }
                        s.push_str("</function>\n</tool_call>");
                    }
                    s.push_str("<|im_end|>\n");
                }
            }
            "assistant" => {
                let content = if qwen35 { content.trim() } else { content };
                let (mut reasoning, mut content) = (String::new(), content.to_string());
                if let Some(r) = m.get("reasoning_content").and_then(Json::as_str) {
                    reasoning = r.to_string();
                } else if let Some((before, after)) = content.split_once("</think>") {
                    reasoning = before.trim_end_matches('\n').rsplit("<think>").next().unwrap_or("").trim_start_matches('\n').to_string();
                    content = after.trim_start_matches('\n').to_string();
                }
                s.push_str("<|im_start|>assistant\n");
                if qwen35 && i > last_query {
                    s.push_str(&format!("<think>\n{}\n</think>\n\n{content}", reasoning.trim()));
                } else if i > last_query && (i == messages.len() - 1 || !reasoning.is_empty()) {
                    s.push_str(&format!("<think>\n{}\n</think>\n\n{}", reasoning.trim_matches('\n'), content.trim_start_matches('\n')));
                } else {
                    s.push_str(&content);
                }
                if qwen35 {
                    // each call as Qwen3-Coder writes it, values that are not text as JSON
                    for (k, call) in m.get("tool_calls").and_then(Json::as_array).unwrap_or(&[]).iter().enumerate() {
                        let f = call.get("function").unwrap_or(call);
                        let lead = if k > 0 { "\n" } else if content.trim().is_empty() { "" } else { "\n\n" };
                        s.push_str(&format!("{lead}<tool_call>\n<function={}>\n", f.get("name").and_then(Json::as_str).unwrap_or("")));
                        let args = match f.get("arguments") {
                            Some(Json::Str(a)) => Json::parse(a).unwrap_or(Json::Null),
                            Some(a) => a.clone(),
                            None => Json::Null,
                        };
                        if let Json::Obj(fields) = args {
                            for (k, v) in fields {
                                let v = match v {
                                    Json::Str(t) => t,
                                    other => other.to_python_string(),
                                };
                                s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
                            }
                        }
                        s.push_str("</function>\n</tool_call>");
                    }
                    s.push_str("<|im_end|>\n");
                    continue;
                }
                for (k, call) in m.get("tool_calls").and_then(Json::as_array).unwrap_or(&[]).iter().enumerate() {
                    if k > 0 || !content.is_empty() {
                        s.push('\n');
                    }
                    let f = call.get("function").unwrap_or(call);
                    let name = f.get("name").and_then(Json::as_str).unwrap_or("");
                    let args = match f.get("arguments") {
                        Some(Json::Str(a)) => a.clone(),
                        Some(a) => a.to_python_string(),
                        None => "{}".into(),
                    };
                    s.push_str(&format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>"));
                }
                s.push_str("<|im_end|>\n");
            }
            "tool" if coder && !qwen35 => {
                if i == 0 || roles[i - 1] != "tool" {
                    s.push_str("<|im_start|>user\n");
                }
                s.push_str(&format!("<tool_response>\n{content}\n</tool_response>\n"));
                if i + 1 == messages.len() || roles[i + 1] != "tool" {
                    s.push_str("<|im_end|>\n");
                }
            }
            _ => {
                // tool results: consecutive ones share one user turn
                if i == 0 || roles[i - 1] != "tool" {
                    s.push_str("<|im_start|>user");
                }
                s.push_str(&format!("\n<tool_response>\n{content}\n</tool_response>"));
                if i + 1 == messages.len() || roles[i + 1] != "tool" {
                    s.push_str("<|im_end|>\n");
                }
            }
        }
    }
    s.push_str("<|im_start|>assistant\n");
    // thinking off: chat_template_kwargs.enable_thinking = false, or OpenAI's reasoning_effort
    // "none"/"minimal" (sent by agents such as Hermes); otherwise think_default (serve --no-think)
    let thinking = body.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")).and_then(Json::as_bool).or_else(|| {
        body.get("reasoning_effort").and_then(Json::as_str).map(|e| !matches!(e.to_ascii_lowercase().as_str(), "none" | "minimal"))
    });
    if switch && !thinking.unwrap_or(think_default) {
        s.push_str("<think>\n\n</think>\n\n");
    } else if qwen35 {
        s.push_str("<think>\n");
    }
    Ok(s)
}

/// Text a chat prompt already holds that the reply is read with: a prompt that opens the reply's reasoning
/// (GLM's `<think>`, Qwen3.5's `<think>\n`) gets a reply that starts with it too.
pub(crate) fn reply_lead(prompt: &str) -> &'static str {
    if prompt.ends_with("<|assistant|><think>") {
        "<think>"
    } else if prompt.ends_with("<|im_start|>assistant\n<think>\n") {
        "<think>\n"
    } else {
        ""
    }
}

/// The end of Qwen3-Coder's tool list and its instructions for calls, as its template writes them.
const CODER_CALLS: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// One tool in Qwen3-Coder's list, as its template writes it: <function> with its name, description and a
/// <parameter> per argument; keys the template has no tag of its own for become tags named after them.
fn coder_tool(s: &mut String, tool: &Json) {
    let f = tool.get("function").unwrap_or(tool);
    let text = |j: &Json| j.get("description").map(|v| jinja_str(v).trim().to_string()).unwrap_or_default();
    s.push_str(&format!("\n<function>\n<name>{}</name>\n<description>{}</description>\n<parameters>", f.get("name").map(jinja_str).unwrap_or_default(), text(f)));
    let params = f.get("parameters").unwrap_or(&Json::Null);
    if let Some(Json::Obj(props)) = params.get("properties") {
        for (name, fields) in props {
            s.push_str(&format!("\n<parameter>\n<name>{name}</name>"));
            if let Some(t) = fields.get("type") {
                s.push_str(&format!("\n<type>{}</type>", jinja_str(t)));
            }
            if fields.get("description").is_some() {
                s.push_str(&format!("\n<description>{}</description>", text(fields)));
            }
            coder_list(s, fields.get("enum"), "enum");
            if let Json::Obj(keys) = fields {
                for (k, v) in keys.iter().filter(|(k, _)| !matches!(k.as_str(), "type" | "description" | "enum" | "required")) {
                    let k = k.replace(['-', ' '], "_").replace('$', "");
                    let v = if matches!(v, Json::Obj(_)) { v.to_python_string() } else { jinja_str(v) };
                    s.push_str(&format!("\n<{k}>{v}</{k}>"));
                }
            }
            coder_list(s, fields.get("required"), "required");
            s.push_str("\n</parameter>");
        }
    }
    coder_list(s, params.get("required"), "required");
    s.push_str("\n</parameters>");
    if let Some(r) = f.get("return") {
        s.push_str(&format!("\n<return>{}</return>", if matches!(r, Json::Obj(_)) { r.to_python_string() } else { jinja_str(r) }));
    }
    s.push_str("\n</function>");
}

/// The template's render_item_list: `\n<tag>[`a`, `b`]</tag>` for a list that is not empty, text in backticks.
fn coder_list(s: &mut String, list: Option<&Json>, tag: &str) {
    let Some(items) = list.and_then(Json::as_array).filter(|l| !l.is_empty()) else {
        return;
    };
    let items: Vec<String> = items.iter().map(|i| if let Json::Str(t) = i { format!("`{t}`") } else { jinja_str(i) }).collect();
    s.push_str(&format!("\n<{tag}>[{}]</{tag}>", items.join(", ")));
}

/// A value as a template's `string` filter writes it: text as it is, True, False and None as Python writes
/// them, the rest as JSON.
fn jinja_str(v: &Json) -> String {
    match v {
        Json::Str(t) => t.clone(),
        Json::Bool(b) => (if *b { "True" } else { "False" }).into(),
        Json::Null => "None".into(),
        other => other.to_python_string(),
    }
}

/// `chat_prompt` for GLM-4 (GLM-4.7-Flash), as its template writes it: `[gMASK]<sop>`, then each message
/// after its role token with no newlines, earlier replies without their reasoning, tools and tool calls
/// in GLM's XML form, and `<|assistant|>` followed by `<think>` (thinking) or `</think>` (not).
fn chat_prompt_glm(body: &Json, think_default: bool, switch: bool) -> Result<String, String> {
    let messages = body.get("messages").and_then(Json::as_array).ok_or("messages must be a list")?;
    if messages.is_empty() {
        return Err("messages is empty".into());
    }
    let mut roles = Vec::with_capacity(messages.len());
    let mut contents = Vec::with_capacity(messages.len());
    for m in messages {
        roles.push(match m.get("role").and_then(Json::as_str) {
            Some("developer") => "system",
            Some(r @ ("system" | "user" | "assistant" | "tool")) => r,
            Some(other) => return Err(format!("unknown message role {other}")),
            None => return Err("a message has no role".into()),
        });
        contents.push(content_text(m.get("content"))?);
    }
    let mut s = String::from("[gMASK]<sop>");
    let tools = request_tools(body);
    if !tools.is_empty() {
        s.push_str("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
        for t in tools {
            s.push_str(&t.to_python_string());
            s.push('\n');
        }
        s.push_str("</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>");
    }
    let last_user = (0..messages.len()).rev().find(|&i| roles[i] == "user" && !aside(&messages[i])).unwrap_or(0);
    for (i, m) in messages.iter().enumerate() {
        let content = contents[i].as_str();
        match roles[i] {
            r @ ("system" | "user") => s.push_str(&format!("<|{r}|>{content}")),
            "assistant" => {
                let (mut reasoning, mut content) = (String::new(), content.to_string());
                if let Some(r) = m.get("reasoning_content").and_then(Json::as_str) {
                    reasoning = r.to_string();
                } else if let Some((before, after)) = content.split_once("</think>") {
                    reasoning = before.rsplit("<think>").next().unwrap_or("").to_string();
                    content = after.to_string();
                }
                s.push_str("<|assistant|>");
                let reasoning = reasoning.trim();
                if i > last_user && !reasoning.is_empty() {
                    s.push_str(&format!("<think>{reasoning}</think>"));
                } else {
                    s.push_str("</think>");
                }
                s.push_str(content.trim());
                let calls = m.get("tool_calls").and_then(Json::as_array).unwrap_or(&[]);
                if !calls.is_empty() {
                    for call in calls {
                        let f = call.get("function").unwrap_or(call);
                        s.push_str(&format!("<tool_call>{}", f.get("name").and_then(Json::as_str).unwrap_or("")));
                        let args = match f.get("arguments") {
                            Some(Json::Str(a)) => Json::parse(a).unwrap_or(Json::Null),
                            Some(a) => a.clone(),
                            None => Json::Null,
                        };
                        if let Json::Obj(fields) = args {
                            for (k, v) in fields {
                                let v = match v {
                                    Json::Str(t) => t,
                                    other => other.to_python_string(),
                                };
                                s.push_str(&format!("<arg_key>{k}</arg_key><arg_value>{v}</arg_value>"));
                            }
                        }
                        s.push_str("</tool_call>");
                    }
                }
            }
            _ => {
                // tool results: one <|observation|> for a run of them
                if i == 0 || roles[i - 1] != "tool" {
                    s.push_str("<|observation|>");
                }
                s.push_str(&format!("<tool_response>{content}</tool_response>"));
            }
        }
    }
    let thinking = body.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")).and_then(Json::as_bool).or_else(|| {
        body.get("reasoning_effort").and_then(Json::as_str).map(|e| !matches!(e.to_ascii_lowercase().as_str(), "none" | "minimal"))
    });
    s.push_str("<|assistant|>");
    s.push_str(if switch && thinking.unwrap_or(think_default) { "<think>" } else { "</think>" });
    Ok(s)
}

/// `chat_prompt` for Mistral's format (Mistral Nemo and its merges, as Ayla), as Nemo's template writes it:
/// `[INST]user[/INST]reply</s>`, turns strictly taking turns (a run of messages from one side is one turn),
/// the system prompt in front of the last user turn, and no tools: their results are read as the user's
/// text and earlier calls are left out. The tokenizer adds the <s> in front.
fn chat_prompt_mistral(body: &Json) -> Result<String, String> {
    let messages = body.get("messages").and_then(Json::as_array).ok_or("messages must be a list")?;
    if messages.is_empty() {
        return Err("messages is empty".into());
    }
    let mut system = Vec::new();
    let mut turns: Vec<(bool, Vec<String>)> = Vec::new(); // (the user's, its messages)
    for m in messages {
        let text = content_text(m.get("content"))?;
        let (user, text) = match m.get("role").and_then(Json::as_str) {
            Some("system" | "developer") => {
                system.push(text);
                continue;
            }
            Some("user") => (true, text),
            Some("tool") => (true, format!("Tool result:\n{text}")),
            // earlier replies without their reasoning
            Some("assistant") => (false, text.rsplit("</think>").next().unwrap_or(&text).to_string()),
            Some(other) => return Err(format!("unknown message role {other}")),
            None => return Err("a message has no role".into()),
        };
        let text = text.trim().to_string();
        if text.is_empty() {
            continue;
        }
        match turns.last_mut() {
            Some((u, texts)) if *u == user => texts.push(text),
            _ => turns.push((user, vec![text])),
        }
    }
    let system = system.join("\n\n");
    let last_user = turns.iter().rposition(|(u, _)| *u);
    if last_user.is_none() && !system.is_empty() {
        turns.push((true, Vec::new()));
    }
    let last_user = turns.iter().rposition(|(u, _)| *u);
    let mut s = String::new();
    for (i, (user, texts)) in turns.iter().enumerate() {
        let text = texts.join("\n\n");
        if !user {
            s.push_str(&format!("{text}</s>"));
        } else if Some(i) == last_user && !system.is_empty() {
            s.push_str(&format!("[INST]{}[/INST]", if text.is_empty() { system.clone() } else { format!("{system}\n\n{text}") }));
        } else {
            s.push_str(&format!("[INST]{text}[/INST]"));
        }
    }
    Ok(s)
}

/// `chat_prompt` for Llama 3's format, as its template writes it: each message
/// `<|start_header_id|>role<|end_header_id|>\n\ntext<|eot_id|>`, the system prompts first as one, then the
/// reply's header. No tools: their results are read as the user's text and earlier calls are left out. The
/// tokenizer adds <|begin_of_text|> in front.
fn chat_prompt_llama3(body: &Json) -> Result<String, String> {
    let messages = body.get("messages").and_then(Json::as_array).ok_or("messages must be a list")?;
    if messages.is_empty() {
        return Err("messages is empty".into());
    }
    let mut system = Vec::new();
    let mut turns = String::new();
    for m in messages {
        let text = content_text(m.get("content"))?;
        let (role, text) = match m.get("role").and_then(Json::as_str) {
            Some("system" | "developer") => {
                system.push(text.trim().to_string());
                continue;
            }
            Some("user") => ("user", text),
            Some("tool") => ("user", format!("Tool result:\n{text}")),
            // earlier replies without their reasoning
            Some("assistant") => ("assistant", text.rsplit("</think>").next().unwrap_or(&text).to_string()),
            Some(other) => return Err(format!("unknown message role {other}")),
            None => return Err("a message has no role".into()),
        };
        let text = text.trim();
        if !text.is_empty() {
            turns.push_str(&format!("<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"));
        }
    }
    system.retain(|s| !s.is_empty());
    let mut s = String::new();
    if !system.is_empty() {
        s.push_str(&format!("<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", system.join("\n\n")));
    }
    s.push_str(&turns);
    s.push_str(LLAMA3_REPLY_HEAD);
    Ok(s)
}

/// How a Llama 3 prompt ends: the reply's header.
const LLAMA3_REPLY_HEAD: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";

/// A chat request's prompt in the loaded model's format.
fn chat_prompt_for(loaded: &Loaded, body: &Json, think_default: bool) -> Result<String, String> {
    if loaded.glm {
        chat_prompt_glm(body, think_default, loaded.thinking)
    } else if loaded.mistral {
        chat_prompt_mistral(body)
    } else if loaded.llama3 {
        chat_prompt_llama3(body)
    } else {
        chat_prompt(body, loaded.default_system.as_deref(), think_default, loaded.thinking, loaded.coder)
    }
}

/// One piece of a reply, once tool calls are split out of the text.
#[derive(Debug, PartialEq)]
enum Part {
    Text(String),
    Call { name: String, arguments: String },
    /// a `<tool_call>` block that could not be read, as written (the API passes it on as text; the agent
    /// tells the model, so it sends the call again)
    Bad(String),
}

/// Splits `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks out of streamed text,
/// holding back what could still become a tag. Whitespace-only text after a call is dropped; JSON with
/// its closing braces wrong is mended (mend_json), and a block that still cannot be read is a Bad part.
#[derive(Default)]
struct ToolCalls {
    buf: String,
    in_call: bool,
    calls: usize,
}

impl ToolCalls {
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";

    fn push(&mut self, text: &str) -> Vec<Part> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            if self.in_call {
                let Some(end) = self.buf.find(Self::CLOSE) else { break };
                let block: String = self.buf.drain(..end + Self::CLOSE.len()).collect();
                self.in_call = false;
                out.push(match self.call(&block[..end]) {
                    Part::Bad(t) => Part::Bad(t + Self::CLOSE),
                    call => call,
                });
            } else if let Some(at) = self.buf.find(Self::OPEN) {
                let before: String = self.buf.drain(..at + Self::OPEN.len()).collect();
                self.text(&before[..at], &mut out);
                self.in_call = true;
            } else {
                let ready = self.buf.len() - partial_suffix(&self.buf, Self::OPEN);
                // after a call, wait for something other than whitespace before sending text
                let ready = if self.calls > 0 && self.buf[..ready].trim().is_empty() { 0 } else { ready };
                let rest = self.buf.split_off(ready);
                let text = std::mem::replace(&mut self.buf, rest);
                self.text(&text, &mut out);
                break;
            }
        }
        out
    }

    /// While a call is being written (nothing of it comes out until it is whole): its tool's name once
    /// that can be read, "" before.
    fn writing(&self) -> Option<&str> {
        if !self.in_call {
            return None;
        }
        let b = self.buf.trim_start();
        let name = if let Some(rest) = b.strip_prefix("<function=") {
            rest.split_once('>').map(|(n, _)| n)
        } else if b.starts_with('{') {
            let after = b.split_once("\"name\"").and_then(|(_, r)| r.trim_start().strip_prefix(':'));
            after.and_then(|r| r.trim_start().strip_prefix('"')).and_then(|r| r.split_once('"')).map(|(n, _)| n)
        } else {
            b.split_once("<arg_key>").or_else(|| b.split_once('\n')).map(|(n, _)| n)
        };
        Some(name.map_or("", str::trim))
    }

    /// While a call is being written: its "path" argument once that is whole (a file being written shows
    /// its name), and how many bytes of the call have come so far.
    fn writing_path(&self) -> Option<(Option<String>, usize)> {
        if !self.in_call {
            return None;
        }
        let b = self.buf.as_str();
        let path = if let Some((_, r)) = b.split_once("<parameter=path>") {
            r.split_once("</parameter>").map(|(p, _)| p.trim().to_string())
        } else if let Some((_, r)) = b.split_once("<arg_key>path</arg_key>") {
            r.split_once("</arg_value>").map(|(p, _)| p.trim().trim_start_matches("<arg_value>").trim().to_string())
        } else if let Some((_, r)) = b.split_once("\"path\"") {
            // JSON: the string up to its closing quote (one not escaped)
            r.trim_start().strip_prefix(':').map(str::trim_start).and_then(|r| r.strip_prefix('"')).and_then(|r| {
                let mut out = String::new();
                let mut chars = r.chars();
                while let Some(c) = chars.next() {
                    match c {
                        '"' => return Some(out),
                        '\\' => out.push(match chars.next()? {
                            'n' => '\n',
                            't' => '\t',
                            c => c,
                        }),
                        c => out.push(c),
                    }
                }
                None
            })
        } else {
            None
        };
        Some((path.filter(|p| !p.is_empty()), b.len()))
    }

    /// What is left at the end of the reply.
    fn finish(&mut self) -> Vec<Part> {
        let rest = std::mem::take(&mut self.buf);
        let mut out = Vec::new();
        if self.in_call {
            // cut off before the closing tag: still a call if the JSON is complete
            self.in_call = false;
            out.push(self.call(&rest));
        } else {
            self.text(&rest, &mut out);
        }
        out
    }

    fn text(&self, t: &str, out: &mut Vec<Part>) {
        if !t.is_empty() && !(self.calls > 0 && t.trim().is_empty()) {
            out.push(Part::Text(t.to_string()));
        }
    }

    fn call(&mut self, inner: &str) -> Part {
        let json = Json::parse(inner.trim()).ok().or_else(|| {
            let mended = mend_json(inner.trim());
            if mended.is_some() {
                eprintln!("[a tool call's JSON had its closing braces wrong: mended]");
            }
            mended
        });
        let parsed = json.and_then(|j| {
            let name = j.get("name")?.as_str()?.to_string();
            let arguments = match j.get("arguments") {
                Some(Json::Str(s)) => s.clone(),
                Some(a) => a.to_python_string(),
                None => "{}".into(),
            };
            Some(Part::Call { name, arguments })
        });
        let parsed = parsed.or_else(|| glm_call(inner)).or_else(|| coder_call(inner));
        match parsed {
            Some(p) => {
                self.calls += 1;
                p
            }
            None => Part::Bad(format!("{}{inner}", Self::OPEN)),
        }
    }
}

/// A tool call's JSON with its closing braces or brackets wrong, as models write it now and then
/// (Qwen3 drops the last `}` after nested arguments): what is still open is closed, and a closer too
/// many (or the wrong one) is cut. None if that does not make it valid.
fn mend_json(s: &str) -> Option<Json> {
    if !s.starts_with('{') {
        return None;
    }
    let mut open = Vec::new();
    let (mut in_str, mut escaped, mut end) = (false, false, s.len());
    for (i, c) in s.char_indices() {
        if in_str {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => open.push('}'),
            '[' => open.push(']'),
            '}' | ']' => {
                if open.last() != Some(&c) {
                    end = i; // not what is open: cut here, then close properly
                    break;
                }
                open.pop();
                if open.is_empty() {
                    end = i + 1; // complete: anything after it is extra
                    break;
                }
            }
            _ => {}
        }
    }
    if in_str {
        return None;
    }
    let mut mended = s[..end].to_string();
    mended.extend(open.iter().rev());
    Json::parse(&mended).ok()
}

/// GLM's tool call: `name<arg_key>k</arg_key><arg_value>v</arg_value>...`. Values that read as JSON
/// (numbers, true/false, lists, objects) keep that type; anything else is a string.
fn glm_call(inner: &str) -> Option<Part> {
    let (name, mut rest) = inner.split_once("<arg_key>").map_or((inner, ""), |(n, r)| (n, r));
    let name = name.trim();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.')) {
        return None; // a function name is one word
    }
    let mut fields = Vec::new();
    while !rest.is_empty() {
        let (key, after) = rest.split_once("</arg_key>")?;
        let after = after.trim_start().strip_prefix("<arg_value>")?;
        let (value, after) = after.split_once("</arg_value>")?;
        let v = match Json::parse(value.trim()) {
            Ok(j @ (Json::Num(_) | Json::Bool(_) | Json::Arr(_) | Json::Obj(_) | Json::Null)) => j,
            _ => Json::Str(value.to_string()),
        };
        fields.push((key.trim().to_string(), v));
        rest = after.trim_start().strip_prefix("<arg_key>").unwrap_or(after.trim());
    }
    Some(Part::Call { name: name.to_string(), arguments: Json::Obj(fields).to_string() })
}

/// A call in Qwen3-Coder's own form, as it was trained to write them: `<function=name>`, then each argument as
/// `<parameter=key>` value `</parameter>`, then `</function>`. Values that read as JSON (numbers, true, lists)
/// are taken as such; the rest is text, without the line breaks the template puts around it.
fn coder_call(inner: &str) -> Option<Part> {
    let rest = inner.trim().strip_prefix("<function=")?;
    let (name, mut body) = rest.split_once('>')?;
    let name = name.trim();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.')) {
        return None;
    }
    let mut fields = Vec::new();
    while let Some(at) = body.find("<parameter=") {
        let (key, after) = body[at + "<parameter=".len()..].split_once('>')?;
        let (value, after) = after.split_once("</parameter>")?;
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let v = match Json::parse(value.trim()) {
            Ok(j @ (Json::Num(_) | Json::Bool(_) | Json::Arr(_) | Json::Obj(_) | Json::Null)) => j,
            _ => Json::Str(value.to_string()),
        };
        fields.push((key.trim().to_string(), v));
        body = after;
    }
    Some(Part::Call { name: name.to_string(), arguments: Json::Obj(fields).to_string() })
}

fn completion_prompt(body: &Json) -> Result<String, String> {
    match body.get("prompt") {
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(Json::Arr(a)) if a.len() == 1 => a[0].as_str().map(String::from).ok_or_else(|| "prompt must be a string".into()),
        Some(Json::Arr(_)) => Err("only one prompt per request".into()),
        _ => Err("prompt must be a string".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_string_prefixes() {
        assert_eq!(partial_suffix("hello wor", "world"), 3);
        assert_eq!(partial_suffix("hello", "world"), 0);
        assert_eq!(partial_suffix("ab", "abc"), 2);
    }

    #[test]
    fn chat_template() {
        let body = Json::parse(r#"{"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]}"#).unwrap();
        assert_eq!(chat_prompt(&body, Some("Be brief."), true, true, false).unwrap(), "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn tool_template() {
        // what the Qwen3 chat template (Jinja, in the GGUF) gives for this conversation
        let body = Json::parse(
            r#"{"tools":[{"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"tz":{"type":"string"}}}}}],
            "messages":[{"role":"system","content":"Sys"},{"role":"user","content":"Time?"},
            {"role":"assistant","content":"<think>\nhm\n</think>\n\nLet me check.","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_time","arguments":"{\"tz\": \"UTC\"}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"12:00"},{"role":"tool","tool_call_id":"c2","content":"x"}]}"#,
        )
        .unwrap();
        let want = concat!(
            "<|im_start|>system\nSys\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
            r#"{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}"#,
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
            "<|im_start|>user\nTime?<|im_end|>\n",
            "<|im_start|>assistant\n<think>\nhm\n</think>\n\nLet me check.\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UTC\"}}\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\n12:00\n</tool_response>\n<tool_response>\nx\n</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(chat_prompt(&body, None, true, true, false).unwrap(), want);
    }

    #[test]
    fn reasoning_effort_none_turns_thinking_off() {
        let off = Json::parse(r#"{"reasoning_effort":"none","messages":[{"role":"user","content":"Hi"}]}"#).unwrap();
        assert!(chat_prompt(&off, None, true, true, false).unwrap().ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        let on = Json::parse(r#"{"reasoning_effort":"medium","messages":[{"role":"user","content":"Hi"}]}"#).unwrap();
        assert!(chat_prompt(&on, None, true, true, false).unwrap().ends_with("<|im_start|>assistant\n"));
        // an explicit chat_template_kwargs wins over reasoning_effort
        let kw = Json::parse(r#"{"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":true},"messages":[{"role":"user","content":"Hi"}]}"#).unwrap();
        assert!(chat_prompt(&kw, None, true, true, false).unwrap().ends_with("<|im_start|>assistant\n"));
        // serve --no-think: off unless the request asks
        let plain = Json::parse(r#"{"messages":[{"role":"user","content":"Hi"}]}"#).unwrap();
        assert!(chat_prompt(&plain, None, false, true, false).unwrap().ends_with("<think>\n\n</think>\n\n"));
        assert!(chat_prompt(&on, None, false, true, false).unwrap().ends_with("<|im_start|>assistant\n"));
        // a model without a thinking switch (Qwen3-2507 Instruct, SmolLM2) never gets the empty block
        assert!(chat_prompt(&off, None, true, false, false).unwrap().ends_with("<|im_start|>assistant\n"));
        assert!(chat_prompt(&plain, None, false, false, false).unwrap().ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn coder_tool_template() {
        // what Qwen3-Coder's chat template (Jinja, in the GGUF) gives for this conversation, rendered as transformers does
        let body = Json::parse(
            r#"{"tools":[{"type":"function","function":{"name":"get_time","description":"Current time. ","parameters":{"type":"object",
            "properties":{"tz":{"type":"string","description":"Time zone","enum":["UTC","MYT"]},"n":{"type":"integer","default":2}},"required":["tz"]}}}],
            "messages":[{"role":"system","content":"Sys"},{"role":"user","content":"Time?"},
            {"role":"assistant","content":"Let me check.","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_time","arguments":"{\"tz\": \"UTC\", \"n\": 2}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"12:00"},{"role":"tool","tool_call_id":"c2","content":"x"}]}"#,
        )
        .unwrap();
        let want = concat!(
            "<|im_start|>system\nSys\n\nYou have access to the following functions:\n\n<tools>\n<function>\n<name>get_time</name>\n<description>Current time.</description>\n<parameters>",
            "\n<parameter>\n<name>tz</name>\n<type>string</type>\n<description>Time zone</description>\n<enum>[`UTC`, `MYT`]</enum>\n</parameter>",
            "\n<parameter>\n<name>n</name>\n<type>integer</type>\n<default>2</default>\n</parameter>\n<required>[`tz`]</required>\n</parameters>\n</function>\n</tools>",
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>",
            "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n",
            "<|im_start|>user\nTime?<|im_end|>\n",
            "<|im_start|>assistant\nLet me check.\n\n<tool_call>\n<function=get_time>\n<parameter=tz>\nUTC\n</parameter>\n<parameter=n>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\n12:00\n</tool_response>\n<tool_response>\nx\n</tool_response>\n<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(chat_prompt(&body, None, true, false, true).unwrap(), want);
    }

    #[test]
    fn qwen35_tool_template() {
        // what Qwen3.6's chat template gives for these conversations, rendered as transformers does
        const CALLS: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
        let first = Json::parse(
            r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Time in Tokyo?"}],
            "tools":[{"type":"function","function":{"name":"get_time","description":"Current time","parameters":{"type":"object","properties":{"tz":{"type":"string","description":"IANA zone"}},"required":["tz"]}}}]}"#,
        )
        .unwrap();
        let want = format!(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{}{CALLS}\n\nBe brief.<|im_end|>\n<|im_start|>user\nTime in Tokyo?<|im_end|>\n<|im_start|>assistant\n<think>\n",
            r#"{"type": "function", "function": {"name": "get_time", "description": "Current time", "parameters": {"type": "object", "properties": {"tz": {"type": "string", "description": "IANA zone"}}, "required": ["tz"]}}}"#
        );
        let prompt = chat_prompt(&first, None, true, true, true).unwrap();
        assert_eq!(prompt, want);
        assert_eq!(reply_lead(&prompt), "<think>\n");

        // calls after reasoning and after none, tool results, thinking off
        let later = Json::parse(
            r#"{"messages":[{"role":"user","content":"Time?"},
            {"role":"assistant","content":"<think>\nneed tool\n</think>\n\nLet me check.","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_time","arguments":"{\"tz\": \"UTC\", \"n\": 2}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"12:00"},
            {"role":"assistant","content":"","tool_calls":[{"id":"c2","type":"function","function":{"name":"get_time","arguments":"{\"tz\": \"Asia/Tokyo\"}"}}]},
            {"role":"tool","tool_call_id":"c2","content":"21:00"}],
            "tools":[{"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"tz":{"type":"string"},"n":{"type":"integer"}}}}}],
            "chat_template_kwargs":{"enable_thinking":false}}"#,
        )
        .unwrap();
        let want = format!(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{}{CALLS}<|im_end|>\n{}",
            r#"{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {"tz": {"type": "string"}, "n": {"type": "integer"}}}}}"#,
            concat!(
                "<|im_start|>user\nTime?<|im_end|>\n",
                "<|im_start|>assistant\n<think>\nneed tool\n</think>\n\nLet me check.\n\n<tool_call>\n<function=get_time>\n<parameter=tz>\nUTC\n</parameter>\n<parameter=n>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
                "<|im_start|>user\n<tool_response>\n12:00\n</tool_response><|im_end|>\n",
                "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_time>\n<parameter=tz>\nAsia/Tokyo\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
                "<|im_start|>user\n<tool_response>\n21:00\n</tool_response><|im_end|>\n",
                "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            )
        );
        let prompt = chat_prompt(&later, None, true, true, true).unwrap();
        assert_eq!(prompt, want);
        assert_eq!(reply_lead(&prompt), "");
    }

    #[test]
    fn coder_tool_calls() {
        let mut tc = ToolCalls::default();
        let text = "Let me run it.\n<tool_call>\n<function=run_command>\n<parameter=command>\necho hi\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>";
        let mut parts = tc.push(text);
        parts.extend(tc.finish());
        let call = parts.iter().find_map(|p| match p {
            Part::Call { name, arguments } => Some((name.clone(), Json::parse(arguments).unwrap())),
            _ => None,
        });
        let (name, args) = call.expect("a call");
        assert_eq!(name, "run_command");
        assert_eq!(args.get("command").and_then(Json::as_str), Some("echo hi"));
        assert_eq!(args.get("timeout").and_then(Json::as_f64), Some(30.0));
    }

    #[test]
    fn tool_calls_split_from_stream() {
        let mut tc = ToolCalls::default();
        let mut parts = Vec::new();
        let text = "<think>\nok\n</think>\n\nSure.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>\n<tool_call>\n{\"name\": \"g\", \"arguments\": {}}\n</tool_call>\n";
        // one character at a time, as tokens arrive
        for c in text.chars() {
            parts.extend(tc.push(&c.to_string()));
        }
        parts.extend(tc.finish());
        let text: String = parts.iter().filter_map(|p| if let Part::Text(t) = p { Some(t.as_str()) } else { None }).collect();
        assert_eq!(text, "<think>\nok\n</think>\n\nSure.\n");
        let calls: Vec<_> = parts.iter().filter(|p| matches!(p, Part::Call { .. })).collect();
        assert_eq!(calls, [&Part::Call { name: "f".into(), arguments: "{\"a\": 1}".into() }, &Part::Call { name: "g".into(), arguments: "{}".into() }]);

        // a block that cannot be read: a Bad part, which the API passes on as text
        let mut tc = ToolCalls::default();
        let mut parts = tc.push("a <tool_call>not json</tool_call> b");
        parts.extend(tc.finish());
        assert_eq!(parts.iter().map(|p| match p { Part::Text(t) | Part::Bad(t) => t.as_str(), _ => "" }).collect::<String>(), "a <tool_call>not json</tool_call> b");
        assert!(parts.contains(&Part::Bad("<tool_call>not json</tool_call>".into())));
    }

    #[test]
    fn an_aside_keeps_the_prompt_start() {
        // the harness's own question after a reply's rounds (to name or compact the chat): the rounds keep their
        // reasoning, so the prompt starts as the round's did, and the server's cache still holds it
        let chat = r#"[{"role":"user","content":"make a page"},{"role":"assistant","content":"<think>plan</think>ok","tool_calls":[{"id":"c1","type":"function","function":{"name":"list_files","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"a.html"}"#;
        let body = |more: &str| Json::parse(&format!(r#"{{"messages":{chat}{more}],"chat_template_kwargs":{{"enable_thinking":true}}}}"#)).unwrap();
        for (switch, coder) in [(true, true), (true, false)] {
            let round = chat_prompt(&body(""), None, true, switch, coder).unwrap();
            let start = round.rsplit_once("<|im_start|>assistant").unwrap().0;
            assert!(start.contains("plan"));
            let asked = chat_prompt(&body(r#",{"role":"user","content":"Name this chat","aside":true}"#), None, true, switch, coder).unwrap();
            assert!(asked.starts_with(start), "{asked}");
            // a new question: the round's reasoning goes, and the prompt changes before it
            let plain = chat_prompt(&body(r#",{"role":"user","content":"Name this chat"}"#), None, true, switch, coder).unwrap();
            assert!(!plain.starts_with(start));
        }
    }

    #[test]
    fn mistral_prompt() {
        // the system prompt in front of the last user turn, replies closed with </s>, a side's run of messages one turn
        let body = Json::parse(r#"{"messages":[{"role":"system","content":"Be Maya."},{"role":"user","content":"hey"},{"role":"assistant","content":"<think>x</think>Hi!"},{"role":"user","content":"long day"},{"role":"user","content":"ugh"}]}"#).unwrap();
        assert_eq!(chat_prompt_mistral(&body).unwrap(), "[INST]hey[/INST]Hi!</s>[INST]Be Maya.\n\nlong day\n\nugh[/INST]");
        let only = Json::parse(r#"{"messages":[{"role":"system","content":"Say hi."}]}"#).unwrap();
        assert_eq!(chat_prompt_mistral(&only).unwrap(), "[INST]Say hi.[/INST]");
        assert_eq!(recent_replies("[INST]a[/INST]one</s>[INST]b[/INST]two</s>[INST]c[/INST]", 5), vec!["one", "two"]);
    }

    #[test]
    fn llama3_prompt() {
        let body = Json::parse(r#"{"messages":[{"role":"system","content":"Be Maya."},{"role":"user","content":"hey"},{"role":"assistant","content":"<think>x</think>Hi!"},{"role":"user","content":"long day"}]}"#).unwrap();
        let p = chat_prompt_llama3(&body).unwrap();
        assert_eq!(
            p,
            "<|start_header_id|>system<|end_header_id|>\n\nBe Maya.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhey<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nlong day<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert_eq!(recent_replies(&p, 5), vec!["Hi!"]);
    }

    #[test]
    fn a_call_being_written() {
        // its tool's name as soon as it can be read, in each form; cut off by the length limit: Bad
        let mut tc = ToolCalls::default();
        tc.push("Writing it.\n<tool_call>\n<function=wri");
        assert_eq!(tc.writing(), Some(""));
        tc.push("te_file>\n<parameter=path>\nindex.html\n</parameter>\n<parameter=content>\n<!DOCTYPE html>\n<html>");
        assert_eq!(tc.writing(), Some("write_file"));
        assert!(matches!(tc.finish().as_slice(), [Part::Bad(_)]));
        assert_eq!(tc.writing(), None);
        let mut tc = ToolCalls::default();
        tc.push("<tool_call>\n{\"name\": \"read_file\", \"argu");
        assert_eq!(tc.writing(), Some("read_file"));
        let mut tc = ToolCalls::default();
        tc.push("<tool_call>write_file\n<arg_key>path</arg_key>");
        assert_eq!(tc.writing(), Some("write_file"));
        tc.push("<arg_value>a.txt</arg_value></tool_call>");
        assert_eq!(tc.writing(), None);
    }

    #[test]
    fn the_path_of_a_call_being_written() {
        let path = |text: &str| {
            let mut tc = ToolCalls::default();
            tc.push(text);
            tc.writing_path().and_then(|(p, _)| p)
        };
        assert_eq!(path("<tool_call>\n<function=write_file>\n<parameter=path>\nD:\\g\\index.html\n</parameter>\n<parameter=content>\n<html>"), Some("D:\\g\\index.html".into()));
        assert_eq!(path("<tool_call>\n<function=write_file>\n<parameter=path>\nD:\\g\\ind"), None);
        assert_eq!(path("<tool_call>write_file\n<arg_key>path</arg_key>\n<arg_value>a.txt</arg_value>\n<arg_key>content"), Some("a.txt".into()));
        assert_eq!(path("<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"D:\\\\g\\\\a \\\"b\\\".js\", \"content\": \"x"), Some("D:\\g\\a \"b\".js".into()));
        assert_eq!(path("<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"D:\\\\g"), None);
        assert_eq!(path("no call here"), None);
    }

    #[test]
    fn tool_calls_with_braces_wrong_are_mended() {
        let calls = |text: &str| {
            let mut tc = ToolCalls::default();
            let mut parts = tc.push(text);
            parts.extend(tc.finish());
            parts
        };
        // as Qwen3-30B-A3B wrote it: the outer object never closed
        let missing = "<tool_call>\n{\"name\": \"update_tasks\", \"arguments\": {\"tasks\": [{\"text\": \"Fix total()\", \"status\": \"done\"}]}\n</tool_call>";
        assert_eq!(calls(missing), vec![Part::Call { name: "update_tasks".into(), arguments: "{\"tasks\": [{\"text\": \"Fix total()\", \"status\": \"done\"}]}".into() }]);
        // one brace too many, and a bracket left open
        assert_eq!(calls("<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}}}</tool_call>"), vec![Part::Call { name: "f".into(), arguments: "{\"a\": 1}".into() }]);
        assert_eq!(calls("<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": [1, 2}}</tool_call>"), vec![Part::Call { name: "f".into(), arguments: "{\"a\": [1, 2]}".into() }]);
        // braces inside strings do not count
        assert_eq!(calls("<tool_call>{\"name\": \"f\", \"arguments\": {\"s\": \"}{\"}</tool_call>"), vec![Part::Call { name: "f".into(), arguments: "{\"s\": \"}{\"}".into() }]);
    }

    #[test]
    fn glm_prompt_and_tool_calls() {
        let body = Json::parse(
            r#"{"messages":[{"role":"system","content":"Sys"},{"role":"user","content":"Time?"},
            {"role":"assistant","content":"<think>hm</think>Let me check.","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_time","arguments":"{\"tz\": \"UTC\", \"n\": 2}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"12:00"},{"role":"tool","tool_call_id":"c2","content":"x"}]}"#,
        )
        .unwrap();
        assert_eq!(
            chat_prompt_glm(&body, true, true).unwrap(),
            "[gMASK]<sop><|system|>Sys<|user|>Time?<|assistant|><think>hm</think>Let me check.<tool_call>get_time<arg_key>tz</arg_key><arg_value>UTC</arg_value><arg_key>n</arg_key><arg_value>2</arg_value></tool_call><|observation|><tool_response>12:00</tool_response><tool_response>x</tool_response><|assistant|><think>"
        );
        // tools, as llama.cpp (b9106) renders GLM-4.7-Flash's own template
        let tools = Json::parse(r#"{"messages":[{"role":"user","content":"Time?"}],"tools":[{"type":"function","function":{"name":"get_time","description":"Current time","parameters":{"type":"object","properties":{"tz":{"type":"string"}}}}}]}"#).unwrap();
        assert_eq!(
            chat_prompt_glm(&tools, true, true).unwrap(),
            concat!(
                "[gMASK]<sop><|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
                r#"{"type": "function", "function": {"name": "get_time", "description": "Current time", "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}"#,
                "\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>",
                "<|user|>Time?<|assistant|><think>"
            )
        );
        let earlier = Json::parse(r#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"<think>r</think>b"},{"role":"user","content":"c"}],"chat_template_kwargs":{"enable_thinking":false}}"#).unwrap();
        assert_eq!(chat_prompt_glm(&earlier, true, true).unwrap(), "[gMASK]<sop><|user|>a<|assistant|></think>b<|user|>c<|assistant|></think>");

        let mut tc = ToolCalls::default();
        let mut parts = tc.push("Checking.\n<tool_call>get_time<arg_key>tz</arg_key><arg_value>UTC</arg_value><arg_key>n</arg_key><arg_value>2</arg_value></tool_call>\n");
        parts.extend(tc.finish());
        assert_eq!(parts, vec![Part::Text("Checking.\n".into()), Part::Call { name: "get_time".into(), arguments: r#"{"tz":"UTC","n":2}"#.into() }]);
        assert!(glm_call("get_system_info").is_some_and(|p| p == Part::Call { name: "get_system_info".into(), arguments: "{}".into() }));
    }
}
