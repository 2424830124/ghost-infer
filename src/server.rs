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
mod mcp;
mod schedule;
mod skills;

const MAX_BODY: usize = 32 << 20;

enum Kind {
    Chat,
    Completion,
}

/// Work for the model thread.
enum Work {
    Job(Job),
    /// switch to this GGUF file; the reply comes once a model is running again (Err: it stayed on the old one)
    Load(PathBuf, mpsc::Sender<Result<(), String>>),
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
    coder: bool,    // Qwen3-Coder's tool format (XML) instead of Qwen3's (JSON)
    backend: String, // where it runs, e.g. "GPU NVIDIA GeForce RTX 2060 SUPER (sm_75, ...)"
    tok: Arc<crate::tokenizer::Tokenizer>, // to count a prompt's tokens before sending it (the agent fits chats to the context)
}

#[derive(Default)]
struct Status {
    current: Option<Loaded>,
    loading: Option<String>, // the model being loaded; requests get 503 meanwhile
}

/// What the connection threads share: the model's status, and where the models are.
struct Shared {
    status: Mutex<Status>,
    dirs: Vec<PathBuf>, // the model's folder, then any --models folders
    download_dir: Option<PathBuf>, // the first --models folder: where a downloaded model goes
    defaults: (f32, usize, f32, usize),
    think_default: bool,
    harness: Mutex<crate::tools::Settings>,
    /// the tool groups of the last agent request: after a model loads, the agent prompt's opening with these is
    /// read in ahead of time
    agent_groups: Mutex<Vec<String>>,
    /// made at startup and written into the chat page: the agent tools answer only requests carrying it
    token: String,
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
        status: Mutex::new(Status { current: None, loading: (!start_idle).then(|| model_id(&o.model)) }),
        dirs,
        download_dir: o.models.first().cloned(),
        defaults: (o.temp, o.top_k, o.top_p, o.max_new),
        think_default: !o.no_think,
        harness: Mutex::new(crate::tools::Settings::load()),
        agent_groups: Mutex::new(vec!["files".to_string()]),
        token: new_token(),
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

    // the model thread: the engine (GPU state included) never leaves it. Each turn of this loop loads a
    // model and serves it until a switch is asked for; the engine and its file are dropped before the next.
    // Started without one, it first waits for a model to be picked, and waits again if that one fails to load.
    let mut path = o.model.clone();
    let mut idle = start_idle;
    let mut last_good: Option<PathBuf> = None;
    let mut waiting: Option<(mpsc::Sender<Result<(), String>>, Option<String>)> = None; // who asked for this switch, and an error to report
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
            path = next;
            waiting = Some((reply, None));
            idle = false;
        }
        let id = model_id(&path);
        shared.status.lock().unwrap().loading = Some(id.clone());
        let t = Instant::now();
        // a failed switch goes back to the model that was running; failing at startup (or on the way back) ends the server
        let mut fall_back = |e: String, path: &mut PathBuf| -> Result<(), String> {
            if start_idle && last_good.is_none() {
                // nothing ran yet: report it and wait for another pick
                eprintln!("[could not load {}: {e}]", model_id(path));
                if let Some((w, _)) = waiting.take() {
                    let _ = w.send(Err(format!("could not load {}: {e}", model_id(path))));
                }
                idle = true;
                return Ok(());
            }
            match (&mut waiting, &last_good) {
                (Some(w), Some(prev)) if prev != path => {
                    eprintln!("[could not load {}: {e}; going back to {}]", model_id(path), model_id(prev));
                    w.1 = Some(format!("could not load {}: {e}", model_id(path)));
                    *path = prev.clone();
                    Ok(())
                }
                _ => Err(e),
            }
        };
        let g = match crate::gguf::Gguf::open(&path) {
            Ok(g) => g,
            Err(e) => {
                fall_back(e, &mut path)?;
                continue;
            }
        };
        // the draft model only fits the model it was given with
        let draft = if path == o.model { o.draft.as_ref() } else { None };
        let mut engine = match crate::load_with_draft(&g, o.gpu, o.ctx.unwrap_or(0), draft, o.draft_k) {
            Ok(e) => e,
            Err(e) => {
                fall_back(e, &mut path)?;
                continue;
            }
        };
        let fmt = ChatFormat::detect(&g).ok();
        eprintln!(
            "[loaded {} on {} in {:.2}s, {}-token context{}]",
            id,
            engine.backend.describe(),
            t.elapsed().as_secs_f64(),
            engine.backend.ctx(),
            if engine.draft.is_some() { ", with a draft model for greedy requests" } else { "" }
        );
        if last_good.is_none() && !start_idle {
            eprintln!("chat in your browser: http://{host}:{port}/");
            eprintln!("OpenAI-compatible API: http://{host}:{port}/v1 (/v1/chat/completions, /v1/completions, /v1/models)");
        }
        if fmt.is_none() {
            eprintln!("[note: no chat template this server knows (ChatML or GLM) in this model, so /v1/chat/completions is off]");
        }
        *shared.status.lock().unwrap() = Status {
            current: Some(Loaded {
                id: id.clone(),
                ctx: engine.backend.ctx(),
                chat_ok: fmt.is_some(),
                thinking: fmt.as_ref().is_some_and(|f| f.thinking),
                glm: fmt.as_ref().is_some_and(|f| f.glm),
                coder: fmt.as_ref().is_some_and(|f| f.coder),
                backend: engine.backend.describe(),
                tok: engine.tok.clone(),
                default_system: fmt.and_then(|f| f.default_system),
            }),
            loading: None,
        };
        if let Some((w, err)) = waiting.take() {
            let _ = w.send(err.map_or(Ok(()), Err));
        }
        last_good = Some(path.clone());
        // a few short replies first (the weights into memory), then the agent's opening read in, unless a
        // request comes in meanwhile: it goes first
        let warm = shared.harness.lock().unwrap().warm_up;
        WARMING.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut early = if warm { warm_up(&mut engine, &id, &rx) } else { None };
        if early.is_none() {
            warm_agent(&mut engine, &shared);
        }
        WARMING.store(false, std::sync::atomic::Ordering::Relaxed);

        let (next, reply) = loop {
            let work = match early.take() {
                Some(w) => Ok(w),
                None => rx.recv(),
            };
            match work {
                Err(_) => return Ok(()),
                Ok(Work::Job(job)) => run(&mut engine, &id, job),
                Ok(Work::Load(p, reply)) if p == path => {
                    let _ = reply.send(Ok(()));
                }
                Ok(Work::Load(p, reply)) => break (p, reply),
            }
        };
        eprintln!("[switching from {id} to {}]", model_id(&next));
        path = next;
        waiting = Some((reply, None));
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
        let job = Job { model: id.to_string(), kind: Kind::Completion, prompt: prompt.into(), max_tokens: Some(48), temperature: 0.0, top_k: 1, top_p: 1.0, seed: 1, stop: Vec::new(), lead: "", events: tx };
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
    let t0 = Instant::now();
    let ids = engine.tok.encode(&job.prompt);
    let ctx = engine.backend.ctx();
    if ids.is_empty() || ids.len() >= ctx {
        let _ = job.events.send(Event::Error(400, format!("the prompt has {} tokens; the context holds {ctx}", ids.len())));
        return;
    }
    // reuse the cache for the part of the prompt it already holds (at least one token is read anew,
    // which gives the logits for the first reply token)
    let keep = engine.hist.iter().zip(&ids).take_while(|(a, b)| a == b).count().min(ids.len() - 1);
    let keep = engine.truncate(keep);
    let reused = keep;
    // a recurrent (DeltaNet) model goes back only to a snapshot, and one is taken where a read ends: so the
    // prompt is read in two, up to the reply's own header and then the header. The next prompt may write this
    // turn's header differently (without its empty thinking once a message follows it, as the compaction's
    // question or a nudge does): it then goes back to the header, not to the start.
    let keep = match ["<|im_start|>assistant", "<|assistant|>"].iter().filter_map(|h| job.prompt.rfind(h)).max() {
        Some(at) => {
            let head = engine.tok.encode(&job.prompt[..at]);
            let b = head.len();
            if b > keep && b < ids.len() && ids[..b] == head[..] && engine.read_in(&ids[keep..b]).is_ok() { b } else { keep }
        }
        None => keep,
    };
    let max_new = job.max_tokens.unwrap_or(usize::MAX).min(ctx - ids.len()).max(1);
    let mut sampler = Sampler::new(job.temperature, job.top_k, job.top_p, job.seed);

    let tok = engine.tok.clone();
    let mut utf8 = Utf8Stream::default();
    // text not sent yet: it could be the start of a stop string (the lead goes out with the first token)
    let mut held = job.lead.to_string();
    let mut hit_stop = false;
    let events = job.events.clone();
    let stops = job.stop.clone();
    let result = engine.generate_with(&ids[keep..], max_new, &mut sampler, &mut |id| {
        held.push_str(&utf8.push(&tok.token_bytes(id, true)));
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
    let (current, loading) = {
        let s = shared.status.lock().unwrap();
        (s.current.clone(), s.loading.clone())
    };
    let (defaults, think_default) = (shared.defaults, shared.think_default);
    // the agent tools and their settings: private to the chat page on this computer
    let path = req.path.trim_end_matches('/');
    if req.method == "GET" && req.path.starts_with("/preview/") {
        return respond_preview(&mut conn, &req, &req.path["/preview/".len()..]);
    }
    if ["/api/tools", "/api/harness", "/api/chats", "/api/agent", "/api/memory", "/api/quit", "/api/skills", "/api/mcp", "/api/instructions", "/api/background", "/api/schedules", "/api/downloads"].iter().any(|p| path.starts_with(p)) {
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
            ("POST", "/api/agent") => {
                let Some(l) = current.clone().filter(|_| loading.is_none()) else {
                    return respond_private(&mut conn, 503, &error_body("no model is running yet; try again in a moment"));
                };
                match body() {
                    Ok(b) => agent::run(&mut conn, &b, work, shared, &l),
                    Err(e) => respond_private(&mut conn, 400, &error_body(&e)),
                }
            }
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
            let models = list_models(&shared.dirs)
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
        // {"id": "<file name without .gguf>"}: answers once the switch is done (it can take a minute)
        ("POST", "/api/models/load") => {
            let id = std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse).ok().and_then(|b| b.get("id").and_then(Json::as_str).map(String::from));
            let Some(id) = id else { return respond(&mut conn, 400, &error_body("send {\"id\": \"<model>\"}")) };
            let Some((_, path, _)) = list_models(&shared.dirs).into_iter().find(|m| m.0 == id) else {
                let dirs: Vec<String> = shared.dirs.iter().map(|d| d.display().to_string()).collect();
                return respond(&mut conn, 404, &error_body(&format!("no {id}.gguf in {}", dirs.join(" or "))));
            };
            if let Some(l) = &loading {
                return respond(&mut conn, 409, &error_body(&format!("{l} is loading; wait for it")));
            }
            let (tx, done) = mpsc::channel();
            work.send(Work::Load(path, tx)).map_err(|_| "the model thread stopped")?;
            return match done.recv() {
                Ok(Ok(())) => respond(&mut conn, 200, &obj(vec![("current", str(&id))])),
                Ok(Err(e)) => respond(&mut conn, 500, &error_body(&e)),
                Err(_) => respond(&mut conn, 500, &error_body("the model thread stopped")),
            };
        }
        _ => {}
    }
    // everything else needs a running model
    let Some(Loaded { id: model_id, ctx, chat_ok, default_system, thinking, glm, coder, .. }) = current.filter(|_| loading.is_none()) else {
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
    let prompt = match (chat, glm) {
        (true, true) => chat_prompt_glm(&body, think_default, thinking),
        (true, false) => chat_prompt(&body, default_system, think_default, thinking, coder),
        (false, _) => completion_prompt(&body),
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
