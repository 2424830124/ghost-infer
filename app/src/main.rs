//! ghost-infer's native desktop app: a light window (drawn on the CPU, no GPU memory) onto the local
//! ghost-infer server. The server runs the model and the agent loop and keeps the chats; this shows them.

#![windows_subsystem = "windows"]

mod api;
mod call;
mod ghost;
// the app's words in English or Chinese
#[macro_use]
mod i18n;
#[path = "../../src/json.rs"]
#[allow(dead_code)]
mod json;
mod orb;
mod preview;
mod setup;
mod tray;
mod view;
mod voice;

use crate::i18n::tr;
use std::time::Instant;

use iced::futures::channel::{mpsc, oneshot};
use iced::widget::{markdown, text_editor};
use iced::{task, Task, Theme};

use api::Api;
use json::{num, obj, str, Json};

/// How many of the latest messages are drawn; older ones wait behind "Show earlier" (text layout is the
/// app's main memory cost, so it stays bounded however long a chat gets).
pub const SHOWN: usize = 30;

/// Command line: `ghost-app [host:port] [--open <chat id>] [--send <message>] [--menu plus|chat|settings|skill-link] [--call]`
/// (the last four for testing; --call opens the Voice call mode and starts a call).
#[derive(Debug, Clone, Default)]
struct Args {
    addr: String,
    open: Option<String>,
    send: Option<String>,
    menu: Option<String>,
    call: bool,
}

fn main() -> iced::Result {
    let mut args = Args { addr: "127.0.0.1:8080".into(), ..Default::default() };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--open" => args.open = it.next(),
            "--send" => args.send = it.next(),
            "--menu" => args.menu = it.next(),
            "--call" => args.call = true,
            _ => args.addr = a,
        }
    }
    // already running (perhaps hidden in the tray): that one comes to the front instead
    if tray::already_running(&args.addr) {
        return Ok(());
    }
    iced::application(move || App::new(args.clone()), App::update, App::view)
        .subscription(App::subscription)
        .title(|app: &App| match &app.title {
            Some(t) => format!("{t} - ghost-infer"),
            None => "ghost-infer".into(),
        })
        .theme(|app: &App| app.theme.clone())
        // tiny-skia repaints only what it thinks changed, and misses some text changes (a growing reply's
        // last lines, an icon swapped for a wider one); a background off by a hair makes it repaint the
        // whole window, so every real change does that, while the ghost's ticks stay cheap
        .style(|app: &App, _: &Theme| {
            let s = app.skin();
            let bg = if app.repaint { iced::Color { r: s.bg.r + 0.0001, ..s.bg } } else { s.bg };
            iced::theme::Style { background_color: bg, text_color: s.text }
        })
        .font(include_bytes!("../fonts/InstrumentSans.ttf").as_slice())
        .default_font(view::BODY)
        // no Windows frame: the app draws its own title bar in the skin's colours, like Claude Desktop
        .window(iced::window::Settings {
            size: iced::Size::new(1180.0, 780.0),
            min_size: Some(iced::Size::new(560.0, 420.0)),
            decorations: false,
            icon: ghost::icon(view::GRAPHITE.accent),
            platform_specific: iced::window::settings::PlatformSpecific { undecorated_shadow: true, ..Default::default() },
            // Alt+F4 hides it to the tray as the X does; the tray's menu quits
            exit_on_close_request: false,
            ..Default::default()
        })
        .run()
}

/// Messages drawn when a chat opens: SHOWN, or GHOST_APP_SHOWN=n (for measuring what a long chat costs).
fn window() -> usize {
    std::env::var("GHOST_APP_SHOWN").ok().and_then(|v| v.parse().ok()).unwrap_or(SHOWN)
}

/// A blocking call on its own thread, as a future iced can wait on.
fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> impl std::future::Future<Output = T> {
    let (tx, rx) = oneshot::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    async move { rx.await.expect(tr("the worker thread ended without an answer")) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Role {
    User,
    Assistant,
}

/// A tool call as the reply shows it: what was called and what came back.
#[derive(Debug)]
pub struct ToolLine {
    id: String,
    name: String,
    arguments: String,
    result: Option<String>,
    denied: bool,
    open: bool,          // its arguments and result shown
    ask: Option<String>, // waiting for the user's OK: the group it needs ("edit", "cmd", "web")
    pub round: usize,    // the reply's round that called it
    secs: Option<f64>,   // how long it ran
}

impl ToolLine {
    fn new(id: &str, name: &str, arguments: &str) -> ToolLine {
        ToolLine { id: id.to_string(), name: name.to_string(), arguments: arguments.to_string(), result: None, denied: false, open: false, ask: None, round: 0, secs: None }
    }

    /// The harness sending the reply back, shown as a step (its words inside): to check the code it changed,
    /// or (`goal`) to carry on with its task list's open steps.
    fn nudge(text: &str, goal: bool) -> ToolLine {
        ToolLine { result: Some(text.to_string()), ..ToolLine::new("nudge", if goal { "keep_going" } else { "check_work" }, "{}") }
    }

    /// The chat compacted to fit the context: the summary that now stands for it, opened from the line.
    fn compacted(summary: &str) -> ToolLine {
        ToolLine { result: Some(summary.to_string()), ..ToolLine::new("nudge", "compacted", "{}") }
    }

    /// Whether it changed a file (so the reply can be undone).
    fn changed_files(&self) -> bool {
        matches!(self.name.as_str(), "write_file" | "edit_file") && !self.denied && self.result.as_deref().is_some_and(|r| !r.starts_with("Error"))
    }
}

/// A command of the open chat, as Claude's background tasks list shows it: one it ran (run_command), or one
/// it left running (start_process).
pub struct BgTask {
    pub key: String,
    pub title: String, // the model's description of it, else the command
    pub command: String,
    pub output: String,
    pub secs: Option<f64>,
    pub state: TaskState,
    pub process: Option<usize>, // a background process: its id (it can be stopped, its output read)
}

#[derive(Clone, Copy, PartialEq)]
pub enum TaskState {
    Running,
    Done,
    Failed,
    Stopped,
}

/// A chat's /goal as the pin above the message box shows it.
#[derive(Debug, Clone)]
pub struct Goal {
    pub text: String,
    pub state: String, // active, met or stopped
    pub tries: usize,  // checks that said not met (or, met: the checks made)
    pub limit: usize,
    pub reason: String, // the last check's
    pub checking: bool, // a check is running
}

fn goal_of(j: Option<&Json>) -> Option<Goal> {
    let j = j.filter(|j| **j != Json::Null)?;
    let s = |k: &str| j.get(k).and_then(Json::as_str).unwrap_or("").to_string();
    let n = |k: &str| j.get(k).and_then(Json::as_f64).unwrap_or(0.0) as usize;
    Some(Goal { text: s("text"), state: s("state"), tries: n("tries"), limit: n("limit"), reason: s("reason"), checking: false })
}

/// Claude Code's permission modes: the id the server takes, the name, and what it does. Every tool is always
/// there; the mode decides which calls wait for the user's OK.
pub const MODES: [(&str, &str, &str); 5] = [
    ("auto", "Auto", "Ghost handles permission decisions"),
    ("manual", "Manual", "Always ask before making changes"),
    ("accept_edits", "Accept edits", "Automatically accept all file edits"),
    ("plan", "Plan", "Create a plan before making changes"),
    ("bypass", "Bypass permissions", "Accepts all permissions"),
];

/// The mode a number key picks in the mode menu ("1" is the first).
fn mode_key(c: &str) -> Option<Message> {
    c.parse::<usize>().ok().filter(|n| (1..=MODES.len()).contains(n)).map(|n| Message::SetMode(n - 1))
}

/// A folder's last part, for a label.
fn folder_name(path: &str) -> String {
    path.rsplit(['\\', '/']).find(|p| !p.is_empty()).unwrap_or(path).to_string()
}

/// What "Always allow" covers, by approval group.
fn group_name(group: &str) -> &'static str {
    match group {
        "edit" => tr("file changes"),
        "cmd" => tr("commands"),
        "mcp" => tr("MCP tools"),
        "skills" => tr("skill installs"),
        _ => tr("web access"),
    }
}

/// One model turn of a reply, drawn in the order it happened: its thinking, its text, then the tools it
/// called (which belong to `Msg::tools`, marked with the round's index).
#[derive(Debug)]
pub struct Round {
    raw: String, // as streamed (a thinking one starts with <think>)
    pub think: String,
    pub answer: String,
    pub md: markdown::Content, // the answer parsed
}

/// One message as drawn: a user's text, or a whole reply (its rounds and their tool calls).
#[derive(Debug)]
struct Msg {
    role: Role,
    raw: String,       // a user message's text
    rounds: Vec<Round>,
    new_round: bool,   // a tool ran since the last text: the next text starts a round
    think_open: bool,  // the thinking shown (every round's at once)
    tools: Vec<ToolLine>,
    stats: String,
    checkpoint: Option<String>, // where the reply's file changes are kept, for undo
    changed: bool,              // it changed files
    undone: bool,
    scheduled: bool, // a user message the scheduler sent (a task), shown by its label
    plan: Vec<(String, String)>, // the task list this reply finished: shown in it, no longer pinned above the box
}

impl Msg {
    fn user(text: &str) -> Msg {
        Msg {
            role: Role::User,
            raw: text.to_string(),
            rounds: Vec::new(),
            new_round: true,
            think_open: false,
            tools: Vec::new(),
            stats: String::new(),
            checkpoint: None,
            changed: false,
            undone: false,
            scheduled: false,
            plan: Vec::new(),
        }
    }

    fn reply() -> Msg {
        Msg { role: Role::Assistant, ..Msg::user("") }
    }

    /// More reply text, to the current round (a new one after a tool ran). The answer is parsed as it grows
    /// (it only ever grows at its end, so the Markdown is fed just the new part).
    fn push(&mut self, more: &str) {
        if self.new_round || self.rounds.is_empty() {
            self.rounds.push(Round { raw: String::new(), think: String::new(), answer: String::new(), md: markdown::Content::new() });
            self.new_round = false;
        }
        let r = self.rounds.last_mut().unwrap();
        r.raw.push_str(more);
        let (think, answer) = split_think(&r.raw);
        let (think, answer) = (think.to_string(), answer.trim().to_string());
        r.think = think;
        if let Some(new) = answer.strip_prefix(r.answer.as_str()) {
            if !new.is_empty() {
                r.md.push_str(new);
                r.answer = answer;
            }
        } else {
            // trailing spaces trimmed away later: parse it all again
            r.md = markdown::Content::parse(&answer);
            r.answer = answer;
        }
    }

    /// A tool call, in the round that made it (a round of its own when the model wrote nothing first).
    fn add_tool(&mut self, mut t: ToolLine) {
        if self.new_round || self.rounds.is_empty() {
            self.push("");
        }
        t.round = self.rounds.len() - 1;
        self.tools.push(t);
        self.new_round = true;
    }

    /// The whole answer: every round's text, a blank line between them (for copying).
    fn answer_text(&self) -> String {
        self.rounds.iter().map(|r| r.answer.as_str()).filter(|a| !a.is_empty()).collect::<Vec<_>>().join("\n\n")
    }
}

/// A reply's thinking (inside <think>...</think> at its start) and its answer.
fn split_think(raw: &str) -> (&str, &str) {
    let t = raw.trim_start();
    let Some(rest) = t.strip_prefix("<think>") else { return ("", raw) };
    match rest.find("</think>") {
        Some(end) => (rest[..end].trim(), rest[end + 8..].trim_start()),
        None => (rest.trim(), ""),
    }
}

/// A task list as the server keeps it: [{"text","status"}].
fn tasks_of(j: Option<&Json>) -> Vec<(String, String)> {
    j.and_then(Json::as_array)
        .unwrap_or(&[])
        .iter()
        .filter_map(|t| Some((t.get("text")?.as_str()?.to_string(), t.get("status").and_then(Json::as_str).unwrap_or("pending").to_string())))
        .collect()
}

/// Text on one line, cut to `max` characters.
pub fn clip_line(s: &str, max: usize) -> String {
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.chars().count() > max { format!("{}…", s.chars().take(max - 1).collect::<String>()) } else { s }
}

/// An MCP tool's server and tool, from its name mcp__<server>__<tool>.
fn mcp_parts(name: &str) -> (String, String) {
    let rest = name.trim_start_matches("mcp__");
    match rest.split_once("__") {
        Some((s, t)) => (s.to_string(), t.to_string()),
        None => (rest.to_string(), String::new()),
    }
}

/// A short label for a tool call, like the chat page's.
fn tool_label(t: &ToolLine) -> String {
    let a = Json::parse(&t.arguments).unwrap_or(Json::Null);
    let s = |k: &str| short(a.get(k).and_then(Json::as_str).unwrap_or(""));
    let label = match t.name.as_str() {
        "read_file" => format!("Read {}", s("path")),
        "list_files" => format!("Listed {}", if s("path").is_empty() { "the folder".into() } else { s("path") }),
        "search_files" => format!("Searched for {}", s("pattern")),
        "write_file" if a.get("append").and_then(Json::as_bool) == Some(true) => format!("Added to {}", s("path")),
        "write_file" => format!("Wrote {}", s("path")),
        "edit_file" => format!("Edited {}", s("path")),
        "run_command" => format!("Ran {}", s("command")),
        "start_process" => format!("Started {}", s("command")),
        "check_process" => format!("Checked process {}", a.get("id").map(|v| v.to_string()).unwrap_or_default()),
        "stop_process" => format!("Stopped process {}", a.get("id").map(|v| v.to_string()).unwrap_or_default()),
        "web_search" => format!("Searched the web for \"{}\"", s("query")),
        "fetch_url" => format!("Read {}", s("url")),
        "get_system_info" => "Checked this computer's specs".into(),
        "update_tasks" => "Updated the task list".into(),
        "remember" => format!("Remembered: {}", s("fact")),
        "forget" => format!("Forgot: {}", s("fact")),
        "use_skill" => format!("Used the {} skill", s("name")),
        "install_skill" => format!("Installed {}", if s("skill").is_empty() { s("source") } else { format!("{} from {}", s("skill"), s("source")) }),
        "search_chats" => format!("Searched earlier chats for \"{}\"", s("query")),
        "read_chat" => "Read an earlier chat".into(),
        "check_work" => "Asked to check its work".into(),
        "keep_going" => "Sent back to work".into(),
        "subagent" => format!("Ran a subagent: {}", s("description")),
        "compacted" => "Compacted the chat to fit the context".into(),
        "unreadable_tool_call" if t.result.as_deref().is_some_and(|r| r.contains("reached its limit")) => "Wrote a tool call the length limit cut off".into(),
        "unreadable_tool_call" => "Wrote a tool call that could not be read".into(),
        "schedule_task" => format!("Scheduled \"{}\"", if s("name").is_empty() { s("prompt") } else { s("name") }),
        "list_schedules" => "Listed the scheduled tasks".into(),
        "cancel_schedule" => format!("Cancelled scheduled task {}", s("id")),
        n if n.starts_with("mcp__") => {
            let (server, tool) = mcp_parts(n);
            format!("Used {tool} ({server} MCP)")
        }
        other => format!("Used {other}"),
    };
    if t.result.is_some() && !t.denied {
        return label;
    }
    // not done (waiting for an OK, or refused): "Edit notes.txt", not "Edited"
    let (verb, rest) = label.split_once(' ').unwrap_or((&label, ""));
    let verb = match verb {
        "Wrote" => "Write",
        "Added" => "Add",
        "Edited" => "Edit",
        "Ran" => "Run",
        "Started" => "Start",
        "Searched" => "Search",
        "Installed" => "Install",
        other => other,
    };
    let label = format!("{verb} {rest}");
    if t.denied { format!("{label} (you said no)") } else { label }
}

/// The approval card's question and what it would do, like the chat page's.
fn approval_text(t: &ToolLine) -> (String, String) {
    let a = Json::parse(&t.arguments).unwrap_or(Json::Null);
    let s = |k: &str| a.get(k).and_then(Json::as_str).unwrap_or("").to_string();
    let dir = s("cwd");
    match t.name.as_str() {
        "write_file" if a.get("append").and_then(Json::as_bool) == Some(true) => {
            let (n, path) = (s("content").lines().count(), s("path"));
            (tf!("Add {n} lines to {path}?", "向 {path} 追加 {n} 行？"), s("content"))
        }
        "write_file" => (
            tf!("Write {} ({} lines)?", "写入 {}（{} 行）？", s("path"), s("content").lines().count()),
            s("content"),
        ),
        "edit_file" => {
            let lines = |k: &str, mark: &str| s(k).lines().map(|l| format!("{mark} {l}")).collect::<Vec<_>>().join("\n");
            (tf!("Change {}?", "修改 {}？", s("path")), format!("{}\n{}", lines("old_string", "-"), lines("new_string", "+")))
        }
        "run_command" if dir.is_empty() => (tf!("Run this in PowerShell?", "在 PowerShell 中运行这条？"), s("command")),
        "run_command" => (tf!("Run this in PowerShell in {}?", "在 PowerShell 中（{}）运行这条？", &dir), s("command")),
        "start_process" if dir.is_empty() => (tf!("Start this in the background?", "在后台启动这个？"), s("command")),
        "start_process" => (tf!("Start this in the background in {}?", "在后台（{}）启动这个？", &dir), s("command")),
        "web_search" => (tf!("Search the web for \"{}\"?", "联网搜索“{}”？", s("query")), String::new()),
        "fetch_url" => (tf!("Open {}?", "打开 {}？", s("url")), String::new()),
        "install_skill" => (
            tf!("Install a skill from {}?", "从 {} 安装一个技能？", s("source")),
            if s("skill").is_empty() { String::new() } else { tf!("Skill: {}", "技能：{}", s("skill")) },
        ),
        n if n.starts_with("mcp__") => {
            let (server, tool) = mcp_parts(n);
            let args = match Json::parse(&t.arguments) {
                Ok(Json::Obj(f)) => f.iter().map(|(k, v)| format!("{k}: {}", v.as_str().map(String::from).unwrap_or_else(|| v.to_string()))).collect::<Vec<_>>().join("\n"),
                _ => t.arguments.clone(),
            };
            (tf!("Use {tool} from the {server} MCP server?", "使用 {server} MCP 服务器的 {tool}？"), args)
        }
        other => (tf!("Use {}?", "使用 {}？", other), t.arguments.clone()),
    }
}

/// A call's details for the opened tool line: its arguments, one per line, then what came back (cut short).
fn tool_detail(t: &ToolLine) -> String {
    let mut out = String::new();
    if let Ok(Json::Obj(fields)) = Json::parse(&t.arguments) {
        for (k, v) in fields {
            let v = match v {
                Json::Str(s) => s,
                other => other.to_string(),
            };
            out.push_str(&format!("{k}: {}\n", clip(&v, 12, 1200)));
        }
    }
    if let Some(r) = &t.result {
        out.push_str(&format!("\n{}", clip(r, 40, 4000)));
    }
    out.trim_end().to_string()
}

/// Text cut to its first `lines` lines and `chars` characters, saying how much was left out.
fn clip(s: &str, lines: usize, chars: usize) -> String {
    let total = s.lines().count();
    let mut out: String = s.lines().take(lines).collect::<Vec<_>>().join("\n");
    if out.chars().count() > chars {
        out = out.chars().take(chars).collect();
    }
    if total > lines || out.len() < s.trim_end().len() {
        out.push_str(&format!("\n… ({total} lines in all)"));
    }
    out
}

/// A label value kept short: a long path by its last two parts, other long text cut.
fn short(v: &str) -> String {
    let v = v.trim();
    if v.chars().count() <= 60 {
        return v.to_string();
    }
    let parts: Vec<&str> = v.split(['\\', '/']).filter(|p| !p.is_empty()).collect();
    if parts.len() > 2 && !v.contains(' ') {
        return format!("…\\{}", parts[parts.len() - 2..].join("\\"));
    }
    format!("{}…", v.chars().take(57).collect::<String>())
}

/// A model in the picker: its file name, shown by a short name.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelChoice {
    pub id: String,
    pub label: String,
    pub size: u64, // the file's bytes (0: not known)
}

/// The largest model file a voice call runs on: what fits in an 8 GB GPU beside the voice and Whisper (about
/// 2.7 GB together) with room for its context. A bigger one (a 30B mixture of experts) is swapped for the largest
/// model under this for the call, and back after it.
const CALL_MAX: u64 = 3_500_000_000;

impl std::fmt::Display for ModelChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label)
    }
}

/// A model's file name without its quantisation and the noise around it, for people.
pub fn short_model(id: &str) -> String {
    let base = id.rsplit_once(['-', '.']).filter(|(_, q)| q.starts_with(['Q', 'q', 'I', 'F'])).map_or(id, |(b, _)| b);
    let base = base.trim_end_matches(".i1").trim_start_matches("Huihui-").trim_start_matches("JOSIEFIED-");
    if base.chars().count() > 28 { format!("{}…", base.chars().take(27).collect::<String>()) } else { base.to_string() }
}

/// A saved chat as the sidebar lists it.
#[derive(Debug, Clone)]
pub struct ChatItem {
    pub id: String,
    pub title: String,
    pub updated: f64, // milliseconds since 1970
    pub unread: bool, // a scheduled run wrote to it since the user last opened it
}

/// The open menu, if any.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Menu {
    Chat,
    Settings,
    Plus,
    Model,  // under the message box: the models to switch to
    CallModel, // on the call screen: the model calls talk to
    Mode,   // the permission modes
    Folder, // a new chat's working directory
}

impl Menu {
    /// The widget a menu opens above, if it opens above one.
    fn anchor(self) -> Option<&'static str> {
        match self {
            Menu::Plus => Some("plus"),
            Menu::Model => Some("model"),
            Menu::CallModel => Some("call_model"),
            Menu::Mode => Some("mode"),
            Menu::Folder => Some("folder"),
            Menu::Chat | Menu::Settings => None,
        }
    }
}

/// A section of the settings window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SettingsPage {
    General,
    Appearance,
    Models,
    Voice,
    Maya,
    About,
}

/// Light or dark: Windows' own setting unless the user picked one (kept in %LOCALAPPDATA%\ghost-infer\app.json).
#[derive(Debug, Clone, Copy, PartialEq)]
enum ThemeMode {
    System,
    Dark,
    Light,
}

fn app_file() -> Option<std::path::PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|d| std::path::PathBuf::from(d).join("ghost-infer").join("app.json"))
}

/// The app's own settings, kept in app.json (start-ghost-app reads load_last and last_model from it too).
struct Saved {
    mode: ThemeMode,
    perm: usize,                // the permission mode (MODES)
    workdir: Option<String>,    // the folder new chats work in
    rail: bool,                 // the rail is shown
    rail_open: Vec<String>,     // its cards (and their inner lists) that are open
    load_last: bool,            // the launcher starts the server with the last picked model
    last_model: Option<String>, // the model last picked in the app
    model_dirs: Vec<String>,    // folders of models already on this computer (an installed app's server reads them)
    call_timings: bool,         // a call shows how long each reply took (Settings, Voice)
    call_model: Option<String>, // the model calls talk to (None: the Code model's)
    lang: crate::i18n::Lang,    // the app's words: English, Chinese, or whatever Windows is set to
}

fn load_settings() -> Saved {
    let j = app_file().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| Json::parse(&t).ok()).unwrap_or(Json::Null);
    let mode = match j.get("theme").and_then(Json::as_str) {
        Some("dark") => ThemeMode::Dark,
        Some("light") => ThemeMode::Light,
        _ => ThemeMode::System,
    };
    let perm = j.get("permission_mode").and_then(Json::as_str).and_then(|m| MODES.iter().position(|(id, _, _)| *id == m));
    Saved {
        mode,
        perm: perm.unwrap_or(1), // manual, as Claude Code starts
        workdir: j.get("workdir").and_then(Json::as_str).map(String::from),
        rail: j.get("rail").and_then(Json::as_bool).unwrap_or(false),
        rail_open: match j.get("rail_open").and_then(Json::as_array) {
            Some(a) => a.iter().filter_map(Json::as_str).map(String::from).collect(),
            None => ["environment", "context", "changes", "tasks", "background"].map(String::from).to_vec(),
        },
        load_last: j.get("load_last").and_then(Json::as_bool).unwrap_or(false),
        last_model: j.get("last_model").and_then(Json::as_str).map(String::from),
        model_dirs: j.get("model_dirs").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect(),
        call_timings: j.get("call_timings").and_then(Json::as_bool).unwrap_or(false),
        call_model: j.get("call_model").and_then(Json::as_str).map(String::from),
        lang: match j.get("lang").and_then(Json::as_str) {
            Some("en") => crate::i18n::Lang::En,
            Some("zh") => crate::i18n::Lang::Zh,
            _ => crate::i18n::Lang::System,
        },
    }
}

/// A link from a reply, in the default browser. Only web links: a reply could link to a program or a file.
/// Handed to Windows as it is (ShellExecuteW): `cmd /C start` would run whatever follows an & in it.
fn open_link(url: &str) {
    let lower = url.trim().to_ascii_lowercase();
    if !(lower.starts_with("https://") || lower.starts_with("http://")) {
        return;
    }
    #[link(name = "shell32")]
    unsafe extern "system" {
        fn ShellExecuteW(hwnd: isize, op: *const u16, file: *const u16, params: *const u16, dir: *const u16, show: i32) -> isize;
    }
    const SW_SHOWNORMAL: i32 = 1;
    let w = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let (op, file) = (w("open"), w(url.trim()));
    unsafe { ShellExecuteW(0, op.as_ptr(), file.as_ptr(), std::ptr::null(), std::ptr::null(), SW_SHOWNORMAL) };
}

/// Whether Windows is set to dark mode for apps (HKCU\...\Personalize\AppsUseLightTheme = 0).
fn windows_dark() -> bool {
    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn RegGetValueW(key: isize, sub: *const u16, value: *const u16, flags: u32, kind: *mut u32, data: *mut u32, len: *mut u32) -> i32;
    }
    const HKEY_CURRENT_USER: isize = 0x8000_0001u32 as i32 as isize;
    const RRF_RT_REG_DWORD: u32 = 0x10;
    let w = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let (sub, value) = (w(r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"), w("AppsUseLightTheme"));
    let (mut data, mut len) = (1u32, 4u32);
    let r = unsafe { RegGetValueW(HKEY_CURRENT_USER, sub.as_ptr(), value.as_ptr(), RRF_RT_REG_DWORD, std::ptr::null_mut(), &mut data, &mut len) };
    r == 0 && data == 0
}

struct App {
    theme: Theme,
    mode: ThemeMode,
    lang: crate::i18n::Lang,
    dark: bool,
    menu: Option<Menu>,
    clock: u64,               // milliseconds since the app started, for the ghost
    born: Instant,
    last_chunk_at: Option<u64>, // when text last arrived (the ghost's mouth moves while it does)
    happy_until: u64,         // the ghost smiles a moment after a reply
    last_active: Instant,     // the user's last action (the ghost dozes off after a while)
    searching: bool,
    filter: String,
    recents_open: bool,
    api: Option<Api>,
    status: String,
    models: Vec<ModelChoice>,
    model: Option<ModelChoice>,
    loading: Option<String>, // the model the server is loading (none: idle, or started without one)
    warming: bool,           // the model loaded and is warming up (a few short replies, the agent's opening read in)
    chats: Vec<ChatItem>,
    chat: Option<String>,
    title: Option<String>,
    messages: Vec<Msg>,
    shown: usize,
    composer: text_editor::Content,
    thinking: bool,
    busy: Option<task::Handle>,
    started: Option<Instant>,
    chunks: usize,               // text pieces of this round so far (about one token each)
    writing_since: Option<Instant>, // this round's first text: the speed shown counts from it
    reply_tokens: f64,           // tokens this reply's rounds have generated (the server's count)
    reply_writing: (f64, f64),   // over its rounds: tokens after each round's first, and the seconds they took
    context: (f64, f64), // tokens used, context length
    start: Args,         // what the command line asked to open or send once connected
    sidebar: bool,       // the chat list shown (it hides with the pane button)
    hovered: Option<String>, // the chat row under the pointer (its delete button shows)
    folders: Vec<String>,    // the other folders file tools may use (the server's harness settings)
    workdir: Option<String>, // the folder a new chat works in (None: the first of `folders`)
    chat_cwd: Option<String>, // the open chat's working directory
    perm: usize,             // the permission mode (MODES)
    goal: Option<Goal>,      // the open chat's /goal
    repaint: bool,          // flipped on every change but the ghost's tick: the whole window is drawn again
    maximized: bool,         // the window fills the screen (no resize edges; the button shows "restore")
    window: iced::Size,      // the window's size, to place menus
    menu_at: Option<iced::Rectangle>, // where the open menu's button is (the menu opens above it)
    rail: bool,              // the session rail on the right is shown
    rail_open: Vec<String>,  // its cards (and their inner lists) that are open
    parts: Json,             // how the open chat's last prompt split ({"system","tools","skills","instructions","memory"})
    load_last: bool,         // start with the last picked model (the launcher reads it)
    last_model: Option<String>, // the model last picked here
    system: Json,            // this computer, as /api/system reports it (for the rail)
    tasks: Vec<(String, String)>, // the model's task list for this chat: text, and pending / in_progress / done
    plan_open: Option<bool>, // the plan above the message box opened or closed by hand (None: open until it is all done)
    memory: Vec<String>,     // what the model remembers across chats
    memory_draft: String,    // a memory being typed in the rail
    instructions: Vec<(String, bool, bool)>, // the GHOST.md files: path, whether it exists, whether it is the global one
    skills: Vec<(String, String)>, // installed skills: name, description
    skill_link: String,      // a skill link being pasted in the rail
    skill_choices: Option<(String, Vec<(String, String, String)>)>, // a source with several skills: it, and each one's path, name, description
    mcp: Vec<(String, String, usize)>, // MCP servers: name, state, how many tools
    mcp_config: String,      // where mcp.json is
    background: Json,        // what runs on its own: the scheduled tasks (/api/background)
    polled_at: u64,          // when /api/background was last asked (the clock), while no answer is pending
    processes: Json,         // the commands the model left running (start_process), from the same poll
    tasks_panel: bool,       // the background tasks panel beside the rail is open
    tasks_wide: bool,        // and wider
    task_open: Vec<String>,  // its tasks unfolded (BgTask keys)
    tasks_hidden: Vec<String>, // finished ones cleared from it
    tasks_shown: usize,      // how many finished ones it lists
    finished_open: bool,     // its finished group unfolded
    proc_output: std::collections::HashMap<usize, String>, // running processes' output, while the panel shows them
    polling: bool,
    own_server: bool,        // installed with its server (ghost-infer.exe beside it), which it starts itself
    model_dirs: Vec<String>, // folders of models added in the app (given to the server when it starts, and when the app connects)
    model_folders: Vec<String>, // every folder the server lists models from (its own and the app's)
    download: Json,          // a model downloading for the first run (/api/downloads)
    download_polled: u64,    // when it was last asked (the clock)
    download_polling: bool,
    download_rate: (f64, Option<Instant>, f64), // bytes at the last look, when, and bytes a second
    pick_model: bool,        // a folder of models was just added: the model menu opens once connected
    warm_up: bool,           // the server runs a few short replies after a model loads (its harness setting)
    preview_url: Option<String>, // the web page the agent last wrote, for the preview pane
    preview_open: bool,      // the pane is shown
    chip: Option<String>,    // a command or skill picked with /: a chip before the text, sent as /name
    slash_pick: usize,       // the / list's highlighted line
    slash_shut: Option<String>, // the text the / list was closed at (Esc): shut until the text changes
    composer_at: Option<iced::Rectangle>, // the message box on screen, for the / list above it
    voice: bool,             // the Voice call mode (the top-left switch) instead of Code
    voice_models: (Option<String>, Option<String>), // the server's speech and speech-to-text models (--tts, --stt)
    call: Option<call::Call>, // a voice call going on
    call_state: String,      // listening, thinking, speaking, or what went wrong
    settings_page: SettingsPage, // the section the settings window shows
    maya_memory: Json,           // what Maya remembers of the user (GET /api/maya), for its settings page
    maya_name: String,           // the name she calls them, as typed on that page
    call_lines: Vec<(bool, String)>, // the call's turns: (the user's, the text)
    call_mood: Option<(String, String)>, // Maya's mood and way of talking, as her last reply said
    call_chat: Option<String>, // the chat the call writes in
    call_pending: bool,      // a call asked for while the model loads: it starts once the model runs
    call_timing: String,     // how long the last turn took, step by step
    call_timings: bool,      // ...shown under the call (app.json's call_timings)
    call_typing: bool,       // the call's message box is open
    call_task: Option<task::Handle>, // the call's events (stopped when it is hung up)
    call_choice: Option<String>, // the model calls talk to (None: the Code model's; app.json's call_model)
    call_ready: Option<String>, // the model a call would talk to now, loaded with the voice
    together: bool,          // the Code model and the call model (with the voice) both loaded: no switch either way
    call_confirm: Option<(String, &'static str)>, // the question before a call that unloads the Code model, and its yes
    asked_pair: bool,        // the server was asked for the Code and call models together (once a model runs)
    voice_on: bool,          // the server has its voice models loaded (they load with a model, for calls)
    call_wait: Option<String>, // the model a call switched to, with the voice: it starts once that runs
    call_text: String,       // a message being typed during a call
    orb: orb::Sim,           // the call's face: the pixel orb's motion
    call_started: Option<std::time::Instant>, // when the call going on began (Maya is told how long it was)
}

/// A line of the / list: a command of the server's, or an installed skill.
pub struct Slash {
    pub name: String,
    pub about: String,
}

/// The commands / offers before the skills.
fn commands() -> [(&'static str, &'static str); 2] {
    [
        ("goal", tr("Keep working until what you describe is done, checked after each try")),
        ("loop", tr("Repeat a prompt in this chat every few minutes: /loop 10m check the build")),
    ]
}

/// Where the text names one of `names` with a slash (/design-taste-frontend), as byte ranges: a / at the start
/// or after a space, then the name, ending at a space, a line's end or a mark such as a comma.
pub fn slash_spans(text: &str, names: &[String]) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    for (at, _) in text.match_indices('/') {
        if at > 0 && !text[..at].ends_with(char::is_whitespace) {
            continue;
        }
        let rest = &text[at + 1..];
        let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '-' || c == '_')).unwrap_or(rest.len());
        if end > 0 && names.iter().any(|n| n.eq_ignore_ascii_case(&rest[..end])) {
            out.push(at..at + 1 + end);
        }
    }
    out
}

/// A command's or skill's name as the chip shows it: design-taste-frontend is Design Taste Frontend.
pub fn slash_label(name: &str) -> String {
    let word = |w: &str| {
        let mut c = w.chars();
        c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
    };
    name.split(['-', '_', ' ']).filter(|w| !w.is_empty()).map(word).collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Clone)]
enum Message {
    Connected(Result<(Api, Json, Json, Json), String>),
    Chats(Result<Json, String>),
    OpenChat(String),
    ChatLoaded(Result<Json, String>),
    NewChat,
    DeleteChat(String),
    PickModel(ModelChoice),
    ModelLoaded(Result<Json, String>),
    Edit(text_editor::Action),
    Tray(tray::Ask),           // the tray icon's click or menu, or a second start of the app
    SlashMove(i32),            // Up/Down in the / list
    SlashPick(Option<String>), // a line of the / list (None: the highlighted one)
    SlashClose,                // Esc: the / list shuts until the text changes
    ChipRemove,                // Backspace before any text
    ComposerAt(Option<iced::Rectangle>),
    Send,
    Stop,
    Agent(Json),
    ToggleThinkView(usize),
    ToggleThinking,
    ShowEarlier,
    Link(String),
    Retry,
    ToggleSidebar,
    Hover(Option<String>),
    Copy(usize),
    Tick,
    Menu(Option<Menu>),
    SetTheme(ThemeMode),
    SetLang(crate::i18n::Lang),
    SettingsPage(SettingsPage),
    MayaMemory(Result<Json, String>),
    MayaForget(&'static str, usize),
    MayaName(String),
    SaveMayaName,
    ClearImpression,
    ResetMaya,
    ToggleCallTimings,
    ToggleCallTyping,
    ToggleSearch,
    Filter(String),
    ToggleRecents,
    CopyChat,
    DeleteCurrent,
    ToggleTool(usize, usize), // a reply's tool line opened or closed
    SetMode(usize),           // a permission mode (MODES)
    ClearGoal,                // the goal pin's x: sends /goal clear
    ToggleCard(&'static str), // a rail card (or a list inside one) opened or closed
    TrimChat,                 // the chat's old tool outputs taken out now
    StopProcess(usize),       // a background process ended (the rail's stop button)
    ShowTasks,                // the background tasks panel
    CloseTasks,
    WideTasks,
    ToggleTask(String),       // a task in the panel unfolded or folded
    ToggleFinished,
    ClearFinished(Vec<String>),
    MoreTasks,
    ProcOutput(usize, Result<Json, String>),
    Trimmed(Result<Json, String>),
    CycleMode,                // Shift+Tab, as in Claude Code
    PickWorkdir,              // a new chat's folder, from Windows' folder dialog
    SetWorkdir(Option<String>),
    Approve(String, &'static str), // a call's id, and "once", "always" or "deny"
    Approved(Result<Json, String>),
    Undo(usize),
    Undone(usize, Result<Json, String>),
    // the window, which has no Windows frame
    DragWindow,
    ToggleMaximize,
    Minimize,
    CloseWindow,
    ResizeWindow(iced::window::Direction),
    WindowResized(iced::Size),
    OpenMenu(Menu), // a menu that opens above its button: found first
    MenuAt(Menu, Option<iced::Rectangle>),
    Maximized(bool),
    ToggleRail,
    TogglePlan,
    ToggleLoadLast,
    ToggleWarmUp,
    Quit,
    System(Result<Json, String>),
    UndoAll,
    AddFolder,
    FolderPicked(Option<String>),
    RemoveFolder(usize),
    Harness(Result<Json, String>),
    Memory(Result<Json, String>),
    Forget(usize),
    MemoryDraft(String),
    Instructions(Result<Json, String>),
    EditInstructions(String),
    AddMemory,
    InstallSkill(bool), // from an archive (true) or a folder
    SkillPicked(Option<String>),
    SkillLink(String),
    InstallLink,
    OpenSkillLink, // the + menu: the rail, with its link field focused
    InstallChoice(Option<String>), // one of a source's skills, by path (None: all of them)
    CancelChoice,
    Skills(Result<Json, String>),
    RemoveSkill(String),
    Mcp(Result<Json, String>),
    McpReload,
    McpImport,
    McpEdit,
    Background(Result<Json, String>),
    RunScheduled(String), // a task's id: now
    CancelScheduled(String),
    Read(Result<Json, String>),
    // the first run: a model to download (setup::OFFERS), or a folder of models already on this computer
    GetModel(usize),
    DownloadStatus(Result<Json, String>),
    CancelDownload,
    UseModelFolder,
    ModelFolder(Option<String>),
    RemoveModelFolder(String),
    ModelFolders(Result<Json, String>), // the server's models folders, after a change
    // the preview pane: the page's view made, and the pane's buttons
    PreviewMade(Result<(), String>),
    PreviewOpen,
    PreviewClose,
    PreviewReload,
    PreviewExternal,
    FocusApp, // a click in the app: the keyboard comes back from the page
    // voice calls
    SetVoice(bool), // the top-left switch: Voice call (true) or Code
    PickCallModel(Option<String>), // the model calls talk to (None: the Code model's)
    ConfirmCall(bool), // the answer to call_confirm: unload the Code model and call, or not
    StartCall,
    EndCall,
    Reflected(Result<Json, String>), // Maya has thought the call over (her memory now)
    ToggleMute,
    Call(call::CallEvent),
    CallText(String), // the call's text box changed
    SendCallText,     // Enter in it: a typed turn
}

impl App {
    fn new(args: Args) -> (App, Task<Message>) {
        let addr = args.addr.clone();
        if let Some((rgba, side)) = ghost::icon_rgba(view::GRAPHITE.accent) {
            tray::install(rgba, side);
        }
        let Saved { mode, perm, workdir, rail, rail_open, load_last, last_model, model_dirs, call_timings, call_model, lang } = load_settings();
        // an installed app starts its own server (with the last model if it is set to), as the launcher does
        let launch = (model_dirs.clone(), if load_last { last_model.clone() } else { None });
        let dark = match mode {
            ThemeMode::Dark => true,
            ThemeMode::Light => false,
            ThemeMode::System => windows_dark(),
        };
        // the app's words, before anything is drawn
        crate::i18n::set(lang);
        let app = App {
            theme: if dark { view::GRAPHITE.theme() } else { view::FOG.theme() },
            mode,
            lang,
            dark,
            menu: None,
            clock: 0,
            born: Instant::now(),
            last_chunk_at: None,
            happy_until: 0,
            last_active: Instant::now(),
            searching: false,
            filter: String::new(),
            recents_open: true,
            api: None,
            status: format!("Connecting to {addr}…"),
            models: Vec::new(),
            model: None,
            loading: None,
            warming: false,
            chats: Vec::new(),
            chat: None,
            title: None,
            messages: Vec::new(),
            shown: window(),
            composer: text_editor::Content::new(),
            thinking: true,
            busy: None,
            started: None,
            chunks: 0,
            writing_since: None,
            reply_tokens: 0.0,
            reply_writing: (0.0, 0.0),
            context: (0.0, 0.0),
            start: args,
            sidebar: true,
            hovered: None,
            folders: Vec::new(),
            workdir,
            chat_cwd: None,
            perm,
            goal: None,
            repaint: false,
            maximized: false,
            window: iced::Size::new(1180.0, 780.0),
            menu_at: None,
            rail,
            rail_open,
            parts: Json::Null,
            load_last,
            last_model,
            system: Json::Null,
            tasks: Vec::new(),
            plan_open: None,
            memory: Vec::new(),
            memory_draft: String::new(),
            instructions: Vec::new(),
            skills: Vec::new(),
            chip: None,
            slash_pick: 0,
            slash_shut: None,
            composer_at: None,
            skill_link: String::new(),
            skill_choices: None,
            mcp: Vec::new(),
            mcp_config: String::new(),
            background: Json::Null,
            polled_at: 0,
            processes: Json::Null,
            tasks_panel: false,
            tasks_wide: false,
            task_open: Vec::new(),
            tasks_hidden: Vec::new(),
            tasks_shown: 12,
            finished_open: true,
            proc_output: Default::default(),
            polling: false,
            own_server: setup::own_server().is_some(),
            model_dirs,
            download: Json::Null,
            download_polled: 0,
            download_polling: false,
            download_rate: (0.0, None, 0.0),
            pick_model: false,
            model_folders: Vec::new(),
            warm_up: true,
            preview_url: None,
            preview_open: false,
            voice: false,
            orb: orb::Sim::new(),
            voice_models: (None, None),
            call: None,
            call_state: String::new(),
            settings_page: SettingsPage::General,
            maya_memory: Json::Null,
            maya_name: String::new(),
            call_lines: Vec::new(),
            call_mood: None,
            call_chat: None,
            call_pending: false,
            call_timing: String::new(),
            call_timings,
            call_typing: false,
            call_task: None,
            call_choice: call_model,
            call_ready: None,
            together: false,
            call_confirm: None,
            asked_pair: false,
            voice_on: false,
            call_wait: None,
            call_text: String::new(),
            call_started: None,
        };
        let connecting = blocking(move || {
            setup::start_server(&addr, &launch.0, launch.1.as_deref());
            connect(&addr)
        });
        (app, Task::perform(connecting, Message::Connected))
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        if !matches!(message, Message::Tick | Message::Agent(_) | Message::Hover(_) | Message::Background(_) | Message::Read(_)) {
            self.last_active = Instant::now();
        }
        // the background poll repaints only when it brings news (it sets `repaint` itself)
        if !matches!(message, Message::Tick | Message::Background(_)) {
            self.repaint = !self.repaint;
        }
        match message {
            Message::Connected(Ok((api, models, chats, harness))) => {
                self.api = Some(api);
                self.set_harness(&harness);
                self.set_models(&models);
                self.set_chats(&chats);
                self.status = String::new();
                let menu = self.start.menu.take();
                self.menu = match menu.as_deref() {
                    Some("chat") => Some(Menu::Chat),
                    Some("settings") => Some(Menu::Settings),
                    _ => None,
                };
                if self.model.is_none() {
                    self.status = self.no_model_status();
                } else if let Some(m) = self.model.as_ref().filter(|_| self.warming) {
                    self.status = format!("Warming up {}…", m.label);
                }
                // the + menu opens as a click opens it: once the button is on screen
                let plus = match menu.as_deref() {
                    Some("plus") => Task::done(Message::OpenMenu(Menu::Plus)),
                    Some("model") => Task::done(Message::OpenMenu(Menu::Model)),
                    Some("mode") => Task::done(Message::OpenMenu(Menu::Mode)),
                    Some("tasks") => Task::done(Message::ShowTasks),
                    Some("skill-link") => Task::done(Message::OpenSkillLink),
                    _ => Task::none(),
                };
                // no model at all yet: a download that may be under way; models but none running after a folder of
                // them was added: the model menu, to pick one
                let first_run = if self.models.is_empty() {
                    self.call(|api| api.get("/api/downloads"), Message::DownloadStatus)
                } else if std::mem::take(&mut self.pick_model) && self.model.is_none() && self.loading.is_none() {
                    Task::done(Message::OpenMenu(Menu::Model))
                } else {
                    Task::none()
                };
                let call = if std::mem::take(&mut self.start.call) {
                    self.voice = true;
                    Task::done(Message::StartCall)
                } else {
                    Task::none()
                };
                // the folders added in the app, given to a server started without them (the launcher's), and the list
                let dirs = self.model_dirs.clone();
                let folders = self.call(
                    move |api| {
                        for d in &dirs {
                            let _ = api.post("/api/models/folders", &obj(vec![("add", str(d))]));
                        }
                        api.get("/api/models/folders")
                    },
                    Message::ModelFolders,
                );
                let system = Task::batch([self.refresh_system(), self.refresh_memory(), self.refresh_extras(), self.wait_for_model(), plus, first_run, call, folders]);
                if let Some(id) = self.start.open.take() {
                    return Task::batch([system, self.call(move |api| api.get(&format!("/api/chats/{id}")), Message::ChatLoaded)]);
                }
                if let Some(text) = self.start.send.take() {
                    self.composer = text_editor::Content::with_text(&text);
                    return Task::batch([system, self.send(None)]);
                }
                return system;
            }
            // the server not up yet (the launcher starts both at once): try again in a moment
            Message::Connected(Err(_)) => {
                let addr = self.start.addr.clone();
                self.status = format!("Waiting for ghost-infer at {addr}…");
                return Task::perform(
                    blocking(move || {
                        std::thread::sleep(std::time::Duration::from_secs(2));
                        connect(&addr)
                    }),
                    Message::Connected,
                );
            }
            Message::Chats(Ok(c)) => self.set_chats(&c),
            Message::Chats(Err(e)) | Message::ChatLoaded(Err(e)) => self.status = e,
            Message::ModelLoaded(Err(e)) => {
                self.status = e.clone();
                // the switch for a call failed: no call, and the model from before it back
                if std::mem::take(&mut self.call_pending) {
                    self.call_wait = None;
                    self.call_state = format!("Couldn't switch models for the call: {e}");
                    return self.back_to_code();
                }
            }
            Message::OpenChat(id) => {
                if self.busy.is_none() {
                    return self.call(move |api| api.get(&format!("/api/chats/{id}")), Message::ChatLoaded);
                }
            }
            Message::ChatLoaded(Ok(chat)) => {
                self.show_chat(&chat);
                // opened with a scheduled run's reply in it: seen now
                if chat.get("unread").and_then(Json::as_bool).unwrap_or(false) {
                    let body = obj(vec![("id", chat.get("id").cloned().unwrap_or(Json::Null))]);
                    return self.call(
                        move |api| {
                            api.post("/api/chats/read", &body)?;
                            api.get("/api/chats")
                        },
                        Message::Read,
                    );
                }
            }
            Message::NewChat => {
                if self.busy.is_none() {
                    self.clear_chat();
                }
            }
            // the open chat deleted: a new chat in its place, as New chat leaves it (its task list and goal
            // went on showing above the message box)
            Message::DeleteChat(id) => {
                if self.chat.as_deref() == Some(id.as_str()) {
                    self.clear_chat();
                }
                return self.call(
                    move |api| {
                        api.delete(&format!("/api/chats/{id}"))?;
                        api.get("/api/chats")
                    },
                    Message::Chats,
                );
            }
            Message::PickModel(choice) => {
                self.menu = None;
                let id = choice.id;
                self.last_model = Some(id.clone());
                self.save_settings();
                self.status = format!("Loading {}…", choice.label);
                return self.load_models(false);
            }
            Message::ModelLoaded(Ok(models)) => {
                let was_warming = self.warming;
                self.set_models(&models);
                // loaded, warming up (asked again until it is done), then ready; a reply's own status stays
                if self.busy.is_none() {
                    self.status = match &self.model {
                        None => self.no_model_status(),
                        Some(m) if self.warming => format!("Warming up {}…", m.label),
                        Some(_) if was_warming => "Warmed up and ready".into(),
                        Some(_) => String::new(),
                    };
                }
                // a call asked for while the model loaded
                // (a poll while it switched models for the call reports the old ones: only the call model it switched
                // to counts, once running, beside the Code model or alone)
                let ready = self.loading.is_none()
                    && match &self.call_wait {
                        None => self.model.is_some(),
                        Some(w) => self.call_ready.as_ref() == Some(w) || self.model.as_ref().is_some_and(|m| m.id == *w),
                    };
                let call = if ready && std::mem::take(&mut self.call_pending) { Task::done(Message::StartCall) } else { Task::none() };
                // a model running that the app did not ask for with its call model (the launcher's): the call model
                // and the voice beside it too, if they fit
                if !self.asked_pair && self.loading.is_none() && self.model.is_some() && self.call.is_none() && !self.call_pending && self.voice_models.0.is_some() && self.voice_models.1.is_some() {
                    self.asked_pair = true;
                    if self.call_ready.is_none() {
                        return Task::batch([self.refresh_system(), self.load_models(false)]);
                    }
                }
                return Task::batch([self.wait_for_model(), self.refresh_system(), call]);
            }
            Message::Edit(action) => {
                let (edits, shape) = (action.is_edit(), self.composer_shape());
                self.composer.perform(action);
                let task = if edits { self.after_edit() } else { Task::none() };
                return self.keep_focus(shape, task);
            }
            Message::SlashMove(d) => {
                let n = self.slash_list().len().max(1) as i32;
                self.slash_pick = (self.slash_pick as i32 + d).rem_euclid(n) as usize;
            }
            // at the start of the box the pick becomes the chip; further on, its name goes in the text in place
            // of what was typed, styled there
            Message::SlashPick(name) => {
                let list = self.slash_list();
                let name = name.or_else(|| list.get(self.slash_pick.min(list.len().saturating_sub(1))).map(|e| e.name.clone()));
                if let (Some(name), Some((q, at, first))) = (name, self.slash_typed()) {
                    if first {
                        let text = self.composer.text();
                        self.chip = Some(name);
                        self.composer = text_editor::Content::with_text(text[at + 1 + q.len()..].trim());
                    } else {
                        for _ in 0..=q.chars().count() {
                            self.composer.perform(text_editor::Action::Select(text_editor::Motion::Left));
                        }
                        self.composer.perform(text_editor::Action::Edit(text_editor::Edit::Paste(std::sync::Arc::new(format!("/{name} ")))));
                    }
                    self.slash_pick = 0;
                }
                return iced::widget::operation::focus("editor");
            }
            Message::SlashClose => self.slash_shut = Some(self.composer.text()),
            Message::ChipRemove => {
                self.chip = None;
                return iced::widget::operation::focus("editor");
            }
            Message::ComposerAt(at) => self.composer_at = at,
            Message::Send => {
                let shape = self.composer_shape();
                let task = self.send(None);
                return self.keep_focus(shape, task);
            }
            Message::Retry => return self.send(Some(())),
            Message::Stop => {
                if let Some(h) = self.busy.take() {
                    h.abort();
                }
                self.finish("stopped");
            }
            Message::Agent(ev) => return self.on_event(ev),
            Message::ToggleThinkView(i) => {
                if let Some(m) = self.messages.get_mut(i) {
                    m.think_open = !m.think_open;
                }
            }
            Message::ToggleThinking => self.thinking = !self.thinking,
            Message::ShowEarlier => self.shown += SHOWN,
            Message::ToggleSidebar => self.sidebar = !self.sidebar,
            Message::Tick => {
                self.clock = self.born.elapsed().as_millis() as u64;
                // a call: the microphone's level and the ghost's mouth move on every tick
                if self.call.is_some() {
                    self.repaint = !self.repaint;
                }
                if self.voice {
                    let (phase, level) = self.orb_input();
                    self.orb.step(self.clock, phase, level);
                }
                // a model downloading: a look every second
                let downloading = self.download.get("state").and_then(Json::as_str) == Some("downloading");
                if downloading && !self.download_polling && self.clock.saturating_sub(self.download_polled) > 1000 {
                    self.download_polling = true;
                    return self.call(|api| api.get("/api/downloads"), Message::DownloadStatus);
                }
                // scheduled runs happen on the server on their own: a look every few seconds (the first at once)
                let due = self.background == Json::Null || self.clock.saturating_sub(self.polled_at) > 4000;
                if !self.polling && self.api.is_some() && due {
                    self.polling = true;
                    return self.call(|api| api.get("/api/background"), Message::Background);
                }
            }
            Message::Background(r) => {
                self.polling = false;
                self.polled_at = self.clock;
                let Ok(j) = r else { return Task::none() };
                // background processes change on their own (their running time too): drawn again when they do
                let procs = j.get("processes").cloned().unwrap_or(Json::Null);
                if procs != self.processes {
                    self.processes = procs;
                    self.repaint = !self.repaint;
                }
                let outputs = self.fetch_outputs();
                let seq = |j: &Json| j.get("seq").and_then(Json::as_f64).unwrap_or(0.0);
                let running = |j: &Json| j.get("running").and_then(Json::as_str).map(String::from);
                if seq(&j) == seq(&self.background) {
                    return outputs;
                }
                self.repaint = !self.repaint;
                let (was, now) = (running(&self.background), running(&j));
                let first = self.background == Json::Null;
                self.background = j;
                if first {
                    return outputs;
                }
                let mut tasks = vec![outputs, self.call(|api| api.get("/api/chats"), Message::Chats)];
                // a run ended: if it wrote to the open chat, that chat is shown again with it
                if let Some(c) = was.filter(|_| now.is_none()) {
                    if self.busy.is_none() && (self.chat.as_deref() == Some(c.as_str()) || c.is_empty()) {
                        if let Some(id) = self.chat.clone() {
                            tasks.push(self.call(move |api| api.get(&format!("/api/chats/{id}")), Message::ChatLoaded));
                        }
                    }
                    if self.busy.is_none() {
                        self.status = String::new();
                    }
                } else if now.is_some() && self.busy.is_none() {
                    self.status = tr("A scheduled task is running").into();
                }
                return Task::batch(tasks);
            }
            Message::RunScheduled(id) => {
                self.status = tr("It runs as soon as nothing else is").into();
                let body = obj(vec![("id", str(&id))]);
                return self.call(move |api| api.post("/api/schedules/run", &body), Message::Background);
            }
            Message::CancelScheduled(id) => {
                let body = obj(vec![("id", str(&id))]);
                return self.call(move |api| api.post("/api/schedules/cancel", &body), Message::Background);
            }
            Message::Read(Ok(c)) => self.set_chats(&c),
            Message::Read(Err(_)) => {}
            Message::Menu(m) => self.menu = m,
            Message::SetTheme(mode) => {
                self.mode = mode;
                self.set_dark(match mode {
                    ThemeMode::Dark => true,
                    ThemeMode::Light => false,
                    ThemeMode::System => windows_dark(),
                });
                self.save_settings();
            }
            Message::SetLang(lang) => {
                self.lang = lang;
                crate::i18n::set(lang);
                self.save_settings();
            }
            Message::SettingsPage(page) => {
                self.settings_page = page;
                if page == SettingsPage::Maya {
                    return self.call(|api| api.get("/api/maya"), Message::MayaMemory);
                }
            }
            Message::MayaMemory(Ok(j)) => {
                self.maya_name = j.get("name").and_then(Json::as_str).unwrap_or("").to_string();
                self.maya_memory = j;
            }
            Message::MayaMemory(Err(e)) => self.status = format!("Couldn't read Maya's memory: {e}"),
            Message::MayaForget(list, i) => {
                let body = obj(vec![("forget", str(list)), ("index", Json::Num(i as f64))]);
                return self.call(move |api| api.post("/api/maya/edit", &body), Message::MayaMemory);
            }
            Message::MayaName(n) => self.maya_name = n,
            Message::SaveMayaName => {
                let body = obj(vec![("name", str(self.maya_name.trim()))]);
                return self.call(move |api| api.post("/api/maya/edit", &body), Message::MayaMemory);
            }
            Message::ClearImpression => {
                let body = obj(vec![("impression", str(""))]);
                return self.call(move |api| api.post("/api/maya/edit", &body), Message::MayaMemory);
            }
            Message::ResetMaya => return self.call(|api| api.post("/api/maya/reset", &Json::Null), Message::MayaMemory),
            Message::ToggleCallTimings => {
                self.call_timings = !self.call_timings;
                self.save_settings();
            }
            Message::ToggleCallTyping => self.call_typing = !self.call_typing,
            Message::ToggleTool(i, j) => {
                if let Some(t) = self.messages.get_mut(i).and_then(|m| m.tools.get_mut(j)) {
                    t.open = !t.open;
                }
            }
            // the menu stays open, so several groups can be switched at once
            Message::SetMode(i) => {
                self.perm = i.min(MODES.len() - 1);
                self.menu = None;
                self.save_settings();
            }
            Message::ToggleCard(key) => {
                match self.rail_open.iter().position(|k| k == key) {
                    Some(i) => {
                        self.rail_open.remove(i);
                    }
                    None => self.rail_open.push(key.to_string()),
                }
                self.save_settings();
            }
            Message::ShowTasks => {
                self.tasks_panel = true;
                return self.fetch_outputs();
            }
            Message::CloseTasks => self.tasks_panel = false,
            Message::WideTasks => self.tasks_wide = !self.tasks_wide,
            Message::ToggleTask(key) => match self.task_open.iter().position(|k| *k == key) {
                Some(i) => {
                    self.task_open.remove(i);
                }
                None => self.task_open.push(key),
            },
            Message::ToggleFinished => self.finished_open = !self.finished_open,
            Message::ClearFinished(keys) => self.tasks_hidden.extend(keys),
            Message::MoreTasks => self.tasks_shown += 12,
            Message::ProcOutput(id, Ok(j)) => {
                if let Some(out) = j.get("output").and_then(Json::as_str) {
                    self.proc_output.insert(id, out.to_string());
                }
            }
            Message::ProcOutput(_, Err(_)) => {}
            Message::StopProcess(id) => {
                let body = obj(vec![("name", str("stop_process")), ("arguments", obj(vec![("id", num(id as f64))]))]);
                return self.call(
                    move |api| {
                        api.post("/api/tools/run", &body)?;
                        api.get("/api/background")
                    },
                    Message::Background,
                );
            }
            Message::TrimChat => {
                let Some(id) = self.chat.clone() else { return Task::none() };
                let body = obj(vec![("id", str(&id))]);
                return self.call(move |api| api.post("/api/chats/trim", &body), Message::Trimmed);
            }
            Message::Trimmed(Ok(j)) => {
                self.status = match j.get("trimmed").and_then(Json::as_f64).unwrap_or(0.0) as usize {
                    0 => tr("Nothing to trim: only the last 3 tool outputs are left").into(),
                    n => if crate::i18n::is_zh() { format!("裁剪了 {n} 条旧工具输出；下一条回复会重新读一遍对话") } else { format!("Trimmed {n} old tool output{}; the next reply reads the chat again", if n == 1 { "" } else { "s" }) },
                };
            }
            Message::Trimmed(Err(e)) => self.status = format!("Couldn't trim: {e}"),
            Message::ClearGoal => {
                self.composer = text_editor::Content::with_text("/goal clear");
                return self.send(None);
            }
            Message::CycleMode => {
                self.perm = (self.perm + 1) % MODES.len();
                self.save_settings();
            }
            Message::PickWorkdir => {
                self.menu = None;
                return Task::perform(blocking(|| rfd::FileDialog::new().set_title(tr("Choose the folder this chat works in")).pick_folder().map(|p| p.display().to_string())), Message::SetWorkdir);
            }
            Message::SetWorkdir(Some(f)) => {
                self.workdir = Some(f);
                self.menu = None;
                self.save_settings();
            }
            Message::SetWorkdir(None) => {}
            Message::Approve(id, decision) => {
                if let Some(t) = self.messages.last_mut().and_then(|m| m.tools.iter_mut().find(|t| t.id == id)) {
                    t.ask = None;
                }
                self.status = if decision == "deny" { "Reading the results".into() } else { "Using a tool".into() };
                let body = obj(vec![("id", str(&id)), ("decision", str(decision))]);
                return self.call(move |api| api.post("/api/agent/approve", &body), Message::Approved);
            }
            Message::Approved(Ok(_)) => {}
            Message::Approved(Err(e)) => self.status = format!("Couldn't send your answer: {e}"),
            Message::Undo(i) => {
                if let Some(cp) = self.messages.get(i).and_then(|m| m.checkpoint.clone()) {
                    let body = obj(vec![("checkpoint", str(&cp))]);
                    return self.call(move |api| api.post("/api/tools/undo", &body), move |r| Message::Undone(i, r));
                }
            }
            Message::Undone(i, r) => {
                let note = match r {
                    Ok(j) => match j.get("undone").and_then(Json::as_array).map_or(0, |a| a.len()) {
                        0 => tr("Nothing to undo").to_string(),
                        1 => tr("Undone: 1 file put back").to_string(),
                        n => tf!("Undone: {n} files put back", "已撤销：放回 {n} 个文件"),
                    },
                    Err(e) => format!("Couldn't undo: {e}"),
                };
                if let Some(m) = self.messages.get_mut(i) {
                    m.undone = true;
                    m.stats = note;
                }
            }
            Message::ToggleSearch => {
                self.searching = !self.searching;
                if !self.searching {
                    self.filter.clear();
                }
            }
            Message::Filter(f) => self.filter = f,
            Message::ToggleRecents => self.recents_open = !self.recents_open,
            Message::CopyChat => {
                self.menu = None;
                let md: Vec<String> = self.messages.iter().map(|m| if m.role == Role::User { tf!("**You:** {}", "**你：** {}", m.raw) } else { m.answer_text() }).collect();
                return iced::clipboard::write(md.join("\n\n"));
            }
            Message::DeleteCurrent => {
                self.menu = None;
                if let Some(id) = self.chat.clone() {
                    return self.update(Message::DeleteChat(id));
                }
            }
            Message::DragWindow => return iced::window::latest().and_then(iced::window::drag),
            Message::ToggleMaximize => return iced::window::latest().and_then(iced::window::toggle_maximize),
            Message::Minimize => return iced::window::latest().and_then(|id| iced::window::minimize(id, true)),
            // the X (or Alt+F4) hides the window to the tray: the model and the chats go on, and the tray icon
            // brings it back or quits
            Message::CloseWindow => return iced::window::latest().and_then(|id| iced::window::set_mode(id, iced::window::Mode::Hidden)),
            Message::Tray(tray::Ask::Show) => {
                return iced::window::latest().and_then(|id| iced::window::set_mode::<Message>(id, iced::window::Mode::Windowed).chain(iced::window::gain_focus(id)));
            }
            Message::Tray(tray::Ask::Quit) => return Task::done(Message::Quit),
            Message::ResizeWindow(d) => return iced::window::latest().and_then(move |id| iced::window::drag_resize(id, d)),
            // maximized by the button, a double click, snapping or Win+Up: the size changes every time
            Message::WindowResized(size) => {
                self.window = size;
                return iced::window::latest().and_then(iced::window::is_maximized).map(Message::Maximized);
            }
            // the + menu: find the button on screen first, then open the menu above it
            Message::OpenMenu(menu) => match menu.anchor() {
                Some(id) => return iced::widget::selector::find(iced::widget::selector::id(id)).map(move |t| Message::MenuAt(menu, t.map(|t| t.bounds()))),
                None => self.menu = Some(menu),
            },
            Message::MenuAt(menu, at) => {
                self.menu_at = at;
                self.menu = Some(menu);
            }
            Message::Maximized(m) => self.maximized = m,
            Message::ToggleWarmUp => {
                self.warm_up = !self.warm_up;
                let body = obj(vec![("folders", Json::Arr(self.folders.iter().map(|f| str(f)).collect())), ("warm_up", Json::Bool(self.warm_up))]);
                return self.call(move |api| api.post("/api/harness", &body), Message::Harness);
            }
            Message::ToggleLoadLast => {
                self.load_last = !self.load_last;
                // on with nothing picked yet: the model running now is the one to start with
                if self.last_model.is_none() {
                    self.last_model = self.model.as_ref().map(|m| m.id.clone());
                }
                self.save_settings();
            }
            // the server runs without a window, so the app stops it: the model leaves the GPU, then the app closes
            Message::Quit => {
                let Some(api) = self.api.clone() else { return iced::exit() };
                return Task::perform(blocking(move || api.post("/api/quit", &Json::Null)), |_| ()).discard().chain(iced::exit());
            }
            Message::TogglePlan => self.plan_open = Some(!self.plan_shown()),
            Message::ToggleRail => {
                self.rail = !self.rail;
                self.save_settings();
                if self.rail {
                    return self.refresh_system();
                }
            }
            Message::System(Ok(j)) => self.system = j,
            Message::System(Err(_)) => {}
            Message::GetModel(i) => {
                let Some(o) = setup::OFFERS.get(i) else { return Task::none() };
                self.download_rate = (0.0, None, 0.0);
                let body = obj(vec![("url", str(o.url)), ("file", str(o.file)), ("size", num(o.size as f64))]);
                return self.call(move |api| api.post("/api/downloads", &body), Message::DownloadStatus);
            }
            Message::CancelDownload => return self.call(|api| api.post("/api/downloads/cancel", &Json::Null), Message::DownloadStatus),
            Message::DownloadStatus(r) => {
                self.download_polling = false;
                self.download_polled = self.clock;
                let j = match r {
                    Ok(j) => j,
                    Err(e) => {
                        self.status = e;
                        return Task::none();
                    }
                };
                let state = |j: &Json| j.get("state").and_then(Json::as_str).unwrap_or("").to_string();
                let was = state(&self.download);
                // the speed, smoothed over the looks
                let bytes = j.get("bytes").and_then(Json::as_f64).unwrap_or(0.0);
                let now = Instant::now();
                if let (Some(at), true) = (self.download_rate.1, bytes >= self.download_rate.0) {
                    let secs = now.duration_since(at).as_secs_f64();
                    if secs > 0.2 {
                        let rate = (bytes - self.download_rate.0) / secs;
                        self.download_rate.2 = if self.download_rate.2 == 0.0 { rate } else { 0.8 * self.download_rate.2 + 0.2 * rate };
                    }
                }
                self.download_rate.0 = bytes;
                self.download_rate.1 = Some(now);
                self.download = j;
                // done: the model loads, and the app starts with it from now on
                if state(&self.download) == "done" && was == "downloading" {
                    let file = self.download.get("file").and_then(Json::as_str).unwrap_or("").to_string();
                    let id = file.trim_end_matches(".gguf").to_string();
                    self.last_model = Some(id.clone());
                    self.load_last = true;
                    self.save_settings();
                    self.status = format!("Loading {}…", short_model(&id));
                    let body = obj(vec![("id", str(&id))]);
                    return self.call(
                        move |api| {
                            api.post("/api/models/load", &body)?;
                            api.get("/api/models")
                        },
                        Message::ModelLoaded,
                    );
                }
            }
            Message::PreviewMade(Ok(())) => {}
            Message::PreviewMade(Err(e)) => {
                self.preview_open = false;
                self.status = format!("Couldn't show the page here: {e}");
            }
            Message::PreviewOpen => {
                if let Some(url) = self.preview_url.clone() {
                    return self.show_preview(url);
                }
            }
            Message::PreviewClose => {
                self.preview_open = false;
                preview::set_visible(false);
            }
            Message::PreviewReload => {
                if let Some(url) = &self.preview_url {
                    preview::navigate(url);
                }
            }
            Message::PreviewExternal => {
                if let Some(url) = &self.preview_url {
                    open_link(url);
                }
            }
            Message::FocusApp => preview::focus_app(),
            // back in Code mode: the Code model with its whole memory, if a call took it
            Message::SetVoice(on) => {
                self.voice = on;
                if !on && self.call.is_none() && !self.call_pending {
                    return self.back_to_code();
                }
            }
            // the model calls talk to: loaded beside the Code model if they both fit (else at the next call's start)
            Message::PickCallModel(id) => {
                self.menu = None;
                self.call_choice = id;
                self.save_settings();
                if self.call.is_none() && self.loading.is_none() && self.code_model().is_some() {
                    return self.load_models(false);
                }
            }
            Message::ConfirmCall(yes) => {
                self.call_confirm = None;
                if yes {
                    return self.switch_for_call();
                }
            }
            Message::StartCall => {
                let Some(api) = self.api.clone() else { return Task::none() };
                if self.call.is_some() {
                    return Task::none();
                }
                if self.voice_models.0.is_none() || self.voice_models.1.is_none() {
                    self.call_state = tr("The server has no voice: start it with --tts and --stt.").into();
                    return Task::none();
                }
                if self.loading.is_some() && self.call_wait.is_none() {
                    // one is loading: the call starts once it has
                    self.call_pending = true;
                    self.call_state = tr("Waiting for the model to load…").into();
                    return self.wait_for_model();
                }
                let Some(target) = self.call_target() else {
                    self.call_state = tr("Pick a model first: the call talks to it.").into();
                    return Task::none();
                };
                // the call model and the voice not loaded yet: switch to them (asking first when that unloads the
                // Code model or shrinks its context); switched already and still not ready: the voice did not load
                if self.call_ready.as_deref() != Some(target.id.as_str()) {
                    if self.call_wait.as_deref() == Some(target.id.as_str()) {
                        self.call_wait = None;
                        self.call_state = tr("The voice didn't load: the server's log says why.").into();
                        return self.back_to_code();
                    }
                    match self.model.clone() {
                        Some(code) if code.id != target.id => {
                            let ask = tf!(
                                "Your graphics card doesn't have room for {} and {} with the voice at the same time.\n\nUnload {} and start the call? It loads again when you switch back to Code.",
                                "显卡装不下 {}、{} 和语音同时都在。\n\n卸载 {} 并开始通话？切回「代码」时会重新加载。",
                                code.label, target.label, code.label
                            );
                            self.call_confirm = Some((ask, tr("Unload and call")));
                            return Task::none();
                        }
                        Some(code) => {
                            let ask = tf!(
                                "Your graphics card doesn't have room for the voice beside {} at its full context.\n\nReload {} with a smaller context and start the call? It gets its full context back when you switch back to Code.",
                                "显卡装不下满上下文的 {} 再加语音。\n\n用更小的上下文重新加载 {} 并开始通话？切回「代码」时它会拿回完整的上下文。",
                                code.label, code.label
                            );
                            self.call_confirm = Some((ask, tr("Reload and call")));
                            return Task::none();
                        }
                        None => return self.switch_for_call(),
                    }
                }
                let (tx, rx) = mpsc::unbounded();
                match call::start(api, None, MODES[self.perm].0.to_string(), tx) {
                    Ok(c) => {
                        self.call = Some(c);
                        self.call_wait = None;
                        self.call_lines.clear();
                        self.call_mood = None;
                        self.orb.set_mood("");
                        self.call_chat = None;
                        self.call_timing.clear();
                        self.call_started = Some(std::time::Instant::now());
                        self.call_state = "Listening".into();
                        let (task, handle) = Task::run(rx, Message::Call).abortable();
                        self.call_task = Some(handle);
                        return task;
                    }
                    Err(e) => self.call_state = format!("Couldn't start the call: {e}"),
                }
            }
            // at once: the devices close, and what the call's thread still sends (it may be waiting on a reply) is
            // not heard any more
            Message::EndCall => {
                // still switching models for it: the call does not start
                let pending = std::mem::take(&mut self.call_pending);
                self.call_wait = None;
                if let Some(c) = self.call.take() {
                    c.end();
                    if let Some(h) = self.call_task.take() {
                        h.abort();
                    }
                } else if !pending {
                    return Task::none();
                }
                self.call_state = tr("Call ended").into();
                return Task::batch([self.call(|api| api.get("/api/chats"), Message::Chats), self.after_call()]);
            }
            Message::Reflected(r) => {
                // another call began meanwhile: it keeps the model
                if self.call.is_some() || self.call_pending {
                    return Task::none();
                }
                self.call_state = match r {
                    Ok(_) => tr("Call ended").into(),
                    Err(e) => tf!("Call ended (Maya couldn't think it over: {e})", "通话已结束（Maya 没能回想起来：{e}）"),
                };
                return self.back_to_code();
            }
            Message::ToggleMute => {
                if let Some(c) = &self.call {
                    c.set_muted(!c.muted());
                }
            }
            Message::CallText(t) => {
                // something being typed: Maya does not speak up meanwhile
                if let Some(c) = &self.call {
                    c.set_typing(!t.trim().is_empty());
                }
                self.call_text = t;
            }
            Message::SendCallText => {
                let text = self.call_text.trim().to_string();
                if let (Some(c), false) = (&self.call, text.is_empty()) {
                    c.send_text(text);
                    c.set_typing(false);
                    self.call_text.clear();
                }
            }
            Message::Call(ev) => match ev {
                call::CallEvent::State(s) => self.call_state = s.into(),
                call::CallEvent::You(t) => {
                    self.call_lines.push((true, t));
                    self.call_lines.push((false, String::new()));
                }
                call::CallEvent::Her => self.call_lines.push((false, String::new())),
                call::CallEvent::Mood(mood, talk) => {
                    self.orb.set_mood(&mood);
                    self.call_mood = Some((mood, talk));
                }
                call::CallEvent::Ghost(t) => {
                    if let Some(l) = self.call_lines.last_mut().filter(|l| !l.0) {
                        l.1 = t;
                    }
                }
                call::CallEvent::Chat(id) => {
                    let new = self.call_chat.is_none();
                    self.call_chat = Some(id);
                    if new {
                        return self.call(|api| api.get("/api/chats"), Message::Chats);
                    }
                }
                call::CallEvent::Timing(t) => self.call_timing = t,
                call::CallEvent::Error(e) => self.call_state = e,
                call::CallEvent::Ended => {
                    self.call = None;
                    self.call_task = None;
                    self.call_state = if self.call_lines.is_empty() { String::new() } else { tr("Call ended").into() };
                    return Task::batch([self.call(|api| api.get("/api/chats"), Message::Chats), self.after_call()]);
                }
            },
            Message::UseModelFolder => {
                return Task::perform(blocking(|| rfd::FileDialog::new().set_title(tr("Choose a folder with GGUF models")).pick_folder().map(|p| p.display().to_string())), Message::ModelFolder);
            }
            Message::ModelFolder(None) => {}
            // a folder of models: listed at once, kept for the next start; added from the first-run screen with no
            // model running, the model menu opens (from the settings, they stay open)
            Message::ModelFolder(Some(f)) => {
                if !self.model_dirs.contains(&f) {
                    self.model_dirs.push(f.clone());
                }
                self.save_settings();
                self.pick_model = self.model.is_none() && self.loading.is_none() && self.menu != Some(Menu::Settings);
                let body = obj(vec![("add", str(&f))]);
                return self.call(move |api| api.post("/api/models/folders", &body), Message::ModelFolders);
            }
            Message::RemoveModelFolder(f) => {
                self.model_dirs.retain(|d| *d != f);
                self.save_settings();
                let body = obj(vec![("remove", str(&f))]);
                return self.call(move |api| api.post("/api/models/folders", &body), Message::ModelFolders);
            }
            Message::ModelFolders(Ok(j)) => {
                self.model_folders = j.get("folders").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect();
                let menu = if std::mem::take(&mut self.pick_model) { Task::done(Message::OpenMenu(Menu::Model)) } else { Task::none() };
                return Task::batch([self.call(|api| api.get("/api/models"), Message::ModelLoaded), menu]);
            }
            Message::ModelFolders(Err(e)) => self.status = e,
            Message::AddFolder => {
                self.menu = None;
                // the dialog blocks, so it opens on its own thread
                return Task::perform(blocking(|| rfd::FileDialog::new().set_title(tr("Choose a folder the model may use")).pick_folder().map(|p| p.display().to_string())), Message::FolderPicked);
            }
            Message::FolderPicked(None) => {}
            Message::FolderPicked(Some(f)) => {
                let mut folders = self.folders.clone();
                if !folders.contains(&f) {
                    folders.push(f);
                }
                return self.save_folders(folders);
            }
            Message::RemoveFolder(i) => {
                let mut folders = self.folders.clone();
                if i < folders.len() {
                    folders.remove(i);
                }
                return self.save_folders(folders);
            }
            Message::Harness(Ok(j)) => {
                self.set_harness(&j);
                return self.refresh_instructions(); // a new folder may bring its own
            }
            Message::Harness(Err(e)) => self.status = format!("Couldn't save the settings: {e}"),
            Message::Memory(Ok(j)) => self.memory = j.get("facts").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect(),
            Message::Memory(Err(e)) => self.status = format!("Couldn't read the memory: {e}"),
            Message::MemoryDraft(d) => self.memory_draft = d,
            Message::AddMemory => {
                let fact = std::mem::take(&mut self.memory_draft);
                if !fact.trim().is_empty() {
                    let body = obj(vec![("fact", str(fact.trim()))]);
                    return self.call(move |api| api.post("/api/memory/add", &body), Message::Memory);
                }
            }
            Message::InstallSkill(archive) => {
                self.menu = None;
                return Task::perform(
                    blocking(move || {
                        let d = rfd::FileDialog::new();
                        let picked = if archive { d.set_title(tr("Choose a skill archive")).add_filter("Skill", &["zip", "skill"]).pick_file() } else { d.set_title(tr("Choose a skill's folder (it has a SKILL.md)")).pick_folder() };
                        picked.map(|p| p.display().to_string())
                    }),
                    Message::SkillPicked,
                );
            }
            Message::SkillPicked(None) => {}
            Message::SkillLink(l) => self.skill_link = l,
            Message::InstallLink => {
                let link = std::mem::take(&mut self.skill_link);
                if !link.trim().is_empty() {
                    self.status = tr("Downloading the skill…").into();
                    let body = obj(vec![("path", str(link.trim()))]);
                    return self.call(move |api| api.post("/api/skills/install", &body), Message::Skills);
                }
            }
            Message::OpenSkillLink => {
                self.menu = None;
                // the field sits low in the rail: scrolled to, then focused
                let focus = iced::widget::operation::snap_to_end("rail").chain(iced::widget::operation::focus("skill-link"));
                if !self.rail {
                    self.rail = true;
                    self.save_settings();
                    return Task::batch([self.refresh_system(), focus]);
                }
                return focus;
            }
            Message::SkillPicked(Some(path)) => {
                let body = obj(vec![("path", str(&path))]);
                return self.call(move |api| api.post("/api/skills/install", &body), Message::Skills);
            }
            Message::InstallChoice(path) => {
                if let Some((source, _)) = &self.skill_choices {
                    self.status = tr("Installing…").into();
                    let body = obj(vec![("path", str(source)), ("skills", Json::Arr(vec![str(path.as_deref().unwrap_or("all"))]))]);
                    return self.call(move |api| api.post("/api/skills/install", &body), Message::Skills);
                }
            }
            Message::CancelChoice => self.skill_choices = None,
            Message::Skills(Ok(j)) => {
                self.skills = j.get("skills").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(|s| Some((s.get("name")?.as_str()?.to_string(), s.get("description").and_then(Json::as_str).unwrap_or("").to_string()))).collect();
                if let Some(name) = j.get("installed").and_then(Json::as_str) {
                    self.skill_choices = None;
                    self.status = if name.contains(", ") { tf!("Installed the skills {name}", "已安装技能 {name}") } else { tf!("Installed the {name} skill", "已安装技能 {name}") };
                }
                if let Some(list) = j.get("choose").and_then(Json::as_array) {
                    let s = |c: &Json, k: &str| c.get(k).and_then(Json::as_str).unwrap_or("").to_string();
                    let choices: Vec<_> = list.iter().map(|c| (s(c, "path"), s(c, "name"), s(c, "description"))).collect();
                    self.status = tf!("That link has {} skills: pick one in the rail", "这个链接里有 {} 个技能：在右侧栏挑一个", choices.len());
                    self.skill_choices = Some((s(&j, "source"), choices));
                }
            }
            Message::Skills(Err(e)) => self.status = format!("Couldn't install the skill: {e}"),
            Message::RemoveSkill(name) => {
                let body = obj(vec![("name", str(&name))]);
                return self.call(move |api| api.post("/api/skills/remove", &body), Message::Skills);
            }
            Message::Mcp(Ok(j)) => {
                self.mcp = j.get("servers").and_then(Json::as_array).unwrap_or(&[]).iter().filter(|s| s.get("name").and_then(Json::as_str).is_some_and(|n| !n.is_empty())).filter_map(|s| {
                    Some((s.get("name")?.as_str()?.to_string(), s.get("status").and_then(Json::as_str).unwrap_or("").to_string(), s.get("tools").and_then(Json::as_array).map_or(0, |t| t.len())))
                }).collect();
                if let Some(c) = j.get("config").and_then(Json::as_str) {
                    self.mcp_config = c.to_string();
                }
                // servers still starting: look again in a moment
                if self.mcp.iter().any(|m| m.1 == "starting") {
                    return self.call(|api| { std::thread::sleep(std::time::Duration::from_secs(3)); api.get("/api/mcp") }, Message::Mcp);
                }
            }
            Message::Mcp(Err(_)) => {}
            Message::McpReload => {
                return self.call(|api| { api.post("/api/mcp/reload", &Json::Null)?; std::thread::sleep(std::time::Duration::from_millis(500)); api.get("/api/mcp") }, Message::Mcp);
            }
            Message::McpImport => {
                return self.call(|api| { api.post("/api/mcp/import-claude", &Json::Null)?; std::thread::sleep(std::time::Duration::from_millis(500)); api.get("/api/mcp") }, Message::Mcp);
            }
            // mcp.json in Notepad; Reload picks the changes up
            Message::Instructions(Ok(j)) => {
                self.instructions = j.get("files").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(|f| {
                    Some((f.get("path")?.as_str()?.to_string(), f.get("exists").and_then(Json::as_bool).unwrap_or(false), f.get("scope").and_then(Json::as_str) == Some("global")))
                }).collect();
            }
            Message::Instructions(Err(_)) => {}
            // opened in Notepad, made first when missing (the model reads it at its next reply)
            Message::EditInstructions(path) => {
                let file = std::path::Path::new(&path);
                if !file.exists() {
                    let note = "<!-- ghost reads this file into its chats, like a CLAUDE.md: write your instructions for the model below. Comments like this one are left out. -->\n\n";
                    if let Err(e) = std::fs::write(file, note) {
                        self.status = tf!("Couldn't make {path}: {e}", "无法创建 {path}：{e}");
                        return Task::none();
                    }
                }
                let _ = std::process::Command::new("notepad").arg(file).spawn();
                return self.refresh_instructions();
            }
            Message::McpEdit => {
                if !self.mcp_config.is_empty() {
                    let _ = std::process::Command::new("notepad").arg(&self.mcp_config).spawn();
                }
            }
            Message::Forget(i) => {
                let body = obj(vec![("index", num(i as f64))]);
                return self.call(move |api| api.post("/api/memory/forget", &body), Message::Memory);
            }
            // every reply's file changes put back, the newest first, one after another (they share a journal)
            Message::UndoAll => {
                let mut task = Task::none();
                for i in (0..self.messages.len()).rev() {
                    let m = &self.messages[i];
                    if let (true, false, Some(cp)) = (m.changed, m.undone, m.checkpoint.clone()) {
                        let body = obj(vec![("checkpoint", str(&cp))]);
                        task = task.chain(self.call(move |api| api.post("/api/tools/undo", &body), move |r| Message::Undone(i, r)));
                    }
                }
                return task;
            }
            Message::Hover(h) => self.hovered = h,
            Message::Copy(i) => {
                if let Some(m) = self.messages.get(i) {
                    return iced::clipboard::write(m.answer_text());
                }
            }
            Message::Link(url) => open_link(&url),
        }
        Task::none()
    }

    /// A server call on a worker thread, its result as a message.
    fn call<T: Send + 'static>(&self, f: impl FnOnce(&Api) -> Result<T, String> + Send + 'static, done: impl FnOnce(Result<T, String>) -> Message + Send + 'static) -> Task<Message> {
        let Some(api) = self.api.clone() else { return Task::none() };
        Task::perform(blocking(move || f(&api)), done)
    }

    /// This computer's GPU, memory and model, for the rail (only fetched while it is shown).
    fn save_settings(&self) {
        let Some(p) = app_file() else { return };
        let v = match self.mode {
            ThemeMode::System => "system",
            ThemeMode::Dark => "dark",
            ThemeMode::Light => "light",
        };
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let mut fields = vec![
            ("theme", str(v)),
            ("permission_mode", str(MODES[self.perm].0)),
            ("rail", Json::Bool(self.rail)),
            ("rail_open", Json::Arr(self.rail_open.iter().map(|k| str(k)).collect())),
            ("load_last", Json::Bool(self.load_last)),
            ("call_timings", Json::Bool(self.call_timings)),
            ("lang", str(match self.lang { crate::i18n::Lang::En => "en", crate::i18n::Lang::Zh => "zh", crate::i18n::Lang::System => "system" })),
        ];
        if let Some(m) = &self.last_model {
            fields.push(("last_model", str(m)));
        }
        if let Some(m) = &self.call_choice {
            fields.push(("call_model", str(m)));
        }
        if let Some(w) = &self.workdir {
            fields.push(("workdir", str(w)));
        }
        if !self.model_dirs.is_empty() {
            fields.push(("model_dirs", Json::Arr(self.model_dirs.iter().map(|d| str(d)).collect())));
        }
        let _ = std::fs::write(p, obj(fields).to_string());
    }

    /// The server's harness settings: the folders file tools may use.
    fn set_harness(&mut self, j: &Json) {
        self.folders = j.get("folders").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect();
        self.warm_up = j.get("warm_up").and_then(Json::as_bool).unwrap_or(true);
    }

    /// Sends the new folder list to the server (it checks each one is a folder).
    fn save_folders(&self, folders: Vec<String>) -> Task<Message> {
        let body = obj(vec![("folders", Json::Arr(folders.iter().map(|f| str(f)).collect()))]);
        self.call(move |api| api.post("/api/harness", &body), Message::Harness)
    }

    /// The status line while no model runs: the one loading, or a hint to pick one (the server may have
    /// started without a model).
    fn no_model_status(&self) -> String {
        match &self.loading {
            Some(id) => format!("Loading {}…", short_model(id)),
            None => "No model loaded".into(),
        }
    }

    /// While the server is still loading its model, or warming it up, asks again every two seconds until it
    /// is done.
    fn wait_for_model(&self) -> Task<Message> {
        let loading = self.model.is_none() && self.loading.is_some();
        if !loading && !self.warming {
            return Task::none();
        }
        self.call(
            |api| {
                std::thread::sleep(std::time::Duration::from_secs(2));
                api.get("/api/models")
            },
            Message::ModelLoaded,
        )
    }

    /// The installed skills, the MCP servers and the GHOST.md files (the rail lists them).
    fn refresh_extras(&self) -> Task<Message> {
        Task::batch([self.call(|api| api.get("/api/skills"), Message::Skills), self.call(|api| api.get("/api/mcp"), Message::Mcp), self.refresh_instructions()])
    }

    /// Whether the plan above the message box shows its steps: as set by hand, else until every step is done.
    pub fn plan_shown(&self) -> bool {
        self.plan_open.unwrap_or_else(|| self.tasks.iter().any(|t| t.1 != "done"))
    }

    fn refresh_instructions(&self) -> Task<Message> {
        self.call(|api| api.get("/api/instructions"), Message::Instructions)
    }

    /// What the model remembers (the rail shows it, with a remove button for each).
    fn refresh_memory(&self) -> Task<Message> {
        self.call(|api| api.get("/api/memory"), Message::Memory)
    }

    fn refresh_system(&self) -> Task<Message> {
        if !self.rail {
            return Task::none();
        }
        self.call(|api| api.get("/api/system"), Message::System)
    }

    /// The files this chat changed and not undone, with lines added and removed, for the rail.
    pub fn changes(&self) -> Vec<(String, usize, usize)> {
        let mut out: Vec<(String, usize, usize)> = Vec::new();
        for t in self.messages.iter().filter(|m| !m.undone).flat_map(|m| &m.tools).filter(|t| t.changed_files()) {
            let a = Json::parse(&t.arguments).unwrap_or(Json::Null);
            let s = |k: &str| a.get(k).and_then(Json::as_str).unwrap_or("").to_string();
            let (add, del) = if t.name == "write_file" { (s("content").lines().count(), 0) } else { (s("new_string").lines().count(), s("old_string").lines().count()) };
            let path = s("path");
            match out.iter_mut().find(|c| c.0 == path) {
                Some(c) => {
                    c.1 += add;
                    c.2 += del;
                }
                None => out.push((path, add, del)),
            }
        }
        out
    }

    /// The model a call runs on when the current one is too big (CALL_MAX): the largest that fits, preferring
    /// one of at least 1 GB (a tiny one answers too poorly to talk to).
    fn call_model(&self) -> Option<ModelChoice> {
        let fits = |m: &&ModelChoice| m.size > 0 && m.size <= CALL_MAX;
        let best = |min: u64| self.models.iter().filter(fits).filter(|m| m.size >= min).max_by_key(|m| m.size).cloned();
        best(1_000_000_000).or_else(|| best(0))
    }

    /// The Code model: the one last picked for chats, else the one running.
    fn code_model(&self) -> Option<ModelChoice> {
        let id = self.last_model.clone().or_else(|| self.model.as_ref().map(|m| m.id.clone()))?;
        self.models.iter().find(|m| m.id == id).cloned().or_else(|| self.model.clone().filter(|m| m.id == id))
    }

    /// The model calls talk to: the one picked for calls, else the Code model, unless that is too big to answer
    /// quickly beside the voice (then the largest that fits).
    fn call_target(&self) -> Option<ModelChoice> {
        if let Some(m) = self.call_choice.as_ref().and_then(|id| self.models.iter().find(|m| m.id == *id)) {
            return Some(m.clone());
        }
        let code = self.code_model()?;
        if code.size > CALL_MAX { self.call_model() } else { Some(code) }
    }

    /// Asks the server for the Code model and the call model: both kept loaded with the voice when they fit,
    /// else the call model with the voice (`for_call`) or the Code model alone.
    fn load_models(&mut self, for_call: bool) -> Task<Message> {
        let (code, target) = (self.code_model(), self.call_target());
        let Some(main) = code.clone().or_else(|| target.clone()) else { return Task::none() };
        self.asked_pair = true;
        let has_voice = self.voice_models.0.is_some() && self.voice_models.1.is_some();
        self.loading = Some(if for_call { target.as_ref().unwrap_or(&main).id.clone() } else { main.id.clone() });
        self.model = None;
        let mut body = vec![("id", str(&main.id))];
        match target.filter(|_| has_voice) {
            Some(t) => {
                body.push(("call", str(&t.id)));
                body.push(("for_call", Json::Bool(for_call)));
            }
            None if for_call => body.push(("voice", Json::Bool(true))),
            None => {}
        }
        let body = obj(body);
        self.call(
            move |api| {
                api.post("/api/models/load", &body)?;
                api.get("/api/models")
            },
            Message::ModelLoaded,
        )
    }

    /// At a call's end: Maya thinks it over (with the call's model, while it still runs), then the model from
    /// before the call comes back.
    fn after_call(&mut self) -> Task<Message> {
        let (Some(started), Some(chat)) = (self.call_started.take(), self.call_chat.clone()) else { return self.back_to_code() };
        self.call_state = tr("Maya is thinking back over the call…").into();
        let body = obj(vec![("chat", str(&chat)), ("seconds", Json::Num(started.elapsed().as_secs_f64()))]);
        self.call(move |api| api.post("/api/maya/reflect", &body), Message::Reflected)
    }

    /// Loads the call model with the voice (beside the Code model when they fit), then starts the call.
    fn switch_for_call(&mut self) -> Task<Message> {
        let Some(target) = self.call_target() else { return Task::none() };
        self.call_wait = Some(target.id.clone());
        self.call_pending = true;
        self.call_state = if self.model.as_ref().is_some_and(|m| m.id == target.id) { tr("Loading the voice…").into() } else { format!("Switching to {} for the call…", target.label) };
        self.load_models(true)
    }

    /// Back in Code mode (or a call that could not start, there): the Code model with its whole memory again, when
    /// a call took it (the call model swapped in, or the voice squeezed in beside it).
    fn back_to_code(&mut self) -> Task<Message> {
        let Some(code) = self.code_model() else { return Task::none() };
        if self.voice || self.loading.is_some() || (self.model.as_ref().is_some_and(|m| m.id == code.id) && (!self.voice_on || self.together)) {
            return Task::none();
        }
        self.status = format!("Loading {} again…", code.label);
        self.load_models(false)
    }

    fn set_models(&mut self, j: &Json) {
        let listed = j.get("models").and_then(Json::as_array).unwrap_or(&[]);
        let size = |id: &str| listed.iter().find(|m| m.get("id").and_then(Json::as_str) == Some(id)).and_then(|m| m.get("size")).and_then(Json::as_f64).unwrap_or(0.0) as u64;
        let choice = |id: &str| ModelChoice { id: id.to_string(), label: short_model(id), size: size(id) };
        self.models = listed.iter().filter_map(|m| m.get("id").and_then(Json::as_str).map(choice)).collect();
        self.model = j.get("current").and_then(Json::as_str).map(choice);
        self.loading = j.get("loading").and_then(Json::as_str).map(String::from);
        self.warming = j.get("warming").and_then(Json::as_bool).unwrap_or(false);
        self.context.1 = j.get("context_length").and_then(Json::as_f64).unwrap_or(0.0);
        let name = |k: &str| j.get(k).and_then(Json::as_str).map(String::from);
        self.voice_models = (name("speech"), name("listen"));
        self.voice_on = j.get("voice").and_then(Json::as_bool).unwrap_or(false);
        self.call_ready = name("call_model");
        self.together = j.get("together").and_then(Json::as_bool).unwrap_or(false);
    }

    fn set_chats(&mut self, j: &Json) {
        self.chats = j
            .get("chats")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .filter_map(|c| {
                Some(ChatItem {
                    id: c.get("id")?.as_str()?.to_string(),
                    title: c.get("title").and_then(Json::as_str).unwrap_or("").to_string(),
                    updated: c.get("updated").and_then(Json::as_f64).unwrap_or(0.0),
                    unread: c.get("unread").and_then(Json::as_bool).unwrap_or(false),
                })
            })
            .collect();
    }

    pub fn skin(&self) -> &'static view::Skin {
        if self.dark { &view::GRAPHITE } else { &view::FOG }
    }

    fn set_dark(&mut self, dark: bool) {
        self.dark = dark;
        self.theme = self.skin().theme();
    }

    pub fn idle_secs(&self) -> f64 {
        self.last_active.elapsed().as_secs_f64()
    }

    /// The open chat's commands, as the background tasks list shows them: running ones first, then the
    /// finished ones newest first (those cleared from the panel left out).
    pub fn bg_tasks(&self) -> Vec<BgTask> {
        let Some(chat) = self.chat.as_deref() else { return Vec::new() };
        let mut tasks = Vec::new();
        // in the order the chat ran them: a command, or a process it started (by the number its result gave,
        // the chat and the command, so a process from before the server restarted is not taken for a new one)
        for (i, m) in self.messages.iter().enumerate() {
            for (j, t) in m.tools.iter().enumerate().filter(|(_, t)| matches!(t.name.as_str(), "run_command" | "start_process") && !t.denied) {
                let a = Json::parse(&t.arguments).unwrap_or(Json::Null);
                let s = |k: &str| a.get(k).and_then(Json::as_str).unwrap_or("").trim().to_string();
                let title = if s("description").is_empty() { s("command") } else { s("description") };
                if t.name == "start_process" {
                    let Some(id) = t.result.as_deref().and_then(|r| r.strip_prefix("started as process ")).and_then(|r| r.split(';').next()).and_then(|n| n.trim().parse::<usize>().ok()) else { continue };
                    let p = self.processes.as_array().unwrap_or(&[]).iter().find(|p| {
                        p.get("id").and_then(Json::as_f64) == Some(id as f64) && p.get("chat").and_then(Json::as_str) == Some(chat) && p.get("command").and_then(Json::as_str) == Some(s("command").as_str())
                    });
                    let state = match p.map(|p| (p.get("running").and_then(Json::as_bool).unwrap_or(false), p.get("stopped").and_then(Json::as_bool).unwrap_or(false), p.get("exit_code").and_then(Json::as_f64))) {
                        Some((true, _, _)) => TaskState::Running,
                        Some((_, true, _)) => TaskState::Stopped,
                        Some((_, _, Some(c))) if c != 0.0 => TaskState::Failed,
                        _ => TaskState::Done,
                    };
                    let output = self.proc_output.get(&id).cloned().unwrap_or_default();
                    let secs = p.and_then(|p| p.get("secs")).and_then(Json::as_f64);
                    tasks.push(BgTask { key: format!("c{i}.{j}"), title, command: s("command"), output, secs, state, process: p.map(|_| id) });
                    continue;
                }
                let (state, output) = match &t.result {
                    None => (TaskState::Running, String::new()),
                    Some(r) => {
                        let (head, rest) = r.split_once('\n').unwrap_or((r.as_str(), ""));
                        let ok = head.trim() == "exit code 0";
                        (if ok { TaskState::Done } else { TaskState::Failed }, if head.starts_with("exit code") { rest.trim().to_string() } else { r.clone() })
                    }
                };
                tasks.push(BgTask { key: format!("c{i}.{j}"), title, command: s("command"), output, secs: t.secs, state, process: None });
            }
        }
        tasks.retain(|t| t.state == TaskState::Running || !self.tasks_hidden.contains(&t.key));
        let (mut running, mut done): (Vec<BgTask>, Vec<BgTask>) = tasks.into_iter().partition(|t| t.state == TaskState::Running);
        done.reverse();
        running.append(&mut done);
        running
    }

    /// The running processes' output, asked for while the panel shows them.
    fn fetch_outputs(&self) -> Task<Message> {
        if !self.tasks_panel {
            return Task::none();
        }
        let ids: Vec<usize> = self.bg_tasks().iter().filter(|t| t.state == TaskState::Running).filter_map(|t| t.process).collect();
        Task::batch(ids.into_iter().map(|id| {
            let body = obj(vec![("id", num(id as f64))]);
            self.call(move |api| api.post("/api/background/output", &body), move |r| Message::ProcOutput(id, r))
        }))
    }

    /// The folder a new chat will work in: the one picked, else the server's first allowed folder.
    pub fn new_chat_dir(&self) -> Option<&str> {
        self.workdir.as_deref().or(self.folders.first().map(String::as_str))
    }

    /// The folder under the message box: where the open chat works (it keeps the one it started in, as a
    /// Claude Code session does), or where a new one will.
    pub fn folder_label(&self) -> String {
        let dir = if self.messages.is_empty() { self.new_chat_dir() } else { self.chat_cwd.as_deref().or(self.folders.first().map(String::as_str)) };
        dir.map(folder_name).unwrap_or_else(|| tr("Choose a folder").into())
    }

    /// What moves the call orb: where the call is, and its level: the microphone while the user may speak, the
    /// voice while it plays.
    fn orb_input(&self) -> (orb::Phase, f32) {
        use orb::Phase;
        let Some(c) = &self.call else { return (Phase::Ready, 0.0) };
        // her voice moves it whether or not the user's microphone is muted
        if c.talking() {
            return (Phase::Speaking, c.shared.voice());
        }
        if c.muted() {
            return (Phase::Muted, 0.0);
        }
        match self.call_state.as_str() {
            "Thinking" => (Phase::Thinking, 0.0),
            "Hearing you" => (Phase::Hearing, c.shared.level()),
            _ => (Phase::Listening, c.shared.level()),
        }
    }

    /// The ghost's clock: a tick a few times a second (the sprite moves in steps, so no more is needed).
    fn subscription(&self) -> iced::Subscription<Message> {
        let resized = iced::window::resize_events().map(|(_, size)| Message::WindowResized(size));
        let tick = iced::Subscription::run(|| {
            use iced::futures::StreamExt;
            let (tx, rx) = mpsc::unbounded::<()>();
            std::thread::spawn(move || {
                while tx.unbounded_send(()).is_ok() {
                    std::thread::sleep(std::time::Duration::from_millis(110));
                }
            });
            rx.map(|_| Message::Tick)
        });
        // the mode menu's number keys, as Claude's (typed into the message box, the box's own binding takes them)
        let keys = if self.menu == Some(Menu::Mode) {
            iced::keyboard::listen().filter_map(|e| match e {
                iced::keyboard::Event::KeyPressed { key: iced::keyboard::Key::Character(c), .. } => mode_key(&c),
                _ => None,
            })
        } else if self.menu == Some(Menu::Settings) {
            // Esc closes the settings window
            iced::keyboard::listen().filter_map(|e| match e {
                iced::keyboard::Event::KeyPressed { key: iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape), .. } => Some(Message::Menu(None)),
                _ => None,
            })
        } else {
            iced::Subscription::none()
        };
        // a click in the app's own area takes the keyboard back from the preview page (a click in the page goes
        // to the page, not here)
        let clicks = if preview::exists() {
            iced::event::listen_with(|e, _, _| matches!(e, iced::Event::Mouse(iced::mouse::Event::ButtonPressed(_))).then_some(Message::FocusApp))
        } else {
            iced::Subscription::none()
        };
        // the tray icon, a second start of the app, and Alt+F4 (which hides the window as the X does)
        let tray = iced::Subscription::run(|| {
            use iced::futures::StreamExt;
            tray::events().map(Message::Tray)
        });
        let close = iced::window::close_requests().map(|_| Message::CloseWindow);
        // the Voice call view: the orb steps a dozen times a second (its mouth and level need more than the clock)
        let orb = if self.voice {
            iced::Subscription::run(|| {
                use iced::futures::StreamExt;
                let (tx, rx) = mpsc::unbounded::<()>();
                std::thread::spawn(move || {
                    while tx.unbounded_send(()).is_ok() {
                        std::thread::sleep(std::time::Duration::from_millis(83));
                    }
                });
                rx.map(|_| Message::Tick)
            })
        } else {
            iced::Subscription::none()
        };
        iced::Subscription::batch([tick, resized, keys, clicks, tray, close, orb])
    }

    /// Shows a web page in the preview pane, making the pane's view the first time (on the window's thread).
    fn show_preview(&mut self, url: String) -> Task<Message> {
        if url.is_empty() {
            return Task::none();
        }
        self.preview_url = Some(url.clone());
        self.preview_open = true;
        if preview::exists() {
            preview::set_visible(true);
            preview::navigate(&url);
            return Task::none();
        }
        iced::window::latest()
            .and_then(move |id| {
                let url = url.clone();
                iced::window::run(id, move |w| preview::create(w, &url))
            })
            .map(Message::PreviewMade)
    }

    /// A task list with every step done leaves the pin above the message box: it goes into the latest reply,
    /// as Claude Code does, and scrolls away with the chat.
    fn settle_plan(&mut self) {
        if self.tasks.is_empty() || self.tasks.iter().any(|t| t.1 != "done") {
            return;
        }
        if let Some(m) = self.messages.iter_mut().rev().find(|m| m.role == Role::Assistant) {
            m.plan = std::mem::take(&mut self.tasks);
            self.plan_open = None;
        }
    }

    /// A saved chat as messages: each user message, then its reply's rounds folded into one.
    fn show_chat(&mut self, chat: &Json) {
        self.chat = chat.get("id").and_then(Json::as_str).map(String::from);
        self.title = chat.get("title").and_then(Json::as_str).map(String::from);
        self.chat_cwd = chat.get("cwd").and_then(Json::as_str).filter(|c| !c.is_empty()).map(String::from);
        self.goal = goal_of(chat.get("goal"));
        self.parts = chat.get("parts").cloned().unwrap_or(Json::Null);
        self.context.0 = chat.get("tokens").and_then(Json::as_f64).unwrap_or(0.0); // chats saved before this have none
        self.tasks = tasks_of(chat.get("tasks"));
        self.plan_open = None;
        self.messages.clear();
        self.shown = window();
        for m in chat.get("messages").and_then(Json::as_array).unwrap_or(&[]) {
            let content = m.get("content").and_then(Json::as_str).unwrap_or("");
            match m.get("role").and_then(Json::as_str) {
                // the harness sending the reply back (check your work, keep going): a step in the reply, not
                // a message of the user
                Some("user") if m.get("auto").and_then(Json::as_bool).unwrap_or(false) => {
                    if let Some(reply) = self.messages.last_mut().filter(|r| r.role == Role::Assistant) {
                        let compact = m.get("compact").and_then(Json::as_bool).unwrap_or(false);
                        reply.add_tool(if compact { ToolLine::compacted(content) } else { ToolLine::nudge(content, m.get("goal").and_then(Json::as_bool).unwrap_or(false)) });
                    }
                }
                // sent by the scheduler: its short label, not the whole prompt the model got
                Some("user") if m.get("shown").is_some() => {
                    let mut msg = Msg::user(m.get("shown").and_then(Json::as_str).unwrap_or(content));
                    msg.scheduled = true;
                    self.messages.push(msg);
                }
                Some("user") => self.messages.push(Msg::user(content)),
                Some("assistant") => {
                    if self.messages.last().is_none_or(|l| l.role != Role::Assistant) {
                        self.messages.push(Msg::reply());
                    }
                    let reply = self.messages.last_mut().unwrap();
                    reply.new_round = true;
                    reply.push(content);
                    if let Some(cp) = m.get("checkpoint").and_then(Json::as_str) {
                        reply.checkpoint = Some(cp.to_string());
                    }
                    reply.changed |= m.get("changed").and_then(Json::as_bool).unwrap_or(false);
                    for c in m.get("tool_calls").and_then(Json::as_array).unwrap_or(&[]) {
                        let f = c.get("function");
                        let field = |k: &str| f.and_then(|f| f.get(k)).and_then(Json::as_str).unwrap_or("");
                        reply.add_tool(ToolLine::new(c.get("id").and_then(Json::as_str).unwrap_or(""), field("name"), field("arguments")));
                    }
                }
                Some("tool") => {
                    if let Some(reply) = self.messages.last_mut() {
                        let id = m.get("tool_call_id").and_then(Json::as_str).unwrap_or("");
                        // by its call's id; chats saved without ids answer the first open call
                        let k = reply.tools.iter().position(|t| t.result.is_none() && t.id == id).or_else(|| reply.tools.iter().position(|t| t.result.is_none()));
                        if let Some(t) = k.map(|k| &mut reply.tools[k]) {
                            t.result = Some(content.to_string());
                            t.denied = m.get("denied").and_then(Json::as_bool).unwrap_or(false);
                            t.secs = m.get("secs").and_then(Json::as_f64);
                        }
                    }
                }
                _ => {}
            }
        }
        self.settle_plan();
    }

    /// `task`, and the focus back in the message box if its layout changed since `shape` (it grew or shrank,
    /// or a chip came or went): the box's editor is then made anew, without the focus.
    fn keep_focus(&self, shape: (bool, bool), task: Task<Message>) -> Task<Message> {
        if self.composer_shape() == shape {
            return task;
        }
        Task::batch([task, iced::widget::operation::focus("editor")])
    }

    /// No chat open: what a new one starts with (no messages, task list, goal or context).
    fn clear_chat(&mut self) {
        self.chat = None;
        self.title = None;
        self.chat_cwd = None;
        self.goal = None;
        self.parts = Json::Null;
        self.messages.clear();
        self.shown = window();
        self.context.0 = 0.0;
        self.tasks.clear();
        self.plan_open = None;
    }

    /// What / offers: the server's commands, then the installed skills.
    fn slash_all(&self) -> Vec<Slash> {
        let commands = commands().into_iter().map(|(n, a)| Slash { name: n.to_string(), about: a.to_string() });
        let skills = self.skills.iter().map(|(n, d)| Slash { name: n.clone(), about: d.lines().next().unwrap_or("").to_string() });
        commands.chain(skills).collect()
    }

    /// The installed skills' names: a /name in the text that is one of them is styled, and loaded with it.
    pub fn skill_names(&self) -> Vec<String> {
        self.skills.iter().map(|(n, _)| n.clone()).collect()
    }

    /// The / being typed: the part of a name between a / and the cursor, where the / starts the box or follows
    /// a space and the cursor ends the word. With where it starts in its line (bytes), and whether that is the
    /// start of the box, where the commands are offered too and a pick becomes the chip.
    fn slash_typed(&self) -> Option<(String, usize, bool)> {
        let c = self.composer.cursor();
        if c.selection.is_some() {
            return None;
        }
        let line = self.composer.line(c.position.line)?.text.to_string();
        let col = c.position.column.min(line.len());
        if !line.is_char_boundary(col) || line[col..].starts_with(|ch: char| !ch.is_whitespace()) {
            return None;
        }
        let before = &line[..col];
        let at = before.rfind(char::is_whitespace).map_or(0, |i| i + before[i..].chars().next().map_or(1, char::len_utf8));
        let q = before[at..].strip_prefix('/').filter(|q| !q.contains('/'))?;
        Some((q.to_string(), at, c.position.line == 0 && at == 0 && self.chip.is_none()))
    }

    /// The / list for the / being typed, wherever it is in the text: the skills, and at the start of the box
    /// the commands too. Names that start with what is typed come first, then those that have it elsewhere.
    pub fn slash_list(&self) -> Vec<Slash> {
        if self.slash_shut.as_ref() == Some(&self.composer.text()) {
            return Vec::new();
        }
        let Some((q, _, first)) = self.slash_typed() else { return Vec::new() };
        let q = q.to_lowercase();
        let offered = self.slash_all().into_iter().filter(|e| first || !commands().iter().any(|(n, _)| *n == e.name));
        let (mut list, then): (Vec<Slash>, Vec<Slash>) = offered.filter(|e| e.name.to_lowercase().contains(&q)).partition(|e| e.name.to_lowercase().starts_with(&q));
        list.extend(then);
        // a whole name typed, and nothing longer to offer: nothing to pick, so Enter sends
        if let [only] = list.as_slice() {
            if only.name.to_lowercase() == q {
                return Vec::new();
            }
        }
        list
    }

    /// After typing: "/name " with a name / knows becomes its chip; while the / list is open, the box is
    /// found on screen, for the list to sit above it.
    fn after_edit(&mut self) -> Task<Message> {
        self.slash_pick = 0;
        let text = self.composer.text();
        if self.slash_shut.as_ref() != Some(&text) {
            self.slash_shut = None;
        }
        if self.chip.is_none() {
            let typed = text.strip_prefix('/').and_then(|t| t.split_once(char::is_whitespace));
            if let Some((e, rest)) = typed.and_then(|(name, rest)| Some((self.slash_all().into_iter().find(|e| e.name.eq_ignore_ascii_case(name))?, rest))) {
                self.chip = Some(e.name);
                self.composer = text_editor::Content::with_text(rest);
                self.composer.perform(text_editor::Action::Move(text_editor::Motion::DocumentEnd));
                return Task::none();
            }
        }
        if self.slash_list().is_empty() {
            return Task::none();
        }
        iced::widget::selector::find(iced::widget::selector::id("composer")).map(|t| Message::ComposerAt(t.map(|t| t.bounds())))
    }

    /// Sends the composer's text (or, for a retry, answers the last message again) and streams the reply.
    fn send(&mut self, retry: Option<()>) -> Task<Message> {
        let Some(api) = self.api.clone() else { return Task::none() };
        if self.busy.is_some() {
            return Task::none();
        }
        // no model running: nothing would answer, so the message waits in the box and the model menu opens (or,
        // while one loads, the status says so)
        if self.model.is_none() {
            if self.loading.is_none() {
                return Task::done(Message::OpenMenu(Menu::Model));
            }
            self.status = self.no_model_status();
            return Task::none();
        }
        let mut body = vec![("thinking", Json::Bool(self.thinking)), ("mode", str(MODES[self.perm].0)), ("max_tokens", num(16384.0)), ("preview", Json::Bool(true))];
        match &self.chat {
            Some(id) => body.push(("chat", str(id))),
            // a new chat: the folder it works in from now on
            None => {
                self.chat_cwd = self.new_chat_dir().map(String::from);
                if let Some(dir) = &self.chat_cwd {
                    body.push(("cwd", str(dir)));
                }
            }
        }
        match retry {
            None => {
                let text = self.composer.text().trim().to_string();
                // a chip goes to the server as the command it stands for: /design-taste-frontend make a page
                let text = match self.chip.take() {
                    Some(c) => format!("/{c} {text}").trim_end().to_string(),
                    None => text,
                };
                if text.is_empty() {
                    return Task::none();
                }
                self.composer = text_editor::Content::new();
                self.messages.push(Msg::user(&text));
                body.push(("message", str(&text)));
            }
            Some(()) => {
                if self.messages.last().is_some_and(|m| m.role == Role::Assistant) {
                    self.messages.pop();
                }
            }
        }
        self.messages.push(Msg::reply());
        self.started = Some(Instant::now());
        (self.chunks, self.writing_since) = (0, None);
        (self.reply_tokens, self.reply_writing) = (0.0, (0.0, 0.0));
        // the server holds this message until a scheduled run answering in this chat is done
        let waits = self.chat.is_some() && self.background.get("running").and_then(Json::as_str) == self.chat.as_deref();
        self.status = if waits { tr("Waiting for the scheduled task in this chat to finish").into() } else { "Reading".into() };
        let body = obj(body);
        let (tx, rx) = mpsc::unbounded();
        std::thread::spawn(move || {
            let end = match api.stream("/api/agent", &body, |ev| tx.unbounded_send(ev).is_ok()) {
                Ok(()) => obj(vec![("type", str("end"))]),
                Err(e) => obj(vec![("type", str("error")), ("message", str(&e))]),
            };
            let _ = tx.unbounded_send(end);
        });
        let (task, handle) = Task::run(rx, Message::Agent).abortable();
        self.busy = Some(handle);
        task
    }

    fn on_event(&mut self, ev: Json) -> Task<Message> {
        let kind = ev.get("type").and_then(Json::as_str).unwrap_or("");
        let s = |k: &str| ev.get(k).and_then(Json::as_str).unwrap_or("").to_string();
        match kind {
            "preview" => return self.show_preview(s("url")),
            // the model is replying in another chat: this one goes next
            "waiting" => self.status = s("text"),
            "chat" => {
                let new = self.chat.is_none();
                self.chat = Some(s("id"));
                self.title = Some(s("title"));
                if let Some(m) = self.messages.last_mut() {
                    m.checkpoint = Some(s("checkpoint")).filter(|c| !c.is_empty());
                }
                if new {
                    return self.call(|api| api.get("/api/chats"), Message::Chats);
                }
            }
            "text" => {
                self.chunks += 1;
                self.writing_since.get_or_insert_with(Instant::now);
                self.last_chunk_at = Some(self.born.elapsed().as_millis() as u64);
                if let Some(m) = self.messages.last_mut() {
                    m.push(&s("text"));
                    self.status = if m.rounds.last().is_some_and(|r| r.answer.is_empty() && !r.think.is_empty()) { "Thinking".into() } else { "Writing".into() };
                }
            }
            // a tool call being written, a piece at a time: counted in the speed as text is
            "calling" => {
                self.chunks += 1;
                self.writing_since.get_or_insert_with(Instant::now);
                self.last_chunk_at = Some(self.born.elapsed().as_millis() as u64);
                self.status = match s("name").as_str() {
                    "write_file" | "edit_file" => tr("Writing a file").into(),
                    // a subagent writing (its own text stays out of the reply)
                    "subagent" => tr("A subagent is working").into(),
                    _ => tr("Writing a tool call").into(),
                };
            }
            "tool_call" => {
                if let Some(m) = self.messages.last_mut() {
                    m.add_tool(ToolLine::new(&s("id"), &s("name"), &s("arguments")));
                }
                self.status = "Using a tool".into();
            }
            "tasks" => {
                self.tasks = tasks_of(ev.get("tasks"));
                self.settle_plan();
            }
            "nudge" => {
                let kind = s("kind");
                match kind.as_str() {
                    // the chat outgrew the context: the model summarizes it (Claude Code's compaction)
                    "compacting" => self.status = tr("Compacting the chat").into(),
                    "compact" => {
                        if let Some(m) = self.messages.last_mut() {
                            m.add_tool(ToolLine::compacted(&s("text")));
                        }
                        self.status = tr("Compacted the chat").into();
                    }
                    _ => {
                        let goal = kind == "goal";
                        if let Some(m) = self.messages.last_mut() {
                            m.add_tool(ToolLine::nudge(&s("text"), goal));
                        }
                        self.status = if goal { tr("Back to work on the goal").into() } else { tr("Checking its work").into() };
                    }
                }
            }
            // the chat's /goal: set, checked, met or given up on (null: cleared)
            "goal" => {
                self.goal = goal_of(ev.get("goal"));
                if ev.get("checking").and_then(Json::as_bool).unwrap_or(false) {
                    self.status = tr("Checking the goal").into();
                    if let Some(g) = &mut self.goal {
                        g.checking = true;
                    }
                }
            }
            "tool_result" => {
                let id = s("id");
                if let Some(m) = self.messages.last_mut() {
                    if let Some(t) = m.tools.iter_mut().find(|t| t.id == id) {
                        t.result = Some(s("result"));
                        t.denied = ev.get("denied").and_then(Json::as_bool).unwrap_or(false);
                        t.secs = ev.get("secs").and_then(Json::as_f64);
                        t.ask = None;
                        let changed = t.changed_files();
                        m.changed |= changed;
                    }
                }
                self.status = "Reading the results".into();
            }
            // the run waits until the card under the reply is answered
            "approval" => {
                let id = s("id");
                if let Some(t) = self.messages.last_mut().and_then(|m| m.tools.iter_mut().find(|t| t.id == id)) {
                    t.ask = Some(s("group"));
                }
                self.status = "Waiting for your OK".into();
            }
            // the model's title for a new chat, a moment after its first reply
            "title" => {
                let (id, title) = (s("id"), s("title"));
                if self.chat.as_deref() == Some(id.as_str()) {
                    self.title = Some(title.clone());
                }
                if let Some(c) = self.chats.iter_mut().find(|c| c.id == id) {
                    c.title = title;
                }
            }
            // a round ended: the next one's speed counts afresh (not over the tools run in between)
            "context" => {
                let f = |k: &str| ev.get(k).and_then(Json::as_f64).unwrap_or(0.0);
                self.context = (f("prompt_tokens") + f("completion_tokens"), f("context_length"));
                self.parts = ev.get("parts").cloned().unwrap_or(Json::Null);
                self.end_round(f("completion_tokens"));
            }
            "done" => {
                self.finish(&s("finish"));
                return Task::batch([self.refresh_system(), self.refresh_memory(), self.refresh_extras()]); // the reply may have changed what the GPU holds
            }
            "error" => {
                self.status = s("message");
                self.busy = None;
            }
            "end" => {
                if self.busy.is_some() {
                    self.finish("stop");
                }
            }
            _ => {}
        }
        Task::none()
    }

    /// A round of the reply ended, having generated `tokens`: its share of the reply's count and speed.
    fn end_round(&mut self, tokens: f64) {
        self.reply_tokens += tokens;
        if let Some(t) = self.writing_since.take() {
            self.reply_writing.0 += (tokens - 1.0).max(0.0);
            self.reply_writing.1 += t.elapsed().as_secs_f64();
        }
        self.chunks = 0;
    }

    fn finish(&mut self, how: &str) {
        self.busy = None;
        if how != "stopped" {
            self.happy_until = self.born.elapsed().as_millis() as u64 + 3000;
        }
        let secs = self.started.take().map_or(0.0, |t| t.elapsed().as_secs_f64());
        if self.writing_since.is_some() {
            self.end_round(self.chunks as f64); // a round cut short has no count from the server: its text pieces
        }
        let (tokens, (after_first, writing)) = (self.reply_tokens, self.reply_writing);
        if let Some(m) = self.messages.last_mut().filter(|m| m.role == Role::Assistant) {
            // tokens, the whole wait, and the writing speed (from each round's first token: not counting
            // reading the prompt or the tools run between rounds)
            let speed = if writing > 0.0 && after_first > 0.0 { format!(" · {:.1} tok/s", after_first / writing) } else { String::new() };
            let count = if tokens > 0.0 { tf!("{tokens:.0} tokens · ", "{tokens:.0} 词元 · ") } else { String::new() };
            m.stats = format!("{count}{secs:.1} s{speed}{}", match how {
                "length" => tr(", stopped at the length limit"),
                "stopped" => tr(", stopped"),
                _ => "",
            });
            // a card left open when the reply stopped has nothing to answer any more
            for t in &mut m.tools {
                t.ask = None;
            }
        }
        self.status = String::new();
    }
}

/// Connects and reads the models and chats.
fn connect(addr: &str) -> Result<(Api, Json, Json, Json), String> {
    let api = Api::connect(addr)?;
    let models = api.get("/api/models")?;
    let chats = api.get("/api/chats")?;
    let harness = api.get("/api/harness").unwrap_or(Json::Null);
    Ok((api, models, chats, harness))
}
