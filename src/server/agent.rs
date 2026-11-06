//! The agent loop on the server, for the native app and any other client: a chat's messages go to the model
//! with the tools it may use, its tool calls run (after the user's OK where needed) and their results go
//! back, round after round. Every step streams to the client as an event, and the chat is saved as it goes
//! in %LOCALAPPDATA%\ghost-infer\chats, so every client sees the same chats.
//!
//! A request: {"chat"?, "message"?, "mode"?, "cwd"?, "thinking"?, "max_tokens"?, "tools"?, "keep_going"?,
//! "preview"?, "persona"?}. "persona": "maya" has Maya answer (maya.rs) instead of ghost. "mode" is the permission mode (auto, manual, accept_edits, plan, bypass; default manual), "cwd"
//! a new chat's working directory (kept with the chat), "tools" the tool groups (default: all of them),
//! "preview" true from a client that shows the web pages the model writes (the "preview" event). A message
//! may be a command: /goal <what done looks like> (or /goal clear), /loop <interval> <prompt> (or /loop stop),
//! and a /<skill> anywhere in it loads that installed skill before the model starts.
//!
//! Events, one JSON object per server-sent event:
//!   {"type":"chat","id","title","checkpoint"}     the chat this run belongs to (new chats get an id here), and
//!                                                 the checkpoint its file changes go under (POST /api/tools/undo)
//!   {"type":"waiting","text"}                     the model is replying in another chat: this reply waits for it
//!                                                 (one at a time, so neither reads its whole prompt again)
//!   {"type":"text","text"}                        reply text as it streams (a thinking reply starts with <think>)
//!   {"type":"calling","name","bytes","path"?}     a piece of a tool call being written, one per piece as "text"
//!                                                 is (the call itself comes whole, as "tool_call"); "name" its
//!                                                 tool, "" until that can be read; "bytes" written so far;
//!                                                 "path" its path argument once that is whole
//!   {"type":"tool_call","id","name","arguments"}  the model called a tool
//!   {"type":"approval","id","name","arguments","group"}  waiting for POST /api/agent/approve {"id","decision"}
//!   {"type":"tool_result","id","name","result","denied","secs"?}  ("secs": how long it ran)
//!   {"type":"context","prompt_tokens","completion_tokens","context_length"}  after each round
//!   {"type":"done","finish"} or {"type":"error","message"}
//!   {"type":"tasks","tasks"}                      the model's task list changed: [{"text","status"}]
//!   {"type":"title","id","title"}                 after a chat's first reply: the title the model gave it
//!   {"type":"nudge","kind","text"}                the harness sent the model back: "check" its work, or keep
//!                                                 going ("goal": a /goal not met, or a task list left open);
//!                                                 or the chat outgrew the context: "compacting", then
//!                                                 "compact" with the summary that now stands for it
//!   {"type":"goal","goal","checking"?}            the chat's /goal: {"text","limit","tries","state","reason"}
//!                                                 (state active, met or stopped; null once cleared)
//!   {"type":"preview","url"}                      a web page the model wrote or changed, to show (served
//!                                                 at /preview/...); only to a client that sent "preview"
//!
//! Scheduled tasks (schedule.rs) run here too, with nobody watching: their events go
//! nowhere, the chat file keeps it all, and the chat is marked unread for the clients.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{chat_prompt_for, sse, Event, Job, Kind, Loaded, Part, Style, Shared, ToolCalls, Work};
use crate::json::{num, obj, str, Json};

/// Model turns one reply may take (each tool round is one).
const MAX_ROUNDS: usize = 40;

/// What one round may write for a scheduled run (the app asks for the same): room for a whole web page in
/// one write_file call. The context's end stops it sooner when the chat is long.
const ROUND_TOKENS: usize = 16384;

// ---- tools

/// The tools, by group: what the client turns on. The system group (this computer's specs, the task list
/// and memory) is always there.
const TOOLS: &str = r#"{
 "system": [
  {"type":"function","function":{"name":"get_system_info","description":"Get the specs of the user's computer: CPU, RAM, GPU and VRAM, operating system, free disk space, and what is running on it right now.","parameters":{"type":"object","properties":{},"required":[]}}},
  {"type":"function","function":{"name":"update_tasks","description":"Keep a task list for the current request; the user sees it pinned above the message box. Use it when the work has 3 or more distinct steps, when the user gives you several things to do, or when they ask for a plan. Do not use it for one simple step, for something done in under 3 easy steps, or for chat and questions: just do or answer those. Send the whole list every time; mark the task you start in_progress before working on it and done as soon as it is finished, and add tasks you discover on the way. Example worth a list: 'add a dark mode toggle and make sure the build passes' -> add the toggle to the settings page; store the choice; style the dark theme; build and fix any errors. Not worth one: 'what does this error mean?' (answer it).","parameters":{"type":"object","properties":{"tasks":{"type":"array","items":{"type":"object","properties":{"text":{"type":"string","description":"The step, in a few words"},"status":{"type":"string","enum":["pending","in_progress","done"]}},"required":["text","status"]}}},"required":["tasks"]}}},
  {"type":"function","function":{"name":"remember","description":"Save a lasting fact for future chats. Use it only when the user tells you something worth keeping: a preference, where their projects live, how a program runs here. Not for details of the current task, and not for what you already remember (listed in the system prompt).","parameters":{"type":"object","properties":{"fact":{"type":"string","description":"The fact, in one short sentence"}},"required":["fact"]}}},
  {"type":"function","function":{"name":"forget","description":"Forget a remembered fact that is wrong or no longer true.","parameters":{"type":"object","properties":{"fact":{"type":"string","description":"The fact as it was saved"}},"required":["fact"]}}},
  {"type":"function","function":{"name":"search_chats","description":"Search the user's earlier chats with you, for when they refer to something you talked about before. Returns matching chats with their id, title, date and the lines that match.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Words to look for"}},"required":["query"]}}},
  {"type":"function","function":{"name":"read_chat","description":"Read an earlier chat found with search_chats.","parameters":{"type":"object","properties":{"id":{"type":"string","description":"The chat's id"}},"required":["id"]}}},
  {"type":"function","function":{"name":"schedule_task","description":"Do something later, or again and again, even while the user is away: remind them, check on something, run a job. At that time the prompt is sent to you in this chat, and you work on it with the same tools (only those the user allows without asking). Give exactly one of in_minutes, at, every_minutes or daily_at.","parameters":{"type":"object","properties":{"prompt":{"type":"string","description":"What to do then, as an instruction to yourself, with every detail needed (you may not remember this chat by then)"},"name":{"type":"string","description":"A short name for it, a few words"},"in_minutes":{"type":"integer","description":"Once, this many minutes from now"},"at":{"type":"string","description":"Once, at this local time: HH:MM (today, or tomorrow if it has passed) or YYYY-MM-DD HH:MM"},"every_minutes":{"type":"integer","description":"Again and again, this many minutes apart (at least 5)"},"daily_at":{"type":"string","description":"Every day at this local time, HH:MM"},"days":{"type":"string","description":"With daily_at: only on these days, e.g. mon,tue,wed,thu,fri"}},"required":["prompt"]}}},
  {"type":"function","function":{"name":"list_schedules","description":"The scheduled tasks: id, name, when each runs next, and the time now.","parameters":{"type":"object","properties":{},"required":[]}}},
  {"type":"function","function":{"name":"cancel_schedule","description":"Cancel a scheduled task by its id from list_schedules.","parameters":{"type":"object","properties":{"id":{"type":"string","description":"The task's id"}},"required":["id"]}}},
  {"type":"function","function":{"name":"subagent","description":"Hand a self-contained task to a subagent: it does it with the same tools in a fresh context of its own, and only its final report comes back to you. Use it for work that needs a lot of reading you will not need afterwards (reviewing a long file against guidelines, searching many files, researching a question): that reading then never fills your own context. It does not see this chat, so put everything it needs in the prompt: the paths, what to check, and whether it should change files or only report. Not for reading one known file or a quick search: do those yourself. Its report is not shown to the user: tell them what matters from it.","parameters":{"type":"object","properties":{"description":{"type":"string","description":"A few words saying what it will do, for the user: Review the page"},"prompt":{"type":"string","description":"The task, with everything needed to do it"}},"required":["description","prompt"]}}}
 ],
 "files": [
  {"type":"function","function":{"name":"read_file","description":"Read a text file, with line numbers. For a big file, read part of it with offset and limit.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"offset":{"type":"integer","description":"First line to read, from 1"},"limit":{"type":"integer","description":"How many lines (default 300)"}},"required":["path"]}}},
  {"type":"function","function":{"name":"list_files","description":"List a folder, or with pattern every file under it matching a glob such as **/*.py.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Folder (default: the first allowed folder)"},"pattern":{"type":"string","description":"Glob, e.g. *.md or src/**/*.rs"}},"required":[]}}},
  {"type":"function","function":{"name":"search_files","description":"Search inside files for a regular expression. Returns path:line:text.","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Regular expression"},"path":{"type":"string","description":"Folder or file to search"},"glob":{"type":"string","description":"Only files matching this, e.g. *.rs"},"ignore_case":{"type":"boolean","description":"Ignore upper/lower case"}},"required":["pattern"]}}}
 ],
 "edit": [
  {"type":"function","function":{"name":"write_file","description":"Create a file, or replace a file's whole content. A file too long for one reply goes in parts: the first part, then each further part with append.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"content":{"type":"string","description":"The complete content, or with append the part to add"},"append":{"type":"boolean","description":"Add content to the end of the file instead of replacing it"}},"required":["path","content"]}}},
  {"type":"function","function":{"name":"edit_file","description":"Replace exact text in a file. Read the file first; old_string must match exactly once, so include enough surrounding lines.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"old_string":{"type":"string","description":"The exact text to replace"},"new_string":{"type":"string","description":"The text to put in its place"},"replace_all":{"type":"boolean","description":"Replace every match"}},"required":["path","old_string","new_string"]}}}
 ],
 "cmd": [
  {"type":"function","function":{"name":"run_command","description":"Run programs, builds, tests and scripts: a Windows PowerShell command and its output. To find or read files use list_files, search_files and read_file instead (stops after 60 s unless timeout is set). Each call starts afresh in cwd: a cd does not carry over to the next call, so set cwd instead.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"PowerShell command"},"cwd":{"type":"string","description":"Folder to run it in"},"timeout":{"type":"integer","description":"Seconds, up to 600"},"description":{"type":"string","description":"A few words saying what it does, for the user: Run the unit tests"}},"required":["command"]}}},
  {"type":"function","function":{"name":"start_process","description":"Start a long-running PowerShell command (a server, a build) in the background. Returns its id.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"PowerShell command"},"cwd":{"type":"string","description":"Folder to run it in"},"description":{"type":"string","description":"A few words saying what it does, for the user: Run the unit tests"}},"required":["command"]}}},
  {"type":"function","function":{"name":"check_process","description":"A background process's state and the output it printed since the last check.","parameters":{"type":"object","properties":{"id":{"type":"integer","description":"Process id"}},"required":["id"]}}},
  {"type":"function","function":{"name":"stop_process","description":"Stop a background process.","parameters":{"type":"object","properties":{"id":{"type":"integer","description":"Process id"}},"required":["id"]}}}
 ],
 "web": [
  {"type":"function","function":{"name":"web_search","description":"Search the web. Returns titles, addresses and snippets. When your reply uses these results, name the site each fact comes from and end it with a Sources list of the result links you used; a reply that uses no web results has no Sources list. To see what a site or page itself says, open it with fetch_url instead.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"What to search for"}},"required":["query"]}}},
  {"type":"function","function":{"name":"fetch_url","description":"Open a web page, or a site the user names (example.com), and read what it says as text. Answer only from what the page says.","parameters":{"type":"object","properties":{"url":{"type":"string","description":"http(s) address"}},"required":["url"]}}}
 ]
}"#;

fn tool_groups() -> &'static Json {
    static T: OnceLock<Json> = OnceLock::new();
    T.get_or_init(|| Json::parse(TOOLS).expect("the built-in tool list is valid JSON"))
}

/// Loading and installing skills: always offered, so a skill installed during a reply can be used in it and
/// the tool list (part of the cached prompt) does not change.
const SKILL_TOOLS: [&str; 2] = [
    r#"{"type":"function","function":{"name":"use_skill","description":"Load an installed skill's instructions (the installed skills are listed in the system prompt). Call it before a task one of them is for.","parameters":{"type":"object","properties":{"name":{"type":"string","description":"The skill's name"}},"required":["name"]}}}"#,
    r#"{"type":"function","function":{"name":"install_skill","description":"Install a skill (a folder with a SKILL.md and any files it uses) so it can be loaded with use_skill. The source can be a GitHub link (a repository, a folder in one, or a SKILL.md in one), a link to a .zip, or a folder or .zip on this computer. Use this, not git or a download, when the user asks to install or add a skill. When the source holds several skills, the result lists them; call again with skill set to the path of the one wanted, or all.","parameters":{"type":"object","properties":{"source":{"type":"string","description":"The link, folder or .zip"},"skill":{"type":"string","description":"Which skill, when the source holds several: its path from the list, or all"}},"required":["source"]}}}"#,
];

/// The tools described in full in every prompt, as Claude Code keeps its file and shell tools, and three of
/// ghost's that small models would not load first: get_system_info ("my PC" questions are common; loading it
/// cost GLM-4.7-Flash four extra rounds), update_tasks (Qwen3 4B kept no task list even when asked for one) and
/// remember (Qwen3 4B said it remembered without saving anything). The turned-on groups' other tools are named
/// in the system prompt only, and tool_search loads a description when the model needs one: a short prompt
/// reads faster and leaves a small model less to choose from.
const CORE_TOOLS: [&str; 13] = ["read_file", "list_files", "search_files", "write_file", "edit_file", "run_command", "web_search", "fetch_url", "use_skill", "get_system_info", "update_tasks", "remember", "subagent"];

const TOOL_SEARCH: &str = r#"{"type":"function","function":{"name":"tool_search","description":"Load the full description of tools that the system prompt lists by name only, so you can call them. Use \"select:<name>,<name>\" for tools you know by name, or a few words saying what you need to find tools by what they do. Call it before using such a tool for the first time in a chat. Not for telling the user what you can do: every tool is already named in the system prompt.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"select:<names>, or keywords"},"max_results":{"type":"integer","description":"Most tools to return for keywords (default 5)"}},"required":["query"]}}}"#;

/// The tools the model gets described in the prompt: the core ones of these groups, and tool_search when
/// others wait to be loaded.
fn tools_for(groups: &[String]) -> Vec<Json> {
    let (mut full, deferred): (Vec<Json>, Vec<Json>) = all_tools(groups).into_iter().partition(|t| CORE_TOOLS.contains(&tool_name(t)));
    if !deferred.is_empty() {
        full.push(Json::parse(TOOL_SEARCH).expect("valid tool"));
    }
    full
}

/// The turned-on tools named in the prompt only, loaded with tool_search.
fn deferred_tools(groups: &[String]) -> Vec<Json> {
    all_tools(groups).into_iter().filter(|t| !CORE_TOOLS.contains(&tool_name(t))).collect()
}

/// Whether a tool has parameters to get right (one without can be called by its name alone).
fn takes_arguments(t: &Json) -> bool {
    t.get("function").and_then(|f| f.get("parameters")).and_then(|p| p.get("properties")).is_some_and(|p| matches!(p, Json::Obj(f) if !f.is_empty()))
}

/// Whether this chat already holds the tool's description: a tool_search result, or a refused early call.
fn loaded_in(chat: &Chat, name: &str) -> bool {
    let needle = format!("\"name\":\"{name}\"");
    chat.messages.iter().any(|m| m.get("role").and_then(Json::as_str) == Some("tool") && m.get("content").and_then(Json::as_str).is_some_and(|c| c.contains(&needle)))
}

/// The tool_search tool: deferred tools' full descriptions, by exact name ("select:a,b") or by the words of
/// the query (ranked by how many a tool's name and description hold, its name counting double).
fn tool_search(deferred: &[Json], arguments: &str) -> String {
    let args = Json::parse(arguments).unwrap_or(Json::Null);
    let query = args.get("query").and_then(Json::as_str).unwrap_or("").trim().to_string();
    let max = args.get("max_results").and_then(Json::as_f64).map_or(5, |n| (n as usize).clamp(1, 20));
    // asked for tools that are described already (a model showing what it can do): nothing to load, and no
    // search on their words either, which would load unrelated ones
    if let Some(names) = query.strip_prefix("select:") {
        let names: Vec<&str> = names.split(',').map(str::trim).filter(|n| !n.is_empty()).collect();
        let loaded = |n: &&str| CORE_TOOLS.contains(n) || *n == "tool_search";
        if !names.is_empty() && names.iter().all(loaded) {
            return format!("Nothing to load: {} are described in your prompt already, so call them directly. To tell the user what you can do, answer from the system prompt.", names.join(", "));
        }
    }
    // exact names; a name that is no tool's ("select:disk_space") is searched for by its words instead
    let selected: Vec<&Json> = query.strip_prefix("select:").map_or(Vec::new(), |names| names.split(',').map(str::trim).filter_map(|n| deferred.iter().find(|t| tool_name(t) == n)).collect());
    let found: Vec<&Json> = match selected.is_empty() {
        false => selected,
        true => {
            let words: Vec<String> = query.trim_start_matches("select:").to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 1).map(String::from).collect();
            let mut scored: Vec<(usize, &Json)> = deferred
                .iter()
                .map(|t| {
                    let (name, all) = (tool_name(t).to_lowercase(), t.to_string().to_lowercase());
                    (words.iter().map(|w| 2 * name.contains(w.as_str()) as usize + all.contains(w.as_str()) as usize).sum(), t)
                })
                .filter(|(s, _)| *s > 0)
                .collect();
            scored.sort_by(|a, b| b.0.cmp(&a.0));
            scored.into_iter().take(max).map(|(_, t)| t).collect()
        }
    };
    if found.is_empty() {
        let names: Vec<&str> = deferred.iter().map(tool_name).collect();
        return format!("No tool matches \"{query}\". The tools you can load: {}.", names.join(", "));
    }
    format!("Loaded; you can call these now:\n{}", found.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("\n"))
}

/// Every tool of these groups (and the system one): with the skill tools, and the MCP servers' tools when the
/// mcp group is on.
fn all_tools(groups: &[String]) -> Vec<Json> {
    let all = tool_groups();
    let mut tools: Vec<Json> = std::iter::once("system").chain(groups.iter().map(String::as_str)).filter_map(|g| all.get(g).and_then(Json::as_array)).flatten().cloned().collect();
    tools.extend(SKILL_TOOLS.iter().map(|t| Json::parse(t).expect("valid tool")));
    if groups.iter().any(|g| g == "mcp") {
        tools.extend(super::mcp::tools());
    }
    tools
}

/// Every tool group. A reply gets them all unless the client names some (as in Claude Code, the tools are
/// always there): the permission mode decides which calls need the user's OK.
const ALL_GROUPS: [&str; 5] = ["files", "edit", "cmd", "web", "mcp"];

/// The permission modes, as Claude Code has them: auto (ghost decides, asking only about risky commands,
/// MCP tools and skill installs), manual (ask before every change), accept_edits (file changes go ahead,
/// the rest is asked), plan (nothing is changed or run: the model reads, then proposes a plan) and bypass
/// (nothing is asked).
const MODES: [&str; 5] = ["auto", "manual", "accept_edits", "plan", "bypass"];

/// What a call needs under a permission mode: to run, the user's OK, or nothing done (the reason why).
enum Gate {
    Run,
    Ask,
    Refuse(&'static str),
}

/// Told to the model when plan mode refuses a change.
const PLAN_REFUSAL: &str = "Not done: plan mode is on, so nothing is changed or run. Read and search what you need, then reply with your plan, and ask the user to switch out of plan mode to carry it out.";

/// Rides on each message sent in plan mode.
const PLAN_NOTE: &str = "(Note from ghost, not from the user: plan mode is on. Do not change files or run commands. Read and search what you need, then reply with a short numbered plan, and ask the user to switch out of plan mode to carry it out.)";

fn gate(mode: &str, group: &str, args: &Json) -> Gate {
    match (mode, group) {
        ("bypass", _) => Gate::Run,
        ("plan", "edit" | "cmd" | "skills") => Gate::Refuse(PLAN_REFUSAL),
        ("accept_edits", "edit") | ("auto", "edit" | "web") => Gate::Run,
        ("auto", "cmd") if !risky_command(args.get("command").and_then(Json::as_str).unwrap_or("")) => Gate::Run,
        _ => Gate::Ask,
    }
}

/// A command auto mode still asks about: one that deletes, rewrites or publishes history, installs, changes
/// the system's settings or stops other programs.
fn risky_command(cmd: &str) -> bool {
    let c = format!(" {} ", cmd.to_lowercase().replace(['\n', '\r', '\t', ';', '|', '&', '(', ')', '{', '}'], " "));
    const WORDS: [&str; 26] = [
        "remove-item", "rm", "del", "erase", "rd", "ri", "rmdir", "clear-content", "format", "format-volume", "diskpart", "shutdown", "restart-computer", "stop-computer", "stop-process", "kill", "taskkill", "set-executionpolicy", "reg", "set-itemproperty", "new-itemproperty", "remove-itemproperty", "takeown", "icacls", "invoke-expression", "iex",
    ];
    const PHRASES: [&str; 14] = ["git push", "git reset --hard", "git clean", "git checkout --", "git branch -d", "npm publish", "cargo publish", "pip install", "npm install -g", "npm i -g", "winget ", "choco ", "msiexec", "-verb runas"];
    WORDS.iter().any(|w| c.contains(&format!(" {w} "))) || PHRASES.iter().any(|p| c.contains(p))
}

/// The folders a chat's tools may use: its working directory first (relative paths and commands start
/// there), then the other folders the user allows.
fn run_folders(cwd: &str, allowed: &[PathBuf]) -> Vec<PathBuf> {
    let mut folders: Vec<PathBuf> = Vec::new();
    for f in std::iter::once(PathBuf::from(cwd)).filter(|_| !cwd.is_empty()).chain(allowed.iter().cloned()) {
        if !folders.contains(&f) {
            folders.push(f);
        }
    }
    folders
}

/// The group whose calls need the user's OK, if any.
fn approval_group(name: &str) -> Option<&'static str> {
    match name {
        "write_file" | "edit_file" => Some("edit"),
        "run_command" | "start_process" => Some("cmd"),
        "web_search" | "fetch_url" => Some("web"),
        "install_skill" => Some("skills"),
        n if n.starts_with("mcp__") => Some("mcp"),
        _ => None,
    }
}

/// Told to the model when tools are on: where it may work, how this computer runs commands, and (when it may
/// run them) to check what it changes.
fn tool_prompt(folders: &[PathBuf], can_run: bool, deferred: &[Json]) -> String {
    let list: Vec<String> = folders.iter().map(|f| f.display().to_string()).collect();
    let where_ = match list.split_first() {
        None => "No working directory is set, so file tools will fail; if you need files, ask the user to pick a folder for this chat.".to_string(),
        Some((cwd, [])) => format!("Your working directory is {cwd}: relative paths and commands start there."),
        Some((cwd, more)) => format!("Your working directory is {cwd}: relative paths and commands start there. You may also use files in {}.", more.join(", ")),
    };
    let check = if can_run { " After you change code, check it: run it or its tests with run_command. If something fails, fix it and run it again. Say it is done only once it works, and say how you checked." } else { "" };
    let more = if deferred.is_empty() {
        String::new()
    } else {
        let names: Vec<&str> = deferred.iter().map(tool_name).collect();
        format!(" These tools are named here without their descriptions: {}. Before calling one of them for the first time in a chat, load it with tool_search (\"select:<name>\").", names.join(", "))
    };
    format!("You can use tools to do things on the user's computer. {where_} It runs Windows; commands run in Windows PowerShell. Answer greetings, chat and questions you can answer yourself directly, without tools; that includes questions about what you can do, which the tools below answer. Use tools when the request needs them; calls that do not depend on each other can go together in one turn. Before saying you cannot do something, check whether one of your tools can. Report what tools show faithfully: if a command failed or a search did not answer the question, say so, and never present a guess as a finding. To find and read files use list_files, search_files and read_file rather than commands, and read a file before you edit it. For a task of 3 or more steps, keep a task list with update_tasks. Save a fact with remember only when the user tells you something worth keeping for later chats. When the user refers to an earlier conversation, look it up with search_chats.{check}{more} Use tools until the task is done, then answer briefly.")
}

/// Files whose changes can be checked by running something (a notes file is not; nor a web page or its
/// stylesheet, which the preview pane shows as it changes: sent back to "run" one, a model reads it all again).
const CODE: [&str; 29] = [
    "rs", "py", "js", "mjs", "cjs", "ts", "tsx", "jsx", "go", "java", "kt", "c", "h", "cc", "cpp", "hpp", "cs", "rb", "php", "swift", "lua", "sh", "ps1", "bat", "cmd", "vue", "svelte", "sql", "dart",
];

fn is_code(path: &str) -> bool {
    path.rsplit_once('.').is_some_and(|(_, ext)| CODE.iter().any(|c| c.eq_ignore_ascii_case(ext)))
}

/// A `<tool_call>` block the model wrote that could not be read even mended: it stays in the chat as a call
/// of this name (its arguments the block as written), and its result asks for the call again.
const BAD_CALL: &str = "unreadable_tool_call";

/// The result of an unreadable call: send it again, in the form the model wrote it in (Qwen3-Coder's
/// <function=...>, GLM's <arg_key>, or JSON).
fn bad_call_result(arguments: &str) -> String {
    let written = Json::parse(arguments).ok().and_then(|a| a.get("text").and_then(Json::as_str).map(String::from)).unwrap_or_default();
    let form = if written.trim_start().starts_with("<function=") {
        "<tool_call><function=TOOL><parameter=NAME>value</parameter></function></tool_call>, with every tag closed"
    } else if written.contains("<arg_key>") {
        "<tool_call>TOOL<arg_key>NAME</arg_key><arg_value>value</arg_value></tool_call>, with every tag closed"
    } else {
        "<tool_call>{\"name\": \"TOOL\", \"arguments\": {...}}</tool_call> as valid JSON, with every brace and bracket closed"
    };
    format!("Error: this tool call could not be read, so nothing was run. Send it again as {form}.")
}

/// The most of a tool's output the model gets as it is, in characters: an eighth of its context (8,000 in 64K),
/// from 4,000 to 30,000. Claude Code's limits are 30,000 to 100,000 in 200K.
fn output_chars(ctx: usize) -> usize {
    (ctx / 8).clamp(4000, 30000)
}

/// A tool's output too long for the context (output_chars), saved to a file in %LOCALAPPDATA%\ghost-infer\
/// outputs\<chat>, and the model given its start and where the rest is, as Claude Code's <persisted-output>
/// does: it reads on with read_file or searches it with search_files if it needs more. A recursive folder
/// listing came to 16,000 characters twice in one chat, each pushing it into a compaction. A file it read
/// (lines it asked for) and a skill's instructions (their own limit) are not saved, only cut if very long.
fn spill(output: String, name: &str, chat_id: &str, call_id: &str, ctx: usize) -> String {
    let n = output.chars().count();
    if n <= output_chars(ctx) {
        return output;
    }
    if matches!(name, "read_file" | "use_skill" | "subagent") {
        return crate::tools::clip(&output, crate::tools::MAX_RESULT);
    }
    let dir = crate::tools::outputs_dir().map(|d| d.join(chat_id));
    let path = dir.filter(|d| std::fs::create_dir_all(d).is_ok()).map(|d| d.join(format!("{call_id}.txt")));
    let Some(path) = path.filter(|p| std::fs::write(p, &output).is_ok()) else {
        return crate::tools::clip(&output, crate::tools::MAX_RESULT);
    };
    let preview: String = output.chars().take(2000).collect();
    format!(
        "<persisted-output>\nOutput too large ({n} characters, {} lines). Full output saved to: {}\nRead it with read_file (offset and limit) or search it with search_files when you need more of it.\n\nPreview (first 2,000 characters):\n{preview}\n...\n</persisted-output>",
        output.lines().count(),
        path.display()
    )
}

/// The file a cut-off call was writing, from what of it came: Qwen3-Coder's <parameter=path>, GLM's
/// <arg_key>path</arg_key><arg_value>, or JSON's "path".
fn cut_path(raw: &str) -> Option<String> {
    let after = |open: &str, close: &str| raw.split_once(open).and_then(|(_, r)| r.split_once(close)).map(|(p, _)| p.trim().to_string());
    after("<parameter=path>", "</parameter>")
        .or_else(|| after("<arg_key>path</arg_key><arg_value>", "</arg_value>"))
        .or_else(|| after("\"path\": \"", "\"").or_else(|| after("\"path\":\"", "\"")).map(|p| p.replace("\\\\", "\\")))
        .filter(|p| !p.is_empty())
}

/// The result of a call the reply's length limit cut off (`tokens` generated): sending it again whole would
/// be cut off again, so a long file goes in parts.
fn cut_call_result(tokens: usize) -> String {
    format!(
        "Error: your reply reached its limit ({tokens} tokens) in the middle of this tool call, so it was cut off and nothing was run. \
Do not send it again whole: it would be cut off again. Write a long file in parts of at most about {} tokens: write_file with the first part, \
then a write_file call with \"append\": true for each further part, one call per reply.",
        (tokens / 2).max(500)
    )
}

/// What the harness says when a reply would end with changed code that nothing has run since.
const NUDGE: &str = "(Automatic check) You changed code in this reply and have not run anything since. Check your work now: run the code or its tests with run_command. If something fails, fix it and run it again. If it cannot be run here, or the user asked you not to run it, say so in one line.";

/// The goal loop, when the user turns it on ("keep going until done"; scheduled runs always have it): a
/// reply that stops with steps of its task list still open is sent back to them, up to this many times in a
/// row without a step getting done. A reply to the nudge with no step done is the model's answer to it (the
/// user said to wait, or a step cannot be done), so once is enough. Not on by default: told nothing, a model
/// takes the nudge for the user's word and goes past a "do step 1, then wait for me" about half the time.
const GOAL_NUDGES: usize = 1;

/// Times a /goal sends the model back when the goal gives no limit of its own ("stop after 5 tries").
const GOAL_TRIES: usize = 8;

/// A /goal: what done looks like in the user's words (Claude Code's goal-based loop). When the model stops,
/// a check reads the chat against it; not met, the model goes back to work with what is missing, until it is
/// met or `limit` checks have said no.
fn new_goal(text: &str) -> Json {
    obj(vec![("text", str(text)), ("limit", num(goal_limit(text) as f64)), ("tries", num(0.0)), ("state", str("active")), ("reason", str(""))])
}

/// The limit a goal sets itself: a number before tries, attempts, turns, times or rounds.
fn goal_limit(text: &str) -> usize {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    words.windows(2).find_map(|w| ["try", "tries", "attempts", "turns", "times", "rounds"].contains(&w[1]).then(|| w[0].parse::<usize>().ok()).flatten()).map_or(GOAL_TRIES, |n| n.clamp(1, 20))
}

fn goal_active(chat: &Chat) -> bool {
    chat.goal.get("state").and_then(Json::as_str) == Some("active")
}

/// Whether the chat's goal is met, asked as Claude Code's evaluator asks when Claude tries to stop: a question
/// after the chat (so what the model has read stays read), without thinking, a JSON verdict back. No verdict
/// counts as not met.
fn check_goal(req: &Request, chat: &Chat, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> (bool, String) {
    let goal = chat.goal.get("text").and_then(Json::as_str).unwrap_or("");
    let tools = if req.groups.is_empty() { Vec::new() } else { tools_for(&req.groups) };
    let mut history = for_model(&chat.messages);
    let ask = format!("(Check from ghost, not a message from the user.) The user's goal for this chat: {goal}\nJudge whether it is met from what the tool results above show; a reply saying it is done is not proof. Reply with JSON only and no tool calls: {{\"met\": true, \"reason\": \"the result that shows it\"}} or {{\"met\": false, \"reason\": \"what is still missing\"}}.");
    history.push(obj(vec![("role", str("user")), ("content", str(&ask)), ("aside", Json::Bool(true))]));
    let Ok(prompt) = prompt_for(req, &history, &tools, false, shared, loaded) else { return (false, "the check could not be made".into()) };
    let (tx, events) = mpsc::channel();
    let job = Job { model: loaded.id.clone(), kind: Kind::Chat, prompt, max_tokens: Some(120), temperature: 0.0, top_k: shared.defaults.1, top_p: shared.defaults.2, seed: 1, style: Style::default(), stop: Vec::new(), lead: "", events: tx };
    if work.send(Work::Job(job)).is_err() {
        return (false, "the check could not be made".into());
    }
    let mut out = String::new();
    for ev in events {
        match ev {
            Event::Text(t) => out.push_str(&t),
            Event::Done { .. } | Event::Error(..) => break,
        }
    }
    let out = out.rsplit("</think>").next().unwrap_or(&out);
    let verdict = out.find('{').zip(out.rfind('}')).filter(|(a, b)| a < b).and_then(|(a, b)| Json::parse(&out[a..=b]).ok());
    let met = verdict.as_ref().and_then(|v| v.get("met")).and_then(Json::as_bool).unwrap_or(false);
    let reason = verdict.as_ref().and_then(|v| v.get("reason")).and_then(Json::as_str).unwrap_or("no verdict").trim().to_string();
    (met, reason.chars().take(300).collect())
}

/// A message that is a command rather than words for the model.
enum Command {
    Goal(String),       // /goal <what done looks like>
    ClearGoal,          // /goal clear
    Loop(u32, String),  // /loop <interval> <prompt>: Claude Code's time-based loop, in this chat
    StopLoops,          // /loop stop
}

fn command(m: &str) -> Option<Result<Command, String>> {
    let m = m.trim();
    let (head, rest) = m.split_once(char::is_whitespace).map_or((m, ""), |(h, r)| (h, r.trim()));
    match head {
        "/goal" if rest.is_empty() => Some(Err("Give the goal: /goal <what done looks like>, for example /goal all tests pass, stop after 5 tries. /goal clear ends one.".into())),
        "/goal" if matches!(rest.to_lowercase().as_str(), "clear" | "off" | "stop" | "none") => Some(Ok(Command::ClearGoal)),
        "/goal" => Some(Ok(Command::Goal(rest.to_string()))),
        "/loop" if matches!(rest.to_lowercase().as_str(), "stop" | "off" | "clear") => Some(Ok(Command::StopLoops)),
        "/loop" => {
            let (every, prompt) = rest.split_once(char::is_whitespace).map_or((rest, ""), |(e, p)| (e, p.trim()));
            match (interval_minutes(every), prompt.is_empty()) {
                (Some(n), false) => Some(Ok(Command::Loop(n, prompt.to_string()))),
                _ => Some(Err("Give how often and what to do: /loop 10m check the build and fix what fails (m for minutes, h for hours; at least 5m). /loop stop ends this chat's loops.".into())),
            }
        }
        _ => None,
    }
}

/// The installed skills a message names after a slash, anywhere in it (/design-taste-frontend make a page,
/// or make a page with /design-taste-frontend): a word that is a / and a skill's name, perhaps with a mark
/// such as a comma after it. Each once, in the order named.
fn skill_commands(m: &str) -> Vec<String> {
    let skills = super::skills::list();
    let mut named: Vec<String> = Vec::new();
    for word in m.split_whitespace() {
        let Some(name) = word.strip_prefix('/') else { continue };
        let name = name.trim_end_matches(|c: char| !(c.is_alphanumeric() || c == '-' || c == '_'));
        if let Some(s) = skills.iter().find(|s| s.name.eq_ignore_ascii_case(name)) {
            if !named.contains(&s.name) {
                named.push(s.name.clone());
            }
        }
    }
    named
}

/// "10m", "10min", "2h", "90" (minutes): minutes, at least the scheduler's 5.
fn interval_minutes(s: &str) -> Option<u32> {
    let s = s.to_lowercase();
    let digits = s.trim_end_matches(|c: char| c.is_alphabetic());
    let n: f64 = digits.parse().ok().filter(|n: &f64| *n > 0.0)?;
    let minutes = match &s[digits.len()..] {
        "" | "m" | "min" | "mins" | "minute" | "minutes" => n,
        "h" | "hr" | "hrs" | "hour" | "hours" => n * 60.0,
        _ => return None,
    };
    Some((minutes.round() as u32).max(5))
}

/// What the harness says when a reply would end with its task list unfinished.
fn goal_nudge(open: &[String]) -> String {
    format!("(Automatic check from ghost, not a message from the user) Your reply ended with steps of your task list still open: {}. If you have done them already, mark them done with update_tasks. If the user asked you to stop or wait before them, do not do them: reply in one line that they are waiting for the user. If a step cannot be done here, say why in one line. Otherwise the user's request is not finished: carry on with the next step now.", open.join("; "))
}

/// The task list's open steps, unless the reply is about to close them: when only the step in progress is
/// left, the run marks it done (close_last_task), so nothing is open.
fn open_tasks(chat: &Chat) -> Vec<String> {
    let tasks = chat.tasks.as_array().unwrap_or(&[]);
    let status = |t: &Json| t.get("status").and_then(Json::as_str).unwrap_or("").to_string();
    let open: Vec<&Json> = tasks.iter().filter(|t| status(t) != "done").collect();
    if let [only] = open[..] {
        if status(only) == "in_progress" {
            return Vec::new();
        }
    }
    open.iter().map(|t| t.get("text").and_then(Json::as_str).unwrap_or("").to_string()).collect()
}

/// Model replies without a task list update after which a reminder rides on the next message, and again at
/// most this often (Claude Code waits 10 turns for both).
const REMINDER_TURNS: usize = 10;

/// The task-list reminder for a new message, when the last REMINDER_TURNS replies neither touched the list
/// nor got a reminder.
fn task_reminder(chat: &Chat) -> Option<String> {
    let mut replies = 0;
    for m in chat.messages.iter().rev() {
        let calls = m.get("tool_calls").and_then(Json::as_array).unwrap_or(&[]);
        if m.get("reminder").is_some() || calls.iter().any(|c| c.get("function").and_then(|f| f.get("name")).and_then(Json::as_str) == Some("update_tasks")) {
            return None;
        }
        if m.get("role").and_then(Json::as_str) == Some("assistant") {
            replies += 1;
            if replies >= REMINDER_TURNS {
                break;
            }
        }
    }
    if replies < REMINDER_TURNS {
        return None;
    }
    let mut note = "(Note from ghost, not from the user: the task list has not been used for a while. If the work ahead has several steps, a task list (update_tasks) helps you and the user follow it; if not, ignore this note. Do not mention it.)".to_string();
    let open = open_tasks(chat);
    if !open.is_empty() {
        note.push_str(&format!(" Open on the current list: {}.", open.join("; ")));
    }
    Some(note)
}

/// The harness's own messages in a chat (the nudge): not the user's, so retrying and searching pass them by.
fn is_auto(m: &Json) -> bool {
    m.get("auto").and_then(Json::as_bool).unwrap_or(false)
}

// ---- fitting a chat to the context: the chat keeps every message, marked with what the model no longer gets

/// What the model gets in place of a tool's output taken out to make room.
const TRIMMED: &str = "[This output was taken out to make room in the context. Call the tool again if it is needed.]";

/// Put before the first message the model still gets, once earlier ones were left out.
const LEFT_OUT: &str = "(Earlier messages of this chat were left out to make room in the context.)";

fn flag(m: &Json, k: &str) -> bool {
    m.get(k).and_then(Json::as_bool).unwrap_or(false)
}

/// A call's arguments once taken out to make room ("args_trimmed" on its message): each long text in them (a
/// file the model wrote, an edit's old and new text) is only how long it was.
fn trimmed_args(name: &str, arguments: &str) -> String {
    let Ok(Json::Obj(fields)) = Json::parse(arguments) else { return arguments.to_string() };
    let where_ = if matches!(name, "write_file" | "edit_file") { ": the file has them, read it if needed" } else { "" };
    let fields = fields
        .into_iter()
        .map(|(k, v)| match v {
            Json::Str(s) if s.len() > 300 => (k, str(&format!("[{} lines, taken out to make room in the context{where_}]", s.lines().count()))),
            v => (k, v),
        })
        .collect();
    Json::Obj(fields).to_string()
}

/// The arguments of a message's calls as the model gets them, all together (empty when it made none).
fn call_args(m: &Json) -> String {
    let calls = m.get("tool_calls").and_then(Json::as_array).unwrap_or(&[]);
    let cut = flag(m, "args_trimmed");
    calls
        .iter()
        .filter_map(|c| {
            let f = c.get("function")?;
            let (name, args) = (f.get("name").and_then(Json::as_str).unwrap_or(""), f.get("arguments").and_then(Json::as_str)?);
            Some(if cut { trimmed_args(name, args) } else { args.to_string() })
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn role(m: &Json) -> &str {
    m.get("role").and_then(Json::as_str).unwrap_or("")
}

/// Sets a message's field `k`.
fn mark(m: &mut Json, k: &str, v: Json) {
    if let Json::Obj(fields) = m {
        fields.retain(|(f, _)| f != k);
        fields.push((k.into(), v));
    }
}

/// The chat as the model reads it: messages left out ("dropped") skipped, tool outputs taken out ("trimmed")
/// replaced by a note, messages cut down ("clip": the characters kept) cut in the middle, and the long texts in
/// calls' arguments taken out ("args_trimmed") said in a line.
fn for_model(messages: &[Json]) -> Vec<Json> {
    let mut out: Vec<Json> = Vec::with_capacity(messages.len());
    let mut left_out = false;
    for m in messages {
        if flag(m, "dropped") {
            left_out = true;
            continue;
        }
        // (a compaction's summary says itself what it stands for)
        let (trimmed, clip, note) = (flag(m, "trimmed"), m.get("clip").and_then(Json::as_f64).map(|c| c as usize), left_out && out.is_empty() && !flag(m, "compact"));
        let mut m = m.clone();
        if let (Some(r), Json::Obj(fields)) = (m.get("reminder").and_then(Json::as_str).map(String::from), &mut m) {
            if let Some((_, Json::Str(t))) = fields.iter_mut().find(|(k, _)| k == "content") {
                t.push_str("\n\n");
                t.push_str(&r);
            }
        }
        if let (true, Json::Obj(fields)) = (trimmed || clip.is_some() || note, &mut m) {
            if let Some((_, Json::Str(t))) = fields.iter_mut().find(|(k, _)| k == "content") {
                if trimmed {
                    *t = TRIMMED.into();
                } else if let Some(c) = clip {
                    *t = crate::tools::clip(t, c);
                }
                if note {
                    *t = format!("{LEFT_OUT}\n\n{t}");
                }
            }
        }
        if let (true, Json::Obj(fields)) = (flag(&m, "args_trimmed"), &mut m) {
            if let Some((_, Json::Arr(calls))) = fields.iter_mut().find(|(k, _)| k == "tool_calls") {
                for call in calls {
                    let Some(Json::Obj(f)) = (match call {
                        Json::Obj(c) => c.iter_mut().find(|(k, _)| k == "function").map(|(_, f)| f),
                        _ => None,
                    }) else {
                        continue;
                    };
                    let name = f.iter().find(|(k, _)| k == "name").and_then(|(_, v)| v.as_str()).unwrap_or("").to_string();
                    if let Some((_, Json::Str(a))) = f.iter_mut().find(|(k, _)| k == "arguments") {
                        *a = trimmed_args(&name, a);
                    }
                }
            }
        }
        out.push(m);
    }
    out
}

/// Makes about `need` tokens of room, the least missed first: tool outputs taken out, oldest first, all but
/// the last 3; the long texts in the model's earlier calls (a file it wrote, an edit's texts: the file still
/// has them), all but the latest; then the oldest turns left out whole; then, when the turn being answered is
/// too big on its own, the latest call's texts, and its biggest messages cut in half. Whether it found
/// anything to take out.
fn make_room(chat: &mut Chat, need: usize, tok: &crate::tokenizer::Tokenizer) -> bool {
    // a message's text and its calls' arguments as the model gets them now, in tokens
    let text_cost = |m: &Json| -> usize {
        if flag(m, "dropped") {
            return 0;
        }
        if flag(m, "trimmed") {
            return 25;
        }
        let t = m.get("content").and_then(Json::as_str).unwrap_or("");
        let shown = m.get("clip").and_then(Json::as_f64).map(|c| crate::tools::clip(t, c as usize));
        tok.encode(shown.as_deref().unwrap_or(t)).len()
    };
    let args_cost = |m: &Json| -> usize {
        let args = call_args(m);
        if flag(m, "dropped") || args.is_empty() { 0 } else { tok.encode(&args).len() }
    };
    let cost = |m: &Json| text_cost(m) + args_cost(m);
    // takes out a message's long call texts: the tokens that freed
    let cut_args = |m: &mut Json| -> usize {
        let before = args_cost(m);
        mark(m, "args_trimmed", Json::Bool(true));
        before.saturating_sub(args_cost(m))
    };
    let msgs = &mut chat.messages;
    let current = msgs.iter().rposition(|m| role(m) == "user" && !is_auto(m)).unwrap_or(0);
    let mut freed = 0;
    let outputs: Vec<usize> = (0..msgs.len()).filter(|&i| role(&msgs[i]) == "tool" && !flag(&msgs[i], "trimmed") && !flag(&msgs[i], "dropped")).collect();
    for &i in &outputs[..outputs.len().saturating_sub(3)] {
        if freed >= need {
            break;
        }
        let c = text_cost(&msgs[i]);
        if c > 100 {
            mark(&mut msgs[i], "trimmed", Json::Bool(true));
            freed += c - 25;
        }
    }
    let writes: Vec<usize> = (0..msgs.len()).filter(|&i| !flag(&msgs[i], "dropped") && !flag(&msgs[i], "args_trimmed") && call_args(&msgs[i]).len() > 1200).collect();
    for &i in &writes[..writes.len().saturating_sub(1)] {
        if freed >= need {
            break;
        }
        freed += cut_args(&mut msgs[i]);
    }
    // a turn: a user's message and everything up to their next one
    let mut i = 0;
    while freed < need && i < current {
        let end = (i + 1..current).find(|&j| role(&msgs[j]) == "user" && !is_auto(&msgs[j])).unwrap_or(current);
        for j in i..end {
            freed += cost(&msgs[j]);
            mark(&mut msgs[j], "dropped", Json::Bool(true));
        }
        i = end;
    }
    if let Some(&last) = writes.last() {
        if freed < need && !flag(&msgs[last], "dropped") {
            freed += cut_args(&mut msgs[last]);
        }
    }
    let mut big: Vec<(usize, usize)> = (current..msgs.len()).map(|j| (text_cost(&msgs[j]), j)).filter(|(c, _)| *c > 1000).collect();
    big.sort_by(|a, b| b.cmp(a));
    for (c, j) in big {
        if freed >= need {
            break;
        }
        let chars = msgs[j].get("content").and_then(Json::as_str).map_or(0, |t| t.chars().count());
        let kept = msgs[j].get("clip").and_then(Json::as_f64).map_or(chars, |c| c as usize);
        mark(&mut msgs[j], "clip", num((kept / 2) as f64));
        freed += c / 2;
    }
    freed > 0
}

/// What the model is asked when a chat is compacted: Claude Code's summary (its nine sections made five, for
/// a small model), after the chat itself so the server's cache still holds the prompt.
const COMPACT_PROMPT: &str = "(From ghost, not from the user) This chat has grown too long for your context, so it is being compacted: everything above will be replaced by your summary of it. Do not call any tools; write only the summary, inside <summary></summary>, under these headings:
1. The user's requests: every request and instruction the user gave, in their own words, the latest last.
2. Done so far: the files created or changed (full paths) and what is in them now, the commands run and what they showed. Only what a tool's result shows happened counts: a call that failed or was cut off changed nothing, so say what it was meant to do and that it is still to do.
3. What was learned: facts and decisions that matter for the rest of the work, errors and how they were fixed, anything the user said to do differently.
4. Current work: exactly what you were doing right before this, and where it stopped.
5. Next step: the next thing to do for the user's latest request, if anything is left.
Be specific (names, paths, numbers) and leave out what no longer matters.";

/// Put before the summary, in place of the chat it covers.
const COMPACTED: &str = "(From ghost, not from the user) This chat was compacted to fit your context: the summary below takes the place of everything before it. Continue the work from where it stopped, without recapping it or asking anything; read again any file you need.";

/// The summary's length at most: a twelfth of the context, from 1,024 to 4,096 tokens.
fn summary_tokens(loaded: &Loaded) -> usize {
    (loaded.ctx / 12).clamp(1024, 4096)
}

/// Compacts the chat as Claude Code does once its context is full: the model writes a summary of everything so
/// far, and the summary takes the place of the messages it covers (left out of the model's prompt; the chat
/// file keeps them). The prompt after it is short, so many rounds go by before the next one, each reading
/// only what is new. Ok(false) when no usable summary came back.
fn compact(out: &mut Out, req: &Request, chat: &mut Chat, tools: &[Json], work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<bool, String> {
    emit(out, obj(vec![("type", str("nudge")), ("kind", str("compacting")), ("text", str("Compacting the chat to fit the context"))]))?;
    // the summary begun for the model, so it writes one (asked, a small model may call a tool instead)
    const START: &str = "<summary>\n1. The user's requests:";
    let budget = summary_tokens(loaded);
    let mut asked = None;
    for _ in 0..4 {
        let mut history = for_model(&chat.messages);
        history.push(obj(vec![("role", str("user")), ("content", str(COMPACT_PROMPT)), ("aside", Json::Bool(true))]));
        let prompt = format!("{}{START}", prompt_for(req, &history, tools, false, shared, loaded)?);
        // the question and the summary have to fit: old tool outputs out first if they do not
        let n = loaded.tok.encode(&prompt).len();
        if n + budget <= loaded.ctx {
            asked = Some(prompt);
            break;
        }
        if !make_room(chat, n + budget + 100 - loaded.ctx, &loaded.tok) {
            break;
        }
    }
    let Some(prompt) = asked else { return Ok(false) };
    let (tx, events) = mpsc::channel();
    let job = Job { model: loaded.id.clone(), kind: Kind::Chat, prompt, max_tokens: Some(budget), temperature: 0.3, top_k: shared.defaults.1, top_p: shared.defaults.2, seed: 1, style: Style::default(), stop: vec!["</summary>".into()], lead: "", events: tx };
    work.send(Work::Job(job)).map_err(|_| "the model thread stopped")?;
    let mut text = String::new();
    for ev in events {
        match ev {
            Event::Text(t) => text.push_str(&t),
            Event::Done { .. } => break,
            Event::Error(_, e) => return Err(e),
        }
    }
    // the summary alone: not a tool call it went on to anyway
    let written = text.split("<tool_call>").next().unwrap_or(&text);
    let summary = format!("1. The user's requests:{}", written.split("</summary>").next().unwrap_or(written));
    let summary = summary.trim();
    if summary.len() < 200 {
        return Ok(false);
    }
    for m in chat.messages.iter_mut() {
        mark(m, "dropped", Json::Bool(true));
    }
    let mut msg = vec![("role", str("user")), ("content", str(&format!("{COMPACTED}\n\n{summary}"))), ("auto", Json::Bool(true)), ("compact", Json::Bool(true))];
    if let Some(r) = task_reminder(chat) {
        msg.push(("reminder", str(&r)));
    }
    chat.messages.push(obj(msg));
    chat.save()?;
    emit(out, obj(vec![("type", str("nudge")), ("kind", str("compact")), ("text", str(summary))]))?;
    Ok(true)
}

/// This round's prompt, made to fit: when it would leave the reply less room than it asks for (up to 3/8 of
/// the context: a whole web page in one write_file, about 8,000 tokens, with its thinking, in 32K; at a
/// quarter one was cut off), the chat is compacted into a summary (compact); if no summary comes, make_room
/// takes out enough that it fills at most half of the rest, so the next rounds fit too and start the same
/// way (the server's cache still holds them: each taking out changes the prompt's start, and the whole of it
/// is read again).
fn fitted_prompt(out: &mut Out, req: &Request, chat: &mut Chat, tools: &[Json], work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<String, String> {
    let limit = loaded.ctx - req.max_tokens.min(loaded.ctx * 3 / 8);
    let mut compacted = false;
    for _ in 0..8 {
        let prompt = prompt_for(req, &for_model(&chat.messages), tools, req.thinking, shared, loaded)?;
        // a token is at least one byte, so a prompt this short fits without counting
        if prompt.len() <= limit {
            return Ok(prompt);
        }
        let n = loaded.tok.encode(&prompt).len();
        if n <= limit {
            return Ok(prompt);
        }
        // once: a summary in place of the chat (more than the message being answered to summarize)
        if !compacted && chat.messages.iter().filter(|m| !flag(m, "dropped")).count() > 2 {
            compacted = true;
            if compact(out, req, chat, tools, work, shared, loaded)? {
                continue;
            }
        }
        if !make_room(chat, n - limit / 2, &loaded.tok) {
            return Err(format!(
                "this chat no longer fits the model's context: it needs {n} tokens, and the context holds {} with {} kept for the reply. Start a new chat (or shorten the message, if it is the long part)",
                loaded.ctx,
                loaded.ctx - limit
            ));
        }
    }
    Err("this chat cannot be made to fit the model's context; start a new chat".into())
}

// ---- memory: facts the model saved, kept in %LOCALAPPDATA%\ghost-infer\memory.json for every chat

/// %LOCALAPPDATA%\ghost-infer, made if missing.
pub(super) fn data_dir() -> Result<PathBuf, String> {
    let d = std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer")).ok_or("LOCALAPPDATA is not set")?;
    std::fs::create_dir_all(&d).map_err(|e| format!("cannot make {}: {e}", d.display()))?;
    Ok(d)
}

fn memory_file() -> Result<PathBuf, String> {
    Ok(data_dir()?.join("memory.json"))
}

/// The remembered facts, oldest first.
pub fn memory() -> Vec<String> {
    let text = memory_file().ok().and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
    let j = Json::parse(&text).unwrap_or(Json::Null);
    j.get("facts").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect()
}

fn save_memory(facts: &[String]) -> Result<(), String> {
    let j = obj(vec![("facts", Json::Arr(facts.iter().map(|f| str(f)).collect()))]);
    std::fs::write(memory_file()?, j.to_string()).map_err(|e| e.to_string())
}

/// Forgets the fact at `index` (the clients' remove button); the facts left.
pub fn forget_at(index: usize) -> Result<Vec<String>, String> {
    let mut facts = memory();
    if index >= facts.len() {
        return Err("no such fact".into());
    }
    facts.remove(index);
    save_memory(&facts)?;
    Ok(facts)
}

/// A fact the user adds by hand (the rail's field); the facts.
pub fn add_fact(fact: &str) -> Result<Vec<String>, String> {
    let fact = fact.trim();
    if fact.is_empty() {
        return Err("the fact is empty".into());
    }
    let mut facts = memory();
    if !facts.iter().any(|f| f.trim().eq_ignore_ascii_case(fact)) {
        facts.push(fact.to_string());
        save_memory(&facts)?;
    }
    Ok(facts)
}

// ---- earlier chats: search_chats and read_chat

/// A chat's text as the model reads it: what the user and the assistant said, without tool calls or thinking.
pub(super) fn chat_text(c: &Json) -> Vec<(String, String)> {
    c.get("messages")
        .and_then(Json::as_array)
        .unwrap_or(&[])
        .iter()
        .filter_map(|m| {
            let who = match m.get("role").and_then(Json::as_str)? {
                "user" if is_auto(m) => return None,
                "user" => "User",
                "assistant" => "Assistant",
                _ => return None,
            };
            let text = m.get("content").and_then(Json::as_str)?;
            let text = text.rsplit("</think>").next().unwrap_or(text).trim();
            (!text.is_empty()).then(|| (who.to_string(), text.to_string()))
        })
        .collect()
}

/// The search_chats tool: chats whose title or text has the query's words, best first, with matching lines.
fn search_chats(query: &str, current: &str) -> String {
    let words: Vec<String> = query.to_lowercase().split_whitespace().filter(|w| w.len() > 1).map(String::from).collect();
    if words.is_empty() {
        return "Error: give some words to look for".into();
    }
    let Ok(dir) = chats_dir() else { return "Error: the chats cannot be read".into() };
    let mut hits: Vec<(usize, f64, String)> = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let Some(c) = std::fs::read_to_string(e.path()).ok().and_then(|t| Json::parse(&t).ok()) else { continue };
        let id = c.get("id").and_then(Json::as_str).unwrap_or("").to_string();
        if id == current || id.is_empty() {
            continue;
        }
        let title = c.get("title").and_then(Json::as_str).unwrap_or("").to_string();
        let lines = chat_text(&c);
        let score_of = |t: &str| {
            let t = t.to_lowercase();
            words.iter().filter(|w| t.contains(w.as_str())).count()
        };
        let mut score = 3 * score_of(&title);
        let mut matched = Vec::new();
        for (who, text) in &lines {
            for line in text.lines() {
                let n = score_of(line);
                if n > 0 {
                    score += n;
                    if matched.len() < 3 {
                        matched.push(format!("  {who}: {}", crate::tools::clip(line.trim(), 200)));
                    }
                }
            }
        }
        if score > 0 {
            let at = c.get("updated").and_then(Json::as_f64).unwrap_or(0.0);
            let days = ((now_ms() - at) / 86_400_000.0).floor() as i64;
            let when = match days {
                ..=0 => "today".to_string(),
                1 => "yesterday".to_string(),
                d => format!("{d} days ago"),
            };
            hits.push((score, at, format!("- id {id}: \"{title}\" ({when})\n{}", matched.join("\n"))));
        }
    }
    if hits.is_empty() {
        return format!("No earlier chat mentions {query}.");
    }
    hits.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)));
    let list: Vec<String> = hits.into_iter().take(5).map(|h| h.2).collect();
    format!("{}\n\nRead one with read_chat and its id.", list.join("\n"))
}

/// The read_chat tool: an earlier chat's conversation (its end, when long).
fn read_chat(id: &str) -> String {
    match load_chat(id.trim()) {
        Ok(c) => {
            let title = c.get("title").and_then(Json::as_str).unwrap_or("");
            let text: Vec<String> = chat_text(&c).into_iter().map(|(who, t)| format!("{who}: {t}")).collect();
            crate::tools::clip(&format!("Chat \"{title}\":\n\n{}", text.join("\n\n")), 12000)
        }
        Err(_) => format!("Error: no chat with id {id}"),
    }
}

/// The remember and forget tools; their result for the model.
fn memory_tool(name: &str, args: &Json) -> String {
    let fact = args.get("fact").and_then(Json::as_str).unwrap_or("").trim().to_string();
    if fact.is_empty() {
        return "Error: give the fact".into();
    }
    let mut facts = memory();
    let same = |f: &String| f.trim().eq_ignore_ascii_case(&fact);
    let result = if name == "remember" {
        if facts.iter().any(same) {
            return "Already remembered.".into();
        }
        facts.push(fact);
        "Remembered.".to_string()
    } else {
        let before = facts.len();
        facts.retain(|f| !same(f));
        if facts.len() == before {
            return format!("Error: no remembered fact reads exactly that. The facts are: {}", memory().join(" | "));
        }
        "Forgotten.".to_string()
    };
    match save_memory(&facts) {
        Ok(()) => result,
        Err(e) => format!("Error: {e}"),
    }
}

/// The facts as the system prompt gives them, or nothing.
fn memory_prompt() -> String {
    let facts = memory();
    if facts.is_empty() {
        return String::new();
    }
    format!("What you remember from earlier chats:\n{}", facts.iter().map(|f| format!("- {f}")).collect::<Vec<_>>().join("\n"))
}

// ---- instructions: GHOST.md files, read into every reply's system prompt (as Claude Code reads CLAUDE.md)

/// A folder's instructions are its GHOST.md, else a CLAUDE.md or AGENTS.md written for other agents.
const INSTRUCTION_FILES: [&str; 3] = ["GHOST.md", "CLAUDE.md", "AGENTS.md"];
const INSTRUCTIONS_MAX: usize = 12000; // characters, all the files together

/// The instruction files, each with whether it exists: the global GHOST.md, then one per folder the model may
/// use (the GHOST.md it would have when it has none of the three).
pub fn instruction_files(folders: &[PathBuf]) -> Vec<(PathBuf, bool)> {
    let mut files = Vec::new();
    if let Ok(d) = data_dir() {
        let p = d.join("GHOST.md");
        files.push((p.clone(), p.is_file()));
    }
    for f in folders {
        match INSTRUCTION_FILES.iter().map(|n| f.join(n)).find(|p| p.is_file()) {
            Some(p) => files.push((p, true)),
            None => files.push((f.join("GHOST.md"), false)),
        }
    }
    files
}

pub fn instructions_json(folders: &[PathBuf]) -> Json {
    Json::Arr(
        instruction_files(folders)
            .iter()
            .enumerate()
            .map(|(i, (p, exists))| obj(vec![("path", str(&p.display().to_string())), ("exists", Json::Bool(*exists)), ("scope", str(if i == 0 { "global" } else { "folder" }))]))
            .collect(),
    )
}

/// A file's text without its <!-- comments --> (notes for whoever edits it, not for the model).
pub(super) fn without_comments(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("<!--") {
        out.push_str(&rest[..at]);
        rest = rest[at..].find("-->").map_or("", |end| &rest[at + end + 3..]);
    }
    out.push_str(rest);
    out
}

/// What the system prompt says from the instruction files, or nothing.
fn instructions_prompt(folders: &[PathBuf]) -> String {
    let mut parts = Vec::new();
    let mut left = INSTRUCTIONS_MAX;
    for (path, exists) in instruction_files(folders) {
        let text = if exists { std::fs::read_to_string(&path).unwrap_or_default() } else { String::new() };
        let text = without_comments(text.trim_start_matches('\u{feff}'));
        let text = text.trim();
        if text.is_empty() || left < 200 {
            continue;
        }
        let text = crate::tools::clip(text, left);
        left = left.saturating_sub(text.chars().count());
        parts.push(format!("From {}:\n{text}", path.display()));
    }
    if parts.is_empty() {
        return String::new();
    }
    format!("The user's instructions, from their GHOST.md files. Follow them:\n\n{}", parts.join("\n\n"))
}

// ---- runs: how many are going (the API's requests too), when the last one ended, and the chats they hold
// (two runs in one chat would each save their own copy of it)

static RUNNING: AtomicUsize = AtomicUsize::new(0);
static LAST_END: AtomicU64 = AtomicU64::new(0); // milliseconds since 1970

fn busy_chats() -> &'static Mutex<HashSet<String>> {
    static B: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    B.get_or_init(Default::default)
}

/// A run going (or a request to /v1): counted while it lives, and holding its chat once it has one.
pub(super) struct Going {
    chat: Option<String>,
}

impl Going {
    pub(super) fn new() -> Going {
        RUNNING.fetch_add(1, Ordering::SeqCst);
        Going { chat: None }
    }

    /// Takes the chat, waiting while another run holds it; false if `still_wanted` says to give up.
    fn hold(&mut self, id: &str, mut still_wanted: impl FnMut() -> bool) -> bool {
        loop {
            if busy_chats().lock().unwrap().insert(id.to_string()) {
                self.chat = Some(id.to_string());
                return true;
            }
            if !still_wanted() {
                return false;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }
}

impl Drop for Going {
    fn drop(&mut self) {
        if let Some(id) = &self.chat {
            busy_chats().lock().unwrap().remove(id);
        }
        LAST_END.store(now_ms() as u64, Ordering::SeqCst);
        RUNNING.fetch_sub(1, Ordering::SeqCst);
    }
}

/// One reply on the model at a time: two chats taking turns round by round throw out each other's cache, so
/// each round reads its whole prompt again (two landing pages at once took 25 minutes, 17 of them reading
/// prompts again). A reply waits for the one going to end.
static REPLYING: AtomicBool = AtomicBool::new(false);

/// The model's turn, held while a reply runs.
struct Turn;

impl Drop for Turn {
    fn drop(&mut self) {
        REPLYING.store(false, Ordering::SeqCst);
    }
}

/// Waits for the model's turn, saying so to the client once; None if the client went away meanwhile.
fn take_turn(out: &mut Out) -> Option<Turn> {
    let mut said = false;
    loop {
        if REPLYING.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            return Some(Turn);
        }
        if !said {
            said = true;
            let _ = emit(out, obj(vec![("type", str("waiting")), ("text", str("Waiting for the reply in another chat to finish"))]));
        }
        if !ping(out) {
            return None;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Milliseconds since the last run ended, or None while one is going (scheduled runs wait for quiet).
pub fn idle_ms() -> Option<u64> {
    (RUNNING.load(Ordering::SeqCst) == 0).then(|| (now_ms() as u64).saturating_sub(LAST_END.load(Ordering::SeqCst)))
}

// ---- approvals: the run waits on a channel the approve endpoint answers

fn pending() -> &'static Mutex<HashMap<String, mpsc::Sender<String>>> {
    static P: OnceLock<Mutex<HashMap<String, mpsc::Sender<String>>>> = OnceLock::new();
    P.get_or_init(Default::default)
}

/// The user's answer to approval `id`: "once", "always" or "deny". False if nothing waits for it.
pub fn answer(id: &str, decision: &str) -> bool {
    match pending().lock().unwrap().remove(id) {
        Some(tx) => tx.send(decision.to_string()).is_ok(),
        None => false,
    }
}

// ---- the chat store

fn chats_dir() -> Result<PathBuf, String> {
    let d = std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer").join("chats")).ok_or("LOCALAPPDATA is not set")?;
    std::fs::create_dir_all(&d).map_err(|e| format!("cannot make {}: {e}", d.display()))?;
    Ok(d)
}

fn chat_file(id: &str) -> Result<PathBuf, String> {
    if id.is_empty() || id.len() > 64 || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(format!("{id} is not a chat id"));
    }
    Ok(chats_dir()?.join(format!("{id}.json")))
}

pub(super) fn now_ms() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as f64).unwrap_or(0.0)
}

/// Every saved chat, newest first: id, title, when it last changed.
pub fn list_chats() -> Result<Json, String> {
    let mut chats: Vec<(f64, Json)> = std::fs::read_dir(chats_dir()?)
        .map_err(|e| e.to_string())?
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| Json::parse(&std::fs::read_to_string(e.path()).ok()?).ok())
        .map(|c| {
            let at = c.get("updated").and_then(Json::as_f64).unwrap_or(0.0);
            let unread = Json::Bool(c.get("unread").and_then(Json::as_bool).unwrap_or(false));
            let title = c.get("title").and_then(Json::as_str).map(|t| str(clean_title(t))).unwrap_or(Json::Null);
            let summary = obj(vec![("id", c.get("id").cloned().unwrap_or(Json::Null)), ("title", title), ("updated", num(at)), ("unread", unread)]);
            (at, summary)
        })
        .collect();
    chats.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(Json::Arr(chats.into_iter().map(|(_, c)| c).collect()))
}

pub fn load_chat(id: &str) -> Result<Json, String> {
    let path = chat_file(id)?;
    let text = std::fs::read_to_string(&path).map_err(|_| format!("no chat {id}"))?;
    Json::parse(&text)
}

pub fn delete_chat(id: &str) -> Result<(), String> {
    std::fs::remove_file(chat_file(id)?).map_err(|_| format!("no chat {id}"))?;
    // the outputs it saved go with it (the id was checked by chat_file)
    if let Some(dir) = crate::tools::outputs_dir() {
        let _ = std::fs::remove_dir_all(dir.join(id));
    }
    Ok(())
}

/// The user has seen what a scheduled run added to this chat. Left alone while a run holds the chat (it
/// saves the chat when it ends, unread again).
pub fn mark_read(id: &str) -> Result<(), String> {
    let mut going = Going::new();
    if !going.hold(id, || false) {
        return Ok(());
    }
    let mut chat = Chat::load(id)?;
    if chat.unread {
        chat.unread = false;
        chat.save_as_is()?;
    }
    Ok(())
}

/// A chat in memory: its id, title and messages (OpenAI-style, with a few extra fields the clients read).
struct Chat {
    id: String,
    title: String,
    created: f64,
    updated: f64,
    messages: Vec<Json>,
    tokens: f64, // how much of the context the last round used (the clients show it)
    tasks: Json, // the model's task list (update_tasks), as [{"text","status"}]
    unread: bool, // a scheduled run wrote to it and the user has not opened it since
    cwd: String,  // its working directory, set when it starts (as a Claude Code session's)
    goal: Json,   // its /goal (new_goal), or null
    parts: Json,  // how its last prompt split (prompt_parts), for the clients
    named: bool,  // the model has named it (after its first request that is more than a greeting)
    transient: bool, // a subagent's chat: never saved
}

impl Chat {
    fn new() -> Chat {
        let t = now_ms();
        // time, then a little randomness so two chats made in the same millisecond differ
        let rnd = {
            use std::hash::{BuildHasher, Hasher};
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u64(t as u64);
            h.finish() & 0xffff
        };
        Chat { id: format!("{:x}-{rnd:04x}", t as u64), title: String::new(), created: t, updated: t, messages: Vec::new(), tokens: 0.0, tasks: Json::Arr(Vec::new()), unread: false, cwd: String::new(), goal: Json::Null, parts: Json::Null, named: false, transient: false }
    }

    fn load(id: &str) -> Result<Chat, String> {
        let j = load_chat(id)?;
        Ok(Chat {
            id: id.to_string(),
            title: clean_title(j.get("title").and_then(Json::as_str).unwrap_or("")).to_string(),
            created: j.get("created").and_then(Json::as_f64).unwrap_or(0.0),
            updated: j.get("updated").and_then(Json::as_f64).unwrap_or(0.0),
            messages: j.get("messages").and_then(Json::as_array).map(<[Json]>::to_vec).unwrap_or_default(),
            tokens: j.get("tokens").and_then(Json::as_f64).unwrap_or(0.0),
            tasks: j.get("tasks").cloned().unwrap_or(Json::Arr(Vec::new())),
            unread: j.get("unread").and_then(Json::as_bool).unwrap_or(false),
            cwd: j.get("cwd").and_then(Json::as_str).unwrap_or("").to_string(),
            goal: j.get("goal").cloned().unwrap_or(Json::Null),
            parts: j.get("parts").cloned().unwrap_or(Json::Null),
            // chats saved before this were named after their first reply
            named: j.get("named").and_then(Json::as_bool).unwrap_or_else(|| j.get("messages").and_then(Json::as_array).is_some_and(|m| m.iter().any(|m| role(m) == "assistant"))),
            transient: false,
        })
    }

    fn save(&mut self) -> Result<(), String> {
        self.updated = now_ms();
        self.save_as_is()
    }

    /// Saved without counting as a change (the chat list is newest first, and reading a chat is not news).
    fn save_as_is(&self) -> Result<(), String> {
        if self.transient {
            return Ok(());
        }
        let j = obj(vec![
            ("id", str(&self.id)),
            ("title", str(&self.title)),
            ("created", num(self.created)),
            ("updated", num(self.updated)),
            ("tokens", num(self.tokens)),
            ("tasks", self.tasks.clone()),
            ("unread", Json::Bool(self.unread)),
            ("cwd", str(&self.cwd)),
            ("goal", self.goal.clone()),
            ("parts", self.parts.clone()),
            ("named", Json::Bool(self.named)),
            ("messages", Json::Arr(self.messages.clone())),
        ]);
        let path = chat_file(&self.id)?;
        // written whole to a side file first, so a crash never leaves half a chat
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, j.to_string()).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
    }
}

// ---- the run

/// Maya's DRY penalty: on a 4B model her replies were built from her earlier ones, sentence by sentence ("It's a
/// bit of a joke, but I like that you noticed", "the fact that I'm a voice on someone's computer"). A presence
/// penalty on words made it worse: she kept the sentences and swapped words for tags and filler.
const MAYA_DRY: f32 = 0.8;

/// The most sentences a reply of Maya's has (her prompt asks for one or two; the 4B wrote four or five, the
/// later ones saying again what she had said before in other words).
const MAYA_SENTENCES: usize = 3;

/// What the client asks for: the chat (none: a new one), a new user message (none: answer the last one
/// again), and the reply's settings.
#[derive(Clone)]
struct Request {
    chat: Option<String>,
    message: Option<String>,
    system: String,
    thinking: bool,
    temperature: f32,
    style: Style, // against repeating itself (Maya's calls: a small model rebuilds its earlier replies)
    max_tokens: usize,
    groups: Vec<String>,
    goal: bool,           // keep going until the task list is done (the goal loop)
    mode: String,         // the permission mode (MODES)
    cwd: Option<String>,  // a new chat's working directory
    folders: Vec<PathBuf>, // where its tools may work, the working directory first (in_chat)
    instructions: String, // the GHOST.md files as the reply started
    memory: String, // the remembered facts, likewise
    skills: String, // the installed skills, likewise
    preview: bool,  // the client shows web pages the model writes itself (the app's preview pane)
    sub: String,    // a subagent's: the id of the call that started it, its own calls' ids begun with it ("" for a reply)
    persona: String, // who answers instead of ghost ("persona": "maya", the voice call's companion): her prompt (maya.rs)
    cut: Option<String>, // a call's: the last reply was cut off by the user's voice after these words (it is kept as said)
    auto: bool,          // the message is the call's, not the user's ("they went quiet"): hidden, and not what they said
}

impl Request {
    fn parse(b: &Json, shared: &Shared) -> Request {
        let f = |k: &str| b.get(k).and_then(Json::as_f64);
        let folders = shared.harness.lock().unwrap().folders.clone();
        // Maya has her own memory, and no use for ghost's, its GHOST.md files or its skills
        let maya = b.get("persona").and_then(Json::as_str) == Some("maya");
        Request {
            chat: b.get("chat").and_then(Json::as_str).map(String::from),
            message: b.get("message").and_then(Json::as_str).map(String::from),
            system: b.get("system").and_then(Json::as_str).unwrap_or("").to_string(),
            thinking: b.get("thinking").and_then(Json::as_bool).unwrap_or(shared.think_default),
            temperature: f("temperature").map(|t| t as f32).unwrap_or(shared.defaults.0),
            style: Style {
                presence: f("presence_penalty").map(|p| p as f32).unwrap_or(0.0),
                dry: f("dry_multiplier").map(|d| d as f32).unwrap_or(if maya { MAYA_DRY } else { 0.0 }),
                sentences: f("max_sentences").map(|n| n as usize).unwrap_or(if maya { MAYA_SENTENCES } else { 0 }),
                spoken: maya,
                repetition: 1.0,
            },
            max_tokens: f("max_tokens").map(|n| n.max(1.0) as usize).unwrap_or(shared.defaults.3),
            groups: match b.get("tools").and_then(Json::as_array) {
                Some(a) => a.iter().filter_map(Json::as_str).map(String::from).collect(),
                None => ALL_GROUPS.iter().map(|g| g.to_string()).collect(),
            },
            goal: b.get("keep_going").and_then(Json::as_bool).unwrap_or(false),
            mode: b.get("mode").and_then(Json::as_str).filter(|m| MODES.contains(m)).unwrap_or("manual").to_string(),
            cwd: b.get("cwd").and_then(Json::as_str).map(str::trim).filter(|c| !c.is_empty()).map(String::from),
            instructions: if maya { String::new() } else { instructions_prompt(&folders) },
            folders,
            memory: if maya { String::new() } else { memory_prompt() },
            skills: if maya { String::new() } else { super::skills::prompt() },
            preview: b.get("preview").and_then(Json::as_bool).unwrap_or(false),
            sub: String::new(),
            persona: if maya { super::maya::prompt() } else { String::new() },
            cut: b.get("cut").and_then(Json::as_str).map(String::from),
            auto: b.get("auto").and_then(Json::as_bool).unwrap_or(false),
        }
    }

    /// For a reply in `chat`: its working directory, then the other allowed folders, and their GHOST.md files.
    fn in_chat(&self, chat: &Chat, shared: &Shared) -> Request {
        let folders = run_folders(&chat.cwd, &shared.harness.lock().unwrap().folders);
        let instructions = if self.persona.is_empty() { instructions_prompt(&folders) } else { String::new() };
        Request { instructions, folders, ..self.clone() }
    }
}

/// Where a run's events go: the client that asked for it, or nobody (a scheduled run).
enum Out<'a> {
    Client(&'a mut TcpStream),
    Away,
}

impl Out<'_> {
    fn away(&self) -> bool {
        matches!(self, Out::Away)
    }
}

/// One event to the client; Err when it has gone away. While a subagent runs, the user sees its calls, their
/// results, approvals and pages, and its writing as progress; not its text, task list or context.
fn emit(out: &mut Out, ev: Json) -> Result<(), String> {
    let ev = if SUBAGENT.with(std::cell::Cell::get) == 0 {
        ev
    } else {
        match ev.get("type").and_then(Json::as_str) {
            Some("tool_call" | "tool_result" | "approval" | "preview" | "waiting") => ev,
            Some("text" | "calling") => obj(vec![("type", str("calling")), ("name", str("subagent"))]),
            _ => return Ok(()),
        }
    };
    match out {
        Out::Client(conn) => sse(conn, &ev.to_string()).map_err(|_| "the client went away".to_string()),
        Out::Away => Ok(()),
    }
}

/// A comment line, to notice a client that went away while the run waits on the user.
fn ping(out: &mut Out) -> bool {
    let Out::Client(conn) = out else { return true };
    let c = ": waiting\n\n";
    write!(conn, "{:x}\r\n{c}\r\n", c.len()).and_then(|_| conn.flush()).is_ok()
}

/// POST /api/agent: runs the agent for one reply, streaming events (see the module comment).
pub fn run(conn: &mut TcpStream, body: &Json, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<(), String> {
    let req = Request::parse(body, shared);
    *shared.agent_groups.lock().unwrap() = req.groups.clone(); // what to read in ahead after the next model load
    let mut going = Going::new();
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    conn.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    let mut out = Out::Client(conn);
    // a scheduled task answering in this chat: this message goes after its reply
    if let Some(id) = &req.chat {
        if !going.hold(id, || ping(&mut out)) {
            return Err("the client went away".into());
        }
    }
    let result = (|| {
        let mut chat = match &req.chat {
            Some(id) => Chat::load(id)?,
            None => Chat::new(),
        };
        chat.unread = false;
        // a new chat works in the folder the client picked (else the first allowed one), from then on
        if chat.cwd.is_empty() {
            let picked = req.cwd.clone().filter(|c| std::path::Path::new(c).is_dir());
            chat.cwd = picked.or_else(|| shared.harness.lock().unwrap().folders.first().map(|f| f.display().to_string())).unwrap_or_default();
        }
        let req = req.in_chat(&chat, shared);
        match &req.message {
            Some(m) => {
                // cut off by the user's voice: her last reply is what she got out before it, as the model
                // should remember it
                // (a reply stopped while it was written was not kept: it is added)
                if let Some(said) = &req.cut {
                    let said = said.trim();
                    let kept = if said.is_empty() { "—".to_string() } else { format!("{said} —") };
                    match chat.messages.last_mut() {
                        Some(Json::Obj(fields)) if fields.iter().any(|(k, v)| k == "role" && v.as_str() == Some("assistant")) => {
                            fields.retain(|(k, _)| k != "content");
                            fields.push(("content".to_string(), str(&kept)));
                        }
                        Some(_) => chat.messages.push(obj(vec![("role", str("assistant")), ("content", str(&kept))])),
                        None => {}
                    }
                }
                if req.auto {
                    chat.messages.push(obj(vec![("role", str("user")), ("content", str(m)), ("auto", Json::Bool(true))]));
                    chat.save()?;
                    let Some(_turn) = take_turn(&mut out) else { return Err("the client went away".into()) };
                    return reply(&mut out, &req, &mut chat, false, work, shared, loaded);
                }
                if chat.title.is_empty() {
                    let line = m.trim().lines().next().unwrap_or("");
                    chat.title = if line.chars().count() > 60 { format!("{}…", line.chars().take(57).collect::<String>().trim_end()) } else { line.to_string() };
                }
                // a command: /goal sets the chat's goal and goes on to the model; the others are answered here
                let done = match command(m) {
                    None => None,
                    Some(Ok(Command::Goal(goal))) => {
                        chat.goal = new_goal(&goal);
                        None
                    }
                    Some(Ok(Command::ClearGoal)) => {
                        chat.goal = Json::Null;
                        Some("Goal cleared.".to_string())
                    }
                    Some(Ok(Command::Loop(every, prompt))) => Some(super::schedule::add_loop(&chat.id, &prompt, every, &req.groups, req.thinking).unwrap_or_else(|e| format!("Not looping: {e}."))),
                    Some(Ok(Command::StopLoops)) => Some(match super::schedule::stop_loops(&chat.id) {
                        0 => "This chat has no loop running.".to_string(),
                        1 => "Loop stopped.".to_string(),
                        n => format!("{n} loops stopped."),
                    }),
                    Some(Err(help)) => Some(help),
                };
                if let Some(text) = done {
                    return answer_here(&mut out, &mut chat, m, &text);
                }
                let mut msg = vec![("role", str("user")), ("content", str(m))];
                let notes: Vec<String> = (req.mode == "plan").then(|| PLAN_NOTE.to_string()).into_iter().chain((!req.groups.is_empty()).then(|| task_reminder(&chat)).flatten()).collect();
                if !notes.is_empty() {
                    msg.push(("reminder", str(&notes.join("\n\n"))));
                }
                chat.messages.push(obj(msg));
                // /<skill> anywhere in it (the app's skill chip, or a name typed further on): each skill's
                // instructions loaded before the model starts, as use_skill calls of their own, so the model
                // need not think to load them
                let skills = skill_commands(m);
                if !skills.is_empty() {
                    let loads: Vec<(String, String, String)> = skills.iter().enumerate().map(|(i, name)| (format!("call_skill_{i}"), obj(vec![("name", str(name))]).to_string(), super::skills::use_skill(name, super::skills::skill_chars(loaded.ctx)))).collect();
                    let calls = loads.iter().map(|(id, args, _)| obj(vec![("id", str(id)), ("type", str("function")), ("function", obj(vec![("name", str("use_skill")), ("arguments", str(args))]))])).collect();
                    chat.messages.push(obj(vec![("role", str("assistant")), ("content", str("")), ("tool_calls", Json::Arr(calls))]));
                    for (id, args, result) in &loads {
                        chat.messages.push(obj(vec![("role", str("tool")), ("tool_call_id", str(id)), ("name", str("use_skill")), ("content", str(result))]));
                        emit(&mut out, obj(vec![("type", str("tool_call")), ("id", str(id)), ("name", str("use_skill")), ("arguments", str(args))]))?;
                        emit(&mut out, obj(vec![("type", str("tool_result")), ("id", str(id)), ("name", str("use_skill")), ("result", str(result)), ("denied", Json::Bool(false))]))?;
                    }
                }
            }
            // answer again: drop the reply after the last user message
            None => {
                let last_user = chat.messages.iter().rposition(|m| m.get("role").and_then(Json::as_str) == Some("user") && !is_auto(m)).ok_or("this chat has no message to answer")?;
                chat.messages.truncate(last_user + 1);
                // a summary that came after it (compact) went with the reply: what it stood for is back in
                // the prompt, and is made to fit again
                for m in chat.messages.iter_mut() {
                    if let Json::Obj(fields) = m {
                        fields.retain(|(k, _)| k != "dropped");
                    }
                }
            }
        }
        chat.save()?;
        let first = !chat.named && req.message.as_deref().is_some_and(|m| !is_greeting(m));
        let Some(_turn) = take_turn(&mut out) else { return Err("the client went away".into()) };
        reply(&mut out, &req, &mut chat, first, work, shared, loaded)
    })();
    if let Err(e) = result {
        let _ = emit(&mut out, obj(vec![("type", str("error")), ("message", str(&e))]));
    }
    let _ = sse(conn, "[DONE]");
    let _ = conn.write_all(b"0\r\n\r\n");
    Ok(())
}

/// How a reply's prompt splits, in tokens: the system prompt's pieces and the tools' descriptions (the chat
/// is the rest), as Claude Code's context breakdown shows it.
fn prompt_parts(req: &Request, loaded: &Loaded) -> Json {
    let tools = if req.groups.is_empty() { Vec::new() } else { tools_for(&req.groups) };
    let n = |s: &str| if s.trim().is_empty() { 0.0 } else { loaded.tok.encode(s).len() as f64 };
    let (schemas, guide) = if tools.is_empty() {
        (String::new(), String::new())
    } else {
        let schemas = tools.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("\n");
        (schemas, tool_prompt(&req.folders, tools.iter().any(|t| tool_name(t) == "run_command"), &deferred_tools(&req.groups)))
    };
    obj(vec![
        ("system", num(n(&req.system) + n(&guide))),
        ("tools", num(n(&schemas))),
        ("skills", num(if tools.is_empty() { 0.0 } else { n(&req.skills) })),
        ("instructions", num(n(&req.instructions))),
        ("memory", num(n(&req.memory))),
    ])
}

/// POST /api/chats/trim {"id"}: the chat's tool outputs taken out now, all but the last 3, as Claude Code's
/// "Compact session" (the chat file keeps them; the model gets a note in their place). How many went.
pub fn trim_chat(id: &str) -> Result<usize, String> {
    let mut going = Going::new();
    if !going.hold(id, || false) {
        return Err("a reply is going in that chat".into());
    }
    let mut chat = Chat::load(id)?;
    let outputs: Vec<usize> = (0..chat.messages.len()).filter(|&i| role(&chat.messages[i]) == "tool" && !flag(&chat.messages[i], "trimmed") && !flag(&chat.messages[i], "dropped")).collect();
    let mut n = 0;
    for &i in &outputs[..outputs.len().saturating_sub(3)] {
        if chat.messages[i].get("content").and_then(Json::as_str).is_some_and(|c| c.len() > 400) {
            mark(&mut chat.messages[i], "trimmed", Json::Bool(true));
            n += 1;
        }
    }
    if n > 0 {
        chat.save_as_is()?;
    }
    Ok(n)
}

/// A command's answer, from the harness rather than the model: the message and the answer kept in the chat,
/// and sent as a reply's events.
fn answer_here(out: &mut Out, chat: &mut Chat, message: &str, text: &str) -> Result<(), String> {
    chat.messages.push(obj(vec![("role", str("user")), ("content", str(message))]));
    chat.messages.push(obj(vec![("role", str("assistant")), ("content", str(text))]));
    chat.save()?;
    emit(out, obj(vec![("type", str("chat")), ("id", str(&chat.id)), ("title", str(&chat.title)), ("checkpoint", str(""))]))?;
    if chat.goal != Json::Null || text == "Goal cleared." {
        emit(out, obj(vec![("type", str("goal")), ("goal", chat.goal.clone())]))?;
    }
    emit(out, obj(vec![("type", str("text")), ("text", str(text))]))?;
    emit(out, obj(vec![("type", str("done")), ("finish", str("stop"))]))
}

/// One reply in a chat whose message is in place: the model's rounds, then "done" (or "error"), and for a
/// chat's first reply a title. Err only when the chat cannot be saved.
#[allow(clippy::too_many_arguments)]
fn reply(out: &mut Out, req: &Request, chat: &mut Chat, first: bool, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<(), String> {
    let checkpoint = format!("{}-{}", chat.id, now_ms() as u64); // this reply's file changes, for undo
    emit(out, obj(vec![("type", str("chat")), ("id", str(&chat.id)), ("title", str(&chat.title)), ("checkpoint", str(&checkpoint))]))?;
    if chat.goal != Json::Null {
        emit(out, obj(vec![("type", str("goal")), ("goal", chat.goal.clone())]))?;
    }
    let result = rounds(out, req, chat, &checkpoint, work, shared, loaded);
    chat.save()?;
    // named even when the reply was stopped or failed: the title is about the request, and a chat left
    // unnamed would keep its first line as its title
    let named = first;
    // the model answered with only its current step left open: that step is done
    if matches!(result, Ok("stop")) && close_last_task(chat) {
        chat.save()?;
        let _ = emit(out, obj(vec![("type", str("tasks")), ("tasks", chat.tasks.clone())]));
    }
    let end = match &result {
        Ok(finish) => obj(vec![("type", str("done")), ("finish", str(finish))]),
        Err(e) => obj(vec![("type", str("error")), ("message", str(e))]),
    };
    let _ = emit(out, end);
    // a chat's first reply: the model names it, after "done" so nobody waits for it
    if named {
        chat.named = true;
        if let Some(title) = name_chat(req, chat, work, shared, loaded) {
            chat.title = title;
            chat.save()?;
            let _ = emit(out, obj(vec![("type", str("title")), ("id", str(&chat.id)), ("title", str(&chat.title))]));
        }
    }
    if out.away() {
        result.map(|_| ())
    } else {
        Ok(())
    }
}

/// What a scheduled run is: the message sent to the model (`text`) and the line the clients show for it.
pub struct Wakeup<'a> {
    pub chat: Option<&'a str>, // the chat to answer in (None, or one that is gone: a new one, titled `title`)
    pub title: &'a str,
    pub text: &'a str,
    pub shown: &'a str,
    pub groups: &'a [String],
    pub thinking: bool,
}

/// A reply nobody is watching (a scheduled task), saved in its chat and marked unread
/// unless `keep` says the reply is not worth the user's time (then the exchange is taken out of the chat).
/// Returns the chat's id, the reply's text, and whether it was kept.
pub fn run_away(w: &Wakeup, keep: impl Fn(&str) -> bool, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<(String, String, bool), String> {
    let mut going = Going::new();
    let existing = w.chat.filter(|id| load_chat(id).is_ok());
    let mut chat = Chat::new();
    let id = existing.map_or_else(|| chat.id.clone(), String::from);
    if !going.hold(&id, || false) {
        return Err("a reply is going in that chat".into());
    }
    match existing {
        Some(id) => chat = Chat::load(id)?, // as it is now that it is held
        None => chat.title = w.title.to_string(),
    }
    if chat.cwd.is_empty() {
        chat.cwd = shared.harness.lock().unwrap().folders.first().map(|f| f.display().to_string()).unwrap_or_default();
    }
    let folders = run_folders(&chat.cwd, &shared.harness.lock().unwrap().folders);
    let req = Request {
        chat: Some(chat.id.clone()),
        message: Some(w.text.to_string()),
        system: String::new(),
        thinking: w.thinking,
        temperature: shared.defaults.0,
        style: Style::default(),
        max_tokens: shared.defaults.3.max(ROUND_TOKENS),
        groups: w.groups.to_vec(),
        goal: true,                // nobody is there to say "go on"
        mode: "manual".to_string(), // nobody is there to ask either: only what the user always allows runs
        cwd: None,
        instructions: instructions_prompt(&folders),
        folders,
        memory: memory_prompt(),
        skills: super::skills::prompt(),
        preview: false,
        sub: String::new(),
        persona: String::new(),
        cut: None,
        auto: false,
    };
    let before = chat.messages.len();
    chat.messages.push(obj(vec![("role", str("user")), ("content", str(w.text)), ("shown", str(w.shown))]));
    chat.save()?;
    let _turn = take_turn(&mut Out::Away);
    reply(&mut Out::Away, &req, &mut chat, false, work, shared, loaded)?;
    let text = chat.messages[before..].iter().rev().find(|m| role(m) == "assistant").and_then(|m| m.get("content").and_then(Json::as_str)).unwrap_or("").to_string();
    let kept = keep(&text);
    if kept {
        chat.unread = true;
    } else {
        chat.messages.truncate(before);
    }
    chat.save()?;
    if chat.messages.is_empty() {
        let _ = delete_chat(&chat.id); // made for this run, and nothing of it kept
    }
    Ok((chat.id.clone(), text, kept))
}

/// The prompt for these messages: the client's system text plus where tools may work, then the chat.
fn prompt_for(req: &Request, history: &[Json], tools: &[Json], thinking: bool, shared: &Shared, loaded: &Loaded) -> Result<String, String> {
    let me = if req.persona.is_empty() {
        format!("You are ghost, an assistant running on the user's own computer through ghost-infer, with the {} model. Today is {}.", loaded.id, super::schedule::today())
    } else {
        req.persona.clone()
    };
    let system = [me, req.system.trim().to_string(), req.instructions.clone(), req.memory.clone(), if tools.is_empty() { String::new() } else { tool_prompt(&req.folders, tools.iter().any(|t| tool_name(t) == "run_command"), &deferred_tools(&req.groups)) },if tools.is_empty() { String::new() } else { req.skills.clone() }].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
    let mut messages = Vec::new();
    if !system.is_empty() {
        messages.push(obj(vec![("role", str("system")), ("content", str(&system))]));
    }
    messages.extend(history.iter().cloned());
    let mut fields = vec![("messages", Json::Arr(messages)), ("chat_template_kwargs", obj(vec![("enable_thinking", Json::Bool(thinking))]))];
    if !tools.is_empty() {
        fields.push(("tools", Json::Arr(tools.to_vec())));
    }
    let body = obj(fields);
    chat_prompt_for(loaded, &body, shared.think_default)
}

/// How an agent prompt begins with these tool groups: everything before the first message (the server reads
/// it in after loading a model, so a chat's first reply does not wait for it). The longest start that two
/// prompts with different messages share.
pub fn prompt_opening(groups: &[String], shared: &Shared, loaded: &Loaded) -> Option<String> {
    let body = obj(vec![("tools", Json::Arr(groups.iter().map(|g| str(g)).collect()))]);
    let req = Request::parse(&body, shared);
    let tools = if req.groups.is_empty() { Vec::new() } else { tools_for(&req.groups) };
    let with = |m: &str| prompt_for(&req, &[obj(vec![("role", str("user")), ("content", str(m))])], &tools, req.thinking, shared, loaded).ok();
    let (a, b) = (with("a")?, with("b")?);
    let n = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    // a whole character, not half of one
    let n = (0..=n).rev().find(|&i| a.is_char_boundary(i))?;
    (n > 0).then(|| a[..n].to_string())
}

/// A message that only greets, thanks or asks how things are: a chat is named after the first message that
/// asks for something, as Claude names one after what the user wants.
fn is_greeting(m: &str) -> bool {
    const WORDS: [&str; 34] = [
        "hi", "hello", "hellow", "helo", "hey", "hai", "hiya", "howdy", "yo", "sup", "greetings", "good", "morning", "afternoon", "evening", "night", "there", "ghost", "thanks", "thank", "you", "ok", "okay", "how", "are", "is", "it", "going", "doing", "what's", "whats", "up", "u", "all",
    ];
    let lower = m.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric() && c != '\'').filter(|w| !w.is_empty()).collect();
    !words.is_empty() && words.len() <= 6 && words.iter().all(|w| WORDS.contains(w))
}

/// A short title for a chat after its first request, written by the model; None if it gave nothing usable.
/// The question goes after the chat itself, so the prompt starts as the reply's did and the server's cache
/// already holds it: only the question is read, and the next message finds the chat still cached.
fn name_chat(req: &Request, chat: &Chat, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Option<String> {
    let tools = if req.groups.is_empty() { Vec::new() } else { tools_for(&req.groups) };
    let mut history = for_model(&chat.messages);
    let ask = "Name this chat for the user's chat list: 3 to 7 words in sentence case (a capital only on the first word and on names) saying what the user wants. Reply with JSON only and no tool calls, like {\"title\": \"Fix the login button on mobile\"} or {\"title\": \"Plan a budget trip to Penang\"}. Too vague: {\"title\": \"Chat\"}. Too long: {\"title\": \"Help me find out why the login button does not work on my phone\"}.";
    history.push(obj(vec![("role", str("user")), ("content", str(ask)), ("aside", Json::Bool(true))]));
    let prompt = prompt_for(req, &history, &tools, false, shared, loaded).ok()?;
    let (tx, events) = mpsc::channel();
    let job = Job { model: loaded.id.clone(), kind: Kind::Chat, prompt, max_tokens: Some(48), temperature: 0.0, top_k: shared.defaults.1, top_p: shared.defaults.2, seed: 1, style: Style::default(), stop: Vec::new(), lead: "", events: tx };
    work.send(Work::Job(job)).ok()?;
    let mut out = String::new();
    for ev in events {
        match ev {
            Event::Text(t) => out.push_str(&t),
            Event::Done { .. } => break,
            Event::Error(..) => return None,
        }
    }
    // the title alone: no thinking, quotes, markdown or "Title:" around it
    let out = out.rsplit("</think>").next().unwrap_or(&out);
    // {"title": "..."} as asked; else the first line, cleaned
    let json = out.find('{').zip(out.rfind('}')).filter(|(a, b)| a < b).and_then(|(a, b)| Json::parse(&out[a..=b]).ok());
    let line = json.as_ref().and_then(|j| j.get("title")).and_then(Json::as_str).unwrap_or_else(|| out.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or(""));
    let title = clean_title(line);
    // nothing to name (a greeting), or only a label ("Chat Title"): the first message's opening words instead
    let lower = title.to_lowercase();
    let generic = ["", "chat title", "title", "new chat", "chat", "untitled", "conversation"].contains(&lower.as_str()) || ask.to_lowercase().contains(&lower);
    if generic || title.contains('<') || title.split_whitespace().count() > 10 {
        let first = chat.messages.iter().rev().find(|m| m.get("role").and_then(Json::as_str) == Some("user") && !is_auto(m))?;
        let words: Vec<&str> = first.get("content").and_then(Json::as_str)?.split_whitespace().take(6).collect();
        return (!words.is_empty()).then(|| words.join(" ").chars().take(80).collect());
    }
    Some(title.chars().take(80).collect())
}

/// A title as shown: without a label the model put before it ("Chat Title: ...", "**Title:** ..."), quotes,
/// markdown or a full stop.
fn clean_title(s: &str) -> &str {
    let bare = |s: &str| -> usize { s.len() - s.trim_start_matches(|c: char| c == '*' || c == '#' || c == '"' || c.is_whitespace()).len() };
    let mut s = &s[bare(s)..];
    if let Some(at) = s.to_lowercase().find("title:") {
        // a label of a few words only, so a title that has "title:" in it later stays whole
        if s[..at].split_whitespace().count() <= 2 {
            s = &s[at + "title:".len()..];
        }
    }
    s.trim_matches(|c: char| c == '"' || c == '\'' || c == '*' || c == '#' || c == '`' || c.is_whitespace()).trim_end_matches('.')
}

fn tool_name(t: &Json) -> &str {
    t.get("function").and_then(|f| f.get("name")).and_then(Json::as_str).unwrap_or("")
}

/// The model's rounds for one reply; Ok is how it finished ("stop", "length", or "rounds" at MAX_ROUNDS).
/// A reply about to end with code it changed and nothing run since is sent back, once, to check it; with
/// the goal loop on, one about to end with steps of the task list it keeps in this reply still open is sent
/// back to them.
fn rounds(out: &mut Out, req: &Request, chat: &mut Chat, checkpoint: &str, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<&'static str, String> {
    let reply_start = chat.messages.len();
    let parts = prompt_parts(req, loaded);
    // a subagent's tools: not those SUB_LEFT_OUT keeps from it
    let kept = |t: &Json| req.sub.is_empty() || !SUB_LEFT_OUT.contains(&tool_name(t));
    let tools: Vec<Json> = if req.groups.is_empty() { Vec::new() } else { tools_for(&req.groups) }.into_iter().filter(kept).collect();
    let deferred: Vec<Json> = if req.groups.is_empty() { Vec::new() } else { deferred_tools(&req.groups) }.into_iter().filter(kept).collect();
    // a subagent's calls' ids begin with its own call's, so its approvals and results find their lines
    let prefix = if req.sub.is_empty() { String::new() } else { format!("{}.", req.sub) };
    let mut changed = false;
    let can_run = tools.iter().any(|t| tool_name(t) == "run_command");
    let (mut unchecked, mut nudged) = (false, false); // code changed and nothing run since; the nudge sent
    let mut planned = false; // the task list was written in this reply (an old one is not this reply's goal)
    let (mut goal_nudges, mut done_then) = (0, 0); // goal nudges in a row, and the steps done at the last one
    let mut cut_write: Option<String> = None; // a file write the length limit cut off, not done again since
    for round in 0..MAX_ROUNDS {
        let prompt = fitted_prompt(out, req, chat, &tools, work, shared, loaded)?;
        let lead = super::reply_lead(&prompt);

        let (tx, events) = mpsc::channel();
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
        let job = Job { model: loaded.id.clone(), kind: Kind::Chat, prompt, max_tokens: Some(req.max_tokens), temperature: req.temperature, top_k: shared.defaults.1, top_p: shared.defaults.2, seed, style: req.style, stop: Vec::new(), lead, events: tx };
        work.send(Work::Job(job)).map_err(|_| "the model thread stopped")?;

        // stream it: text to the client as it comes, tool calls split out
        let mut split = ToolCalls::default();
        let (mut text, mut calls) = (String::new(), Vec::new());
        let mut finish = "stop";
        let mut usage = (0usize, 0usize);
        let take = |parts: Vec<Part>, out: &mut Out, text: &mut String, calls: &mut Vec<(String, String, String)>| -> Result<(), String> {
            for part in parts {
                match part {
                    Part::Text(t) => {
                        text.push_str(&t);
                        emit(out, obj(vec![("type", str("text")), ("text", str(&t))]))?;
                    }
                    Part::Call { name, arguments } => {
                        let id = format!("{prefix}call_{}_{}", round, calls.len());
                        emit(out, obj(vec![("type", str("tool_call")), ("id", str(&id)), ("name", str(&name)), ("arguments", str(&arguments))]))?;
                        calls.push((id, name, arguments));
                    }
                    // a call it could not read: kept as a call, so its result tells the model to send it again
                    Part::Bad(raw) => {
                        let id = format!("{prefix}call_{}_{}", round, calls.len());
                        // what it wrote as a string (the prompt shows calls inside tags of their own)
                        let written = raw.trim_start_matches(ToolCalls::OPEN).trim_end_matches(ToolCalls::CLOSE).trim();
                        let arguments = obj(vec![("text", str(written))]).to_string();
                        emit(out, obj(vec![("type", str("tool_call")), ("id", str(&id)), ("name", str(BAD_CALL)), ("arguments", str(&arguments))]))?;
                        calls.push((id, BAD_CALL.to_string(), arguments));
                    }
                }
            }
            Ok(())
        };
        for ev in events {
            match ev {
                // the client going away drops `events`, which stops the generation
                Event::Text(t) => {
                    take(if tools.is_empty() { vec![Part::Text(t)] } else { split.push(&t) }, out, &mut text, &mut calls)?;
                    // a call shows nothing until it is whole (a file can take minutes): a sign of each piece
                    if let Some(name) = split.writing() {
                        let mut ev = vec![("type", str("calling")), ("name", str(name))];
                        if let Some((path, bytes)) = split.writing_path() {
                            ev.push(("bytes", num(bytes as f64)));
                            if let Some(p) = path {
                                ev.push(("path", str(&p)));
                            }
                        }
                        emit(out, obj(ev))?;
                    }
                }
                Event::Done { finish: f, prompt_tokens, completion_tokens } => {
                    finish = f;
                    usage = (prompt_tokens, completion_tokens);
                    break;
                }
                Event::Error(_, e) => return Err(e),
            }
        }
        let open_call = split.writing().is_some();
        take(split.finish(), out, &mut text, &mut calls)?;
        // a call the length limit cut off: its result says so, not that it was written wrong
        let cut = (finish == "length" && open_call).then(|| calls.last().map(|c| c.0.clone())).flatten();
        chat.tokens = (usage.0 + usage.1) as f64;
        chat.parts = parts.clone();
        emit(out, obj(vec![("type", str("context")), ("prompt_tokens", num(usage.0 as f64)), ("completion_tokens", num(usage.1 as f64)), ("context_length", num(loaded.ctx as f64)), ("parts", parts.clone())]))?;

        // the round as the chat keeps it: what the model wrote, and its calls in OpenAI's form
        let mut msg = vec![("role", str("assistant")), ("content", str(text.trim())), ("checkpoint", str(checkpoint))];
        if !calls.is_empty() {
            let tc = calls.iter().map(|(id, name, args)| obj(vec![("id", str(id)), ("type", str("function")), ("function", obj(vec![("name", str(name)), ("arguments", str(args))]))])).collect();
            msg.push(("tool_calls", Json::Arr(tc)));
        }
        // a call cut off did nothing: its text out of the prompt from now on (a page of it filled the context)
        if let Some((_, _, args)) = cut.as_ref().and_then(|id| calls.iter().find(|c| &c.0 == id)) {
            msg.push(("args_trimmed", Json::Bool(true)));
            let raw = Json::parse(args).ok().and_then(|a| a.get("text").and_then(Json::as_str).map(String::from)).unwrap_or_default();
            if raw.contains("write_file") || raw.contains("edit_file") {
                cut_write = Some(cut_path(&raw).unwrap_or_else(|| "the file".into()));
            }
        }
        chat.messages.push(obj(msg));
        // the reply about to end with a cut-off write never done again (a model can lose track of it, as one
        // did after a compaction: it reviewed the old page instead): sent back to it, once
        if calls.is_empty() && round + 1 < MAX_ROUNDS {
            if let Some(path) = cut_write.take() {
                let say = format!("(Automatic check from ghost, not a message from the user) Your write of {path} was cut off earlier and never done again, so the file does not have it. Write it now, in parts (write_file with the first part, then write_file with \"append\": true for each further part), or say in one line why not.");
                chat.messages.push(obj(vec![("role", str("user")), ("content", str(&say)), ("auto", Json::Bool(true)), ("goal", Json::Bool(true))]));
                emit(out, obj(vec![("type", str("nudge")), ("kind", str("goal")), ("text", str(&say))]))?;
                mark_changed(chat, reply_start, changed);
                chat.save()?;
                continue;
            }
        }
        if calls.is_empty() && unchecked && can_run && !nudged && round + 1 < MAX_ROUNDS {
            nudged = true;
            chat.messages.push(obj(vec![("role", str("user")), ("content", str(NUDGE)), ("auto", Json::Bool(true))]));
            emit(out, obj(vec![("type", str("nudge")), ("kind", str("check")), ("text", str(NUDGE))]))?;
            mark_changed(chat, reply_start, changed);
            chat.save()?;
            continue;
        }
        // the goal-based loop (/goal): stopping is checked against the user's goal; not met, back to work
        if calls.is_empty() && goal_active(chat) && round + 1 < MAX_ROUNDS {
            emit(out, obj(vec![("type", str("goal")), ("goal", chat.goal.clone()), ("checking", Json::Bool(true))]))?;
            let (met, reason) = check_goal(req, chat, work, shared, loaded);
            let f = |k: &str| chat.goal.get(k).and_then(Json::as_f64).unwrap_or(0.0) as usize;
            let (tries, limit) = (f("tries") + 1, f("limit").max(1));
            let state = if met { "met" } else if tries >= limit { "stopped" } else { "active" };
            mark(&mut chat.goal, "tries", num(tries as f64));
            mark(&mut chat.goal, "state", str(state));
            mark(&mut chat.goal, "reason", str(&reason));
            emit(out, obj(vec![("type", str("goal")), ("goal", chat.goal.clone())]))?;
            if state == "active" {
                let goal = chat.goal.get("text").and_then(Json::as_str).unwrap_or("").to_string();
                let say = format!("(Check from ghost, not a message from the user) The goal is not met yet: {reason}. Keep working toward it: {goal}");
                chat.messages.push(obj(vec![("role", str("user")), ("content", str(&say)), ("auto", Json::Bool(true)), ("goal", Json::Bool(true))]));
                emit(out, obj(vec![("type", str("nudge")), ("kind", str("goal")), ("text", str(&say))]))?;
                mark_changed(chat, reply_start, changed);
                chat.save()?;
                continue;
            }
            chat.save()?;
        }
        // the task-list loop (scheduled runs, nobody there to say go on): open steps go back to the model,
        // while it keeps getting steps done
        let open = if calls.is_empty() && planned && req.goal { open_tasks(chat) } else { Vec::new() };
        if !open.is_empty() && round + 1 < MAX_ROUNDS {
            let done = chat.tasks.as_array().map_or(0, |t| t.iter().filter(|t| t.get("status").and_then(Json::as_str) == Some("done")).count());
            if done > done_then {
                goal_nudges = 0;
            }
            if goal_nudges < GOAL_NUDGES {
                (goal_nudges, done_then) = (goal_nudges + 1, done);
                let say = goal_nudge(&open);
                chat.messages.push(obj(vec![("role", str("user")), ("content", str(&say)), ("auto", Json::Bool(true)), ("goal", Json::Bool(true))]));
                emit(out, obj(vec![("type", str("nudge")), ("kind", str("goal")), ("text", str(&say))]))?;
                mark_changed(chat, reply_start, changed);
                chat.save()?;
                continue;
            }
        }
        if calls.is_empty() {
            mark_changed(chat, reply_start, changed);
            return Ok(if finish == "length" { "length" } else { "stop" });
        }

        for (id, name, arguments) in calls {
            planned |= name == "update_tasks";
            let known = name == BAD_CALL || tools.iter().chain(&deferred).any(|t| tool_name(t) == name);
            let unloaded = deferred.iter().find(|t| tool_name(t) == name && takes_arguments(t) && !loaded_in(chat, &name));
            let (result, denied) = match name.as_str() {
                _ if !known => (format!("Error: there is no tool named {name} here. Use the tools the system prompt lists; the others are not turned on."), false),
                // as Claude Code refuses a deferred tool called before it was loaded; its description comes with
                // the refusal, so the next call can be right without a tool_search round
                _ if unloaded.is_some() => (format!("Error: {name} was called before its description was loaded, so nothing was run. Its description:
{}
Call it again with these parameters.", unloaded.unwrap().to_string()), false),
                "tool_search" => (tool_search(&deferred, &arguments), false),
                "update_tasks" => (update_tasks(out, chat, &arguments)?, false),
                "subagent" => (subagent(out, &id, &arguments, chat, checkpoint, req, work, shared, loaded)?, false),
                BAD_CALL if cut.as_ref() == Some(&id) => (cut_call_result(usage.1), false),
                BAD_CALL => (bad_call_result(&arguments), false),
                _ => {
                    RUN_SECS.with(|c| c.set(-1.0));
                    call(out, &id, &name, &arguments, chat, reply_start, checkpoint, req, shared, loaded)?
                }
            };
            let result = spill(result, &name, &chat.id, &id, loaded.ctx);
            let secs = RUN_SECS.with(|c| c.replace(-1.0));
            let secs = (secs >= 0.0).then(|| num((secs * 10.0).round() / 10.0));
            if !denied && approval_group(&name) == Some("edit") && !result.starts_with("Error") {
                changed = true;
                cut_write = None; // done again
                unchecked |= Json::parse(&arguments).ok().and_then(|a| a.get("path").and_then(Json::as_str).map(is_code)).unwrap_or(false);
            }
            if !denied && approval_group(&name) == Some("cmd") {
                unchecked = false;
            }
            let mut ev = vec![("type", str("tool_result")), ("id", str(&id)), ("name", str(&name)), ("result", str(&result)), ("denied", Json::Bool(denied))];
            let mut m = vec![("role", str("tool")), ("tool_call_id", str(&id)), ("name", str(&name)), ("content", str(&result))];
            if let Some(secs) = secs {
                ev.push(("secs", secs.clone()));
                m.push(("secs", secs));
            }
            emit(out, obj(ev))?;
            if denied {
                m.push(("denied", Json::Bool(true)));
            }
            chat.messages.push(obj(m));
        }
        mark_changed(chat, reply_start, changed);
        chat.save()?;
    }
    Ok("rounds")
}

/// The update_tasks tool: the chat's new task list, sent to the client as a tasks event. Its result for the model.
fn update_tasks(out: &mut Out, chat: &mut Chat, arguments: &str) -> Result<String, String> {
    let args = Json::parse(arguments).unwrap_or(Json::Null);
    let Some(list) = args.get("tasks").and_then(Json::as_array) else {
        return Ok("Error: give tasks, a list of {\"text\", \"status\"}".into());
    };
    let tasks: Vec<Json> = list
        .iter()
        .filter_map(|t| {
            let text = t.get("text").and_then(Json::as_str)?.trim();
            let status = match t.get("status").and_then(Json::as_str).unwrap_or("pending") {
                s @ ("in_progress" | "done") => s,
                _ => "pending",
            };
            (!text.is_empty()).then(|| obj(vec![("text", str(text)), ("status", str(status))]))
        })
        .collect();
    let done = tasks.iter().filter(|t| t.get("status").and_then(Json::as_str) == Some("done")).count();
    let total = tasks.len();
    chat.tasks = Json::Arr(tasks);
    emit(out, obj(vec![("type", str("tasks")), ("tasks", chat.tasks.clone())]))?;
    Ok(format!("Task list updated: {done} of {total} done."))
}

/// Marks the task list's one in_progress task done when every other task is done; whether it did.
fn close_last_task(chat: &mut Chat) -> bool {
    let Json::Arr(tasks) = &mut chat.tasks else { return false };
    let status = |t: &Json| t.get("status").and_then(Json::as_str).unwrap_or("").to_string();
    let open: Vec<usize> = (0..tasks.len()).filter(|&i| status(&tasks[i]) != "done").collect();
    let [i] = open[..] else { return false };
    if status(&tasks[i]) != "in_progress" {
        return false;
    }
    let text = tasks[i].get("text").and_then(Json::as_str).unwrap_or("").to_string();
    tasks[i] = obj(vec![("text", str(&text)), ("status", str("done"))]);
    true
}

/// Notes on the reply's last assistant message whether it changed files (the clients offer undo then).
fn mark_changed(chat: &mut Chat, reply_start: usize, changed: bool) {
    if !changed {
        return;
    }
    let last = chat.messages[reply_start..].iter().rposition(|m| m.get("role").and_then(Json::as_str) == Some("assistant"));
    for (i, m) in chat.messages[reply_start..].iter_mut().enumerate() {
        if let Json::Obj(fields) = m {
            fields.retain(|(k, _)| k != "changed");
            if Some(i) == last {
                fields.push(("changed".into(), Json::Bool(true)));
            }
        }
    }
}

thread_local! {
    /// How long the last tool `call` took to run (not counting the wait for the user's OK); negative: it ran
    /// nothing. The clients show it for commands, as Claude's background tasks do.
    static RUN_SECS: std::cell::Cell<f64> = const { std::cell::Cell::new(-1.0) };
}

/// One tool call: refused if it repeats itself, put to the user if its group needs an OK, then run.
/// Returns its result for the model and whether the user said no.
#[allow(clippy::too_many_arguments)]
/// What a subagent does without: scheduling, memory, earlier chats and subagents of its own.
const SUB_LEFT_OUT: [&str; 8] = ["subagent", "schedule_task", "list_schedules", "cancel_schedule", "remember", "forget", "search_chats", "read_chat"];

/// Put before a subagent's task: Claude Code's brief for its general-purpose agent.
const SUB_INTRO: &str = "(From ghost, not from the user) You are a subagent: the main agent has handed you the task below. Do it fully with your tools; do not gold-plate it, but do not leave it half-done. You do not see the main chat or the user, and the main agent sees only your last message: end with a concise report of what you did and found, with the paths and specifics it needs, and nothing else.";

thread_local! {
    /// Subagents running on this thread (a reply's subagent call runs on its thread): their events are what
    /// the user sees of them (emit).
    static SUBAGENT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The subagent tool, as Claude Code's Agent tool: the task done in a fresh chat of its own (the reply's tools
/// but those SUB_LEFT_OUT keeps, the chat's folder and permission mode), its file changes under the reply's
/// checkpoint, and only its last message back as the result, so its reading never fills the main chat's
/// context. Its calls and approvals show in the reply; its writing counts as progress (emit).
#[allow(clippy::too_many_arguments)]
fn subagent(out: &mut Out, id: &str, arguments: &str, parent: &Chat, checkpoint: &str, req: &Request, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<String, String> {
    if !req.sub.is_empty() {
        return Ok("Error: a subagent cannot start another one; do this part yourself.".into());
    }
    let args = Json::parse(arguments).unwrap_or(Json::Null);
    let task = args.get("prompt").and_then(Json::as_str).unwrap_or("").trim();
    if task.is_empty() {
        return Ok("Error: subagent needs \"prompt\": the task, with everything needed to do it.".into());
    }
    // the main chat's id, so the outputs it saves (spill) go with the main chat when that is deleted
    let mut chat = Chat { id: parent.id.clone(), cwd: parent.cwd.clone(), transient: true, ..Chat::new() };
    chat.messages.push(obj(vec![("role", str("user")), ("content", str(&format!("{SUB_INTRO}\n\nThe task:\n{task}")))]));
    let sub = Request { sub: id.to_string(), chat: None, message: None, goal: false, ..req.clone() };
    SUBAGENT.with(|d| d.set(d.get() + 1));
    let done = rounds(out, &sub, &mut chat, checkpoint, work, shared, loaded);
    SUBAGENT.with(|d| d.set(d.get() - 1));
    let finish = done?;
    let last = chat.messages.iter().rev().find(|m| role(m) == "assistant").and_then(|m| m.get("content").and_then(Json::as_str)).unwrap_or("");
    let report = last.rsplit("</think>").next().unwrap_or(last).trim();
    let note = match finish {
        "length" => " (it stopped at the length limit)",
        "rounds" => " (it stopped after its most rounds)",
        _ => "",
    };
    Ok(if report.is_empty() { format!("The subagent finished without a report{note}.") } else { format!("(The subagent's report{note}; the user has not seen it.)\n{report}") })
}

/// A read_file of lines the prompt still holds, unchanged since (no write or edit of the file after that read,
/// and its output not taken out): a line saying so in place of the lines again, as Claude Code answers an
/// unchanged re-read. One landing page's run read the same 300 lines of it three times.
fn unchanged_read(chat: &Chat, this: &str, arguments: &str) -> Option<String> {
    let path = |a: &Json| a.get("path").and_then(Json::as_str).map(|p| p.replace('\\', "/").to_lowercase());
    let range = |a: &Json| (a.get("offset").and_then(Json::as_f64).map_or(1, |o| o.max(1.0) as i64), a.get("limit").and_then(Json::as_f64).map(|l| l as i64));
    let want = Json::parse(arguments).ok()?;
    let file = path(&want)?;
    let msgs = &chat.messages;
    for (i, m) in msgs.iter().enumerate().rev() {
        if flag(m, "dropped") {
            break; // left out: what is before it too
        }
        // (this call is in the chat already, its output not yet)
        for c in m.get("tool_calls").and_then(Json::as_array).unwrap_or(&[]).iter().rev().filter(|c| c.get("id").and_then(Json::as_str) != Some(this)) {
            let Some(f) = c.get("function") else { continue };
            let args = f.get("arguments").and_then(Json::as_str).and_then(|a| Json::parse(a).ok()).unwrap_or(Json::Null);
            if path(&args).as_ref() != Some(&file) {
                continue;
            }
            match f.get("name").and_then(Json::as_str) {
                Some("write_file" | "edit_file") => return None, // changed since
                Some("read_file") if range(&args) == range(&want) => {
                    let call_id = c.get("id").and_then(Json::as_str)?;
                    let out = msgs[i + 1..].iter().find(|r| r.get("tool_call_id").and_then(Json::as_str) == Some(call_id))?;
                    let text = out.get("content").and_then(Json::as_str).unwrap_or("");
                    if flag(out, "trimmed") || out.get("clip").is_some() || text.starts_with("Error") || text.starts_with("(Unchanged") {
                        return None;
                    }
                    return Some("(Unchanged since you read these lines above: they are still in this chat, so they are not sent again. Use them from there.)".into());
                }
                _ => {}
            }
        }
    }
    None
}

fn call(out: &mut Out, id: &str, name: &str, arguments: &str, chat: &Chat, reply_start: usize, checkpoint: &str, req: &Request, shared: &Shared, loaded: &Loaded) -> Result<(String, bool), String> {
    if name == "read_file" {
        if let Some(note) = unchanged_read(chat, id, arguments) {
            return Ok((note, false));
        }
    }
    // the same call a third time in one reply: a small model going round in circles
    let same = chat.messages[reply_start..]
        .iter()
        .filter_map(|m| m.get("tool_calls").and_then(Json::as_array))
        .flatten()
        .filter(|c| c.get("function").is_some_and(|f| f.get("name").and_then(Json::as_str) == Some(name) && f.get("arguments").and_then(Json::as_str) == Some(arguments)))
        .count();
    if same > 2 {
        return Ok(("You already made this exact call twice; its result is above. Do something different, or answer with what you have.".into(), false));
    }
    if let Some(group) = approval_group(name) {
        let ask = match gate(&req.mode, group, &Json::parse(arguments).unwrap_or(Json::Null)) {
            Gate::Run => false,
            Gate::Refuse(why) => return Ok((why.to_string(), true)),
            Gate::Ask => !shared.harness.lock().unwrap().always.iter().any(|g| g == group),
        };
        if ask {
            // a scheduled run has nobody to ask: only what the user always allows runs in it
            if out.away() {
                return Ok((format!("Not done: this needs the user's OK, and this is a scheduled run with nobody here to give it (they can allow {group} tools always). Do what you can without it, and say in your reply what needs their OK."), true));
            }
            let (tx, rx) = mpsc::channel();
            pending().lock().unwrap().insert(id.to_string(), tx);
            emit(out, obj(vec![("type", str("approval")), ("id", str(id)), ("name", str(name)), ("arguments", str(arguments)), ("group", str(group))]))?;
            let decision = loop {
                match rx.recv_timeout(Duration::from_secs(2)) {
                    Ok(d) => break d,
                    Err(mpsc::RecvTimeoutError::Timeout) if ping(out) => continue,
                    _ => {
                        pending().lock().unwrap().remove(id);
                        return Err("the client went away".into());
                    }
                }
            };
            match decision.as_str() {
                "always" => {
                    let mut s = shared.harness.lock().unwrap();
                    if !s.always.iter().any(|g| g == group) {
                        s.always.push(group.to_string());
                        let _ = s.save();
                    }
                }
                "once" => {}
                _ => return Ok(("The user declined this action. Do not try it again. If there is no other way to proceed, explain the situation clearly and stop.".into(), true)),
            }
        }
    }
    if name == "use_skill" || name == "install_skill" {
        let args = Json::parse(arguments).unwrap_or(Json::Null);
        let s = |k: &str| args.get(k).and_then(Json::as_str).unwrap_or("").to_string();
        return Ok((if name == "use_skill" { super::skills::use_skill(&s("name"), super::skills::skill_chars(loaded.ctx)) } else { super::skills::install_tool(&s("source"), &s("skill")) }, false));
    }
    if name == "search_chats" || name == "read_chat" {
        let args = Json::parse(arguments).unwrap_or(Json::Null);
        let s = |k: &str| args.get(k).and_then(Json::as_str).unwrap_or("").to_string();
        return Ok((if name == "search_chats" { search_chats(&s("query"), &chat.id) } else { read_chat(&s("id")) }, false));
    }
    if matches!(name, "schedule_task" | "list_schedules" | "cancel_schedule") {
        let args = Json::parse(arguments).unwrap_or(Json::Null);
        let always = shared.harness.lock().unwrap().always.clone();
        return Ok((super::schedule::tool(name, &args, &chat.id, &req.groups, req.thinking, &always), false));
    }
    if name.starts_with("mcp__") {
        let args = if arguments.trim().is_empty() { Json::Obj(Vec::new()) } else { Json::parse(arguments).unwrap_or(Json::Null) };
        return Ok((super::mcp::call(name, &args), false));
    }
    if name == "remember" || name == "forget" {
        return Ok((memory_tool(name, &Json::parse(arguments).unwrap_or(Json::Null)), false));
    }
    if name == "get_system_info" {
        let first = vec![
            ("inference_engine", str(concat!("ghost-infer ", env!("CARGO_PKG_VERSION"), ", running locally on this computer"))),
            ("model_loaded", str(&loaded.id)),
            ("model_runs_on", str(&loaded.backend)),
            ("context_length", num(loaded.ctx as f64)),
        ];
        return Ok((crate::sysinfo::report(first).to_string(), false));
    }
    let args = if arguments.trim().is_empty() { Json::Obj(Vec::new()) } else { Json::parse(arguments).unwrap_or(Json::Null) };
    if args == Json::Null {
        return Ok((format!("Error: invalid arguments for {name}: not valid JSON. Arguments given: {}. Fix them and call the tool again.", crate::tools::clip(arguments, 300)), false));
    }
    // a copy of the settings: a command may run for minutes, and the settings must stay usable meanwhile;
    // the chat's own folders, its working directory first
    let settings = crate::tools::Settings { folders: req.folders.clone(), ..shared.harness.lock().unwrap().clone() };
    let t = std::time::Instant::now();
    // whole: one too long for the context is saved to a file after (spill)
    let result = crate::tools::run_full(name, &args, &settings, checkpoint).unwrap_or_else(|e| format!("Error: {e}"));
    RUN_SECS.with(|c| c.set(t.elapsed().as_secs_f64()));
    let result = match show_page(out, req.preview, name, &args, &settings, &result) {
        Some(note) => format!("{result}\n{note}"),
        None => result,
    };
    Ok((result, false))
}

/// A web page the model just wrote, shown to the user as it is made, without asking for it: an .html file
/// itself (served by this server, see preview_url), and for a .css or .js file the page last shown, again. The
/// app shows it in its preview pane (a "preview" event); another client gets it in a browser MCP server's
/// window, if one has a browser_navigate tool (as Playwright's has). A note for the model if it was shown.
fn show_page(out: &mut Out, client_shows: bool, name: &str, args: &Json, settings: &crate::tools::Settings, result: &str) -> Option<String> {
    static LAST: Mutex<Option<String>> = Mutex::new(None);
    if !matches!(name, "write_file" | "edit_file") || result.starts_with("Error") {
        return None;
    }
    let path = settings.resolve(args.get("path")?.as_str()?).ok()?;
    let url = match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "html" | "htm" => super::preview_url(&path)?,
        "css" | "js" => LAST.lock().unwrap().clone()?,
        _ => return None,
    };
    let note = if client_shows {
        emit(out, obj(vec![("type", str("preview")), ("url", str(&url))])).ok()?;
        format!("(ghost is showing {url} to the user in the app's preview pane, and shows it again after each change: there is no need to open it.)")
    } else {
        let tool = super::mcp::find_tool("browser_navigate")?;
        if super::mcp::call(&tool, &obj(vec![("url", str(&url))])).starts_with("Error") {
            return None;
        }
        format!("(ghost opened {url} in the browser for the user to see; the browser tools can look at it.)")
    };
    *LAST.lock().unwrap() = Some(url);
    Some(note)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_output_is_saved_to_a_file() {
        let (chat, long) = ("spill-test-0001", "x".repeat(70) + "\n");
        let output = long.repeat(200); // 14,200 characters: more than 8,000 in a 64K context
        let got = spill(output.clone(), "run_command", chat, "call_1_0", 65536);
        let path = got.split("Full output saved to: ").nth(1).and_then(|r| r.lines().next()).unwrap().to_string();
        assert!(got.starts_with("<persisted-output>") && got.contains("14200 characters, 200 lines"), "{got}");
        assert!(got.len() < 3000);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), output);
        // short enough, or a file the model read: as it is (cut only past MAX_RESULT)
        assert_eq!(spill(long.clone(), "run_command", chat, "call_1_1", 65536), long);
        assert_eq!(spill(output.clone(), "read_file", chat, "call_1_2", 65536), output);
        let _ = std::fs::remove_dir_all(crate::tools::outputs_dir().unwrap().join(chat));
    }

    #[test]
    fn an_unchanged_reread_is_not_sent_again() {
        let call = |id: &str, name: &str, args: &str| obj(vec![("role", str("assistant")), ("content", str("")), ("tool_calls", Json::Arr(vec![obj(vec![("id", str(id)), ("type", str("function")), ("function", obj(vec![("name", str(name)), ("arguments", str(args))]))])]))]);
        let result = |id: &str, text: &str| obj(vec![("role", str("tool")), ("tool_call_id", str(id)), ("content", str(text))]);
        let read = r#"{"path": "D:\\site\\index.html", "offset": 1, "limit": 300}"#;
        let mut chat = Chat::new();
        chat.messages = vec![obj(vec![("role", str("user")), ("content", str("make a page"))]), call("c1", "read_file", read), result("c1", "1\t<html>"), call("c2", "read_file", r#"{"path": "D:/site/index.html", "limit": 300}"#)];
        // the same lines (the same path written either way, offset 1 as none), still in the prompt
        assert!(unchanged_read(&chat, "c2", r#"{"path": "D:/site/index.html", "limit": 300}"#).is_some());
        // other lines, or the file written since, or its output taken out: read again
        assert!(unchanged_read(&chat, "c2", r#"{"path": "D:/site/index.html", "offset": 300}"#).is_none());
        let mut written = chat.messages.clone();
        written.insert(3, call("w1", "write_file", r#"{"path": "D:/site/index.html", "content": "x"}"#));
        assert!(unchanged_read(&Chat { messages: written, ..Chat::new() }, "c2", read).is_none());
        mark(&mut chat.messages[2], "trimmed", Json::Bool(true));
        assert!(unchanged_read(&chat, "c2", read).is_none());
    }

    #[test]
    fn a_cut_off_write_and_its_arguments() {
        // the file a cut-off call was writing, in each form a model writes calls in
        assert_eq!(cut_path("<function=write_file>\n<parameter=path>\nD:\\site\\index.html\n</parameter>\n<parameter=content>\n<!DOCTYPE").as_deref(), Some("D:\\site\\index.html"));
        assert_eq!(cut_path("{\"name\": \"write_file\", \"arguments\": {\"path\": \"D:\\\\site\\\\a.html\", \"content\": \"<ht").as_deref(), Some("D:\\site\\a.html"));
        assert_eq!(cut_path("write_file<arg_key>path</arg_key><arg_value>a.html</arg_value><arg_key>content"), Some("a.html".into()));
        assert_eq!(cut_path("<function=write_file>\n<parameter=content>\n<html>"), None);
        // its long text taken out, the rest kept
        let args = obj(vec![("path", str("a.html")), ("content", str(&"<p>x</p>\n".repeat(100)))]).to_string();
        let cut = Json::parse(&trimmed_args("write_file", &args)).unwrap();
        assert_eq!(cut.get("path").and_then(Json::as_str), Some("a.html"));
        assert!(cut.get("content").and_then(Json::as_str).unwrap().starts_with("[100 lines, taken out"));
    }

    #[test]
    fn greetings_wait_for_a_request() {
        for m in ["hi there.", "Hello!", "hey ghost", "good morning", "how are you", "thanks!", "what's up"] {
            assert!(is_greeting(m), "{m}");
        }
        for m in ["write a nursery rhyme song", "Hello from the other side", "what is your name", "hi, check my disk space", ""] {
            assert!(!is_greeting(m), "{m}");
        }
    }

    #[test]
    fn goal_limits_and_loop_intervals() {
        assert_eq!(goal_limit("all tests pass, stop after 5 tries"), 5);
        assert_eq!(goal_limit("the page loads"), GOAL_TRIES);
        assert_eq!(interval_minutes("10m"), Some(10));
        assert_eq!(interval_minutes("2h"), Some(120));
        assert_eq!(interval_minutes("1m"), Some(5));
        assert_eq!(interval_minutes("soon"), None);
    }
}
