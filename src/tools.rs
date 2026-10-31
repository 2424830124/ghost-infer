//! The agent tools the chat page offers the model, run here because a browser page cannot touch files:
//! reading, listing and searching files inside the folders the user allows. The page decides what needs
//! the user's approval; this side checks that every path stays inside an allowed folder and keeps every
//! result short enough for a small context.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::json::{num, obj, str, Json};

/// Characters a tool result may hand the model; longer ones keep their start and end.
pub const MAX_RESULT: usize = 16_000;
/// Lines read_file returns when the model does not say, and the longest line it shows whole.
const READ_LINES: usize = 300;
const MAX_LINE: usize = 400;
/// Entries list_files and search_files return at most.
const MAX_ENTRIES: usize = 300;
/// Folders a listing never walks into.
const SKIP_DIRS: [&str; 7] = [".git", "node_modules", "target", "__pycache__", ".venv", "venv", ".next"];

/// The harness settings, kept in %LOCALAPPDATA%\ghost-infer\harness.json: the folders tools may use (the
/// first is where relative paths start).
#[derive(Clone)]
pub struct Settings {
    pub folders: Vec<PathBuf>,
    /// tool groups the user said to always allow (edit, cmd, web), for clients that run the agent here
    pub always: Vec<String>,
    /// a few short replies right after a model loads, so its weights are in memory before the first chat
    pub warm_up: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings { folders: Vec::new(), always: Vec::new(), warm_up: true }
    }
}

/// Where installed skills live (%LOCALAPPDATA%\ghost-infer\skills): the file tools may read there, and
/// commands may run there (for a skill's scripts), but nothing is written there except by installing.
pub fn skills_dir() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer").join("skills"))
}

/// Where tool outputs too long for the model's context are saved, a folder per chat
/// (%LOCALAPPDATA%\ghost-infer\outputs): the file tools may read there, as in the skills folder.
pub fn outputs_dir() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer").join("outputs"))
}

fn settings_file() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer").join("harness.json"))
}

impl Settings {
    pub fn load() -> Settings {
        let Some(text) = settings_file().and_then(|p| std::fs::read_to_string(p).ok()) else { return Settings::default() };
        let folders = Json::parse(&text)
            .ok()
            .and_then(|j| j.get("folders").and_then(Json::as_array).map(|a| a.iter().filter_map(Json::as_str).map(PathBuf::from).collect()))
            .unwrap_or_default();
        let always = Json::parse(&text)
            .ok()
            .and_then(|j| j.get("always").and_then(Json::as_array).map(|a| a.iter().filter_map(Json::as_str).map(String::from).collect()))
            .unwrap_or_default();
        let warm_up = Json::parse(&text).ok().and_then(|j| j.get("warm_up").and_then(Json::as_bool)).unwrap_or(true);
        Settings { folders, always, warm_up }
    }

    pub fn save(&self) -> Result<(), String> {
        let path = settings_file().ok_or("LOCALAPPDATA is not set")?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        std::fs::write(&path, self.to_json().to_string()).map_err(|e| format!("cannot save {}: {e}", path.display()))
    }

    pub fn to_json(&self) -> Json {
        obj(vec![
            ("folders", Json::Arr(self.folders.iter().map(|f| str(&f.display().to_string())).collect())),
            ("always", Json::Arr(self.always.iter().map(|g| str(g)).collect())),
            ("warm_up", Json::Bool(self.warm_up)),
        ])
    }

    /// Settings from a client: folders that exist, as given (duplicates dropped); "always" and "warm_up" if
    /// sent, else kept.
    pub fn from_json(j: &Json, old: &Settings) -> Result<Settings, String> {
        let mut folders: Vec<PathBuf> = Vec::new();
        for f in j.get("folders").and_then(Json::as_array).ok_or("send {\"folders\": [...]}")? {
            let f = f.as_str().ok_or("folders are strings")?.trim();
            if f.is_empty() {
                continue;
            }
            let p = PathBuf::from(f);
            if !p.is_dir() {
                return Err(format!("{f} is not a folder"));
            }
            if !folders.contains(&p) {
                folders.push(p);
            }
        }
        let always = match j.get("always").and_then(Json::as_array) {
            Some(a) => a.iter().filter_map(Json::as_str).filter(|g| matches!(*g, "edit" | "cmd" | "web" | "mcp")).map(String::from).collect(),
            None => old.always.clone(),
        };
        let warm_up = j.get("warm_up").and_then(Json::as_bool).unwrap_or(old.warm_up);
        Ok(Settings { folders, always, warm_up })
    }

    /// `path` (absolute, or relative to the first allowed folder) if it lies inside an allowed folder. It is
    /// resolved through `..`, symbolic links and junctions first, so none of them can lead outside; a path
    /// that does not exist yet (a new file) is checked through its nearest existing parent.
    pub fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let first = self.folders.first().ok_or("no folders are allowed yet: the user can add some in the chat page's Settings")?;
        let p = Path::new(path.trim());
        let full = if p.is_absolute() { p.to_path_buf() } else { first.join(p) };
        // the nearest part that exists, made canonical, with the rest put back
        let mut existing = full.as_path();
        let mut rest = Vec::new();
        while !existing.exists() {
            rest.push(existing.file_name().ok_or_else(|| format!("{path} is not a valid path"))?.to_owned());
            existing = existing.parent().ok_or_else(|| format!("{path} is not a valid path"))?;
        }
        let mut real = std::fs::canonicalize(existing).map_err(|e| format!("{path}: {e}"))?;
        for part in rest.iter().rev() {
            if part == ".." || part == "." {
                return Err(format!("{path} is not a valid path"));
            }
            real.push(part);
        }
        for root in &self.folders {
            if let Ok(r) = std::fs::canonicalize(root) {
                if real.starts_with(&r) {
                    return Ok(real);
                }
            }
        }
        Err(format!("{path} is outside the allowed folders ({})", self.folder_list()))
    }

    /// `resolve`, or else an absolute path inside the skills folder or the saved outputs' folder: for reading
    /// and for a command's folder, never for writing (the write tools use `resolve`).
    pub fn resolve_read(&self, path: &str) -> Result<PathBuf, String> {
        let err = match self.resolve(path) {
            Ok(p) => return Ok(p),
            Err(e) => e,
        };
        let p = Path::new(path.trim());
        if p.is_absolute() {
            for root in [skills_dir(), outputs_dir()].into_iter().flatten() {
                if let (Ok(real), Ok(r)) = (std::fs::canonicalize(p), std::fs::canonicalize(root)) {
                    if real.starts_with(&r) {
                        return Ok(real);
                    }
                }
            }
        }
        Err(err)
    }

    fn folder_list(&self) -> String {
        self.folders.iter().map(|f| f.display().to_string()).collect::<Vec<_>>().join(", ")
    }

    /// A canonical path as the model should see it: without Windows' \\?\ prefix.
    fn show(&self, p: &Path) -> String {
        let s = p.display().to_string();
        s.strip_prefix(r"\\?\").map(String::from).unwrap_or(s)
    }
}

/// `text` cut to `max` characters: its start and end, with how much was left out between them.
pub fn clip(text: &str, max: usize) -> String {
    let n = text.chars().count();
    if n <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max * 2 / 3).collect();
    let tail: String = text.chars().skip(n - max / 3).collect();
    format!("{head}\n... [{} characters cut] ...\n{tail}", n - head.chars().count() - tail.chars().count())
}

/// Runs tool `name` with its JSON arguments; Ok is the result for the model (at most MAX_RESULT characters, cut
/// in the middle), Err an error it can act on. `checkpoint` groups file changes for undo (the chat page sends
/// one per reply).
pub fn run(name: &str, args: &Json, s: &Settings, checkpoint: &str) -> Result<String, String> {
    run_full(name, args, s, checkpoint).map(|out| clip(&out, MAX_RESULT))
}

/// `run`, its result not cut (only as long as each tool keeps: a command's last 1 MB): the agent saves one too
/// long for the model's context to a file, for the model to read on in.
pub fn run_full(name: &str, args: &Json, s: &Settings, checkpoint: &str) -> Result<String, String> {
    let text = |k: &str| args.get(k).and_then(Json::as_str);
    let need = |k: &str| text(k).ok_or_else(|| format!("{name} needs \"{k}\""));
    let count = |k: &str| args.get(k).and_then(Json::as_f64).filter(|v| *v >= 1.0).map(|v| v as usize);
    let out = match name {
        "read_file" => read_file(&s.resolve_read(need("path")?)?, count("offset").unwrap_or(1), count("limit").unwrap_or(READ_LINES))?,
        "list_files" => list_files(s, &s.resolve_read(text("path").unwrap_or("."))?, text("pattern"))?,
        "search_files" => search_files(s, need("pattern")?, &s.resolve_read(text("path").unwrap_or("."))?, text("glob"), args.get("ignore_case").and_then(Json::as_bool).unwrap_or(false))?,
        "write_file" => write_file(s, &writable(s.resolve(need("path")?)?)?, need("content")?, args.get("append").and_then(Json::as_bool).unwrap_or(false), checkpoint)?,
        "edit_file" => edit_file(s, &writable(s.resolve(need("path")?)?)?, need("old_string")?, text("new_string").unwrap_or(""), args.get("replace_all").and_then(Json::as_bool).unwrap_or(false), checkpoint)?,
        "run_command" => {
            let secs = count("timeout").unwrap_or(COMMAND_SECS).min(MAX_COMMAND_SECS);
            run_command(need("command")?, &s.resolve_read(text("cwd").unwrap_or("."))?, secs)?
        }
        // the chat that started it: a checkpoint is the chat's id and a time
        "start_process" => start_process(need("command")?, &s.resolve_read(text("cwd").unwrap_or("."))?, text("description").unwrap_or(""), checkpoint.rsplit_once('-').map_or("", |(c, _)| c))?,
        "check_process" => check_process(count("id").ok_or("check_process needs \"id\"")?)?,
        "stop_process" => stop_process(count("id").ok_or("stop_process needs \"id\"")?)?,
        "web_search" => web_search(need("query")?)?,
        // a site by its name alone (penipu.my) is its https address
        "fetch_url" => {
            let url = need("url")?.trim();
            fetch_url(&if url.contains("://") { url.to_string() } else { format!("https://{url}") })?
        }
        _ => return Err(format!("no tool named {name}")),
    };
    Ok(out)
}

/// Lines `offset..offset + limit` of a text file, numbered (cat -n style), with a note of what is left.
fn read_file(path: &Path, offset: usize, limit: usize) -> Result<String, String> {
    if path.is_dir() {
        return Err(format!("{} is a folder; use list_files", path.display()));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if bytes[..bytes.len().min(8192)].contains(&0) {
        return Err(format!("{} is a binary file ({} bytes); it cannot be shown as text", path.display(), bytes.len()));
    }
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Ok("(empty file)".into());
    }
    let start = offset.min(lines.len());
    let end = (start - 1 + limit).min(lines.len());
    let mut out = String::new();
    for (i, line) in lines[start - 1..end].iter().enumerate() {
        let line = line.trim_end_matches('\r');
        let shown: String = line.chars().take(MAX_LINE).collect();
        let cut = if shown.len() < line.len() { " [line cut]" } else { "" };
        out.push_str(&format!("{:>6}\t{shown}{cut}\n", start + i));
    }
    if end < lines.len() || start > 1 {
        out.push_str(&format!("(lines {start}-{end} of {}; read more with offset)\n", lines.len()));
    }
    Ok(out)
}

/// A folder's entries (folders first, with a slash), or with `pattern` the files under it whose path
/// relative to it matches the glob (`*`, `?`, `**` for any folders).
fn list_files(s: &Settings, dir: &Path, pattern: Option<&str>) -> Result<String, String> {
    if !dir.is_dir() {
        return Err(format!("{} is not a folder", dir.display()));
    }
    let mut out = Vec::new();
    match pattern.filter(|p| !p.is_empty()) {
        None => {
            let mut entries: Vec<(bool, String, u64)> = std::fs::read_dir(dir)
                .map_err(|e| e.to_string())?
                .flatten()
                .map(|e| {
                    let meta = e.metadata().ok();
                    (meta.as_ref().is_some_and(|m| m.is_dir()), e.file_name().to_string_lossy().into_owned(), meta.map_or(0, |m| m.len()))
                })
                .collect();
            entries.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.to_lowercase().cmp(&b.1.to_lowercase())));
            for (is_dir, name, size) in entries.iter().take(MAX_ENTRIES) {
                out.push(if *is_dir { format!("{name}/") } else { format!("{name}  ({})", human(*size)) });
            }
            if entries.len() > MAX_ENTRIES {
                out.push(format!("... and {} more", entries.len() - MAX_ENTRIES));
            }
        }
        Some(pat) => {
            let pat = pat.replace('\\', "/");
            let mut found = 0;
            walk(dir, dir, &mut |rel, size| {
                if glob_match(&pat, rel) {
                    found += 1;
                    if found <= MAX_ENTRIES {
                        out.push(format!("{rel}  ({})", human(size)));
                    }
                }
            });
            if found > MAX_ENTRIES {
                out.push(format!("... and {} more matches", found - MAX_ENTRIES));
            }
            if found == 0 {
                out.push(format!("no files matching {pat}"));
            }
        }
    }
    Ok(format!("{}:\n{}", s.show(dir), out.join("\n")))
}

/// Every file under `dir` as (path relative to `root` with forward slashes, size), skipping SKIP_DIRS.
fn walk(root: &Path, dir: &Path, f: &mut dyn FnMut(&str, u64)) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let Ok(meta) = e.metadata() else { continue };
        let path = e.path();
        if meta.is_dir() {
            if !SKIP_DIRS.contains(&e.file_name().to_string_lossy().as_ref()) {
                walk(root, &path, f);
            }
        } else if let Ok(rel) = path.strip_prefix(root) {
            f(&rel.to_string_lossy().replace('\\', "/"), meta.len());
        }
    }
}

/// Glob matching on a relative path: `*` and `?` stay inside one folder level, `**/` spans any number of
/// them; case-insensitive (Windows). A pattern without a slash matches the file name at any depth.
fn glob_match(pat: &str, path: &str) -> bool {
    let (pat, path) = (pat.to_lowercase(), path.to_lowercase());
    if !pat.contains('/') {
        let name = path.rsplit('/').next().unwrap_or(&path);
        return glob(pat.as_bytes(), name.as_bytes());
    }
    glob(pat.as_bytes(), path.as_bytes())
}

fn glob(p: &[u8], s: &[u8]) -> bool {
    match p {
        [] => s.is_empty(),
        [b'*', b'*', b'/', rest @ ..] => glob(rest, s) || s.iter().enumerate().any(|(i, &c)| c == b'/' && glob(rest, &s[i + 1..])),
        [b'*', b'*'] => true,
        [b'*', rest @ ..] => (0..=s.len()).take_while(|&i| i == 0 || s[i - 1] != b'/').any(|i| glob(rest, &s[i..])),
        [b'?', rest @ ..] => !s.is_empty() && s[0] != b'/' && glob(rest, &s[1..]),
        [c, rest @ ..] => !s.is_empty() && s[0] == *c && glob(rest, &s[1..]),
    }
}

fn human(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.1} GB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / (1u64 << 20) as f64),
        b if b >= 1 << 10 => format!("{:.0} KB", b as f64 / 1024.0),
        b => format!("{b} B"),
    }
}

/// Lines matching a regular expression in the files under `dir` (or one file), through bettergrep:
/// `path:line:text`, at most MAX_ENTRIES of them.
fn search_files(s: &Settings, pattern: &str, dir: &Path, glob: Option<&str>, ignore_case: bool) -> Result<String, String> {
    let mut cmd = Command::new("bettergrep");
    cmd.arg("grep").arg(pattern).arg(dir).args(["--limit", &MAX_ENTRIES.to_string(), "--timeout", "30"]);
    if let Some(g) = glob.filter(|g| !g.is_empty()) {
        cmd.args(["-g", g]);
    }
    if ignore_case {
        cmd.arg("-i");
    }
    let out = cmd.output().map_err(|e| format!("cannot run bettergrep (is it on PATH?): {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    match out.status.code() {
        Some(1) if text.trim().is_empty() => Ok(format!("no matches for {pattern} in {}", s.show(dir))),
        Some(0) | Some(1) => Ok(text.lines().map(|l| l.strip_prefix(r"\\?\").unwrap_or(l)).collect::<Vec<_>>().join("\n")),
        _ => Err(format!("search failed: {}", String::from_utf8_lossy(&out.stderr).trim())),
    }
}

// ---- changing files, with undo

/// Where file changes are journalled for undo: %LOCALAPPDATA%\ghost-infer\checkpoints, one line per change
/// in journal.txt (checkpoint, sequence, path, whether the file existed) and the old content in <seq>.bak.
fn journal_dir() -> Result<PathBuf, String> {
    let d = std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer").join("checkpoints")).ok_or("LOCALAPPDATA is not set")?;
    std::fs::create_dir_all(&d).map_err(|e| format!("cannot make {}: {e}", d.display()))?;
    Ok(d)
}

/// Saves `path`'s current state (its bytes, or that it did not exist) under `checkpoint` before a change.
fn remember(path: &Path, checkpoint: &str) -> Result<(), String> {
    use std::io::Write;
    let dir = journal_dir()?;
    let seq = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let existed = path.exists();
    if existed {
        std::fs::copy(path, dir.join(format!("{seq}.bak"))).map_err(|e| format!("cannot back up {}: {e}", path.display()))?;
    }
    let mut j = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("journal.txt")).map_err(|e| e.to_string())?;
    // tabs cannot appear in a checkpoint id or a Windows path
    writeln!(j, "{}\t{seq}\t{}\t{}", checkpoint.replace(['\t', '\n'], " "), path.display(), existed as u8).map_err(|e| e.to_string())
}

/// A path for people and models: without Windows' \\?\ prefix.
fn plain(p: &Path) -> String {
    let s = p.display().to_string();
    s.strip_prefix(r"\\?\").map(String::from).unwrap_or(s)
}

/// Puts back every file changed under `checkpoint`, newest change first; returns what it did.
pub fn undo(checkpoint: &str) -> Result<Vec<String>, String> {
    let dir = journal_dir()?;
    let journal = std::fs::read_to_string(dir.join("journal.txt")).unwrap_or_default();
    let mut done = Vec::new();
    let mut keep = Vec::new();
    let entries: Vec<&str> = journal.lines().collect();
    for line in entries.iter().rev() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 4 || f[0] != checkpoint {
            keep.push(*line);
            continue;
        }
        let (bak, path) = (dir.join(format!("{}.bak", f[1])), Path::new(f[2]));
        if f[3] == "1" {
            std::fs::copy(&bak, path).map_err(|e| format!("cannot restore {}: {e}", path.display()))?;
            let _ = std::fs::remove_file(&bak);
            done.push(format!("restored {}", plain(path)));
        } else {
            let _ = std::fs::remove_file(path);
            done.push(format!("removed {} (it was new)", plain(path)));
        }
    }
    keep.reverse();
    std::fs::write(dir.join("journal.txt"), keep.iter().map(|l| format!("{l}\n")).collect::<String>()).map_err(|e| e.to_string())?;
    done.dedup();
    Ok(done)
}

/// A path the model may change: not inside a .git folder (a repository's history is not the model's to edit).
fn writable(path: PathBuf) -> Result<PathBuf, String> {
    if path.components().any(|c| c.as_os_str().eq_ignore_ascii_case(".git")) {
        return Err(format!("{} is inside a .git folder; it cannot be changed", path.display()));
    }
    Ok(path)
}

/// Writes `content` as the file's whole content, or with `append` adds it to the end (a file too long for
/// one reply, written in parts).
fn write_file(s: &Settings, path: &Path, content: &str, append: bool, checkpoint: &str) -> Result<String, String> {
    if path.is_dir() {
        return Err(format!("{} is a folder", path.display()));
    }
    let old = std::fs::read(path).ok();
    if !append && old.as_deref() == Some(content.as_bytes()) {
        return Ok(format!("{} already has this content; nothing changed", s.show(path)));
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot make {}: {e}", dir.display()))?;
    }
    let existed = path.exists();
    remember(path, checkpoint)?;
    let whole = match (append, old) {
        (true, Some(old)) => [old.as_slice(), content.as_bytes()].concat(),
        _ => content.as_bytes().to_vec(),
    };
    std::fs::write(path, &whole).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    let lines = String::from_utf8_lossy(&whole).lines().count();
    Ok(if append && existed {
        let n = content.lines().count();
        format!("added {n} line{} to {} ({lines} lines now)", if n == 1 { "" } else { "s" }, s.show(path))
    } else {
        format!("{} {} ({lines} lines)", if existed { "overwrote" } else { "created" }, s.show(path))
    })
}

/// Replaces `old` with `new` in a text file: exactly once (or every time with `all`), else an error that
/// says why. Windows line ends in the file are kept whatever the model writes, and when the exact text is
/// missing, a match that differs only in trailing spaces is accepted if there is exactly one.
fn edit_file(s: &Settings, path: &Path, old: &str, new: &str, all: bool, checkpoint: &str) -> Result<String, String> {
    if old.is_empty() {
        return Err("old_string is empty; to create a file use write_file".into());
    }
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let text = String::from_utf8(bytes).map_err(|_| format!("{} is not UTF-8 text", path.display()))?;
    let crlf = text.contains("\r\n");
    let (body, old, new) = (text.replace("\r\n", "\n"), old.replace("\r\n", "\n"), new.replace("\r\n", "\n"));
    if old == new {
        return Err("old_string and new_string are the same".into());
    }
    let line_of = |at: usize| body[..at].matches('\n').count() + 1;
    let hits: Vec<usize> = body.match_indices(&old).map(|(i, _)| i).collect();
    let (edited, count, first) = match hits.len() {
        0 => {
            // the same lines, give or take trailing spaces
            let want: Vec<&str> = old.lines().map(str::trim_end).collect();
            let lines: Vec<&str> = body.split('\n').collect();
            let starts: Vec<usize> = (0..lines.len().saturating_sub(want.len() - 1)).filter(|&i| want.iter().enumerate().all(|(k, w)| lines[i + k].trim_end() == *w)).collect();
            match starts[..] {
                [i] => {
                    let mut out: Vec<String> = lines[..i].iter().map(|l| l.to_string()).collect();
                    out.extend(new.split('\n').map(String::from));
                    out.extend(lines[i + want.len()..].iter().map(|l| l.to_string()));
                    (out.join("\n"), 1, i + 1)
                }
                [] => {
                    // where the first line of old_string does appear (give or take spaces), to set the model straight
                    let first = want.iter().find(|l| !l.trim().is_empty()).map(|l| l.trim()).unwrap_or("");
                    let near = lines.iter().position(|l| !first.is_empty() && l.contains(first)).map(|i| {
                        let shown: String = (i.saturating_sub(2)..(i + 3).min(lines.len())).map(|n| format!("{:>6}\t{}\n", n + 1, lines[n])).collect();
                        format!(" Its first line appears here; did you mean:\n{shown}")
                    });
                    return Err(format!(
                        "old_string was not found in {}. It must match exactly, including spaces and indentation, without the line-number prefix of read_file.{}",
                        s.show(path),
                        near.unwrap_or_else(|| " Read the file again before editing.".into())
                    ));
                }
                _ => return Err(format!("old_string matches {} places (ignoring trailing spaces); include more surrounding lines", starts.len())),
            }
        }
        1 => (body.replacen(&old, &new, 1), 1, line_of(hits[0])),
        n if all => (body.replace(&old, &new), n, line_of(hits[0])),
        n => {
            let lines: Vec<String> = hits.iter().take(10).map(|&i| line_of(i).to_string()).collect();
            return Err(format!("old_string matches {n} places (lines {}); include more surrounding lines to pick one, or set replace_all", lines.join(", ")));
        }
    };
    remember(path, checkpoint)?;
    let out = if crlf { edited.replace('\n', "\r\n") } else { edited.clone() };
    std::fs::write(path, out).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    // the changed lines and a few around them, numbered, so the model sees the result
    let lines: Vec<&str> = edited.split('\n').collect();
    let (from, to) = (first.saturating_sub(3).max(1), (first + new.matches('\n').count() + 3).min(lines.len()));
    let around: String = (from..=to).map(|n| format!("{n:>6}\t{}\n", lines[n - 1])).collect();
    Ok(format!("edited {}: {count} replacement{} from line {first}\n{around}", s.show(path), if count == 1 { "" } else { "s" }))
}

// ---- commands

/// run_command's timeout when the model gives none, and the longest it may ask for (seconds).
const COMMAND_SECS: usize = 60;
const MAX_COMMAND_SECS: usize = 600;
/// Output a background process keeps (its latest part), in bytes.
const BG_KEEP: usize = 256 << 10;

/// Run before every command: UTF-8 output whatever the console's code page. PowerShell quotes the whole
/// script in its error messages, so text_of takes this line back out.
const PREAMBLE: &str = "[Console]::OutputEncoding = [Text.Encoding]::UTF8";

/// Bash's `a && b` and `a || b` as Windows PowerShell 5.1 (which has neither; 7 has) runs them: `a; if ($?) { b }`
/// and `a; if (-not $?) { b }`, nested for a chain. Models write them out of habit (Qwen3-Coder ran
/// `cd x && dir` three times running, a parse error each time). Only outside quotes and brackets.
fn and_chains(command: &str) -> String {
    let (mut parts, mut ops) = (Vec::new(), Vec::new());
    let (b, mut depth, mut quote, mut start, mut i) = (command.as_bytes(), 0i32, None, 0, 0);
    while i < b.len() {
        let c = b[i];
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None => match c {
                b'\'' | b'"' => quote = Some(c),
                b'(' | b'{' | b'[' => depth += 1,
                b')' | b'}' | b']' => depth -= 1,
                b'&' | b'|' if depth == 0 && b.get(i + 1) == Some(&c) => {
                    parts.push(command[start..i].trim());
                    ops.push(c);
                    i += 2;
                    start = i;
                    continue;
                }
                _ => {}
            },
        }
        i += 1;
    }
    if ops.is_empty() {
        return command.to_string();
    }
    parts.push(command[start..].trim());
    let mut out = parts[parts.len() - 1].to_string();
    for k in (0..ops.len()).rev() {
        out = format!("{}; if ({}$?) {{ {out} }}", parts[k], if ops[k] == b'|' { "-not " } else { "" });
    }
    out
}

/// `command` in Windows PowerShell, started in `cwd`, stdout and stderr captured together.
fn shell(command: &str, cwd: &Path) -> Result<std::process::Child, String> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"])
        .arg(format!("{PREAMBLE}\n{}", and_chains(command)))
        .current_dir(cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map_err(|e| format!("cannot start PowerShell: {e}"))
}

/// Reads a child's stdout and stderr on two threads into one shared buffer, keeping its last `keep` bytes.
fn collect(child: &mut std::process::Child, keep: usize) -> std::sync::Arc<std::sync::Mutex<Vec<u8>>> {
    use std::io::Read;
    let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let pipes: Vec<Box<dyn Read + Send>> = [child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>), child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>)].into_iter().flatten().collect();
    for mut pipe in pipes {
        let buf = buf.clone();
        std::thread::spawn(move || {
            let mut chunk = [0u8; 8192];
            while let Ok(n) = pipe.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                let mut b = buf.lock().unwrap();
                b.extend_from_slice(&chunk[..n]);
                if b.len() > keep {
                    let cut = b.len() - keep;
                    b.drain(..cut);
                }
            }
        });
    }
    buf
}

/// Ends a process and everything it started (taskkill /T), by process id.
fn kill_tree(pid: u32) {
    use std::os::windows::process::CommandExt;
    let _ = Command::new("taskkill").args(["/T", "/F", "/PID", &pid.to_string()]).creation_flags(0x0800_0000).output();
}

/// Runs a command to completion (or `secs`), then its exit code and output.
fn run_command(command: &str, cwd: &Path, secs: usize) -> Result<String, String> {
    let mut child = shell(command, cwd)?;
    // up to 1 MB: an output too long for the model is saved whole to a file (the agent's spill), a shorter one cut
    let buf = collect(&mut child, 1 << 20);
    let start = std::time::Instant::now();
    let status = loop {
        if let Some(st) = child.try_wait().map_err(|e| e.to_string())? {
            break Some(st);
        }
        if start.elapsed().as_secs() as usize >= secs {
            kill_tree(child.id());
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    std::thread::sleep(std::time::Duration::from_millis(50)); // the readers' last bytes
    let out = text_of(&buf.lock().unwrap());
    let head = match status {
        Some(st) => format!("exit code {}", st.code().map_or("?".into(), |c| c.to_string())),
        None => format!("stopped after {secs} s (timeout); use start_process for long tasks"),
    };
    Ok(format!("{head}\n{}", if out.is_empty() { "(no output)" } else { &out }))
}

/// Process output as text: lossy UTF-8, Windows line ends made plain, trailing blank lines dropped.
fn text_of(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).replace("\r\n", "\n").replace(&format!("{PREAMBLE}\n"), "").trim_end().to_string()
}

/// A command left running: its process and the latest of its output, and how much the model has seen.
struct Background {
    command: String,
    child: std::process::Child,
    out: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    seen: usize,
    started: std::time::Instant,
    ended: Option<(std::time::Duration, Option<i32>)>, // how long it ran, and its exit code, once seen to end
    stopped: bool,                                       // ended by stop_process
    title: String,                                       // what it does, in the model's words (its description)
    chat: String,                                        // the chat that started it
}

impl Background {
    /// Whether it still runs, noting when it is first seen to have ended.
    fn running(&mut self) -> bool {
        if self.ended.is_none() {
            if let Ok(Some(st)) = self.child.try_wait() {
                self.ended = Some((self.started.elapsed(), st.code()));
            }
        }
        self.ended.is_none()
    }
}

fn background() -> &'static std::sync::Mutex<Vec<Option<Background>>> {
    static BG: std::sync::OnceLock<std::sync::Mutex<Vec<Option<Background>>>> = std::sync::OnceLock::new();
    BG.get_or_init(Default::default)
}

fn start_process(command: &str, cwd: &Path, title: &str, chat: &str) -> Result<String, String> {
    let mut child = shell(command, cwd)?;
    let out = collect(&mut child, BG_KEEP);
    let mut bg = background().lock().unwrap();
    bg.push(Some(Background { command: command.to_string(), child, out, seen: 0, started: std::time::Instant::now(), ended: None, stopped: false, title: title.trim().to_string(), chat: chat.to_string() }));
    Ok(format!("started as process {}; read its output with check_process, end it with stop_process", bg.len()))
}

/// A background process's state and the output it printed since the last check.
fn check_process(id: usize) -> Result<String, String> {
    let mut bg = background().lock().unwrap();
    let p = bg.get_mut(id - 1).and_then(Option::as_mut).ok_or_else(|| format!("no process {id}"))?;
    std::thread::sleep(std::time::Duration::from_millis(100));
    let state = match (p.running(), p.ended) {
        (true, _) => "still running".to_string(),
        (false, _) if p.stopped => "stopped".to_string(),
        (false, end) => format!("finished, exit code {}", end.and_then(|e| e.1).map_or("?".into(), |c| c.to_string())),
    };
    let out = p.out.lock().unwrap();
    // `seen` counts bytes of the kept tail; if old output was dropped since, start from what is left
    let from = p.seen.min(out.len());
    let new = text_of(&out[from..]);
    p.seen = out.len();
    Ok(format!("process {id} ({}): {state}\n{}", p.command, if new.is_empty() { "(no new output)".into() } else { new }))
}

/// Ends a background process (it stays listed, as stopped, for the clients).
fn stop_process(id: usize) -> Result<String, String> {
    let mut bg = background().lock().unwrap();
    let p = bg.get_mut(id - 1).and_then(Option::as_mut).ok_or_else(|| format!("no process {id}"))?;
    if !p.running() {
        return Ok(format!("process {id} had already ended"));
    }
    kill_tree(p.child.id());
    let _ = p.child.wait();
    (p.stopped, p.ended) = (true, Some((p.started.elapsed(), None)));
    let buf = p.out.lock().unwrap();
    let out = text_of(&buf[p.seen.min(buf.len())..]);
    Ok(format!("process {id} stopped{}", if out.is_empty() { String::new() } else { format!("; its last output:\n{out}") }))
}

/// The background processes, for the clients: each one's command, whether it runs, for how long it has (or
/// had), and how it ended (its exit code, or stopped).
pub fn processes() -> Json {
    let mut bg = background().lock().unwrap();
    let list = bg
        .iter_mut()
        .enumerate()
        .filter_map(|(i, p)| p.as_mut().map(|p| (i + 1, p)))
        .map(|(id, p)| {
            let running = p.running();
            let secs = p.ended.map_or(p.started.elapsed(), |e| e.0).as_secs_f64().floor();
            let code = p.ended.and_then(|e| e.1).map_or(Json::Null, |c| num(c as f64));
            obj(vec![("id", num(id as f64)), ("command", str(&p.command)), ("title", str(&p.title)), ("chat", str(&p.chat)), ("running", Json::Bool(running)), ("secs", num(secs)), ("exit_code", code), ("stopped", Json::Bool(p.stopped))])
        })
        .collect();
    Json::Arr(list)
}

/// What a background process has printed lately (the kept tail), for the clients; reading it does not count
/// as the model's check.
pub fn process_output(id: usize) -> Option<String> {
    let bg = background().lock().unwrap();
    let p = bg.get(id.checked_sub(1)?)?.as_ref()?;
    let out = p.out.lock().unwrap();
    Some(clip(&text_of(&out), MAX_RESULT))
}

// ---- web (through Windows' own curl.exe: no TLS library here)

const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36";

/// GET `url` (http or https), at most 5 MB in 20 s: the body and its Content-Type.
fn http_get(url: &str) -> Result<(Vec<u8>, String), String> {
    use std::os::windows::process::CommandExt;
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(format!("{url} is not an http(s) address"));
    }
    const MARK: &[u8] = b"\n__ghost_infer_content_type__:";
    let out = Command::new("curl.exe")
        .args(["-sSL", "--compressed", "--max-time", "20", "--max-filesize", "5000000", "-A", USER_AGENT, "-w"])
        .arg(format!("{}%{{content_type}}", String::from_utf8_lossy(MARK)))
        .arg(url)
        .creation_flags(0x0800_0000)
        .output()
        .map_err(|e| format!("cannot run curl.exe: {e}"))?;
    if !out.status.success() {
        return Err(format!("could not fetch {url}: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    let at = out.stdout.windows(MARK.len()).rposition(|w| w == MARK).ok_or("no response")?;
    let ctype = String::from_utf8_lossy(&out.stdout[at + MARK.len()..]).to_lowercase();
    Ok((out.stdout[..at].to_vec(), ctype))
}

/// A page as readable text: HTML turned into text with its links, plain text as it is.
fn fetch_url(url: &str) -> Result<String, String> {
    let (body, ctype) = http_get(url)?;
    if ctype.contains("html") || (ctype.is_empty() && body.starts_with(b"<")) {
        return Ok(format!("{url}\n\n{}", html_to_text(&String::from_utf8_lossy(&body), url)));
    }
    if ctype.starts_with("text/") || ctype.contains("json") || ctype.contains("xml") || ctype.contains("javascript") {
        return Ok(format!("{url}\n\n{}", String::from_utf8_lossy(&body)));
    }
    Err(format!("{url} is {ctype} ({} bytes), not text", body.len()))
}

/// Web results for `query` from DuckDuckGo's plain-HTML page: title, address and snippet of the first ten.
fn web_search(query: &str) -> Result<String, String> {
    let (body, _) = http_get(&format!("https://html.duckduckgo.com/html/?q={}", url_encode(query)))?;
    let page = String::from_utf8_lossy(&body);
    let mut out = Vec::new();
    for part in page.split("class=\"result__a\"").skip(1).take(10) {
        let href = attr(part, "href").unwrap_or_default();
        // results go through a redirect: the real address is its uddg parameter
        let link = href.split("uddg=").nth(1).map(|u| url_decode(u.split('&').next().unwrap_or(u))).unwrap_or_else(|| href.replace("&amp;", "&"));
        let title = part.split_once('>').map(|(_, r)| r.split("</a>").next().unwrap_or("")).map(strip_tags).unwrap_or_default();
        let snippet = part.split("class=\"result__snippet\"").nth(1).and_then(|s| s.split_once('>')).map(|(_, r)| strip_tags(r.split("</a>").next().unwrap_or(""))).unwrap_or_default();
        out.push(format!("{}. {title}\n   {link}\n   {snippet}", out.len() + 1));
    }
    if out.is_empty() {
        return Ok(format!("no results for {query}. Try a shorter or less specific query (or DuckDuckGo may be limiting requests for a while)."));
    }
    Ok(out.join("\n"))
}

fn attr(tag: &str, name: &str) -> Option<String> {
    let at = tag.find(&format!("{name}=\""))? + name.len() + 2;
    Some(tag[at..].split('"').next()?.to_string())
}

fn url_encode(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { (b as char).to_string() } else if b == b' ' { "+".into() } else { format!("%{b:02X}") }).collect()
}

fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
        match b[i] {
            b'%' if i + 2 < b.len() => match (hex(b[i + 1]), hex(b[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push(h << 4 | l);
                    i += 3;
                    continue;
                }
                _ => out.push(b'%'),
            },
            b'+' => out.push(b' '),
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The text of an HTML fragment: tags dropped, entities decoded, whitespace collapsed.
fn strip_tags(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    decode_entities(&out).split_whitespace().collect::<Vec<_>>().join(" ")
}

fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let end = rest[..rest.len().min(12)].find(';');
        let decoded = end.and_then(|e| {
            let name = &rest[1..e];
            let c = match name {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" | "#39" => Some('\''),
                "nbsp" => Some(' '),
                n if n.starts_with("#x") || n.starts_with("#X") => u32::from_str_radix(&n[2..], 16).ok().and_then(char::from_u32),
                n if n.starts_with('#') => n[1..].parse().ok().and_then(char::from_u32),
                _ => None,
            };
            c.map(|c| (c, e + 1))
        });
        match decoded {
            Some((c, len)) => {
                out.push(c);
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// HTML as readable text: scripts, styles and page chrome (nav, footer, ...) dropped; headings marked with
/// #, list items with -, links as [text](address) made absolute against `base`; blank runs collapsed.
fn html_to_text(html: &str, base: &str) -> String {
    const SKIP: [&str; 9] = ["script", "style", "noscript", "svg", "nav", "footer", "aside", "form", "iframe"];
    const BLOCK: [&str; 18] = ["p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "header", "main", "pre", "blockquote", "h1", "h2", "h3", "h4"];
    let origin = base.splitn(4, '/').take(3).collect::<Vec<_>>().join("/"); // scheme://host
    // the page's main content if it marks it (<main>, else <article>), but the title first
    let title = html.find("<title").and_then(|t| html[t..].split_once('>')).map(|(_, r)| strip_tags(r.split("</title>").next().unwrap_or(""))).unwrap_or_default();
    let lower = html.to_ascii_lowercase();
    let html = ["<main", "<article"]
        .iter()
        .find_map(|t| {
            let start = lower.find(t)?;
            let end = lower.rfind(&format!("</{}>", &t[1..])).filter(|&e| e > start)?;
            Some(&html[start..end])
        })
        .unwrap_or(html);
    let mut out = if title.is_empty() { String::new() } else { format!("{title}\n\n") };
    let (mut skip_depth, mut link): (usize, Option<(String, usize)>) = (0, None);
    let mut rest = html;
    while let Some(lt) = rest.find('<') {
        if skip_depth == 0 {
            out.push_str(&decode_entities(&rest[..lt]));
        }
        rest = &rest[lt..];
        if rest.starts_with("<!--") {
            rest = rest.find("-->").map_or("", |e| &rest[e + 3..]);
            continue;
        }
        let Some(gt) = rest.find('>') else { break };
        let tag = &rest[1..gt];
        rest = &rest[gt + 1..];
        let closing = tag.starts_with('/');
        let name: String = tag.trim_start_matches('/').chars().take_while(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase();
        if SKIP.contains(&name.as_str()) {
            if closing {
                skip_depth = skip_depth.saturating_sub(1);
            } else if !tag.ends_with('/') {
                skip_depth += 1;
            }
            continue;
        }
        if skip_depth > 0 {
            continue;
        }
        match (name.as_str(), closing) {
            ("a", false) => {
                if let Some(h) = attr(tag, "href").map(|h| h.replace("&amp;", "&")) {
                    let abs = if h.starts_with("http") { Some(h) } else if h.starts_with("//") { Some(format!("https:{h}")) } else if h.starts_with('/') { Some(format!("{origin}{h}")) } else { None };
                    link = abs.map(|a| (a, out.len()));
                }
            }
            ("a", true) => {
                if let Some((href, start)) = link.take() {
                    let text = out[start..].trim().to_string();
                    if !text.is_empty() && text.len() < 200 {
                        out.truncate(start);
                        out.push_str(&format!("[{text}]({href})"));
                    }
                }
            }
            (h @ ("h1" | "h2" | "h3" | "h4"), false) => out.push_str(&format!("\n\n{} ", "#".repeat(h[1..].parse().unwrap_or(2)))),
            ("li", false) => out.push_str("\n- "),
            (n, _) if BLOCK.contains(&n) => out.push('\n'),
            ("td" | "th", false) => out.push_str(" | "),
            _ => {}
        }
    }
    // collapse spaces within lines and blank lines between them
    let lines: Vec<String> = out.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|l| !l.is_empty() && l != "-" && l != "|").collect();
    // a run of 8 or more lines that are nothing but a link is a menu: kept as a count
    let bare = |l: &str| {
        let l = l.trim_start_matches("- ");
        l.starts_with('[') && l.ends_with(')') && l.matches("](").count() == 1
    };
    let mut text = String::new();
    let mut i = 0;
    while i < lines.len() {
        let run = lines[i..].iter().take_while(|l| bare(l)).count();
        if run >= 8 {
            text.push_str(&format!("({run} links)\n"));
            i += run;
            continue;
        }
        text.push_str(&lines[i]);
        text.push('\n');
        i += 1;
    }
    text
}

/// What the chat page shows about the tools' state: the allowed folders.
pub fn status(s: &Settings) -> Json {
    let j = s.to_json();
    obj(vec![
        ("folders", j.get("folders").cloned().unwrap_or(Json::Null)),
        ("always", j.get("always").cloned().unwrap_or(Json::Null)),
        ("warm_up", Json::Bool(s.warm_up)),
        ("max_result", num(MAX_RESULT as f64)),
        ("processes", processes()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        assert!(glob_match("*.rs", "src/main.rs"));
        assert!(glob_match("src/*.rs", "src/main.rs"));
        assert!(!glob_match("src/*.rs", "src/a/main.rs"));
        assert!(glob_match("src/**/*.rs", "src/a/b/main.rs"));
        assert!(glob_match("src/**/*.rs", "src/main.rs"));
        assert!(glob_match("**/*.TXT", "a/b.txt"));
        assert!(glob_match("?.md", "x/a.md"));
        assert!(!glob_match("*.md", "notes.txt"));
    }

    #[test]
    fn clip_keeps_both_ends() {
        let t: String = (0..1000).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
        let c = clip(&t, 90);
        assert!(c.starts_with(&t[..60]) && c.ends_with(&t[t.len() - 30..]) && c.contains("characters cut"));
        assert_eq!(clip("short", 90), "short");
    }

    #[test]
    fn edit_and_undo() {
        let root = std::env::temp_dir().join("ghost_infer_edit_test");
        let _ = std::fs::create_dir_all(&root);
        let s = Settings { folders: vec![root.clone()], ..Settings::default() };
        let f = root.join("a.txt");
        let original = "one\r\ntwo  \r\nthree\r\ntwo  \r\n";
        std::fs::write(&f, original).unwrap();
        let cp = "test-checkpoint";
        // two matches: refused, with where they are
        let e = edit_file(&s, &f, "two", "2", false, cp).unwrap_err();
        assert!(e.contains("2 places") && e.contains("lines 2, 4"), "{e}");
        // the model's LF text against the file's CRLF, and trailing spaces left out
        edit_file(&s, &f, "one\ntwo\nthree", "ONE\nTHREE", false, cp).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "ONE\r\nTHREE\r\ntwo  \r\n");
        edit_file(&s, &f, "two  ", "2", true, cp).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "ONE\r\nTHREE\r\n2\r\n");
        assert!(edit_file(&s, &f, "missing", "x", false, cp).is_err());
        // a file written in parts; undo takes back both
        write_file(&s, &root.join("new.txt"), "hello\n", false, cp).unwrap();
        let r = write_file(&s, &root.join("new.txt"), "world\n", true, cp).unwrap();
        assert!(r.starts_with("added 1 line to") && r.contains("2 lines now"), "{r}");
        assert_eq!(std::fs::read_to_string(root.join("new.txt")).unwrap(), "hello\nworld\n");
        let done = undo(cp).unwrap();
        assert!(done.iter().any(|d| d.contains("new.txt")), "{done:?}");
        assert_eq!(std::fs::read_to_string(&f).unwrap(), original);
        assert!(!root.join("new.txt").exists());
    }

    #[test]
    fn and_chains_in_powershell() {
        assert_eq!(and_chains("cd site && dir"), "cd site; if ($?) { dir }");
        assert_eq!(and_chains("a && b && c"), "a; if ($?) { b; if ($?) { c } }");
        assert_eq!(and_chains("npm test || echo failed"), "npm test; if (-not $?) { echo failed }");
        // in quotes, in brackets, and a pipe: as they are
        assert_eq!(and_chains("echo 'a && b'"), "echo 'a && b'");
        assert_eq!(and_chains("if ($x) { a && b }"), "if ($x) { a && b }");
        assert_eq!(and_chains("dir | select -first 3"), "dir | select -first 3");
        // and it runs: the second part only after the first worked
        let out = run_command("cd .. && echo ran", &std::env::temp_dir(), 30).unwrap();
        assert!(out.starts_with("exit code 0") && out.contains("ran"), "{out}");
    }

    #[test]
    fn paths_stay_inside() {
        let root = std::env::temp_dir().join("ghost_infer_tools_test");
        let _ = std::fs::create_dir_all(root.join("sub"));
        let s = Settings { folders: vec![root.clone()], ..Settings::default() };
        assert!(s.resolve("sub").is_ok());
        assert!(s.resolve("sub/new.txt").is_ok());
        assert!(s.resolve("..").is_err());
        assert!(s.resolve("sub/../../x").is_err());
        assert!(s.resolve(r"C:\Windows\win.ini").is_err());
        assert!(Settings::default().resolve("a").is_err());
    }
}
