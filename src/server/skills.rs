//! Skills, in the Agent Skills format Claude uses (agentskills.io): a folder with a SKILL.md, YAML front matter
//! with a name and a description, then instructions, plus any files those refer to. They are installed into
//! %LOCALAPPDATA%\ghost-infer\skills\<name>. The model sees every skill's name and description; use_skill
//! gives it a skill's instructions when a task fits, and its other files stay readable for when they are needed.

use std::path::{Path, PathBuf};

use crate::json::{obj, str, Json};

pub struct Skill {
    pub name: String,
    pub description: String,
    pub dir: PathBuf,
}

fn root() -> Result<PathBuf, String> {
    let d = crate::tools::skills_dir().ok_or("LOCALAPPDATA is not set")?;
    std::fs::create_dir_all(&d).map_err(|e| format!("cannot make {}: {e}", d.display()))?;
    Ok(d)
}

/// A SKILL.md's front matter fields and its body. Only what skills use: `key: value` lines, quoted or not,
/// and `key: >` or `key: |` followed by indented lines.
fn parse(text: &str) -> (Vec<(String, String)>, String) {
    let text = text.trim_start_matches('\u{feff}').replace("\r\n", "\n");
    let Some(rest) = text.strip_prefix("---\n") else { return (Vec::new(), text) };
    let Some(end) = rest.find("\n---") else { return (Vec::new(), text.clone()) };
    let (head, body) = (&rest[..end], rest[end + 4..].trim_start_matches(['-', '\n']));
    let mut fields: Vec<(String, String)> = Vec::new();
    let mut block: Option<usize> = None; // the field a `>` or `|` block belongs to
    for line in head.lines() {
        if let (Some(i), true) = (block, line.starts_with([' ', '\t'])) {
            let v = &mut fields[i].1;
            if !v.is_empty() {
                v.push(' ');
            }
            v.push_str(line.trim());
            continue;
        }
        block = None;
        let Some((k, v)) = line.split_once(':') else { continue };
        if k.starts_with([' ', '\t']) {
            continue; // a nested map (metadata): not needed
        }
        let v = v.trim();
        if matches!(v, ">" | "|" | ">-" | "|-") {
            fields.push((k.trim().to_string(), String::new()));
            block = Some(fields.len() - 1);
        } else {
            fields.push((k.trim().to_string(), v.trim_matches(|c| c == '"' || c == '\'').to_string()));
        }
    }
    (fields, body.to_string())
}

fn read(dir: &Path) -> Option<Skill> {
    let text = std::fs::read_to_string(dir.join("SKILL.md")).ok()?;
    let (fields, _) = parse(&text);
    let get = |k: &str| fields.iter().find(|(f, _)| f == k).map(|(_, v)| v.clone()).filter(|v| !v.is_empty());
    let folder = dir.file_name()?.to_string_lossy().to_string();
    Some(Skill { name: get("name").unwrap_or(folder), description: get("description").unwrap_or_default(), dir: dir.to_path_buf() })
}

/// The installed skills, by name.
pub fn list() -> Vec<Skill> {
    let Ok(root) = root() else { return Vec::new() };
    let mut skills: Vec<Skill> = std::fs::read_dir(root).into_iter().flatten().flatten().filter(|e| e.path().is_dir()).filter_map(|e| read(&e.path())).collect();
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

pub fn list_json() -> Json {
    Json::Arr(list().iter().map(|s| obj(vec![("name", str(&s.name)), ("description", str(&s.description))])).collect())
}

/// What the system prompt says about the skills, or nothing when none are installed.
pub fn prompt() -> String {
    let skills = list();
    if skills.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = skills.iter().map(|s| format!("- {}: {}", s.name, crate::tools::clip(&s.description, 400))).collect();
    format!("Skills you can use. When a task matches one, call use_skill with its name before you start, then follow its instructions:\n{}", lines.join("\n"))
}

/// The most of a skill's instructions use_skill gives at once, in characters (about 6,000 tokens), in a
/// context of 48K tokens or more.
const SKILL_CHARS: usize = 24000;

/// The most of a skill use_skill gives with a context of `ctx` tokens: about an eighth of it (16,000
/// characters in 32K), up to SKILL_CHARS. Taste's 24,000 with a 26 KB page to read and rewrite overflowed
/// a 32K context each time it was loaded again after a compaction, round after round.
pub fn skill_chars(ctx: usize) -> usize {
    (ctx / 2).clamp(4000, SKILL_CHARS)
}

/// The use_skill tool: a skill's instructions (at most `max` characters of them), and where its other files are.
pub fn use_skill(name: &str, max: usize) -> String {
    let skills = list();
    let Some(s) = skills.iter().find(|s| s.name.eq_ignore_ascii_case(name.trim())) else {
        return format!("Error: no skill named {name}. The skills are: {}", skills.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", "));
    };
    let path = s.dir.join("SKILL.md");
    let text = std::fs::read_to_string(&path).unwrap_or_default().replace("\r\n", "\n");
    let (_, body) = parse(&text);
    let dir = s.dir.display().to_string();
    format!(
        "{}\n\n(This skill's folder is {dir}. Read the files it mentions with read_file, using full paths in that folder; run its scripts with run_command and cwd set to that folder.)",
        skill_text(body.trim(), &text, &path.display().to_string(), max)
    )
}

/// A skill's instructions as use_skill gives them: whole when they fit in `max` characters; else their start,
/// to a line's end, then the headings of the rest with their lines in SKILL.md, so the model reads only the
/// part a task needs (Taste's SKILL.md is 85 KB: read whole, it fills a 32K context on its own).
fn skill_text(body: &str, file: &str, path: &str, max: usize) -> String {
    if body.len() <= max {
        return body.to_string();
    }
    let lines: Vec<&str> = body.lines().collect();
    let (mut kept, mut chars) = (0, 0);
    while kept < lines.len() && chars + lines[kept].len() < max {
        chars += lines[kept].len() + 1;
        kept += 1;
    }
    // the body's first line in the file (after the front matter), so the numbers are read_file's
    let first = file.lines().position(|l| l == lines[0]).unwrap_or(0) + 1;
    let sections: Vec<String> = lines[kept..].iter().enumerate().filter(|(_, l)| l.starts_with('#')).take(60).map(|(i, l)| format!("line {}: {}", first + kept + i, l.trim())).collect();
    format!(
        "{}\n\n[{path} goes on for {} more lines, left out to save room in the context. Its sections from here: {}. Read one with read_file (offset and limit) only when the task needs it; do not read the whole file.]",
        lines[..kept].join("\n"),
        lines.len() - kept,
        if sections.is_empty() { "(no headings)".to_string() } else { sections.join("; ") }
    )
}

/// A skill name as the format allows: lowercase letters, digits and single hyphens.
fn clean_name(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().to_lowercase().chars() {
        let c = if c.is_ascii_alphanumeric() { c } else { '-' };
        if !(c == '-' && (out.is_empty() || out.ends_with('-'))) {
            out.push(c);
        }
    }
    out.trim_end_matches('-').chars().take(64).collect()
}

/// Every folder with a SKILL.md under `dir`, a few levels down (a repository can hold many skills).
fn skill_dirs(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 5 {
        return;
    }
    let skip = |p: &Path| p.file_name().is_some_and(|n| n == ".git" || n == "node_modules");
    let mut subs: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_dir() && !skip(p)).collect();
    subs.sort();
    for p in subs {
        if p.join("SKILL.md").is_file() {
            out.push(p);
        } else {
            skill_dirs(&p, depth + 1, out);
        }
    }
}

/// The skills in `dir`: itself, or every one below it.
fn found(dir: &Path) -> Vec<PathBuf> {
    if dir.join("SKILL.md").is_file() {
        return vec![dir.to_path_buf()];
    }
    let mut out = Vec::new();
    skill_dirs(dir, 0, &mut out);
    out
}

/// A skill's path below the folder it was found in, with forward slashes ("" for the folder itself).
fn rel(dir: &Path, base: &Path) -> String {
    dir.strip_prefix(base).unwrap_or(dir).display().to_string().replace('\\', "/")
}

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)?.flatten() {
        let (src, dst) = (e.path(), to.join(e.file_name()));
        if e.file_name() == ".git" {
            continue;
        }
        if src.is_dir() {
            copy_dir(&src, &dst)?;
        } else {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

/// Unpacks a .zip / .skill archive into `into` (Windows' own tar reads zip archives).
fn unpack(file: &Path, into: &Path) -> Result<(), String> {
    std::fs::create_dir_all(into).map_err(|e| e.to_string())?;
    let out = std::process::Command::new("tar").arg("-xf").arg(file).arg("-C").arg(into).output().map_err(|e| format!("cannot unpack it: {e}"))?;
    if !out.status.success() {
        return Err(format!("cannot unpack it: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(())
}

/// A GitHub link as what to download and, for a repository's zip, the folder in it to install from:
/// github.com/<o>/<r>, …/tree/<ref>/<folder>, …/blob/<ref>/<folder>/SKILL.md, raw.githubusercontent.com/<o>/<r>/<ref>/<folder>/SKILL.md.
/// A link to an archive in a repository downloads just that file.
fn github(url: &str) -> Option<(String, Option<String>)> {
    let rest = url.split_once("://")?.1.split(['?', '#']).next()?.trim_end_matches('/');
    let (host, path) = rest.split_once('/')?;
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    let (owner, repo, reference, mut sub) = match (host.trim_start_matches("www."), parts.as_slice()) {
        ("github.com", [o, r]) => (*o, r.trim_end_matches(".git"), "HEAD", Vec::new()),
        ("github.com", [o, r, "tree" | "blob", reference, sub @ ..]) => (*o, *r, *reference, sub.to_vec()),
        ("raw.githubusercontent.com", [o, r, reference, sub @ ..]) => (*o, *r, *reference, sub.to_vec()),
        _ => return None,
    };
    let last = sub.last().map(|l| l.to_lowercase()).unwrap_or_default();
    if last.ends_with(".zip") || last.ends_with(".skill") {
        return Some((format!("https://raw.githubusercontent.com/{owner}/{repo}/{reference}/{}", sub.join("/")), None));
    }
    if last.ends_with(".md") {
        sub.pop();
    }
    Some((format!("https://codeload.github.com/{owner}/{repo}/zip/{reference}"), Some(sub.join("/"))))
}

/// Downloads a skill into `tmp`; the folder to look for skills in.
fn download(url: &str, tmp: &Path) -> Result<PathBuf, String> {
    let (fetch, sub) = github(url).unwrap_or((url.to_string(), None));
    let file = tmp.join("download");
    std::fs::create_dir_all(tmp).map_err(|e| e.to_string())?;
    let out = std::process::Command::new("curl")
        .args(["-fsSL", "--max-time", "300", "--max-filesize", "500000000", "-A", "ghost-infer", "-o"])
        .arg(&file)
        .arg(&fetch)
        .output()
        .map_err(|e| format!("cannot download it: {e}"))?;
    if !out.status.success() {
        let why = String::from_utf8_lossy(&out.stderr);
        let why = why.trim().trim_start_matches("curl: ");
        let why = why.split_once(") ").map_or(why, |(_, w)| w);
        if why.ends_with("error: 404") {
            return Err("nothing is at that link (404)".into());
        }
        return Err(format!("cannot download it: {why}"));
    }
    let dir = tmp.join("files");
    let head = std::fs::read(&file).map_err(|e| e.to_string())?;
    if head.starts_with(b"PK") {
        unpack(&file, &dir)?;
    } else if String::from_utf8_lossy(&head[..head.len().min(64)]).trim_start_matches('\u{feff}').trim_start().starts_with("---") {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::rename(&file, dir.join("SKILL.md")).map_err(|e| e.to_string())?;
    } else {
        return Err("that link gives neither a skill archive nor a SKILL.md".into());
    }
    let Some(sub) = sub else { return Ok(dir) };
    // a repository's zip holds one folder, <repo>-<ref>
    let top = std::fs::read_dir(&dir).ok().and_then(|d| d.flatten().map(|e| e.path()).find(|p| p.is_dir())).ok_or("the repository's download is empty")?;
    let base = if sub.is_empty() { top } else { top.join(&sub) };
    if !base.is_dir() {
        return Err(format!("the repository has no folder {sub}"));
    }
    Ok(base)
}

/// A skill in a source that holds several: its path there, its name and what it is for.
pub struct Choice {
    pub path: String,
    pub name: String,
    pub description: String,
}

/// What installing did: the skills it installed, by name, or (a source with several skills and none
/// picked) the skills to choose from.
pub enum Installed {
    Done(Vec<String>),
    Choose(Vec<Choice>),
}

/// Copies the skill in `dir` into the skills folder, replacing one of the same name. Its name.
fn install_dir(root: &Path, dir: &Path) -> Result<String, String> {
    let skill = read(dir).ok_or("its SKILL.md cannot be read")?;
    let name = clean_name(&skill.name);
    if name.is_empty() {
        return Err("its SKILL.md has no usable name".to_string());
    }
    let target = root.join(&name);
    if target.exists() {
        std::fs::remove_dir_all(&target).map_err(|e| format!("cannot replace the installed {name}: {e}"))?;
    }
    copy_dir(dir, &target).map_err(|e| format!("cannot copy {name}: {e}"))?;
    Ok(name)
}

/// Installs skills from a folder or a .zip / .skill archive, or from a link: GitHub (a repository, a folder
/// in one, or its SKILL.md) or a direct link to an archive or a SKILL.md. Every folder with a SKILL.md
/// counts, with all its files. When the source holds several skills, `pick` says which: their paths or names,
/// or "all"; with none picked, the answer is the list to choose from, and a download is kept (in .pending)
/// so that choosing does not download it again. A skill of the same name is replaced.
pub fn install(source: &str, pick: &[String]) -> Result<Installed, String> {
    let root = root()?;
    let source = source.trim().trim_matches('"');
    let link = source.starts_with("https://") || source.starts_with("http://");
    let tmp = root.join(format!(".install-{}", std::process::id()));
    let held = root.join(".pending");
    let _ = std::fs::remove_dir_all(&tmp);
    let result = (|| {
        let src = PathBuf::from(source);
        // the download kept when the list was shown, if this is the same link
        let kept = std::fs::read_to_string(held.join("source.txt")).ok().filter(|s| link && !pick.is_empty() && s == source);
        let base = if let Some(rel) = kept.and_then(|_| std::fs::read_to_string(held.join("base.txt")).ok()) {
            held.join(rel)
        } else if link {
            download(source, &tmp)?
        } else if src.is_dir() {
            src
        } else if src.is_file() {
            unpack(&src, &tmp.join("files"))?;
            tmp.join("files")
        } else {
            return Err(format!("{source} is not a folder, a file or a link"));
        };
        let all = found(&base);
        if all.is_empty() {
            return Err("there is no SKILL.md in it".into());
        }
        if all.len() > 1 && pick.is_empty() {
            let choices = all.iter().filter_map(|d| read(d).map(|s| Choice { path: rel(d, &base), name: s.name, description: s.description })).collect();
            if link && base.starts_with(&tmp) {
                let _ = std::fs::remove_dir_all(&held);
                std::fs::rename(&tmp, &held).map_err(|e| e.to_string())?;
                let _ = std::fs::write(held.join("source.txt"), source);
                let _ = std::fs::write(held.join("base.txt"), base.strip_prefix(&tmp).unwrap_or(&base).display().to_string());
            }
            return Ok(Installed::Choose(choices));
        }
        // which ones: by path or folder name or skill name; "all" for every one (the first of a name)
        let wanted = |d: &PathBuf| {
            let (path, folder) = (rel(d, &base), d.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default());
            let name = read(d).map(|s| s.name).unwrap_or_default();
            all.len() == 1 || pick.iter().any(|p| {
                let p = p.trim().trim_matches('/').replace('\\', "/");
                p.eq_ignore_ascii_case("all") || p.eq_ignore_ascii_case(&path) || p.eq_ignore_ascii_case(&folder) || p.eq_ignore_ascii_case(&name)
            })
        };
        let mut names: Vec<String> = Vec::new();
        for d in all.iter().filter(|d| wanted(d)) {
            if read(d).is_some_and(|s| names.contains(&clean_name(&s.name))) {
                continue;
            }
            names.push(install_dir(&root, d)?);
        }
        if names.is_empty() {
            let paths: Vec<String> = all.iter().map(|d| rel(d, &base)).collect();
            return Err(format!("it has no skill {}; its skills are: {}", pick.join(", "), paths.join(", ")));
        }
        let _ = std::fs::remove_dir_all(&held);
        Ok(Installed::Done(names))
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    result
}

/// The install_skill tool: installs from `source` (`skill` picks one when it holds several) and says
/// what happened in words the model can act on.
pub fn install_tool(source: &str, skill: &str) -> String {
    let pick: Vec<String> = if skill.trim().is_empty() { Vec::new() } else { vec![skill.trim().to_string()] };
    match install(source, &pick) {
        Ok(Installed::Done(names)) => {
            let skills = list();
            let lines: Vec<String> = names
                .iter()
                .map(|n| {
                    let about = skills.iter().find(|s| clean_name(&s.name) == *n).map(|s| crate::tools::clip(&s.description, 300)).unwrap_or_default();
                    format!("- {n}: {about}")
                })
                .collect();
            format!("Installed:\n{}\nTo use one, call use_skill with its name.", lines.join("\n"))
        }
        Ok(Installed::Choose(choices)) => {
            let lines: Vec<String> = choices.iter().take(40).map(|c| format!("- {} ({}): {}", c.path, c.name, crate::tools::clip(&c.description, 200))).collect();
            let more = if choices.len() > 40 { format!("\n(and {} more)", choices.len() - 40) } else { String::new() };
            format!(
                "Nothing installed yet: this holds {} skills.\n{}{more}\nCall install_skill again with the same source and skill set to the path of the one the user wants, or to \"all\". If it is not clear which one, ask the user.",
                choices.len(),
                lines.join("\n")
            )
        }
        Err(e) => format!("Error: cannot install it: {e}"),
    }
}

/// Removes an installed skill.
pub fn remove(name: &str) -> Result<(), String> {
    let name = clean_name(name);
    let dir = root()?.join(&name);
    if name.is_empty() || !dir.join("SKILL.md").is_file() {
        return Err("no such skill".into());
    }
    std::fs::remove_dir_all(dir).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_skill_is_cut_with_its_sections_listed() {
        // 3,000 lines of 20 characters, a heading every 500: the start whole, then where each later heading is
        let body: String = (0..3000).map(|i| if i % 500 == 0 { format!("## Part {:<12}\n", i / 500) } else { format!("{:<19}\n", "text") }).collect();
        let file = format!("---\nname: big\ndescription: d\n---\n{body}");
        let (_, parsed) = parse(&file);
        let out = skill_text(parsed.trim(), &file, "SKILL.md", SKILL_CHARS);
        assert!(out.len() < SKILL_CHARS + 2000, "{}", out.len());
        assert_eq!(skill_chars(32768), 16384);
        // about 1,200 lines fit; the body starts on the file's 5th line, so its line 1500 (Part 3) is the file's
        // 1505th, and Part 2 (kept) is not listed
        assert!(out.contains("line 1505: ## Part 3"), "{}", &out[out.len() - 400..]);
        assert!(!out.contains("## Part 2;"));
        assert!(out.contains("do not read the whole file"));
        assert_eq!(skill_text("short", "short", "SKILL.md", SKILL_CHARS), "short");
    }

    #[test]
    fn front_matter() {
        let (f, body) = parse("---\nname: pdf-tools\ndescription: >\n  Fill PDF forms.\n  Use for PDFs.\nmetadata:\n  author: x\n---\n# Steps\nDo it.");
        assert_eq!(f[0], ("name".into(), "pdf-tools".into()));
        assert_eq!(f[1], ("description".into(), "Fill PDF forms. Use for PDFs.".into()));
        assert!(body.starts_with("# Steps"));
        assert_eq!(clean_name("My Skill__v2"), "my-skill-v2");
    }

    #[test]
    fn github_links() {
        let zip = |r: &str| format!("https://codeload.github.com/a/b/zip/{r}");
        assert_eq!(github("https://github.com/a/b"), Some((zip("HEAD"), Some(String::new()))));
        assert_eq!(github("https://github.com/a/b.git/"), Some((zip("HEAD"), Some(String::new()))));
        assert_eq!(github("https://github.com/a/b/tree/main/skills/pdf"), Some((zip("main"), Some("skills/pdf".into()))));
        assert_eq!(github("https://github.com/a/b/blob/main/skills/pdf/SKILL.md"), Some((zip("main"), Some("skills/pdf".into()))));
        assert_eq!(github("https://raw.githubusercontent.com/a/b/main/x/SKILL.md"), Some((zip("main"), Some("x".into()))));
        assert_eq!(github("https://github.com/a/b/blob/main/dist/x.zip"), Some(("https://raw.githubusercontent.com/a/b/main/dist/x.zip".into(), None)));
        assert_eq!(github("https://github.com/a/b/releases/download/v1/x.zip"), None);
        assert_eq!(github("https://example.com/x.zip"), None);
    }
}
