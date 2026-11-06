//! Files attached to a message: text or code, picked, dropped on the window or pasted (a file copied in
//! Explorer). The server's /api/agent takes a message as text, so a file goes inside it, after what the user
//! typed, as <file name="...">...</file>; the bubble shows it as a chip again (`split`).

use std::path::Path;

/// The most one file may hold: about 50K tokens, most of the context a model here has.
const MAX_BYTES: usize = 200 * 1024;

/// A file attached to the message being written.
#[derive(Debug, Clone)]
pub struct Attachment {
    pub name: String,
    pub text: String,
}

impl Attachment {
    /// How big it is, for its chip: "212 lines".
    pub fn size(&self) -> String {
        lines(self.text.lines().count())
    }
}

/// A file's size on its chip: "212 lines".
pub fn lines(n: usize) -> String {
    if crate::i18n::is_zh() { format!("{n} 行") } else { format!("{n} line{}", if n == 1 { "" } else { "s" }) }
}

/// Reads a file to attach, if it is text (UTF-8) and not too big.
pub fn read(path: &Path) -> Result<Attachment, String> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.display().to_string());
    let meta = std::fs::metadata(path).map_err(|e| crate::tf!("Could not read {name}: {e}", "无法读取 {name}：{e}"))?;
    if meta.is_dir() {
        return Err(crate::tf!("{name} is a folder: attach files from it, or add it with Add folder", "{name} 是文件夹：请附加其中的文件，或用「添加文件夹」加入它"));
    }
    if meta.len() as usize > MAX_BYTES {
        return Err(crate::tf!("{name} is too big to attach ({} KB; up to {} KB)", "{name} 太大，无法附加（{} KB；上限 {} KB）", meta.len() / 1024, MAX_BYTES / 1024));
    }
    let bytes = std::fs::read(path).map_err(|e| crate::tf!("Could not read {name}: {e}", "无法读取 {name}：{e}"))?;
    let text = String::from_utf8(bytes).ok().filter(|t| !t.contains('\0')).ok_or_else(|| crate::tf!("{name} is not a text file", "{name} 不是文本文件"))?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text).replace("\r\n", "\n");
    Ok(Attachment { name, text })
}

/// The message as the model gets it: the user's words, then each file.
pub fn join(words: &str, files: &[Attachment]) -> String {
    let mut out = words.trim().to_string();
    for f in files {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&format!("<file name=\"{}\">\n{}\n</file>", f.name.replace('"', "'"), f.text.trim_end_matches('\n')));
    }
    out
}

/// A sent message's words and the files in it (name and line count), as `join` put them together.
pub fn split(raw: &str) -> (&str, Vec<(String, usize)>) {
    const OPEN: &str = "<file name=\"";
    let starts: Vec<usize> = raw.match_indices(OPEN).map(|(i, _)| i).filter(|&i| i == 0 || raw[..i].ends_with('\n')).collect();
    let Some(&first) = starts.first() else { return (raw, Vec::new()) };
    let mut files = Vec::new();
    for (k, &at) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(raw.len());
        let block = &raw[at + OPEN.len()..end];
        let Some((name, body)) = block.split_once("\">\n") else { return (raw, Vec::new()) };
        let body = body.trim_end().strip_suffix("</file>").unwrap_or(body);
        files.push((name.to_string(), body.lines().count()));
    }
    (raw[..first].trim_end(), files)
}

/// Files copied in Explorer (on the clipboard as CF_HDROP), for Ctrl+V in the message box.
#[cfg(windows)]
pub fn clipboard_files() -> Vec<std::path::PathBuf> {
    #[link(name = "user32")]
    unsafe extern "system" {
        fn OpenClipboard(hwnd: isize) -> i32;
        fn CloseClipboard() -> i32;
        fn IsClipboardFormatAvailable(format: u32) -> i32;
        fn GetClipboardData(format: u32) -> isize;
    }
    #[link(name = "shell32")]
    unsafe extern "system" {
        fn DragQueryFileW(drop: isize, file: u32, buf: *mut u16, len: u32) -> u32;
    }
    const CF_HDROP: u32 = 15;
    let mut out = Vec::new();
    unsafe {
        if IsClipboardFormatAvailable(CF_HDROP) == 0 || OpenClipboard(0) == 0 {
            return out;
        }
        let drop = GetClipboardData(CF_HDROP);
        if drop != 0 {
            let n = DragQueryFileW(drop, u32::MAX, std::ptr::null_mut(), 0);
            for i in 0..n {
                let len = DragQueryFileW(drop, i, std::ptr::null_mut(), 0);
                let mut buf = vec![0u16; len as usize + 1];
                let got = DragQueryFileW(drop, i, buf.as_mut_ptr(), buf.len() as u32);
                out.push(std::path::PathBuf::from(String::from_utf16_lossy(&buf[..got as usize])));
            }
        }
        CloseClipboard();
    }
    out
}

#[cfg(not(windows))]
pub fn clipboard_files() -> Vec<std::path::PathBuf> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_with_files_splits_back() {
        let files = [Attachment { name: "a.js".into(), text: "let a = 1;\nlet b = 2;\n".into() }, Attachment { name: "b.txt".into(), text: "one".into() }];
        let raw = join("fix this", &files);
        assert_eq!(split(&raw), ("fix this", vec![("a.js".to_string(), 2), ("b.txt".to_string(), 1)]));
        let raw = join("", &files[1..]);
        assert_eq!(split(&raw), ("", vec![("b.txt".to_string(), 1)]));
        assert_eq!(split("no files"), ("no files", vec![]));
        // a mention in the middle of a line is not a file
        assert_eq!(split("say <file name=\"x\"> here"), ("say <file name=\"x\"> here", vec![]));
    }
}
