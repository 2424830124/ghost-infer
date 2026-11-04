//! Byte-level BPE tokenizer (GGUF `tokenizer.ggml.model = "gpt2"`), as used by SmolLM, Qwen, Llama 3 and friends.

use std::collections::HashMap;

use crate::gguf::{Gguf, Value};

/// GGUF token types (llama.cpp `llama_token_type`).
const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreTokenizer {
    /// GPT-2 split regex.
    Gpt2,
    /// GPT-2 split regex, but every digit is its own piece first (SmolLM, StarCoder).
    Gpt2IsolatedDigits,
    /// Qwen2/Qwen3 split regex (see `split_qwen2`).
    Qwen2,
    /// GLM-4 ("glm4"): the Qwen2 regex with numbers in runs of up to three digits (`\p{N}{1,3}`), as in Llama 3.
    Glm4,
    /// Qwen3.5 ("qwen35"): the Qwen2 regex with combining marks (`\p{M}`) counted with the letters.
    Qwen35,
}

pub struct Tokenizer {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), u32>,
    /// Control / user-defined tokens, longest first, matched literally in the input.
    specials: Vec<(String, u32)>,
    is_special: Vec<bool>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pre: PreTokenizer,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub add_bos: bool,
}

/// GPT-2's reversible byte -> printable unicode mapping.
fn byte_level_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0..=255u32 {
        let printable = (b'!' as u32..=b'~' as u32).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            extra += 1;
            char::from_u32(255 + extra).unwrap()
        };
    }
    table
}

fn is_letter(c: char) -> bool { c.is_alphabetic() }
fn is_number(c: char) -> bool { c.is_numeric() }
fn is_other(c: char) -> bool { !c.is_whitespace() && !is_letter(c) && !is_number(c) }

include!("unicode_marks.rs");

/// A combining mark (\p{M}).
fn is_mark(c: char) -> bool {
    let c = c as u32;
    MARKS.binary_search_by(|&(a, b)| if b < c { std::cmp::Ordering::Less } else if a > c { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal }).is_ok()
}

/// Splits text the way the GPT-2 regex does:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
fn split_gpt2(text: &str, out: &mut Vec<String>) {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let run = |mut j: usize, pred: fn(char) -> bool| {
        while j < n && pred(chars[j]) {
            j += 1;
        }
        j
    };
    let mut i = 0;
    while i < n {
        let c = chars[i];
        let start = i;
        // contractions
        if c == '\'' && i + 1 < n {
            let rest: String = chars[i + 1..n.min(i + 3)].iter().collect();
            let len = if rest.starts_with("re") || rest.starts_with("ve") || rest.starts_with("ll") {
                3
            } else if matches!(chars[i + 1], 's' | 't' | 'm' | 'd') {
                2
            } else {
                0
            };
            if len > 0 {
                out.push(chars[i..i + len].iter().collect());
                i += len;
                continue;
            }
        }
        // optional leading space + letters / numbers / other symbols
        let (lead, body) = if c == ' ' && i + 1 < n { (1, chars[i + 1]) } else { (0, c) };
        let class: Option<fn(char) -> bool> = if is_letter(body) {
            Some(is_letter)
        } else if is_number(body) {
            Some(is_number)
        } else if is_other(body) {
            Some(is_other)
        } else {
            None
        };
        if let Some(pred) = class {
            if lead == 1 || !c.is_whitespace() {
                let end = run(i + lead, pred);
                out.push(chars[start..end].iter().collect());
                i = end;
                continue;
            }
        }
        // whitespace: \s+(?!\S) then \s+
        let end = run(i, char::is_whitespace);
        if end == n {
            out.push(chars[i..end].iter().collect());
            i = end;
        } else if end - i > 1 {
            out.push(chars[i..end - 1].iter().collect());
            i = end - 1;
        } else {
            // single whitespace char before a non-space: a space was handled above, anything else stands alone
            out.push(chars[i].to_string());
            i += 1;
        }
    }
}

/// Splits text the way the Qwen2 regex does:
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
/// with numbers in runs of up to `digits` (1 for Qwen, 3 for GLM-4), and with `marks` (Qwen3.5) combining
/// marks counted with the letters (`[\p{L}\p{M}]+`, and not among the symbols).
fn split_qwen2(text: &str, digits: usize, marks: bool, out: &mut Vec<String>) {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let is_letter = |c: char| is_letter(c) || marks && is_mark(c);
    let is_other = |c: char| is_other(c) && !(marks && is_mark(c));
    let is_nl = |c: char| c == '\r' || c == '\n';
    let run = |mut j: usize, pred: &dyn Fn(char) -> bool| {
        while j < n && pred(chars[j]) {
            j += 1;
        }
        j
    };
    let mut i = 0;
    while i < n {
        let c = chars[i];
        let piece_end: usize = 'm: {
            // 1. contractions, any case
            if c == '\'' && i + 1 < n {
                let a = chars[i + 1].to_ascii_lowercase();
                let b = chars.get(i + 2).map(|c| c.to_ascii_lowercase());
                if matches!((a, b), ('r', Some('e')) | ('v', Some('e')) | ('l', Some('l'))) {
                    break 'm i + 3;
                }
                if matches!(a, 's' | 't' | 'm' | 'd') {
                    break 'm i + 2;
                }
            }
            // 2. letters, optionally led by one char that is not a letter, digit or newline
            if is_letter(c) {
                break 'm run(i, &is_letter);
            }
            if !is_nl(c) && !is_number(c) && i + 1 < n && is_letter(chars[i + 1]) {
                break 'm run(i + 1, &is_letter);
            }
            // 3. a digit, or a run of up to `digits` of them
            if is_number(c) {
                break 'm run(i, &is_number).min(i + digits);
            }
            // 4. optional space, a run of symbols, then any newlines
            let start = if c == ' ' && i + 1 < n && is_other(chars[i + 1]) { i + 1 } else { i };
            if is_other(chars[start]) {
                let e = run(start, &is_other);
                break 'm run(e, &is_nl);
            }
            // 5-7. whitespace
            let ws_end = run(i, &char::is_whitespace);
            if let Some(last_nl) = (i..ws_end).rev().find(|&k| is_nl(chars[k])) {
                break 'm last_nl + 1; // \s*[\r\n]+ : up to and including the last newline
            }
            if ws_end == n || ws_end - i == 1 {
                break 'm ws_end; // \s+(?!\S) at the end of the text, or the single-char \s+ fallback
            }
            ws_end - 1 // \s+(?!\S): leave the last space to lead the next piece
        };
        out.push(chars[i..piece_end].iter().collect());
        i = piece_end;
    }
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
        let model = g.get("tokenizer.ggml.model").and_then(Value::as_str).unwrap_or("");
        if model != "gpt2" {
            return Err(format!("tokenizer model \"{model}\" is not supported yet (only byte-level BPE \"gpt2\")"));
        }
        let pre = match g.get("tokenizer.ggml.pre").and_then(Value::as_str).unwrap_or("default") {
            "smollm" | "starcoder" | "refact" => PreTokenizer::Gpt2IsolatedDigits,
            "qwen2" => PreTokenizer::Qwen2,
            // Llama 3's regex is GLM-4's
            "glm4" | "llama-bpe" => PreTokenizer::Glm4,
            "qwen35" => PreTokenizer::Qwen35,
            _ => PreTokenizer::Gpt2,
        };

        let tokens: Vec<String> = g
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or("missing tokenizer.ggml.tokens")?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
        let types: Vec<i64> = g
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| v.as_u64().map(|x| x as i64).unwrap_or(1)).collect())
            .unwrap_or_else(|| vec![1; tokens.len()]);

        let mut merge_rank = HashMap::new();
        if let Some(merges) = g.get("tokenizer.ggml.merges").and_then(Value::as_array) {
            for (rank, m) in merges.iter().enumerate() {
                if let Some((a, b)) = m.as_str().and_then(|s| s.split_once(' ')) {
                    merge_rank.insert((a.to_string(), b.to_string()), rank as u32);
                }
            }
        }

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut specials = Vec::new();
        let mut is_special = vec![false; tokens.len()];
        for (id, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), id as u32);
            let ty = types.get(id).copied().unwrap_or(1);
            if ty == TOKEN_TYPE_CONTROL || ty == TOKEN_TYPE_USER_DEFINED {
                is_special[id] = true;
                if !t.is_empty() {
                    specials.push((t.clone(), id as u32));
                }
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let byte_to_char = byte_level_table();
        let char_to_byte = byte_to_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let id = |k: &str| g.get(k).and_then(Value::as_u64).map(|v| v as u32);

        Ok(Tokenizer {
            tokens,
            token_to_id,
            merge_rank,
            specials,
            is_special,
            byte_to_char,
            char_to_byte,
            pre,
            bos_id: id("tokenizer.ggml.bos_token_id"),
            eos_id: id("tokenizer.ggml.eos_token_id"),
            // unsaid, Llama 3's vocabulary starts with <|begin_of_text|> as llama.cpp's does: its fine-tunes' files often
            // leave the flag out, and without it a long prompt sends the model off the rails
            add_bos: g.get("tokenizer.ggml.add_bos_token").and_then(Value::as_bool).unwrap_or(g.get("tokenizer.ggml.pre").and_then(Value::as_str) == Some("llama-bpe")),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.is_special.get(id as usize).copied().unwrap_or(false)
    }

    /// Encodes text; special tokens written literally (e.g. `<|im_start|>`) become their single id. The
    /// beginning-of-text token is added first, unless the text writes it itself (a prompt that places it, such as
    /// maya1's after its own start marker, would otherwise have two).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let written = self.bos_id.filter(|&b| self.is_special(b)).and_then(|b| self.tokens.get(b as usize)).is_some_and(|t| text.contains(t.as_str()));
        if self.add_bos && !written {
            ids.extend(self.bos_id);
        }
        let mut rest = text;
        while !rest.is_empty() {
            // find the earliest special token occurrence (longest wins on ties)
            let next = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|pos| (pos, s.len(), *id)))
                .min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
            match next {
                Some((pos, len, id)) => {
                    self.encode_plain(&rest[..pos], &mut ids);
                    ids.push(id);
                    rest = &rest[pos + len..];
                }
                None => {
                    self.encode_plain(rest, &mut ids);
                    break;
                }
            }
        }
        ids
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut pieces = Vec::new();
        match self.pre {
            PreTokenizer::Gpt2 => split_gpt2(text, &mut pieces),
            PreTokenizer::Qwen2 => split_qwen2(text, 1, false, &mut pieces),
            PreTokenizer::Glm4 => split_qwen2(text, 3, false, &mut pieces),
            PreTokenizer::Qwen35 => split_qwen2(text, 1, true, &mut pieces),
            PreTokenizer::Gpt2IsolatedDigits => {
                let mut chunk = String::new();
                for c in text.chars() {
                    if c.is_numeric() {
                        if !chunk.is_empty() {
                            split_gpt2(&chunk, &mut pieces);
                            chunk.clear();
                        }
                        pieces.push(c.to_string());
                    } else {
                        chunk.push(c);
                    }
                }
                if !chunk.is_empty() {
                    split_gpt2(&chunk, &mut pieces);
                }
            }
        }
        for piece in pieces {
            self.bpe(&piece, ids);
        }
    }

    fn bpe(&self, piece: &str, ids: &mut Vec<u32>) {
        let mapped: String = piece.bytes().map(|b| self.byte_to_char[b as usize]).collect();
        if let Some(&id) = self.token_to_id.get(&mapped) {
            ids.push(id);
            return;
        }
        let mut parts: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();
        while parts.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len() - 1 {
                if let Some(&rank) = self.merge_rank.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.is_none_or(|(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts[i] = merged;
            parts.remove(i + 1);
        }
        for p in parts {
            match self.token_to_id.get(&p) {
                Some(&id) => ids.push(id),
                // should not happen with a byte-level vocab; fall back to single characters
                None => ids.extend(p.chars().filter_map(|c| self.token_to_id.get(&c.to_string()).copied())),
            }
        }
    }

    /// Raw bytes for a token. Special tokens give their literal text only when `show_special` is set.
    pub fn token_bytes(&self, id: u32, show_special: bool) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else { return Vec::new() };
        if self.is_special(id) {
            return if show_special { t.as_bytes().to_vec() } else { Vec::new() };
        }
        t.chars().map(|c| self.char_to_byte.get(&c).copied().unwrap_or(b'?')).collect()
    }

    pub fn decode(&self, ids: &[u32], show_special: bool) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.token_bytes(id, show_special)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn token_text(&self, id: u32) -> &str {
        self.tokens.get(id as usize).map(String::as_str).unwrap_or("")
    }
}

/// Turns a stream of token bytes into printable UTF-8, holding back incomplete multi-byte characters.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                // an invalid (not just incomplete) sequence: emit it lossily rather than stall forever
                let cut = if e.error_len().is_some() { self.pending.len() } else { valid };
                let s = String::from_utf8_lossy(&self.pending[..cut]).into_owned();
                self.pending.drain(..cut);
                s
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(s: &str) -> Vec<String> {
        let mut v = Vec::new();
        split_gpt2(s, &mut v);
        v
    }

    #[test]
    fn gpt2_split_matches_regex() {
        assert_eq!(split("Hello world"), ["Hello", " world"]);
        assert_eq!(split("I'm here's"), ["I", "'m", " here", "'s"]);
        assert_eq!(split("a  b"), ["a", " ", " b"]);
        assert_eq!(split("x\n\ny"), ["x", "\n", "\n", "y"]);
        assert_eq!(split("end   "), ["end", "   "]);
        assert_eq!(split("hi!!! ok?"), ["hi", "!!!", " ok", "?"]);
        assert_eq!(split(" 42"), [" 42"]);
    }

    #[test]
    fn qwen2_split_matches_regex() {
        let split = |s: &str| {
            let mut v = Vec::new();
            split_qwen2(s, 1, false, &mut v);
            v
        };
        assert_eq!(split("Hello world"), ["Hello", " world"]);
        assert_eq!(split("I'M here"), ["I", "'M", " here"]);
        assert_eq!(split("(hello) 123"), ["(hello", ")", " ", "1", "2", "3"]);
        assert_eq!(split("a.\n\nb"), ["a", ".\n\n", "b"]);
        assert_eq!(split("x  \n  y"), ["x", "  \n", " ", " y"]);
        assert_eq!(split("x\n\ny"), ["x", "\n\n", "y"]);
        assert_eq!(split("a  b"), ["a", " ", " b"]);
        assert_eq!(split("end   "), ["end", "   "]);
        assert_eq!(split("hi !!"), ["hi", " !!"]);
    }

    #[test]
    fn glm4_split_groups_digits_by_three() {
        let mut v = Vec::new();
        split_qwen2("(x) 1234567 a1", 3, false, &mut v);
        assert_eq!(v, ["(x", ")", " ", "123", "456", "7", " a", "1"]);
    }

    #[test]
    fn qwen35_split_keeps_marks_with_letters() {
        let split = |s: &str, marks: bool| {
            let mut v = Vec::new();
            split_qwen2(s, 1, marks, &mut v);
            v
        };
        // e and a combining acute (U+0301), then a mark alone after a space
        let text = "cafe\u{301} x \u{301}\u{301}!";
        assert_eq!(split(text, true), ["cafe\u{301}", " x", " \u{301}\u{301}", "!"]);
        assert_eq!(split(text, false), ["cafe", "\u{301}", " x", " \u{301}\u{301}!"]);
        assert!(is_mark('\u{301}') && is_mark('\u{94d}') && !is_mark('e') && !is_mark('\u{2ff}'));
    }

    #[test]
    fn byte_table_is_a_bijection() {
        let t = byte_level_table();
        let set: std::collections::HashSet<char> = t.iter().copied().collect();
        assert_eq!(set.len(), 256);
        assert_eq!(t[b' ' as usize], 'Ġ');
        assert_eq!(t[b'\n' as usize], 'Ċ');
    }

    #[test]
    fn utf8_stream_holds_partial_chars() {
        let mut s = Utf8Stream::default();
        let euro = "€".as_bytes();
        assert_eq!(s.push(&euro[..1]), "");
        assert_eq!(s.push(&euro[1..]), "€");
    }
}
