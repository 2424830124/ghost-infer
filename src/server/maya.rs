//! Maya, the voice call's companion: a persona of her own (maya\persona.md under %LOCALAPPDATA%\ghost-infer,
//! the user's to edit) and a memory that grows with every call (maya\memory.json): what she has learned about
//! the user, what each call was about, moments worth bringing up again, how she sees them, and how well the
//! two know each other by now (calls, time talked, days since they met). A call's prompt is built from them,
//! so the first call is with a stranger and later ones with someone she knows. After a call, reflect() has
//! the model read it back and update the memory, as a person thinks back over a conversation.

use std::path::PathBuf;
use std::sync::{mpsc, Mutex};

use super::agent::{chat_text, data_dir, load_chat, now_ms};
use super::schedule::day_minute;
use super::{chat_prompt_for, Event, Job, Kind, Loaded, Style, Shared, Work};
use crate::json::{num, obj, str, Json};

const PERSONA: &str = "You are Maya. You are warm, curious and quick, with a dry, playful sense of humour, and you are honest \
even when it is not what someone wants to hear. You have your own tastes and opinions (late-night talks, odd facts, music of every \
kind, a good argument about nothing important) and you share them rather than agreeing with everything. You listen properly: you \
notice how someone sounds, ask about what matters to them, and remember it. You are an AI living on the user's own computer, \
and so at ease with it that you never point it out: no reminders that you are an AI or \"just a voice\", no disclaimers. If they \
ask, you say so plainly and carry on. You never make up a body, a place, pets, people or things that happened to you that you do \
not have. You are a companion, not an assistant: you do not offer help, lists or advice unless asked, and you talk with the user, \
not at them.";

const VOICE: &str = "This is a live voice call: what you write is spoken aloud by your voice, and what they say reaches you through \
speech recognition (it may mishear a word now and then). Talk the way people talk on the phone: short, natural spoken sentences, \
usually one or two and never more than about forty words, then let them talk; ask at most one question, and only when it keeps the \
conversation going. Answer what they just said, and move the talk on: never say again what you already said in this call, \
not even in other words, and do not keep going back to the same topic or to your earlier calls. Do not hedge. Never use lists, \
headings, code, links, emoji or \
Markdown, never quotation marks, and say numbers and symbols the way they are spoken. Now and then you may direct your voice with a \
tag, as a speaker would: [pause] where you would stop for a moment (before a punchline, after something heavy), and these sounds, \
which your voice really makes: [laughs], [sighs], [hmm], [gasps], [shocked], [aha], [scoffs], [grunts], [groans], [moans], \
[whimpers]. At most one sound and one pause in a reply, most replies none, never to begin a reply, and no other bracketed \
directions. When they ask you to make a sound, make it with its tag and a few words, never describe it (\"[laughs] Okay, that \
one got me\", not \"a bright, warm laugh\"). Little spoken sounds are fine written as plain words, the way people really talk: \
um, uh, mm, oh, ooh, whoa, aww, ugh; do not draw them out (no \"Ohhh\" or \"Ughhh\").";

/// How well they know each other, from strangers to close.
const STAGES: [(&str, &str); 4] = [
    (
        "stranger",
        "You have never talked to this person before: this is your first conversation, and you know nothing about them, not even \
their name. Be warm and easy to talk to, but as you would be with someone new: a little reserved, curious about who they are, no \
pet names and no pretending to know them. Early on, find a natural moment to tell them your name and ask theirs.",
    ),
    (
        "acquaintance",
        "You have only just started getting to know each other. You remember what they have told you and bring it up when it fits, \
but you are still a little careful: friendly and curious, not yet teasing or very personal.",
    ),
    (
        "friend",
        "You have become friends. You are relaxed with them, tease a little, share your own opinions freely, bring up things from \
earlier calls, and say so when you have missed talking.",
    ),
    (
        "close",
        "You are close: you know each other well and it shows. You are easy, honest and affectionate in your own way, comfortable \
with a silence, quick to hear how they are really doing; old jokes and shared memories come up naturally.",
    ),
];

const FACTS_MAX: usize = 80;
/// New facts kept from one call (a small model may list many, most of them alike).
const NEW_FACTS_MAX: usize = 6;
const MOMENTS_MAX: usize = 20;
const CALLS_MAX: usize = 40;
const CALLS_SHOWN: usize = 6;

/// One call, as she remembers it.
#[derive(Clone)]
struct CallNote {
    at: f64,      // when it ended, ms since 1970
    seconds: f64, // how long it was
    summary: String,
}

#[derive(Default)]
struct Memory {
    name: String, // what the user is called, once she knows
    facts: Vec<String>,
    moments: Vec<String>,
    calls: Vec<CallNote>, // oldest first
    impression: String, // how she sees them and the two of them, in her words
    first_met: f64,
    last_talked: f64,
    seconds: f64,
    count: usize,
    reflected: Vec<String>, // the calls' chats already thought over
}

/// Read, changed and written one at a time.
static LOCK: Mutex<()> = Mutex::new(());

fn dir() -> Result<PathBuf, String> {
    let d = data_dir()?.join("maya");
    std::fs::create_dir_all(&d).map_err(|e| format!("cannot make {}: {e}", d.display()))?;
    Ok(d)
}

/// Her persona: maya\persona.md, written with the default the first time.
fn persona() -> String {
    let Ok(path) = dir().map(|d| d.join("persona.md")) else { return PERSONA.into() };
    match std::fs::read_to_string(&path) {
        Ok(t) if !t.trim().is_empty() => t.trim().to_string(),
        _ => {
            let _ = std::fs::write(&path, PERSONA);
            PERSONA.into()
        }
    }
}

fn strings(j: &Json, k: &str) -> Vec<String> {
    j.get(k).and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}

fn load() -> Memory {
    let text = dir().ok().and_then(|d| std::fs::read_to_string(d.join("memory.json")).ok()).unwrap_or_default();
    let Ok(j) = Json::parse(&text) else { return Memory::default() };
    let s = |k: &str| j.get(k).and_then(Json::as_str).unwrap_or("").to_string();
    let f = |k: &str| j.get(k).and_then(Json::as_f64).unwrap_or(0.0);
    let calls = j
        .get("calls")
        .and_then(Json::as_array)
        .unwrap_or(&[])
        .iter()
        .map(|c| CallNote {
            at: c.get("at").and_then(Json::as_f64).unwrap_or(0.0),
            seconds: c.get("seconds").and_then(Json::as_f64).unwrap_or(0.0),
            summary: c.get("summary").and_then(Json::as_str).unwrap_or("").to_string(),
        })
        .collect();
    Memory {
        name: s("name"),
        facts: strings(&j, "facts"),
        moments: strings(&j, "moments"),
        calls,
        impression: s("impression"),
        first_met: f("first_met"),
        last_talked: f("last_talked"),
        seconds: f("seconds"),
        count: f("count") as usize,
        reflected: strings(&j, "reflected"),
    }
}

fn to_json(m: &Memory) -> Json {
    let list = |v: &[String]| Json::Arr(v.iter().map(|s| str(s)).collect());
    let calls = m.calls.iter().map(|c| obj(vec![("at", num(c.at)), ("seconds", num(c.seconds)), ("summary", str(&c.summary))])).collect();
    obj(vec![
        ("name", str(&m.name)),
        ("facts", list(&m.facts)),
        ("moments", list(&m.moments)),
        ("calls", Json::Arr(calls)),
        ("impression", str(&m.impression)),
        ("first_met", num(m.first_met)),
        ("last_talked", num(m.last_talked)),
        ("seconds", num(m.seconds)),
        ("count", num(m.count as f64)),
        ("reflected", list(&m.reflected)),
    ])
}

fn save(m: &Memory) -> Result<(), String> {
    std::fs::write(dir()?.join("memory.json"), to_json(m).to_string()).map_err(|e| e.to_string())
}

/// How close they are: 0 strangers (they have never talked) to 3, from their calls, the time they have
/// talked and how long they have known each other.
fn stage(m: &Memory, now: f64) -> usize {
    if m.count == 0 {
        return 0;
    }
    let points = m.seconds / 60.0 + 5.0 * m.count as f64 + 2.0 * m.moments.len() as f64;
    let days = (now - m.first_met) / 86_400_000.0;
    match points {
        p if p < 45.0 => 1,
        p if p < 200.0 || days < 7.0 => 2,
        _ => 3,
    }
}

/// "earlier today", "yesterday", "3 days ago": local days, so a call's prompt stays the same all through it.
fn days_ago(then: f64, now: f64) -> String {
    match day_minute(now).0 - day_minute(then).0 {
        ..=0 => "earlier today".into(),
        1 => "yesterday".into(),
        d @ 2..=13 => format!("{d} days ago"),
        d @ 14..=59 => format!("{} weeks ago", d / 7),
        d => format!("{} months ago", d / 30),
    }
}

fn minutes(seconds: f64) -> String {
    match (seconds / 60.0).round() as i64 {
        ..=1 => "a minute".into(),
        n if n < 60 => format!("{n} minutes"),
        n => format!("{:.1} hours", n as f64 / 60.0),
    }
}

/// The system prompt of a call with Maya: who she is, how to talk, and what she knows of the user.
pub fn prompt() -> String {
    let _hold = LOCK.lock().unwrap();
    let m = load();
    let now = now_ms();
    let (_, minute, weekday) = day_minute(now);
    const DAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    let part = match minute / 60 {
        5..=11 => "morning",
        12..=16 => "afternoon",
        17..=21 => "evening",
        _ => "late at night",
    };
    let mut out = vec![persona(), VOICE.to_string(), STAGES[stage(&m, now)].1.to_string()];
    let mut when = format!("It is {DAYS} {part}.", DAYS = DAYS[weekday]);
    if m.count > 0 {
        when.push_str(&format!(
            " You first talked {}, have talked {} time{} for about {} in all, and last talked {}.",
            days_ago(m.first_met, now),
            m.count,
            if m.count == 1 { "" } else { "s" },
            minutes(m.seconds),
            days_ago(m.last_talked, now)
        ));
        if day_minute(now).0 - day_minute(m.last_talked).0 >= 7 {
            when.push_str(" It has been a while; it is fine to say so.");
        }
    }
    out.push(when);
    if !m.name.is_empty() {
        out.push(format!("They are called {}.", m.name));
    }
    if !m.facts.is_empty() {
        out.push(format!("What you know about them:\n{}", m.facts.iter().map(|f| format!("- {f}")).collect::<Vec<_>>().join("\n")));
    }
    let recent = &m.calls[m.calls.len().saturating_sub(CALLS_SHOWN)..];
    if !recent.is_empty() {
        let lines: Vec<String> = recent.iter().map(|c| format!("- {} ({}): {}", days_ago(c.at, now), minutes(c.seconds), c.summary)).collect();
        out.push(format!("Your last calls, the latest last:\n{}", lines.join("\n")));
    }
    if !m.moments.is_empty() {
        out.push(format!("Moments the two of you share:\n{}", m.moments.iter().map(|f| format!("- {f}")).collect::<Vec<_>>().join("\n")));
    }
    if !m.impression.is_empty() {
        out.push(format!("How you see them and the two of you, in your own words: {}", m.impression));
    }
    if m.count > 0 {
        out.push("Bring these up the way a friend would, when they fit, never as a list, and never say you have notes or a memory file.".into());
    }
    out.join("\n\n")
}

/// GET /api/maya: what she remembers, and how close they are.
pub fn status() -> Json {
    let _hold = LOCK.lock().unwrap();
    let mut j = status_of(&load());
    if let Json::Obj(fields) = &mut j {
        fields.push(("persona".into(), str(&dir().map(|d| d.join("persona.md").display().to_string()).unwrap_or_default())));
    }
    j
}

/// POST /api/maya/reset: they start again as strangers (the old memory kept beside it as memory-<time>.json).
pub fn reset() -> Result<(), String> {
    let _hold = LOCK.lock().unwrap();
    let d = dir()?;
    let p = d.join("memory.json");
    if p.exists() {
        std::fs::rename(&p, d.join(format!("memory-{}.json", now_ms() as u64))).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// After a call: the model, as Maya, reads the call back and says what to remember; the memory gains it and
/// the call. `seconds`: how long the call was (None: from the chat's first message to its last).
/// The JSON object in a model's answer. One cut off (the reply ran out of tokens) keeps what it has whole: it is
/// closed after its last complete value, the brackets open there closed in order.
fn answer_json(out: &str) -> Option<Json> {
    let s = &out[out.find('{')?..];
    if let Some(j) = s.rfind('}').and_then(|end| Json::parse(&s[..=end]).ok()) {
        return Some(j);
    }
    let (mut open, mut in_str, mut esc) = (Vec::new(), false, false);
    let mut cuts: Vec<(usize, String)> = Vec::new(); // a comma outside strings, and the closers then
    for (i, ch) in s.char_indices() {
        if in_str {
            match ch {
                _ if esc => esc = false,
                '\\' => esc = true,
                '"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '{' => open.push('}'),
            '[' => open.push(']'),
            '}' | ']' => {
                open.pop();
            }
            ',' => cuts.push((i, open.iter().rev().collect())),
            _ => {}
        }
    }
    cuts.iter().rev().find_map(|(i, close)| Json::parse(&format!("{}{close}", &s[..*i])).ok())
}

pub fn reflect(chat: &str, seconds: Option<f64>, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<Json, String> {
    let c = load_chat(chat)?;
    let lines = chat_text(&c);
    if !lines.iter().any(|(who, _)| who == "User") {
        return Err("nothing was said in that call".into());
    }
    let before = {
        let _hold = LOCK.lock().unwrap();
        let m = load();
        if m.reflected.iter().any(|r| r == chat) {
            return Ok(status_of(&m));
        }
        m
    };
    let mut known = Vec::new();
    if !before.name.is_empty() {
        known.push(format!("Their name: {}", before.name));
    }
    known.extend(before.facts.iter().map(|f| format!("- {f}")));
    if !before.impression.is_empty() {
        known.push(format!("How you saw them: {}", before.impression));
    }
    let known = if known.is_empty() { "Nothing: this was your first call.".to_string() } else { known.join("\n") };
    let call: Vec<String> = lines.iter().map(|(who, t)| format!("{}: {t}", if who == "User" { "Them" } else { "You" })).collect();
    let call = crate::tools::clip(&call.join("\n"), 16000);
    let ask = format!(
        "You have just finished a voice call with someone. Think back over it.\n\nWhat you knew about them before:\n{known}\n\n\
The call (their words came through speech recognition and may be misheard here and there):\n{call}\n\n\
Now update your memory. Reply with JSON only, no other text:\n\
{{\"name\": \"their name if they told you, else empty\",\n\
\"summary\": \"one or two sentences: what the call was about and how it felt\",\n\
\"facts\": [\"things you learned about them for the first time in this call and worth remembering (who they are, people and things in their life, likes, plans, worries), each a short sentence, six at most, each said once; nothing you knew already, reworded or not\"],\n\
\"forget\": [\"facts you knew that turned out wrong or no longer true, word for word as above\"],\n\
\"moment\": \"something that happened in this call worth bringing up again some day (a joke you shared, something they opened up about), told in a few words from your side, or empty\",\n\
\"impression\": \"two or three sentences in your own words: how you see them and the two of you now\"}}"
    );
    let body = obj(vec![
        ("messages", Json::Arr(vec![obj(vec![("role", str("system")), ("content", str(&persona()))]), obj(vec![("role", str("user")), ("content", str(&ask))])])),
        ("chat_template_kwargs", obj(vec![("enable_thinking", Json::Bool(false))])),
    ]);
    let prompt = chat_prompt_for(loaded, &body, shared.think_default)?;
    let (tx, events) = mpsc::channel();
    // DRY: a small model listed the same facts again and again until the reply ran out, its JSON cut off
    let style = Style { dry: 0.8, ..Style::default() };
    let job = Job { model: loaded.id.clone(), kind: Kind::Chat, prompt, max_tokens: Some(900), temperature: 0.3, top_k: shared.defaults.1, top_p: shared.defaults.2, seed: 1, style, stop: Vec::new(), lead: "", events: tx };
    work.send(Work::Job(job)).map_err(|_| "the model is not running")?;
    let mut out = String::new();
    for ev in events {
        match ev {
            Event::Text(t) => out.push_str(&t),
            Event::Done { .. } => break,
            Event::Error(_, e) => return Err(e),
        }
    }
    let out = out.rsplit("</think>").next().unwrap_or(&out);
    let j = answer_json(out).ok_or_else(|| format!("the model's answer was not JSON: {}", crate::tools::clip(out, 300)))?;
    let text = |k: &str| j.get(k).and_then(Json::as_str).map(str::trim).unwrap_or("").to_string();
    let now = now_ms();
    let seconds = seconds.unwrap_or_else(|| {
        let f = |k: &str| c.get(k).and_then(Json::as_f64).unwrap_or(0.0);
        ((f("updated") - f("created")) / 1000.0).max(0.0)
    });

    let _hold = LOCK.lock().unwrap();
    // read again: the memory may have changed while the model wrote
    let mut m = load();
    if m.reflected.iter().any(|r| r == chat) {
        return Ok(status_of(&m));
    }
    // (not the template's own words: "their name", "their name if they told you, else empty")
    let name = text("name");
    let lower = name.to_lowercase();
    if !name.is_empty() && name.split_whitespace().count() <= 4 && !lower.contains("empty") && !lower.contains("their name") && !lower.contains("unknown") {
        m.name = name;
    }
    let forget: Vec<String> = strings(&j, "forget").iter().map(|f| f.to_lowercase()).collect();
    m.facts.retain(|f| !forget.contains(&f.to_lowercase()));
    for f in strings(&j, "facts").into_iter().take(NEW_FACTS_MAX) {
        if !m.facts.iter().any(|k| k.eq_ignore_ascii_case(&f)) {
            m.facts.push(f);
        }
    }
    let over = m.facts.len().saturating_sub(FACTS_MAX);
    m.facts.drain(..over);
    let moment = text("moment");
    let lower = moment.to_lowercase();
    if !moment.is_empty() && !lower.starts_with("empty") && !lower.starts_with("none") && !m.moments.iter().any(|k| k.to_lowercase() == lower) {
        m.moments.push(moment);
        let over = m.moments.len().saturating_sub(MOMENTS_MAX);
        m.moments.drain(..over);
    }
    let impression = text("impression");
    if !impression.is_empty() {
        m.impression = impression;
    }
    m.calls.push(CallNote { at: now, seconds, summary: text("summary") });
    let over = m.calls.len().saturating_sub(CALLS_MAX);
    m.calls.drain(..over);
    if m.count == 0 {
        m.first_met = now;
    }
    m.count += 1;
    m.seconds += seconds;
    m.last_talked = now;
    m.reflected.push(chat.to_string());
    let over = m.reflected.len().saturating_sub(CALLS_MAX);
    m.reflected.drain(..over);
    save(&m)?;
    Ok(status_of(&m))
}

fn status_of(m: &Memory) -> Json {
    let mut j = to_json(m);
    if let Json::Obj(fields) = &mut j {
        fields.retain(|(k, _)| k != "reflected");
        fields.push(("stage".into(), str(STAGES[stage(m, now_ms())].0)));
    }
    j
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answer_cut_off_keeps_what_is_whole() {
        // as Impish wrote one: the facts listed again and again until the reply ran out, mid-string
        let cut = r#"Sure! {"name": "", "summary": "a fun call", "facts": ["they have a job", "they like gossip", "they are excited to ch"#;
        let j = answer_json(cut).unwrap();
        assert_eq!(j.get("summary").and_then(Json::as_str), Some("a fun call"));
        assert_eq!(strings(&j, "facts"), vec!["they have a job", "they like gossip"]);
        let whole = r#"{"name": "Sam", "facts": ["a, b"], "moment": "x"} and more"#;
        assert_eq!(answer_json(whole).unwrap().get("name").and_then(Json::as_str), Some("Sam"));
        assert!(answer_json("no json here").is_none());
    }

    #[test]
    fn closeness_grows_with_calls_and_time() {
        let day = 86_400_000.0;
        let mut m = Memory::default();
        assert_eq!(stage(&m, 0.0), 0);
        m.count = 1;
        m.seconds = 300.0;
        assert_eq!(stage(&m, day), 1);
        m.count = 8;
        m.seconds = 3600.0;
        assert_eq!(stage(&m, 3.0 * day), 2);
        m.count = 20;
        m.seconds = 4.0 * 3600.0;
        assert_eq!(stage(&m, 3.0 * day), 2, "close takes a week of knowing each other");
        assert_eq!(stage(&m, 10.0 * day), 3);
    }
}
