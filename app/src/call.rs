//! A voice call. The microphone's utterances (voice.rs) become text through the server's speech-to-text
//! (/v1/audio/transcriptions); each is sent to the agent as a message, in a chat of its own, without tools, for
//! Maya to answer (the server's companion persona, who remembers earlier calls); the answer is spoken a sentence at a time as it is written
//! (/v1/audio/speech), in the chat's own voice context, so the voice carries on from one reply to the next.
//! While the call's voice plays, the user can cut in (a headset: the voice does not reach the microphone): it
//! stops at once, and the server keeps her reply as far as it was heard. On speakers (GHOST_APP_NO_BARGE_IN=1)
//! the microphone does not take what it hears while the voice plays for the user (no echo canceller: one side
//! talks at a time). Maya opens the call, and speaks up herself when the user stays quiet (QUIET_SECS).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use iced::futures::channel::mpsc as fmpsc;

use crate::api::Api;
use crate::json::{obj, str, Json};
use crate::voice::{self, Heard, Shared, Speaker};

/// What the call tells the window.
#[derive(Debug, Clone)]
pub enum CallEvent {
    /// listening, hearing you, thinking, speaking
    State(&'static str),
    /// what was heard
    You(String),
    /// a reply of hers that answers nothing the user said (she opened the call, or spoke up)
    Her,
    /// the reply so far
    Ghost(String),
    /// her mood and her way of talking, as the reply begins
    Mood(String, String),
    /// the chat the call writes in
    Chat(String),
    /// how long the last turn took, step by step
    Timing(String),
    Error(String),
    Ended,
}

/// The running call: what the window shows from it (the microphone's level, the voice playing), and its end.
pub struct Call {
    pub shared: Arc<Shared>,
    pub speaker: Arc<Speaker>,
    ended: Arc<AtomicBool>,
    typed: mpsc::Sender<String>,
}

impl Call {
    /// A typed message, answered aloud as a spoken one is (after the answer under way, if there is one).
    pub fn send_text(&self, text: String) {
        let _ = self.typed.send(text);
    }

    pub fn end(&self) {
        self.ended.store(true, Ordering::Relaxed);
        self.shared.stop.store(true, Ordering::Relaxed);
        self.speaker.stop();
    }

    pub fn set_muted(&self, muted: bool) {
        self.shared.muted.store(muted, Ordering::Relaxed);
    }

    /// The user is typing a message: she does not speak up meanwhile.
    pub fn set_typing(&self, typing: bool) {
        self.shared.typing.store(typing, Ordering::Relaxed);
    }

    pub fn muted(&self) -> bool {
        self.shared.muted.load(Ordering::Relaxed)
    }

    /// The voice is playing.
    pub fn talking(&self) -> bool {
        self.speaker.queued() > 0.0
    }
}

/// Opens the microphone and the speaker and starts the call; `chat`: the chat to continue (None: a new one).
pub fn start(api: Api, chat: Option<String>, mode: String, events: fmpsc::UnboundedSender<CallEvent>) -> Result<Call, String> {
    let shared = Arc::new(Shared::default());
    shared.barge.store(std::env::var_os("GHOST_APP_NO_BARGE_IN").is_none_or(|v| v == "0"), Ordering::Relaxed);
    let speaker = Arc::new(Speaker::start(shared.clone())?);
    let (heard_tx, heard) = mpsc::channel();
    if let Err(e) = voice::start_mic(shared.clone(), heard_tx) {
        shared.stop.store(true, Ordering::Relaxed);
        return Err(e);
    }
    let ended = Arc::new(AtomicBool::new(false));
    let (typed_tx, typed) = mpsc::channel();
    let call = Call { shared: shared.clone(), speaker: speaker.clone(), ended: ended.clone(), typed: typed_tx };
    std::thread::spawn(move || {
        run(api, chat, mode, heard, typed, speaker, shared, ended.clone(), &events);
        let _ = events.unbounded_send(CallEvent::Ended);
    });
    Ok(call)
}

/// Quiet after a turn before Maya speaks up herself (doubled each time it stays quiet), and how many times in a
/// row she does before she waits for the user.
const QUIET_SECS: u64 = 20;
const QUIET_TURNS: u32 = 2;
/// The call's own notes to Maya (hidden from the chat): the call has started, the user has gone quiet.
const OPENING: &str = "(The call has just connected. Say hello your way.)";
const QUIET: &str = "(They have gone quiet. Your turn: say something of your own.)";

/// This turn's voice so far: seconds of audio queued, and each piece's words with where its audio starts and ends.
#[derive(Default)]
struct Voiced {
    total: f64,
    pieces: Vec<(String, f64, f64)>,
}

impl Voiced {
    /// The words heard when `unplayed` seconds were still to play: the pieces played, and of the one playing
    /// as many words as its share of time played.
    fn said(&self, unplayed: f64) -> String {
        let played = self.total - unplayed;
        let mut out = Vec::new();
        for (text, start, end) in &self.pieces {
            if *end <= played {
                out.push(text.clone());
                continue;
            }
            if *start < played {
                let words: Vec<&str> = text.split_whitespace().collect();
                let n = ((played - start) / (end - start) * words.len() as f64) as usize;
                out.push(words[..n.min(words.len())].join(" "));
            }
            break;
        }
        without_tags(&out.join(" "))
    }
}

#[allow(clippy::too_many_arguments)]
fn run(api: Api, mut chat: Option<String>, mode: String, heard: mpsc::Receiver<Heard>, typed: mpsc::Receiver<String>, speaker: Arc<Speaker>, shared: Arc<Shared>, ended: Arc<AtomicBool>, events: &fmpsc::UnboundedSender<CallEvent>) {
    let send = |e: CallEvent| events.unbounded_send(e).is_ok();
    // sentences to speak, one after another, on a thread of their own (while the model writes the next ones);
    // each with its turn: a turn cut off is not spoken on
    let (say_tx, say_rx) = mpsc::channel::<(String, String, usize)>();
    let pending = Arc::new(AtomicUsize::new(0));
    let turn = Arc::new(AtomicUsize::new(0));
    let voiced: Arc<Mutex<Voiced>> = Arc::default();
    // when this turn's first sound arrived, for its timings
    let first_audio: Arc<Mutex<Option<Instant>>> = Arc::default();
    {
        let (api, speaker, pending, ended, first_audio, turn, voiced) = (api.clone(), speaker.clone(), pending.clone(), ended.clone(), first_audio.clone(), turn.clone(), voiced.clone());
        let events = events.clone();
        std::thread::spawn(move || {
            for (text, conversation, t) in say_rx {
                let live = || !ended.load(Ordering::Relaxed) && turn.load(Ordering::Relaxed) == t;
                if live() {
                    {
                        let mut v = voiced.lock().unwrap();
                        let at = v.total;
                        v.pieces.push((text.clone(), at, at));
                    }
                    let body = obj(vec![("input", str(&text)), ("voice", str("maya")), ("response_format", str("pcm")), ("conversation", str(&conversation))]);
                    let mut odd: Option<u8> = None;
                    let r = api.stream_bytes("/v1/audio/speech", &body, |chunk| {
                        if !live() {
                            return false;
                        }
                        first_audio.lock().unwrap().get_or_insert_with(Instant::now);
                        let mut bytes = Vec::with_capacity(chunk.len() + 1);
                        bytes.extend(odd.take());
                        bytes.extend_from_slice(chunk);
                        if bytes.len() % 2 == 1 {
                            odd = bytes.pop();
                        }
                        let samples: Vec<f32> = bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect();
                        speaker.play(&samples, 24_000);
                        let mut v = voiced.lock().unwrap();
                        v.total += samples.len() as f64 / 24_000.0;
                        let total = v.total;
                        if let Some(p) = v.pieces.last_mut() {
                            p.2 = total;
                        }
                        true
                    });
                    if let Err(e) = r {
                        let _ = events.unbounded_send(CallEvent::Error(format!("The voice failed: {e}")));
                    }
                }
                pending.fetch_sub(1, Ordering::Relaxed);
            }
        });
    }
    // the user's voice while Maya talks or thinks: kept for the turns after it; true when it cuts her off (a
    // voice over hers, or words while she thinks)
    let barged = |held: &mut VecDeque<Heard>| {
        let mut cut = false;
        while let Ok(h) = heard.try_recv() {
            cut |= matches!(h, Heard::Utterance(..)) || (matches!(h, Heard::Started) && shared.speaking());
            held.push_back(h);
        }
        cut
    };
    // she stops at once: the words heard so far are what she said
    let cut_off = || {
        turn.fetch_add(1, Ordering::Relaxed);
        let said = voiced.lock().unwrap().said(speaker.queued() as f64);
        speaker.stop();
        said
    };
    send(CallEvent::State("Listening"));
    // a transcript started at a pause, in case the pause is the end
    let mut ahead: Option<mpsc::Receiver<Result<String, String>>> = None;
    let mut held: VecDeque<Heard> = VecDeque::new();
    // her last reply was cut off after these words (the server keeps it so)
    let mut cut: Option<String> = None;
    // since when nobody has spoken, and how many times since the user last did she spoke up herself
    let (mut quiet_since, mut nudges, mut opening) = (Instant::now(), 0u32, true);
    while !ended.load(Ordering::Relaxed) {
        // typing is not quiet: the wait starts again once it is sent or cleared
        if shared.typing.load(Ordering::Relaxed) {
            quiet_since = Instant::now();
        }
        let quiet = held.is_empty() && !shared.muted.load(Ordering::Relaxed) && nudges < QUIET_TURNS && quiet_since.elapsed() >= Duration::from_secs(QUIET_SECS << nudges);
        // a turn's words: the call's note, typed, or heard (then `spoke`: when the voice stopped, for the timings)
        let (text, spoke, heard_at, text_at, used_ahead, was_typed, auto) = if opening || quiet {
            let now = Instant::now();
            send(CallEvent::State("Thinking"));
            nudges += u32::from(!opening);
            let note = if opening { OPENING } else { QUIET };
            opening = false;
            (note.to_string(), now, now, now, false, true, true)
        } else if let Ok(t) = typed.try_recv() {
            let now = Instant::now();
            send(CallEvent::State("Thinking"));
            (t, now, now, now, false, true, false)
        } else {
            let h = match held.pop_front() {
                Some(h) => h,
                None => match heard.recv_timeout(Duration::from_millis(100)) {
                    Ok(h) => h,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(_) => break,
                },
            };
            quiet_since = Instant::now();
            let (audio, spoke, same) = match h {
                Heard::Started => {
                    send(CallEvent::State("Hearing you"));
                    continue;
                }
                Heard::Pause(a) => {
                    let (tx, rx) = mpsc::channel();
                    let api = api.clone();
                    std::thread::spawn(move || {
                        let _ = tx.send(transcribe(&api, &a));
                    });
                    ahead = Some(rx);
                    continue;
                }
                Heard::Utterance(a, spoke, same) => (a, spoke, same),
            };
            let heard_at = Instant::now();
            send(CallEvent::State("Thinking"));
            // nothing said since the pause: its transcript (already under way, or done) stands
            let early = ahead.take().filter(|_| same).and_then(|rx| rx.recv().ok());
            let used_ahead = early.is_some();
            let text = match early.unwrap_or_else(|| transcribe(&api, &audio)) {
                Ok(t) => t,
                Err(e) => {
                    send(CallEvent::Error(format!("Couldn't hear that: {e}")));
                    String::new()
                }
            };
            (text, spoke, heard_at, Instant::now(), used_ahead, false, false)
        };
        if text.trim().is_empty() || ended.load(Ordering::Relaxed) {
            send(CallEvent::State("Listening"));
            continue;
        }
        if auto {
            send(CallEvent::Her);
        } else {
            nudges = 0;
            send(CallEvent::You(text.clone()));
        }
        let mut mood_told = false;
        *first_audio.lock().unwrap() = None;
        *voiced.lock().unwrap() = Voiced::default();
        let t = turn.load(Ordering::Relaxed);
        let (mut first_text, mut first_said): (Option<Instant>, Option<Instant>) = (None, None);
        // Maya answers: her persona, how to talk on a call and what she remembers are the server's (maya.rs)
        let mut body = vec![("message", str(&text)), ("persona", str("maya")), ("tools", Json::Arr(Vec::new())), ("thinking", Json::Bool(false)), ("mode", str(&mode))];
        if let Some(c) = &chat {
            body.push(("chat", str(c)));
        }
        if auto {
            body.push(("auto", Json::Bool(true)));
        }
        if let Some(c) = cut.take() {
            body.push(("cut", str(&c)));
        }
        let (mut raw, mut said) = (String::new(), 0usize);
        let say = |reply: &str, said: &mut usize, last: bool, chat: &Option<String>, first: &mut Option<Instant>| {
            if let Some(piece) = next_sentences(reply, said, last) {
                first.get_or_insert_with(Instant::now);
                pending.fetch_add(1, Ordering::Relaxed);
                let _ = say_tx.send((piece, chat.clone().unwrap_or_default(), t));
            }
        };
        let result = api.stream("/api/agent", &obj(body), |ev| {
            let s = |k: &str| ev.get(k).and_then(Json::as_str).unwrap_or("").to_string();
            match ev.get("type").and_then(Json::as_str).unwrap_or("") {
                "chat" => {
                    chat = Some(s("id"));
                    send(CallEvent::Chat(s("id")));
                }
                "text" => {
                    raw.push_str(&s("text"));
                    if !mood_told {
                        if let Some((m, t)) = mood(&raw) {
                            mood_told = true;
                            send(CallEvent::Mood(m, t));
                        }
                    }
                    let reply = answer(&raw);
                    if !reply.trim().is_empty() {
                        first_text.get_or_insert_with(Instant::now);
                    }
                    send(CallEvent::Ghost(without_tags(&reply)));
                    say(&reply, &mut said, false, &chat, &mut first_said);
                }
                "error" => {
                    send(CallEvent::Error(s("message")));
                }
                _ => {}
            }
            if barged(&mut held) {
                cut = Some(cut_off());
            }
            !ended.load(Ordering::Relaxed) && cut.is_none()
        });
        if cut.is_none() {
            if let Err(e) = result {
                send(CallEvent::Error(format!("No answer: {e}")));
            }
            let reply = answer(&raw);
            say(&reply, &mut said, true, &chat, &mut first_said);
            if pending.load(Ordering::Relaxed) > 0 || speaker.queued() > 0.0 {
                send(CallEvent::State("Speaking"));
            }
            // until the whole answer has been spoken, or the user cuts in
            while !ended.load(Ordering::Relaxed) && (pending.load(Ordering::Relaxed) > 0 || speaker.queued() > 0.0) {
                if barged(&mut held) {
                    cut = Some(cut_off());
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        // how long each step took, counted from the last sound of the user's voice (or the message's sending)
        let at = |t: Option<Instant>| t.map_or("-".to_string(), |t| format!("{:.2}", t.duration_since(spoke).as_secs_f64()));
        let line = if was_typed {
            format!("{} · first words {} · first sentence {} · voice {} s", if auto { "she spoke up" } else { "typed" }, at(first_text), at(first_said), at(*first_audio.lock().unwrap()))
        } else {
            format!(
                "end of speech heard {} · text {} · first words {} · first sentence {} · voice {} s{}",
                at(Some(heard_at)),
                at(Some(text_at)),
                at(first_text),
                at(first_said),
                at(*first_audio.lock().unwrap()),
                if used_ahead { " (transcribed at the pause)" } else { "" }
            )
        };
        send(CallEvent::Timing(line.clone()));
        log_times(&format!("{line}{}", if cut.is_some() { " · cut off" } else { "" }));
        quiet_since = Instant::now();
        if cut.is_some() {
            send(CallEvent::State("Hearing you"));
            continue;
        }
        // she said goodbye and hung up, once it was heard
        if hangs_up(&raw) {
            break;
        }
        // without barge-in, what was heard meanwhile is the call's own voice, not the user
        if !shared.barge.load(Ordering::Relaxed) {
            while heard.try_recv().is_ok() {}
            std::thread::sleep(Duration::from_millis(300));
            while heard.try_recv().is_ok() {}
        }
        send(CallEvent::State("Listening"));
    }
    shared.stop.store(true, Ordering::Relaxed);
}

/// A reply as the call screen shows it: the voice's directions ([pause], [laughs]) left out.
fn without_tags(reply: &str) -> String {
    let mut out = String::new();
    let mut rest = reply;
    while let Some(open) = rest.find('[') {
        match rest[open..].find(']') {
            // a [link](...) is Markdown, not a direction
            Some(len) if !rest[open + len + 1..].starts_with('(') => {
                out.push_str(&rest[..open]);
                rest = &rest[open + len + 1..];
            }
            _ => {
                out.push_str(&rest[..open + 1]);
                rest = &rest[open + 1..];
            }
        }
    }
    out.push_str(rest);
    out.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")
}

/// The text of an utterance ("" when the server thinks it held no speech).
fn transcribe(api: &Api, audio: &[f32]) -> Result<String, String> {
    let j = api.post_bytes("/v1/audio/transcriptions", "audio/wav", &wav16(audio))?;
    if j.get("no_speech").and_then(Json::as_f64).unwrap_or(0.0) > 0.6 {
        return Ok(String::new());
    }
    Ok(j.get("text").and_then(Json::as_str).unwrap_or("").trim().to_string())
}

/// A turn's timings, added to %LOCALAPPDATA%\ghost-infer\call-times.log.
fn log_times(line: &str) {
    use std::io::Write;
    let Some(dir) = std::env::var_os("LOCALAPPDATA").map(|d| std::path::PathBuf::from(d).join("ghost-infer")) else { return };
    let _ = std::fs::create_dir_all(&dir);
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("call-times.log")) {
        let _ = writeln!(f, "{line}");
    }
}

/// The answer part of a reply: without a thinking block (a model that thinks may still open an empty one), nor
/// pauses before its first word (only a later start; the model writes one before nearly every reply). Any other
/// direction there stays: an opening [laughs] is heard.
fn answer(raw: &str) -> String {
    let mut a = after_thinking(raw);
    // her mood opens it ({playful · teasing}): shown on the screen, never spoken
    if a.starts_with('{') {
        match a.find('}') {
            Some(end) => a = a[end + 1..].trim_start(),
            None => return String::new(),
        }
    }
    while a.starts_with('[') {
        match a.find(']') {
            Some(end) if a[1..end].to_lowercase().contains("pause") => a = a[end + 1..].trim_start(),
            Some(_) => break,
            // the direction is still being written
            None => return String::new(),
        }
    }
    a.to_string()
}

fn after_thinking(raw: &str) -> &str {
    match (raw.find("<think>"), raw.find("</think>")) {
        (Some(_), Some(end)) => &raw[end + "</think>".len()..],
        (Some(_), None) => "",
        _ => raw,
    }
    .trim_start()
}

/// The reply ends the call: Maya's [hang up], as she writes it after a goodbye.
fn hangs_up(raw: &str) -> bool {
    let r = after_thinking(raw).to_lowercase();
    r.contains("[hang up]") || r.contains("[hangs up]")
}

/// Her mood and way of talking from the reply's opening {mood · way of talking}, once it is written: the mood
/// as one of hers (MOODS, else as written), lower case.
fn mood(raw: &str) -> Option<(String, String)> {
    let a = after_thinking(raw).strip_prefix('{')?;
    let inner = &a[..a.find('}')?];
    let mut parts = inner.splitn(2, ['·', '|', ',', ';', '/', ':', '-', '•']);
    let mood = parts.next()?.trim().trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    let talk = parts.next().unwrap_or("").trim().to_lowercase();
    (!mood.is_empty()).then_some((mood, talk))
}

/// The next whole sentences of `reply` after `said` (all the rest when `last`), as words to speak: no code,
/// no Markdown marks, links by their text. Moves `said` on.
fn next_sentences(reply: &str, said: &mut usize, last: bool) -> Option<String> {
    let rest = reply.get(*said..)?;
    let end = if last {
        rest.len()
    } else {
        // the last sentence end with a space or line after it; the first piece may be short (heard sooner)
        let min = if *said == 0 { 12 } else { 40 };
        let mut end = 0;
        for (i, c) in rest.char_indices() {
            let after = rest[i + c.len_utf8()..].chars().next();
            let boundary = c == '\n' || (matches!(c, '.' | '!' | '?' | '…' | ':' | ';') && after.is_some_and(char::is_whitespace));
            if boundary && i + c.len_utf8() >= min {
                end = i + c.len_utf8();
            }
        }
        end
    };
    if end == 0 {
        return None;
    }
    *said += end;
    let words = speakable(&rest[..end]);
    words.chars().any(char::is_alphanumeric).then_some(words)
}

fn speakable(text: &str) -> String {
    let mut out = String::new();
    let mut in_code = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        // [text](link) as its text, bare links as "a link", inline code and Markdown marks left out
        let mut l = String::new();
        let mut it = line.split('`');
        while let Some(plain) = it.next() {
            l.push_str(plain);
            it.next();
        }
        let mut words = String::new();
        let mut rest = l.as_str();
        while let Some(open) = rest.find('[') {
            words.push_str(&rest[..open]);
            match rest[open..].find("](").and_then(|m| rest[open + m..].find(')').map(|c| (m, open + m + c))) {
                Some((m, close)) => {
                    words.push_str(&rest[open + 1..open + m]);
                    rest = &rest[close + 1..];
                }
                None => {
                    words.push('[');
                    rest = &rest[open + 1..];
                }
            }
        }
        words.push_str(rest);
        let words: Vec<String> = words.split_whitespace().map(|w| if w.starts_with("http://") || w.starts_with("https://") { "a link".to_string() } else { w.to_string() }).collect();
        // letters, digits and the punctuation speech reads; not Markdown's marks, emoji or other symbols
        // (square brackets stay: they are the voice's directions, [pause], [laughs])
        let spoken = |c: &char| c.is_alphanumeric() || c.is_whitespace() || ".,!?;:'\"-()%$&/@+=’‘“”—–…[]".contains(*c);
        let line: String = words.join(" ").chars().filter(spoken).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
        if !line.trim().is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(line.trim());
        }
    }
    out
}

/// A 16 kHz mono WAV file of `samples`.
fn wav16(samples: &[f32]) -> Vec<u8> {
    let data = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&voice::MIC_RATE.to_le_bytes());
    out.extend_from_slice(&(voice::MIC_RATE * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data as u32).to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_come_out_whole() {
        // shorter than the first piece's minimum: it waits for more
        let mut said = 0;
        assert_eq!(next_sentences("Hi there! How are", &mut said, false), None);
        let mut said = 0;
        assert_eq!(next_sentences("Hello there, friend. How are", &mut said, false).as_deref(), Some("Hello there, friend."));
        assert_eq!(next_sentences("Hello there, friend. How are you today?", &mut said, true).as_deref(), Some("How are you today?"));
    }

    #[test]
    fn markdown_is_not_spoken() {
        assert_eq!(speakable("**Sure** - see [the docs](https://x.y) or `code`.\n```\nlet x = 1;\n```\nDone."), "Sure - see the docs or . Done.");
    }

    #[test]
    fn emoji_are_not_spoken() {
        assert_eq!(speakable("They always keep their cool. 😄 What's yours? 🎉"), "They always keep their cool. What's yours?");
    }

    #[test]
    fn directions_reach_the_voice_but_not_the_screen() {
        assert_eq!(speakable("Oh! [short pause] That's wild. [laughs]"), "Oh! [short pause] That's wild. [laughs]");
        assert_eq!(without_tags("Oh! [short pause] That's wild. [laughs]"), "Oh! That's wild.");
        assert_eq!(without_tags("See [the docs](x) [pause] now"), "See [the docs](x) now");
    }

    #[test]
    fn a_cut_reply_keeps_what_was_heard() {
        let v = Voiced { total: 4.0, pieces: vec![("Oh, hey you. [laughs]".into(), 0.0, 1.0), ("I was just thinking about that song again.".into(), 1.0, 4.0)] };
        assert_eq!(v.said(0.0), "Oh, hey you. I was just thinking about that song again.");
        // 1.5 of its 3 seconds played: half its words
        assert_eq!(v.said(1.5), "Oh, hey you. I was just thinking");
        assert_eq!(v.said(3.5), "Oh, hey");
        assert_eq!(v.said(4.0), "");
    }

    #[test]
    fn her_mood_is_read_not_spoken() {
        assert_eq!(mood("{playful · teasing} Oh, you again?"), Some(("playful".into(), "teasing".into())));
        assert_eq!(mood("{Tender | soft and slow}"), Some(("tender".into(), "soft and slow".into())));
        assert_eq!(mood("{curious"), None);
        assert_eq!(mood("Hi there."), None);
        assert_eq!(answer("{playful · teasing} [pause] Oh, you again?"), "Oh, you again?");
        assert_eq!(answer("{playful · tea"), "");
    }

    #[test]
    fn she_hangs_up_after_a_goodbye() {
        assert!(hangs_up("{tender · warm} [hang up] Okay, talk soon. Bye!"));
        assert!(!hangs_up("{calm · easy} So, what are you up to?"));
        assert_eq!(without_tags(&answer("{tender · warm} [hang up] Okay, talk soon. Bye!")), "Okay, talk soon. Bye!");
    }

    #[test]
    fn thinking_is_not_spoken() {
        assert_eq!(answer("<think>\n\n</think>\n\nHello."), "Hello.");
        assert_eq!(answer("<think>hmm"), "");
        assert_eq!(answer("[short pause] Hello there. [laughs] Hi."), "Hello there. [laughs] Hi.");
        assert_eq!(answer("[short pa"), "");
    }
}
