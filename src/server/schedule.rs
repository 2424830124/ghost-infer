//! The agent working while nobody is chatting: scheduled tasks.
//!
//! A scheduled task is a prompt sent to the model at a time (once, every N minutes, or daily at a local
//! time). The model makes them with its schedule_task tool, or the user with /loop (Claude Code's time-based
//! loop), and each is answered in the chat it was made in. (The job file follows goose's.)
//!
//! Kept in %LOCALAPPDATA%\ghost-infer\schedules.json. A run waits until no reply has been going for a while,
//! so it never cuts in on the user, and uses only the tools the user always allows (nobody is there to ask).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use super::agent::{self, Wakeup};
use super::{Shared, Work};
use crate::json::{num, obj, str, Json};

/// How long after the last reply ended a scheduled run may start (the user may be reading, about to type).
const QUIET_MS: u64 = 20_000;

/// The shortest gap between runs of a repeating task, in minutes (each one takes the model a while).
const MIN_EVERY: u32 = 5;

const DAYS: [&str; 7] = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];

// ---- local time: Windows' clock offset, and dates by Howard Hinnant's civil-day algorithms

/// Local time minus UTC, in milliseconds (to the minute), as Windows has it now.
#[cfg(windows)]
fn offset_ms() -> i64 {
    #[repr(C)]
    #[derive(Default)]
    struct WinTime {
        year: u16,
        month: u16,
        day_of_week: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
        ms: u16,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLocalTime(t: *mut WinTime);
        fn GetSystemTime(t: *mut WinTime);
    }
    let (mut l, mut u) = (WinTime::default(), WinTime::default());
    unsafe {
        GetSystemTime(&mut u);
        GetLocalTime(&mut l);
    }
    let ms = |t: &WinTime| (days_from_civil(t.year as i64, t.month as u32, t.day as u32) * 86_400 + t.hour as i64 * 3600 + t.minute as i64 * 60 + t.second as i64) * 1000;
    // the two calls are a moment apart
    ((ms(&l) - ms(&u)) as f64 / 60_000.0).round() as i64 * 60_000
}

#[cfg(not(windows))]
fn offset_ms() -> i64 {
    0
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m as i64 + 9) % 12) + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// A moment as local time: its day (days since 1970), minute of that day, and weekday (0 Sunday).
struct Local {
    day: i64,
    minute: u32,
    weekday: usize,
}

fn local(ms: f64) -> Local {
    let t = ms as i64 + offset_ms();
    let day = t.div_euclid(86_400_000);
    Local { day, minute: (t.rem_euclid(86_400_000) / 60_000) as u32, weekday: (day + 4).rem_euclid(7) as usize }
}

/// A moment as local time: its day (days since 1970), minute of the day and weekday (0 Sunday).
pub(super) fn day_minute(ms: f64) -> (i64, u32, usize) {
    let at = local(ms);
    (at.day, at.minute, at.weekday)
}

/// Today's local date, as the agent's opening line says it: Thursday 30 October 2026.
pub fn today() -> String {
    const NAMES: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    let at = local(agent::now_ms());
    let (y, m, d) = civil_from_days(at.day);
    format!("{} {d} {} {y}", NAMES[at.weekday], MONTHS[(m as usize).clamp(1, 12) - 1])
}

/// The moment a local day and minute is, in milliseconds since 1970.
fn moment(day: i64, minute: u32) -> f64 {
    (day * 86_400_000 + minute as i64 * 60_000 - offset_ms()) as f64
}

/// "today 14:30", "tomorrow 09:00", "Mon 3 Nov 09:00".
fn when_text(ms: f64) -> String {
    let (at, today) = (local(ms), local(agent::now_ms()).day);
    let hm = format!("{:02}:{:02}", at.minute / 60, at.minute % 60);
    match at.day - today {
        0 => format!("today {hm}"),
        1 => format!("tomorrow {hm}"),
        _ => {
            const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
            let (_, m, d) = civil_from_days(at.day);
            let wd = DAYS[at.weekday];
            format!("{}{} {d} {} {hm}", wd[..1].to_uppercase(), &wd[1..], MONTHS[m as usize - 1])
        }
    }
}

/// "14:30" or "9:05" as a minute of the day.
fn parse_hm(s: &str) -> Option<u32> {
    let (h, m) = s.trim().split_once(':')?;
    let (h, m): (u32, u32) = (h.trim().parse().ok()?, m.trim().parse().ok()?);
    (h < 24 && m < 60).then_some(h * 60 + m)
}

/// "mon,tue", "weekdays", "weekends" or full names, as the weekdays (0 Sunday) they name; empty: every day.
fn parse_days(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split([',', ' ']).map(|p| p.trim().to_lowercase()).filter(|p| !p.is_empty()) {
        match part.as_str() {
            "weekdays" => out.extend(1..=5),
            "weekends" => out.extend([0, 6]),
            p => out.push(DAYS.iter().position(|d| p.starts_with(d)).ok_or_else(|| format!("{p} is not a day; use mon, tue, wed, thu, fri, sat, sun"))?),
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// "14:30" (today, or tomorrow once it has passed) or "2026-11-03 14:30", local time.
fn parse_at(s: &str, now: f64) -> Option<f64> {
    let s = s.trim();
    match s.split_once([' ', 'T']) {
        Some((date, time)) => {
            let mut ymd = date.split('-').map(|p| p.parse::<i64>().ok());
            let (y, m, d) = (ymd.next()??, ymd.next()??, ymd.next()??);
            if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
                return None;
            }
            Some(moment(days_from_civil(y, m as u32, d as u32), parse_hm(time)?))
        }
        None => {
            let minute = parse_hm(s)?;
            let today = moment(local(now).day, minute);
            Some(if today > now { today } else { moment(local(now).day + 1, minute) })
        }
    }
}

/// The next time after `after` that is `minute` on one of `days` (every day when empty).
fn next_daily(after: f64, minute: u32, days: &[usize]) -> f64 {
    let start = local(after).day;
    (0..8)
        .map(|k| start + k)
        .filter(|day| days.is_empty() || days.contains(&((day + 4).rem_euclid(7) as usize)))
        .map(|day| moment(day, minute))
        .find(|&t| t > after)
        .unwrap_or(after + 86_400_000.0)
}

// ---- the store

#[derive(Clone, Default)]
struct Job {
    id: String,
    name: String,
    prompt: String,
    chat: String,        // answered in this chat (made anew if it is gone)
    next: f64,           // when it runs next, milliseconds since 1970
    every: u32,          // minutes apart; 0 with no daily: once
    daily: Option<u32>,  // every day at this local minute
    days: Vec<usize>,    // with daily: only these weekdays (0 Sunday); empty: all
    groups: Vec<String>, // the tool groups it may use (those of the chat it was made in)
    thinking: bool,
    last: f64,
    note: String, // how the last run went
}

impl Job {
    fn when(&self) -> String {
        match (self.daily, self.every) {
            (Some(m), _) => {
                let days = if self.days.is_empty() { String::new() } else { format!(" on {}", self.days.iter().map(|&d| DAYS[d]).collect::<Vec<_>>().join(", ")) };
                format!("daily at {:02}:{:02}{days}", m / 60, m % 60)
            }
            (None, 0) => "once".into(),
            (None, n) if n % 60 == 0 => format!("every {} h", n / 60),
            (None, n) => format!("every {n} min"),
        }
    }

    /// When it runs after one at `now`; None: it ran its once.
    fn after(&self, now: f64) -> Option<f64> {
        match (self.daily, self.every) {
            (Some(m), _) => Some(next_daily(now, m, &self.days)),
            (None, 0) => None,
            (None, n) => Some(now + n as f64 * 60_000.0),
        }
    }
}

#[derive(Default)]
struct Store {
    jobs: Vec<Job>,
}

fn file() -> Result<std::path::PathBuf, String> {
    Ok(agent::data_dir()?.join("schedules.json"))
}

fn strings(j: Option<&Json>) -> Vec<String> {
    j.and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect()
}

fn load() -> Store {
    let j = file().ok().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| Json::parse(&t).ok()).unwrap_or(Json::Null);
    let s = |j: &Json, k: &str| j.get(k).and_then(Json::as_str).unwrap_or("").to_string();
    let f = |j: &Json, k: &str| j.get(k).and_then(Json::as_f64).unwrap_or(0.0);
    let jobs = j.get("jobs").and_then(Json::as_array).unwrap_or(&[]).iter().map(|t| Job {
        id: s(t, "id"),
        name: s(t, "name"),
        prompt: s(t, "prompt"),
        chat: s(t, "chat"),
        next: f(t, "next"),
        every: f(t, "every") as u32,
        daily: t.get("daily").and_then(Json::as_f64).map(|m| m as u32),
        days: t.get("days").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_f64).map(|d| d as usize).filter(|&d| d < 7).collect(),
        groups: strings(t.get("tools")),
        thinking: t.get("thinking").and_then(Json::as_bool).unwrap_or(false),
        last: f(t, "last"),
        note: s(t, "note"),
    }).collect();
    Store { jobs }
}

fn save(st: &Store) -> Result<(), String> {
    let tools = |g: &[String]| Json::Arr(g.iter().map(|g| str(g)).collect());
    let jobs = st.jobs.iter().map(|t| {
        let mut f = vec![("id", str(&t.id)), ("name", str(&t.name)), ("prompt", str(&t.prompt)), ("chat", str(&t.chat)), ("next", num(t.next)), ("every", num(t.every as f64))];
        if let Some(m) = t.daily {
            f.push(("daily", num(m as f64)));
            f.push(("days", Json::Arr(t.days.iter().map(|&d| num(d as f64)).collect())));
        }
        f.extend([("tools", tools(&t.groups)), ("thinking", Json::Bool(t.thinking)), ("last", num(t.last)), ("note", str(&t.note))]);
        obj(f)
    });
    let path = file()?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, obj(vec![("jobs", Json::Arr(jobs.collect()))]).to_string()).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
}

/// Bumped on every change, so the clients know to look again (a chat list with a new unread reply, a new task).
static SEQ: AtomicU64 = AtomicU64::new(1);

/// The chat a scheduled run is answering in right now.
static RUNNING_IN: Mutex<Option<String>> = Mutex::new(None);

/// The store, changed by `f` and saved (one change at a time).
fn edit<T>(f: impl FnOnce(&mut Store) -> T) -> T {
    static LOCK: Mutex<()> = Mutex::new(());
    let _g = LOCK.lock().unwrap();
    let mut st = load();
    let r = f(&mut st);
    if let Err(e) = save(&st) {
        eprintln!("[schedules: cannot save: {e}]");
    }
    SEQ.fetch_add(1, Ordering::SeqCst);
    r
}

// ---- the tools

/// schedule_task, list_schedules and cancel_schedule, from the chat `chat` whose reply uses `groups`;
/// `always` is what the user allows without asking. The result for the model.
pub fn tool(name: &str, args: &Json, chat: &str, groups: &[String], thinking: bool, always: &[String]) -> String {
    let now = agent::now_ms();
    match name {
        "schedule_task" => match new_job(args, now, chat, groups, thinking) {
            Ok(job) => {
                let mut text = format!("Scheduled \"{}\" (id {}): {}, next {}. It runs in this chat.", job.name, job.id, job.when(), when_text(job.next));
                // tools of this chat that need an OK, which a run with nobody there cannot get
                let asks: Vec<&str> = [("edit", "changing files"), ("cmd", "running commands"), ("web", "using the web"), ("mcp", "MCP tools")]
                    .iter()
                    .filter(|(g, _)| groups.iter().any(|x| x == g) && !always.iter().any(|x| x == g))
                    .map(|(_, what)| *what)
                    .collect();
                if !asks.is_empty() {
                    text.push_str(&format!(" Nobody will be there to OK {}, so it cannot do those unless the user sets them to always allowed; tell them if the task needs them.", asks.join(", ")));
                }
                let id = job.id.clone();
                edit(|st| st.jobs.push(job));
                eprintln!("[scheduled {id}]");
                text
            }
            Err(e) => format!("Error: {e}"),
        },
        "list_schedules" => {
            let st = load();
            let mut lines = vec![format!("It is now {}.", when_text(now))];
            if st.jobs.is_empty() {
                lines.push("No scheduled tasks.".into());
            }
            for t in &st.jobs {
                lines.push(format!("- {}: \"{}\", {}, next {}{}", t.id, t.name, t.when(), when_text(t.next), if t.chat == chat { " (this chat)" } else { "" }));
            }
            lines.join("\n")
        }
        _ => {
            let id = args.get("id").and_then(Json::as_str).unwrap_or("").trim().to_string();
            let gone = edit(|st| {
                let i = st.jobs.iter().position(|t| t.id == id)?;
                Some(st.jobs.remove(i))
            });
            match gone {
                Some(t) => format!("Cancelled \"{}\".", t.name),
                None => format!("Error: no scheduled task {id}; list_schedules shows them."),
            }
        }
    }
}

/// /loop in a chat (Claude Code's time-based loop): `prompt` sent to it every `every` minutes until
/// /loop stop. What to tell the user.
pub fn add_loop(chat: &str, prompt: &str, every: u32, groups: &[String], thinking: bool) -> Result<String, String> {
    let args = obj(vec![("prompt", str(prompt)), ("every_minutes", num(every as f64))]);
    let job = new_job(&args, agent::now_ms(), chat, groups, thinking)?;
    let text = format!("Looping {} in this chat: \"{}\". The first run is {}. /loop stop ends it. Nobody is here to OK tools while it runs, so it uses only what you always allow.", job.when(), job.prompt, when_text(job.next));
    edit(|st| st.jobs.push(job));
    Ok(text)
}

/// /loop stop: the chat's repeating tasks cancelled; how many there were.
pub fn stop_loops(chat: &str) -> usize {
    edit(|st| {
        let before = st.jobs.len();
        st.jobs.retain(|j| !(j.chat == chat && j.every > 0));
        before - st.jobs.len()
    })
}

fn new_job(args: &Json, now: f64, chat: &str, groups: &[String], thinking: bool) -> Result<Job, String> {
    let s = |k: &str| args.get(k).and_then(Json::as_str).map(str::trim).filter(|v| !v.is_empty());
    let n = |k: &str| args.get(k).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())));
    let prompt = s("prompt").ok_or("give prompt: what to do then")?.to_string();
    let mut job = Job { prompt, chat: chat.to_string(), groups: groups.to_vec(), thinking, ..Default::default() };
    let given = [n("in_minutes").is_some(), s("at").is_some(), n("every_minutes").is_some(), s("daily_at").is_some()].iter().filter(|g| **g).count();
    if given != 1 {
        return Err("give exactly one of in_minutes, at, every_minutes or daily_at".into());
    }
    if let Some(m) = n("in_minutes") {
        job.next = now + m.max(1.0) * 60_000.0;
    } else if let Some(at) = s("at") {
        job.next = parse_at(at, now).ok_or(format!("{at} is not a time; give HH:MM or YYYY-MM-DD HH:MM"))?;
        if job.next <= now {
            return Err(format!("{at} has passed; it is now {}", when_text(now)));
        }
    } else if let Some(m) = n("every_minutes") {
        job.every = (m.round() as u32).max(MIN_EVERY);
        job.next = now + job.every as f64 * 60_000.0;
    } else if let Some(at) = s("daily_at") {
        let minute = parse_hm(at).ok_or(format!("{at} is not a time; give HH:MM"))?;
        job.days = parse_days(s("days").unwrap_or(""))?;
        job.daily = Some(minute);
        job.next = next_daily(now, minute, &job.days);
    }
    job.name = match s("name") {
        Some(n) => n.chars().take(60).collect(),
        None => {
            let words: Vec<&str> = job.prompt.split_whitespace().take(6).collect();
            words.join(" ")
        }
    };
    let taken: Vec<String> = load().jobs.into_iter().map(|t| t.id).collect();
    let mut k = (now as u64 / 1000) & 0xfffff;
    while taken.contains(&format!("{k:05x}")) {
        k += 1;
    }
    job.id = format!("{k:05x}");
    Ok(job)
}

/// The first line of a reply worth showing (its thinking left out).
fn gist(reply: &str) -> String {
    let reply = reply.rsplit("</think>").next().unwrap_or(reply);
    let line = reply.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    crate::tools::clip(line, 140)
}

// ---- for the clients

/// GET /api/background: what runs on its own, and a number that changes whenever any of it does; and the
/// commands the model left running (start_process), which change without it.
pub fn status_json() -> Json {
    let st = load();
    let at = |ms: f64| if ms > 0.0 { str(&when_text(ms)) } else { Json::Null };
    let jobs = st.jobs.iter().map(|t| obj(vec![("id", str(&t.id)), ("name", str(&t.name)), ("prompt", str(&t.prompt)), ("when", str(&t.when())), ("next", at(t.next)), ("last", at(t.last)), ("note", str(&t.note)), ("chat", str(&t.chat))]));
    let running = RUNNING_IN.lock().unwrap().clone().map_or(Json::Null, |c| str(&c));
    obj(vec![("seq", num(SEQ.load(Ordering::SeqCst) as f64)), ("running", running), ("jobs", Json::Arr(jobs.collect())), ("processes", crate::tools::processes())])
}

/// POST /api/schedules/run {"id"}: the task runs as soon as things are quiet.
pub fn run_now(id: &str) -> Result<Json, String> {
    let found = edit(|st| {
        st.jobs.iter_mut().find(|t| t.id == id).map(|t| t.next = 0.0).is_some()
    });
    if !found {
        return Err(format!("no scheduled task {id}"));
    }
    Ok(status_json())
}

/// POST /api/schedules/cancel {"id"}.
pub fn cancel(id: &str) -> Result<Json, String> {
    let gone = edit(|st| {
        let i = st.jobs.iter().position(|t| t.id == id)?;
        Some(st.jobs.remove(i))
    });
    gone.map(|_| status_json()).ok_or(format!("no scheduled task {id}"))
}

// ---- the scheduler

/// Starts the thread that runs what is due, one at a time, whenever the agent is quiet.
pub fn start(work: mpsc::Sender<Work>, shared: Arc<Shared>) {
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(5));
        tick(&work, &shared);
    });
}

fn tick(work: &mpsc::Sender<Work>, shared: &Shared) {
    if agent::idle_ms().is_none_or(|ms| ms < QUIET_MS) {
        return;
    }
    let Some(loaded) = ({
        let s = shared.status.lock().unwrap();
        s.current.clone().filter(|c| s.loading.is_none() && c.chat_ok)
    }) else {
        return;
    };
    let now = agent::now_ms();
    // the task due first, moved on before it runs (or taken out if it ran its once), so a run that fails or a server stopped mid-run does not repeat it at once
    let due = edit(|st| {
        if let Some(i) = st.jobs.iter().enumerate().filter(|(_, t)| t.next <= now).min_by(|a, b| a.1.next.total_cmp(&b.1.next)).map(|(i, _)| i) {
            let job = st.jobs[i].clone();
            match job.after(now) {
                Some(next) => st.jobs[i].next = next,
                None => {
                    st.jobs.remove(i);
                }
            }
            return Some(job);
        }
        None
    });
    if let Some(job) = due {
        run_task(job, now, work, shared, &loaded);
    }
}

fn run_task(job: Job, now: f64, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &super::Loaded) {
    eprintln!("[scheduled task {} \"{}\"]", job.id, job.name);
    let text = format!("(Scheduled task \"{}\", set to run {}. It is now {}. The user may not be here: do the task, then report briefly.)\n\n{}", job.name, job.when(), when_text(now), job.prompt);
    let shown = format!("Scheduled: {}", job.name);
    let w = Wakeup { chat: Some(&job.chat).filter(|c| !c.is_empty()).map(String::as_str), title: &job.name, text: &text, shown: &shown, groups: &job.groups, thinking: job.thinking };
    *RUNNING_IN.lock().unwrap() = Some(job.chat.clone());
    SEQ.fetch_add(1, Ordering::SeqCst);
    let result = agent::run_away(&w, |_| true, work, shared, loaded);
    *RUNNING_IN.lock().unwrap() = None;
    let (chat, note) = match result {
        Ok((chat, text, _)) => (chat, gist(&text)),
        Err(e) => {
            eprintln!("[scheduled task {}: {e}]", job.id);
            (job.chat.clone(), format!("failed: {e}"))
        }
    };
    edit(|st| {
        if let Some(t) = st.jobs.iter_mut().find(|t| t.id == job.id) {
            (t.last, t.note, t.chat) = (now, note, chat);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_days_round_trip() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2026, 10, 29), 20_755);
        for z in [-1000, 0, 59, 60, 20_755, 30_000] {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
        assert_eq!((days_from_civil(2026, 10, 29) + 4).rem_euclid(7), 4); // a Thursday
    }

    #[test]
    fn times_and_days() {
        assert_eq!(parse_hm("09:05"), Some(545));
        assert_eq!(parse_hm("24:00"), None);
        assert_eq!(parse_days("mon, Friday").unwrap(), vec![1, 5]);
        assert_eq!(parse_days("weekdays").unwrap(), vec![1, 2, 3, 4, 5]);
        assert!(parse_days("someday").is_err());
        let now = moment(days_from_civil(2026, 10, 29), 14 * 60); // Thursday 14:00
        assert_eq!(parse_at("15:30", now), Some(moment(days_from_civil(2026, 10, 29), 15 * 60 + 30)));
        assert_eq!(parse_at("13:00", now), Some(moment(days_from_civil(2026, 10, 30), 13 * 60)));
        assert_eq!(parse_at("2026-11-03 09:00", now), Some(moment(days_from_civil(2026, 11, 3), 9 * 60)));
        // daily at 09:00 on Mondays, from Thursday: the next Monday
        assert_eq!(next_daily(now, 9 * 60, &[1]), moment(days_from_civil(2026, 11, 2), 9 * 60));
        assert_eq!(next_daily(now, 15 * 60, &[]), moment(days_from_civil(2026, 10, 29), 15 * 60));
    }
}
