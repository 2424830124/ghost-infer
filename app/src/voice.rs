//! Voice calls: the microphone, a detector that cuts it into utterances, and the speaker. The microphone is
//! read at the device's rate, made mono and brought to 16 kHz (what speech-to-text reads); an utterance is
//! the audio from a little before the voice rose above the room's noise until it has been quiet a moment.
//! The speaker plays 24 kHz speech (the server's voice) at the device's rate from a queue, so pieces that
//! arrive while it plays follow on without a gap.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// The rate speech-to-text reads.
pub const MIC_RATE: u32 = 16_000;
/// 30 ms frames for the detector.
const FRAME: usize = 480;
/// Audio kept from before the voice rose (its first sounds are quiet).
const PRE_ROLL: usize = 10; // frames: 300 ms
/// Quiet that ends an utterance (GHOST_APP_END_MS), quiet after which what was said so far is offered for
/// transcribing ahead (GHOST_APP_PAUSE_MS; 0: never), and the longest utterance (then it is sent as it is).
/// Measured with the 4B model, from the last word to the first sound: 750 ms and none took 1.5–1.95 s; 450 and
/// 210 took 0.85–1.17 s (the transcript was ready when the quiet ended). 500 leaves room for a short pause in a
/// sentence.
fn quiet_frames(var: &str, default_ms: usize) -> usize {
    std::env::var(var).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default_ms) / 30
}
const LONGEST: usize = 20 * MIC_RATE as usize;

/// What the microphone side tells the call.
pub enum Heard {
    /// someone started speaking (the call can stop its own speech: a barge-in)
    Started,
    /// a short quiet: the words so far (16 kHz mono), to transcribe ahead in case it is the end
    Pause(Vec<f32>),
    /// an utterance (16 kHz mono); when the voice last sounded; and whether it is the last Pause's words
    /// with only quiet after (that transcript stands)
    Utterance(Vec<f32>, Instant, bool),
}

/// The state both audio threads and the call share.
#[derive(Default)]
pub struct Shared {
    /// the microphone's loudness, 0..1 (f32 bits), for the call screen
    pub level: AtomicU32,
    /// muted: the microphone is read but nothing is heard
    pub muted: AtomicBool,
    /// when the speaker last played something (ms, `now_ms`)
    played_at: AtomicU64,
    /// the voice playing: its loudness, 0..1 (f32 bits), for the call's orb
    voice: AtomicU32,
    /// ends both threads
    pub stop: AtomicBool,
}

fn now_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64 + 10_000
}

impl Shared {
    pub fn level(&self) -> f32 {
        f32::from_bits(self.level.load(Ordering::Relaxed))
    }

    /// The voice playing: how loud (0..1).
    pub fn voice(&self) -> f32 {
        f32::from_bits(self.voice.load(Ordering::Relaxed))
    }

    /// The speaker is playing, or stopped a moment ago (the room still echoes it): the microphone hears the
    /// call's own voice then, which is not taken for the user's.
    pub fn speaking(&self) -> bool {
        now_ms() < self.played_at.load(Ordering::Relaxed) + 350
    }
}

/// A linear resampler for one channel: good enough for speech.
struct Resampler {
    step: f64, // input samples per output sample
    pos: f64,  // where the next output falls, from the last input sample of the previous call
    last: f32,
}

impl Resampler {
    fn new(from: u32, to: u32) -> Resampler {
        Resampler { step: from as f64 / to as f64, pos: 0.0, last: 0.0 }
    }

    fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        // sample i of the input is at position i + 1 (the previous call's last one is at 0)
        while self.pos <= input.len() as f64 {
            let i = self.pos.floor() as usize;
            let frac = (self.pos - i as f64) as f32;
            let a = if i == 0 { self.last } else { input[i - 1] };
            let b = if i < input.len() { input[i] } else { a };
            out.push(a + (b - a) * frac);
            self.pos += self.step;
        }
        self.pos -= input.len() as f64;
        if let Some(&l) = input.last() {
            self.last = l;
        }
    }
}

/// Mono samples as f32 from any of the formats a device gives.
fn mono<T: Copy>(data: &[T], channels: usize, conv: impl Fn(T) -> f32, out: &mut Vec<f32>) {
    for frame in data.chunks(channels) {
        out.push(frame.iter().map(|&v| conv(v)).sum::<f32>() / channels as f32);
    }
}

/// The default microphone, its mono blocks into `tx`: its rate and the stream (kept open while held).
fn open_mic(tx: mpsc::Sender<Vec<f32>>) -> Result<(u32, cpal::Stream), String> {
    let host = cpal::default_host();
    let dev = host.default_input_device().ok_or("no microphone found")?;
    let cfg = dev.default_input_config().map_err(|e| format!("the microphone: {e}"))?;
    let (rate, channels, format) = (cfg.sample_rate().0, cfg.channels() as usize, cfg.sample_format());
    let err = |e| eprintln!("[microphone: {e}]");
    let config: cpal::StreamConfig = cfg.into();
    let stream = match format {
        cpal::SampleFormat::F32 => dev.build_input_stream(&config, move |d: &[f32], _: &_| { let mut m = Vec::new(); mono(d, channels, |v| v, &mut m); let _ = tx.send(m); }, err, None),
        cpal::SampleFormat::I16 => dev.build_input_stream(&config, move |d: &[i16], _: &_| { let mut m = Vec::new(); mono(d, channels, |v| v as f32 / 32768.0, &mut m); let _ = tx.send(m); }, err, None),
        cpal::SampleFormat::U16 => dev.build_input_stream(&config, move |d: &[u16], _: &_| { let mut m = Vec::new(); mono(d, channels, |v| (v as f32 - 32768.0) / 32768.0, &mut m); let _ = tx.send(m); }, err, None),
        f => return Err(format!("the microphone gives {f:?} samples, which this app does not read")),
    };
    let stream = stream.map_err(|e| format!("the microphone: {e}"))?;
    stream.play().map_err(|e| format!("the microphone: {e}"))?;
    Ok((rate, stream))
}

/// GHOST_APP_FAKE_MIC=<16-bit WAVs, separated by ;>: those files instead of the microphone, in real time, each
/// after a second of quiet and once the call's voice has been quiet a while (as a person takes turns), then
/// quiet (for testing calls without speaking). All at the first file's rate.
fn fake_mic(paths: &std::ffi::OsStr, tx: mpsc::Sender<Vec<f32>>, shared: Arc<Shared>) -> Result<u32, String> {
    let mut clips = Vec::new();
    let mut rate = 0;
    for path in paths.to_string_lossy().split(';').filter(|p| !p.is_empty()) {
        let b = std::fs::read(path).map_err(|e| format!("GHOST_APP_FAKE_MIC {path}: {e}"))?;
        let fmt = b.windows(4).position(|w| w == b"fmt ").ok_or("GHOST_APP_FAKE_MIC: not a WAV file")? + 8;
        let channels = u16::from_le_bytes([b[fmt + 2], b[fmt + 3]]) as usize;
        rate = u32::from_le_bytes(b[fmt + 4..fmt + 8].try_into().unwrap());
        let data = b.windows(4).position(|w| w == b"data").ok_or("GHOST_APP_FAKE_MIC: no data")? + 8;
        let samples: Vec<f32> = b[data..].chunks_exact(2).map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0).collect();
        clips.push(samples.chunks(channels.max(1)).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect::<Vec<f32>>());
    }
    std::thread::spawn(move || {
        let block = rate as usize * 30 / 1000;
        let mut queue: VecDeque<f32> = VecDeque::new();
        let mut clips = clips.into_iter();
        let (mut quiet_since, mut answered) = (Instant::now(), true);
        while !shared.stop.load(Ordering::Relaxed) {
            // the next clip: once the voice has answered the last one and been quiet 1.5 s since
            if shared.speaking() {
                quiet_since = Instant::now();
                answered = true;
            }
            if queue.is_empty() && answered && quiet_since.elapsed().as_millis() > 1500 {
                if let Some(c) = clips.next() {
                    queue.extend(std::iter::repeat_n(0.0, rate as usize));
                    queue.extend(c);
                    answered = false;
                }
            }
            let mut b: Vec<f32> = (0..block).map(|_| queue.pop_front().unwrap_or(0.0)).collect();
            b.resize(block, 0.0);
            if tx.send(b).is_err() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
    });
    Ok(rate)
}

/// Starts the microphone on a thread of its own; utterances come out of `heard`.
pub fn start_mic(shared: Arc<Shared>, heard: mpsc::Sender<Heard>) -> Result<(), String> {
    let (ready, ok) = mpsc::channel();
    std::thread::spawn(move || {
        let (tx, rx) = mpsc::channel::<Vec<f32>>();
        let opened = match std::env::var_os("GHOST_APP_FAKE_MIC") {
            Some(p) => fake_mic(&p, tx, shared.clone()).map(|r| (r, None)),
            None => open_mic(tx).map(|(r, s)| (r, Some(s))),
        };
        let (rate, stream) = match opened {
            Ok(o) => o,
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        };
        let _ = ready.send(Ok(()));
        let mut rs = Resampler::new(rate, MIC_RATE);
        let mut det = Detector::default();
        let mut pending = Vec::new();
        while !shared.stop.load(Ordering::Relaxed) {
            let Ok(block) = rx.recv_timeout(std::time::Duration::from_millis(200)) else { continue };
            rs.push(&block, &mut pending);
            while pending.len() >= FRAME {
                let frame: Vec<f32> = pending.drain(..FRAME).collect();
                // the call's own voice, or muted: heard as silence (and the noise floor is left alone)
                let deaf = shared.muted.load(Ordering::Relaxed) || shared.speaking();
                if let Some(h) = det.push(&frame, deaf, &shared) {
                    if heard.send(h).is_err() {
                        return;
                    }
                }
            }
        }
        drop(stream);
    });
    ok.recv().map_err(|_| "the microphone thread stopped".to_string())?
}

/// Cuts the microphone into utterances by loudness against a slowly followed noise floor.
struct Detector {
    floor: f32, // the room's noise, as RMS
    before: VecDeque<Vec<f32>>,
    speech: Option<Speech>,
    loud_run: usize,
    end: usize,   // quiet frames that end an utterance
    pause: usize, // quiet frames that offer the words so far (0: never)
}

/// An utterance under way.
struct Speech {
    audio: Vec<f32>,
    quiet: usize,       // quiet frames at its end
    loud: usize,        // loud frames in it
    last_loud: Instant, // when the voice last sounded
    paused: bool,       // offered as a Pause, and quiet since
}

impl Default for Detector {
    fn default() -> Detector {
        Detector { floor: 0.0, before: VecDeque::new(), speech: None, loud_run: 0, end: quiet_frames("GHOST_APP_END_MS", 500).max(5), pause: quiet_frames("GHOST_APP_PAUSE_MS", 210) }
    }
}

/// The utterance's audio with 240 ms of its quiet end kept.
fn trimmed(audio: &[f32], quiet: usize) -> Vec<f32> {
    audio[..audio.len() - quiet.saturating_sub(8) * FRAME].to_vec()
}

impl Detector {
    fn push(&mut self, frame: &[f32], deaf: bool, shared: &Shared) -> Option<Heard> {
        let rms = (frame.iter().map(|v| v * v).sum::<f32>() / frame.len() as f32).sqrt();
        shared.level.store(((rms * 12.0).min(1.0)).to_bits(), Ordering::Relaxed);
        if self.floor == 0.0 {
            self.floor = rms.max(1e-4);
        }
        if deaf {
            self.speech = None;
            self.loud_run = 0;
            self.before.clear();
            return None;
        }
        // speech: well above the floor, and above a fixed minimum (a quiet room's floor is tiny)
        let loud = rms > (self.floor * 3.2).max(0.012);
        let quiet = rms < (self.floor * 2.0).max(0.008);
        match &mut self.speech {
            None => {
                // the floor follows the room while nobody speaks: down at once, up slowly
                self.floor = if rms < self.floor { rms.max(1e-4) } else { self.floor * 0.995 + rms * 0.005 };
                self.loud_run = if loud { self.loud_run + 1 } else { 0 };
                self.before.push_back(frame.to_vec());
                if self.before.len() > PRE_ROLL {
                    self.before.pop_front();
                }
                // three loud frames in a row (90 ms): a voice, not a click
                if self.loud_run >= 3 {
                    let audio: Vec<f32> = self.before.drain(..).flatten().collect();
                    self.speech = Some(Speech { audio, quiet: 0, loud: 3, last_loud: Instant::now(), paused: false });
                    return Some(Heard::Started);
                }
                None
            }
            Some(sp) => {
                sp.audio.extend_from_slice(frame);
                sp.quiet = if quiet { sp.quiet + 1 } else { 0 };
                if loud {
                    sp.loud += 1;
                    sp.last_loud = Instant::now();
                    sp.paused = false;
                }
                // too little voice in it: a cough or a bump, not words
                let words = sp.loud >= 8;
                if sp.quiet >= self.end || sp.audio.len() >= LONGEST {
                    let sp = self.speech.take().unwrap();
                    self.loud_run = 0;
                    return words.then(|| Heard::Utterance(trimmed(&sp.audio, sp.quiet), sp.last_loud, sp.paused));
                }
                if self.pause > 0 && sp.quiet == self.pause && words && !sp.paused {
                    sp.paused = true;
                    return Some(Heard::Pause(trimmed(&sp.audio, sp.quiet)));
                }
                None
            }
        }
    }
}

/// The speaker: samples queued at their own rate come out at the device's.
pub struct Speaker {
    queue: Arc<Mutex<VecDeque<f32>>>,
    rate: u32,
    resampler: Mutex<Option<(u32, Resampler)>>,
}

impl Speaker {
    /// Starts the default output device on a thread of its own (it ends with `shared.stop`).
    pub fn start(shared: Arc<Shared>) -> Result<Speaker, String> {
        let queue = Arc::new(Mutex::new(VecDeque::<f32>::new()));
        let (ready, ok) = mpsc::channel();
        let q = queue.clone();
        std::thread::spawn(move || {
            let host = cpal::default_host();
            let Some(dev) = host.default_output_device() else {
                let _ = ready.send(Err("no speaker found".to_string()));
                return;
            };
            let cfg = match dev.default_output_config() {
                Ok(c) => c,
                Err(e) => {
                    let _ = ready.send(Err(format!("the speaker: {e}")));
                    return;
                }
            };
            let (rate, channels, format) = (cfg.sample_rate().0, cfg.channels() as usize, cfg.sample_format());
            let config: cpal::StreamConfig = cfg.into();
            let err = |e| eprintln!("[speaker: {e}]");
            let sh = shared.clone();
            // each device frame: the next queued sample, on every channel
            let next = move |frames: usize| -> Vec<f32> {
                let mut q = q.lock().unwrap();
                let n = frames.min(q.len());
                if n > 0 {
                    sh.played_at.store(now_ms(), Ordering::Relaxed);
                }
                let mut v: Vec<f32> = q.drain(..n).collect();
                let rms = if n > 0 { (v.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt() } else { 0.0 };
                sh.voice.store((rms * 6.0).min(1.0).to_bits(), Ordering::Relaxed);
                v.resize(frames, 0.0);
                v
            };
            let stream = match format {
                cpal::SampleFormat::F32 => dev.build_output_stream(&config, move |d: &mut [f32], _: &_| {
                    let v = next(d.len() / channels);
                    for (f, v) in d.chunks_mut(channels).zip(v) {
                        f.fill(v);
                    }
                }, err, None),
                cpal::SampleFormat::I16 => dev.build_output_stream(&config, move |d: &mut [i16], _: &_| {
                    let v = next(d.len() / channels);
                    for (f, v) in d.chunks_mut(channels).zip(v) {
                        f.fill((v.clamp(-1.0, 1.0) * 32767.0) as i16);
                    }
                }, err, None),
                f => {
                    let _ = ready.send(Err(format!("the speaker takes {f:?} samples, which this app does not write")));
                    return;
                }
            };
            let stream = match stream.map_err(|e| e.to_string()).and_then(|s| s.play().map(|()| s).map_err(|e| e.to_string())) {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready.send(Err(format!("the speaker: {e}")));
                    return;
                }
            };
            let _ = ready.send(Ok(rate));
            while !shared.stop.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            drop(stream);
        });
        let rate = ok.recv().map_err(|_| "the speaker thread stopped".to_string())??;
        Ok(Speaker { queue, rate, resampler: Mutex::new(None) })
    }

    /// Queues mono `samples` at `rate`, after what is already queued.
    pub fn play(&self, samples: &[f32], rate: u32) {
        let mut rs = self.resampler.lock().unwrap();
        if rs.as_ref().is_none_or(|(r, _)| *r != rate) {
            *rs = Some((rate, Resampler::new(rate, self.rate)));
        }
        let mut out = Vec::with_capacity(samples.len() * self.rate as usize / rate as usize + 2);
        rs.as_mut().unwrap().1.push(samples, &mut out);
        self.queue.lock().unwrap().extend(out);
    }

    /// Drops everything queued (the call was interrupted).
    pub fn stop(&self) {
        self.queue.lock().unwrap().clear();
        *self.resampler.lock().unwrap() = None;
    }

    /// Seconds still queued.
    pub fn queued(&self) -> f32 {
        self.queue.lock().unwrap().len() as f32 / self.rate as f32
    }
}

