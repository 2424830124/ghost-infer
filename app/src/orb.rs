//! The call's face: the pixel orb of design/call-orb-mockup.html. A sphere of short streaks turning, drawn small
//! (a cell every few pixels) as light, each cell's light dithered onto five colours of the skin; round it a broken
//! ring, ticks, the microphone's arc, a chase while thinking and bars with the voice, plotted as whole cells; at its
//! centre a glowing core with strands winding round it. It moves in steps, like the ghost.

use crate::view::Skin;
use iced::widget::canvas::{self, Frame, Geometry};
use iced::{mouse, Color, Point, Rectangle, Renderer, Size, Theme};
use std::f32::consts::{PI, TAU};

/// Where the call is, as the orb shows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Ready,
    Listening,
    Hearing,
    Thinking,
    Speaking,
    Muted,
}

// what each phase asks of the orb (it eases between them), as indices into the array
const SPIN: usize = 0;
const AMP: usize = 1;
const GAIN: usize = 2;
const BRIGHT: usize = 3;
const CONTRACT: usize = 4;
const SWIRL: usize = 5;
const RING: usize = 6;
const SPIKES: usize = 7;
const SPARKS: usize = 8;
const METER: usize = 9;
const CHASE: usize = 10;
const CORE: usize = 11;

fn params(p: Phase) -> [f32; 12] {
    match p {
        Phase::Ready => [0.10, 0.025, 0.0, 0.55, 1.0, 0.45, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        Phase::Listening => [0.16, 0.03, 0.22, 0.72, 1.0, 0.6, 0.08, 0.0, 0.0, 1.0, 0.0, 0.3],
        Phase::Hearing => [0.22, 0.04, 0.34, 0.88, 1.03, 0.85, 0.12, 0.0, 0.0, 1.0, 0.0, 0.5],
        Phase::Thinking => [0.42, 0.035, 0.0, 0.84, 0.84, 2.6, 0.35, 0.0, 0.0, 0.0, 1.0, 0.0],
        Phase::Speaking => [0.26, 0.045, 0.42, 0.92, 1.04, 1.1, 0.1, 1.0, 1.0, 0.0, 0.0, 1.0],
        Phase::Muted => [0.08, 0.02, 0.0, 0.4, 0.97, 0.35, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
}

// the scene is drawn in light alone: how bright each part's colour is
const CORE_L: f32 = 1.0;
const HOT_L: f32 = 0.92;
const MID_L: f32 = 0.65;
const DEEP_L: f32 = 0.41;

/// Maya's moods and the colour each gives the orb (the first, calm, is its own indigo); a mood not here keeps
/// the colour it has.
pub const MOODS: [(&str, [u8; 3]); 10] = [
    ("calm", [88, 104, 220]),
    ("happy", [236, 184, 52]),
    ("playful", [232, 92, 168]),
    ("curious", [36, 184, 184]),
    ("excited", [246, 126, 40]),
    ("tender", [232, 116, 138]),
    ("thoughtful", [132, 108, 222]),
    ("sad", [86, 122, 176]),
    ("annoyed", [222, 70, 58]),
    ("tired", [138, 138, 164]),
];

/// The five colours a cell can take, faint to bright, from the orb's colour: from deep to white on Graphite,
/// from pale to ink on Fog.
fn ramp(s: &Skin, tint: [f32; 3]) -> [Color; 5] {
    let scale = |k: f32| Color::from_rgb(tint[0] * k, tint[1] * k, tint[2] * k);
    let pale = |k: f32| Color::from_rgb(tint[0] + (1.0 - tint[0]) * k, tint[1] + (1.0 - tint[1]) * k, tint[2] + (1.0 - tint[2]) * k);
    if s.dark {
        [scale(0.32), scale(0.85), pale(0.55), pale(0.88), Color::WHITE]
    } else {
        [pale(0.78), pale(0.45), scale(0.72), scale(0.45), scale(0.1)]
    }
}

fn rgb(c: [u8; 3]) -> [f32; 3] {
    c.map(|v| v as f32 / 255.0)
}

fn bayer(x: usize, y: usize) -> f32 {
    const B: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
    (B[y & 3][x & 3] as f32 + 0.5) / 16.0
}

struct Spark {
    a: f32,
    r: f32,
    v: f32,
    life: f32,
    max: f32,
    len: f32,
}

/// The orb's motion, stepped on the app's clock.
pub struct Sim {
    t: f32,
    rot: f32,
    ring_rot: f32,
    swirl: f32,
    chase: f32,
    lv: f32,
    cur: [f32; 12],
    sparks: Vec<Spark>,
    spark_acc: f32,
    seed: u32,
    phase: Phase,
    last: Option<u64>,
    tint: [f32; 3],    // its colour now, easing to `tint_to` (her mood's)
    tint_to: [f32; 3],
}

impl Sim {
    pub fn new() -> Sim {
        Sim {
            t: 0.0,
            rot: 0.0,
            ring_rot: 0.0,
            swirl: 0.0,
            chase: 0.0,
            lv: 0.0,
            cur: params(Phase::Ready),
            sparks: Vec::new(),
            spark_acc: 0.0,
            seed: 0x9e37_79b9,
            phase: Phase::Ready,
            last: None,
            tint: rgb(MOODS[0].1),
            tint_to: rgb(MOODS[0].1),
        }
    }

    /// Her mood, as the orb's colour ("" for calm, the call's start); one not in MOODS leaves it as it is.
    pub fn set_mood(&mut self, mood: &str) {
        let mood = if mood.is_empty() { "calm" } else { mood };
        if let Some((_, c)) = MOODS.iter().find(|(m, _)| *m == mood) {
            self.tint_to = rgb(*c);
        }
    }

    fn rand(&mut self) -> f32 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        (self.seed >> 8) as f32 / (1 << 24) as f32
    }

    /// Moves on to `clock` (ms): the call's phase, the level that moves the orb (the microphone while listening,
    /// the voice while speaking).
    pub fn step(&mut self, clock: u64, phase: Phase, level: f32) {
        let dt = self.last.map_or(0.0, |l| clock.saturating_sub(l) as f32 / 1000.0).min(0.25);
        self.last = Some(clock);
        self.phase = phase;
        self.t += dt;
        let k = 1.0 - (-dt * 3.5).exp();
        let tgt = params(phase);
        for (c, t) in self.cur.iter_mut().zip(tgt) {
            *c += (t - *c) * k;
        }
        // a new mood's colour comes in over about a second
        let kt = 1.0 - (-dt * 2.5).exp();
        for (c, t) in self.tint.iter_mut().zip(self.tint_to) {
            *c += (t - *c) * kt;
        }
        // quick up, slow down (per sixtieth of a second, as the mockup's frames)
        let raw = if phase == Phase::Muted { 0.0 } else { level };
        let rate: f32 = if raw > self.lv { 0.45 } else { 0.12 };
        self.lv += (raw - self.lv) * (1.0 - (1.0 - rate).powf(dt * 60.0));
        let c = self.cur;
        self.rot += c[SPIN] * dt;
        self.ring_rot += c[RING] * dt;
        self.swirl += c[SWIRL] * dt;
        self.chase += dt * (1.0 + 3.5 * c[CHASE]);

        // sparks thrown off the shell while speaking
        self.spark_acc += dt * 110.0 * self.lv * c[SPARKS];
        while self.spark_acc >= 1.0 {
            self.spark_acc -= 1.0;
            let s = Spark { a: self.rand() * TAU, r: 0.9 + self.rand() * 0.12, v: 0.45 + self.rand() * 0.9, life: 0.0, max: 0.45 + self.rand() * 0.5, len: 0.04 + self.rand() * 0.12 };
            self.sparks.push(s);
        }
        for s in &mut self.sparks {
            s.r += s.v * dt;
            s.life += dt;
        }
        self.sparks.retain(|s| s.life < s.max);
    }
}

// the particles, strands and ring segments, the same every time (a seeded generator)
struct Part {
    x: f32,
    y: f32,
    z: f32,
    s: f32,
    len: f32,
    seed: f32,
    lum: f32,
}
struct Strand {
    u: [f32; 3],
    v: [f32; 3],
    r: f32,
    speed: f32,
    off: f32,
    span: f32,
    a: f32,
}
struct Seg {
    a: f32,
    len: f32,
    al: f32,
}
struct Set {
    parts: Vec<Part>,
    strands: Vec<Strand>,
    segs: Vec<Seg>,
}

impl Set {
    fn new(n: usize) -> Set {
        let mut x: i32 = 7;
        let mut rnd = move || {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            ((x as u32) % 100000) as f32 / 100000.0
        };
        let golden = PI * (3.0 - 5f32.sqrt());
        let parts = (0..n)
            .map(|i| {
                let y = 1.0 - (i as f32 + 0.5) / n as f32 * 2.0;
                let rr = (1.0 - y * y).sqrt();
                let th = golden * i as f32 + (rnd() - 0.5) * 0.3;
                let inner = rnd() < 0.2;
                let s = if inner { 0.48 + rnd() * 0.28 } else { 0.9 + rnd() * 0.1 };
                Part { x: th.cos() * rr, y, z: th.sin() * rr, s, len: 0.4 + rnd() * 1.3, seed: rnd(), lum: 0.45 + rnd() * 0.55 }
            })
            .collect();
        let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let unit = |a: [f32; 3]| {
            let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            [a[0] / l, a[1] / l, a[2] / l]
        };
        let strands = (0..13)
            .map(|i| {
                let wide = i == 0;
                let n = unit([rnd() - 0.5, rnd() - 0.5, rnd() - 0.5]);
                let a = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
                let u = unit(cross(n, a));
                let v = cross(n, u);
                let r = if wide { 0.7 } else { 0.08 + rnd() * 0.4 };
                let speed = (if wide { 0.6 } else { 1.0 + rnd() * 1.6 }) * if rnd() < 0.3 { -1.0 } else { 1.0 };
                let off = rnd() * 6.28;
                let span = if wide { 2.3 } else { 1.4 + rnd() * 2.2 };
                let _w = if wide { 1.6 } else { 0.7 + rnd() * 1.1 };
                let a = if wide { 0.55 } else { 0.45 + rnd() * 0.5 };
                Strand { u, v, r, speed, off, span, a }
            })
            .collect();
        let mut segs = Vec::new();
        let mut a = 0.0;
        while a < TAU {
            let len = 0.04 + rnd() * 0.42;
            let _w = rnd();
            let al = 0.15 + rnd() * 0.35;
            let _dbl = rnd();
            segs.push(Seg { a, len: len.min(TAU - a), al });
            a += len + 0.02 + rnd() * 0.16;
        }
        Set { parts, strands, segs }
    }
}

/// An n x n field of light, drawn into: lines, discs fading out.
struct Light {
    n: usize,
    v: Vec<f32>,
}

impl Light {
    fn add(&mut self, x: f32, y: f32, w: f32) {
        // shared between the four nearest cell centres
        let (fx, fy) = (x - 0.5, y - 0.5);
        let (ix, iy) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - ix, fy - iy);
        for (dx, dy, k) in [(0, 0, (1.0 - tx) * (1.0 - ty)), (1, 0, tx * (1.0 - ty)), (0, 1, (1.0 - tx) * ty), (1, 1, tx * ty)] {
            let (cx, cy) = (ix as i32 + dx, iy as i32 + dy);
            if cx >= 0 && cy >= 0 && (cx as usize) < self.n && (cy as usize) < self.n {
                self.v[cy as usize * self.n + cx as usize] += w * k;
            }
        }
    }

    /// A line a cell wide, `w` of light along each cell of its length.
    fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, w: f32) {
        let len = (x1 - x0).hypot(y1 - y0);
        if len < 1e-3 {
            return;
        }
        let steps = (len * 2.0).ceil().max(1.0);
        let each = w * len / steps;
        for i in 0..steps as usize {
            let s = (i as f32 + 0.5) / steps;
            self.add(x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, each);
        }
    }

    /// A disc of radius `r` whose light falls off through `stops` (where, how much), as a radial gradient.
    fn disc(&mut self, cx: f32, cy: f32, r: f32, stops: &[(f32, f32)]) {
        let n = self.n;
        for y in 0..n {
            for x in 0..n {
                let d = (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy) / r;
                if d >= 1.0 {
                    continue;
                }
                let i = stops.iter().position(|s| s.0 >= d).unwrap_or(stops.len() - 1).max(1);
                let ((p0, v0), (p1, v1)) = (stops[i - 1], stops[i]);
                let k = if p1 > p0 { ((d - p0) / (p1 - p0)).clamp(0.0, 1.0) } else { 1.0 };
                self.v[y * n + x] += v0 + (v1 - v0) * k;
            }
        }
    }
}

/// The orb, drawn on a canvas: `cell` is the size of one of its cells.
pub struct Orb<'a> {
    pub sim: &'a Sim,
    pub skin: &'static Skin,
    pub cell: f32,
}

impl Orb<'_> {
    /// The sphere, its haze and core, the strands round the core and the sparks, as light.
    fn scene(&self, set: &Set, n: usize) -> Light {
        let (sim, c) = (self.sim, &self.sim.cur);
        let (lv, t) = (sim.lv, sim.t);
        let nf = n as f32;
        let (r, cx, cy) = (nf * 0.3, nf / 2.0, nf / 2.0);
        let bright = (c[BRIGHT] + 0.3 * lv * (c[GAIN] * 4.0).min(1.0)).min(1.25);
        let amp = c[AMP] + c[GAIN] * lv;
        let breath = 1.0 + 0.018 * (t * 1.3).sin();
        let mut l = Light { n, v: vec![0.0; n * n] };

        l.disc(cx, cy, r * 1.55, &[(0.0, 0.16 * bright * DEEP_L), (0.7, 0.05 * bright * DEEP_L), (1.0, 0.0)]);
        l.disc(cx, cy, r * 0.85, &[(0.0, 0.15 * bright * MID_L), (1.0, 0.0)]);

        // the shell, as streaks along the turn
        let tilt = 0.36 + 0.08 * (t * 0.21).sin();
        let (ct, st) = (tilt.cos(), tilt.sin());
        let trail = (0.045 + c[SPIN] * 0.14) * 1.6;
        let contract = c[CONTRACT] * breath;
        let persp = 3.2;
        let project = |x: f32, y: f32, z: f32, a: f32| {
            let (ca, sa) = (a.cos(), a.sin());
            let (x1, z1) = (x * ca + z * sa, -x * sa + z * ca);
            let (y2, z2) = (y * ct - z1 * st, y * st + z1 * ct);
            let k = persp / (persp + z2);
            (cx + x1 * r * k, cy + y2 * r * k, z2)
        };
        for p in &set.parts {
            let f = 0.5 * (2.1 * p.x + 1.7 * t).sin() * (1.9 * p.y - 1.1 * t).cos()
                + 0.32 * (3.3 * p.z + 2.3 * p.y + 1.5 * t).sin()
                + 0.22 * (5.1 * p.x - 4.2 * p.z + 2.7 * t + 9.0 * lv).sin();
            let rr = p.s * contract * (1.0 + amp * f * if p.s > 0.8 { 1.0 } else { 0.5 });
            let (x, y, z) = (p.x * rr, p.y * rr, p.z * rr);
            let (x1, y1, z2) = project(x, y, z, sim.rot);
            let (x0, y0, _) = project(x, y, z, sim.rot - trail * p.len);
            let rim = 1.0 - (z2 / rr).abs().min(1.0);
            let mut al = bright * p.lum * (0.22 + 0.78 * rim.powf(1.6)) * if z2 < 0.0 { 1.0 } else { 0.5 };
            if p.s < 0.8 {
                al *= 0.6;
            }
            let light = if rim > 0.75 && p.seed > 0.45 { HOT_L } else if p.seed > 0.3 { MID_L } else { DEEP_L };
            let q = (al * 6.0).floor().clamp(0.0, 5.0);
            l.line(x0, y0, x1, y1, (q + 0.6) / 6.0 * light);
        }

        // the strands winding round the core
        {
            let sa = sim.rot * 0.5;
            let segs = 16;
            for s in &set.strands {
                let head = sim.swirl * s.speed + s.off;
                let rad = s.r * (0.85 + 0.15 * contract) * (1.0 + 0.25 * lv * c[CORE]);
                let mut prev = (0.0, 0.0);
                for k in 0..=segs {
                    let th = head - s.span * (1.0 - k as f32 / segs as f32) * s.speed.signum();
                    let (cs, sn) = (th.cos() * rad, th.sin() * rad);
                    let p = [s.u[0] * cs + s.v[0] * sn, s.u[1] * cs + s.v[1] * sn, s.u[2] * cs + s.v[2] * sn];
                    let (x, y, _) = project(p[0], p[1], p[2], sa);
                    if k > 0 {
                        let f = k as f32 / segs as f32;
                        l.line(prev.0, prev.1, x, y, f.powf(1.6) * s.a * bright * if f > 0.7 { CORE_L } else { HOT_L });
                    }
                    prev = (x, y);
                }
            }
        }

        // the core
        let pulse = if c[CHASE] > 0.5 { 0.9 + 0.12 * (t * 9.0).sin() } else { 1.0 };
        let core_r = r * (0.095 + 0.07 * lv * c[CORE] + 0.01 * (t * 2.2).sin()) * pulse * 1.4;
        let stops = [(0.0, (0.9 * bright).min(1.0) * CORE_L), (0.14, 0.5 * bright * CORE_L), (0.4, 0.18 * bright * HOT_L), (1.0, 0.0)];
        l.disc(cx, cy, core_r * 2.6, &stops);

        for s in &sim.sparks {
            let f = 1.0 - s.life / s.max;
            let (ca, sa) = (s.a.cos(), s.a.sin());
            l.line(cx + ca * r * (s.r - s.len), cy + sa * r * (s.r - s.len), cx + ca * r * s.r, cy + sa * r * s.r, 0.8 * f * HOT_L);
        }
        l
    }

    /// The rings, plotted as whole cells of the ramp (a dithered line a cell wide breaks up): the broken ring and
    /// its ticks, the meter ring with the microphone's arc, the thinking chase, the speaking bars.
    fn rings(&self, set: &Set, lvl: &mut [u8], n: usize) {
        let (sim, c) = (self.sim, &self.sim.cur);
        let nf = n as f32;
        let (r, cx, cy) = (nf * 0.3, nf / 2.0, nf / 2.0);
        let mut plot = |x: f32, y: f32, l: u8| {
            let (x, y) = (x.floor(), y.floor());
            if x >= 0.0 && y >= 0.0 && (x as usize) < n && (y as usize) < n {
                lvl[y as usize * n + x as usize] = l;
            }
        };
        let arc = |plot: &mut dyn FnMut(f32, f32, u8), rad: f32, a0: f32, a1: f32, l: u8| {
            let mut a = a0;
            while a <= a1 {
                plot(cx + a.cos() * rad, cy + a.sin() * rad, l);
                a += 0.5 / rad;
            }
        };
        for s in &set.segs {
            let a = s.a + sim.ring_rot;
            arc(&mut plot, r * 1.22, a, a + s.len, if s.al > 0.35 { 3 } else { 2 });
        }
        let (r2, ticks) = (r * 1.38, 32);
        for i in 0..ticks {
            let fi = i as f32;
            if fi > ticks as f32 * 0.65 && fi < ticks as f32 * 0.8 {
                continue;
            }
            let a = fi / ticks as f32 * TAU - sim.ring_rot * 0.6;
            plot(cx + a.cos() * r2, cy + a.sin() * r2, 2);
            if i % 4 == 0 {
                plot(cx + a.cos() * (r2 + 1.0), cy + a.sin() * (r2 + 1.0), 2);
            }
        }
        let r3 = r * 1.09;
        arc(&mut plot, r3, 0.0, TAU, 1);
        if c[METER] > 0.3 {
            let hl = 0.04 + sim.lv * PI * 0.9;
            arc(&mut plot, r3, -PI / 2.0 - hl, -PI / 2.0 + hl, 4);
        }
        if c[CHASE] > 0.3 {
            for k in 0..2 {
                let a = sim.chase * 2.2 + k as f32 * PI;
                arc(&mut plot, r3, a, a + 0.55 + 0.25 * (sim.t * 3.0).sin(), 4);
            }
        }
        if c[SPIKES] > 0.3 {
            let bars = 36;
            for i in 0..bars {
                let fi = i as f32;
                let a = fi / bars as f32 * TAU - PI / 2.0;
                let w = 0.5 + 0.5 * (fi * 1.7 + sim.t * 6.3).sin() * (fi * 0.61 - sim.t * 3.3).sin();
                let len = (r * 0.2 * sim.lv * (0.2 + 0.8 * w)).round() as i32;
                for j in 0..=len {
                    let d = r3 + 2.0 + j as f32;
                    plot(cx + a.cos() * d, cy + a.sin() * d, if j == len { 5 } else { 3 });
                }
            }
        }
    }
}

/// Fills a w x h grid of cells from `x0, y0`, a run of same-coloured cells in a row as one rectangle.
fn runs(frame: &mut Frame, w: usize, h: usize, x0: f32, y0: f32, cell: f32, f: impl Fn(usize, usize) -> Option<Color>) {
    for y in 0..h {
        let mut x = 0;
        while x < w {
            let Some(c) = f(x, y) else {
                x += 1;
                continue;
            };
            let mut e = x + 1;
            while e < w && f(e, y) == Some(c) {
                e += 1;
            }
            frame.fill_rectangle(Point::new(x0 + x as f32 * cell, y0 + y as f32 * cell), Size::new((e - x) as f32 * cell, cell), c);
            x = e;
        }
    }
}

impl<Message> canvas::Program<Message> for Orb<'_> {
    type State = ();

    fn draw(&self, _: &(), renderer: &Renderer, _: &Theme, bounds: Rectangle, _: mouse::Cursor) -> Vec<Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());
        let cell = self.cell;
        let n = (bounds.width.min(bounds.height) / cell).floor() as usize;
        if n < 8 {
            return vec![frame.into_geometry()];
        }
        let (ox, oy) = (((bounds.width - n as f32 * cell) / 2.0).floor(), ((bounds.height - n as f32 * cell) / 2.0).floor());
        let set = Set::new(((n * n) as f32 * 0.17).clamp(90.0, 700.0) as usize);

        // the light, dithered onto the ramp's five colours; then the rings over it
        let light = self.scene(&set, n);
        let mut lvl = vec![0u8; n * n];
        for y in 0..n {
            for x in 0..n {
                let l = light.v[y * n + x].min(1.0).powf(1.35) * 5.0;
                let mut v = l.floor();
                if l - v > bayer(x, y) {
                    v += 1.0;
                }
                lvl[y * n + x] = v.min(5.0) as u8;
            }
        }
        self.rings(&set, &mut lvl, n);
        let ramp = ramp(self.skin, self.sim.tint);
        runs(&mut frame, n, n, ox, oy, cell, |x, y| match lvl[y * n + x] {
            0 => None,
            l => Some(ramp[l as usize - 1]),
        });
        vec![frame.into_geometry()]
    }
}
