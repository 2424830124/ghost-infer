//! The pixel ghost, the same sprite as the chat page's pixel style: a hand-drawn 14x14 body on a 24x22 grid,
//! its face set by its mood, its outline worked out from the body, drawn as squares on a canvas. It moves in
//! steps a few times a second (`t`, in milliseconds), like an old game sprite.

use iced::widget::canvas::{self, Frame, Geometry};
use iced::{mouse, Color, Point, Rectangle, Renderer, Size, Theme};

pub(crate) const W: usize = 24;
pub(crate) const H: usize = 22;
pub(crate) const OX: usize = 3;
pub(crate) const OY: usize = 6;

const TOP: [&str; 12] = [
    "....wwwwww....",
    "..wwwwwwwwww..",
    ".wwwwwwwwwwww.",
    ".wwwwwwwwwwww.",
    "wwwwwwwwwwwwww",
    "wwwwwwwwwwwwww",
    "wwwwwwwwwwwwww",
    "wwwwwwwwwwwwww",
    "wwwwwwwwwwwwww",
    "wwwwwwwwwwwwww",
    "wwwwwwwwwwwwww",
    "wwwwwwwwwwwwww",
];
/// Two hem frames: it wiggles.
const HEM: [[&str; 2]; 2] = [["wwww..wwww..ww", ".ww....ww....w"], ["ww..wwww..wwww", "w....ww....ww."]];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mood {
    Idle,
    Wave,
    Thinking,
    Talking,
    Tool,
    Waiting,
    Happy,
    Sad,
    Sleepy,
}

pub(crate) fn color(c: u8, accent: Color) -> Color {
    let rgb = |r: u8, g: u8, b: u8| Color::from_rgb8(r, g, b);
    match c {
        b'o' => rgb(0x0b, 0x0e, 0x17),
        b's' => rgb(0xae, 0xb6, 0xcc),
        b'h' => rgb(0xff, 0xff, 0xff),
        b'e' => rgb(0x12, 0x17, 0x26),
        b'b' => rgb(0xff, 0x9d, 0xb5),
        b'a' => accent,
        b'A' => Color { a: 0.75, ..accent },
        b't' => rgb(0x8f, 0xc0, 0xff),
        _ => rgb(0xe8, 0xeb, 0xf3),
    }
}

/// The ghost: `pixel` is the size of one sprite pixel, `t` the clock, `talk` whether text is arriving.
pub struct Ghost {
    pub pixel: f32,
    pub mood: Mood,
    pub t: u64,
    pub talk: bool,
    pub accent: Color,
}

impl Ghost {
    pub fn still(pixel: f32, accent: Color) -> Ghost {
        Ghost { pixel, mood: Mood::Idle, t: 1, talk: false, accent }
    }

    pub(crate) fn grid(&self) -> [[u8; W]; H] {
        let t = self.t;
        let step = (t / 420) as usize;
        let mut dy: isize = (t / if self.mood == Mood::Sleepy { 1100 } else { 420 }) as isize % 2;
        match self.mood {
            Mood::Sleepy | Mood::Sad => dy += 2,
            Mood::Happy | Mood::Wave => dy -= [0, 1, 2, 1][(t / 200) as usize % 4],
            _ => {}
        }
        let mut g = [[0u8; W]; H];
        let mut put = |x: isize, y: isize, c: u8| {
            let (px, py) = (OX as isize + x, OY as isize + y + dy);
            if px >= 0 && py >= 0 && (px as usize) < W && (py as usize) < H {
                g[py as usize][px as usize] = c;
            }
        };
        let rows: Vec<&str> = TOP.iter().chain(HEM[step % 2].iter()).copied().collect();
        let body = |x: usize, y: usize| y < rows.len() && x < 14 && rows[y].as_bytes()[x] == b'w';
        for (y, row) in rows.iter().enumerate() {
            for (x, c) in row.bytes().enumerate() {
                if c == b'w' {
                    put(x as isize, y as isize, if !body(x + 1, y) || !body(x, y + 1) { b's' } else { b'w' });
                }
            }
        }
        put(3, 1, b'h');
        put(2, 2, b'h');
        let blink = step % 9 == 0;
        let mut px = |list: &[(isize, isize)], c: u8| list.iter().for_each(|&(x, y)| put(x, y, c));
        let eyes = |px: &mut dyn FnMut(&[(isize, isize)], u8), x0: isize, y0: isize, h: isize, shine: bool| {
            for (ex, ey) in [(x0, y0), (x0 + 6, y0)] {
                for i in 0..h {
                    px(&[(ex, ey + i), (ex + 1, ey + i)], b'e');
                }
                if shine {
                    px(&[(ex, ey)], b'h');
                }
            }
        };
        let arcs = [(2, 5), (3, 4), (4, 5), (8, 5), (9, 4), (10, 5)];
        let smile = [(5, 7), (6, 8), (7, 8), (8, 7)];
        let lines = [(3, 5), (4, 5), (9, 5), (10, 5)];
        match self.mood {
            Mood::Idle => {
                if blink {
                    px(&lines, b'e');
                } else {
                    eyes(&mut px, 3, 4, 2, true);
                }
            }
            Mood::Wave => {
                px(&arcs, b'e');
                px(&smile, b'e');
                px(&[(1, 7), (12, 7)], b'b');
                px(if (t / 180) % 2 == 1 { &[(14, 6), (15, 5), (16, 4), (16, 3)] } else { &[(14, 6), (15, 5), (16, 5), (17, 4)] }, b'w');
            }
            Mood::Thinking => {
                eyes(&mut px, 4, 3, 2, false);
                for i in 0..3 {
                    if step % 4 > i {
                        px(&[(13 + 2 * i as isize, -2)], b'a');
                    }
                }
            }
            Mood::Talking => {
                eyes(&mut px, 3, 4, 2, true);
                px(&[(6, 8), (7, 8)], b'e');
                if self.talk && (t / 110) % 2 == 1 {
                    px(&[(6, 9), (7, 9)], b'e');
                }
            }
            Mood::Tool | Mood::Waiting => {
                if self.mood == Mood::Waiting {
                    eyes(&mut px, 4, 4, 2, false);
                    px(&[(10, 1), (11, 1)], b'e');
                } else {
                    eyes(&mut px, 4, 4, 2, false);
                }
                px(&[(15, 3), (16, 2), (17, 2), (18, 3), (18, 4), (17, 5), (16, 5), (15, 4)], b'a');
                if step % 3 != 0 {
                    px(&[(16, 3)], b'h');
                }
                px(&[(14, 6), (13, 7)], b'A');
            }
            Mood::Happy => {
                px(&arcs, b'e');
                px(&smile, b'e');
                px(&[(1, 7), (12, 7)], b'b');
            }
            Mood::Sad => {
                px(&lines, b'e');
                px(&[(3, 3), (4, 2), (9, 2), (10, 3), (5, 9), (6, 8), (7, 8), (8, 9)], b'e');
                px(&[(3, 6 + (step % 4) as isize)], b't');
            }
            Mood::Sleepy => {
                px(&lines, b'e');
                if step % 6 < 4 {
                    let up = if step % 6 > 1 { 1 } else { 0 };
                    px(&[(15, -3 - up), (16, -3 - up), (17, -3 - up), (17, -2 - up), (16, -1 - up), (15, 0 - up), (16, 0 - up), (17, 0 - up)], b'a');
                }
            }
        }
        // the outline: every empty cell touching the ghost (not its props)
        let solid = |g: &[[u8; W]; H], x: isize, y: isize| x >= 0 && y >= 0 && (x as usize) < W && (y as usize) < H && matches!(g[y as usize][x as usize], b'w' | b's' | b'h' | b'e' | b'b');
        let mut out = g;
        for y in 0..H {
            for x in 0..W {
                let (xi, yi) = (x as isize, y as isize);
                if g[y][x] == 0 && (solid(&g, xi - 1, yi) || solid(&g, xi + 1, yi) || solid(&g, xi, yi - 1) || solid(&g, xi, yi + 1)) {
                    out[y][x] = b'o';
                }
            }
        }
        out
    }
}

/// The window and taskbar icon: the idle ghost, two screen pixels per sprite pixel, centred in a square.
pub fn icon(accent: Color) -> Option<iced::window::Icon> {
    let (rgba, side) = icon_rgba(accent)?;
    iced::window::icon::from_rgba(rgba, side, side).ok()
}

/// The icon's pixels, RGBA, and its side (it is square): for the window and the tray.
pub fn icon_rgba(accent: Color) -> Option<(Vec<u8>, u32)> {
    let g = Ghost::still(1.0, accent).grid();
    let cells: Vec<(usize, usize)> = (0..H).flat_map(|y| (0..W).map(move |x| (x, y))).filter(|&(x, y)| g[y][x] != 0).collect();
    let (x0, x1) = (cells.iter().map(|c| c.0).min()?, cells.iter().map(|c| c.0).max()?);
    let (y0, y1) = (cells.iter().map(|c| c.1).min()?, cells.iter().map(|c| c.1).max()?);
    let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
    const SCALE: usize = 2;
    let side = w.max(h) * SCALE + 2;
    let (ox, oy) = ((side - w * SCALE) / 2, (side - h * SCALE) / 2);
    let mut rgba = vec![0u8; side * side * 4];
    for &(x, y) in &cells {
        let c = color(g[y][x], accent).into_rgba8();
        for dy in 0..SCALE {
            for dx in 0..SCALE {
                let (px, py) = (ox + (x - x0) * SCALE + dx, oy + (y - y0) * SCALE + dy);
                rgba[(py * side + px) * 4..][..4].copy_from_slice(&c);
            }
        }
    }
    Some((rgba, side as u32))
}

impl<Message> canvas::Program<Message> for Ghost {
    type State = ();

    fn draw(&self, _: &(), renderer: &Renderer, _: &Theme, bounds: Rectangle, _: mouse::Cursor) -> Vec<Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());
        let p = self.pixel;
        for (y, row) in self.grid().iter().enumerate() {
            for (x, &c) in row.iter().enumerate() {
                if c != 0 {
                    frame.fill_rectangle(Point::new(x as f32 * p, y as f32 * p), Size::new(p, p), color(c, self.accent));
                }
            }
        }
        vec![frame.into_geometry()]
    }
}
