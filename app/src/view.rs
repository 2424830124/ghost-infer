//! How the app looks: Claude Desktop's calm structure in ghost-infer's colours, as two skins: Graphite
//! (dark, neutral greys) and Fog (light). Icons come from Segoe MDL2 Assets, the icon font every Windows 10
//! and 11 has; text is Instrument Sans, the greeting Georgia.

use iced::widget::{button, canvas, column, container, markdown, mouse_area, opaque, rich_text, row, scrollable, stack, text, text_editor, text_input, tooltip, Space};
use iced::{border, keyboard, Alignment, Background, Border, Color, Element, Fill, Font, Length, Padding, Pixels, Theme};

use crate::ghost::{Ghost, Mood};
use crate::json::Json;
use crate::{approval_text, group_name, tool_detail, tool_label, App, Menu, Message, Msg, Role, TaskState};

/// A skin: the colours every part of the app is drawn with.
#[derive(Debug)]
pub struct Skin {
    pub bg: Color,
    pub side: Color,
    pub raise: Color,
    pub raise2: Color,
    pub hover: Color,
    pub menu: Color,
    pub line: Color,
    pub line2: Color,
    pub text: Color,
    pub muted: Color,
    pub faint: Color,
    pub accent: Color,
    pub ink: Color,
    pub ok: Color,
    pub bad: Color,
    pub warn: Color,
    pub bubble: Color,
    pub dark: bool,
}

pub const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color { r: r as f32 / 255.0, g: g as f32 / 255.0, b: b as f32 / 255.0, a: 1.0 }
}
const fn rgba(r: u8, g: u8, b: u8, a: f32) -> Color {
    Color { a, ..rgb(r, g, b) }
}

pub static GRAPHITE: Skin = Skin {
    bg: rgb(0x1d, 0x1d, 0x1f),
    side: rgb(0x17, 0x17, 0x19),
    raise: rgb(0x2a, 0x2a, 0x2d),
    raise2: rgb(0x33, 0x33, 0x37),
    hover: rgb(0x26, 0x26, 0x29),
    menu: rgb(0x2b, 0x2b, 0x2f),
    line: rgba(255, 255, 255, 0.06),
    line2: rgba(255, 255, 255, 0.10),
    text: rgb(0xec, 0xec, 0xee),
    muted: rgb(0xa3, 0xa3, 0xaa),
    faint: rgb(0x6f, 0x6f, 0x77),
    accent: rgb(0xb9, 0xc7, 0xff),
    ink: rgb(0x14, 0x17, 0x2a),
    ok: rgb(0x8f, 0xd6, 0xa2),
    bad: rgb(0xff, 0x8f, 0x80),
    warn: rgb(0xf2, 0xc2, 0x7a),
    bubble: rgb(0x2f, 0x2f, 0x33),
    dark: true,
};

pub static FOG: Skin = Skin {
    bg: rgb(0xf5, 0xf5, 0xf7),
    side: rgb(0xec, 0xec, 0xef),
    raise: rgb(0xff, 0xff, 0xff),
    raise2: rgb(0xe9, 0xe9, 0xee),
    hover: rgb(0xe4, 0xe4, 0xea),
    menu: rgb(0xff, 0xff, 0xff),
    line: rgba(20, 22, 30, 0.07),
    line2: rgba(20, 22, 30, 0.12),
    text: rgb(0x18, 0x19, 0x1f),
    muted: rgb(0x5c, 0x5e, 0x6b),
    faint: rgb(0x92, 0x94, 0x9f),
    accent: rgb(0x3d, 0x4a, 0xa8),
    ink: rgb(0xff, 0xff, 0xff),
    ok: rgb(0x1f, 0x8a, 0x4c),
    bad: rgb(0xc4, 0x45, 0x2f),
    warn: rgb(0xa8, 0x6a, 0x12),
    bubble: rgb(0xe6, 0xe8, 0xf2),
    dark: false,
};

impl Skin {
    pub fn theme(&self) -> Theme {
        Theme::custom(
            if self.dark { "Graphite" } else { "Fog" },
            iced::theme::Palette { background: self.bg, text: self.text, primary: self.accent, success: self.ok, warning: self.warn, danger: self.bad },
        )
    }
}

pub const BODY: Font = Font::with_name("Instrument Sans");
const MEDIUM: Font = Font { weight: iced::font::Weight::Medium, ..BODY };
const SERIF: Font = Font::with_name("Georgia");
const ICONS: Font = Font::with_name("Segoe MDL2 Assets");
const MONO: Font = Font::with_name("Consolas");

/// Segoe MDL2 Assets code points.
mod ic {
    pub const ADD: char = '\u{E710}';
    pub const DELETE: char = '\u{E74D}';
    pub const SETTINGS: char = '\u{E713}';
    pub const CLOSE_PANE: char = '\u{E89F}';
    pub const MINIMIZE: char = '\u{E921}';
    pub const MAXIMIZE: char = '\u{E922}';
    pub const RESTORE: char = '\u{E923}';
    pub const CLOSE: char = '\u{E8BB}';
    pub const DOCK_RIGHT: char = '\u{E90D}';
    pub const CHIP: char = '\u{E950}';
    pub const POWER: char = '\u{E7E8}';
    pub const SKILL: char = '\u{E8F1}';
    pub const PLUG: char = '\u{E71B}';
    pub const TASKS: char = '\u{E9D5}';
    pub const MEMORY: char = '\u{E734}';
    pub const NOW: char = '\u{E91F}';
    pub const TODO: char = '\u{EA3A}';
    pub const OPEN_PANE: char = '\u{E8A0}';
    pub const UP: char = '\u{E74A}';
    pub const CHEVRON_UP: char = '\u{E70E}';
    pub const RIGHT: char = '\u{E76C}';
    pub const DOWN: char = '\u{E70D}';
    pub const COPY: char = '\u{E8C8}';
    pub const RETRY: char = '\u{E72C}';
    pub const FILE: char = '\u{E8A5}';
    pub const FOLDER: char = '\u{E8B7}';
    pub const SEARCH: char = '\u{E721}';
    pub const EDIT: char = '\u{E70F}';
    pub const TERMINAL: char = '\u{E756}';
    pub const GLOBE: char = '\u{E774}';
    pub const PC: char = '\u{E7F8}';
    pub const BOLT: char = '\u{E945}';
    pub const CHECK: char = '\u{E73E}';
    pub const CANCEL: char = '\u{E711}';
    pub const CHAT: char = '\u{E8BD}';
    pub const SUN: char = '\u{E706}';
    pub const MOON: char = '\u{E708}';
    pub const CLOCK: char = '\u{E916}';
    pub const UNDO: char = '\u{E7A7}';
    pub const DOWNLOAD: char = '\u{E896}';
    pub const OPEN_NEW: char = '\u{E8A7}';
    pub const FLAG: char = '\u{E7C1}';
    pub const REPEAT: char = '\u{E8EE}';
}

fn icon<'a>(c: char, size: impl Into<Pixels>) -> text::Text<'a> {
    text(c.to_string()).font(ICONS).size(size).line_height(1.0)
}

/// The icon for a tool's call in the reply.
fn tool_icon(name: &str) -> char {
    match name {
        "read_file" => ic::FILE,
        "list_files" => ic::FOLDER,
        "search_files" | "web_search" => ic::SEARCH,
        "write_file" | "edit_file" => ic::EDIT,
        "run_command" | "start_process" | "check_process" | "stop_process" => ic::TERMINAL,
        "fetch_url" => ic::GLOBE,
        "update_tasks" => ic::TASKS,
        "remember" | "forget" => ic::MEMORY,
        "use_skill" => ic::SKILL,
        "search_chats" => ic::SEARCH,
        "read_chat" => ic::CHAT,
        "check_work" => ic::RETRY,
        "keep_going" => ic::TASKS,
        "compacted" => ic::MEMORY,
        "subagent" => ic::CHAT,
        "schedule_task" | "list_schedules" | "cancel_schedule" => ic::CLOCK,
        n if n.starts_with("mcp__") => ic::PLUG,
        _ => ic::PC,
    }
}

/// The context breakdown's colours (the other parts are greys of the theme's muted text).
const PART_BLUE: Color = rgb(0x5b, 0x8d, 0xef);
const PART_ORANGE: Color = rgb(0xe0, 0x7b, 0x39);
const PART_GREEN: Color = rgb(0x4c, 0xb0, 0x8c);

/// A stop button as Claude's background tasks have it: a small square.
fn stop_button(s: &'static Skin, msg: Message) -> Element<'static, Message> {
    button(container(Space::new().width(8).height(8)).style(move |_| container::Style { background: Some(Background::Color(s.text)), border: border::rounded(2), ..Default::default() }))
        .on_press(msg)
        .padding(5)
        .style(move |_, st| bare(s, st))
        .into()
}

/// How long something has run, as Claude's background tasks show it: 16s, 2m 5s, 1h 3m.
fn duration(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m {}s", s / 60, s % 60),
        _ => format!("{}h {}m", s / 3600, s % 3600 / 60),
    }
}

/// A token count as Claude's rail writes it: 400, 9.7k, 534.9k, 1M.
fn tokens_k(n: f64) -> String {
    if n >= 1_000_000.0 {
        format!("{:.1}M", n / 1_000_000.0).replace(".0M", "M")
    } else if n >= 1000.0 {
        format!("{:.1}k", n / 1000.0).replace(".0k", "k")
    } else {
        format!("{n:.0}")
    }
}

/// A menu line: its icon, name and a line under it, what it does, drawn in red, a check (shown when on), and
/// a key at its right.
struct Item {
    icon: Option<char>,
    label: String,
    detail: Option<String>,
    hint: Option<String>,
    msg: Message,
    danger: bool,
    check: Option<bool>,
}

/// "Good morning" etc., from the computer's clock.
fn greeting() -> &'static str {
    #[repr(C)]
    #[derive(Default)]
    struct SystemTime {
        year: u16,
        month: u16,
        dow: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
        ms: u16,
    }
    unsafe extern "system" {
        fn GetLocalTime(t: *mut SystemTime);
    }
    let mut t = SystemTime::default();
    unsafe { GetLocalTime(&mut t) };
    match t.hour {
        5..=11 => "Good morning",
        12..=17 => "Good afternoon",
        _ => "Good evening",
    }
}

/// How long ago a chat changed, roughly: Today, Yesterday, 3 days ago.
fn ago(ms: f64) -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as f64).unwrap_or(0.0);
    match ((now - ms) / 86_400_000.0).floor() as i64 {
        ..=0 => "Today".into(),
        1 => "Yesterday".into(),
        d => format!("{d} days ago"),
    }
}

impl App {
    pub fn view(&self) -> Element<'_, Message> {
        let s = self.skin();
        let mut body = row![];
        if self.sidebar {
            body = body.push(self.sidebar());
        }
        body = body.push(self.main());
        if self.preview_shown() {
            body = body.push(self.preview_pane());
        }
        if self.tasks_panel {
            body = body.push(self.tasks_panel());
        }
        if self.rail {
            body = body.push(self.rail());
        }
        let body: Element<'_, Message> = body.into();
        let Some(menu) = self.menu else { return self.framed(self.with_slash(body)) };
        // an open menu: a click anywhere else closes it
        let shade = mouse_area(container(Space::new()).width(Fill).height(Fill)).on_press(Message::Menu(None));
        let item = |c: char, label: &str, msg: Message| Item { icon: Some(c), label: label.to_string(), detail: None, hint: None, msg, danger: false, check: None };
        // a menu above its button: its left edge with the button's, or (right) its right edge
        let above = |right: bool| match self.menu_at {
            Some(at) if right => Padding { right: (self.window.width - at.x - at.width).max(0.0), bottom: (self.window.height - at.y + 6.0).max(0.0), ..Padding::ZERO },
            Some(at) => Padding { left: at.x, bottom: (self.window.height - at.y + 6.0).max(0.0), ..Padding::ZERO },
            None => Padding { left: if self.sidebar { 264.0 + 60.0 } else { 60.0 }, bottom: 130.0, ..Padding::ZERO },
        };
        let (title, items, pad, width, right): (Option<&str>, Vec<Item>, Padding, f32, bool) = match menu {
            Menu::Chat => (
                None,
                vec![item(ic::COPY, "Copy as Markdown", Message::CopyChat), Item { danger: true, ..item(ic::DELETE, "Delete", Message::DeleteCurrent) }],
                Padding { top: 44.0, left: if self.sidebar { 264.0 + 16.0 } else { 56.0 }, ..Padding::ZERO },
                232.0,
                false,
            ),
            Menu::Settings => (
                None,
                vec![
                    item(if s.dark { ic::SUN } else { ic::MOON }, if s.dark { "Light theme (Fog)" } else { "Dark theme (Graphite)" }, Message::ToggleTheme),
                    item(ic::SETTINGS, "Follow Windows' theme", Message::SystemTheme),
                    Item { check: Some(self.load_last), ..item(ic::BOLT, "Start with the last model", Message::ToggleLoadLast) },
                    Item { check: Some(self.warm_up), ..item(ic::NOW, "Warm up after loading", Message::ToggleWarmUp) },
                    item(ic::POWER, "Quit and stop the model", Message::Quit),
                ],
                Padding { left: 12.0, bottom: 60.0, ..Padding::ZERO },
                232.0,
                false,
            ),
            // adding things; the tools themselves are always there, and the mode decides what needs an OK
            Menu::Plus => (
                None,
                vec![item(ic::FOLDER, "Add folder…", Message::AddFolder), item(ic::SKILL, "Install skill…", Message::InstallSkill(false)), item(ic::DOWNLOAD, "Install skill from a link…", Message::OpenSkillLink)],
                above(false),
                232.0,
                false,
            ),
            Menu::Model => {
                let current = self.model.as_ref().map(|m| m.id.as_str());
                let items = self.models.iter().map(|m| Item { icon: None, check: Some(current == Some(m.id.as_str())), ..item(ic::BOLT, &m.label, Message::PickModel(m.clone())) }).collect();
                (Some("Model"), items, above(true), 260.0, true)
            }
            Menu::Mode => {
                let items = crate::MODES
                    .iter()
                    .enumerate()
                    .map(|(i, (_, name, what))| Item { icon: None, detail: Some(what.to_string()), hint: Some((i + 1).to_string()), check: Some(i == self.perm), ..item(ic::BOLT, name, Message::SetMode(i)) })
                    .collect();
                (Some("Mode"), items, above(true), 300.0, true)
            }
            Menu::Folder => {
                let current = self.new_chat_dir();
                let mut dirs: Vec<&str> = self.folders.iter().map(String::as_str).collect();
                if let Some(w) = self.workdir.as_deref().filter(|w| !dirs.contains(w)) {
                    dirs.insert(0, w);
                }
                let mut items: Vec<Item> = dirs
                    .into_iter()
                    .map(|d| Item { detail: Some(d.to_string()), check: Some(current == Some(d)), ..item(ic::FOLDER, &crate::folder_name(d), Message::SetWorkdir(Some(d.to_string()))) })
                    .collect();
                items.push(item(ic::ADD, "Choose a folder…", Message::PickWorkdir));
                (Some("Work in"), items, above(false), 300.0, false)
            }
        };
        let mut list = column![].spacing(1);
        if let Some(title) = title {
            list = list.push(container(text(title).size(12.5).font(MEDIUM).color(s.faint)).padding(Padding { top: 4.0, right: 10.0, bottom: 4.0, left: 10.0 }));
        }
        for it in items {
            let color = if it.danger { s.bad } else { s.text };
            let mut name = column![text(it.label).size(14).color(color)].spacing(2).width(Fill);
            if let Some(d) = it.detail {
                name = name.push(text(d).size(12).color(s.faint));
            }
            let mut line = row![].spacing(12).align_y(Alignment::Center);
            if let Some(c) = it.icon {
                line = line.push(icon(c, 13).color(if it.danger { s.bad } else { s.muted }));
            }
            line = line.push(name);
            if let Some(on) = it.check {
                line = line.push(icon(ic::CHECK, 12).color(if on { s.accent } else { Color::TRANSPARENT }));
            }
            if let Some(h) = it.hint {
                line = line.push(text(h).size(12).color(s.faint));
            }
            list = list.push(button(line).on_press(it.msg).width(Fill).padding([7, 10]).style(move |_, st| row_style(s, st, false)));
        }
        let card = container(list).padding(6).width(width).style(move |_| container::Style {
            background: Some(Background::Color(s.menu)),
            border: Border { radius: 12.0.into(), width: 1.0, color: s.line2 },
            ..Default::default()
        });
        let placed = container(opaque(card))
            .padding(pad)
            .width(Fill)
            .height(Fill)
            .align_x(if right { Alignment::End } else { Alignment::Start })
            .align_y(if matches!(menu, Menu::Chat) { Alignment::Start } else { Alignment::End });
        self.framed(stack![body, opaque(shade), placed].into())
    }

    /// The / list over the chat, just above the message box: the commands and skills whose names have what
    /// is typed after the /. A click or Tab/Enter picks one, the arrows move, Esc closes it.
    /// A stack either way: the body's place in the tree stays the same as the list opens and closes, so the
    /// message box keeps its focus.
    fn with_slash<'a>(&'a self, body: Element<'a, Message>) -> Element<'a, Message> {
        let list = self.slash_list();
        let Some(at) = self.composer_at.filter(|_| !list.is_empty()) else { return stack![body, Space::new()].into() };
        let s = self.skin();
        let pick = self.slash_pick.min(list.len() - 1);
        // seven lines at most, the highlighted one among them
        let mut lines = column![].spacing(1);
        for (i, e) in list.iter().enumerate().skip(pick.saturating_sub(6)).take(7) {
            let name = row![text(crate::slash_label(&e.name)).size(14).color(s.text), text(format!("/{}", e.name)).size(12).color(s.faint)].spacing(8).align_y(Alignment::Center);
            let about = container(text(e.about.clone()).size(12).color(s.faint).wrapping(text::Wrapping::None)).width(Fill).clip(true);
            let line = row![container(self.slash_icon(&e.name, 14.0, s.muted)).center_x(18), column![name, about].spacing(2).width(Fill)].spacing(10).align_y(Alignment::Center);
            let on = i == pick;
            let style = move |_: &Theme, st| button::Style { background: Some(Background::Color(if on || hovered(st) { s.hover } else { Color::TRANSPARENT })), text_color: s.text, border: border::rounded(8), ..Default::default() };
            lines = lines.push(button(line).on_press(Message::SlashPick(Some(e.name.clone()))).width(Fill).padding([6, 10]).style(style));
        }
        let card = container(lines).padding(6).width(at.width).style(move |_| container::Style {
            background: Some(Background::Color(s.menu)),
            border: Border { radius: 12.0.into(), width: 1.0, color: s.line2 },
            ..Default::default()
        });
        let placed = container(opaque(card)).padding(Padding { left: at.x, bottom: (self.window.height - at.y + 6.0).max(0.0), ..Padding::ZERO }).width(Fill).height(Fill).align_y(Alignment::End);
        stack![body, placed].into()
    }

    /// A command's or skill's icon: a wand for a skill, a flag for /goal, the repeat arrows for /loop.
    fn slash_icon(&self, name: &str, size: f32, color: Color) -> Element<'static, Message> {
        match name {
            "goal" => icon(ic::FLAG, size).color(color).into(),
            "loop" => icon(ic::REPEAT, size).color(color).into(),
            _ => canvas(Wand { color }).width(size).height(size).into(),
        }
    }

    /// The command or skill picked with /, before the text: its icon and name, as Claude shows one.
    fn chip(&self, name: &str) -> Element<'static, Message> {
        let s = self.skin();
        let label = row![self.slash_icon(name, 12.0, s.accent), text(crate::slash_label(name)).size(14).font(MEDIUM).color(s.accent)].spacing(6).align_y(Alignment::Center);
        container(label)
            .padding([3, 8])
            .style(move |_| container::Style { background: Some(Background::Color(Color { a: 0.14, ..s.accent })), border: border::rounded(7), ..Default::default() })
            .into()
    }

    /// The window's resize edges, invisible strips over everything (the window has no Windows frame to
    /// grab); a press hands the resize to Windows. None when maximized.
    fn framed<'a>(&'a self, body: Element<'a, Message>) -> Element<'a, Message> {
        use iced::mouse::Interaction as I;
        use iced::window::Direction as D;
        if self.maximized {
            return body;
        }
        const E: Length = Length::Fixed(5.0);
        let edge = |w: Length, h: Length, d: D, i: I| -> Element<'a, Message> { mouse_area(Space::new().width(w).height(h)).on_press(Message::ResizeWindow(d)).interaction(i).into() };
        let top = row![edge(E, E, D::NorthWest, I::ResizingDiagonallyDown), edge(Fill, E, D::North, I::ResizingVertically), edge(E, E, D::NorthEast, I::ResizingDiagonallyUp)];
        let middle = row![edge(E, Fill, D::West, I::ResizingHorizontally), Space::new().width(Fill).height(Fill), edge(E, Fill, D::East, I::ResizingHorizontally)];
        let bottom = row![edge(E, E, D::SouthWest, I::ResizingDiagonallyUp), edge(Fill, E, D::South, I::ResizingVertically), edge(E, E, D::SouthEast, I::ResizingDiagonallyDown)];
        stack![body, column![top, middle, bottom]].into()
    }

    /// The session rail on the right: this computer and model, how full the context is, and the files this
    /// chat changed (with undo), the model's task list, and what it remembers across chats.
    fn rail(&self) -> Element<'_, Message> {
        let s = self.skin();
        let open = |k: &str| self.rail_open.iter().any(|o| o == k);
        // a row as Claude's rail has them: an icon, a label that clips, a value at the right
        let line = |c: char, label: String, right: String| -> Element<'static, Message> {
            row![icon(c, 13).color(s.faint).width(18), container(text(label).size(13).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true), text(right).size(12.5).color(s.faint)]
                .spacing(8)
                .align_y(Alignment::Center)
                .padding([5, 0])
                .into()
        };
        let link = |label: &'static str, msg: Message| -> Element<'static, Message> {
            button(text(label).size(12.5).color(s.accent)).on_press(msg).padding(Padding { top: 4.0, right: 10.0, ..Padding::ZERO }).style(move |_, _| button::Style { background: None, ..Default::default() }).into()
        };
        // a small outlined button, as Claude's "Create PR"
        let action = |label: &'static str, msg: Option<Message>| -> Element<'static, Message> {
            let on = msg.is_some();
            button(text(label).size(12.5).color(if on { s.text } else { s.faint }))
                .on_press_maybe(msg)
                .padding([4, 10])
                .style(move |_, st| button::Style {
                    background: Some(Background::Color(if on && hovered(st) { s.hover } else { Color::TRANSPARENT })),
                    border: Border { radius: 8.0.into(), width: 1.0, color: s.line2 },
                    ..Default::default()
                })
                .into()
        };
        let remove = |msg: Message| -> Element<'static, Message> { button(icon(ic::CANCEL, 9).color(s.faint)).on_press(msg).padding([4, 5]).style(move |_, st| bare(s, st)).into() };
        // a list inside a card that folds: a chevron, its name, how many
        let fold = |key: &'static str, label: &'static str, count: String| -> Element<'static, Message> {
            let chevron = if open(key) { ic::DOWN } else { ic::RIGHT };
            button(row![icon(chevron, 8).color(s.faint).width(18), text(label).size(13).color(s.muted).width(Fill), text(count).size(12.5).color(s.faint)].spacing(8).align_y(Alignment::Center))
                .on_press(Message::ToggleCard(key))
                .width(Fill)
                .padding([5, 0])
                .style(move |_, _| button::Style { background: None, ..Default::default() })
                .into()
        };
        // a card: its title and a chevron (a click folds it), what it holds at the right while folded, its body
        let card = |key: &'static str, title: &'static str, folded: Element<'static, Message>, body: Element<'static, Message>| -> Element<'static, Message> {
            let on = open(key);
            let mut head = row![text(title).size(13).font(MEDIUM).color(s.text), icon(if on { ic::DOWN } else { ic::RIGHT }, 8).color(s.faint), Space::new().width(Fill)].spacing(8).align_y(Alignment::Center);
            if !on {
                head = head.push(folded);
            }
            let head = button(head).on_press(Message::ToggleCard(key)).width(Fill).padding(0).style(move |_, _| button::Style { background: None, ..Default::default() });
            let mut col = column![head].spacing(10);
            if on {
                col = col.push(body);
            }
            container(col)
                .width(Fill)
                .padding([12, 14])
                .style(move |_| container::Style { background: Some(Background::Color(s.bg)), border: Border { radius: 12.0.into(), width: 1.0, color: s.line2 }, ..Default::default() })
                .into()
        };
        let faint = |t: String| -> Element<'static, Message> { text(t).size(12.5).color(s.faint).into() };
        // a name and, fainter, what goes with it, on one line that clips
        let pair = |name: String, more: String, c: char| -> Element<'static, Message> {
            let words: [text::Span<'static, (), Font>; 2] = [iced::widget::span(name).color(s.text), iced::widget::span(more).color(s.faint)];
            row![icon(c, 12).color(s.faint).width(18), container(rich_text(words).size(13).wrapping(text::Wrapping::None)).width(Fill).clip(true)].spacing(8).align_y(Alignment::Center).into()
        };

        // environment: the GPU, the memory, the model, and the folder the chat works in
        let sys = &self.system;
        let f = |j: &Json, k: &str| j.get(k).and_then(Json::as_f64).unwrap_or(0.0);
        let gpu = sys.get("gpus").and_then(Json::as_array).and_then(|g| g.first()).cloned().unwrap_or(Json::Null);
        let mut env = column![];
        if let Some(name) = gpu.get("name").and_then(Json::as_str) {
            let name = name.trim_start_matches("NVIDIA ").trim_start_matches("GeForce ").to_string();
            env = env.push(line(ic::PC, name, format!("{:.1} / {:.0} GB", f(&gpu, "vram_used_gb"), f(&gpu, "vram_total_gb"))));
        }
        if f(sys, "ram_total_gb") > 0.0 {
            env = env.push(line(ic::CHIP, "Memory".into(), format!("{:.1} / {:.0} GB", f(sys, "ram_total_gb") - f(sys, "ram_available_gb"), f(sys, "ram_total_gb"))));
        }
        let model = self.model.as_ref().map(|m| m.label.clone()).unwrap_or_else(|| "No model".into());
        let runs = sys.get("model_runs_on").and_then(Json::as_str).map(|b| if b.starts_with("GPU") { "GPU" } else { "CPU" }).unwrap_or("");
        env = env.push(line(ic::BOLT, model.clone(), runs.into()));
        let dir = if self.messages.is_empty() { self.new_chat_dir() } else { self.chat_cwd.as_deref().or(self.folders.first().map(String::as_str)) };
        env = env.push(match dir {
            Some(d) => {
                let place = std::path::Path::new(d).parent().map(|p| format!(" · {}", p.display())).unwrap_or_default();
                container(pair(crate::folder_name(d), place, ic::FOLDER)).padding([5, 0]).into()
            }
            None => line(ic::FOLDER, "No folder".into(), String::new()),
        });
        // the folders every chat may use besides its own
        env = env.push(fold("environment.folders", "Other folders", self.folders.len().to_string()));
        if open("environment.folders") {
            for (i, d) in self.folders.iter().enumerate() {
                env = env.push(row![Space::new().width(18), container(text(d.clone()).size(12.5).color(s.muted).wrapping(text::Wrapping::None)).width(Fill).clip(true), remove(Message::RemoveFolder(i))].spacing(8).align_y(Alignment::Center));
            }
            env = env.push(row![Space::new().width(18), link("Add folder", Message::AddFolder)].spacing(8));
        }
        let environment = card("environment", "Environment", faint(model), env.into());

        // context: how full it is, and what fills it, as Claude's rail breaks it down
        let (used, total) = self.context;
        let pct = if total > 0.0 { (100.0 * used / total).clamp(0.0, 100.0) } else { 0.0 };
        let part = |k: &str| self.parts.get(k).and_then(Json::as_f64).unwrap_or(0.0);
        let counted = ["system", "tools", "skills", "instructions", "memory"].iter().map(|k| part(k)).sum::<f64>();
        let grey = |a: f32| Color { a, ..s.muted };
        let parts: Vec<(&str, f64, Color)> = if self.parts != Json::Null && used > 0.0 {
            vec![
                ("Messages", (used - counted).max(0.0), PART_BLUE),
                ("Tools", part("tools"), PART_ORANGE),
                ("Skills", part("skills"), PART_GREEN),
                ("System prompt", part("system"), grey(0.85)),
                ("Instructions", part("instructions"), grey(0.6)),
                ("Memory", part("memory"), grey(0.4)),
            ]
        } else {
            vec![("Used", used, PART_BLUE)]
        };
        let free = (total - used).max(0.0);
        let portion = |n: f64| if total > 0.0 { ((n / total) * 1000.0).round().max(1.0) as u16 } else { 1 };
        let mut bar = row![].spacing(2).height(6);
        for (_, n, c) in parts.iter().filter(|p| p.1 > 0.0) {
            let c = *c;
            bar = bar.push(container(Space::new()).width(Length::FillPortion(portion(*n))).height(6).style(move |_| container::Style { background: Some(Background::Color(c)), border: border::rounded(2), ..Default::default() }));
        }
        bar = bar.push(container(Space::new()).width(Length::FillPortion(portion(free))).height(6).style(move |_| container::Style { background: Some(Background::Color(s.raise2)), border: border::rounded(2), ..Default::default() }));
        let share = |n: f64| if total > 0.0 { format!("{:.1}%", 100.0 * n / total) } else { String::new() };
        let entry = |label: &str, n: f64, c: Color| -> Element<'static, Message> {
            row![
                container(Space::new().width(9).height(9)).style(move |_| container::Style { background: Some(Background::Color(c)), border: border::rounded(2), ..Default::default() }),
                text(label.to_string()).size(13).color(s.muted).width(Fill),
                text(tokens_k(n)).size(12.5).color(s.faint).width(52).align_x(Alignment::End),
                text(share(n)).size(12.5).color(s.text).width(48).align_x(Alignment::End),
            ]
            .spacing(10)
            .align_y(Alignment::Center)
            .padding([3, 0])
            .into()
        };
        let mut ctx = column![].spacing(8);
        if total == 0.0 {
            ctx = ctx.push(faint("No model".into()));
        } else {
            ctx = ctx.push(row![text(format!("{pct:.0}% used")).size(13).font(MEDIUM).color(s.text), Space::new().width(Fill), text(format!("{} / {}", tokens_k(used), tokens_k(total))).size(12.5).color(s.faint)].align_y(Alignment::Center));
            ctx = ctx.push(bar);
            let mut rows = column![];
            for (label, n, c) in parts.iter().filter(|_| used > 0.0) {
                rows = rows.push(entry(label, *n, *c));
            }
            rows = rows.push(entry("Free space", free, s.raise2));
            ctx = ctx.push(rows);
        }
        // the GHOST.md files and the remembered facts, as Claude's memory files
        let in_use = self.instructions.iter().filter(|f| f.1).count();
        let mut lists = column![fold("context.instructions", "Instructions", in_use.to_string())];
        if open("context.instructions") {
            for (path, exists, global) in &self.instructions {
                let file = path.rsplit(['\\', '/']).next().unwrap_or(path).to_string();
                let note = if *global { "every chat".to_string() } else { path.rsplit(['\\', '/']).nth(1).unwrap_or("").to_string() };
                let words: [text::Span<'static, (), Font>; 2] = [iced::widget::span(file).color(if *exists { s.text } else { s.faint }), iced::widget::span(format!(" · {note}")).color(s.faint)];
                lists = lists.push(row![Space::new().width(18), container(rich_text(words).size(12.5).wrapping(text::Wrapping::None)).width(Fill).clip(true), link(if *exists { "Edit" } else { "Add" }, Message::EditInstructions(path.clone()))].spacing(8).align_y(Alignment::Center));
            }
        }
        lists = lists.push(fold("context.memory", "Memory", self.memory.len().to_string()));
        if open("context.memory") {
            for (i, fact) in self.memory.iter().enumerate() {
                lists = lists.push(row![Space::new().width(18), text(fact.clone()).size(12.5).color(s.muted).width(Fill), remove(Message::Forget(i))].spacing(8).align_y(Alignment::Center));
            }
            let add_memory = text_input("Add a memory", &self.memory_draft).on_input(Message::MemoryDraft).on_submit(Message::AddMemory).size(12.5).padding([5, 8]).style(move |t, st| input_style(s, t, st));
            lists = lists.push(row![Space::new().width(18), add_memory].spacing(8).padding(Padding { top: 4.0, ..Padding::ZERO }));
        }
        ctx = ctx.push(lists);
        // when old output starts being taken out (the server keeps the reply's length free, at most a quarter)
        if total > 0.0 && self.chat.is_some() {
            let room = (total - 4096f64.min(total / 4.0) - used).max(0.0);
            let can_trim = self.busy.is_none().then_some(Message::TrimChat);
            let until = container(text(format!("{} until old output is trimmed", tokens_k(room))).size(12).color(s.faint).wrapping(text::Wrapping::None)).width(Fill).clip(true);
            ctx = ctx.push(row![until, action("Trim now", can_trim)].spacing(8).align_y(Alignment::Center));
        }
        let filled = (pct.round() as u16).clamp(1, 99);
        let mini = container(row![
            container(Space::new()).width(Length::FillPortion(filled)).height(4).style(move |_| container::Style { background: Some(Background::Color(PART_BLUE)), border: border::rounded(99), ..Default::default() }),
            Space::new().width(Length::FillPortion(100 - filled)),
        ])
        .width(26)
        .style(move |_| container::Style { background: Some(Background::Color(s.raise2)), border: border::rounded(99), ..Default::default() });
        let folded: Element<'static, Message> = if total > 0.0 { row![faint(format!("{pct:.0}%")), mini].spacing(8).align_y(Alignment::Center).into() } else { Space::new().into() };
        let context = card("context", "Context", folded, ctx.into());

        // changes: the files this chat changed, their lines, and undo for all of them
        let changes = self.changes();
        let (adds, dels) = changes.iter().fold((0, 0), |(a, d), c| (a + c.1, d + c.2));
        let counts = |a: usize, d: usize| -> Element<'static, Message> { row![text(format!("+{a}")).size(12.5).font(MONO).color(s.ok), text(format!("−{d}")).size(12.5).font(MONO).color(s.bad)].spacing(6).into() };
        let mut ch = column![];
        if changes.is_empty() {
            ch = ch.push(faint("No changes".into()));
        }
        for (path, add, del) in &changes {
            let name = path.rsplit(['\\', '/']).next().unwrap_or(path).to_string();
            ch = ch.push(row![icon(ic::FILE, 12).color(s.faint).width(18), container(text(name).size(13).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true), counts(*add, *del)].spacing(8).align_y(Alignment::Center).padding([4, 0]));
        }
        if !changes.is_empty() {
            ch = ch.push(row![icon(ic::EDIT, 12).color(s.faint).width(18), counts(adds, dels), Space::new().width(Fill), action("Undo all", Some(Message::UndoAll))].spacing(8).align_y(Alignment::Center).padding(Padding { top: 6.0, ..Padding::ZERO }));
        }
        let changed: Element<'static, Message> = if changes.is_empty() { Space::new().into() } else { counts(adds, dels) };
        let changes = card("changes", "Changes", changed, ch.into());

        // the model's task list, while it keeps one
        let done = self.tasks.iter().filter(|t| t.1 == "done").count();
        let tasks = column(self.tasks.iter().map(|(task, status)| {
            let (mark, mc, tc) = match status.as_str() {
                "done" => (ic::CHECK, s.ok, s.faint),
                "in_progress" => (ic::NOW, s.accent, s.text),
                _ => (ic::TODO, s.faint, s.muted),
            };
            row![icon(mark, 10).color(mc).width(18), text(task.clone()).size(13).color(tc).width(Fill)].spacing(8).align_y(Alignment::Center).padding([3, 0]).into()
        }));
        let tasks = card("tasks", "Tasks", faint(format!("{done} of {}", self.tasks.len())), tasks.into());

        // background tasks, as Claude's: the chat's commands running now, each with its time and a stop button,
        // and how many finished; a click opens the panel with all of them
        let list = self.bg_tasks();
        let mut bt = column![];
        for t in list.iter().filter(|t| t.state == TaskState::Running) {
            let title = button(container(text(t.title.clone()).size(13).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true)).on_press(Message::ShowTasks).width(Fill).padding(0).style(move |_, _| button::Style { background: None, ..Default::default() });
            let mut r = row![icon(ic::TERMINAL, 12).color(s.faint).width(18), title].spacing(8).align_y(Alignment::Center).padding([2, 0]);
            if let Some(secs) = t.secs {
                r = r.push(text(duration(secs)).size(12.5).color(s.faint));
            }
            if let Some(id) = t.process {
                r = r.push(stop_button(s, Message::StopProcess(id)));
            }
            bt = bt.push(r);
        }
        let finished = list.iter().filter(|t| t.state != TaskState::Running).count();
        if finished > 0 {
            bt = bt.push(button(row![icon(ic::TASKS, 12).color(s.faint).width(18), text(format!("{finished} finished")).size(13).color(s.text)].spacing(8).align_y(Alignment::Center)).on_press(Message::ShowTasks).width(Fill).padding([2, 0]).style(move |_, _| button::Style { background: None, ..Default::default() }));
        }
        let running_now = list.len() - finished;
        let tasks_note = if running_now > 0 { format!("{running_now} running") } else { format!("{finished} finished") };
        let background = (!list.is_empty()).then(|| card("background", "Background tasks", faint(tasks_note), bt.into()));

        // scheduled: the tasks the model set and the /loop ones, and the one running now
        let jobs = self.background.get("jobs").and_then(Json::as_array).unwrap_or(&[]);
        let running = self.background.get("running").and_then(Json::as_str).is_some();
        let mut sc = column![];
        if running {
            sc = sc.push(line(ic::NOW, "Running one now".into(), String::new()));
        }
        for j in jobs {
            let g = |k: &str| j.get(k).and_then(Json::as_str).unwrap_or("").to_string();
            sc = sc.push(row![pair(g("name"), format!(" · {}, next {}", g("when"), g("next")), ic::CLOCK), link("Run", Message::RunScheduled(g("id"))), remove(Message::CancelScheduled(g("id")))].spacing(4).align_y(Alignment::Center).padding([3, 0]));
        }
        let due = match (running, jobs.len()) {
            (true, _) => "running".to_string(),
            (false, 0) => String::new(),
            (false, n) => n.to_string(),
        };
        let scheduled = (running || !jobs.is_empty()).then(|| card("scheduled", "Scheduled", faint(due), sc.into()));

        // skills: installed ones, and installing
        let mut sk = column![];
        for (name, desc) in &self.skills {
            sk = sk.push(row![pair(name.clone(), format!(" · {desc}"), ic::SKILL), remove(Message::RemoveSkill(name.clone()))].spacing(8).align_y(Alignment::Center).padding([2, 0]));
        }
        sk = sk.push(row![link("Install from folder", Message::InstallSkill(false)), link("From .zip", Message::InstallSkill(true))]);
        let skill_link = text_input("Skill link (GitHub or .zip)", &self.skill_link).id("skill-link").on_input(Message::SkillLink).on_submit(Message::InstallLink).size(12.5).padding([5, 8]).style(move |t, st| input_style(s, t, st));
        sk = sk.push(container(skill_link).padding(Padding { top: 4.0, ..Padding::ZERO }));
        // a link or folder with several skills: which to install
        if let Some((_, choices)) = &self.skill_choices {
            sk = sk.push(container(faint(format!("It has {} skills. Install which?", choices.len()))).padding(Padding { top: 6.0, ..Padding::ZERO }));
            for (path, name, desc) in choices {
                let place = if path.is_empty() { desc.clone() } else { format!("{path} · {desc}") };
                sk = sk.push(row![pair(name.clone(), format!(" · {place}"), ic::SKILL), link("Install", Message::InstallChoice(Some(path.clone())))].spacing(8).align_y(Alignment::Center));
            }
            sk = sk.push(row![link("Install all", Message::InstallChoice(None)), link("Cancel", Message::CancelChoice)]);
        }
        let skills = card("skills", "Skills", faint(if self.skills.is_empty() { String::new() } else { self.skills.len().to_string() }), sk.into());

        // MCP servers: each with its state, and editing the list
        let mut mc = column![];
        for (name, state, tools) in &self.mcp {
            let (dot, note) = match state.as_str() {
                "ready" => (s.ok, format!("{tools} tools")),
                "starting" => (s.warn, "starting".to_string()),
                other => (s.bad, crate::clip_line(other, 40)),
            };
            mc = mc.push(
                row![
                    container(container(Space::new().width(7).height(7)).style(move |_| container::Style { background: Some(Background::Color(dot)), border: border::rounded(99), ..Default::default() })).width(18).align_x(Alignment::Center),
                    container(text(name.clone()).size(13).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true),
                    text(note).size(12.5).color(s.faint),
                ]
                .spacing(8)
                .align_y(Alignment::Center)
                .padding([4, 0]),
            );
        }
        mc = mc.push(row![link("Edit servers", Message::McpEdit), link("Reload", Message::McpReload), link("From Claude", Message::McpImport)]);
        let ready = self.mcp.iter().filter(|m| m.1 == "ready").count();
        let broken = self.mcp.iter().filter(|m| m.1 != "ready" && m.1 != "starting").count();
        let mcp_note = match (ready, broken) {
            (_, b) if b > 0 => format!("{b} not working"),
            (0, _) => String::new(),
            (r, _) => format!("{r} ready"),
        };
        let mcp = card("mcp", "MCP servers", faint(mcp_note), mc.into());

        let mut body = column![environment, context, changes].spacing(10);
        if !self.tasks.is_empty() {
            body = body.push(tasks);
        }
        for c in [background, scheduled].into_iter().flatten() {
            body = body.push(c);
        }
        let body = body.push(skills).push(mcp);
        let top = row![drag_area(container(Space::new()).width(Fill).height(32)), self.window_controls()];
        container(column![top, scrollable(container(body).padding(Padding { top: 0.0, right: 14.0, bottom: 14.0, left: 14.0 })).id("rail").height(Fill).style(move |t, st| scroll_style(s, t, st))])
            .width(300)
            .height(Fill)
            .style(move |_| container::Style { background: Some(Background::Color(s.side)), border: Border { width: 0.0, ..Default::default() }, ..Default::default() })
            .into()
    }

    /// The background tasks panel beside the rail, as Claude's: the running commands, then the finished ones
    /// (newest first), each unfolding to its command and what it printed.
    fn tasks_panel(&self) -> Element<'_, Message> {
        let s = self.skin();
        let list = self.bg_tasks();
        let tool = |c: char, msg: Message| -> Element<'static, Message> { button(icon(c, 12).color(s.muted)).on_press(msg).padding(7).style(move |_, st| bare(s, st)).into() };
        let finished: Vec<String> = list.iter().filter(|t| t.state != TaskState::Running).map(|t| t.key.clone()).collect();
        let head = row![
            text("Background tasks").size(14).font(MEDIUM).color(s.text),
            Space::new().width(Fill),
            tool(ic::DELETE, Message::ClearFinished(finished.clone())),
            tool(if self.tasks_wide { ic::RESTORE } else { ic::MAXIMIZE }, Message::WideTasks),
            tool(ic::CANCEL, Message::CloseTasks),
        ]
        .spacing(2)
        .align_y(Alignment::Center)
        .padding(Padding { top: 8.0, right: 8.0, bottom: 8.0, left: 16.0 });
        let item = |t: &crate::BgTask| -> Element<'static, Message> {
            let open = t.state == TaskState::Running || self.task_open.contains(&t.key);
            let (right, color) = match (t.state, t.secs) {
                (TaskState::Failed, _) => ("Failed".to_string(), s.bad),
                (TaskState::Stopped, Some(n)) => (format!("Stopped · {}", duration(n)), s.faint),
                (TaskState::Stopped, None) => ("Stopped".to_string(), s.faint),
                (_, Some(n)) => (duration(n), s.faint),
                (_, None) => (String::new(), s.faint),
            };
            let mut top = row![
                icon(if open { ic::DOWN } else { ic::RIGHT }, 8).color(s.faint).width(14),
                container(text(t.title.clone()).size(13).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true),
                text(right).size(12.5).color(color),
            ]
            .spacing(8)
            .align_y(Alignment::Center);
            if let (TaskState::Running, Some(id)) = (t.state, t.process) {
                top = top.push(stop_button(s, Message::StopProcess(id)));
            }
            let top = button(top).on_press(Message::ToggleTask(t.key.clone())).width(Fill).padding(0).style(move |_, _| button::Style { background: None, ..Default::default() });
            let mut body = column![top].spacing(8);
            if open {
                let shown = if t.output.is_empty() { format!("$ {}", t.command) } else { format!("$ {}\n{}", t.command, t.output) };
                let out = scrollable(container(text(shown).size(12).font(MONO).color(s.muted)).width(Fill).padding(Padding { right: 10.0, ..Padding::ZERO })).height(Length::Shrink).style(move |t, st| scroll_style(s, t, st));
                body = body.push(container(out).max_height(260).padding(Padding { left: 22.0, ..Padding::ZERO }));
            }
            container(body).width(Fill).padding([10, 12]).style(move |_| container::Style { background: Some(Background::Color(s.raise)), border: border::rounded(10), ..Default::default() }).into()
        };
        let mut col = column![].spacing(6);
        for t in list.iter().filter(|t| t.state == TaskState::Running) {
            col = col.push(item(t));
        }
        if !finished.is_empty() {
            let group = button(row![icon(if self.finished_open { ic::DOWN } else { ic::RIGHT }, 8).color(s.faint).width(14), text("Finished").size(12.5).color(s.faint), text(finished.len().to_string()).size(12.5).color(s.faint)].spacing(6).align_y(Alignment::Center))
                .on_press(Message::ToggleFinished)
                .padding(Padding { top: 10.0, right: 0.0, bottom: 4.0, left: 2.0 })
                .style(move |_, _| button::Style { background: None, ..Default::default() });
            col = col.push(group);
            if self.finished_open {
                for t in list.iter().filter(|t| t.state != TaskState::Running).take(self.tasks_shown) {
                    col = col.push(item(t));
                }
                if finished.len() > self.tasks_shown {
                    col = col.push(button(text("Show more").size(12.5).color(s.muted)).on_press(Message::MoreTasks).padding([6, 4]).style(move |_, st| bare(s, st)));
                }
            }
        }
        if list.is_empty() {
            col = col.push(text("None").size(12.5).color(s.faint));
        }
        let body = scrollable(container(col).padding(Padding { top: 0.0, right: 12.0, bottom: 12.0, left: 12.0 })).height(Fill).style(move |t, st| scroll_style(s, t, st));
        container(column![head, body])
            .width(if self.tasks_wide { 600 } else { 400 })
            .height(Fill)
            .style(move |_| container::Style { background: Some(Background::Color(s.side)), border: Border { width: 1.0, color: s.line, radius: 0.0.into() }, ..Default::default() })
            .into()
    }

    /// Minimise, maximise (or restore) and close, drawn like Windows' own: 46 by 32, close turning red.
    fn window_controls(&self) -> Element<'_, Message> {
        let s = self.skin();
        let b = |c: char, msg: Message, close: bool| {
            button(container(icon(c, 10)).width(46).height(32).align_x(Alignment::Center).align_y(Alignment::Center))
                .on_press(msg)
                .padding(0)
                .style(move |_, st| {
                    let (bg, fg) = match (hovered(st), close) {
                        (true, true) => (rgb(0xc4, 0x2b, 0x1c), Color::WHITE),
                        (true, false) => (s.hover, s.text),
                        _ => (Color::TRANSPARENT, s.muted),
                    };
                    button::Style { background: Some(Background::Color(bg)), text_color: fg, ..Default::default() }
                })
        };
        row![
            b(ic::MINIMIZE, Message::Minimize, false),
            b(if self.maximized { ic::RESTORE } else { ic::MAXIMIZE }, Message::ToggleMaximize, false),
            b(ic::CLOSE, Message::CloseWindow, true)
        ]
        .into()
    }

    fn sidebar(&self) -> Element<'_, Message> {
        let s = self.skin();
        // the title bar's left part: drag the window by it, double-click to maximize
        let chrome = drag_area(container(icon_button(s, ic::CLOSE_PANE, Message::ToggleSidebar)).width(Fill).height(44).align_y(Alignment::Center));
        let nav = |c: char, label: &'static str, msg: Message| {
            button(row![icon(c, 14).color(s.muted), text(label).size(14).color(s.text)].spacing(12).align_y(Alignment::Center))
                .on_press(msg)
                .width(Fill)
                .padding([7, 10])
                .style(move |_, st| row_style(s, st, false))
        };
        let mut col = column![chrome, nav(ic::ADD, "New chat", Message::NewChat), nav(ic::SEARCH, "Search", Message::ToggleSearch)].spacing(1);
        if self.searching {
            col = col.push(
                text_input("Search chats", &self.filter)
                    .on_input(Message::Filter)
                    .size(13.5)
                    .padding([7, 10])
                    .style(move |t, st| input_style(s, t, st)),
            );
        }
        let recents_head = button(row![text("Recents").size(12).color(s.faint), icon(if self.recents_open { ic::DOWN } else { ic::RIGHT }, 8).color(s.faint)].spacing(6).align_y(Alignment::Center))
            .on_press(Message::ToggleRecents)
            .padding(Padding { top: 14.0, right: 10.0, bottom: 4.0, left: 10.0 })
            .style(move |_, _| button::Style { background: None, ..Default::default() });
        col = col.push(recents_head);
        let filter = self.filter.to_lowercase();
        let rows = column(self.chats.iter().filter(|c| filter.is_empty() || c.title.to_lowercase().contains(&filter)).map(|c| {
            let on = self.chat.as_deref() == Some(c.id.as_str());
            let hover = self.hovered.as_deref() == Some(c.id.as_str());
            let name = container(text(if c.title.is_empty() { "New chat" } else { c.title.as_str() }).size(14).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true);
            // a long title fades out at the right instead of being cut (the fade is the row's own colour)
            let bg = if on { s.raise } else if hover { s.hover } else { s.side };
            let fade = iced::gradient::Linear::new(iced::Radians(std::f32::consts::FRAC_PI_2)).add_stop(0.0, Color { a: 0.0, ..bg }).add_stop(1.0, bg);
            let veil = container(Space::new().width(32).height(Fill)).style(move |_| container::Style { background: Some(Background::Gradient(fade.into())), ..Default::default() });
            // only a title that runs past the row fades (iced draws the gradient a shade off the row's colour,
            // which would show as a block on a short one)
            let label: Element<'_, Message> = if c.title.chars().count() > 26 { stack![name, container(veil).width(Fill).height(Fill).align_x(Alignment::End)].into() } else { name.into() };
            let mut line = row![label].align_y(Alignment::Center).spacing(6);
            // a scheduled run wrote here since it was last opened
            if c.unread && !on {
                line = line.push(container(Space::new().width(7).height(7)).style(move |_| container::Style { background: Some(Background::Color(s.accent)), border: border::rounded(99), ..Default::default() }));
            }
            if hover || on {
                line = line.push(button(icon(ic::DELETE, 12).color(s.faint)).on_press(Message::DeleteChat(c.id.clone())).padding([3, 5]).style(move |_, st| bare(s, st)));
            }
            let b = button(line).on_press(Message::OpenChat(c.id.clone())).width(Fill).padding([7, 10]).style(move |_, _| button::Style { background: Some(Background::Color(bg)), text_color: s.text, border: border::rounded(8), ..Default::default() }); // the same colour as the fade
            mouse_area(b).on_enter(Message::Hover(Some(c.id.clone()))).on_exit(Message::Hover(None)).into()
        }))
        .spacing(1);
        let list: Element<'_, Message> = if self.recents_open { scrollable(rows).height(Fill).style(move |t, st| scroll_style(s, t, st)).into() } else { Space::new().height(Fill).into() };
        col = col.push(list);
        // you, at the bottom: this computer, the model, settings
        let model = self.model.as_ref().map(|m| m.label.clone()).unwrap_or_else(|| "No model".into());
        let me = button(
            row![
                canvas(Ghost::still(1.0, s.accent)).width(24).height(22),
                column![text("This PC").size(13.5).color(s.text), text(model).size(12).color(s.faint).wrapping(text::Wrapping::None)].width(Fill),
                icon(ic::SETTINGS, 13).color(s.muted),
            ]
            .spacing(10)
            .align_y(Alignment::Center),
        )
        .on_press(Message::Menu(Some(Menu::Settings)))
        .width(Fill)
        .padding([8, 8])
        .style(move |_, st| row_style(s, st, false));
        container(column![container(col).height(Fill).padding([4, 10]), container(me).padding(Padding { top: 8.0, right: 10.0, bottom: 10.0, left: 10.0 }).style(move |_| container::Style { border: Border { width: 0.0, ..Default::default() }, ..Default::default() })])
            .width(264)
            .height(Fill)
            .style(move |_| container::Style { background: Some(Background::Color(s.side)), ..Default::default() })
            .into()
    }

    fn main(&self) -> Element<'_, Message> {
        let s = self.skin();
        let mut top = row![].spacing(8).align_y(Alignment::Center).padding(Padding { top: 6.0, right: 14.0, bottom: 6.0, left: 14.0 }).height(44);
        if !self.sidebar {
            top = top.push(icon_button(s, ic::OPEN_PANE, Message::ToggleSidebar));
        }
        if let (Some(title), false) = (&self.title, self.messages.is_empty()) {
            // a long title clips rather than running under what is beside the chat
            let title = button(row![text(title.as_str()).size(14).font(MEDIUM).color(s.text).wrapping(text::Wrapping::None), icon(ic::DOWN, 9).color(s.muted)].spacing(8).align_y(Alignment::Center))
                .on_press(Message::Menu(Some(Menu::Chat)))
                .padding([5, 8])
                .style(move |_, st| row_style(s, st, false));
            top = top.push(container(row![title, Space::new().width(Fill)]).width(Fill).clip(true));
        } else {
            top = top.push(Space::new().width(Fill));
        }
        // the title bar: drag the window by its empty part, the window's buttons at its right
        // the rail's switch; with the rail open, the window's buttons move to its top
        // a page the agent wrote, with its pane closed: the button that shows it again
        if self.preview_url.is_some() && !self.preview_open {
            top = top.push(button(icon(ic::GLOBE, 13).color(s.muted)).on_press(Message::PreviewOpen).padding(8).style(move |_, st| row_style(s, st, false)));
        }
        let on = self.rail;
        top = top.push(button(icon(ic::DOCK_RIGHT, 13).color(if on { s.text } else { s.muted })).on_press(Message::ToggleRail).padding(8).style(move |_, st| row_style(s, st, on)));
        // the window's buttons at the window's top right: here, unless the preview pane or the rail is to the right
        let top: Element<'_, Message> = if self.rail || self.preview_shown() { drag_area(top) } else { row![drag_area(top), self.window_controls()].into() };

        if self.messages.is_empty() {
            // a new chat: the greeting, the ghost on the message box, recent chats; on the first run, with no
            // model yet, the models to download instead of the message box
            let hello = text(if self.api.is_some() { greeting() } else { "Connecting…" }).size(38).font(SERIF).color(s.text);
            let first_run = self.api.is_some() && self.models.is_empty() && self.model.is_none() && self.loading.is_none();
            let below = if first_run { self.setup_card() } else { self.dock_parts() };
            let mut center = column![Space::new().height(Fill), hello, Space::new().height(26), below].align_x(Alignment::Center).max_width(760).width(Fill);
            let recent: Vec<_> = self.chats.iter().take(3).collect();
            if !recent.is_empty() {
                let cards = row(recent.into_iter().map(|c| {
                    button(column![icon(ic::CHAT, 14).color(s.accent), text(c.title.as_str()).size(14).font(MEDIUM).color(s.text), Space::new().height(Fill), text(ago(c.updated)).size(12).color(s.faint)].spacing(8))
                        .on_press(Message::OpenChat(c.id.clone()))
                        .width(Fill)
                        .height(104)
                        .padding([12, 14])
                        .style(move |_, st| card_style(s, st))
                        .into()
                }))
                .spacing(12);
                center = center.push(Space::new().height(30)).push(row![text("Recent chats").size(13).color(s.muted)].padding([0, 2])).push(Space::new().height(10)).push(cards);
            }
            center = center.push(Space::new().height(Fill));
            return container(column![top, container(center).center_x(Fill).height(Fill).padding([0, 24])]).style(move |_| container::Style { background: Some(Background::Color(s.bg)), ..Default::default() }).into();
        }

        let first = self.messages.len().saturating_sub(self.shown);
        let mut items: Vec<Element<'_, Message>> = Vec::new();
        if first > 0 {
            items.push(container(button(text(format!("Show {} earlier messages", first.min(crate::SHOWN))).size(13).color(s.muted)).on_press(Message::ShowEarlier).padding([6, 14]).style(move |_, st| pill(s, st, false))).center_x(Fill).into());
        }
        for (i, m) in self.messages.iter().enumerate().skip(first) {
            items.push(self.message(i, m));
        }
        let transcript = scrollable(container(column(items).spacing(28).max_width(740)).center_x(Fill).padding(Padding { top: 18.0, right: 28.0, bottom: 28.0, left: 28.0 }))
            .anchor_bottom()
            .height(Fill)
            .style(move |t, st| scroll_style(s, t, st));
        let dock = container(column![self.dock_parts()].max_width(740)).center_x(Fill).padding(Padding { top: 0.0, right: 24.0, bottom: 12.0, left: 24.0 });
        container(column![top, transcript, dock]).style(move |_| container::Style { background: Some(Background::Color(s.bg)), ..Default::default() }).into()
    }

    /// The first run: the models to download, the download under way, and, for an installed app, a folder of
    /// models already on this computer. Which one is the user's choice: none is marked.
    fn setup_card(&self) -> Element<'_, Message> {
        let s = self.skin();
        let state = self.download.get("state").and_then(Json::as_str).unwrap_or("");
        let file = self.download.get("file").and_then(Json::as_str).unwrap_or("");
        let busy = state == "downloading";
        let mut rows = column![].spacing(8);
        for (i, o) in crate::setup::OFFERS.iter().enumerate() {
            let this = file == o.file;
            let name = text(o.name).size(15).font(MEDIUM).color(s.text);
            let mut left = column![name, text(o.note).size(13).color(s.muted)].spacing(4).width(Fill);
            let size = text(format!("{:.1} GB", o.size as f64 / 1e9)).size(13).color(s.faint);
            let action: Element<'_, Message> = if this && busy {
                button(text("Cancel").size(13).color(s.text)).on_press(Message::CancelDownload).padding([6, 14]).style(move |_, st| pill(s, st, false)).into()
            } else if busy {
                Space::new().into()
            } else {
                let label = if this && matches!(state, "failed" | "cancelled") { "Resume" } else { "Download" };
                button(row![icon(ic::DOWNLOAD, 12).color(s.text), text(label).size(13).color(s.text)].spacing(8).align_y(Alignment::Center))
                    .on_press(Message::GetModel(i))
                    .padding([6, 14])
                    .style(move |_, st| pill(s, st, false))
                    .into()
            };
            if this && busy {
                let (bytes, total) = (self.download.get("bytes").and_then(Json::as_f64).unwrap_or(0.0), o.size as f64);
                let done = ((bytes / total).clamp(0.0, 1.0) * 1000.0) as u16;
                let bar = |c: Color| move |_: &Theme| container::Style { background: Some(Background::Color(c)), border: border::rounded(3), ..Default::default() };
                let track = row![
                    container(Space::new()).width(Length::FillPortion(done.max(1))).height(6).style(bar(s.accent)),
                    container(Space::new()).width(Length::FillPortion((1000 - done).max(1))).height(6).style(bar(s.raise2)),
                ];
                let rate = self.download_rate.2;
                let mut note = crate::setup::progress_text(bytes, total);
                if rate > 1e5 {
                    let mins = ((total - bytes) / rate / 60.0).ceil();
                    note += &if mins <= 1.0 { " · about a minute left".to_string() } else { format!(" · {mins:.0} min left") };
                }
                left = left.push(Space::new().height(6)).push(track).push(text(note).size(12).color(s.muted));
            } else if this && state == "failed" {
                let why = self.download.get("error").and_then(Json::as_str).unwrap_or("The download stopped");
                left = left.push(text(why).size(12).color(s.bad));
            }
            let line = row![left, size, action].spacing(16).align_y(Alignment::Center);
            rows = rows.push(container(line).padding([14, 16]).width(Fill).style(move |_| container::Style {
                background: Some(Background::Color(s.hover)),
                border: Border { radius: 12.0.into(), width: if s.dark { 0.0 } else { 1.0 }, color: s.line },
                ..Default::default()
            }));
        }
        let mut card = column![row![text("Choose a model").size(13).color(s.muted)].padding([0, 2]), rows].spacing(10).max_width(640).width(Fill);
        if self.own_server && !busy {
            let folder = button(row![icon(ic::FOLDER, 13).color(s.muted), text("Use models already on this computer").size(13).color(s.muted)].spacing(8).align_y(Alignment::Center))
                .on_press(Message::UseModelFolder)
                .padding([6, 10])
                .style(move |_, st| bare(s, st));
            card = card.push(folder);
        }
        card.into()
    }

    fn preview_shown(&self) -> bool {
        self.preview_open && self.preview_url.is_some()
    }

    /// The preview pane beside the chat: the page the agent last wrote, under a bar with its name, reload, open
    /// in the browser and close. The page itself is a native view laid over the area below the bar (preview.rs).
    fn preview_pane(&self) -> Element<'_, Message> {
        let s = self.skin();
        let url = self.preview_url.as_deref().unwrap_or("");
        let name = url.rsplit('/').next().unwrap_or(url).replace("%20", " ");
        let bar = row![
            icon(ic::GLOBE, 13).color(s.muted),
            container(text(name).size(13).font(MEDIUM).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true),
            icon_button(s, ic::RETRY, Message::PreviewReload),
            icon_button(s, ic::OPEN_NEW, Message::PreviewExternal),
            icon_button(s, ic::CANCEL, Message::PreviewClose),
        ]
        .spacing(4)
        .align_y(Alignment::Center)
        .padding(Padding { top: 6.0, right: 8.0, bottom: 6.0, left: 16.0 })
        .height(44);
        let bar: Element<'_, Message> = if self.rail { drag_area(bar) } else { row![drag_area(bar), self.window_controls()].into() };
        let page = canvas(PageArea).width(Fill).height(Fill);
        container(column![bar, page])
            .width(Fill)
            .height(Fill)
            .style(move |_| container::Style { background: Some(Background::Color(s.bg)), border: Border { width: 1.0, color: s.line, ..Default::default() }, ..Default::default() })
            .into()
    }

    /// The message box's layout: whether it has grown past one line, and whether it holds a chip. One line,
    /// as Claude's box, until the text needs more (a new line, or more than fits on one line at about 7.6 px
    /// a character): then it grows, to about 10 lines, and scrolls beyond.
    pub fn composer_shape(&self) -> (bool, bool) {
        let main = self.window.width - if self.sidebar { 264.0 } else { 0.0 } - if self.rail { 300.0 } else { 0.0 } - if self.tasks_panel { if self.tasks_wide { 600.0 } else { 400.0 } } else { 0.0 };
        let chip_chars = self.chip.as_deref().map_or(0, |c| crate::slash_label(c).len() + 5);
        let fits = ((main.min(760.0) - 130.0) / 7.6).max(20.0) as usize - chip_chars.min(20);
        (self.composer.line_count() > 1 || self.composer.text().chars().count() > fits, self.chip.is_some())
    }

    /// The ghost perched on the message box, the box itself, and the pickers under it.
    fn dock_parts(&self) -> Element<'_, Message> {
        let s = self.skin();
        let ghost = canvas(Ghost { pixel: 3.0, mood: self.mood(), t: self.clock, talk: self.talking(), accent: s.accent }).width(72).height(66);
        let say = if self.status.is_empty() && self.messages.is_empty() && self.api.is_some() { "Hello".to_string() } else { self.status_text() };
        // the ghost's hem tucks behind the box: the perch overlaps the card by a few pixels
        let perch = row![Space::new().width(18), container(ghost).height(58).clip(true), container(text(say).size(13).color(s.muted)).padding(Padding { bottom: 22.0, left: 10.0, ..Padding::ZERO })]
            .align_y(Alignment::End)
            .height(58);
        let mode_menu = self.menu == Some(Menu::Mode);
        let grown = self.composer_shape().0;
        // the / list open: the arrows move in it, Tab or Enter picks, Esc closes; Backspace before any text
        // takes the chip away
        let slash_open = !self.slash_list().is_empty();
        let cursor = self.composer.cursor();
        let chip_back = self.chip.is_some() && cursor.position.line == 0 && cursor.position.column == 0 && cursor.selection.is_none();
        let editor = text_editor(&self.composer)
            .id("editor")
            .highlight_with::<SlashNames>((self.skill_names(), s.accent), |c, _| iced::advanced::text::highlighter::Format { color: Some(*c), font: Some(MEDIUM) })
            .placeholder(match (&self.chip, self.messages.is_empty()) {
                (Some(_), _) => "Say what to do with it",
                (None, true) => "How can I help you today?",
                (None, false) => "Reply to ghost-infer",
            })
            .on_action(Message::Edit)
            .padding(if grown { Padding { top: 4.0, right: 6.0, bottom: 10.0, left: 6.0 } } else { Padding::from([6, 6]) })
            .height(Length::Shrink)
            .min_height(22)
            .max_height(240)
            .size(15)
            .key_binding(move |kp| {
                use keyboard::key::Named;
                let custom = |m| Some(text_editor::Binding::Custom(m));
                if slash_open {
                    match kp.key {
                        keyboard::Key::Named(Named::ArrowUp) => return custom(Message::SlashMove(-1)),
                        keyboard::Key::Named(Named::ArrowDown) => return custom(Message::SlashMove(1)),
                        keyboard::Key::Named(Named::Enter | Named::Tab) if !kp.modifiers.shift() => return custom(Message::SlashPick(None)),
                        keyboard::Key::Named(Named::Escape) => return custom(Message::SlashClose),
                        _ => {}
                    }
                }
                if chip_back && matches!(kp.key, keyboard::Key::Named(Named::Backspace)) {
                    return custom(Message::ChipRemove);
                }
                // Enter sends; Shift+Enter starts a new line
                if matches!(kp.key, keyboard::Key::Named(keyboard::key::Named::Enter)) && !kp.modifiers.shift() {
                    return Some(text_editor::Binding::Custom(Message::Send));
                }
                // Shift+Tab goes to the next permission mode, as in Claude Code
                if matches!(kp.key, keyboard::Key::Named(keyboard::key::Named::Tab)) && kp.modifiers.shift() {
                    return Some(text_editor::Binding::Custom(Message::CycleMode));
                }
                // the mode menu open: its number keys pick a mode
                if let (true, keyboard::Key::Character(c)) = (mode_menu, &kp.key) {
                    if let Some(m) = crate::mode_key(c) {
                        return Some(text_editor::Binding::Custom(m));
                    }
                }
                text_editor::Binding::from_key_press(kp)
            })
            .style(move |t, st| editor_style(s, t, st));
        let plus = container(button(container(icon(ic::ADD, 13).color(s.muted)).center(32)).on_press(Message::OpenMenu(Menu::Plus)).padding(0).style(move |_, st| round_style(s, st))).id("plus");
        let busy = self.busy.is_some();
        let send = if busy {
            // stop: a small square, as Claude's (the button around it only shows on hover)
            let mark = container(Space::new()).width(12).height(12).style(move |_| container::Style { background: Some(Background::Color(s.text)), border: border::rounded(3), ..Default::default() });
            button(container(mark).center(32)).on_press(Message::Stop).padding(0).style(move |_, st| stop_style(s, st))
        } else {
            button(container(icon(ic::UP, 13).color(s.ink)).center(32)).on_press(Message::Send).padding(0).style(move |_, st| send_style(s, st))
        };
        let chip = self.chip.as_deref().map(|c| self.chip(c));
        let (inside, pad, radius): (Element<'_, Message>, Padding, f32) = if grown {
            let mut col = column![].spacing(4);
            if let Some(chip) = chip {
                col = col.push(row![chip]);
            }
            (col.push(editor).push(row![plus, Space::new().width(Fill), send].align_y(Alignment::Center)).into(), Padding { top: 12.0, right: 12.0, bottom: 10.0, left: 14.0 }, 18.0)
        } else {
            let mut line = row![plus].spacing(6).align_y(Alignment::Center);
            if let Some(chip) = chip {
                line = line.push(chip);
            }
            (line.push(editor).push(send).into(), Padding::from([6, 8]), 16.0)
        };
        let card = container(inside).id("composer").padding(pad).style(move |_| container::Style {
            background: Some(Background::Color(s.raise)),
            // no soft shadow: tiny-skia repaints only the ghost's patch each tick and leaves holes in it
            border: Border { radius: radius.into(), width: 1.0, color: s.line2 },
            ..Default::default()
        });
        let think = button(row![icon(ic::BOLT, 11).color(if self.thinking { s.accent } else { s.faint }), text(if self.thinking { "Think" } else { "Quick" }).size(13).color(s.muted)].spacing(6).align_y(Alignment::Center))
            .on_press(Message::ToggleThinking)
            .padding([4, 8])
            .style(move |_, st| row_style(s, st, false));
        // the model by name, as Claude shows it: a label that opens the model menu
        let name = match &self.model {
            Some(m) if self.warming => format!("{} · warming up", m.label),
            Some(m) => m.label.clone(),
            None => "Model".into(),
        };
        let model = container(button(text(name).size(13).color(s.text)).on_press(Message::OpenMenu(Menu::Model)).padding([4, 8]).style(move |_, st| row_style(s, st, false))).id("model");
        let mode = crate::MODES[self.perm].1;
        let mode = container(button(text(mode).size(13).color(s.text)).on_press(Message::OpenMenu(Menu::Mode)).padding([4, 8]).style(move |_, st| row_style(s, st, true))).id("mode");
        // where the chat works: picked for a new chat, then kept (as a Claude Code session keeps its folder)
        let folder = row![icon(ic::FOLDER, 12).color(s.muted), text(self.folder_label()).size(13).color(s.muted)].spacing(7).align_y(Alignment::Center);
        let folder: Element<'_, Message> = if self.messages.is_empty() {
            container(button(folder).on_press(Message::OpenMenu(Menu::Folder)).padding([4, 8]).style(move |_, st| row_style(s, st, false))).id("folder").into()
        } else {
            container(folder).padding([4, 8]).into()
        };
        let mut under = row![folder, Space::new().width(Fill), model, think, mode].spacing(2).align_y(Alignment::Center).padding(Padding { top: 8.0, right: 4.0, bottom: 0.0, left: 4.0 });
        // how full the context is, as Claude shows it: a ring, its numbers on hover, the side panel on a click
        let (used, total) = self.context;
        if total > 0.0 {
            let frac = (used / total).clamp(0.0, 1.0) as f32;
            let ring = canvas(Ring { frac, track: s.line2, fill: s.accent }).width(16).height(16);
            let ring = button(container(ring).center(24)).on_press(Message::ToggleRail).padding(2).style(move |_, st| row_style(s, st, false));
            let tip = format!("{:.0}% of context used: {}", 100.0 * frac, self.context_note());
            let tip = container(text(tip).size(12).color(s.text)).padding([5, 9]).style(move |_| container::Style {
                background: Some(Background::Color(s.raise)),
                border: Border { radius: 8.0.into(), width: 1.0, color: s.line2 },
                ..Default::default()
            });
            under = under.push(tooltip(ring, tip, tooltip::Position::Top).gap(6));
        }
        let mut dock = column![];
        if let Some(goal) = self.goal_pin() {
            dock = dock.push(goal);
        }
        if let Some(plan) = self.plan() {
            dock = dock.push(plan);
        }
        dock.push(perch).push(card).push(under).into()
    }

    /// The chat's /goal pinned above the message box: what done looks like, and how the checks went.
    fn goal_pin(&self) -> Option<Element<'_, Message>> {
        let g = self.goal.as_ref()?;
        let s = self.skin();
        let (mark, color, state) = match g.state.as_str() {
            _ if g.checking => (ic::NOW, s.accent, "checking…".to_string()),
            "met" => (ic::CHECK, s.ok, "met".to_string()),
            "stopped" => (ic::TODO, s.faint, format!("stopped after {} checks", g.limit)),
            _ if g.tries == 0 => (ic::NOW, s.accent, "working on it".to_string()),
            _ => (ic::NOW, s.accent, format!("not met yet · {} of {} checks", g.tries, g.limit)),
        };
        let mut head = row![
            icon(mark, 12).color(color),
            container(text(format!("Goal: {}", g.text)).size(13).font(MEDIUM).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true),
            text(state).size(12.5).color(s.faint),
        ]
        .spacing(10)
        .align_y(Alignment::Center);
        if g.state == "active" && self.busy.is_none() {
            head = head.push(button(icon(ic::CANCEL, 9).color(s.faint)).on_press(Message::ClearGoal).padding([4, 5]).style(move |_, st| bare(s, st)));
        }
        let mut body = column![head].spacing(4);
        if !g.reason.is_empty() && !g.checking {
            body = body.push(container(text(g.reason.clone()).size(12).color(s.faint)).padding(Padding { left: 22.0, ..Padding::ZERO }));
        }
        let card = container(body).padding([8, 12]).width(Fill).style(move |_| container::Style {
            background: Some(Background::Color(s.raise)),
            border: Border { radius: 14.0.into(), width: 1.0, color: s.line2 },
            ..Default::default()
        });
        Some(container(card).padding(Padding { bottom: 6.0, ..Padding::ZERO }).into())
    }

    /// The model's task list pinned above the message box: the step it is on and how far along it is, and the
    /// steps themselves when open (until all are done, unless opened or closed by hand).
    fn plan(&self) -> Option<Element<'_, Message>> {
        if self.tasks.is_empty() {
            return None;
        }
        let s = self.skin();
        let done = self.tasks.iter().filter(|t| t.1 == "done").count();
        let total = self.tasks.len();
        let open = self.plan_shown();
        let now = self.tasks.iter().find(|t| t.1 == "in_progress").or_else(|| self.tasks.iter().find(|t| t.1 != "done"));
        let title = match now {
            Some((task, _)) => task.clone(),
            None if total == 1 => "The task is done".to_string(),
            None => format!("All {total} tasks done"),
        };
        let head = row![
            icon(if done == total { ic::CHECK } else { ic::TASKS }, 12).color(if done == total { s.ok } else { s.accent }),
            container(text(title).size(13).font(MEDIUM).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true),
            text(format!("{done} of {total}")).size(12.5).color(s.faint),
            icon(if open { ic::DOWN } else { ic::CHEVRON_UP }, 8).color(s.faint),
        ]
        .spacing(10)
        .align_y(Alignment::Center);
        let head = button(head).on_press(Message::TogglePlan).width(Fill).padding([7, 10]).style(move |_, st| bare(s, st));
        let mut body = column![head];
        if open {
            let steps = column(self.tasks.iter().map(|(task, status)| {
                let (mark, mc, tc) = match status.as_str() {
                    "done" => (ic::CHECK, s.ok, s.faint),
                    "in_progress" => (ic::NOW, s.accent, s.text),
                    _ => (ic::TODO, s.faint, s.muted),
                };
                row![icon(mark, 10).color(mc).width(14), text(task.clone()).size(13).color(tc).width(Fill)].spacing(10).align_y(Alignment::Center).padding([4, 0]).into()
            }));
            let steps = scrollable(container(steps).padding(Padding { top: 0.0, right: 12.0, bottom: 8.0, left: 10.0 })).height(Length::Shrink).style(move |t, st| scroll_style(s, t, st));
            body = body.push(container(steps).max_height(190));
        }
        let card = container(body).padding(2).width(Fill).style(move |_| container::Style {
            background: Some(Background::Color(s.raise)),
            border: Border { radius: 14.0.into(), width: 1.0, color: s.line2 },
            ..Default::default()
        });
        Some(container(card).padding(Padding { bottom: 6.0, ..Padding::ZERO }).into())
    }

    /// The context's tokens in words: used of all, or free in a new chat.
    fn context_note(&self) -> String {
        let (used, total) = self.context;
        match (used > 0.0, self.messages.is_empty()) {
            (true, _) => format!("{} of {} tokens", thousands(used), thousands(total)),
            (false, true) => format!("{} tokens free", thousands(total)),
            (false, false) => "Shown after the next reply".to_string(), // a chat saved before tokens were kept
        }
    }

    /// The ghost's line. While a reply runs, after what it is doing: how long the reply has taken, the tokens it
    /// has written (its rounds' counts and this round's pieces so far), and its speed (this round's while it
    /// writes, else the reply's so far), as Claude Code shows them: Writing a file · 1m 23s · 2,345 tokens · 19 tok/s
    fn status_text(&self) -> String {
        let (Some(started), true) = (self.started, self.busy.is_some()) else { return self.status.clone() };
        let secs = started.elapsed().as_secs();
        let mut line = format!("{} · {}", self.status, if secs < 60 { format!("{secs}s") } else { format!("{}m {:02}s", secs / 60, secs % 60) });
        let tokens = self.reply_tokens + self.chunks as f64;
        if tokens > 0.0 {
            line.push_str(&format!(" · {} tokens", thousands(tokens)));
        }
        let rate = match self.writing_since {
            Some(t) if self.chunks > 1 => Some((self.chunks - 1) as f64 / t.elapsed().as_secs_f64().max(0.001)),
            _ if self.reply_writing.1 > 0.0 => Some(self.reply_writing.0 / self.reply_writing.1),
            _ => None,
        };
        if let Some(r) = rate {
            line.push_str(&format!(" · {r:.0} tok/s"));
        }
        line
    }

    fn message<'a>(&'a self, i: usize, m: &'a Msg) -> Element<'a, Message> {
        let s = self.skin();
        if m.role == Role::User && m.scheduled {
            // sent by the scheduler, not typed: a quiet line instead of a bubble
            let line = row![icon(ic::CLOCK, 12).color(s.faint), text(m.raw.as_str()).size(13).color(s.muted)].spacing(8).align_y(Alignment::Center);
            return row![Space::new().width(Fill), container(line).padding([6, 12]).style(move |_| container::Style { border: Border { width: 1.0, color: s.hover, radius: 16.0.into() }, ..Default::default() })].into();
        }
        if m.role == Role::User {
            // sent with a chip (/design-taste-frontend make a page): the chip again, then the words, with any
            // skill named further on styled as the message box showed it
            let names = self.skill_names();
            let said = move |t: &'a str| -> Element<'a, Message> {
                let spans = crate::slash_spans(t, &names);
                if spans.is_empty() {
                    return text(t).size(15).line_height(1.5).color(s.text).into();
                }
                let mut words: Vec<text::Span<'a, (), Font>> = Vec::new();
                let mut at = 0;
                for r in spans {
                    words.push(iced::widget::span(&t[at..r.start]).color(s.text));
                    // one piece when the line wraps: a word joiner (invisible) after the / and each hyphen,
                    // where a line could otherwise break
                    let name = t[r.clone()].replace('/', "/\u{2060}").replace('-', "-\u{2060}");
                    words.push(iced::widget::span(name).color(s.accent).font(MEDIUM));
                    at = r.end;
                }
                words.push(iced::widget::span(&t[at..]).color(s.text));
                rich_text(words).size(15).line_height(1.5).into()
            };
            let picked = m.raw.strip_prefix('/').map(|t| t.split_once(char::is_whitespace).unwrap_or((t, ""))).filter(|(name, _)| self.slash_all().iter().any(|e| e.name == *name));
            let content: Element<'a, Message> = match picked {
                Some((name, rest)) if rest.trim().is_empty() => self.chip(name),
                Some((name, rest)) => column![row![self.chip(name)], said(rest.trim_start())].spacing(6).into(),
                None => said(m.raw.as_str()).into(),
            };
            let bubble = container(content).padding([10, 16]).max_width(560).style(move |_| container::Style {
                background: Some(Background::Color(s.bubble)),
                border: border::rounded(16),
                ..Default::default()
            });
            return row![Space::new().width(Fill), bubble].into();
        }
        let live = self.busy.is_some() && i + 1 == self.messages.len();
        // in the order it happened: each round's thinking and text, then the tools it called; tool calls of
        // rounds with no text between them share one box
        let settings = self.md_settings();
        let mut parts: Vec<Element<'a, Message>> = Vec::new();
        let mut group: Vec<usize> = Vec::new();
        let flush = |group: &mut Vec<usize>, parts: &mut Vec<Element<'a, Message>>| {
            if group.is_empty() {
                return;
            }
            parts.push(self.tool_group(i, m, group));
            // a call waiting for the user's OK: its card right under it
            if let Some(t) = group.iter().map(|&j| &m.tools[j]).find(|t| t.ask.is_some()) {
                parts.push(self.approval_card(t));
            }
            group.clear();
        };
        let last = m.rounds.len().saturating_sub(1);
        for (k, r) in m.rounds.iter().enumerate() {
            if !r.think.is_empty() {
                flush(&mut group, &mut parts);
                let thinking = live && k == last && r.answer.is_empty() && !m.tools.iter().any(|t| t.round == k);
                let chevron = if m.think_open { ic::DOWN } else { ic::RIGHT };
                parts.push(
                    button(row![text(if thinking { "Thinking…" } else { "Thought" }).size(13.5).color(s.muted), icon(chevron, 9).color(s.faint)].spacing(7).align_y(Alignment::Center))
                        .on_press(Message::ToggleThinkView(i))
                        .padding([2, 0])
                        .style(move |_, _| button::Style { background: None, ..Default::default() })
                        .into(),
                );
                if m.think_open {
                    parts.push(container(text(r.think.as_str()).size(13.5).color(s.muted).line_height(1.55)).padding(Padding { left: 14.0, ..Padding::ZERO }).into());
                }
            }
            if !r.answer.is_empty() {
                flush(&mut group, &mut parts);
                parts.push(markdown::view_with(r.md.items(), settings, &Md(s)));
            }
            // task list updates show in the rail, not as lines in the reply
            group.extend(m.tools.iter().enumerate().filter(|(_, t)| t.round == k && t.name != "update_tasks").map(|(j, _)| j));
        }
        flush(&mut group, &mut parts);
        if !m.plan.is_empty() {
            parts.push(self.finished_plan(&m.plan));
        }
        if !live && (m.rounds.iter().any(|r| !r.answer.is_empty()) || m.changed) {
            let mut foot = row![small_button(s, ic::COPY, Message::Copy(i))].spacing(2).align_y(Alignment::Center);
            if i + 1 == self.messages.len() {
                foot = foot.push(small_button(s, ic::RETRY, Message::Retry));
            }
            if m.changed && !m.undone && m.checkpoint.is_some() {
                foot = foot.push(
                    container(button(row![icon(ic::UNDO, 11).color(s.muted), text("Undo file changes").size(12.5).color(s.muted)].spacing(7).align_y(Alignment::Center)).on_press(Message::Undo(i)).padding([4, 10]).style(move |_, st| pill(s, st, false)))
                        .padding(Padding { left: 6.0, ..Padding::ZERO }),
                );
            }
            if !m.stats.is_empty() {
                foot = foot.push(container(text(m.stats.as_str()).size(12).color(s.faint)).padding(Padding { left: 8.0, ..Padding::ZERO }));
            }
            parts.push(foot.into());
        }
        column(parts).spacing(12).into()
    }

    /// The task list a reply finished, inside the reply: its steps, each checked.
    fn finished_plan<'a>(&'a self, plan: &'a [(String, String)]) -> Element<'a, Message> {
        let s = self.skin();
        let title = if plan.len() == 1 { "The task is done".to_string() } else { format!("All {} tasks done", plan.len()) };
        let head = row![icon(ic::CHECK, 12).color(s.ok), text(title).size(13).font(MEDIUM).color(s.text)].spacing(10).align_y(Alignment::Center);
        let steps = column(plan.iter().map(|(task, _)| {
            row![icon(ic::CHECK, 10).color(s.ok).width(14), text(task.as_str()).size(13).color(s.faint).width(Fill)].spacing(10).align_y(Alignment::Center).padding([3, 0]).into()
        }));
        let body = column![head, container(steps).padding(Padding { top: 6.0, left: 0.0, right: 0.0, bottom: 0.0 })].padding([9, 12]);
        container(body)
            .width(Fill)
            .style(move |_| container::Style {
                background: Some(Background::Color(s.raise)),
                border: Border { radius: 14.0.into(), width: 1.0, color: s.line2 },
                ..Default::default()
            })
            .into()
    }

    /// A box of tool calls (`which`: indexes into the reply's tools), each a line; a click shows what it was
    /// given and what came back.
    fn tool_group<'a>(&'a self, i: usize, m: &'a Msg, which: &[usize]) -> Element<'a, Message> {
        let s = self.skin();
        let lines = column(which.iter().map(|&j| {
            let t = &m.tools[j];
            let (mark, color) = match (&t.result, t.denied) {
                (None, _) => (ic::CLOCK, s.warn),
                (Some(_), true) => (ic::CANCEL, s.bad),
                (Some(r), false) if r.starts_with("Error") => (ic::CANCEL, s.bad),
                (Some(_), false) => (ic::CHECK, s.ok),
            };
            let head = row![
                icon(tool_icon(&t.name), 12).color(s.muted),
                container(text(tool_label(t)).size(13).color(s.muted).wrapping(text::Wrapping::None)).width(Fill).clip(true),
                icon(mark, 11).color(color),
                icon(if t.open { ic::DOWN } else { ic::RIGHT }, 8).color(s.faint),
            ]
            .spacing(10)
            .align_y(Alignment::Center);
            let line = button(head).on_press(Message::ToggleTool(i, j)).width(Fill).padding([5, 8]).style(move |_, st| bare(s, st));
            if !t.open {
                return line.into();
            }
            let detail = container(scrollable(text(tool_detail(t)).size(12.5).font(MONO).color(s.muted).line_height(1.45)).height(Length::Shrink).style(move |t, st| scroll_style(s, t, st)))
                .max_height(260)
                .width(Fill)
                .padding([8, 10])
                .style(move |_| container::Style { background: Some(Background::Color(s.bg)), border: border::rounded(8), ..Default::default() });
            column![line, container(detail).padding(Padding { top: 2.0, right: 4.0, bottom: 6.0, left: 30.0 })].into()
        }))
        .spacing(1);
        container(lines).padding([4, 4]).style(move |_| container::Style { background: Some(Background::Color(s.hover)), border: border::rounded(12), ..Default::default() }).into()
    }

    /// The card for a call that needs the user's OK: what it would do, and Allow / Always allow / Deny.
    fn approval_card<'a>(&'a self, t: &'a crate::ToolLine) -> Element<'a, Message> {
        let s = self.skin();
        let (what, preview) = approval_text(t);
        let mut card = column![row![icon(tool_icon(&t.name), 13).color(s.accent), text(what).size(14).font(MEDIUM).color(s.text)].spacing(10).align_y(Alignment::Center)].spacing(10);
        if !preview.is_empty() {
            card = card.push(
                container(scrollable(text(preview).size(12.5).font(MONO).color(s.text).line_height(1.45)).height(Length::Shrink).style(move |t, st| scroll_style(s, t, st)))
                    .max_height(240)
                    .width(Fill)
                    .padding([8, 10])
                    .style(move |_| container::Style { background: Some(Background::Color(s.bg)), border: border::rounded(8), ..Default::default() }),
            );
        }
        let group = t.ask.as_deref().unwrap_or("");
        let answer = |label: String, decision: &'static str, main: bool| {
            button(text(label).size(13.5).color(if main { s.ink } else { s.text }))
                .on_press(Message::Approve(t.id.clone(), decision))
                .padding([6, 14])
                .style(move |_, st| answer_style(s, st, main))
        };
        card = card.push(row![answer("Allow".into(), "once", true), answer(format!("Always allow {}", group_name(group)), "always", false), answer("Deny".into(), "deny", false)].spacing(8));
        container(card)
            .padding([12, 14])
            .width(Fill)
            .style(move |_| container::Style { background: Some(Background::Color(s.raise)), border: Border { radius: 12.0.into(), width: 1.0, color: Color { a: 0.45, ..s.accent } }, ..Default::default() })
            .into()
    }

    /// How a reply's Markdown is drawn: the body font, headings a step above the text (not twice its size).
    fn md_settings(&self) -> markdown::Settings {
        let s = self.skin();
        let mut style = markdown::Style::from_palette(self.theme.palette());
        style.font = BODY;
        style.inline_code_font = MONO;
        style.code_block_font = MONO;
        style.inline_code_color = s.text;
        style.inline_code_highlight.background = Background::Color(s.raise2);
        style.inline_code_padding = Padding { top: 1.0, right: 4.0, bottom: 1.0, left: 4.0 };
        style.link_color = s.accent;
        let mut settings = markdown::Settings::with_text_size(15.5, style);
        settings.code_size = 13.5.into();
        settings.h1_size = 22.into();
        settings.h2_size = 19.5.into();
        settings.h3_size = 17.5.into();
        settings.h4_size = 16.into();
        settings.spacing = 12.into();
        settings
    }

    /// The ghost's mood, from what is happening.
    fn mood(&self) -> Mood {
        match self.status.as_str() {
            _ if self.api.is_none() && !self.status.starts_with("Connecting") => Mood::Sad,
            "Thinking" | "Reading" | "Reading the results" => Mood::Thinking,
            "Writing" => Mood::Talking,
            "Using a tool" => Mood::Tool,
            "Waiting for your OK" => Mood::Waiting,
            _ if self.idle_secs() > 120.0 => Mood::Sleepy,
            _ if self.messages.is_empty() && self.clock < 2600 => Mood::Wave,
            _ if self.happy_until > self.clock => Mood::Happy,
            _ => Mood::Idle,
        }
    }

    fn talking(&self) -> bool {
        self.last_chunk_at.is_some_and(|t| self.clock.saturating_sub(t) < 350)
    }
}

/// Draws a reply's Markdown; code blocks in the skin's colours (iced's own are near-black in every theme).
struct Md(&'static Skin);

impl<'a> markdown::Viewer<'a, Message> for Md {
    fn on_link_click(url: markdown::Uri) -> Message {
        Message::Link(url)
    }

    fn code_block(&self, settings: markdown::Settings, _: Option<&'a str>, _: &'a str, lines: &'a [markdown::Text]) -> Element<'a, Message> {
        let s = self.0;
        let code = column(lines.iter().map(|line| rich_text(line.spans(settings.style)).on_link_click(Self::on_link_click).font(settings.style.code_block_font).size(settings.code_size).into()));
        let bar = scrollable::Scrollbar::default().width(6).scroller_width(6);
        container(scrollable(container(code).padding([12, 14])).direction(scrollable::Direction::Horizontal(bar)).style(move |t, st| scroll_style(s, t, st)))
            .width(Fill)
            .style(move |_| container::Style { background: Some(Background::Color(s.raise2)), border: border::rounded(10), text_color: Some(s.text), ..Default::default() })
            .into()
    }
}

/// A box in the rail.
/// A count with thousands separators: 16384 -> "16,384".
fn thousands(n: f64) -> String {
    let digits = (n.max(0.0).round() as u64).to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Part of the title bar: a press drags the window, a double click maximizes it (buttons inside keep
/// their own clicks).
fn drag_area<'a>(content: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    mouse_area(content).on_press(Message::DragWindow).on_double_click(Message::ToggleMaximize).into()
}

fn icon_button<'a>(s: &'static Skin, c: char, on_press: Message) -> Element<'a, Message> {
    button(icon(c, 13).color(s.muted)).on_press(on_press).padding(8).style(move |_, st| row_style(s, st, false)).into()
}

fn small_button<'a>(s: &'static Skin, c: char, on_press: Message) -> Element<'a, Message> {
    button(icon(c, 12).color(s.faint)).on_press(on_press).padding(7).style(move |_, st| bare(s, st)).into()
}

// ---- styles

fn hovered(st: button::Status) -> bool {
    matches!(st, button::Status::Hovered | button::Status::Pressed)
}

fn row_style(s: &Skin, st: button::Status, on: bool) -> button::Style {
    button::Style { background: Some(Background::Color(if on { s.raise } else if hovered(st) { s.hover } else { Color::TRANSPARENT })), text_color: s.text, border: border::rounded(8), ..Default::default() }
}

fn bare(s: &Skin, st: button::Status) -> button::Style {
    button::Style { background: Some(Background::Color(if hovered(st) { s.hover } else { Color::TRANSPARENT })), text_color: s.muted, border: border::rounded(7), ..Default::default() }
}

fn pill(s: &Skin, st: button::Status, on: bool) -> button::Style {
    button::Style {
        background: Some(Background::Color(if on { Color { a: 0.14, ..s.accent } } else if hovered(st) { s.hover } else { Color::TRANSPARENT })),
        text_color: s.text,
        border: Border { radius: 999.0.into(), width: 1.0, color: s.line2 },
        ..Default::default()
    }
}

/// An approval card's buttons: Allow filled with the accent, the others outlined.
fn answer_style(s: &Skin, st: button::Status, main: bool) -> button::Style {
    let bg = match (main, hovered(st)) {
        (true, false) => s.accent,
        (true, true) => Color { a: 0.85, ..s.accent },
        (false, true) => s.hover,
        (false, false) => Color::TRANSPARENT,
    };
    button::Style { background: Some(Background::Color(bg)), text_color: s.text, border: Border { radius: 8.0.into(), width: if main { 0.0 } else { 1.0 }, color: s.line2 }, ..Default::default() }
}

fn round_style(s: &Skin, st: button::Status) -> button::Style {
    button::Style { background: Some(Background::Color(if hovered(st) { s.raise2 } else { Color::TRANSPARENT })), border: border::rounded(999), ..Default::default() }
}

fn send_style(s: &Skin, st: button::Status) -> button::Style {
    let c = if hovered(st) { Color { a: 0.85, ..s.accent } } else { s.accent };
    button::Style { background: Some(Background::Color(c)), border: border::rounded(999), ..Default::default() }
}

fn stop_style(s: &Skin, st: button::Status) -> button::Style {
    button::Style { background: hovered(st).then_some(Background::Color(s.hover)), border: border::rounded(8), ..Default::default() }
}

fn card_style(s: &Skin, st: button::Status) -> button::Style {
    button::Style {
        background: Some(Background::Color(if hovered(st) { s.raise } else { s.hover })),
        text_color: s.text,
        border: Border { radius: 12.0.into(), width: if s.dark { 0.0 } else { 1.0 }, color: s.line },
        ..Default::default()
    }
}

fn editor_style(s: &Skin, theme: &Theme, st: text_editor::Status) -> text_editor::Style {
    let base = text_editor::default(theme, st);
    text_editor::Style { background: Background::Color(Color::TRANSPARENT), border: Border { width: 0.0, ..base.border }, placeholder: s.faint, value: s.text, selection: Color { a: 0.3, ..s.accent }, ..base }
}

fn input_style(s: &Skin, _: &Theme, _: text_input::Status) -> text_input::Style {
    text_input::Style { background: Background::Color(s.hover), border: Border { radius: 8.0.into(), width: 1.0, color: s.line2 }, placeholder: s.faint, value: s.text, selection: Color { a: 0.3, ..s.accent }, icon: s.faint }
}

fn scroll_style(s: &Skin, theme: &Theme, st: scrollable::Status) -> scrollable::Style {
    let mut out = scrollable::default(theme, st);
    for rail in [&mut out.vertical_rail, &mut out.horizontal_rail] {
        rail.background = None;
        rail.border = Border::default();
        rail.scroller.background = Background::Color(s.line2);
        rail.scroller.border = border::rounded(4);
    }
    out
}
/// How full the context is: a faint track, and the used part clockwise from the top.
struct Ring {
    frac: f32,
    track: Color,
    fill: Color,
}

/// The preview page's area: nothing is drawn here (the page's own view covers it); drawing it moves the page
/// to wherever the layout put it.
struct PageArea;

impl<M> canvas::Program<M> for PageArea {
    type State = ();

    fn draw(&self, _: &(), _: &iced::Renderer, _: &Theme, bounds: iced::Rectangle, _: iced::mouse::Cursor) -> Vec<canvas::Geometry> {
        crate::preview::place(bounds);
        Vec::new()
    }
}

/// The message box's highlighter: a /name of an installed skill, anywhere in the text, in the accent colour
/// (a chip cannot sit inside the text, so a name past the start is styled where it is).
struct SlashNames {
    names: Vec<String>,
    color: Color,
    line: usize,
}

impl text::Highlighter for SlashNames {
    type Settings = (Vec<String>, Color);
    type Highlight = Color;
    type Iterator<'a> = std::vec::IntoIter<(std::ops::Range<usize>, Color)>;

    fn new(settings: &Self::Settings) -> Self {
        SlashNames { names: settings.0.clone(), color: settings.1, line: 0 }
    }

    fn update(&mut self, settings: &Self::Settings) {
        (self.names, self.color, self.line) = (settings.0.clone(), settings.1, 0);
    }

    fn change_line(&mut self, line: usize) {
        self.line = line;
    }

    fn highlight_line(&mut self, line: &str) -> Self::Iterator<'_> {
        self.line += 1;
        let color = self.color;
        crate::slash_spans(line, &self.names).into_iter().map(|r| (r, color)).collect::<Vec<_>>().into_iter()
    }

    fn current_line(&self) -> usize {
        self.line
    }
}

/// A magic wand, for skills (Segoe MDL2 has none): a stick up to the right, a four-point star at its tip
/// and two small sparkles.
struct Wand {
    color: Color,
}

impl<M> canvas::Program<M> for Wand {
    type State = ();

    fn draw(&self, _: &(), renderer: &iced::Renderer, _: &Theme, bounds: iced::Rectangle, _: iced::mouse::Cursor) -> Vec<canvas::Geometry> {
        use canvas::{LineCap, Path, Stroke};
        use iced::Point;
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let s = bounds.width.min(bounds.height);
        let stick = Path::line(Point::new(0.14 * s, 0.86 * s), Point::new(0.56 * s, 0.44 * s));
        frame.stroke(&stick, Stroke::default().with_width((0.13 * s).max(1.4)).with_color(self.color).with_line_cap(LineCap::Round));
        let star = |cx: f32, cy: f32, r: f32| {
            Path::new(|b| {
                for i in 0..8 {
                    let a = i as f32 * std::f32::consts::FRAC_PI_4 - std::f32::consts::FRAC_PI_2;
                    let rr = if i % 2 == 0 { r } else { r * 0.32 };
                    let p = Point::new(cx + rr * a.cos(), cy + rr * a.sin());
                    if i == 0 { b.move_to(p) } else { b.line_to(p) }
                }
                b.close();
            })
        };
        frame.fill(&star(0.7 * s, 0.3 * s, 0.28 * s), self.color);
        frame.fill(&star(0.24 * s, 0.2 * s, 0.13 * s), self.color);
        frame.fill(&star(0.86 * s, 0.7 * s, 0.11 * s), self.color);
        vec![frame.into_geometry()]
    }
}

impl<M> canvas::Program<M> for Ring {
    type State = ();

    fn draw(&self, _: &(), renderer: &iced::Renderer, _: &Theme, bounds: iced::Rectangle, _: iced::mouse::Cursor) -> Vec<canvas::Geometry> {
        use canvas::{path::Arc, LineCap, Path, Stroke};
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let (c, r) = (frame.center(), bounds.width.min(bounds.height) / 2.0 - 1.5);
        frame.stroke(&Path::circle(c, r), Stroke::default().with_width(2.0).with_color(self.track));
        if self.frac > 0.0 {
            let start = -std::f32::consts::FRAC_PI_2;
            // a sliver at least, so a new chat's few tokens still show
            let end = start + self.frac.max(0.02) * std::f32::consts::TAU;
            let arc = Path::new(|b| b.arc(Arc { center: c, radius: r, start_angle: iced::Radians(start), end_angle: iced::Radians(end) }));
            frame.stroke(&arc, Stroke::default().with_width(2.0).with_color(self.fill).with_line_cap(LineCap::Round));
        }
        vec![frame.into_geometry()]
    }
}
