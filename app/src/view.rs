//! How the app looks: Claude Desktop's calm structure in ghost-infer's colours, as two skins: Graphite
//! (dark, neutral greys) and Fog (light). Icons come from Segoe MDL2 Assets, the icon font every Windows 10
//! and 11 has; text is Instrument Sans, the greeting Georgia.

use crate::i18n::tr;
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
    pub const CLOCK: char = '\u{E916}';
    pub const UNDO: char = '\u{E7A7}';
    pub const DOWNLOAD: char = '\u{E896}';
    pub const OPEN_NEW: char = '\u{E8A7}';
    pub const FLAG: char = '\u{E7C1}';
    pub const REPEAT: char = '\u{E8EE}';
    pub const PHONE: char = '\u{E717}';
    pub const CODE: char = '\u{E943}';
    pub const MIC: char = '\u{E720}';
    pub const MIC_OFF: char = '\u{EC54}';
    pub const HANG_UP: char = '\u{E778}';
    pub const INFO: char = '\u{E946}';
    pub const COLOR: char = '\u{E790}';
    pub const KEYBOARD: char = '\u{E765}';
    pub const ATTACH: char = '\u{E723}';
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
        5..=11 => tr("Good morning"),
        12..=17 => tr("Good afternoon"),
        _ => tr("Good evening"),
    }
}

/// How long ago a chat changed, roughly: Today, Yesterday, 3 days ago.
fn ago(ms: f64) -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as f64).unwrap_or(0.0);
    match ((now - ms) / 86_400_000.0).floor() as i64 {
        ..=0 => tr("Today").into(),
        1 => tr("Yesterday").into(),
        d => tf!("{d} days ago", "{d} 天前"),
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
        if menu == Menu::Settings {
            return self.framed(stack![body, self.settings()].into());
        }
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
                vec![item(ic::COPY, tr("Copy as Markdown"), Message::CopyChat), Item { danger: true, ..item(ic::DELETE, tr("Delete"), Message::DeleteCurrent) }],
                Padding { top: 44.0, left: if self.sidebar { 264.0 + 16.0 } else { 56.0 }, ..Padding::ZERO },
                232.0,
                false,
            ),
            Menu::Settings => unreachable!("{}", tr("the settings are a window of their own")),
            // adding things; the tools themselves are always there, and the mode decides what needs an OK
            Menu::Plus => (
                None,
                vec![item(ic::ATTACH, tr("Attach files…"), Message::AttachFiles), item(ic::FOLDER, tr("Add folder…"), Message::AddFolder), item(ic::SKILL, tr("Install skill…"), Message::InstallSkill(false)), item(ic::DOWNLOAD, tr("Install skill from a link…"), Message::OpenSkillLink)],
                above(false),
                232.0,
                false,
            ),
            Menu::Model => {
                let current = self.model.as_ref().map(|m| m.id.as_str());
                let items = self.models.iter().map(|m| Item { icon: None, check: Some(current == Some(m.id.as_str())), ..item(ic::BOLT, &m.label, Message::PickModel(m.clone())) }).collect();
                (Some(tr("Model")), items, above(true), 260.0, true)
            }
            Menu::CallModel => {
                let same = Item { icon: None, detail: self.code_model().map(|m| m.label), check: Some(self.call_choice.is_none()), ..item(ic::BOLT, tr("Same as Code"), Message::PickCallModel(None)) };
                let items = std::iter::once(same)
                    .chain(self.models.iter().map(|m| Item { icon: None, check: Some(self.call_choice.as_deref() == Some(m.id.as_str())), ..item(ic::BOLT, &m.label, Message::PickCallModel(Some(m.id.clone()))) }))
                    .collect();
                // under its chip, which sits near the top of the call screen
                let below = match self.menu_at {
                    Some(at) => Padding { left: (at.x + at.width / 2.0 - 130.0).max(0.0), top: at.y + at.height + 6.0, ..Padding::ZERO },
                    None => Padding { top: 140.0, ..Padding::ZERO },
                };
                (Some(tr("Maya talks with")), items, below, 260.0, false)
            }
            Menu::Mode => {
                let items = crate::MODES
                    .iter()
                    .enumerate()
                    .map(|(i, (_, name, what))| Item { icon: None, detail: Some(tr(what).to_string()), hint: Some((i + 1).to_string()), check: Some(i == self.perm), ..item(ic::BOLT, tr(name), Message::SetMode(i)) })
                    .collect();
                (Some(tr("Mode")), items, above(true), 300.0, true)
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
                items.push(item(ic::ADD, tr("Choose a folder…"), Message::PickWorkdir));
                (Some(tr("Work in")), items, above(false), 300.0, false)
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
            .align_y(if matches!(menu, Menu::Chat | Menu::CallModel) { Alignment::Start } else { Alignment::End });
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
    /// An attached file: its name and size; `remove` gives it an x (in the message box, before it is sent).
    fn file_chip(&self, name: String, size: String, remove: Option<Message>) -> Element<'static, Message> {
        let s = self.skin();
        let mut label = row![icon(ic::FILE, 12).color(s.accent), text(name).size(13).font(MEDIUM).color(s.text), text(size).size(12).color(s.faint)].spacing(7).align_y(Alignment::Center);
        if let Some(m) = remove {
            label = label.push(button(icon(ic::CLOSE, 8).color(s.muted)).on_press(m).padding([2, 3]).style(move |_, st| bare(s, st)));
        }
        container(label)
            .padding([4, 9])
            .style(move |_| container::Style { background: Some(Background::Color(s.hover)), border: Border { radius: 8.0.into(), width: 1.0, color: s.line2 }, ..Default::default() })
            .into()
    }

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

    /// The settings window, over the app dimmed: its sections down the left, the one open on the right as
    /// groups of rows (a name, a line on what it does, its control at the right). Esc, the X or a click
    /// outside closes it.
    fn settings(&self) -> Element<'_, Message> {
        use crate::{SettingsPage as P, ThemeMode};
        let s = self.skin();
        let page = self.settings_page;
        let shade = mouse_area(container(Space::new()).width(Fill).height(Fill).style(move |_| container::Style {
            background: Some(Background::Color(Color { a: if s.dark { 0.55 } else { 0.28 }, ..Color::BLACK })),
            ..Default::default()
        }))
        .on_press(Message::Menu(None));

        // the sections
        let nav_item = |p: P, c: char, label: &'static str| -> Element<'static, Message> {
            let on = page == p;
            let line = row![icon(c, 14).color(if on { s.text } else { s.muted }).width(18), text(label).size(13.5).font(if on { MEDIUM } else { BODY }).color(if on { s.text } else { s.muted })].spacing(10).align_y(Alignment::Center);
            button(line).on_press(Message::SettingsPage(p)).width(Fill).padding([8, 10]).style(move |_, st| row_style(s, st, on)).into()
        };
        let nav = column![
            container(text(tr("Settings")).size(15).font(MEDIUM).color(s.text)).padding(Padding { top: 4.0, right: 10.0, bottom: 14.0, left: 10.0 }),
            nav_item(P::General, ic::SETTINGS, tr("General")),
            nav_item(P::Appearance, ic::COLOR, tr("Appearance")),
            nav_item(P::Models, ic::FOLDER, tr("Models")),
            nav_item(P::Voice, ic::MIC, tr("Voice")),
            nav_item(P::Maya, ic::MEMORY, tr("Maya")),
            nav_item(P::About, ic::INFO, tr("About")),
        ]
        .spacing(2);
        let nav = container(nav).width(208).height(Fill).padding([18, 12]).style(move |_| container::Style {
            background: Some(Background::Color(s.side)),
            border: Border { radius: iced::border::Radius { top_left: 14.0, bottom_left: 14.0, ..Default::default() }, ..Default::default() },
            ..Default::default()
        });

        // a row: its name and what it does, the control at the right
        let row_of = |name: &'static str, about: String, control: Element<'static, Message>| -> Element<'static, Message> {
            let words = column![text(name).size(14).color(s.text), text(about).size(12.5).color(s.faint)].spacing(3).width(Fill);
            row![words, control].spacing(24).align_y(Alignment::Center).padding([14, 18]).into()
        };
        // rows in a group, a hairline between them
        let group = |title: &'static str, rows: Vec<Element<'static, Message>>| -> Element<'static, Message> {
            let mut list = column![];
            let n = rows.len();
            for (i, r) in rows.into_iter().enumerate() {
                list = list.push(r);
                if i + 1 < n {
                    list = list.push(container(Space::new().height(1)).width(Fill).padding(Padding { left: 18.0, right: 18.0, ..Padding::ZERO }).style(move |_| container::Style { background: Some(Background::Color(s.line)), ..Default::default() }));
                }
            }
            let card = container(list).width(Fill).style(move |_| container::Style {
                background: Some(Background::Color(s.raise)),
                border: Border { radius: 12.0.into(), width: 1.0, color: s.line2 },
                ..Default::default()
            });
            column![text(title).size(12.5).font(MEDIUM).color(s.muted), card].spacing(8).into()
        };
        // a switch: a track that fills with the accent when on, its knob at that end
        let switch = |on: bool, msg: Message| -> Element<'static, Message> {
            let knob = container(Space::new().width(14).height(14)).style(move |_| container::Style {
                background: Some(Background::Color(if on { s.ink } else { s.muted })),
                border: border::rounded(7),
                ..Default::default()
            });
            let track = container(knob).width(38).height(22).padding(4).align_x(if on { Alignment::End } else { Alignment::Start }).align_y(Alignment::Center).style(move |_| container::Style {
                background: Some(Background::Color(if on { s.accent } else { Color::TRANSPARENT })),
                border: Border { radius: 11.0.into(), width: if on { 0.0 } else { 1.0 }, color: s.line2 },
                ..Default::default()
            });
            button(track).on_press(msg).padding(0).style(move |_, _| button::Style { background: None, ..Default::default() }).into()
        };
        // an outlined button, red for one that ends something
        let action = |label: &'static str, msg: Message, danger: bool| -> Element<'static, Message> {
            let c = if danger { s.bad } else { s.text };
            button(text(label).size(13).color(c))
                .on_press(msg)
                .padding([6, 14])
                .style(move |_, st| button::Style {
                    background: Some(Background::Color(if hovered(st) { s.hover } else { Color::TRANSPARENT })),
                    border: Border { radius: 8.0.into(), width: 1.0, color: if danger { Color { a: 0.5, ..s.bad } } else { s.line2 } },
                    ..Default::default()
                })
                .into()
        };
        // a value shown, not changed
        let value = |v: String| -> Element<'static, Message> { text(v).size(13).color(s.muted).into() };

        // the language picker: like the theme's, three in a bordered row
        let lang_choice = |label: &'static str, m: crate::i18n::Lang| -> Element<'static, Message> {
            let on = self.lang == m;
            button(text(label).size(13).color(if on { s.text } else { s.muted }).font(if on { MEDIUM } else { BODY }))
                .on_press(Message::SetLang(m))
                .padding([5, 14])
                .style(move |_, st| button::Style {
                    background: Some(Background::Color(if on { s.raise2 } else if hovered(st) { s.hover } else { Color::TRANSPARENT })),
                    border: border::rounded(7),
                    ..Default::default()
                })
                .into()
        };
        let picker_box = |inner: Element<'static, Message>| -> Element<'static, Message> {
            container(inner).padding(3).style(move |_| container::Style {
                border: Border { radius: 10.0.into(), width: 1.0, color: s.line2 },
                ..Default::default()
            }).into()
        };
        let (title, about, groups): (&str, &str, Vec<Element<'static, Message>>) = match page {
            P::General => (
                tr("General"),
                tr("How ghost starts and stops."),
                vec![
                    group(
                        tr("Language"),
                        vec![row_of(
                            tr("Language"),
                            tr("English by default. Chinese follows Windows unless you pick one.").into(),
                            picker_box(
                                row![
                                    lang_choice(tr("English"), crate::i18n::Lang::En),
                                    lang_choice(tr("中文"), crate::i18n::Lang::Zh),
                                    lang_choice(tr("System"), crate::i18n::Lang::System),
                                ]
                                .spacing(2)
                                .into(),
                            ),
                        )],
                    ),
                    group(
                        tr("Startup"),
                        vec![
                            row_of(tr("Open with the last model"), tr("Loads the model you used last when ghost starts.").into(), switch(self.load_last, Message::ToggleLoadLast)),
                            row_of(tr("Warm up after loading"), tr("Runs a few short replies after a model loads, so the first real answer comes quickly.").into(), switch(self.warm_up, Message::ToggleWarmUp)),
                        ],
                    ),
                    group(
                        tr("Quitting"),
                        vec![row_of(tr("Quit ghost"), tr("Closing the window keeps ghost in the tray. This stops it and frees the model's memory.").into(), action(tr("Quit"), Message::Quit, true))],
                    ),
                ],
            ),
            P::Appearance => {
                let choice = |label: &'static str, m: ThemeMode| -> Element<'static, Message> {
                    let on = self.mode == m;
                    button(text(label).size(13).color(if on { s.text } else { s.muted }).font(if on { MEDIUM } else { BODY }))
                        .on_press(Message::SetTheme(m))
                        .padding([5, 14])
                        .style(move |_, st| button::Style {
                            background: Some(Background::Color(if on { s.raise2 } else if hovered(st) { s.hover } else { Color::TRANSPARENT })),
                            border: border::rounded(7),
                            ..Default::default()
                        })
                        .into()
                };
                let picker = container(row![choice(tr("Light"), ThemeMode::Light), choice(tr("Dark"), ThemeMode::Dark), choice(tr("System"), ThemeMode::System)].spacing(2)).padding(3).style(move |_| container::Style {
                    border: Border { radius: 10.0.into(), width: 1.0, color: s.line2 },
                    ..Default::default()
                });
                (tr("Appearance"), tr("How ghost looks."), vec![group(tr("Theme"), vec![row_of(tr("Colour theme"), tr("Fog is light, Graphite dark. System follows Windows.").into(), picker.into())])])
            }
            P::Models => {
                // each folder by its path; the ones added here can be removed (the server's own come back when it
                // starts again, so they stay)
                let mut rows: Vec<Element<'static, Message>> = self
                    .model_folders
                    .iter()
                    .map(|f| {
                        let ours = self.model_dirs.contains(f);
                        let words = column![text(f.clone()).size(13.5).color(s.text), text(if ours { tr("Added in ghost") } else { tr("Set when ghost started") }).size(12.5).color(s.faint)].spacing(3).width(Fill);
                        let control: Element<'static, Message> = if ours { action(tr("Remove"), Message::RemoveModelFolder(f.clone()), false) } else { Space::new().width(0).into() };
                        row![words, control].spacing(24).align_y(Alignment::Center).padding([14, 18]).into()
                    })
                    .collect();
                rows.push(row_of(tr("Add a folder"), tr("Its GGUF files show in the model menus at once, and from then on.").into(), action(tr("Add folder…"), Message::UseModelFolder, false)));
                (
                    tr("Models"),
                    tr("Where ghost finds models."),
                    vec![
                        group(tr("Folders"), rows),
                        text("Voice calls find their files in these folders too: a speech folder (a maya1 GGUF with snac.gguf) and a whisper folder with a whisper*.gguf, from huggingface.co/affifuddin-hazam/ghost-voice. They are read when ghost starts.").size(12.5).color(s.faint).into(),
                    ],
                )
            }
            P::Voice => {
                let (speech, listen) = &self.voice_models;
                let named = |v: &Option<String>| v.clone().unwrap_or_else(|| tr("Not set up").into());
                (
                    tr("Voice"),
                    tr("Voice calls with Maya."),
                    vec![
                        group(
                            tr("Models"),
                            vec![
                                row_of(tr("Voice"), tr("What Maya speaks with.").into(), value(named(speech))),
                                row_of(tr("Listening"), tr("What turns your speech into text.").into(), value(named(listen))),
                                row_of(
                                    tr("Call model"),
                                    tr("Picked on the call screen. Kept loaded beside the Code model when your graphics card has room for both and the voice; otherwise a call asks before swapping them.").into(),
                                    value(match (&self.call_choice, self.call_target()) {
                                        (None, _) => tr("Same as Code").into(),
                                        (Some(_), Some(m)) => m.label,
                                        (Some(id), None) => crate::short_model(id),
                                    }),
                                ),
                            ],
                        ),
                        group(tr("During a call"), vec![row_of(tr("Show response timings"), tr("How long each reply took to start: for checking speed.").into(), switch(self.call_timings, Message::ToggleCallTimings))]),
                        text(tr("The voice models are chosen when the server starts (its --tts and --stt options).")).size(12.5).color(s.faint).into(),
                    ],
                )
            }
            P::Maya => {
                let m = &self.maya_memory;
                let list = |k: &str| -> Vec<String> { m.get(k).and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect() };
                // a remembered line, with an X that makes her forget it
                let line = |t: String, msg: Message| -> Element<'static, Message> {
                    let x = button(icon(ic::CANCEL, 9).color(s.faint)).on_press(msg).padding([4, 5]).style(move |_, st| bare(s, st));
                    row![text(t).size(13).color(s.text).width(Fill), x].spacing(12).align_y(Alignment::Center).padding([10, 18]).into()
                };
                let none = |t: &'static str| -> Vec<Element<'static, Message>> { vec![container(text(t).size(12.5).color(s.faint)).padding([12, 18]).into()] };
                let or_none = |rows: Vec<Element<'static, Message>>, t: &'static str| if rows.is_empty() { none(t) } else { rows };

                let count = m.get("count").and_then(Json::as_f64).unwrap_or(0.0) as u64;
                let minutes = (m.get("seconds").and_then(Json::as_f64).unwrap_or(0.0) / 60.0).round() as u64;
                let stage = m.get("stage").and_then(Json::as_str).unwrap_or("stranger");
                let known = if count == 0 { tr("You haven't talked yet.").to_string() } else { if crate::i18n::is_zh() { format!("共 {count} 次通话，约 {minutes} 分钟。") } else { format!("{count} call{}, about {minutes} min in all.", if count == 1 { "" } else { "s" }) } };
                let stage = stage[..1].to_uppercase() + &stage[1..];
                let name = row![
                    text_input(tr("What she calls you"), &self.maya_name).on_input(Message::MayaName).on_submit(Message::SaveMayaName).size(13).padding([6, 10]).width(200).style(move |t, st| input_style(s, t, st)),
                    action(tr("Save"), Message::SaveMayaName, false),
                ]
                .spacing(8)
                .align_y(Alignment::Center);

                let facts = list("facts").into_iter().enumerate().map(|(i, f)| line(f, Message::MayaForget("facts", i))).collect();
                let moments = list("moments").into_iter().enumerate().map(|(i, f)| line(f, Message::MayaForget("moments", i))).collect();
                // the calls, latest first, each by its summary
                let calls = m.get("calls").and_then(Json::as_array).unwrap_or(&[]);
                let calls = calls.iter().enumerate().rev().map(|(i, c)| line(c.get("summary").and_then(Json::as_str).unwrap_or("").to_string(), Message::MayaForget("calls", i))).collect();
                let impression = m.get("impression").and_then(Json::as_str).unwrap_or("").to_string();
                let impression = if impression.is_empty() { none(tr("Nothing yet.")) } else { vec![line(impression, Message::ClearImpression)] };
                (
                    tr("Maya"),
                    tr("What she remembers of you, from your calls. Remove anything you don't want her to know."),
                    vec![
                        group(tr("You"), vec![row_of(tr("Your name"), tr("What she calls you.").into(), name.into()), row_of(tr("How close you are"), known, value(stage))]),
                        group(tr("What she knows about you"), or_none(facts, tr("Nothing yet."))),
                        group(tr("Moments you share"), or_none(moments, tr("None yet."))),
                        group(tr("Calls she remembers"), or_none(calls, tr("None yet."))),
                        group(tr("How she sees you"), impression),
                        group(tr("Start over"), vec![row_of(tr("Forget everything"), tr("She meets you again as a stranger. The old memory is kept beside it as a backup.").into(), action(tr("Start over"), Message::ResetMaya, true))]),
                    ],
                )
            }
            P::About => {
                let sys = &self.system;
                let gpu = sys.get("gpus").and_then(Json::as_array).and_then(|g| g.first()).and_then(|g| g.get("name")).and_then(Json::as_str).map(|n| n.trim_start_matches("NVIDIA ").to_string()).unwrap_or_else(|| tr("None found").into());
                let model = self.model.as_ref().map(|m| m.label.clone()).unwrap_or_else(|| tr("None loaded").into());
                let head = row![
                    canvas(Ghost::still(2.0, s.accent)).width(48).height(44),
                    column![text("ghost").size(18).font(MEDIUM).color(s.text), text(tf!("Version {}", "版本 {}", env!("CARGO_PKG_VERSION"))).size(12.5).color(s.faint)].spacing(2),
                ]
                .spacing(14)
                .align_y(Alignment::Center);
                (
                    tr("About"),
                    tr("This copy of ghost and the computer it runs on."),
                    vec![
                        head.into(),
                        group(
                            "Running",
                            vec![
                                row_of(tr("Engine"), tr("Runs the models on this computer.").into(), value("ghost-infer".into())),
                                row_of(tr("Model"), tr("The model loaded now.").into(), value(model)),
                                row_of(tr("Graphics"), tr("Where the model runs.").into(), value(gpu)),
                            ],
                        ),
                    ],
                )
            }
        };
        let close = button(icon(ic::CANCEL, 11).color(s.muted)).on_press(Message::Menu(None)).padding(8).style(move |_, st| bare(s, st));
        let header = row![column![text(title).size(20).font(MEDIUM).color(s.text), text(about).size(13).color(s.faint)].spacing(4).width(Fill), close].align_y(Alignment::Start);
        let mut content = column![header].spacing(22);
        for g in groups {
            content = content.push(g);
        }
        let content = scrollable(container(content).padding(Padding { top: 22.0, right: 26.0, bottom: 26.0, left: 28.0 })).height(Fill).style(move |t, st| scroll_style(s, t, st));

        let (w, h) = ((self.window.width - 64.0).clamp(560.0, 860.0), (self.window.height - 64.0).clamp(420.0, 600.0));
        let card = container(row![nav, content]).width(w).height(h).style(move |_| container::Style {
            background: Some(Background::Color(s.bg)),
            // no shadow: the CPU renderer's redraw of what changed leaves a blurred one (and the card over it)
            // stale and black
            border: Border { radius: 14.0.into(), width: 1.0, color: s.line2 },
            ..Default::default()
        });
        stack![shade, container(opaque(card)).width(Fill).height(Fill).center_x(Fill).center_y(Fill)].into()
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
            env = env.push(line(ic::CHIP, tr("Memory").into(), format!("{:.1} / {:.0} GB", f(sys, "ram_total_gb") - f(sys, "ram_available_gb"), f(sys, "ram_total_gb"))));
        }
        let model = self.model.as_ref().map(|m| m.label.clone()).unwrap_or_else(|| tr("No model").into());
        let runs = sys.get("model_runs_on").and_then(Json::as_str).map(|b| if b.starts_with("GPU") { "GPU" } else { "CPU" }).unwrap_or("");
        env = env.push(line(ic::BOLT, model.clone(), runs.into()));
        let dir = if self.messages.is_empty() { self.new_chat_dir() } else { self.chat_cwd.as_deref().or(self.folders.first().map(String::as_str)) };
        env = env.push(match dir {
            Some(d) => {
                let place = std::path::Path::new(d).parent().map(|p| format!(" · {}", p.display())).unwrap_or_default();
                container(pair(crate::folder_name(d), place, ic::FOLDER)).padding([5, 0]).into()
            }
            None => line(ic::FOLDER, tr("No folder").into(), String::new()),
        });
        // the folders every chat may use besides its own
        env = env.push(fold("environment.folders", tr("Other folders"), self.folders.len().to_string()));
        if open("environment.folders") {
            for (i, d) in self.folders.iter().enumerate() {
                env = env.push(row![Space::new().width(18), container(text(d.clone()).size(12.5).color(s.muted).wrapping(text::Wrapping::None)).width(Fill).clip(true), remove(Message::RemoveFolder(i))].spacing(8).align_y(Alignment::Center));
            }
            env = env.push(row![Space::new().width(18), link(tr("Add folder"), Message::AddFolder)].spacing(8));
        }
        let environment = card("environment", tr("Environment"), faint(model), env.into());

        // context: how full it is, and what fills it, as Claude's rail breaks it down
        let (used, total) = self.context;
        let pct = if total > 0.0 { (100.0 * used / total).clamp(0.0, 100.0) } else { 0.0 };
        let part = |k: &str| self.parts.get(k).and_then(Json::as_f64).unwrap_or(0.0);
        let counted = ["system", "tools", "skills", "instructions", "memory"].iter().map(|k| part(k)).sum::<f64>();
        let grey = |a: f32| Color { a, ..s.muted };
        let parts: Vec<(&str, f64, Color)> = if self.parts != Json::Null && used > 0.0 {
            vec![
                (tr("Messages"), (used - counted).max(0.0), PART_BLUE),
                (tr("Tools"), part("tools"), PART_ORANGE),
                (tr("Skills"), part("skills"), PART_GREEN),
                (tr("System prompt"), part("system"), grey(0.85)),
                (tr("Instructions"), part("instructions"), grey(0.6)),
                (tr("Memory"), part("memory"), grey(0.4)),
            ]
        } else {
            vec![(tr("Used"), used, PART_BLUE)]
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
            ctx = ctx.push(faint(tr("No model").into()));
        } else {
            ctx = ctx.push(row![text(tf!("{pct:.0}% used", "已用 {pct:.0}%")).size(13).font(MEDIUM).color(s.text), Space::new().width(Fill), text(format!("{} / {}", tokens_k(used), tokens_k(total))).size(12.5).color(s.faint)].align_y(Alignment::Center));
            ctx = ctx.push(bar);
            let mut rows = column![];
            for (label, n, c) in parts.iter().filter(|_| used > 0.0) {
                rows = rows.push(entry(label, *n, *c));
            }
            rows = rows.push(entry(tr("Free space"), free, s.raise2));
            ctx = ctx.push(rows);
        }
        // the GHOST.md files and the remembered facts, as Claude's memory files
        let in_use = self.instructions.iter().filter(|f| f.1).count();
        let mut lists = column![fold("context.instructions", tr("Instructions"), in_use.to_string())];
        if open("context.instructions") {
            for (path, exists, global) in &self.instructions {
                let file = path.rsplit(['\\', '/']).next().unwrap_or(path).to_string();
                let note = if *global { tr("every chat").to_string() } else { path.rsplit(['\\', '/']).nth(1).unwrap_or("").to_string() };
                let words: [text::Span<'static, (), Font>; 2] = [iced::widget::span(file).color(if *exists { s.text } else { s.faint }), iced::widget::span(format!(" · {note}")).color(s.faint)];
                lists = lists.push(row![Space::new().width(18), container(rich_text(words).size(12.5).wrapping(text::Wrapping::None)).width(Fill).clip(true), link(if *exists { tr("Edit") } else { tr("Add") }, Message::EditInstructions(path.clone()))].spacing(8).align_y(Alignment::Center));
            }
        }
        lists = lists.push(fold("context.memory", tr("Memory"), self.memory.len().to_string()));
        if open("context.memory") {
            for (i, fact) in self.memory.iter().enumerate() {
                lists = lists.push(row![Space::new().width(18), text(fact.clone()).size(12.5).color(s.muted).width(Fill), remove(Message::Forget(i))].spacing(8).align_y(Alignment::Center));
            }
            let add_memory = text_input(tr("Add a memory"), &self.memory_draft).on_input(Message::MemoryDraft).on_submit(Message::AddMemory).size(12.5).padding([5, 8]).style(move |t, st| input_style(s, t, st));
            lists = lists.push(row![Space::new().width(18), add_memory].spacing(8).padding(Padding { top: 4.0, ..Padding::ZERO }));
        }
        ctx = ctx.push(lists);
        // when old output starts being taken out (the server keeps the reply's length free, at most a quarter)
        if total > 0.0 && self.chat.is_some() {
            let room = (total - 4096f64.min(total / 4.0) - used).max(0.0);
            let can_trim = self.busy.is_none().then_some(Message::TrimChat);
            let until = container(text(tf!("{} until old output is trimmed", "距裁剪旧输出还有 {}", tokens_k(room))).size(12).color(s.faint).wrapping(text::Wrapping::None)).width(Fill).clip(true);
            ctx = ctx.push(row![until, action(tr("Trim now"), can_trim)].spacing(8).align_y(Alignment::Center));
        }
        let filled = (pct.round() as u16).clamp(1, 99);
        let mini = container(row![
            container(Space::new()).width(Length::FillPortion(filled)).height(4).style(move |_| container::Style { background: Some(Background::Color(PART_BLUE)), border: border::rounded(99), ..Default::default() }),
            Space::new().width(Length::FillPortion(100 - filled)),
        ])
        .width(26)
        .style(move |_| container::Style { background: Some(Background::Color(s.raise2)), border: border::rounded(99), ..Default::default() });
        let folded: Element<'static, Message> = if total > 0.0 { row![faint(format!("{pct:.0}%")), mini].spacing(8).align_y(Alignment::Center).into() } else { Space::new().into() };
        let context = card("context", tr("Context"), folded, ctx.into());

        // changes: the files this chat changed, their lines, and undo for all of them
        let changes = self.changes();
        let (adds, dels) = changes.iter().fold((0, 0), |(a, d), c| (a + c.1, d + c.2));
        let counts = |a: usize, d: usize| -> Element<'static, Message> { row![text(format!("+{a}")).size(12.5).font(MONO).color(s.ok), text(format!("−{d}")).size(12.5).font(MONO).color(s.bad)].spacing(6).into() };
        let mut ch = column![];
        if changes.is_empty() {
            ch = ch.push(faint(tr("No changes").into()));
        }
        for (path, add, del) in &changes {
            let name = path.rsplit(['\\', '/']).next().unwrap_or(path).to_string();
            ch = ch.push(row![icon(ic::FILE, 12).color(s.faint).width(18), container(text(name).size(13).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true), counts(*add, *del)].spacing(8).align_y(Alignment::Center).padding([4, 0]));
        }
        if !changes.is_empty() {
            ch = ch.push(row![icon(ic::EDIT, 12).color(s.faint).width(18), counts(adds, dels), Space::new().width(Fill), action(tr("Undo all"), Some(Message::UndoAll))].spacing(8).align_y(Alignment::Center).padding(Padding { top: 6.0, ..Padding::ZERO }));
        }
        let changed: Element<'static, Message> = if changes.is_empty() { Space::new().into() } else { counts(adds, dels) };
        let changes = card("changes", tr("Changes"), changed, ch.into());

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
        let tasks = card("tasks", tr("Tasks"), faint(tf!("{done} of {total}", "{done} / {total}")), tasks.into());

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
            bt = bt.push(button(row![icon(ic::TASKS, 12).color(s.faint).width(18), text(tf!("{finished} finished", "{finished} 项已完成")).size(13).color(s.text)].spacing(8).align_y(Alignment::Center)).on_press(Message::ShowTasks).width(Fill).padding([2, 0]).style(move |_, _| button::Style { background: None, ..Default::default() }));
        }
        let running_now = list.len() - finished;
        let tasks_note = if running_now > 0 { tf!("{running_now} running", "{running_now} 项运行中") } else { tf!("{finished} finished", "{finished} 项已完成") };
        let background = (!list.is_empty()).then(|| card("background", tr("Background tasks"), faint(tasks_note), bt.into()));

        // scheduled: the tasks the model set and the /loop ones, and the one running now
        let jobs = self.background.get("jobs").and_then(Json::as_array).unwrap_or(&[]);
        let running = self.background.get("running").and_then(Json::as_str).is_some();
        let mut sc = column![];
        if running {
            sc = sc.push(line(ic::NOW, tr("Running one now").into(), String::new()));
        }
        for j in jobs {
            let g = |k: &str| j.get(k).and_then(Json::as_str).unwrap_or("").to_string();
            sc = sc.push(row![pair(g("name"), tf!(" · {}, next {}", " · {}，下次 {}", g("when"), g("next")), ic::CLOCK), link("Run", Message::RunScheduled(g("id"))), remove(Message::CancelScheduled(g("id")))].spacing(4).align_y(Alignment::Center).padding([3, 0]));
        }
        let due = match (running, jobs.len()) {
            (true, _) => "running".to_string(),
            (false, 0) => String::new(),
            (false, n) => n.to_string(),
        };
        let scheduled = (running || !jobs.is_empty()).then(|| card("scheduled", tr("Scheduled"), faint(due), sc.into()));

        // skills: installed ones, and installing
        let mut sk = column![];
        for (name, desc) in &self.skills {
            sk = sk.push(row![pair(name.clone(), format!(" · {desc}"), ic::SKILL), remove(Message::RemoveSkill(name.clone()))].spacing(8).align_y(Alignment::Center).padding([2, 0]));
        }
        sk = sk.push(row![link(tr("Install from folder"), Message::InstallSkill(false)), link(tr("From .zip"), Message::InstallSkill(true))]);
        let skill_link = text_input(tr("Skill link (GitHub or .zip)"), &self.skill_link).id("skill-link").on_input(Message::SkillLink).on_submit(Message::InstallLink).size(12.5).padding([5, 8]).style(move |t, st| input_style(s, t, st));
        sk = sk.push(container(skill_link).padding(Padding { top: 4.0, ..Padding::ZERO }));
        // a link or folder with several skills: which to install
        if let Some((_, choices)) = &self.skill_choices {
            sk = sk.push(container(faint(tf!("It has {} skills. Install which?", "其中有 {} 个技能，要装哪个？", choices.len()))).padding(Padding { top: 6.0, ..Padding::ZERO }));
            for (path, name, desc) in choices {
                let place = if path.is_empty() { desc.clone() } else { format!("{path} · {desc}") };
                sk = sk.push(row![pair(name.clone(), format!(" · {place}"), ic::SKILL), link(tr("Install"), Message::InstallChoice(Some(path.clone())))].spacing(8).align_y(Alignment::Center));
            }
            sk = sk.push(row![link(tr("Install all"), Message::InstallChoice(None)), link(tr("Cancel"), Message::CancelChoice)]);
        }
        let skills = card("skills", tr("Skills"), faint(if self.skills.is_empty() { String::new() } else { self.skills.len().to_string() }), sk.into());

        // MCP servers: each with its state, and editing the list
        let mut mc = column![];
        for (name, state, tools) in &self.mcp {
            let (dot, note) = match state.as_str() {
                "ready" => (s.ok, tf!("{tools} tools", "{tools} 个工具")),
                "starting" => (s.warn, tr("starting").to_string()),
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
        mc = mc.push(row![link(tr("Edit servers"), Message::McpEdit), link(tr("Reload"), Message::McpReload), link(tr("From Claude"), Message::McpImport)]);
        let ready = self.mcp.iter().filter(|m| m.1 == "ready").count();
        let broken = self.mcp.iter().filter(|m| m.1 != "ready" && m.1 != "starting").count();
        let mcp_note = match (ready, broken) {
            (_, b) if b > 0 => tf!("{b} not working", "{b} 个不可用"),
            (0, _) => String::new(),
            (r, _) => tf!("{r} ready", "{r} 个就绪"),
        };
        let mcp = card("mcp", tr("MCP servers"), faint(mcp_note), mc.into());

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
            text(tr("Background tasks")).size(14).font(MEDIUM).color(s.text),
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
                (TaskState::Failed, _) => (tr("Failed").to_string(), s.bad),
                (TaskState::Stopped, Some(n)) => (tf!("Stopped · {}", "已停止 · {}", duration(n)), s.faint),
                (TaskState::Stopped, None) => (tr("Stopped").to_string(), s.faint),
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
            let group = button(row![icon(if self.finished_open { ic::DOWN } else { ic::RIGHT }, 8).color(s.faint).width(14), text(tr("Finished")).size(12.5).color(s.faint), text(finished.len().to_string()).size(12.5).color(s.faint)].spacing(6).align_y(Alignment::Center))
                .on_press(Message::ToggleFinished)
                .padding(Padding { top: 10.0, right: 0.0, bottom: 4.0, left: 2.0 })
                .style(move |_, _| button::Style { background: None, ..Default::default() });
            col = col.push(group);
            if self.finished_open {
                for t in list.iter().filter(|t| t.state != TaskState::Running).take(self.tasks_shown) {
                    col = col.push(item(t));
                }
                if finished.len() > self.tasks_shown {
                    col = col.push(button(text(tr("Show more")).size(12.5).color(s.muted)).on_press(Message::MoreTasks).padding([6, 4]).style(move |_, st| bare(s, st)));
                }
            }
        }
        if list.is_empty() {
            col = col.push(text(tr("None")).size(12.5).color(s.faint));
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
        let chrome = drag_area(container(row![icon_button(s, ic::CLOSE_PANE, Message::ToggleSidebar), Space::new().width(Fill), self.mode_switch()].align_y(Alignment::Center)).width(Fill).height(44).align_y(Alignment::Center));
        let nav = |c: char, label: &'static str, msg: Message| {
            button(row![icon(c, 14).color(s.muted), text(label).size(14).color(s.text)].spacing(12).align_y(Alignment::Center))
                .on_press(msg)
                .width(Fill)
                .padding([7, 10])
                .style(move |_, st| row_style(s, st, false))
        };
        let mut col = column![chrome, nav(ic::ADD, tr("New chat"), Message::NewChat), nav(ic::SEARCH, tr("Search"), Message::ToggleSearch)].spacing(1);
        if self.searching {
            col = col.push(
                text_input(tr("Search chats"), &self.filter)
                    .on_input(Message::Filter)
                    .size(13.5)
                    .padding([7, 10])
                    .style(move |t, st| input_style(s, t, st)),
            );
        }
        let recents_head = button(row![text(tr("Recents")).size(12).color(s.faint), icon(if self.recents_open { ic::DOWN } else { ic::RIGHT }, 8).color(s.faint)].spacing(6).align_y(Alignment::Center))
            .on_press(Message::ToggleRecents)
            .padding(Padding { top: 14.0, right: 10.0, bottom: 4.0, left: 10.0 })
            .style(move |_, _| button::Style { background: None, ..Default::default() });
        col = col.push(recents_head);
        let filter = self.filter.to_lowercase();
        let rows = column(self.chats.iter().filter(|c| filter.is_empty() || c.title.to_lowercase().contains(&filter)).map(|c| {
            let on = self.chat.as_deref() == Some(c.id.as_str());
            let hover = self.hovered.as_deref() == Some(c.id.as_str());
            let name = container(text(if c.title.is_empty() { tr("New chat") } else { c.title.as_str() }).size(14).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true);
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
        let model = self.model.as_ref().map(|m| m.label.clone()).unwrap_or_else(|| tr("No model").into());
        let me = button(
            row![
                canvas(Ghost::still(1.0, s.accent)).width(24).height(22),
                column![text(tr("This PC")).size(13.5).color(s.text), text(model).size(12).color(s.faint).wrapping(text::Wrapping::None)].width(Fill),
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
            top = top.push(icon_button(s, ic::OPEN_PANE, Message::ToggleSidebar)).push(self.mode_switch());
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
        if self.voice {
            return container(column![top, self.call_view()]).style(move |_| container::Style { background: Some(Background::Color(s.bg)), ..Default::default() }).into();
        }

        if self.messages.is_empty() {
            // a new chat: the greeting, the ghost on the message box, recent chats; on the first run, with no
            // model yet, the models to download instead of the message box
            let hello = text(if self.api.is_some() { greeting() } else { tr("Connecting…") }).size(38).font(SERIF).color(s.text);
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
                center = center.push(Space::new().height(30)).push(row![text(tr("Recent chats")).size(13).color(s.muted)].padding([0, 2])).push(Space::new().height(10)).push(cards);
            }
            center = center.push(Space::new().height(Fill));
            return container(column![top, container(center).center_x(Fill).height(Fill).padding([0, 24])]).style(move |_| container::Style { background: Some(Background::Color(s.bg)), ..Default::default() }).into();
        }

        let first = self.messages.len().saturating_sub(self.shown);
        let mut items: Vec<Element<'_, Message>> = Vec::new();
        if first > 0 {
            items.push(container(button(text(tf!("Show {} earlier messages", "显示更早的 {} 条消息", first.min(crate::SHOWN))).size(13).color(s.muted)).on_press(Message::ShowEarlier).padding([6, 14]).style(move |_, st| pill(s, st, false))).center_x(Fill).into());
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

    /// The top-left switch between the app's two modes, as Claude Desktop's: a voice call, or the chat that
    /// works with code and files.
    fn mode_switch(&self) -> Element<'_, Message> {
        let s = self.skin();
        let seg = |c: char, tip: &'static str, on: bool, msg: Message| {
            let b = button(container(icon(c, 13).color(if on { s.text } else { s.muted })).width(30).height(26).align_x(Alignment::Center).align_y(Alignment::Center))
                .on_press(msg)
                .padding(0)
                .style(move |_, st| button::Style {
                    background: Some(Background::Color(if on { s.raise2 } else if hovered(st) { s.hover } else { Color::TRANSPARENT })),
                    border: border::rounded(7),
                    ..Default::default()
                });
            tooltip(b, container(text(tip).size(12).color(s.text)).padding([4, 8]).style(move |_| container::Style { background: Some(Background::Color(s.raise)), border: border::rounded(6), ..Default::default() }), tooltip::Position::Bottom)
        };
        container(row![seg(ic::PHONE, tr("Voice call"), self.voice, Message::SetVoice(true)), seg(ic::CODE, tr("Code"), !self.voice, Message::SetVoice(false))].spacing(2))
            .padding(2)
            .style(move |_| container::Style { background: Some(Background::Color(s.hover)), border: border::rounded(9), ..Default::default() })
            .into()
    }

    /// The Voice call mode: the call's face (the pixel orb, Maya or the ghost in its core, the microphone's level
    /// and the voice in its rings), where the call is, its last turns, and the call's buttons.
    fn call_view(&self) -> Element<'_, Message> {
        let s = self.skin();
        let in_call = self.call.is_some();
        let talking = self.call.as_ref().is_some_and(|c| c.talking());
        let muted = self.call.as_ref().is_some_and(|c| c.muted());

        // who and how long, as a phone shows a call
        let since = self.call_started.filter(|_| in_call).map(|t| t.elapsed().as_secs()).map(|t| format!("{:02}:{:02}", t / 60, t % 60));
        let header = column![text(tr("Maya")).size(22).font(MEDIUM).color(s.text), text(since.unwrap_or_else(|| tr("Voice call").into())).size(13).color(s.faint)].spacing(4).align_x(Alignment::Center);

        // where the call is: a dot of colour and a word; anything else (loading, an error) as a plain line
        let (dot, word) = match (in_call, self.call_state.as_str()) {
            (true, _) if talking => (Some(s.accent), "Speaking".to_string()),
            (true, _) if muted => (Some(s.faint), tr("Muted").to_string()),
            (true, "Thinking") => (Some(s.warn), "Thinking".to_string()),
            (true, "Hearing you") => (Some(s.ok), "Hearing you".to_string()),
            (true, "Listening") | (true, "") => (Some(s.ok), "Listening".to_string()),
            (false, "") => (None, "Ready when you are".to_string()),
            (_, st) => (None, st.to_string()),
        };
        let status: Element<'_, Message> = match dot {
            Some(c) => container(
                row![container(Space::new().width(7).height(7)).style(move |_| container::Style { background: Some(Background::Color(c)), border: border::rounded(4), ..Default::default() }), text(word).size(13).color(s.muted)]
                    .spacing(8)
                    .align_y(Alignment::Center),
            )
            .padding([5, 12])
            .style(move |_| container::Style { background: Some(Background::Color(s.raise)), border: Border { radius: 999.0.into(), width: 1.0, color: s.line }, ..Default::default() })
            .into(),
            None => text(word).size(13).color(s.muted).into(),
        };

        // her mood and way of talking, as her last reply opened
        let mood: Element<'_, Message> = match self.call_mood.as_ref().filter(|_| in_call) {
            Some((m, t)) => {
                let cap = |w: &str| w.chars().next().map(|c| c.to_uppercase().collect::<String>() + &w[c.len_utf8()..]).unwrap_or_default();
                let line = |k: &'static str, v: String| row![text(k).size(12.5).color(s.faint), text(v).size(12.5).color(s.muted)].spacing(6);
                let mut col = column![line(tr("Current mood"), cap(m))].spacing(3).align_x(Alignment::Center);
                if !t.is_empty() {
                    col = col.push(line(tr("Way of talk"), t.clone()));
                }
                container(col).padding([10, 0]).into()
            }
            None => Space::new().height(0).into(),
        };

        let face = canvas(crate::orb::Orb { sim: &self.orb, skin: s, cell: 6.0 }).width(288).height(288);

        // live captions: the last thing you said, small, and her reply under it
        let last = |you: bool| self.call_lines.iter().rev().find(|l| l.0 == you && !l.1.is_empty()).map(|l| l.1.clone());
        let mut captions = column![].spacing(10).align_x(Alignment::Center).max_width(560);
        if let Some(t) = last(true) {
            captions = captions.push(text(t).size(13.5).color(s.faint).align_x(Alignment::Center));
        }
        if let Some(t) = last(false) {
            captions = captions.push(text(t).size(16).color(s.text).align_x(Alignment::Center));
        }

        // the controls: round buttons with a word under each
        let control = |c: char, label: &'static str, msg: Message, bg: Color, fg: Color, size: f32| -> Element<'_, Message> {
            let disc = button(container(icon(c, size * 0.32).color(fg)).center_x(size).center_y(size))
                .on_press(msg)
                .padding(0)
                .style(move |_, st| button::Style { background: Some(Background::Color(if hovered(st) { Color { a: 0.82, ..bg } } else { bg })), border: border::rounded(size / 2.0), ..Default::default() });
            column![disc, text(label).size(12).color(s.faint)].spacing(8).align_x(Alignment::Center).into()
        };
        let controls: Element<'_, Message> = if in_call {
            let (mic_bg, mic_fg) = if muted { (s.text, s.bg) } else { (s.raise2, s.text) };
            let (key_bg, key_fg) = if self.call_typing { (s.text, s.bg) } else { (s.raise2, s.text) };
            row![
                control(if muted { ic::MIC_OFF } else { ic::MIC }, if muted { tr("Unmute") } else { tr("Mute") }, Message::ToggleMute, mic_bg, mic_fg, 56.0),
                control(ic::KEYBOARD, tr("Type"), Message::ToggleCallTyping, key_bg, key_fg, 56.0),
                control(ic::HANG_UP, tr("End"), Message::EndCall, s.bad, Color::WHITE, 56.0),
            ]
            .spacing(28)
            .into()
        } else {
            control(ic::PHONE, tr("Call Maya"), Message::StartCall, s.ok, Color::WHITE, 64.0)
        };

        // who she talks with: the call model, picked here (its menu opens above it)
        let talks: Element<'_, Message> = match self.call_target() {
            Some(m) if !in_call => {
                let line = row![text(tf!("Talks with {}", "通话用 {}", m.label)).size(12.5).color(s.muted), icon(ic::DOWN, 9).color(s.faint)].spacing(6).align_y(Alignment::Center);
                container(button(line).on_press(Message::OpenMenu(Menu::CallModel)).padding([4, 10]).style(move |_, st| row_style(s, st, false))).id("call_model").into()
            }
            _ => Space::new().height(0).into(),
        };
        let mut col = column![Space::new().height(Fill), header, Space::new().height(6), talks, Space::new().height(8), status, mood, face, captions].align_x(Alignment::Center).width(Fill);
        col = col.push(Space::new().height(28)).push(controls);
        // typing instead of speaking, answered aloud the same way: a box with a send button, when asked for
        if in_call && self.call_typing {
            let field = text_input(tr("Type a message"), &self.call_text)
                .on_input(Message::CallText)
                .on_submit(Message::SendCallText)
                .size(14)
                .padding([10, 14])
                .style(move |t, st| input_style(s, t, st));
            let send = button(icon(ic::UP, 12).color(s.bg)).on_press(Message::SendCallText).padding([8, 10]).style(move |_, st| button::Style {
                background: Some(Background::Color(if hovered(st) { Color { a: 0.85, ..s.text } } else { s.text })),
                border: border::rounded(999),
                ..Default::default()
            });
            col = col.push(Space::new().height(20)).push(container(row![field, send].spacing(8).align_y(Alignment::Center)).max_width(460).width(Fill));
        }
        if in_call && self.call_timings && !self.call_timing.is_empty() {
            col = col.push(Space::new().height(14)).push(text(self.call_timing.as_str()).size(11).color(s.faint));
        }
        // a line under the call only when calls cannot work yet
        if self.voice_models.0.is_none() || self.voice_models.1.is_none() {
            let note = tr("Voice calls need the server started with --tts (a voice folder) and --stt (a Whisper file).");
            col = col.push(Space::new().height(14)).push(container(text(note).size(12).color(s.faint)).max_width(520).center_x(Fill));
        }
        col = col.push(Space::new().height(Fill));
        let screen = container(col).width(Fill).height(Fill).padding([0, 24]);
        let Some((question, yes)) = &self.call_confirm else { return screen.into() };

        // asked before a call that unloads the Code model: a card over the dimmed screen
        let shade = mouse_area(container(Space::new()).width(Fill).height(Fill).style(move |_| container::Style {
            background: Some(Background::Color(Color { a: if s.dark { 0.55 } else { 0.28 }, ..Color::BLACK })),
            ..Default::default()
        }))
        .on_press(Message::ConfirmCall(false));
        let choice = |label: &'static str, msg: Message, primary: bool| -> Element<'_, Message> {
            button(text(label).size(13).color(if primary { Color::WHITE } else { s.text }))
                .on_press(msg)
                .padding([7, 16])
                .style(move |_, st| button::Style {
                    background: Some(Background::Color(if primary { if hovered(st) { Color { a: 0.85, ..s.ok } } else { s.ok } } else if hovered(st) { s.hover } else { Color::TRANSPARENT })),
                    border: Border { radius: 8.0.into(), width: if primary { 0.0 } else { 1.0 }, color: s.line2 },
                    ..Default::default()
                })
                .into()
        };
        let buttons = row![Space::new().width(Fill), choice(tr("Cancel"), Message::ConfirmCall(false), false), choice(yes, Message::ConfirmCall(true), true)].spacing(8);
        let card = container(column![text(tr("Not enough graphics memory for both")).size(16).font(MEDIUM).color(s.text), text(question.clone()).size(13.5).color(s.muted), buttons].spacing(14))
            .padding(22)
            .max_width(440)
            .style(move |_| container::Style { background: Some(Background::Color(s.bg)), border: Border { radius: 14.0.into(), width: 1.0, color: s.line2 }, ..Default::default() });
        stack![screen, shade, container(opaque(card)).width(Fill).height(Fill).center_x(Fill).center_y(Fill)].into()
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
            let mut left = column![name, text(tr(o.note)).size(13).color(s.muted)].spacing(4).width(Fill);
            let size = text(format!("{:.1} GB", o.size as f64 / 1e9)).size(13).color(s.faint);
            let action: Element<'_, Message> = if this && busy {
                button(text(tr("Cancel")).size(13).color(s.text)).on_press(Message::CancelDownload).padding([6, 14]).style(move |_, st| pill(s, st, false)).into()
            } else if busy {
                Space::new().into()
            } else {
                let label = if this && matches!(state, "failed" | "cancelled") { tr("Resume") } else { tr("Download") };
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
                    note += &if mins <= 1.0 { tr(" · about a minute left").to_string() } else { tf!(" · {mins:.0} min left", " · 还剩 {mins:.0} 分钟") };
                }
                left = left.push(Space::new().height(6)).push(track).push(text(note).size(12).color(s.muted));
            } else if this && state == "failed" {
                let why = self.download.get("error").and_then(Json::as_str).unwrap_or(tr("The download stopped"));
                left = left.push(text(why).size(12).color(s.bad));
            }
            let line = row![left, size, action].spacing(16).align_y(Alignment::Center);
            rows = rows.push(container(line).padding([14, 16]).width(Fill).style(move |_| container::Style {
                background: Some(Background::Color(s.hover)),
                border: Border { radius: 12.0.into(), width: if s.dark { 0.0 } else { 1.0 }, color: s.line },
                ..Default::default()
            }));
        }
        let mut card = column![row![text(tr("Choose a model")).size(13).color(s.muted)].padding([0, 2]), rows].spacing(10).max_width(640).width(Fill);
        if self.own_server && !busy {
            let folder = button(row![icon(ic::FOLDER, 13).color(s.muted), text(tr("Use models already on this computer")).size(13).color(s.muted)].spacing(8).align_y(Alignment::Center))
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
        let say = if self.status.is_empty() && self.messages.is_empty() && self.api.is_some() { tr("Hello").to_string() } else { self.status_text() };
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
                (Some(_), _) => tr("Say what to do with it"),
                (None, true) => tr("How can I help you today?"),
                (None, false) => tr("Reply to ghost-infer"),
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
                // Ctrl+V: files copied in Explorer are attached; text is pasted as before
                if kp.modifiers.command() && !kp.modifiers.shift() && !kp.modifiers.alt() && kp.key.as_ref() == keyboard::Key::Character("v") {
                    return custom(Message::PasteKey);
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
        // files attached to the message: chips above the text, each with an x
        let files = (!self.attached.is_empty()).then(|| {
            row(self.attached.iter().enumerate().map(|(k, a)| self.file_chip(a.name.clone(), a.size(), Some(Message::Unattach(k))))).spacing(6).wrap()
        });
        let (inside, pad, radius): (Element<'_, Message>, Padding, f32) = if grown || files.is_some() {
            let mut col = column![].spacing(4);
            if let Some(files) = files {
                col = col.push(container(files).padding(Padding { bottom: 4.0, ..Padding::ZERO }));
            }
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
        let think = button(row![icon(ic::BOLT, 11).color(if self.thinking { s.accent } else { s.faint }), text(if self.thinking { tr("Think") } else { tr("Quick") }).size(13).color(s.muted)].spacing(6).align_y(Alignment::Center))
            .on_press(Message::ToggleThinking)
            .padding([4, 8])
            .style(move |_, st| row_style(s, st, false));
        // the model by name, as Claude shows it: a label that opens the model menu
        let name = match &self.model {
            Some(m) if self.warming => tf!("{} · warming up", "{} · 预热中", m.label),
            Some(m) => m.label.clone(),
            None => tr("Model").into(),
        };
        let model = container(button(text(name).size(13).color(s.text)).on_press(Message::OpenMenu(Menu::Model)).padding([4, 8]).style(move |_, st| row_style(s, st, false))).id("model");
        let mode = tr(crate::MODES[self.perm].1);
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
            let tip = tf!("{:.0}% of context used: {}", "上下文已用 {:.0}%：{}", 100.0 * frac, self.context_note());
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
            _ if g.checking => (ic::NOW, s.accent, tr("checking…").to_string()),
            "met" => (ic::CHECK, s.ok, tr("met").to_string()),
            "stopped" => (ic::TODO, s.faint, tf!("stopped after {} checks", "检查 {} 次后停止", g.limit)),
            _ if g.tries == 0 => (ic::NOW, s.accent, tr("working on it").to_string()),
            _ => (ic::NOW, s.accent, tf!("not met yet · {} of {} checks", "尚未达成 · {}/{} 次检查", g.tries, g.limit)),
        };
        let mut head = row![
            icon(mark, 12).color(color),
            container(text(tf!("Goal: {}", "目标：{}", g.text)).size(13).font(MEDIUM).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true),
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
            None if total == 1 => tr("The task is done").to_string(),
            None => tf!("All {total} tasks done", "{total} 项任务全部完成"),
        };
        let head = row![
            icon(if done == total { ic::CHECK } else { ic::TASKS }, 12).color(if done == total { s.ok } else { s.accent }),
            container(text(title).size(13).font(MEDIUM).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true),
            text(tf!("{done} of {total}", "{done} / {total}")).size(12.5).color(s.faint),
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
            (true, _) => tf!("{} of {} tokens", "{}/{} 词元", thousands(used), thousands(total)),
            (false, true) => tf!("{} tokens free", "剩余 {} 词元", thousands(total)),
            (false, false) => tr("Shown after the next reply").to_string(), // a chat saved before tokens were kept
        }
    }

    /// The ghost's line. While a reply runs, after what it is doing: how long the reply has taken, the tokens it
    /// has written (its rounds' counts and this round's pieces so far), and its speed (this round's while it
    /// writes, else the reply's so far), as Claude Code shows them: Writing a file · 1m 23s · 2,345 tokens · 19 tok/s
    fn status_text(&self) -> String {
        let (Some(started), true) = (self.started, self.busy.is_some()) else { return crate::i18n::status(&self.status) };
        let secs = started.elapsed().as_secs();
        let mut line = format!("{} · {}", crate::i18n::status(&self.status), if secs < 60 { format!("{secs}s") } else { format!("{}m {:02}s", secs / 60, secs % 60) });
        // the count and speed only while the model is writing: not while it waits for an OK, reads the
        // prompt or a tool runs (a round's writing ends with its "context" event)
        if self.writing_since.is_none() {
            return line;
        }
        let tokens = self.reply_tokens + self.chunks as f64;
        if tokens > 0.0 {
            line.push_str(&tf!(" · {} tokens", " · {} 词元", thousands(tokens)));
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
            // files attached to it: chips under the words (the model got them whole, inside the message)
            let (raw, files) = crate::attach::split(m.raw.as_str());
            let picked = raw.strip_prefix('/').map(|t| t.split_once(char::is_whitespace).unwrap_or((t, ""))).filter(|(name, _)| self.slash_all().iter().any(|e| e.name == *name));
            let content: Element<'a, Message> = match picked {
                Some((name, rest)) if rest.trim().is_empty() => self.chip(name),
                Some((name, rest)) => column![row![self.chip(name)], said(rest.trim_start())].spacing(6).into(),
                None => said(raw).into(),
            };
            let content: Element<'a, Message> = if files.is_empty() {
                content
            } else {
                let chips = row(files.into_iter().map(|(name, lines)| self.file_chip(name, crate::attach::lines(lines), None))).spacing(6).wrap();
                if raw.is_empty() { chips.into() } else { column![content, chips].spacing(8).into() }
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
                let thinking = live && k == last && r.answer.is_empty() && m.writing.is_none() && !m.tools.iter().any(|t| t.round == k);
                let chevron = if r.think_open { ic::DOWN } else { ic::RIGHT };
                parts.push(
                    button(row![text(tr(if thinking { "Thinking…" } else { "Thought" })).size(13.5).color(s.muted), icon(chevron, 9).color(s.faint)].spacing(7).align_y(Alignment::Center))
                        .on_press(Message::ToggleThinkView(i, k))
                        .padding([2, 0])
                        .style(move |_, _| button::Style { background: None, ..Default::default() })
                        .into(),
                );
                if r.think_open {
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
        // a tool call still being written: a line that grows with it, until it comes whole as a tool line
        if let Some(w) = m.writing.as_ref().filter(|_| live) {
            parts.push(self.writing_line(w));
        }
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
                    container(button(row![icon(ic::UNDO, 11).color(s.muted), text(tr("Undo file changes")).size(12.5).color(s.muted)].spacing(7).align_y(Alignment::Center)).on_press(Message::Undo(i)).padding([4, 10]).style(move |_, st| pill(s, st, false)))
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

    /// The line for a tool call being written: what it is (the file, once its path can be read), how much has
    /// come so far, and for how long.
    fn writing_line<'a>(&'a self, w: &'a crate::Writing) -> Element<'a, Message> {
        let s = self.skin();
        let file = w.path.as_deref().map(crate::short);
        let what = match (w.name.as_str(), file) {
            ("write_file", Some(f)) => tf!("Writing {f}…", "正在写入 {f}…"),
            ("edit_file", Some(f)) => tf!("Editing {f}…", "正在编辑 {f}…"),
            ("write_file" | "edit_file", None) => tr("Writing a file…").to_string(),
            ("", _) => tr("Writing a tool call…").to_string(),
            (n, _) => tf!("Writing a call to {n}…", "正在写 {n} 调用…"),
        };
        let secs = w.since.elapsed().as_secs();
        let size = if w.bytes > 0 { format!("{:.1} KB", w.bytes as f64 / 1024.0) } else { tf!("{} tokens", "{} 词元", w.pieces) };
        let head = row![
            icon(tool_icon(&w.name), 12).color(s.accent),
            container(text(what).size(13).color(s.text).wrapping(text::Wrapping::None)).width(Fill).clip(true),
            text(format!("{size} · {secs} s")).size(12).color(s.faint),
            icon(ic::CLOCK, 11).color(s.warn),
        ]
        .spacing(10)
        .align_y(Alignment::Center);
        container(container(head).padding([5, 8]))
            .width(Fill)
            .padding([4, 4])
            .style(move |_| container::Style { background: Some(Background::Color(s.hover)), border: border::rounded(12), ..Default::default() })
            .into()
    }

    /// The task list a reply finished, inside the reply: its steps, each checked.
    fn finished_plan<'a>(&'a self, plan: &'a [(String, String)]) -> Element<'a, Message> {
        let s = self.skin();
        let title = if plan.len() == 1 { tr("The task is done").to_string() } else { tf!("All {} tasks done", "{} 项任务全部完成", plan.len()) };
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
                container(text(crate::i18n::label(&tool_label(t))).size(13).color(s.muted).wrapping(text::Wrapping::None)).width(Fill).clip(true),
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
        card = card.push(row![answer(tr("Allow").into(), "once", true), answer(tf!("Always allow {}", "始终允许 {} 操作", tr(group_name(group))), "always", false), answer(tr("Deny").into(), "deny", false)].spacing(8));
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
