//! The ghost by the clock (Windows' notification area). The window's X hides the window there instead of
//! quitting, so an accidental click loses nothing: a click on the icon brings it back, and its menu quits
//! (stopping the model too). Starting the app again while it runs shows this one, as the tray apps Discord and
//! Oryxis do: the first app owns a named event, a second one sets it and leaves.

use crate::i18n::tr;
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use iced::futures::channel::mpsc;
use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::{MouseButton, TrayIcon, TrayIconBuilder, TrayIconEvent};

/// What the icon, or a second start of the app, asks for.
#[derive(Debug, Clone)]
pub enum Ask {
    Show,
    Quit,
}

const OPEN: &str = "open";
const QUIT: &str = "quit";

thread_local! {
    // the icon stays while it is held; only the main thread, which made it, touches it
    static ICON: RefCell<Option<TrayIcon>> = const { RefCell::new(None) };
}

/// The named event a second start sets (0: none made).
static SHOW_EVENT: AtomicUsize = AtomicUsize::new(0);

unsafe extern "system" {
    fn CreateEventW(attributes: *const c_void, manual_reset: i32, initial: i32, name: *const u16) -> *mut c_void;
    fn SetEvent(event: *mut c_void) -> i32;
    fn CloseHandle(handle: *mut c_void) -> i32;
    fn WaitForSingleObject(handle: *mut c_void, ms: u32) -> u32;
    fn GetLastError() -> u32;
}

/// True when ghost-app already runs for this server: it was asked to show its window, and this start should
/// end. Otherwise this app owns the event from now on.
pub fn already_running(addr: &str) -> bool {
    const ERROR_ALREADY_EXISTS: u32 = 183;
    // a name may not have a backslash after Local\
    let name: Vec<u16> = format!("Local\\ghost-app-{}", addr.replace('\\', "/")).encode_utf16().chain([0]).collect();
    let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, name.as_ptr()) };
    if event.is_null() {
        return false;
    }
    if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        unsafe {
            SetEvent(event);
            CloseHandle(event);
        }
        return true;
    }
    SHOW_EVENT.store(event as usize, Ordering::Relaxed);
    false
}

/// Puts the icon by the clock: once, from the main thread (Windows sends the icon's clicks to the thread that
/// made it, and iced's event loop is the one that reads them).
pub fn install(rgba: Vec<u8>, side: u32) {
    let menu = Menu::new();
    let _ = menu.append(&MenuItem::with_id(OPEN, tr("Open ghost-infer"), true, None));
    let _ = menu.append(&MenuItem::with_id(QUIT, tr("Quit and stop the model"), true, None));
    let mut icon = TrayIconBuilder::new().with_tooltip("ghost-infer").with_menu(Box::new(menu)).with_menu_on_left_click(false);
    if let Ok(i) = tray_icon::Icon::from_rgba(rgba, side, side) {
        icon = icon.with_icon(i);
    }
    match icon.build() {
        Ok(t) => ICON.with(|c| *c.borrow_mut() = Some(t)),
        Err(e) => eprintln!("[no tray icon: {e}]"),
    }
}

/// The icon's clicks and menu, and second starts of the app, as they come.
pub fn events() -> mpsc::UnboundedReceiver<Ask> {
    let (tx, rx) = mpsc::unbounded();
    let (click, menu) = (tx.clone(), tx.clone());
    // the button going down as well as up: in Windows 10's panel of hidden icons the panel closes as the
    // button comes up, and tray-icon drops that one (it cannot find the icon any more); showing twice does no harm
    TrayIconEvent::set_event_handler(Some(move |e| {
        if let TrayIconEvent::Click { button: MouseButton::Left, .. } | TrayIconEvent::DoubleClick { button: MouseButton::Left, .. } = e {
            let _ = click.unbounded_send(Ask::Show);
        }
    }));
    MenuEvent::set_event_handler(Some(move |e: MenuEvent| {
        let _ = menu.unbounded_send(if e.id.0 == QUIT { Ask::Quit } else { Ask::Show });
    }));
    let event = SHOW_EVENT.load(Ordering::Relaxed);
    if event != 0 {
        std::thread::spawn(move || {
            // set by a second start: auto-reset, so each one shows the window once
            while unsafe { WaitForSingleObject(event as *mut c_void, u32::MAX) } == 0 {
                if tx.unbounded_send(Ask::Show).is_err() {
                    break;
                }
            }
        });
    }
    rx
}
