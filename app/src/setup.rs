//! What an installed app does on its own: start its server (ghost-infer.exe beside it) when nothing answers
//! yet, and offer models to download when the models folder is empty (the user picks; none is pushed).

use std::path::PathBuf;
use std::time::Duration;

/// A model the first run offers, from Hugging Face.
pub struct Offer {
    pub name: &'static str,
    pub note: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub size: u64,
}

pub const OFFERS: [Offer; 3] = [
    Offer {
        name: "Qwen3.6 35B",
        note: "Thinks before it answers. Best for coding and agent work",
        file: "Qwen3.6-35B-A3B.i1-Q3_K_M.gguf",
        url: "https://huggingface.co/mradermacher/Qwen3.6-35B-A3B-i1-GGUF/resolve/main/Qwen3.6-35B-A3B.i1-Q3_K_M.gguf",
        size: 16_764_764_896,
    },
    Offer {
        name: "Qwen3 Coder 30B",
        note: "Answers at once. Fast for code",
        file: "Qwen3-Coder-30B-A3B-Instruct.i1-Q3_K_M.gguf",
        url: "https://huggingface.co/mradermacher/Qwen3-Coder-30B-A3B-Instruct-i1-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct.i1-Q3_K_M.gguf",
        size: 14_711_850_016,
    },
    Offer {
        name: "Qwen3 4B",
        note: "Small and quick, for less memory",
        file: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        size: 2_497_281_120,
    },
];

/// %LOCALAPPDATA%\ghost-infer: the server's and the app's data.
fn data_dir() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer"))
}

/// Where downloaded models go.
pub fn models_dir() -> Option<PathBuf> {
    data_dir().map(|d| d.join("models"))
}

/// The server beside this program, if the app was installed with it.
pub fn own_server() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let server = exe.with_file_name("ghost-infer.exe");
    server.exists().then_some(server)
}

/// Whether something answers at `addr`.
pub fn listening(addr: &str) -> bool {
    use std::net::ToSocketAddrs;
    addr.to_socket_addrs().ok().and_then(|mut a| a.next()).is_some_and(|a| std::net::TcpStream::connect_timeout(&a, Duration::from_millis(300)).is_ok())
}

/// Starts the server beside this program, without a window, if nothing answers at `addr` yet: the models
/// folder and `extra` folders, and the last picked model if the app is set to start with it.
pub fn start_server(addr: &str, extra: &[String], last: Option<&str>) {
    let (Some(server), Some(data), Some(models)) = (own_server(), data_dir(), models_dir()) else { return };
    if listening(addr) {
        return;
    }
    let _ = std::fs::create_dir_all(&models);
    let port = addr.rsplit(':').next().unwrap_or("8080").to_string();
    let mut dirs = vec![models];
    dirs.extend(extra.iter().map(PathBuf::from));
    let mut args: Vec<std::ffi::OsString> = vec!["serve".into()];
    if let Some(path) = last.and_then(|id| dirs.iter().map(|d| d.join(format!("{id}.gguf"))).find(|p| p.exists())) {
        args.push(path.into());
    }
    // 64K: the context's memory is taken only as a chat grows (from the experts kept on the GPU), and 32K left a
    // skill, a web page and its rewrite compacting round after round
    for a in ["--gpu", "--ctx", "65536", "--port", &port] {
        args.push(a.into());
    }
    for d in &dirs {
        args.push("--models".into());
        args.push(d.into());
    }
    // voice calls: a CSM voice folder and a Whisper file found in the models folders
    if let Some(voice) = dirs.iter().flat_map(|d| std::fs::read_dir(d).into_iter().flatten().flatten()).map(|e| e.path()).find(|p| ["backbone.gguf", "decoder.gguf", "mimi.gguf"].iter().all(|f| p.join(f).is_file())) {
        args.push("--tts".into());
        args.push(voice.into());
    }
    let whisper = |d: &PathBuf| std::fs::read_dir(d).into_iter().flatten().flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|x| x == "gguf") && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.to_lowercase().starts_with("whisper")));
    if let Some(stt) = dirs.iter().find_map(|d| whisper(&d.join("whisper")).or_else(|| whisper(d))) {
        args.push("--stt".into());
        args.push(stt.into());
    }
    let log = |name: &str| std::fs::File::create(data.join(name)).map(std::process::Stdio::from).unwrap_or_else(|_| std::process::Stdio::null());
    use std::os::windows::process::CommandExt;
    let _ = std::process::Command::new(server)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(log("server-out.log"))
        .stderr(log("server.log"))
        .creation_flags(0x0800_0000) // no console window; it goes on when the app closes, as the launcher's did
        .spawn();
}

/// "6.2 of 16.8 GB".
pub fn progress_text(bytes: f64, total: f64) -> String {
    format!("{:.1} of {:.1} GB", bytes / 1e9, total / 1e9)
}
