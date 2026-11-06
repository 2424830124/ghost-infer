//! The app's words, in English or Chinese.
//!
//! English is the source language: every string in the app is written in English, and `tr()` looks each up
//! in the table below when the app is set to Chinese. A string that is not in the table stays English, so a
//! missing line is never a bug.
//!
//! Three things need translating, and each has its own way in:
//! - a fixed label or sentence: `tr("Settings")`
//! - a sentence with values in it: the `tf!` macro, `tf!("Loading {}…", "正在加载 {}…", name)`
//! - a line built elsewhere as a `String` (a reply's status, a tool's name): `status()` and `label()`,
//!   which take the English line apart.

use std::sync::atomic::{AtomicU8, Ordering};

/// Which words the app shows: English, Chinese, or whatever Windows is set to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    System,
    En,
    Zh,
}

/// Whether the app draws Chinese (1) or English (0). Set once at startup and again when the setting
/// changes, so every draw reads a plain flag.
static CHINESE: AtomicU8 = AtomicU8::new(0);

pub fn set(lang: Lang) {
    let chinese = match lang {
        Lang::System => system_is_chinese(),
        Lang::En => false,
        Lang::Zh => true,
    };
    CHINESE.store(chinese as u8, Ordering::Relaxed);
}

/// True when the app is drawing Chinese.
pub fn is_zh() -> bool {
    CHINESE.load(Ordering::Relaxed) == 1
}

/// Windows' own UI language. The low byte is the primary language: 0x04 is Chinese (any of its scripts), and
/// GetUserDefaultUILanguage's 0x0804 (Simplified) and 0x0404 (Traditional) both carry it.
#[cfg(windows)]
fn system_is_chinese() -> bool {
    unsafe extern "system" {
        fn GetUserDefaultUILanguage() -> u16;
    }
    let id = unsafe { GetUserDefaultUILanguage() };
    (id & 0x3ff) == 0x04
}

#[cfg(not(windows))]
fn system_is_chinese() -> bool {
    std::env::var("LANG").map(|l| l.starts_with("zh")).unwrap_or(false)
}

/// A label or a sentence: its Chinese, when the app is set to Chinese and the line is in the table.
pub fn tr(en: &'static str) -> &'static str {
    if !is_zh() {
        return en;
    }
    match en {
        // ---- the rail ----
        "Settings" => "设置",
        "New chat" => "新建对话",
        "Search" => "搜索",
        "Search chats" => "搜索对话",
        "Recents" => "最近",
        "Recent chats" => "最近的对话",
        "This PC" => "本机",
        "No model" => "未加载模型",
        "No folder" => "未选择文件夹",
        "Other folders" => "其他文件夹",
        "Add folder…" => "添加文件夹…",
        "Add folder" => "添加文件夹",
        "Add a folder" => "添加一个文件夹",
        "Choose a folder…" => "选择文件夹…",
        "Install skill…" => "安装技能…",
        "Install skill from a link…" => "从链接安装技能…",
        "Copy as Markdown" => "复制为 Markdown",
        "Delete" => "删除",
        "Work in" => "工作于",

        // ---- settings ----
        "General" => "通用",
        "Appearance" => "外观",
        "Models" => "模型",
        "Voice" => "语音",
        "About" => "关于",
        "Startup" => "启动",
        "Language" => "语言",
        "System" => "跟随系统",
        "English" => "English",
        "中文" => "中文",
        "How ghost starts and stops." => "Ghost 的启动与退出方式。",
        "English by default. Chinese follows Windows unless you pick one." => {
            "默认英文。选「跟随系统」时，Windows 是中文就用中文。"
        }
        "Open with the last model" => "启动时打开上次的模型",
        "Loads the model you used last when ghost starts." => "Ghost 启动时加载你上次使用的模型。",
        "Warm up after loading" => "加载后预热",
        "Runs a few short replies after a model loads, so the first real answer comes quickly." => {
            "模型加载后先跑几段简短回复，好让第一个真正的回答来得更快。"
        }
        "Quitting" => "退出",
        "Quit ghost" => "退出 Ghost",
        "Quit" => "退出",
        "Quitting…" => "正在退出…",
        "Closing the window keeps ghost in the tray. This stops it and frees the model's memory." => {
            "关闭窗口只是把 Ghost 收进托盘。这里则彻底停止它，并释放模型占用的内存。"
        }
        "How ghost looks." => "Ghost 的外观。",
        "Colour theme" => "配色主题",
        "Theme" => "主题",
        "Light" => "浅色",
        "Dark" => "深色",
        "Fog is light, Graphite dark. System follows Windows." => "Fog 为浅色，Graphite 为深色。跟随系统则与 Windows 一致。",
        "Where ghost finds models." => "Ghost 从这些位置查找模型。",
        "Folders" => "文件夹",
        "Its GGUF files show in the model menus at once, and from then on." => {
            "该文件夹里的 GGUF 文件会立刻出现在模型菜单中，之后一直如此。"
        }
        "Added in ghost" => "在 Ghost 内添加",
        "Set when ghost started" => "启动 Ghost 时设定",
        "Remove" => "移除",
        "Not set up" => "未设置",
        "Voice calls with Maya." => "与 Maya 的语音通话。",
        "What Maya speaks with." => "Maya 用哪个模型说话。",
        "What turns your speech into text." => "把你的话转成文字。",
        "Call model" => "通话模型",
        "Same as Code" => "与代码模型相同",
        "During a call" => "通话期间",
        "Show response timings" => "显示回复耗时",
        "How long each reply took to start: for checking speed." => "每句回复开始前的等待时间，用来查看速度。",
        "This copy of ghost and the computer it runs on." => "这个 Ghost 副本，以及它运行的这台电脑。",
        "Running" => "运行状态",
        "Engine" => "引擎",
        "Runs the models on this computer." => "在本机运行模型。",
        "The model loaded now." => "当前加载的模型。",
        "Graphics" => "显卡",
        "Where the model runs." => "模型在哪里运行。",
        "None found" => "未检测到",
        "None loaded" => "未加载",
        "None" => "无",

        // ---- Maya's memory ----
        "You haven't talked yet." => "你们还没聊过。",
        "What she calls you" => "她对你的称呼",
        "What she calls you." => "她对你的称呼。",
        "Your name" => "你的名字",
        "Save" => "保存",
        "Nothing yet." => "还没有。",
        "None yet." => "还没有。",
        "How close you are" => "你们的亲近程度",
        "What she knows about you" => "她知道关于你的事",
        "Moments you share" => "你们的共同回忆",
        "Calls she remembers" => "她记得的通话",
        "How she sees you" => "她眼中的你",
        "What she remembers of you, from your calls. Remove anything you don't want her to know." => {
            "她通过通话记住的关于你的事。不想让她知道的可以删掉。"
        }
        "Start over" => "重新开始",
        "Forget everything" => "忘掉一切",
        "She meets you again as a stranger. The old memory is kept beside it as a backup." => {
            "她会重新把你当陌生人认识。旧记忆会留作备份。"
        }
        "You" => "你",
        "Maya" => "Maya",

        // ---- the rail's cards ----
        "Environment" => "运行环境",
        "Memory" => "记忆",
        "Free space" => "剩余空间",
        "Context" => "上下文",
        "Changes" => "改动",
        "Tasks" => "任务",
        "Skills" => "技能",
        "Tools" => "工具",
        "Messages" => "消息",
        "Instructions" => "指令",
        "Background tasks" => "后台任务",
        "Scheduled" => "已排程",
        "MCP servers" => "MCP 服务器",
        "Mode" => "模式",
        "Model" => "模型",
        "System prompt" => "系统提示词",
        "Add" => "添加",
        "Edit" => "编辑",
        "No changes" => "无改动",
        "Undo all" => "全部撤销",
        "Show more" => "显示更多",
        "Finished" => "已完成",
        "Failed" => "失败",
        "Stopped" => "已停止",
        "Running one now" => "正在运行一项",
        "Trim now" => "立即裁剪",
        "Add a memory" => "添加一条记忆",
        "every chat" => "所有对话",
        "Install" => "安装",
        "Install all" => "全部安装",
        "Install from folder" => "从文件夹安装",
        "Skill link (GitHub or .zip)" => "技能链接（GitHub 或 .zip）",
        "Edit servers" => "编辑服务器",
        "Reload" => "重新加载",
        "From Claude" => "从 Claude 导入",
        "Used" => "已用",

        // ---- the chat ----
        "Good morning" => "早上好",
        "Good afternoon" => "下午好",
        "Good evening" => "晚上好",
        "Today" => "今天",
        "Yesterday" => "昨天",
        "How can I help you today?" => "今天需要我做什么？",
        "Say what to do with it" => "说说要怎么做",
        "Reply to ghost-infer" => "回复 ghost-infer",
        "Hello" => "你好",
        "Type a message" => "输入消息",
        "Connecting…" => "连接中…",
        "Quick" => "快速",
        "Think" => "思考",
        "Thought" => "已思考",
        "Thinking…" => "思考中…",
        "Attach files…" => "附加文件…",
        "Attach text or code files" => "附加文本或代码文件",
        "Writing a file…" => "正在写入文件…",
        "Writing a tool call…" => "正在写工具调用…",
        "Mute" => "静音",
        "Unmute" => "取消静音",
        "Type" => "打字",
        "End" => "结束",
        "Voice call" => "语音通话",
        "Code" => "代码",
        "Call Maya" => "呼叫 Maya",
        "Ready when you are" => "随时可以开始",
        "Current mood" => "当前心情",
        "Way of talk" => "说话方式",
        "Hearing you" => "听见了",
        "Speaking" => "说话中",
        "Muted" => "已静音",
        "Choose a model" => "选择模型",
        "Use models already on this computer" => "使用本机已有的模型",
        "The download stopped" => "下载已中断",
        "Cancel" => "取消",
        "Download" => "下载",
        "Resume" => "继续",
        "Not enough graphics memory for both" => "显存不足以同时容纳两者",
        "The task is done" => "任务已完成",
        "Undo file changes" => "撤销文件改动",
        "Shown after the next reply" => "下一条回复后显示",
        "Show earlier" => "显示更早",
        "Create PR" => "创建 PR",
        "Allow" => "允许",
        "Deny" => "拒绝",
        "Always allow" => "始终允许",
        "file changes" => "文件改动",
        "commands" => "命令",
        "MCP tools" => "MCP 工具",
        "skill installs" => "技能安装",
        "web access" => "网络访问",

        // ---- the app's own words ----
        "Waiting for the model to load…" => "正在等待模型加载…",
        "A scheduled task is running" => "有排程任务正在运行",
        "It runs as soon as nothing else is" => "等其它任务结束后就会运行",
        "Downloading the skill…" => "正在下载技能…",
        "Installing…" => "正在安装…",
        "Compacting the chat" => "正在压缩对话",
        "Compacted the chat" => "已压缩对话",
        "Back to work on the goal" => "继续朝目标推进",
        "Checking its work" => "正在检查成果",
        "Checking the goal" => "正在核对目标",
        "Writing a file" => "正在写入文件",
        "A subagent is working" => "子代理正在工作",
        "Writing a tool call" => "正在写工具调用",
        "Waiting for the scheduled task in this chat to finish" => "正在等待本对话的排程任务结束",
        "Call ended" => "通话已结束",
        "Checking its work…" => "正在检查成果…",
        "the settings are a window of their own" => "设置是独立窗口",
        "the worker thread ended without an answer" => "工作线程结束但没有返回结果",
        "not an HTTP response" => "不是 HTTP 响应",
        "the server did not stream an answer" => "服务端没有流式返回答案",
        "the server did not stream a reply" => "服务端没有流式返回回复",
        "this server has no tool token (is it an older ghost-infer?)" => "此服务端没有工具令牌（是不是旧版 ghost-infer？）",
        "no microphone found" => "未找到麦克风",
        "no speaker found" => "未找到扬声器",
        "the microphone thread stopped" => "麦克风线程已停止",
        "the speaker thread stopped" => "扬声器线程已停止",
        "Open ghost-infer" => "打开 ghost-infer",
        "Quit and stop the model" => "退出并停止模型",
        "Thinks before it answers. Best for coding and agent work" => "先思考再回答。最适合写代码和代理任务",
        "Answers at once. Fast for code" => "立即作答。写代码速度快",
        "Small and quick, for less memory" => "小巧快速，占用内存少",
        "Keep working until what you describe is done, checked after each try" => {
            "持续工作直到你描述的目标达成，每次尝试后都会检查"
        }
        "Repeat a prompt in this chat every few minutes: /loop 10m check the build" => {
            "每隔几分钟在本对话重复一条提示：/loop 10m 检查构建"
        }


        // ---- permission modes ----
        "Auto" => "自动",
        "Ghost handles permission decisions" => "由 Ghost 决定权限",
        "Manual" => "手动",
        "Always ask before making changes" => "每次改动前都询问",
        "Accept edits" => "接受编辑",
        "Automatically accept all file edits" => "自动接受所有文件编辑",
        "Plan" => "计划",
        "Create a plan before making changes" => "改动前先写出计划",
        "Bypass permissions" => "跳过权限",
        "Accepts all permissions" => "接受所有权限",

        // ---- lines under the chat and in the rail ----
        " · about a minute left" => " · 大约还剩一分钟",
        "starting" => "启动中",
        "Nothing to trim: only the last 3 tool outputs are left" => "无需裁剪：只剩最近 3 条工具输出了",
        "Nothing to undo" => "没有可撤销的改动",
        "Undone: 1 file put back" => "已撤销：放回 1 个文件",
        "Maya is thinking back over the call…" => "Maya 正在回想这通电话…",
        "Loading the voice…" => "正在加载语音…",
        "The server has no voice: start it with --tts and --stt." => "服务端没有语音：启动时加上 --tts 和 --stt。",
        "Pick a model first: the call talks to it." => "先选一个模型：通话要用它。",
        "The voice didn't load: the server's log says why." => "语音没能加载：具体原因在服务端日志里。",
        "Unload and call" => "卸载并通话",
        "Reload and call" => "重新加载并通话",
        "Choose a folder" => "选择一个文件夹",
        "Choose the folder this chat works in" => "选择这个对话的工作文件夹",
        "Choose a folder with GGUF models" => "选择存放 GGUF 模型的文件夹",
        "Choose a folder the model may use" => "选择模型可以使用的文件夹",
        "Choose a skill archive" => "选择技能压缩包",
        "Choose a skill's folder (it has a SKILL.md)" => "选择技能文件夹（里面要有 SKILL.md）",
        "From .zip" => "从 .zip",
        ", stopped at the length limit" => "，在长度上限处停止",
        ", stopped" => "，已停止",
        "checking…" => "检查中…",
        "met" => "已达成",
        "working on it" => "正在推进",
        "Picked on the call screen. Kept loaded beside the Code model when your graphics card has room for both and the voice; otherwise a call asks before swapping them." => {
            "在通话界面里选。显卡同时装得下它、代码模型和语音时，它会和代码模型一起常驻；否则通话前会先问你。"
        }
        "The voice models are chosen when the server starts (its --tts and --stt options)." => {
            "语音模型在服务端启动时就定下了（它的 --tts 与 --stt 选项）。"
        }
        "Voice calls need the server started with --tts (a voice folder) and --stt (a Whisper file)." => {
            "语音通话要求服务端以 --tts（语音文件夹）和 --stt（Whisper 文件）启动。"
        }
        "Maya talks with" => "Maya 通话用",
        "Listening" => "聆听",

        // ---- plural-free forms of the counts ----
        "{done} of {total}" => "{done} / {total}",
        _ => en,
    }
}

/// A sentence with values in it. Both languages are written out, so `{}` finds its value either way:
///
/// ```ignore
/// tf!("Loading {}…", "正在加载 {}…", name)
/// ```
#[macro_export]
macro_rules! tf {
    ($en:literal, $zh:literal $(, $arg:expr)* $(,)?) => {
        if $crate::i18n::is_zh() { format!($zh $(, $arg)*) } else { format!($en $(, $arg)*) }
    };
}

/// The line under the chat while a reply runs, and the call's state: built into a `String` from English
/// pieces, so the Chinese is picked apart here instead of at every place that sets it.
pub fn status(en: &str) -> String {
    if !is_zh() {
        return en.to_string();
    }
    // whole lines first
    let whole = tr_match(en);
    if let Some(z) = whole {
        return z.to_string();
    }
    // then the ones with a value in them, by their opening words
    let parts: [(&str, &str); 21] = [
        ("Connecting to ", "正在连接 "),
        ("Waiting for ghost-infer at ", "正在等待 ghost-infer（"),
        ("Loading ", "正在加载 "),
        ("Warming up ", "正在预热 "),
        ("Switching to ", "正在为通话切换到 "),
        ("Couldn't switch models for the call: ", "无法为通话切换模型："),
        ("Couldn't start the call: ", "无法开始通话："),
        ("Couldn't read Maya's memory: ", "无法读取 Maya 的记忆："),
        ("Couldn't save the settings: ", "无法保存设置："),
        ("Couldn't read the memory: ", "无法读取记忆："),
        ("Couldn't send your answer: ", "无法发送你的答复："),
        ("Couldn't show the page here: ", "无法在此显示页面："),
        ("Couldn't trim: ", "无法裁剪："),
        ("Couldn't undo: ", "无法撤销："),
        ("Couldn't hear that: ", "没听清："),
        ("Couldn't install the skill: ", "无法安装技能："),
        ("Couldn't make ", "无法创建 "),
        ("cannot reach ghost-infer at ", "连接不上 ghost-infer："),
        ("[microphone: ", "[麦克风："),
        ("[speaker: ", "[扬声器："),
        ("[no tray icon: ", "[没有托盘图标："),
    ];
    for (en_head, zh_head) in parts {
        if let Some(rest) = en.strip_prefix(en_head) {
            // "Waiting for ghost-infer at {addr}…" closes a bracket instead of repeating "…"
            let tail = if en_head.ends_with(" at ") && rest.ends_with('…') {
                format!("{}）", &rest[..rest.len() - '…'.len_utf8()])
            } else {
                rest.to_string()
            };
            return format!("{zh_head}{tail}");
        }
    }
    en.to_string()
}

/// A tool's name under a reply ("Wrote notes.txt"): English verb, then what it acted on.
pub fn label(en: &str) -> String {
    if !is_zh() {
        return en.to_string();
    }
    // a call the user refused: "Edit notes.txt (you said no)"
    if let Some(head) = en.strip_suffix(" (you said no)") {
        return format!("{}（你拒绝了）", label(head));
    }
    // whole names first, so a verb below does not take the start of one ("Read an earlier chat")
    if let Some(z) = tr_match(en) {
        return z.to_string();
    }
    // names with a value inside them, not only after the verb
    if let Some(name) = en.strip_prefix("Used the ").and_then(|r| r.strip_suffix(" skill")) {
        return format!("使用技能 {name}");
    }
    if let Some(what) = en.strip_prefix("Ran a subagent: ") {
        return format!("运行子代理：{what}");
    }
    // a skill installed from a link: "Installed tidy from https://…"
    if let Some(head) = en.strip_prefix("Installed ") {
        if let Some((what, from)) = head.split_once(" from ") {
            return format!("安装 {what}（来自 {from}）");
        }
    }
    let verbs: [(&str, &str); 16] = [
        ("Wrote ", "写入 "),
        ("Write ", "写入 "),
        ("Added to ", "追加到 "),
        ("Add ", "追加 "),
        ("Edited ", "编辑 "),
        ("Edit ", "编辑 "),
        ("Ran ", "运行 "),
        ("Run ", "运行 "),
        ("Started ", "启动 "),
        ("Start ", "启动 "),
        ("Read ", "读取 "),
        ("Listed ", "列出 "),
        ("Searched the web for ", "联网搜索 "),
        ("Searched earlier chats for ", "在先前对话中搜索 "),
        ("Searched for ", "搜索 "),
        ("Installed ", "安装 "),
    ];
    for (en_head, zh_head) in verbs {
        if let Some(rest) = en.strip_prefix(en_head) {
            return format!("{zh_head}{rest}");
        }
    }
    let others = [
        ("Checked process ", "检查进程 "),
        ("Stopped process ", "停止进程 "),
        ("Remembered: ", "记住了："),
        ("Forgot: ", "忘掉了："),
        ("Used the ", "使用技能 "),
        ("Scheduled ", "排程 "),
        ("Cancelled scheduled task ", "取消排程任务 "),
        ("Ran a subagent: ", "运行子代理："),
        ("Used ", "使用 "),
    ];
    for (en_head, zh_head) in others {
        if let Some(rest) = en.strip_prefix(en_head) {
            return format!("{zh_head}{rest}");
        }
    }
    // a whole name ("Updated the task list", "Checked this computer's specs")
    tr_match(en).map(String::from).unwrap_or_else(|| en.to_string())
}

/// The table `tr()` reads, for lines that reach it as a `&str` rather than as a literal (a tool's whole name,
/// a status that has no value in it).
fn tr_match(en: &str) -> Option<&'static str> {
    let pairs: [(&str, &str); 31] = [
        ("Read an earlier chat", "读取一段先前的对话"),
        ("Listed the folder", "列出文件夹"),
        ("Sent back to work", "被要求继续工作"),
        ("Asked to check its work", "被要求检查成果"),
        ("Compacted the chat to fit the context", "压缩对话以放进上下文"),
        ("Wrote a tool call the length limit cut off", "写的工具调用被长度上限截断"),
        ("Wrote a tool call that could not be read", "写的工具调用无法读取"),
        ("No model loaded", "未加载模型"),
        ("Warmed up and ready", "预热完成，准备就绪"),
        ("Ready when you are", "随时可以开始"),
        ("Thinking", "思考中"),
        ("Thinking…", "思考中…"),
        ("Listening", "聆听中"),
        ("Hearing you", "听见了"),
        ("Speaking", "说话中"),
        ("Reading", "读取中"),
        ("Reading the results", "读取结果"),
        ("Writing", "撰写中"),
        ("Using a tool", "使用工具"),
        ("Waiting for your OK", "等你确认"),
        ("Running", "运行状态"),
        ("Compacting the chat", "正在压缩对话"),
        ("Compacted the chat", "已压缩对话"),
        ("Back to work on the goal", "继续朝目标推进"),
        ("Checking its work", "正在检查成果"),
        ("Checking the goal", "正在核对目标"),
        ("A subagent is working", "子代理正在工作"),
        ("Writing a tool call", "正在写工具调用"),
        ("Writing a file", "正在写入文件"),
        ("Updated the task list", "更新任务列表"),
        ("Checked this computer's specs", "查看本机配置"),
    ];
    pairs.iter().find(|(a, _)| *a == en).map(|(_, z)| *z)
}
