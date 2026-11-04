# The desktop app and the chat page

[Back to the README](../README.md)

## The desktop app and agent

`ghost-app` is a native window onto the server, drawn on the CPU so it takes no GPU memory from the model.
The server runs the agent: the model works on your files and your computer with tools, round after
round, asks before anything that changes something, and keeps every chat in `%LOCALAPPDATA%\ghost-infer`.

<p align="center">
  <img src="screenshot-app.png" width="860" alt="The desktop app: a bug found and fixed, its plan done, and the rail with the model, the context, the tasks, the changed file and the scheduled tasks">
</p>

The same task with Qwen3-Coder, which answers without thinking first:

<p align="center">
  <img src="demo-coder.gif" width="860" alt="The desktop app with Qwen3-Coder-30B-A3B finding and fixing the same bug, then running the tests">
  <br><sub>Real time, not sped up: the desktop app fixing a failing test, Qwen3-Coder-30B-A3B (Huihui abliterated, i1-Q3_K_M) on an RTX 2060 SUPER (8 GB) with 16 GB of RAM.</sub>
</p>

```bash
cargo build --release -p ghost-app
target\release\ghost-infer serve models\Qwen3-8B-Q4_K_M.gguf --gpu
target\release\ghost-app                # connects to 127.0.0.1:8080; give another host:port to change it
```

The app runs on the same computer as the server: the agent only answers requests from this computer.
Installed, the app starts its own server (`ghost-infer.exe` beside it) when none is running, and with no
model yet it offers models to download from Hugging Face, or reads a folder of GGUF files already on the
computer. With models but none running, sending opens the model menu instead.

- **A folder per chat.** A new chat works in the folder shown under the message box (click it to pick
  another), and keeps it, as a Claude Code session does: relative paths and commands start there. Folders
  added with the + menu are open to every chat as well; file tools work nowhere else.
- **Permission modes, as Claude Code has them.** The tools (files, PowerShell, the web, MCP) are always
  there; the mode next to the context ring decides what waits for your OK. **Manual** asks before every
  change, command and web use; **Accept edits** lets file changes through; **Auto** lets file changes, the
  web and ordinary commands through and asks about risky ones (deleting, `git push`, installing, system
  settings); **Plan** changes nothing and answers with a plan; **Bypass permissions** asks nothing.
  Shift+Tab steps through them. An OK can be once or always, and every reply's file changes can be undone.
- **Plans and checks.** The model keeps a task list, pinned above the message box. After changing code it
  is sent back once to run it.
- **Loops, as Claude Code has them.** `/goal all tests pass, stop after 5 tries`: when the model stops, a
  check reads the work against the goal and sends it back with what is missing, until the goal is met or
  the tries run out (8 unless the goal says). `/loop 10m check the build and fix what fails` repeats a
  prompt in the chat; `/loop stop` ends it.
- **Long chats fit, compacted as Claude Code does.** When a chat outgrows the model's context, the model
  writes a summary of it (what you asked, what was done, where it stopped) and works on from that; the
  chat file keeps everything, and the reply shows "Compacted the chat" with the summary. If no summary
  comes, old tool outputs, old file writes and then the oldest turns are taken out instead.
- **Subagents, as Claude Code has them.** The model can hand a self-contained task (reviewing a long
  file against guidelines, searching many files) to a subagent: it works in a fresh context of its own
  with the same tools, its calls and approvals show in the reply, and only its report comes back, so its
  reading never fills the chat's context.
- **Big outputs go to a file.** A tool output too long for the context (an eighth of it: 8,000
  characters at 64K) is saved under `%LOCALAPPDATA%\ghost-infer\outputs`, and the model gets its start and
  the path, to read on in if it needs more. A file read again unchanged is not sent again either.
- **One reply at a time.** A message in another chat waits for the reply going to end, so neither has to
  read its whole prompt again.
- **Progress as it works.** The ghost's line says what the model is doing, how long the reply has taken,
  the tokens written and the speed: "Writing a file · 1m 23s · 2,345 tokens · 19 tok/s".
- **Instructions and memory.** `GHOST.md` goes into every chat: the one in `%LOCALAPPDATA%\ghost-infer`,
  and one in each folder you add (a folder's `CLAUDE.md` or `AGENTS.md` works too). The model saves
  lasting facts with `remember` and can search earlier chats.
- **Skills and MCP servers.** Install [Agent Skills](https://agentskills.io) (a folder with a `SKILL.md`)
  from a folder, a .zip or a GitHub link; a repository with several skills lets you pick. MCP servers go in
  `mcp.json` in Claude Desktop's format, and **From Claude** copies the ones Claude Desktop has.
- **Slash commands.** Type `/` in the message box for the commands and installed skills: the arrows move,
  Tab or Enter picks, Esc closes. At the start of the box a pick (or a whole name and a space) becomes a
  chip before your text, a wand for a skill, and Backspace takes it away; further on, a skill's /name is
  styled in the text ("make a page with /design-taste-frontend"). Every skill named this way is loaded
  before the model starts.
- **Scheduled tasks.** Ask in a chat ("remind me in 20 minutes to…", "every weekday at 9:00, check the
  build"): the model schedules it once, every N minutes or daily at a time, and it runs in that chat even
  while you are away. Its reply waits there, marked unread.
- **Web pages, shown as they are made.** A page the model writes opens in a pane beside the chat (Edge's
  WebView2, built into Windows) and shows again after every change to it, its stylesheet or its script;
  close it, reload it or open it in your browser from its bar. The server serves only that page's folder,
  only to this computer. Another client gets the page in a browser MCP server's window instead (Playwright's,
  if `mcp.json` has it).
- **The tray.** The window's X (or Alt+F4) hides ghost to its icon by the clock, so the model and a reply
  under way go on: click the icon, or start the app again, to bring it back. **Quit and stop the model**
  (the icon's right-click menu, or the settings menu) ends both. Windows 10 may put a new icon behind the
  ^ at first; drag it onto the taskbar to keep it in sight.
- **Warm-up.** After a model loads, the server runs a few short replies so its weights are in memory before
  your first message, which then goes first if it comes sooner. The ghost says "Warming up…" (and the model
  name "warming up") until it is done, then "Warmed up and ready". **Warm up after loading** in the settings
  menu turns it off.

Scheduled tasks and loops wait until nothing else has used the model for 20 seconds, and use only what
you set to always allowed, since nobody is there to ask.

| In `%LOCALAPPDATA%\ghost-infer` | |
|---|---|
| `chats\` | every chat (with its folder and goal), for the app and any other client of the agent |
| `harness.json` | the folders every chat may use, and what is always allowed |
| `GHOST.md`, `memory.json` | instructions and remembered facts, for every chat |
| `skills\`, `mcp.json` | installed skills and MCP servers |
| `schedules.json` | scheduled tasks and loops |
| `app.json` | the app's own settings (theme, permission mode, the folder new chats work in, the call model) |
| `maya\` | Maya's persona (`persona.md`, yours to edit) and what she remembers of your calls (`memory.json`) |

## Voice calls

The switch at the top left turns the app into a phone call with **Maya**, a companion who talks rather than
works: you speak (or type), she answers aloud, a sentence at a time as the model writes it.

<p align="center">
  <img src="demo-voice.gif" width="520" alt="A voice call in the desktop app: a typed question, and Maya's answer spoken while the pixel orb moves with her voice">
  <br><sub>Real time: a call with Impish_LLAMA_3B on an RTX 2060 SUPER (8 GB), with the voice (maya1 with a voice fine-tune
  of our own, not included: plain maya1 sounds different) and Whisper turbo beside it on the same card.
  <a href="demo-voice.mp4">With sound</a>.</sub>
</p>

- **Speaking and listening.** The microphone is cut into what you say by loudness against the room's noise,
  Whisper turns it into text, and Maya's reply is spoken as it is written. Speaking while she talks cuts her
  off. **Type** sends text instead, answered aloud the same way; **Mute** stops the microphone.
- **She remembers.** After each call the model thinks it back over and updates what Maya knows about you,
  what the calls were about and moments worth bringing up, so the first call is with a stranger and later
  ones with someone she knows. Her persona is `maya\persona.md`; the memory is `maya\memory.json`, plain JSON
  you can edit or empty.
- **Her own model.** "Talks with …" under her name picks the model calls use, "Same as Code" by default. A
  small chatty one (3B to 4B) answers fastest and fits beside the voice; a coding model is often too formal.
- **One card, two models.** When the GPU has room for the Code model, the call model and the voice together
  (each model with a working context), all three stay loaded and a call starts at once. When it has not,
  **Call Maya** asks before it unloads the Code model (or reloads it with a smaller context), and switching
  back to Code brings it back.

The voice needs two files the server is started with:

```bash
# speech: maya1 (a GGUF of maya-research/maya1) and the SNAC decoder in one folder
python scripts/snac-convert.py C:\models\maya1\snac.gguf
# listening: Whisper large-v3-turbo as one GGUF
python scripts/whisper-convert.py C:\models\whisper-turbo.gguf
ghost-infer serve --gpu --models C:\models --tts C:\models\maya1 --stt C:\models\whisper-turbo.gguf
```

Installed, the app finds them in its models folders by themselves: a folder with `snac.gguf` (or CSM's three
files) and a `whisper*.gguf`. A `voice.txt` in the speech folder holds maya1's voice description (a
fine-tune's own, if it was trained with one). Sesame's CSM works as the speech folder too (`scripts/csm-convert.py`). Together the voice and
Whisper take about 3.5 GB of GPU memory, loaded only while calls may need them.

## The chat page

`ghost-infer serve` includes a chat page, compiled into the program and working offline:

<p align="center">
  <img src="screenshot-desktop.png" width="860" alt="The chat page: a conversation with code, the ghost on the message box">
</p>

- **A ghost that shows what the model is doing**: it listens while you type, thinks while the model reads
  the prompt or reasons in `<think>`, talks while the reply streams, reaches for a magnifier on tool calls,
  and changes face with the model's mood (the model is asked to tag its replies; the tags stay hidden).
- **Built for coding too**: code blocks with their language and a copy button, tables, copy and
  write-again on every reply, and a reply that keeps streaming while you scroll back to read.
- **Saved conversations** in the browser, grouped by day.
- **Model picker**: lists the GGUF files next to the running model; choosing one swaps it on the GPU.
- **Thinking switch** for reasoning models, and settings for temperature, reply length and system prompt.
- **A tool to check this computer**: the model can call `get_system_info` for the CPU, RAM, GPU, drives and
  what is running, and the reply shows what it looked at. It only reads; a setting turns it off.
- Live speed while a reply streams, and per-reply timing under each answer.
- Works on a phone too: start the server with `--host 0.0.0.0` and open it from another device on your network.

<p align="center">
  <img src="screenshot-mobile.png" width="260" alt="The chat page on a phone">
</p>
