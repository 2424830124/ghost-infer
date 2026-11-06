<p align="center">
  <img src="docs/ghost.png" width="132" alt="The ghost-infer ghost">
</p>

<h1 align="center">ghost-infer</h1>

<p align="center">
  <b>Fast local LLM inference for the GPU you already have.</b><br>
  One small Rust program that runs GGUF models on an NVIDIA card, serves an OpenAI-compatible API,<br>
  and comes with a desktop app where the model works on your files.
</p>

<p align="center">
  <a href="#install"><b>Install</b></a> ·
  <a href="#whats-new">What's new</a> ·
  <a href="#speed">Speed</a> ·
  <a href="#models">Models</a> ·
  <a href="docs/desktop-app.md">Desktop app</a> ·
  <a href="docs/usage.md">Server API</a> ·
  <a href="docs/how-it-works.md">How it works</a>
</p>

<p align="center">
  <a href="https://www.paypal.com/ncp/payment/WPSFYDSR9HVSY"><img src="https://img.shields.io/badge/Support_ghost--infer-PayPal-00457C?logo=paypal&logoColor=white" alt="Support ghost-infer with PayPal"></a>
</p>

<p align="center">
  <img src="docs/demo.gif" width="860" alt="The ghost-infer desktop app finding and fixing a bug, then running the tests">
  <br><sub>Real time, not sped up: the desktop app fixing a failing test, Qwen3.6-35B-A3B (Huihui abliterated, i1-Q3_K_M) on an RTX 2060 SUPER (8 GB) with 16 GB of RAM.</sub>
</p>

Most inference engines are built for data centres: many users, big GPUs, batching. ghost-infer is built
for the other case: **one person, one consumer GPU, one model at a time**, and it is tuned for exactly that.

## What's new

Since 0.2.3:

- **Long chats stay fast on Qwen3.6-35B-A3B.** Its attention (256-wide heads) now runs on the tensor cores,
  both when it writes and when it reads a prompt. On an RTX 2060 SUPER, generation at a 16K context went from
  25.2 to 39.7 tokens/s and at 64K from 12.0 to 40.8; reading a 128K-token prompt takes about 4 minutes
  instead of 27 (81 to 581 tokens/s). `GHOST_INFER_NO_TC_DECODE=1` and `GHOST_INFER_NO_TC_ATT=1` turn the new
  kernels off.
- **Attach files in the desktop app.** Text and code files go with a message: from the + menu, dropped on the
  window, or pasted with Ctrl+V after copying them in Explorer.
- **See a file being written.** While the model writes a file, the reply shows a live line with its name, how
  much is written and for how long, instead of going quiet until the file is done.
- **Smaller fixes in the app.** Each thinking block opens and closes on its own, and the token count and
  speed show only while the model is writing (not while it waits for your OK).

## Install

**[Download GhostSetup for Windows](https://github.com/affifuddin-hazam/ghost-infer/releases/latest)** and run
it. It needs no administrator rights and nothing else installed. The app starts the server and, the first
time, offers models to download, or uses the GGUF files you already have.

<details>
<summary>Build from source</summary>

```bash
git clone https://github.com/affifuddin-hazam/ghost-infer
cd ghost-infer
cargo build --release
target\release\ghost-infer serve models\Qwen3-8B-Q4_K_M.gguf --gpu
```

Open <http://127.0.0.1:8080> to chat, or point any OpenAI client at `http://127.0.0.1:8080/v1`. Built from
source, it needs the CUDA toolkit's NVRTC: see [Building](docs/building.md).

</details>

## Speed

On an 8 GB RTX 2060 SUPER with 16 GB of RAM, on the same GGUF files:

| Generation speed | ghost-infer | llama.cpp | Ollama |
|---|---|---|---|
| Qwen3 8B, 5 GB: fits in the GPU | **70.3 tokens/s** | 64.8 | 66.0 |
| Qwen3-30B-A3B, 13.7 GB: bigger than the GPU | **38.0 tokens/s** | 23.7 | 21.7 |
| Qwen3-Coder-30B-A3B, 14.7 GB \* | **32.7 / 47.3** | not measured | not measured |
| Qwen3.6-35B-A3B, 16.8 GB \* | **31.9 / 31.9** | 26.0 / 24.9 | not measured |
| Qwen3-Coder-Next 80B, 22 GB: bigger than the RAM \* | **11.8 / 14.5** | 9.3 / 12.4 | not measured |

<sub>llama.cpp b11747 and Ollama 0.40.13. The first two rows: GuideLLM and seven varied prompts, llama.cpp at
its best `--n-cpu-moe` setting. Rows marked \*: tokens/s writing code / writing a long answer, the second (warm)
of two runs, llama.cpp with its automatic GPU split. Method and more results in [Benchmarks](docs/benchmarks.md).</sub>

## What you get

- **One small program.** About 3 MB with two Rust dependencies: no Python, no Docker. It compiles its GPU
  code for your card on first start.
- **Models bigger than your GPU.** Mixture-of-experts models keep their experts in RAM and the ones a reply
  is using in VRAM, and an 80B model runs on an 8 GB card.
- **A desktop app with an agent.** The model reads and edits your files and runs commands, asking first,
  with permission modes, a task list and background jobs. [More](docs/desktop-app.md)
- **Voice calls.** Talk with Maya, a companion who answers aloud and remembers your calls, with her own
  model beside the coding one when the GPU has room. Speech and listening run on the same card. The voice
  files (about 3 GB) are on [Hugging Face](https://huggingface.co/affifuddin-hazam/ghost-voice).
  [More](docs/desktop-app.md#voice-calls)
- **Drop-in server.** OpenAI-compatible `/v1/chat/completions` with streaming and tool calls, so editor
  plugins, chat apps and agent harnesses work unchanged. [More](docs/usage.md)
- **Exact where it matters.** Speculative decoding gives the same output as plain greedy decoding, and
  results are checked against exact maths and llama.cpp. [More](docs/how-it-works.md)

<p align="center">
  <img src="docs/demo-voice.gif" width="520" alt="A voice call in the desktop app: Maya answers aloud while the pixel orb moves with her voice">
  <br><sub>A voice call, real time: Impish_LLAMA_3B, the voice (maya1 with a voice fine-tune of our own,
  <a href="https://huggingface.co/affifuddin-hazam/ghost-voice">on Hugging Face</a>) and Whisper turbo together on an 8 GB RTX 2060 SUPER. <a href="docs/demo-voice.mp4">With sound</a>.</sub>
</p>

## Models

The app offers these on its first start:

| Model | Download | Good for |
|---|---|---|
| Qwen3.6-35B-A3B | 16.8 GB | Thinks before it answers: coding and agent work (16 GB of RAM) |
| Qwen3-Coder-30B-A3B | 14.7 GB | Answers at once: fast for code |
| Qwen3-4B-Instruct-2507 | 2.5 GB | Computers with less memory |

Other GGUF files of Llama 3, Mistral Nemo, Qwen3, Qwen3-Next, Qwen3.5 / 3.6 and DeepSeek-style models such as
GLM-4.7-Flash run too, in Q3_K to Q8_0 or the 2-bit IQ2 formats: [the full list](docs/building.md#supported-models).
3-bit files stay 3-bit on the GPU, so a 12B Q3_K_M runs on an 8 GB card.

## Requirements

Windows 10 or 11, an NVIDIA GPU from the GTX 10 series on with a current driver, and a CPU with AVX2.

## Documentation

- [Benchmarks](docs/benchmarks.md): the method, long contexts, mixture of experts, speculative decoding
- [The desktop app and the chat page](docs/desktop-app.md)
- [Command line and server API](docs/usage.md)
- [How it works](docs/how-it-works.md), and its accuracy
- [Building, requirements and models](docs/building.md): from source, GPU requirements, the installer

## Roadmap

- Linux support
- More architectures (Gemma) and formats (Q4_0, the other IQ quants)
- Constrained decoding for tool calls

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work
by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional
terms or conditions.

The chat page includes [three.js](https://threejs.org) (MIT, [`src/vendor/three.LICENSE`](src/vendor/three.LICENSE))
and the [Instrument Sans](https://github.com/Instrument/instrument-sans) typeface (SIL Open Font License 1.1,
[`src/vendor/instrument-sans.LICENSE`](src/vendor/instrument-sans.LICENSE)). The installer includes NVIDIA's
NVRTC under the CUDA Toolkit EULA ([`installer/THIRD-PARTY.txt`](installer/THIRD-PARTY.txt)).
