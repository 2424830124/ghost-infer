# Command line and server API

[Back to the README](../README.md)

## Usage

```bash
ghost-infer serve model.gguf --gpu                  # server + chat page on http://127.0.0.1:8080
ghost-infer chat  model.gguf --gpu                  # chat in the terminal
ghost-infer run   model.gguf -p "Explain DNS." --gpu
ghost-infer bench model.gguf --prompt-len 256 -n 128 --gpu
ghost-infer run   Qwen3-8B-Q4_K_M.gguf -p "Explain DNS." -t 0 --gpu --draft Qwen3-0.6B-Q8_0.gguf
```

| Command | What it does |
|---|---|
| `serve <model>` | OpenAI-compatible HTTP server with the chat page |
| `chat <model>` | interactive chat (`/reset` clears the conversation, empty line or `/bye` quits) |
| `run <model> -p <prompt>` | one reply, streamed, with timing |
| `bench <model>` | fixed workload: `--prompt-len` prompt tokens, then `-n` greedy tokens, best of 3 |
| `top <model> <text> [--gpu]` | the 10 most likely next tokens after raw text, with log-probabilities |
| `tokenize <model> <text>` | token ids and pieces, plus a round-trip check |
| `inspect <model>` | metadata, tensor list and parameter counts |
| `meta <model> <key>` | one metadata value in full (e.g. `tokenizer.chat_template`) |
| `dequant <model> <tensor> <row>` | one row of a weight tensor as floats (for checking formats) |

<details>
<summary>Options for <code>run</code>, <code>chat</code>, <code>serve</code> and <code>bench</code></summary>

| Option | Default | |
|---|---|---|
| `--gpu` | | run on the NVIDIA GPU |
| `-n`, `--max <n>` | 256 | max new tokens per reply |
| `-t`, `--temp <f>` | 0.7 | temperature; `0` = greedy |
| `--top-k <n>` | 40 | keep the k most likely tokens (`0` = all) |
| `--top-p <f>` | 0.9 | nucleus sampling cut-off |
| `--seed <n>` | time | sampling seed |
| `--ctx <n>` | `run`, `bench`: 2048; `chat`, `serve`: as much as fits | context window in tokens. On the GPU `chat` and `serve` take the longest that fits in the free memory, keeping 384 MB spare (on Windows, going over does not fail: the driver quietly moves memory to system RAM and everything slows down); on the CPU 4096 |
| `--kv <f16\|int8>` | int8 (`GHOST_INFER_KV`) | GPU KV cache format: int8 holds about twice the context of f16 |
| `--draft <gguf>` | | speculative decoding with this small model (`--gpu`, greedy replies; same tokenizer needed) |
| `--draft-k <n>` | 3 | tokens the draft proposes per step (1-7) |
| `--system <text>` | template's | system prompt |
| `--no-think` | | reasoning models answer without thinking first (requests can still ask for it) |
| `--raw` | | send the prompt as-is, without the chat template |
| `--threads <n>` | physical cores | CPU threads (generation is memory-bound, so hyper-threads don't help) |
| `--special` | | print special tokens |
| `--host <addr>` | 127.0.0.1 | `serve`: address to listen on (`0.0.0.0` for other machines on your network) |
| `--port <n>` | 8080 | `serve`: port |
| `--models <dir>` | | `serve`: another folder of GGUF files for the model picker, besides the model's own (repeatable) |
| `--tts <dir>` | | `serve`: a speech folder, maya1's (its GGUF and `snac.gguf`) or CSM's, for `/v1/audio/speech` and the app's voice calls ([Voice calls](desktop-app.md#voice-calls)) |
| `--stt <file>` | | `serve`: a Whisper file for `/v1/audio/transcriptions`; with `--tts`, the app's voice calls |

</details>

## Server API

Point any OpenAI client at `http://127.0.0.1:8080/v1` (the API key and model name can be anything):

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | `messages` (system / user / assistant / tool, text content), `tools`, `tool_choice` (`"none"` hides the tools), `stream`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop`, `stream_options.include_usage`, `chat_template_kwargs.enable_thinking` |
| `POST /v1/completions` | the same with a raw `prompt` instead of messages (no chat template) |
| `GET /v1/models` | the loaded model (its file name) and its `context_length` |
| `GET /api/models` | the GGUF files in the model's folder, the current model, one being loaded, and for calls: `voice` (loaded), `call_model` (the model a call talks to now) and `together` (the chat and call models both loaded) |
| `POST /api/models/load` | `{"id": "<file name without .gguf>"}`: switch to that model; answers once it is ready. `"voice": true` loads the speech and Whisper models beside it. `"call": "<model>"` keeps a call model and the voice loaded beside it when they all fit; otherwise `"for_call": true` loads the call model with the voice, else the model alone |
| `GET`/`POST /api/models/folders` | the folders the model list comes from; `{"add": "<folder>"}` or `{"remove": "<folder>"}` changes them while the server runs (the app's Settings, Models; local requests with the page's token only) |
| `POST /v1/audio/speech` | `input` (text), `response_format` (`wav` or streamed `pcm`): speech in the `--tts` voice, once the voice is loaded |
| `POST /v1/audio/transcriptions` | a WAV file (raw or in a form): `text`, `language`, `duration` |
| `GET /api/system` | this computer: CPU, RAM, GPU (VRAM, temperature, load), OS, drives, and the model and engine running (the chat page's `get_system_info` tool) |
| `GET /api/progress` | the prompt being read: `done` and `total` tokens (`total` 0: none), and `disk`: waiting for weights to come off the disk |
| `GET /` | the chat page |
| `GET /health` | `{"status":"ok"}` |

- Requests run one at a time. Unset sampling fields use the command-line defaults.
- The server keeps the last conversation in its cache: when a client resends the history with one new
  message, only the new part is read.
- Greedy requests (`"temperature": 0`) use the `--draft` model when one is loaded, with the same output.
- **Tool calling** works the OpenAI way (`tools`, `tool_calls`, `role: "tool"` results). Tools are written
  into the prompt the way Qwen's chat template does, and the model's `<tool_call>` blocks come back as
  `tool_calls` with `finish_reason: "tool_calls"`; streamed, each call is one `delta.tool_calls` chunk.
- **Switching models** drops the running model (freeing its GPU memory) and loads the new one. Requests
  during a switch get `503`; if the new model fails to load, the server goes back to the previous one.
- **Two models at once**: with a call model kept beside the chat model, a request goes to the model it was
  written for (Maya's to the call model, everything else to the chat model). They are kept together only if
  each has a working context (8K tokens for the chat model, 4K for the call model); the server checks the
  files' sizes first, then the contexts once loaded, and remembers a pair that did not fit.
- The agent's endpoints (`/api/agent`, `/api/tools`, `/api/chats` and the rest the app uses) are not part
  of this API: they answer only requests from this computer that carry the token written into the chat
  page. Other agent harnesses bring their own loop and tools and use `/v1`.
- Not supported: images, log-probabilities, several choices per request (`n`).
