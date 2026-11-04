# Building, requirements and models

[Back to the README](../README.md)

## Build from source

```bash
git clone https://github.com/affifuddin-hazam/ghost-infer
cd ghost-infer
cargo build --release
```

Put some GGUF files in a folder (for example `models\`) and start the server on one of them:

```bash
target\release\ghost-infer serve models\Qwen3-8B-Q4_K_M.gguf --gpu
```

Open <http://127.0.0.1:8080> to chat, or point any OpenAI client at `http://127.0.0.1:8080/v1`. The
other GGUF files in the same folder appear in the chat page's model picker.

Requirements: Windows 10/11, an NVIDIA GPU with compute capability 6.1 or newer (GTX 10-series and
later), a current driver, and, built from source, the CUDA toolkit's NVRTC library (see
[GPU requirements](#gpu-requirements)). Without `--gpu` it runs on the CPU (AVX2).

## The installer

`GhostSetup` installs ghost-infer and its desktop app for one user, in `%LOCALAPPDATA%\Programs\ghost-infer`,
with NVIDIA's runtime compiler (NVRTC) beside them, so nothing else needs installing. Uninstall it from
Windows' Apps settings: your chats, settings and models (in `%LOCALAPPDATA%\ghost-infer`) stay unless you
choose to remove them.

## GPU requirements

- An NVIDIA GPU with compute capability 6.1 or newer (needs `dp4a`; 7.5+ also uses int8 tensor cores),
  and a current driver. Dense models must fit in VRAM (an 8B Q4_K_M needs about 5.5 GB); mixture-of-experts
  models only need their shared weights to fit.
- NVRTC 11.1 or newer. The installer ships it; built from source, it comes from a
  [CUDA toolkit](https://developer.nvidia.com/cuda-downloads). ghost-infer takes the one named by
  `GHOST_INFER_NVRTC`, else an `nvrtc64_*.dll` beside the program, else the newest under
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin`. Compiled kernels are cached in
  `%LOCALAPPDATA%\ghost-infer\kernels` (about 0.6 MB), so only the first start after an update or a GPU
  change compiles them.
- Windows only for now (Linux is on the roadmap).

`.cargo/config.toml` builds with `target-cpu=native`, so the binary is tuned for the machine that built
it. [`scripts/build-installer.ps1`](../scripts/build-installer.ps1) builds for any CPU with AVX2 and FMA
(x86-64-v3, as fast here) and packs the installer with [Inno Setup](https://jrsoftware.org/isinfo.php) 6 or later.

## Supported models

| | |
|---|---|
| Architectures | Llama, Qwen3, Qwen3-MoE, DeepSeek-style models such as GLM-4.7-Flash, Qwen3-Next such as Qwen3-Coder-Next, and Qwen3.5 / Qwen3.6 mixtures of experts such as Qwen3.6-35B-A3B (`general.architecture` `llama`, `qwen3`, `qwen3moe`, `deepseek2`, `qwen3next`, `qwen35moe`; on the GPU, `deepseek2` with a 512-wide latent and up to 32 heads, `qwen3next` and `qwen35moe` with 128-wide DeltaNet heads; text only) |
| Weight formats | Q8_0, Q3_K, Q4_K, Q5_K, Q6_K (so Q3_K_M, Q4_K_M, Q5_K_M, Q6_K and Q8_0 files), IQ2_XXS, IQ2_XS and IQ2_S (the 2-bit IQ2 files), F16/F32 on the CPU; Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, IQ3_S and BF16 by dequantising, for the few tensors llama.cpp stores that way |
| Tokenizer | byte-level BPE (`gpt2`): GPT-2 / Qwen2 / Qwen3.5 / GLM-4 / Llama 3 splitting, SmolLM-style digits |
| Chat template | ChatML (SmolLM2, Qwen) with Qwen3's thinking and tool-call format, Qwen3-Coder's XML tool format, and Qwen3.5 / 3.6's (Qwen3-Coder's calls with thinking); GLM-4's format with its XML tool calls; Mistral's (`[INST]`, Mistral Nemo and its merges) and Llama 3's, without tools |
| Tested with | [SmolLM2-135M-Instruct](https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF), Qwen3 0.6B and 8B fine-tunes, [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF), [GLM-4.7-Flash (Huihui abliterated, i1-Q3_K_M)](https://huggingface.co/mradermacher/Huihui-GLM-4.7-Flash-abliterated-i1-GGUF), [Qwen3-Coder-30B-A3B (Huihui abliterated, i1-Q3_K_M)](https://huggingface.co/mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-i1-GGUF), [Qwen3-Coder-Next (Huihui abliterated, IQ2_S)](https://huggingface.co/bartowski/huihui-ai_Qwen3-Coder-Next-abliterated-GGUF), [Qwen3.6-35B-A3B (Huihui abliterated, i1-Q3_K_M)](https://huggingface.co/mradermacher/Huihui-Qwen3.6-35B-A3B-abliterated-i1-GGUF) |

Not yet: other IQ formats (IQ1, IQ3_XXS, IQ4_XS), Gemma, images.
