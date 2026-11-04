"""Converts a Hugging Face Whisper model to ghost-infer's speech-to-text file (src/whisper.rs): one GGUF.

  python scripts/whisper-convert.py OUT.gguf [--model MahmoudAshraf/acft-whisper-large-v3-turbo]
      [--generation openai/whisper-large-v3-turbo]

Matrices are Q8_0; the two input convolutions are written as matrices over [x(t-1) | x(t) | x(t+1)] (each
tap's input channels in turn), so they run as matmuls. The mel filter bank, the tokenizer (byte-level BPE, for
reading tokens back as text) and the special tokens go in the file too. --generation names the repo whose
generation_config.json lists the tokens never to write (an ACFT fine-tune ships without one).
"""
import argparse
import json
import os

import numpy as np
import torch
import gguf
from gguf import GGMLQuantizationType as Q, TokenType
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer


def f32(t):
    return t.detach().float().cpu().numpy()


def q8(out, name, w):
    out.add_tensor(name, gguf.quants.quantize(np.ascontiguousarray(w, dtype=np.float32), Q.Q8_0), raw_dtype=Q.Q8_0)


def conv(w):
    """[out, in, 3] to [out, 3 * in]: tap-major, as the im2col rows are laid out."""
    o, i, k = w.shape
    return np.ascontiguousarray(w.transpose(0, 2, 1).reshape(o, k * i))


def attn(out, sd, p, b, cross):
    pre = "encoder_attn" if cross else "self_attn"
    q8(out, b + "q.weight", f32(sd[p + pre + ".q_proj.weight"]))
    out.add_tensor(b + "q.bias", f32(sd[p + pre + ".q_proj.bias"]))
    q8(out, b + "k.weight", f32(sd[p + pre + ".k_proj.weight"]))  # no bias
    q8(out, b + "v.weight", f32(sd[p + pre + ".v_proj.weight"]))
    out.add_tensor(b + "v.bias", f32(sd[p + pre + ".v_proj.bias"]))
    q8(out, b + "o.weight", f32(sd[p + pre + ".out_proj.weight"]))
    out.add_tensor(b + "o.bias", f32(sd[p + pre + ".out_proj.bias"]))


def norm(out, sd, src, dst):
    out.add_tensor(dst + ".weight", f32(sd[src + ".weight"]))
    out.add_tensor(dst + ".bias", f32(sd[src + ".bias"]))


def mlp(out, sd, p, b):
    q8(out, b + "fc1.weight", f32(sd[p + "fc1.weight"]))
    out.add_tensor(b + "fc1.bias", f32(sd[p + "fc1.bias"]))
    q8(out, b + "fc2.weight", f32(sd[p + "fc2.weight"]))
    out.add_tensor(b + "fc2.bias", f32(sd[p + "fc2.bias"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="MahmoudAshraf/acft-whisper-large-v3-turbo")
    ap.add_argument("--generation", default="openai/whisper-large-v3-turbo")
    args = ap.parse_args()

    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    cfg = model.config
    sd = model.state_dict()
    tok = WhisperTokenizer.from_pretrained(args.model)
    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    gen = json.load(open(hf_hub_download(args.generation, "generation_config.json")))

    out = gguf.GGUFWriter(args.out, "whisper")
    out.add_name(os.path.basename(args.model))
    for k, v in [("n_mels", cfg.num_mel_bins), ("dim", cfg.d_model), ("heads", cfg.encoder_attention_heads), ("encoder_layers", cfg.encoder_layers),
                 ("decoder_layers", cfg.decoder_layers), ("ffn", cfg.encoder_ffn_dim), ("audio_positions", cfg.max_source_positions),
                 ("text_positions", cfg.max_target_positions), ("vocab", cfg.vocab_size)]:
        out.add_uint32(f"whisper.{k}", int(v))
    ids = lambda t: tok.convert_tokens_to_ids(t)
    for k, t in [("sot", "<|startoftranscript|>"), ("eot", "<|endoftext|>"), ("en", "<|en|>"), ("transcribe", "<|transcribe|>"),
                 ("translate", "<|translate|>"), ("notimestamps", "<|notimestamps|>"), ("nospeech", "<|nospeech|>")]:
        out.add_uint32(f"whisper.token.{k}", int(ids(t)))
    out.add_uint32("whisper.token.timestamp_begin", int(ids("<|0.00|>")))
    out.add_array("whisper.suppress", [int(t) for t in gen.get("suppress_tokens", [])])
    out.add_array("whisper.begin_suppress", [int(t) for t in gen.get("begin_suppress_tokens", [220, ids("<|endoftext|>")])])
    # the languages: name -> token, for a language other than English
    langs = gen.get("lang_to_id", {})
    out.add_array("whisper.languages", [k.strip("<|>") for k in langs])
    out.add_array("whisper.language_tokens", [int(v) for v in langs.values()])

    # the mel filter bank: n_mels rows over the 201 frequency bins of a 400-point FFT
    out.add_tensor("mel_filters", np.ascontiguousarray(fe.mel_filters.T.astype(np.float32)))

    # tokens as byte-level BPE strings (read back as text through GPT-2's byte map)
    vocab = tok.get_vocab()
    rev = {i: t for t, i in vocab.items()}
    added = tok.added_tokens_decoder
    tokens, types = [], []
    for i in range(cfg.vocab_size):
        if i in added:
            tokens.append(added[i].content)
            types.append(TokenType.CONTROL)
        else:
            tokens.append(rev.get(i, f"[PAD{i}]"))
            types.append(TokenType.NORMAL if i in rev else TokenType.UNUSED)
    out.add_tokenizer_model("gpt2")
    out.add_token_list(tokens)
    out.add_token_types(types)

    e = "model.encoder."
    q8(out, "enc.conv1.weight", conv(f32(sd[e + "conv1.weight"])))
    out.add_tensor("enc.conv1.bias", f32(sd[e + "conv1.bias"]))
    q8(out, "enc.conv2.weight", conv(f32(sd[e + "conv2.weight"])))
    out.add_tensor("enc.conv2.bias", f32(sd[e + "conv2.bias"]))
    out.add_tensor("enc.pos", f32(sd[e + "embed_positions.weight"]))
    for i in range(cfg.encoder_layers):
        p, b = f"{e}layers.{i}.", f"enc.blk.{i}."
        norm(out, sd, p + "self_attn_layer_norm", b + "attn_norm")
        attn(out, sd, p, b + "attn_", False)
        norm(out, sd, p + "final_layer_norm", b + "ffn_norm")
        mlp(out, sd, p, b)
    norm(out, sd, e + "layer_norm", "enc.norm")

    d = "model.decoder."
    q8(out, "dec.embd", f32(sd[d + "embed_tokens.weight"]))  # also the output layer (tied)
    out.add_tensor("dec.pos", f32(sd[d + "embed_positions.weight"]))
    for i in range(cfg.decoder_layers):
        p, b = f"{d}layers.{i}.", f"dec.blk.{i}."
        norm(out, sd, p + "self_attn_layer_norm", b + "attn_norm")
        attn(out, sd, p, b + "attn_", False)
        norm(out, sd, p + "encoder_attn_layer_norm", b + "cross_norm")
        attn(out, sd, p, b + "cross_", True)
        norm(out, sd, p + "final_layer_norm", b + "ffn_norm")
        mlp(out, sd, p, b)
    norm(out, sd, d + "layer_norm", "dec.norm")

    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    print(f"{args.out} written")


if __name__ == "__main__":
    with torch.no_grad():
        main()
