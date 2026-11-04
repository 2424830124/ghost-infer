"""Converts Sesame CSM-1B to ghost-infer's speech model: a folder of three GGUF files (see src/csm.rs).

  backbone.gguf  the 16-layer Llama that reads text and audio frames and picks each frame's first codebook
  decoder.gguf   the 4-layer Llama that picks the other 31 codebooks of a frame
  mimi.gguf      Kyutai's Mimi decoder, which turns the codes into 24 kHz audio

Both transformers are written as llama.cpp writes Llama (q and k permuted for adjacent-pair RoPE, Llama 3
frequency scaling as rope_freqs) and are fed vectors: their input tables stay on the CPU.

  python scripts/csm-convert.py OUT_DIR [--base unsloth/csm-1b] [--lora shb777/csm-maya-exp2]

--lora merges a PEFT adapter (a voice fine-tune) into the weights first.
"""
import argparse
import os

import numpy as np
import torch
import gguf
from gguf import GGMLQuantizationType as Q, TokenType
from transformers import AutoTokenizer, CsmForConditionalGeneration
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def f32(t):
    return t.detach().float().cpu().numpy()


def q8(w, name, out):
    out.add_tensor(name, gguf.quants.quantize(np.ascontiguousarray(w, dtype=np.float32), Q.Q8_0), raw_dtype=Q.Q8_0)


def permute(w, heads):
    """HF's rotate-half q/k rows to the adjacent pairs llama-architecture GGUFs use (llama.cpp's permute)."""
    return w.reshape(heads, 2, w.shape[0] // heads // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def rope_freqs(cfg):
    """Llama 3 scaling as llama.cpp keeps it: per rotated pair, what the plain frequency is divided by."""
    inv, scale = ROPE_INIT_FUNCTIONS[cfg.rope_scaling["rope_type"]](cfg, "cpu")
    assert scale == 1.0
    hd = cfg.head_dim
    plain = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    return (plain / inv.double()).float().numpy()


def llama(out, sd, prefix, cfg, ctx):
    """The llama keys and blocks of one transformer (`prefix`: its layers' parent in the state dict)."""
    out.add_context_length(ctx)
    out.add_embedding_length(cfg.hidden_size)
    out.add_block_count(cfg.num_hidden_layers)
    out.add_feed_forward_length(cfg.intermediate_size)
    out.add_head_count(cfg.num_attention_heads)
    out.add_head_count_kv(cfg.num_key_value_heads)
    out.add_key_length(cfg.head_dim)
    out.add_value_length(cfg.head_dim)
    out.add_rope_dimension_count(cfg.head_dim)
    out.add_rope_freq_base(float(cfg.rope_theta))
    out.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    out.add_vocab_size(cfg.vocab_size)
    out.add_tensor("rope_freqs.weight", rope_freqs(cfg))
    for i in range(cfg.num_hidden_layers):
        p, b = f"{prefix}.layers.{i}.", f"blk.{i}."
        out.add_tensor(b + "attn_norm.weight", f32(sd[p + "input_layernorm.weight"]))
        q8(permute(f32(sd[p + "self_attn.q_proj.weight"]), cfg.num_attention_heads), b + "attn_q.weight", out)
        q8(permute(f32(sd[p + "self_attn.k_proj.weight"]), cfg.num_key_value_heads), b + "attn_k.weight", out)
        q8(f32(sd[p + "self_attn.v_proj.weight"]), b + "attn_v.weight", out)
        q8(f32(sd[p + "self_attn.o_proj.weight"]), b + "attn_output.weight", out)
        out.add_tensor(b + "ffn_norm.weight", f32(sd[p + "post_attention_layernorm.weight"]))
        q8(f32(sd[p + "mlp.gate_proj.weight"]), b + "ffn_gate.weight", out)
        q8(f32(sd[p + "mlp.up_proj.weight"]), b + "ffn_up.weight", out)
        q8(f32(sd[p + "mlp.down_proj.weight"]), b + "ffn_down.weight", out)
    out.add_tensor("output_norm.weight", f32(sd[prefix + ".norm.weight"]))


def tokenizer(out, base):
    tok = AutoTokenizer.from_pretrained(base)
    vocab = tok.get_vocab()
    rev = {i: t for t, i in vocab.items()}
    added = tok.added_tokens_decoder
    tokens, types = [], []
    for i in range(max(vocab.values()) + 1):
        if i in added:
            tokens.append(added[i].content)
            types.append(TokenType.CONTROL if added[i].special else TokenType.USER_DEFINED)
        elif i in rev:
            tokens.append(rev[i])
            types.append(TokenType.NORMAL)
        else:
            tokens.append(f"[PAD{i}]")
            types.append(TokenType.UNUSED)
    special = gguf.SpecialVocab(base if os.path.isdir(base) else snapshot(base), load_merges=True)
    out.add_tokenizer_model("gpt2")
    out.add_tokenizer_pre("llama-bpe")
    out.add_token_list(tokens)
    out.add_token_types(types)
    out.add_token_merges(special.merges)
    out.add_bos_token_id(tok.bos_token_id)
    out.add_eos_token_id(tok.eos_token_id)
    out.add_add_bos_token(False)


def snapshot(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", default="unsloth/csm-1b")
    ap.add_argument("--lora", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = CsmForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float32)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora).merge_and_unload()
    model.eval()
    cfg, dcfg, ccfg = model.config, model.config.depth_decoder_config, model.config.codec_config
    sd = model.state_dict()
    name = os.path.basename(os.path.normpath(args.out))

    # the backbone: text and audio frames in, the first codebook's logits and the hidden state out
    out = gguf.GGUFWriter(os.path.join(args.out, "backbone.gguf"), "llama")
    out.add_name(f"{name} backbone")
    llama(out, sd, "backbone_model", cfg, cfg.max_position_embeddings)
    q8(f32(sd["lm_head.weight"]), "output.weight", out)
    q8(f32(sd["embed_text_tokens.weight"]), "csm.text_embd.weight", out)
    # one row per (codebook, code): a frame's input is the sum of its 32 rows
    audio = f32(sd.get("backbone_model.embed_tokens.embed_audio_tokens.weight", sd["depth_decoder.model.embed_tokens.weight"]))
    out.add_tensor("csm.audio_embd.weight", audio.astype(np.float16))
    out.add_uint32("csm.codebooks", cfg.num_codebooks)
    out.add_uint32("csm.audio_vocab", cfg.vocab_size)
    out.add_uint32("csm.codebook_eos", cfg.codebook_eos_token_id)
    tokenizer(out, args.base)
    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    print("backbone.gguf written", flush=True)

    # the depth decoder: position 0 is the projected backbone hidden state, position i the code of codebook
    # i - 1 (its projected row), and the head of codebook i reads position i
    out = gguf.GGUFWriter(os.path.join(args.out, "decoder.gguf"), "llama")
    out.add_name(f"{name} decoder")
    llama(out, sd, "depth_decoder.model", dcfg, dcfg.max_position_embeddings)
    proj = f32(sd["depth_decoder.model.inputs_embeds_projector.weight"])  # 1024 x 2048
    dec_embd = f32(sd["depth_decoder.model.embed_tokens.weight"])
    out.add_tensor("csm.proj_embd.weight", (dec_embd @ proj.T).astype(np.float16))
    q8(proj, "csm.proj.weight", out)
    heads = f32(sd["depth_decoder.codebooks_head.weight"])  # 31 x 1024 x 2051
    for i in range(heads.shape[0]):
        q8(np.ascontiguousarray(heads[i].T), f"csm.head.{i + 1}.weight", out)
    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    print("decoder.gguf written", flush=True)

    # Mimi's decoder half. Each codebook's rows go through its quantizer's output projection here, so a
    # frame's 512-wide input is just the sum of 32 rows.
    out = gguf.GGUFWriter(os.path.join(args.out, "mimi.gguf"), "mimi")
    out.add_name(f"{name} mimi")
    rows = []
    for part, n in (("semantic", ccfg.num_semantic_quantizers), ("acoustic", ccfg.num_quantizers - ccfg.num_semantic_quantizers)):
        p = f"codec_model.quantizer.{part}_residual_vector_quantizer."
        w = f32(sd[p + "output_proj.weight"])[:, :, 0]  # 512 x 256
        for i in range(n):
            cb = p + f"layers.{i}.codebook."
            embed = f32(sd[cb + "embed_sum"]) / np.maximum(f32(sd[cb + "cluster_usage"]), 1e-5)[:, None]
            rows.append(embed @ w.T)
    out.add_tensor("quantizer.codebooks.weight", np.concatenate(rows).astype(np.float32))
    for k, v in sd.items():
        if k.startswith(("codec_model.upsample.", "codec_model.decoder_transformer.", "codec_model.decoder.")):
            out.add_tensor(k[len("codec_model."):], f32(v))
    out.add_uint32("mimi.sample_rate", ccfg.sampling_rate)
    out.add_uint32("mimi.head_count", ccfg.num_attention_heads)
    out.add_uint32("mimi.block_count", ccfg.num_hidden_layers)
    out.add_float32("mimi.rope.freq_base", float(ccfg.rope_theta))
    out.add_float32("mimi.norm_eps", float(ccfg.norm_eps))
    out.add_uint32("mimi.sliding_window", ccfg.sliding_window)
    out.add_array("mimi.upsampling_ratios", list(ccfg.upsampling_ratios))
    out.add_uint32("mimi.compress", ccfg.compress)
    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    print("mimi.gguf written", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
