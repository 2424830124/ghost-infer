"""Reference numbers from the fp32 transformers model for `ghost-infer csm-check` (src/csm.rs).

  python scripts/csm-reference.py OUT.json [--lora shb777/csm-maya-exp2]

The model must be the one scripts/csm-convert.py converted (same base and LoRA).
"""
import argparse
import json

import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--base", default="unsloth/csm-1b")
ap.add_argument("--lora", default=None)
args = ap.parse_args()
OUT = args.out
TEXT = "Hey, it's really nice to hear from you."
dev = "cuda"

proc = AutoProcessor.from_pretrained(args.base)
model = CsmForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float32)
if args.lora:
    model = PeftModel.from_pretrained(model, args.lora).merge_and_unload()
model = model.to(dev).eval()
bb, dd = model.backbone_model, model.depth_decoder
V = model.config.vocab_size

conv = [{"role": "4", "content": [{"type": "text", "text": TEXT}]}]
ids = proc.apply_chat_template(conv, tokenize=True, return_dict=True)["input_ids"].to(dev)
ref = {"text": TEXT, "speaker": 4, "tokens": ids[0].tolist()}


def frame_step(h):
    """Greedy codes of one frame from the backbone's last hidden state; logits of every codebook."""
    c0_logits = model.lm_head(h)
    codes, logits = [int(c0_logits.argmax())], [c0_logits[0].tolist()]
    out = dd(input_ids=torch.tensor([[0, codes[0]]], device=dev), backbone_last_hidden_state=h, use_cache=True,
             cache_position=torch.tensor([0, 1], device=dev), logits_to_keep=1)
    for i in range(1, 32):
        lg = out.logits[0, -1]
        logits.append(lg.tolist())
        codes.append(int(lg.argmax()))
        if i < 31:
            out = dd(input_ids=torch.tensor([[codes[-1]]], device=dev), past_key_values=out.past_key_values,
                     cache_position=torch.tensor([i + 1], device=dev), logits_to_keep=1, use_cache=True)
    return codes, logits


with torch.no_grad():
    emb = model.embed_text_tokens(ids)
    o = bb(inputs_embeds=emb, use_cache=True)
    h = o.last_hidden_state[:, -1]
    ref["hidden"] = h[0].tolist()
    codes, logits = frame_step(h)
    ref["frame0_codes"], ref["frame0_logits"] = codes, logits
    # greedy frames, each fed back as the sum of its codebooks' rows
    frames, pkv = [codes], o.past_key_values
    for _ in range(11):
        x = bb.embed_tokens(torch.tensor([[frames[-1]]], device=dev))
        o = bb(inputs_embeds=x, past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values
        frames.append(frame_step(o.last_hidden_state[:, -1])[0])
    ref["greedy_frames"] = frames

    # sampled speech, and Mimi's audio for its codes
    torch.manual_seed(0)
    inputs = proc.apply_chat_template(conv, tokenize=True, return_dict=True).to(dev)
    seq = model.generate(**inputs, max_new_tokens=375)[0]  # frames x 32
    eos = (seq == 0).all(-1).nonzero()
    seq = seq[: int(eos.min()) if eos.numel() else seq.shape[0]]
    audio = model.codec_model.decode(seq.T.unsqueeze(0)).audio_values[0, 0]
    ref["mimi_codes"] = seq.tolist()
    ref["mimi_audio"] = [round(v, 6) for v in audio.tolist()]
    proc.save_audio([audio], OUT.replace(".json", ".wav"))

with open(OUT, "w") as f:
    json.dump(ref, f)
print("frames", len(ref["mimi_codes"]), "audio", len(ref["mimi_audio"]), "greedy c0:", [fr[0] for fr in frames])
