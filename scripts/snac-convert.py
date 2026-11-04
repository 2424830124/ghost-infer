"""Converts the SNAC 24 kHz decoder (hubertsiuzdak/snac_24khz, the codec maya1 and Orpheus speak in) to
snac.gguf for src/snac.rs: the decoder's weights with weight norm folded in, and each codebook already through
its quantizer's output projection (a code is then a row to add).

  python scripts/snac-convert.py OUT.gguf [--ref REF.json]

--ref also writes a test: random codes and the audio the reference decoder makes of them (noise off), for
`ghost-infer snac-check snac.gguf REF.json`.
"""
import argparse
import json

import numpy as np
import torch
import gguf
from snac import SNAC


def f32(t):
    return t.detach().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--repo", default="hubertsiuzdak/snac_24khz")
    ap.add_argument("--ref")
    args = ap.parse_args()
    m = SNAC.from_pretrained(args.repo).eval()
    assert m.attn_window_size is None, "a decoder with local attention is not supported"
    out = gguf.GGUFWriter(args.out, "snac")
    out.add_name(args.repo.split("/")[-1])
    out.add_uint32("snac.sample_rate", m.sampling_rate)
    out.add_uint32("snac.latent_dim", m.latent_dim)
    out.add_uint32("snac.codebook_size", m.codebook_size)
    out.add_array("snac.vq_strides", list(m.vq_strides))
    out.add_array("snac.decoder_rates", list(m.decoder.model[2 + i].block[1].stride[0] for i in range(len(m.decoder_rates))))
    t = lambda name, w: out.add_tensor(name, np.ascontiguousarray(f32(w), dtype=np.float32))
    with torch.no_grad():
        for i, q in enumerate(m.quantizer.quantizers):
            # out_proj (a 1x1 convolution) applied to every code's vector: [codes][latent]
            table = q.out_proj(q.codebook.weight.t().unsqueeze(0))[0].t()
            t(f"q{i}.table", table)
        d = m.decoder.model
        t("in.dw.weight", d[0].weight[:, 0, :])
        t("in.dw.bias", d[0].bias)
        t("in.pw.weight", d[1].weight[:, :, 0])
        t("in.pw.bias", d[1].bias)
        for b in range(len(m.decoder_rates)):
            blk = d[2 + b].block
            t(f"b{b}.snake", blk[0].alpha.reshape(-1))
            t(f"b{b}.up.weight", blk[1].weight)  # [in][out][k]
            t(f"b{b}.up.bias", blk[1].bias)
            t(f"b{b}.noise.weight", blk[2].linear.weight[:, :, 0])
            for r in range(3):
                ru = blk[3 + r].block
                t(f"b{b}.r{r}.snake1", ru[0].alpha.reshape(-1))
                t(f"b{b}.r{r}.dw.weight", ru[1].weight[:, 0, :])
                t(f"b{b}.r{r}.dw.bias", ru[1].bias)
                t(f"b{b}.r{r}.snake2", ru[2].alpha.reshape(-1))
                t(f"b{b}.r{r}.pw.weight", ru[3].weight[:, :, 0])
                t(f"b{b}.r{r}.pw.bias", ru[3].bias)
        n = len(m.decoder_rates)
        t("out.snake", d[2 + n].alpha.reshape(-1))
        t("out.weight", d[3 + n].weight[0])  # [in][k]
        t("out.bias", d[3 + n].bias)
    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    print(f"{args.out} written", flush=True)

    if args.ref:
        # the reference with every NoiseBlock off (random by design), on 12 frames of random codes
        for blk in m.decoder.model:
            if hasattr(blk, "block") and len(blk.block) > 2 and type(blk.block[2]).__name__ == "NoiseBlock":
                blk.block[2] = torch.nn.Identity()
        g = torch.Generator().manual_seed(0)
        frames = 12
        codes = [torch.randint(0, m.codebook_size, (1, frames * 4 // s), generator=g) for s in m.vq_strides]
        with torch.no_grad():
            audio = m.decode(codes)[0, 0]
        json.dump({"codes": [c[0].tolist() for c in codes], "audio": [round(float(v), 6) for v in audio]}, open(args.ref, "w"))
        print(f"{args.ref} written: {frames} frames, {audio.numel()} samples", flush=True)


if __name__ == "__main__":
    main()
