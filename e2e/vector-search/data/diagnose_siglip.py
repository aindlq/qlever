#!/usr/bin/env python3
"""Root-cause diagnostic: why does SigLIP text->image fail through vLLM?

Runs INSIDE the vLLM container (uses transformers + the cached model weights).
Loads SigLIP with plain HF transformers and computes the text<->image cosine
matrix two ways:

  * padding="max_length" (=64)  -- SigLIP's CANONICAL preprocessing. The model
    pools the text at the fixed 64th position. -> cross-modal retrieval works.
  * padding="longest"           -- what vLLM effectively does (no fixed pad).
    The last *real* token is pooled instead. -> cross-modal retrieval collapses,
    reproducing exactly what the vLLM server returns.

Conclusion: vLLM 0.24.0 CPU does not reproduce SigLIP's fixed-64 text padding,
so served text embeddings are cross-modally misaligned with image embeddings.
The image path (and text<->text, image<->image) are unaffected.

Usage:  docker compose exec -T vllm python3 /work/diagnose_siglip.py [model_id]
"""
import sys

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/siglip-base-patch16-224"
SUBS = ["cat", "dog", "pizza", "mountain", "banana", "sunflower"]


def main():
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    imgs = [Image.open(f"/work/images/{s}.jpg").convert("RGB") for s in SUBS]

    with torch.no_grad():
        io = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
    ie = getattr(io, "pooler_output", io)
    ie = ie / ie.norm(dim=-1, keepdim=True)

    def text_emb(padding):
        ti = proc(text=[f"a photo of a {s}" for s in SUBS],
                  padding=padding, max_length=64, return_tensors="pt")
        with torch.no_grad():
            to = model.get_text_features(**ti)
        te = getattr(to, "pooler_output", to)
        return te / te.norm(dim=-1, keepdim=True)

    print(f"model: {MODEL}\n")
    for padding, label in [
        ("max_length", 'padding="max_length"=64  (SigLIP canonical)'),
        ("longest", 'padding="longest"         (vLLM-equivalent)'),
    ]:
        sim = text_emb(padding) @ ie.T
        hits = 0
        print(f"[{label}]")
        print("          " + " ".join(f"{s[:6]:>7}" for s in SUBS) + "   argmax")
        for r, t in enumerate(SUBS):
            am = SUBS[int(sim[r].argmax())]
            hits += am == t
            print(f"{t:>8}  " + " ".join(f"{v:7.3f}" for v in sim[r].tolist())
                  + f"   -> {am} {'OK' if am == t else 'X'}")
        print(f"diagonal-argmax (correct) hits: {hits}/{len(SUBS)}\n")


if __name__ == "__main__":
    main()
