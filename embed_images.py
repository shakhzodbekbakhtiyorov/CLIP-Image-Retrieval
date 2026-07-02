"""Embed every image in the manifest with OpenCLIP ViT-B/32.

Pipeline per image: load -> CLIP preprocess -> image encoder -> L2-normalize.
Normalized vectors mean inner product == cosine similarity, which is what the
FAISS IndexFlatIP in the next step relies on.

Outputs (saved next to the manifest):
    embeddings.npy   float32 array, shape (N, D), each row unit-norm
    ids.json         list of {"id", "image_path"} aligned row-for-row with the
                     embeddings (row i <-> ids[i]); this is the id->path map

BACKBONE IS FIXED: OpenCLIP ViT-B/32, pretrained 'openai'. Do not change it --
the same model's text encoder is reused unchanged in phase 2.
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
import open_clip

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".", help="repo root (contains val2017/ and manifest.json)")
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument("--out-emb", default="embeddings.npy")
    p.add_argument("--out-ids", default="ids.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="embed only first N images (0 = all); for smoke tests")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    with open(os.path.join(args.data_dir, args.manifest)) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]

    print(f"Loading {MODEL_NAME}/{PRETRAINED} on {args.device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(args.device).eval()

    n = len(records)
    embs = []
    ids = []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            batch = records[start : start + args.batch_size]
            imgs = []
            kept = []
            for r in batch:
                path = os.path.join(args.data_dir, r["image_path"])
                try:
                    img = Image.open(path).convert("RGB")
                except Exception as e:  # noqa: BLE001
                    print(f"  skip {path}: {e}")
                    continue
                imgs.append(preprocess(img))
                kept.append(r)
            if not imgs:
                continue
            x = torch.stack(imgs).to(args.device)
            feats = model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalize
            embs.append(feats.cpu().numpy().astype("float32"))
            for r in kept:
                ids.append({"id": r["id"], "image_path": r["image_path"]})
            print(f"  embedded {min(start + args.batch_size, n)}/{n}", end="\r")

    embeddings = np.concatenate(embs, axis=0)
    np.save(os.path.join(args.data_dir, args.out_emb), embeddings)
    with open(os.path.join(args.data_dir, args.out_ids), "w") as f:
        json.dump(ids, f)

    # --- sanity checks ---
    norms = np.linalg.norm(embeddings, axis=1)
    print()
    print(f"embeddings shape: {embeddings.shape}  dtype: {embeddings.dtype}")
    print(f"id rows:          {len(ids)} (must equal {embeddings.shape[0]})")
    print(f"norm min/max:     {norms.min():.5f} / {norms.max():.5f} (want ~1.0)")
    assert len(ids) == embeddings.shape[0], "id/embedding row count mismatch"
    assert np.allclose(norms, 1.0, atol=1e-3), "vectors are not unit-norm"
    print("OK: vectors are unit-norm and aligned with ids.json")


if __name__ == "__main__":
    main()
