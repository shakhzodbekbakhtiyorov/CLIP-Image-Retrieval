"""Text->image search: CLIP text encoder -> same embeddings.npy, exact cosine.

numpy (not faiss) avoids the faiss+torch OpenMP crash on macOS.
"""

import argparse
import json
import os

import numpy as np

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def embed_text(queries):
    """CLIP text encoder -> L2-normalize. Returns (m, D) float32."""
    import torch
    import open_clip

    device = "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    tokens = tokenizer(queries).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", help="text query, e.g. 'a red bus on a city street'")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--emb", default="embeddings.npy")
    p.add_argument("-k", "--topk", type=int, default=5)
    args = p.parse_args()

    with open(os.path.join(args.data_dir, "ids.json")) as f:
        ids = json.load(f)
    corpus = np.load(os.path.join(args.data_dir, args.emb)).astype("float32")

    q = embed_text([args.query])[0]                # (D,), already normalized
    scores = corpus @ q                            # (N,) cosine
    k = args.topk
    rows = np.argpartition(-scores, range(k))[:k]
    rows = rows[np.argsort(-scores[rows])]

    print(f'Query: "{args.query}"')
    print(f"\nTop {k} images:")
    for row in rows:
        print(f"  {scores[row]:.4f}  {ids[row]['image_path']}  (id {ids[row]['id']})")


if __name__ == "__main__":
    main()
