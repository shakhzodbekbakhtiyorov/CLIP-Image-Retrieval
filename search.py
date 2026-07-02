"""Image -> image search over the flat FAISS index.

Two query modes:
  --image PATH   embed an arbitrary image with CLIP, then search
  --id COCO_ID   reuse the stored vector for a dataset image (no CLIP load);
                 handy for quickly eyeballing results

Returns the top-k most similar images by cosine similarity, as paths + scores.
A self-match (score ~1.0) is expected and skipped when querying by --id.

Backbone is pinned to OpenCLIP ViT-B/32 / openai -- must match embed_images.py.
"""

import argparse
import json
import os

import numpy as np
import faiss

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def load_ids(data_dir: str) -> list:
    with open(os.path.join(data_dir, "ids.json")) as f:
        return json.load(f)


def embed_image(path: str) -> np.ndarray:
    """Embed one image the same way the corpus was embedded."""
    import torch
    from PIL import Image
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        v = model.encode_image(img)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype("float32")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".")
    p.add_argument("--index", default="index_flat.faiss")
    p.add_argument("--image", help="path to a query image")
    p.add_argument("--id", type=int, help="COCO image id already in the index")
    p.add_argument("-k", "--topk", type=int, default=5)
    args = p.parse_args()
    if (args.image is None) == (args.id is None):
        p.error("provide exactly one of --image or --id")

    ids = load_ids(args.data_dir)
    index = faiss.read_index(os.path.join(args.data_dir, args.index))

    skip_self = False
    if args.image is not None:
        q = embed_image(os.path.join(args.data_dir, args.image)
                        if not os.path.isabs(args.image) else args.image)
        print(f"Query image: {args.image}")
    else:
        row = next((i for i, r in enumerate(ids) if r["id"] == args.id), None)
        if row is None:
            p.error(f"id {args.id} not found in ids.json")
        q = index.reconstruct(row).reshape(1, -1)
        skip_self = True
        print(f"Query id {args.id}: {ids[row]['image_path']}")

    # over-fetch by 1 so we can drop the self-match when querying by id
    scores, idxs = index.search(q, args.topk + (1 if skip_self else 0))

    print(f"\nTop {args.topk} matches:")
    shown = 0
    for score, row in zip(scores[0], idxs[0]):
        if skip_self and ids[row]["id"] == args.id:
            continue
        print(f"  {score:.4f}  {ids[row]['image_path']}  (id {ids[row]['id']})")
        shown += 1
        if shown >= args.topk:
            break


if __name__ == "__main__":
    main()
