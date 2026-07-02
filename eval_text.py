"""Text->image eval: each image's true caption as query, gold = that image.

Reports Recall@{1,5,10} + median/mean rank. Recall undercounts, since one caption
can fit many images but only the source is credited.
"""

import argparse
import json
import os

import numpy as np

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def embed_text(queries, batch_size=256):
    """CLIP text encoder -> L2-normalize. Returns (m, D) float32."""
    import torch
    import open_clip

    device = "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    out = []
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            toks = tokenizer(queries[i:i + batch_size]).to(device)
            f = model.encode_text(toks)
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy().astype("float32"))
            print(f"  embedded captions {min(i + batch_size, len(queries))}/{len(queries)}",
                  end="\r")
    print()
    return np.concatenate(out, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="results/text_metrics.md")
    p.add_argument("--ks", default="1,5,10")
    p.add_argument("--per-image", type=int, default=1,
                   help="captions to use per image as queries (default 1)")
    args = p.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    corpus = np.load(os.path.join(args.data_dir, "embeddings.npy")).astype("float32")
    with open(os.path.join(args.data_dir, "ids.json")) as f:
        ids = json.load(f)
    with open(os.path.join(args.data_dir, "manifest.json")) as f:
        manifest = json.load(f)

    row_of_id = {r["id"]: i for i, r in enumerate(ids)}

    # build (caption, gold_row) pairs, dropping contaminated/degenerate captions
    from caption_utils import good_captions
    captions, gold_rows, dropped = [], [], 0
    for m in manifest:
        if m["id"] not in row_of_id:
            continue
        good = good_captions(m["captions"])
        dropped += len(m["captions"]) - len(good)
        for cap in good[:args.per_image]:
            captions.append(cap)
            gold_rows.append(row_of_id[m["id"]])
    gold_rows = np.array(gold_rows)
    print(f"dropped {dropped} low-quality captions via filter")
    print(f"{len(captions)} caption queries over {corpus.shape[0]} images")

    text_emb = embed_text(captions)          # (Q, D)

    # rank of the gold image for each query, without materializing full argsort
    ranks = np.empty(len(captions), dtype=np.int64)
    B = 512
    for s in range(0, len(captions), B):
        sims = text_emb[s:s + B] @ corpus.T          # (b, N)
        gold = gold_rows[s:s + B]
        gold_score = sims[np.arange(len(gold)), gold][:, None]
        # rank = 1 + (# images strictly more similar than the gold)
        ranks[s:s + B] = 1 + (sims > gold_score).sum(axis=1)

    recalls = {k: float((ranks <= k).mean()) for k in ks}
    med_rank = float(np.median(ranks))
    mean_rank = float(np.mean(ranks))

    lines = [
        "# Text -> Image Retrieval Metrics",
        "",
        f"- Corpus: {corpus.shape[0]} images (COCO val2017)",
        f"- Queries: {len(captions)} true captions "
        f"({args.per_image} per image; {dropped} degenerate captions filtered out)",
        "- Gold: the caption's source image (exactly one correct answer)",
        "- Search: exact cosine over CLIP joint space",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k in ks:
        lines.append(f"| Recall@{k} | {recalls[k]:.4f} |")
    lines += [
        f"| Median rank | {med_rank:.0f} |",
        f"| Mean rank | {mean_rank:.1f} |",
        "",
        "## Caveat",
        "",
        "One caption can legitimately describe many images, but only the source "
        "image is credited as correct. When another valid image outranks the "
        "gold, it is scored as a miss, so Recall here **undercounts** true "
        "retrieval quality. Text->image cosine scores are also absolutely lower "
        "than image->image due to CLIP's modality gap; only the ranking matters.",
        "",
    ]
    out = os.path.join(args.data_dir, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines))

    print("  ".join(f"R@{k}={recalls[k]:.4f}" for k in ks)
          + f"  median_rank={med_rank:.0f}  mean_rank={mean_rank:.1f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
