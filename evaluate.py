"""Leave-one-out retrieval evaluation for image->image search."""

import argparse
import json
import os

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="results/metrics.md")
    p.add_argument("--ks", default="1,5,10")
    args = p.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    emb = np.load(os.path.join(args.data_dir, "embeddings.npy")).astype("float32")
    with open(os.path.join(args.data_dir, "ids.json")) as f:
        ids = json.load(f)
    with open(os.path.join(args.data_dir, "manifest.json")) as f:
        manifest = json.load(f)

    cats_by_id = {m["id"]: set(m["category_ids"]) for m in manifest}
    # category set per embedding row, aligned to emb
    row_cats = [cats_by_id.get(r["id"], set()) for r in ids]
    n = len(row_cats)

    # queries with no category labels have no ground truth -> skip them
    valid = [i for i in range(n) if row_cats[i]]

    sims_all = emb @ emb.T          # (n, n) cosine matrix; diagonal = self = 1.0
    np.fill_diagonal(sims_all, -np.inf)   # exclude self-match

    p_at = {k: [] for k in ks}
    aps = []
    first_ranks = []

    for i in valid:
        order = np.argsort(-sims_all[i])          # rows, best first
        q = row_cats[i]
        rel = np.fromiter((1 if row_cats[j] & q else 0 for j in order),
                          dtype=np.int32, count=n)

        for k in ks:
            p_at[k].append(rel[:k].mean())

        total_rel = int(rel.sum())
        if total_rel:
            hit_positions = np.flatnonzero(rel) + 1          # 1-based ranks
            first_ranks.append(int(hit_positions[0]))
            cum = np.cumsum(rel)
            precisions = cum[hit_positions - 1] / hit_positions
            aps.append(precisions.mean())                     # AP for this query

    mAP = float(np.mean(aps))
    med_first = float(np.median(first_ranks))
    p_means = {k: float(np.mean(p_at[k])) for k in ks}

    lines = [
        "# Image -> Image Retrieval Metrics",
        "",
        f"- Corpus: {n} images (COCO val2017)",
        f"- Valid queries (>=1 category label): {len(valid)}  "
        f"(skipped {n - len(valid)} background-only images)",
        "- Relevance: shares >=1 COCO category with the query",
        "- Search: exact cosine (IndexFlatIP baseline)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| mAP | {mAP:.4f} |",
    ]
    for k in ks:
        lines.append(f"| Precision@{k} | {p_means[k]:.4f} |")
    lines += [
        f"| Median rank of first relevant hit | {med_first:.0f} |",
        "",
        "## Caveat",
        "",
        "Relevance is COCO category overlap, which is multi-label and fuzzy: "
        "two images sharing only a common category (e.g. \"person\") can look "
        "entirely different. These numbers are a consistent yardstick across "
        "experiments, not an absolute measure of visual similarity.",
        "",
    ]
    out = os.path.join(args.data_dir, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines))

    print(f"mAP={mAP:.4f}  " + "  ".join(f"P@{k}={p_means[k]:.4f}" for k in ks)
          + f"  median_first_hit={med_first:.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
