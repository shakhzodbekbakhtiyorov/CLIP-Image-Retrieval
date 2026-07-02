"""Near-duplicate study: flag cosine>threshold pairs, drop dup clusters,
re-run leave-one-out eval on full vs. deduped corpus. The gap is the finding.
"""

import argparse
import json
import os

import numpy as np


def evaluate(emb, row_cats, keep_idx, ks):
    """Leave-one-out eval restricted to rows in keep_idx (both as queries and
    as retrievable items). Returns dict of metrics."""
    sub = np.asarray(keep_idx)
    E = emb[sub]
    cats = [row_cats[i] for i in sub]
    n = len(sub)

    sims = E @ E.T
    np.fill_diagonal(sims, -np.inf)

    p_at = {k: [] for k in ks}
    aps, first_ranks = [], []
    for i in range(n):
        q = cats[i]
        if not q:
            continue
        order = np.argsort(-sims[i])
        rel = np.fromiter((1 if cats[j] & q else 0 for j in order),
                          dtype=np.int32, count=n)
        for k in ks:
            p_at[k].append(rel[:k].mean())
        if rel.sum():
            hits = np.flatnonzero(rel) + 1
            first_ranks.append(int(hits[0]))
            cum = np.cumsum(rel)
            aps.append((cum[hits - 1] / hits).mean())

    return {
        "n_queries": len(aps),
        "mAP": float(np.mean(aps)),
        "P": {k: float(np.mean(p_at[k])) for k in ks},
        "median_first": float(np.median(first_ranks)),
    }


def find_duplicate_clusters(emb, threshold):
    """Union-find over pairs with cosine > threshold. Returns (remove_set,
    n_pairs, n_clusters). Keeps the lowest row index in each cluster."""
    n = len(emb)
    sims = emb @ emb.T
    iu = np.triu_indices(n, k=1)
    pair_mask = sims[iu] > threshold
    a = iu[0][pair_mask]
    b = iu[1][pair_mask]

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(a.tolist(), b.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    # any node that participated in a duplicate pair
    involved = set(a.tolist()) | set(b.tolist())
    clusters = {}
    for x in involved:
        clusters.setdefault(find(x), []).append(x)

    remove = set()
    for members in clusters.values():
        members.sort()
        remove.update(members[1:])   # keep the first, drop the rest
    return remove, int(pair_mask.sum()), len(clusters)


def fmt(m, ks):
    return (f"mAP={m['mAP']:.4f}  "
            + "  ".join(f"P@{k}={m['P'][k]:.4f}" for k in ks)
            + f"  median_first_hit={m['median_first']:.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="results/dedup_metrics.md")
    p.add_argument("--threshold", type=float, default=0.98)
    p.add_argument("--ks", default="1,5,10")
    args = p.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    emb = np.load(os.path.join(args.data_dir, "embeddings.npy")).astype("float32")
    ids = json.load(open(os.path.join(args.data_dir, "ids.json")))
    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    cats_by_id = {m["id"]: set(m["category_ids"]) for m in manifest}
    row_cats = [cats_by_id.get(r["id"], set()) for r in ids]
    n = len(emb)

    remove, n_pairs, n_clusters = find_duplicate_clusters(emb, args.threshold)
    keep_full = list(range(n))
    keep_dedup = [i for i in range(n) if i not in remove]

    m_full = evaluate(emb, row_cats, keep_full, ks)
    m_dedup = evaluate(emb, row_cats, keep_dedup, ks)

    print(f"threshold={args.threshold}  dup pairs={n_pairs}  "
          f"clusters={n_clusters}  images removed={len(remove)}")
    print("full  :", fmt(m_full, ks))
    print("dedup :", fmt(m_dedup, ks))

    lines = [
        "# De-duplication Study",
        "",
        f"- Near-duplicate rule: cosine > {args.threshold}",
        f"- Duplicate pairs found: {n_pairs}",
        f"- Duplicate clusters: {n_clusters}  (kept 1 image each)",
        f"- Images removed: {len(remove)}  ({n} -> {len(keep_dedup)})",
        "",
        "| Metric | Full corpus | Duplicates removed |",
        "|---|---|---|",
        f"| Queries | {m_full['n_queries']} | {m_dedup['n_queries']} |",
        f"| mAP | {m_full['mAP']:.4f} | {m_dedup['mAP']:.4f} |",
    ]
    for k in ks:
        lines.append(
            f"| Precision@{k} | {m_full['P'][k]:.4f} | {m_dedup['P'][k]:.4f} |")
    lines += [
        f"| Median first-hit rank | {m_full['median_first']:.0f} "
        f"| {m_dedup['median_first']:.0f} |",
        "",
        "## Finding",
        "",
        "Near-duplicates are trivially easy top hits, so leaving them in "
        "**inflates** precision and mAP. Removing them gives a more honest "
        "estimate of semantic retrieval quality; the drop is the size of that "
        "inflation. Category overlap remains a coarse, multi-label relevance "
        "signal (see metrics.md), so both columns share that caveat.",
        "",
    ]
    out = os.path.join(args.data_dir, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
