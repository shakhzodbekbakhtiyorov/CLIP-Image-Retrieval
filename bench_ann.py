"""ANN benchmark: recall@10 vs. speed for IVFFlat/HNSWFlat, ground truth = flat index."""

import argparse
import json
import os
import time

import numpy as np
import faiss

K = 10


def timed_search(index, queries, k, repeats=5):
    """Return (results_idx, mean_seconds_per_query) over repeats, best run."""
    best = float("inf")
    idx = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, idx = index.search(queries, k)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return idx, best / len(queries)


def recall_at_k(approx_idx, exact_idx):
    """Mean over queries of |approx_topk ∩ exact_topk| / k."""
    n, k = exact_idx.shape
    hits = 0
    for a, e in zip(approx_idx, exact_idx):
        hits += len(set(a.tolist()) & set(e.tolist()))
    return hits / (n * k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="results/ann_metrics.md")
    args = p.parse_args()

    emb = np.load(os.path.join(args.data_dir, "embeddings.npy")).astype("float32")
    n, d = emb.shape
    queries = emb  # every corpus vector is also a query

    # --- exact baseline (ground truth) ---
    flat = faiss.IndexFlatIP(d)
    flat.add(emb)
    exact_idx, flat_lat = timed_search(flat, queries, K)
    rows = [("Flat (exact)", 1.0, flat_lat, 1.0)]

    # --- IVFFlat: sweep nprobe ---
    nlist = 100
    quantizer = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(emb)
    ivf.add(emb)
    for nprobe in [1, 5, 10, 20, 50]:
        ivf.nprobe = nprobe
        aidx, lat = timed_search(ivf, queries, K)
        rows.append((f"IVFFlat nprobe={nprobe}", recall_at_k(aidx, exact_idx),
                     lat, flat_lat / lat))

    # --- HNSWFlat: sweep efSearch (L2 == IP ranking on unit-norm vectors) ---
    hnsw = faiss.IndexHNSWFlat(d, 32)     # M=32 graph connectivity
    hnsw.hnsw.efConstruction = 200
    hnsw.add(emb)
    for ef in [16, 32, 64, 128]:
        hnsw.hnsw.efSearch = ef
        aidx, lat = timed_search(hnsw, queries, K)
        rows.append((f"HNSWFlat efSearch={ef}", recall_at_k(aidx, exact_idx),
                     lat, flat_lat / lat))

    # --- report ---
    lines = [
        "# ANN Index Benchmark (recall vs. speed)",
        "",
        f"- Corpus: {n} vectors, dim {d}",
        f"- Ground truth: flat index exact top-{K} neighbors",
        f"- recall@{K}: fraction of exact neighbors the ANN index also returns",
        f"- Queries: all {n} corpus vectors; latency = best of 5 runs",
        "",
        f"| Index | recall@{K} | µs / query | speedup vs flat |",
        "|---|---|---|---|",
    ]
    for name, rec, lat, spd in rows:
        lines.append(f"| {name} | {rec:.3f} | {lat*1e6:.1f} | {spd:.2f}x |")
    lines += [
        "",
        "## Reading it",
        "",
        "Flat is the correctness baseline (recall = 1.000 by definition). IVF and "
        "HNSW recover most exact neighbors while touching far fewer vectors; "
        "turning up `nprobe` / `efSearch` buys recall back at the cost of speed.",
        "",
        f"**Scale caveat:** at N={n} the exact scan is already ~microseconds, so "
        "measured speedups are small and timing is noisy — HNSW can even look "
        "slower here due to graph overhead. The tradeoff *shape* (recall traded "
        "for fewer comparisons) is what pays off at millions of vectors, where "
        "flat search becomes the bottleneck. Flat is kept as the exact baseline.",
        "",
    ]
    out = os.path.join(args.data_dir, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines))

    for name, rec, lat, spd in rows:
        print(f"{name:22s} recall@{K}={rec:.3f}  {lat*1e6:7.1f} us/q  {spd:.2f}x")
    print("wrote", out)


if __name__ == "__main__":
    main()
