"""Generate README figures: query->results image strip + ANN recall-vs-speed plot.

Outputs to results/figures/. Needs pillow (strip) and matplotlib + faiss (plot).
"""

import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

THUMB = 160
PAD = 6


def load(data_dir):
    emb = np.load(os.path.join(data_dir, "embeddings.npy")).astype("float32")
    ids = json.load(open(os.path.join(data_dir, "ids.json")))
    return emb, ids


def thumb(path, box=False):
    img = Image.open(path).convert("RGB")
    img = img.resize((THUMB, THUMB))
    if box:  # mark the query with a red border
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, THUMB - 1, THUMB - 1], outline=(220, 30, 30), width=5)
    return img


def strip(data_dir, out, query_ids, k=5):
    emb, ids = load(data_dir)
    id_to_row = {r["id"]: i for i, r in enumerate(ids)}
    rows = [id_to_row[q] for q in query_ids if q in id_to_row]

    n_cols = 1 + k
    W = n_cols * THUMB + (n_cols + 1) * PAD + THUMB // 2  # extra gap after query
    H = len(rows) * THUMB + (len(rows) + 1) * PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    for ri, row in enumerate(rows):
        scores = emb @ emb[row]
        order = np.argsort(-scores)
        neigh = [o for o in order if o != row][:k]
        y = PAD + ri * (THUMB + PAD)
        # query
        qpath = os.path.join(data_dir, ids[row]["image_path"])
        canvas.paste(thumb(qpath, box=True), (PAD, y))
        # results (after a small gap)
        x0 = PAD + THUMB + PAD + THUMB // 2
        for ci, o in enumerate(neigh):
            rpath = os.path.join(data_dir, ids[o]["image_path"])
            canvas.paste(thumb(rpath), (x0 + ci * (THUMB + PAD), y))
    canvas.save(out)
    print("wrote", out)


def ann_plot(data_dir, out):
    import faiss
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb, _ = load(data_dir)
    n, d = emb.shape
    q = emb
    K = 10

    def lat(index):
        best = 1e9
        for _ in range(5):
            t = time.perf_counter()
            _, idx = index.search(q, K)
            best = min(best, time.perf_counter() - t)
        return idx, best / n

    flat = faiss.IndexFlatIP(d); flat.add(emb)
    exact, flat_lat = lat(flat)

    def recall(idx):
        return np.mean([len(set(a) & set(e)) / K
                        for a, e in zip(idx.tolist(), exact.tolist())])

    pts = {"IVFFlat": [], "HNSWFlat": []}
    quant = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quant, d, 100, faiss.METRIC_INNER_PRODUCT)
    ivf.train(emb); ivf.add(emb)
    for np_ in [1, 5, 10, 20, 50]:
        ivf.nprobe = np_
        idx, l = lat(ivf)
        pts["IVFFlat"].append((flat_lat / l, recall(idx)))
    hnsw = faiss.IndexHNSWFlat(d, 32); hnsw.hnsw.efConstruction = 200; hnsw.add(emb)
    for ef in [16, 32, 64, 128]:
        hnsw.hnsw.efSearch = ef
        idx, l = lat(hnsw)
        pts["HNSWFlat"].append((flat_lat / l, recall(idx)))

    plt.figure(figsize=(6, 4))
    for name, ps in pts.items():
        ps = sorted(ps)
        xs, ys = zip(*ps)
        plt.plot(xs, ys, "o-", label=name)
    plt.axhline(1.0, ls="--", c="gray", lw=1, label="Flat (exact)")
    plt.xlabel("speedup vs. flat (x)")
    plt.ylabel("recall@10")
    plt.title(f"ANN recall vs. speed (N={n})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=130)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out-dir", default="results/figures")
    # a few visually distinct query images from val2017
    p.add_argument("--query-ids", default="139,285,724,776,785")
    args = p.parse_args()
    os.makedirs(os.path.join(args.data_dir, args.out_dir), exist_ok=True)

    qids = [int(x) for x in args.query_ids.split(",")]
    strip(args.data_dir, os.path.join(args.data_dir, args.out_dir,
          "query_results_strip.png"), qids)
    try:
        ann_plot(args.data_dir, os.path.join(args.data_dir, args.out_dir,
                 "ann_tradeoff.png"))
    except Exception as e:  # matplotlib/faiss optional
        print("skipped ANN plot:", e)


if __name__ == "__main__":
    main()
