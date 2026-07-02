"""Text->image failure analysis: worst-ranked captions + compositional probes
(spatial/counting/negation) + a quantified attribute-binding swap test.
"""

import argparse
import json
import os

import numpy as np

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

# Compositional probe queries grouped by the phenomenon they stress.
PROBES = {
    "spatial relation": [
        "a cat to the left of a dog",
        "a person standing behind a car",
        "a bottle on top of a book",
    ],
    "counting": [
        "exactly three dogs",
        "two people and one bicycle",
    ],
    "negation": [
        "a street with no cars",
        "a plate with food but no fork",
    ],
}
# Attribute-binding swap pairs: same words, swapped colors/objects.
SWAP_PAIRS = [
    ("a red car and a blue bus", "a blue car and a red bus"),
    ("a white dog and a black cat", "a black dog and a white cat"),
    ("a man in a red shirt and a woman in a green shirt",
     "a man in a green shirt and a woman in a red shirt"),
]


def make_encoder():
    """Return a function texts -> (m, D) normalized float32, loading CLIP once."""
    import torch
    import open_clip

    device = "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    def encode(texts, batch_size=256):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                toks = tokenizer(texts[i:i + batch_size]).to(device)
                f = model.encode_text(toks)
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0)

    return encode


def topk_paths(scores_row, ids, k):
    rows = np.argpartition(-scores_row, range(k))[:k]
    rows = rows[np.argsort(-scores_row[rows])]
    return [(float(scores_row[r]), ids[r]["image_path"], ids[r]["id"]) for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="results/failure_analysis.md")
    p.add_argument("--n-worst", type=int, default=8)
    p.add_argument("--rank-thresh", type=int, default=100,
                   help="only count captions ranked worse than this as failures")
    args = p.parse_args()

    corpus = np.load(os.path.join(args.data_dir, "embeddings.npy")).astype("float32")
    ids = json.load(open(os.path.join(args.data_dir, "ids.json")))
    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    row_of_id = {r["id"]: i for i, r in enumerate(ids)}
    cats_by_id = {m["id"]: m["categories"] for m in manifest}

    encode = make_encoder()

    # ---- Part A: worst-ranked captions ----
    captions, gold_rows = [], []
    from caption_utils import good_captions
    for m in manifest:
        if m["id"] in row_of_id:
            good = good_captions(m["captions"])
            if good:
                captions.append(good[0])
                gold_rows.append(row_of_id[m["id"]])
    gold_rows = np.array(gold_rows)
    print(f"encoding {len(captions)} captions ...")
    cap_emb = encode(captions)

    ranks = np.empty(len(captions), dtype=np.int64)
    B = 512
    for s in range(0, len(captions), B):
        sims = cap_emb[s:s + B] @ corpus.T
        g = gold_rows[s:s + B]
        gs = sims[np.arange(len(g)), g][:, None]
        ranks[s:s + B] = 1 + (sims > gs).sum(axis=1)

    worst = np.argsort(-ranks)[:args.n_worst]
    n_fail = int((ranks > args.rank_thresh).sum())

    lines = [
        "# Failure Analysis (text -> image)",
        "",
        f"{n_fail} of {len(captions)} caption queries ranked their true image "
        f"worse than {args.rank_thresh} (mean rank {ranks.mean():.1f}, "
        f"median {np.median(ranks):.0f}).",
        "",
        "## A. Worst-ranked captions (semantic vs. visual mismatch)",
        "",
        "The true image is what the caption describes; the top results are what "
        "CLIP thought matched better. Mismatches show CLIP anchoring on dominant "
        "visual concepts and under-weighting the detail the caption singled out.",
        "",
    ]
    for qi in worst:
        gold_id = ids[gold_rows[qi]]["id"]
        sims = cap_emb[qi] @ corpus.T
        top = topk_paths(sims, ids, 3)
        lines.append(f'**"{captions[qi]}"**  (gold rank {ranks[qi]})')
        lines.append(f"- gold image: {ids[gold_rows[qi]]['image_path']} "
                     f"— categories: {', '.join(cats_by_id.get(gold_id, [])) or '(none)'}")
        for sc, path, iid in top:
            lines.append(f"- returned {sc:.3f}: {path} "
                         f"— {', '.join(cats_by_id.get(iid, [])) or '(none)'}")
        lines.append("")

    # ---- Part B: compositional probes ----
    lines += ["## B. Compositional probes", "",
              "Top-3 images CLIP returns for queries that require relations, "
              "counting, or negation. Inspect whether the results actually honor "
              "the constraint (they usually don't).", ""]
    for group, queries in PROBES.items():
        lines.append(f"### {group}")
        lines.append("")
        pe = encode(queries)
        for qi, qtext in enumerate(queries):
            sims = pe[qi] @ corpus.T
            top = topk_paths(sims, ids, 3)
            lines.append(f'**"{qtext}"**')
            for sc, path, iid in top:
                lines.append(f"- {sc:.3f}: {path} "
                             f"— {', '.join(cats_by_id.get(iid, [])) or '(none)'}")
            lines.append("")

    # ---- Part B2: attribute-binding swap pairs (quantified) ----
    lines += ["### attribute binding (swap test)", "",
              "If binding were understood, swapping the colors would change the "
              "results. Cosine near 1.0 and Jaccard near 1.0 mean CLIP treats the "
              "swapped queries as essentially the same — a binding failure.", "",
              "| Query A | Query B | query cosine | top-10 Jaccard |",
              "|---|---|---|---|"]
    for a, b in SWAP_PAIRS:
        e = encode([a, b])
        cos = float(e[0] @ e[1])
        sa = {r for _, _, r in topk_paths(e[0] @ corpus.T, ids, 10)}
        sb = {r for _, _, r in topk_paths(e[1] @ corpus.T, ids, 10)}
        jac = len(sa & sb) / len(sa | sb)
        lines.append(f"| {a} | {b} | {cos:.3f} | {jac:.2f} |")
    lines.append("")

    out = os.path.join(args.data_dir, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
