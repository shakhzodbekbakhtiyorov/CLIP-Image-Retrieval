"""Build an exact FAISS IndexFlatIP from embeddings.npy (unit-norm -> IP == cosine)."""

import argparse
import os

import numpy as np
import faiss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".")
    p.add_argument("--emb", default="embeddings.npy")
    p.add_argument("--out", default="index_flat.faiss")
    args = p.parse_args()

    emb = np.load(os.path.join(args.data_dir, args.emb)).astype("float32")
    n, d = emb.shape

    # Guard: IndexFlatIP only equals cosine if rows are unit-norm.
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "embeddings are not unit-norm"

    index = faiss.IndexFlatIP(d)
    index.add(emb)
    assert index.ntotal == n

    out = os.path.join(args.data_dir, args.out)
    faiss.write_index(index, out)
    print(f"Built IndexFlatIP: {index.ntotal} vectors, dim {d} -> {out}")


if __name__ == "__main__":
    main()
