"""Build an exact FAISS flat index from the image embeddings.

Vectors are already L2-normalized, so inner product == cosine similarity.
IndexFlatIP does exact (brute-force) search -- no approximation. This is the
correctness baseline that the ANN index (step 8) will be measured against.

Input:  embeddings.npy  (N, D) float32, unit-norm
Output: index_flat.faiss

Row i of the index corresponds to ids.json[i] -- FAISS returns row indices,
which you map back to image paths via ids.json.
"""

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
