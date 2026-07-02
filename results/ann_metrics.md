# ANN Index Benchmark (recall vs. speed)

- Corpus: 5000 vectors, dim 512
- Ground truth: flat index exact top-10 neighbors
- recall@10: fraction of exact neighbors the ANN index also returns
- Queries: all 5000 corpus vectors; latency = best of 5 runs

| Index | recall@10 | µs / query | speedup vs flat |
|---|---|---|---|
| Flat (exact) | 1.000 | 17.7 | 1.00x |
| IVFFlat nprobe=1 | 0.693 | 3.3 | 5.42x |
| IVFFlat nprobe=5 | 0.935 | 9.4 | 1.89x |
| IVFFlat nprobe=10 | 0.974 | 17.0 | 1.04x |
| IVFFlat nprobe=20 | 0.992 | 31.8 | 0.56x |
| IVFFlat nprobe=50 | 1.000 | 83.8 | 0.21x |
| HNSWFlat efSearch=16 | 0.997 | 6.0 | 2.97x |
| HNSWFlat efSearch=32 | 0.999 | 9.9 | 1.78x |
| HNSWFlat efSearch=64 | 1.000 | 16.9 | 1.05x |
| HNSWFlat efSearch=128 | 1.000 | 33.1 | 0.54x |

## Reading it

Flat is the correctness baseline (recall = 1.000 by definition). IVF and HNSW recover most exact neighbors while touching far fewer vectors; turning up `nprobe` / `efSearch` buys recall back at the cost of speed.

**Scale caveat:** at N=5000 the exact scan is already ~microseconds, so measured speedups are small and timing is noisy — HNSW can even look slower here due to graph overhead. The tradeoff *shape* (recall traded for fewer comparisons) is what pays off at millions of vectors, where flat search becomes the bottleneck. Flat is kept as the exact baseline.
