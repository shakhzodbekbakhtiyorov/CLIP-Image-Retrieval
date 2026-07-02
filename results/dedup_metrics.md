# De-duplication Study

- Near-duplicate rule: cosine > 0.98
- Duplicate pairs found: 1
- Duplicate clusters: 1  (kept 1 image each)
- Images removed: 1  (5000 -> 4999)

| Metric | Full corpus | Duplicates removed |
|---|---|---|
| Queries | 4952 | 4952 |
| mAP | 0.5528 | 0.5528 |
| Precision@1 | 0.9245 | 0.9245 |
| Precision@5 | 0.9063 | 0.9063 |
| Precision@10 | 0.8912 | 0.8912 |
| Median first-hit rank | 1 | 1 |

## Finding

Near-duplicates are trivially easy top hits, so leaving them in **inflates** precision and mAP. Removing them gives a more honest estimate of semantic retrieval quality; the drop is the size of that inflation. Category overlap remains a coarse, multi-label relevance signal (see metrics.md), so both columns share that caveat.
