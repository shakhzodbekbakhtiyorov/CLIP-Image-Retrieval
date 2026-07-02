# Image -> Image Retrieval Metrics

- Corpus: 5000 images (COCO val2017)
- Valid queries (>=1 category label): 4952  (skipped 48 background-only images)
- Relevance: shares >=1 COCO category with the query
- Search: exact cosine (IndexFlatIP baseline)

| Metric | Value |
|---|---|
| mAP | 0.5528 |
| Precision@1 | 0.9245 |
| Precision@5 | 0.9063 |
| Precision@10 | 0.8912 |
| Median rank of first relevant hit | 1 |

## Caveat

Relevance is COCO category overlap, which is multi-label and fuzzy: two images sharing only a common category (e.g. "person") can look entirely different. These numbers are a consistent yardstick across experiments, not an absolute measure of visual similarity.
