# Text -> Image Retrieval Metrics

- Corpus: 5000 images (COCO val2017)
- Queries: 5000 true captions (1 per image; 14 degenerate captions filtered out)
- Gold: the caption's source image (exactly one correct answer)
- Search: exact cosine over CLIP joint space

| Metric | Value |
|---|---|
| Recall@1 | 0.2900 |
| Recall@5 | 0.5288 |
| Recall@10 | 0.6282 |
| Median rank | 5 |
| Mean rank | 27.5 |

## Caveat

One caption can legitimately describe many images, but only the source image is credited as correct. When another valid image outranks the gold, it is scored as a miss, so Recall here **undercounts** true retrieval quality. Text->image cosine scores are also absolutely lower than image->image due to CLIP's modality gap; only the ranking matters.
