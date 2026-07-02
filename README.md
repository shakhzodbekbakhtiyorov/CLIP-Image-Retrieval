# CLIP Image Retrieval (COCO val2017)

A content-based image retrieval system over the COCO val2017 slice (~5k images),
built on a single CLIP backbone. Phase 1 is image→image search; the same joint
embedding space later powers text→image search (phase 2) with no re-indexing.

**Backbone:** OpenCLIP `ViT-B/32` (`openai` weights). Fixed — the image and text
encoders share one space, which is what makes phase 2 free.

## Pipeline

```
COCO annotations ──build_manifest.py──> manifest.json   (paths, categories, captions)
val2017 images   ──embed_images.py────> embeddings.npy + ids.json   (unit-norm vectors)
embeddings.npy   ──build_index.py─────> index_flat.faiss   (IndexFlatIP = exact cosine)
query image      ──search.py──────────> top-k similar images
```

Vectors are L2-normalized, so inner product == cosine similarity, and
`IndexFlatIP` does exact brute-force search (the correctness baseline for the
ANN index added later).

## Setup

```bash
python -m venv cvenv && source cvenv/bin/activate
pip install -r requirements.txt
```

Then download the COCO val2017 images into `val2017/` and the annotations into
`annotations/` (`instances_val2017.json`, `captions_val2017.json`).

## Run

```bash
python build_manifest.py            # -> manifest.json (5000 records)
python embed_images.py              # -> embeddings.npy + ids.json (first run downloads weights)
python build_index.py               # -> index_flat.faiss

# image -> image search
python search.py --id 139 -k 5            # query by a dataset image id
python search.py --image path/to/pic.jpg  # query by an arbitrary image
```

## Status

- [x] 1–2. Setup + COCO slice + manifest
- [x] 3. Image embeddings (unit-norm, sanity-checked)
- [x] 4. Flat FAISS index (exact cosine)
- [x] 5. Image→image search — **working milestone**
- [ ] 6. Evaluation (mAP, P@k, median rank)
- [ ] 7. De-dup study
- [ ] 8. ANN index (recall vs speedup)
- [ ] 9–11. Text encoder + text→image + eval
- [ ] 12. Failure analysis
- [ ] 13. Gradio demo
- [ ] 14. README pass

## Caveats

- Relevance is defined by **COCO category overlap**, which is multi-label and
  fuzzy: two images sharing "person" may be visually very different. Metrics
  inherit this noise.
- 48 of 5000 images carry no category labels (background-only); they have no
  relevance ground truth for category-overlap evaluation.
- CLIP was pretrained on large web data and may have seen COCO-like images, so
  retrieval quality is an optimistic estimate.
