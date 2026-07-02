---
title: CLIP Image Search (COCO val2017)
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# CLIP Image Search — COCO val2017

Dual-mode content-based image retrieval over a ~5,000-image COCO val2017 slice,
built on a single OpenCLIP **ViT-B/32** (`openai`) backbone.

- **Image → image:** upload a photo, get the most visually/semantically similar images.
- **Text → image:** type a sentence, get matching images.

Both queries are encoded into the same 512-D CLIP space and ranked by cosine
similarity against precomputed image embeddings (`embeddings.npy`). Result images
are displayed from the COCO CDN, so this Space ships only the embeddings and the
id→path map — not the image files.

See the [main project repo](https://github.com/) for the full pipeline:
embedding, FAISS indexing, evaluation (image→image mAP/P@k, text→image
Recall@k), a de-duplication study, an ANN recall-vs-speed benchmark, and a
failure analysis of CLIP's compositional/negation weaknesses.
