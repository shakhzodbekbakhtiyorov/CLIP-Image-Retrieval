"""Gradio demo: dual-mode CLIP image retrieval (upload an image or type text).

Result images are fetched from the COCO CDN, so the Space ships only
embeddings.npy + ids.json, not the images.
"""

import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import open_clip
from PIL import Image
import gradio as gr

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
HERE = os.path.dirname(os.path.abspath(__file__))
COCO_URL = "http://images.cocodataset.org/val2017/{}"

# ---- load once at startup ----
corpus = np.load(os.path.join(HERE, "embeddings.npy")).astype("float32")
with open(os.path.join(HERE, "ids.json")) as f:
    IDS = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
_model, _, _preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED)
_model = _model.to(device).eval()
_tokenizer = open_clip.get_tokenizer(MODEL_NAME)


def _normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype("float32")


def encode_image(img: Image.Image) -> np.ndarray:
    x = _preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        v = _model.encode_image(x).cpu().numpy()
    return _normalize(v)[0]


def encode_text(text: str) -> np.ndarray:
    toks = _tokenizer([text]).to(device)
    with torch.no_grad():
        v = _model.encode_text(toks).cpu().numpy()
    return _normalize(v)[0]


def retrieve(query_vec: np.ndarray, k: int):
    """Return list of (row, score) for the top-k images by cosine."""
    scores = corpus @ query_vec
    rows = np.argpartition(-scores, range(k))[:k]
    rows = rows[np.argsort(-scores[rows])]
    return [(int(r), float(scores[r])) for r in rows]


_img_cache = {}


def fetch_image(url: str):
    """Download an image server-side (cached). Returns a PIL image or None.

    COCO is served from an S3 bucket whose TLS cert doesn't match the hostname,
    so https fails; we fetch over http here on the server. Fetching server-side
    also avoids browser mixed-content blocking when the Space is served on https.
    """
    if url in _img_cache:
        return _img_cache[url]
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    except Exception:
        img = None
    _img_cache[url] = img
    return img


def to_gallery(hits):
    """Map (row, score) hits to (PIL image, caption) for a gr.Gallery.

    Images are fetched by the server (not the browser), so the gallery works
    over https and doesn't depend on the COCO cert. Fetches run in parallel so
    the wall-clock cost is ~one round-trip, not k sequential ones."""
    urls = [COCO_URL.format(os.path.basename(IDS[r]["image_path"]))
            for r, _ in hits]
    with ThreadPoolExecutor(max_workers=8) as pool:
        imgs = list(pool.map(fetch_image, urls))
    out = []
    for rank, ((r, score), img) in enumerate(zip(hits, imgs), start=1):
        if img is not None:
            out.append((img, f"#{rank}  cos={score:.3f}  (id {IDS[r]['id']})"))
    return out


def search(image, text, k):
    if image is not None:
        q = encode_image(image)
        header = "Image query"
    elif text and text.strip():
        q = encode_text(text.strip())
        header = f'Text query: "{text.strip()}"'
    else:
        return [], "Upload an image or type a query to search."
    hits = retrieve(q, int(k))
    return to_gallery(hits), header


with gr.Blocks(title="CLIP Image Search (COCO val2017)") as demo:
    gr.Markdown(
        "# 🔎 CLIP Image Search — COCO val2017\n"
        "Search ~5,000 images by **uploading an image** or **typing a "
        "description**. Both use one OpenCLIP ViT-B/32 space. "
        "If both are given, the image wins.")
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Query image")
            text_in = gr.Textbox(label="…or text query",
                                 placeholder="a dog playing in the snow")
            k_in = gr.Slider(1, 20, value=8, step=1, label="Results (k)")
            go = gr.Button("Search", variant="primary")
            gr.Examples(
                examples=[
                    [None, "two people riding motorcycles", 8],
                    [None, "a plate of breakfast food", 8],
                    [None, "a red double-decker bus", 8],
                ],
                inputs=[image_in, text_in, k_in])
        with gr.Column(scale=2):
            header = gr.Markdown()
            gallery = gr.Gallery(label="Results", columns=4, height="auto")

    go.click(search, [image_in, text_in, k_in], [gallery, header])
    text_in.submit(search, [image_in, text_in, k_in], [gallery, header])

if __name__ == "__main__":
    demo.launch()
