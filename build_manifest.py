"""Build manifest.json from COCO annotations: per image id, path, categories, captions."""

import argparse
import json
import os
from collections import defaultdict


def build(data_dir: str, out_path: str) -> None:
    ann_dir = os.path.join(data_dir, "annotations")
    img_dir_name = "val2017"

    with open(os.path.join(ann_dir, "instances_val2017.json")) as f:
        inst = json.load(f)
    with open(os.path.join(ann_dir, "captions_val2017.json")) as f:
        caps = json.load(f)

    # id -> readable category name
    catname = {c["id"]: c["name"] for c in inst["categories"]}

    # image_id -> set of category ids
    cats_by_img = defaultdict(set)
    for a in inst["annotations"]:
        cats_by_img[a["image_id"]].add(a["category_id"])

    # image_id -> list of captions
    caps_by_img = defaultdict(list)
    for a in caps["annotations"]:
        caps_by_img[a["image_id"]].append(a["caption"].strip())

    records = []
    for img in inst["images"]:
        iid = img["id"]
        cat_ids = sorted(cats_by_img.get(iid, set()))
        records.append({
            "id": iid,
            "file_name": img["file_name"],
            "image_path": os.path.join(img_dir_name, img["file_name"]),
            "width": img["width"],
            "height": img["height"],
            "category_ids": cat_ids,
            "categories": [catname[c] for c in cat_ids],
            "captions": caps_by_img.get(iid, []),
        })

    records.sort(key=lambda r: r["id"])
    with open(out_path, "w") as f:
        json.dump(records, f)

    n = len(records)
    no_cat = sum(1 for r in records if not r["categories"])
    no_cap = sum(1 for r in records if not r["captions"])
    print(f"Wrote {n} records to {out_path}")
    print(f"  images with no category labels: {no_cat}")
    print(f"  images with no captions:        {no_cap}")
    print(f"  example: {json.dumps(records[0], ensure_ascii=False)[:200]}...")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".", help="repo root containing val2017/ and annotations/")
    p.add_argument("--out", default="manifest.json")
    build(*[p.parse_args().data_dir, p.parse_args().out])
