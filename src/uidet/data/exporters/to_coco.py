"""Export a :class:`UnifiedDataset` to COCO JSON (one file per split).

Layout produced:

    <out_dir>/
        annotations/<dataset>_<taxonomy>_{train,val,test}.json
        images/         (shared with the YOLO export -- exporter assumes images
                         already exist there; if not, call export_yolo first or
                         copy the files yourself)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..unified import UnifiedDataset


def export_coco(
    dataset: UnifiedDataset,
    splits: dict[str, list[str]],
    taxonomy: str,
    out_dir: Path,
    image_subdir: str = "images",
) -> dict[str, Path]:
    """Write COCO JSONs per split. Returns dict {split_name: json_path}."""
    classes = dataset.classes_by_taxonomy.get(taxonomy)
    if not classes:
        raise ValueError(
            f"Taxonomy {taxonomy!r} not found in dataset.classes_by_taxonomy. "
            f"Available: {sorted(dataset.classes_by_taxonomy)}"
        )
    name_to_id = {c: i + 1 for i, c in enumerate(classes)}  # COCO uses 1-based cat ids

    by_id = {im.image_id: im for im in dataset.images}

    out_dir = out_dir.resolve()
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for split_name, ids in splits.items():
        coco = {
            "info": {
                "description": f"{dataset.name} ({taxonomy}) — {split_name} split",
                "version": "1.0",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": cid, "name": cname, "supercategory": "ui"}
                for cname, cid in name_to_id.items()
            ],
        }
        ann_id = 1
        for img_id_str in ids:
            img = by_id.get(img_id_str)
            if img is None:
                continue
            coco_img_id = len(coco["images"]) + 1
            ext = img.image_path.suffix or ".png"
            file_name = f"{image_subdir}/{split_name}/{img_id_str}{ext}"
            coco["images"].append({
                "id": coco_img_id,
                "file_name": file_name,
                "width": img.width,
                "height": img.height,
            })
            for b in img.boxes:
                cname = b.label_for(taxonomy)
                if cname is None:
                    continue
                w = b.width(); h = b.height()
                if w <= 0 or h <= 0:
                    continue
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": coco_img_id,
                    "category_id": name_to_id[cname],
                    "bbox": [b.xmin, b.ymin, w, h],     # COCO is xywh
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

        path = ann_dir / f"{dataset.name}_{taxonomy}_{split_name}.json"
        with path.open("w") as fh:
            json.dump(coco, fh)
        written[split_name] = path

    return written
