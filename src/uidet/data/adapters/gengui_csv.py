"""Adapter for the GenGUI dataset (CSV annotations).

Source layout (after unzipping ``datasets/GenGUI.zip``):

    GenGUI/
        annotations.csv      header: image_id,class,subclass,xmin,ymin,xmax,ymax
        images/Image*.png

250 images, 13 native classes, 20 484 boxes.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from ..unified import UnifiedBox, UnifiedDataset, UnifiedImage
from .common import load_all_taxonomies, load_mapping, map_native_to_all


def load_gengui(root: Path) -> UnifiedDataset:
    csv_path = root / "annotations.csv"
    images_dir = root / "images"
    if not csv_path.exists():
        raise FileNotFoundError(f"GenGUI annotations.csv not found at {csv_path}")

    mapping = load_mapping("gengui")
    taxonomies = load_all_taxonomies()

    by_id: dict[str, UnifiedImage] = {}

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_id = row["image_id"].strip()
            native = row["class"].strip()
            try:
                xmin = float(row["xmin"]); ymin = float(row["ymin"])
                xmax = float(row["xmax"]); ymax = float(row["ymax"])
            except (KeyError, ValueError):
                continue

            labels = map_native_to_all(native, mapping)
            if not any(v is not None for v in labels.values()):
                continue

            if image_id not in by_id:
                img_path = images_dir / f"{image_id}.png"
                if not img_path.exists():
                    cand = list(images_dir.glob(f"{image_id}.*"))
                    if not cand:
                        continue
                    img_path = cand[0]
                with Image.open(img_path) as im:
                    w, h = im.size
                by_id[image_id] = UnifiedImage(
                    image_id=image_id,
                    image_path=img_path,
                    width=w,
                    height=h,
                    boxes=[],
                    source_dataset="gengui",
                )

            box = UnifiedBox(
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                native_class=native, labels=labels,
            )
            host = by_id[image_id]
            if box.is_valid(host.width, host.height):
                host.boxes.append(box)

    images = sorted(by_id.values(), key=lambda im: im.image_id)
    return UnifiedDataset(
        name="gengui",
        images=images,
        classes_by_taxonomy=taxonomies,
    )
