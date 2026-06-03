"""Convert one of our prepared <dataset>/<taxonomy> dirs into RF-DETR's
expected Roboflow layout.

Source layout (what scripts/prepare_*.py produces):
    data/prepared/<ds>/<tax>/
        images/{train,val,test}/<id>.<ext>
        annotations/<ds>_<tax>_{train,val,test}.json
        data.yaml

RF-DETR / Roboflow layout (what this script writes):
    data/prepared_rfdetr/<ds>__<tax>/
        train/_annotations.coco.json   + <id>.<ext> files alongside
        valid/_annotations.coco.json   + <id>.<ext> files
        test/_annotations.coco.json    + <id>.<ext> files

The annotations' ``file_name`` fields are rewritten from
``images/<split>/<id>.<ext>`` to just ``<id>.<ext>`` so RF-DETR can find them.

Usage:
    python scripts/to_rfdetr_layout.py --dataset gengui --taxonomy gengui13
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _safe_remove(p: Path) -> None:
    """Delete ``p`` even if it's a broken/unreadable symlink on Windows.
    Required because ``Path.exists()`` on a dangling reparse point raises
    WinError 1920 instead of returning False."""
    try:
        p.unlink()
        return
    except FileNotFoundError:
        return
    except OSError:
        try:
            os.remove(p)
        except FileNotFoundError:
            return
        except OSError:
            pass


def _exists_safe(p: Path) -> bool:
    """Like Path.exists() but returns False on a broken symlink (WinError 1920)
    instead of raising."""
    try:
        return p.exists()
    except OSError:
        return False

REPO = Path(__file__).resolve().parents[1]
PREPARED = REPO / "data" / "prepared"
RFDETR_OUT = REPO / "data" / "prepared_rfdetr"


# Roboflow uses "valid" not "val"
_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


def convert(dataset: str, taxonomy: str, *, copy: bool = False) -> Path:
    src_root = PREPARED / dataset / taxonomy
    if not src_root.exists():
        raise FileNotFoundError(f"Source not found: {src_root}")
    coco_dir = src_root / "annotations"
    img_dir = src_root / "images"
    if not coco_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(
            f"Expected annotations/ and images/ under {src_root}"
        )

    out_root = RFDETR_OUT / f"{dataset}__{taxonomy}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Converting {src_root} -> {out_root}")

    for our_split in ("train", "val", "test"):
        rf_split = _SPLIT_MAP[our_split]
        out_split = out_root / rf_split
        out_split.mkdir(parents=True, exist_ok=True)

        # Find the matching COCO json (filename has the form
        # <dataset>_<taxonomy>_<split>.json)
        cands = list(coco_dir.glob(f"*_{our_split}.json"))
        if not cands:
            print(f"  WARN: no COCO json for split={our_split}; skipping")
            continue
        src_json = cands[0]
        coco = json.loads(src_json.read_text())

        # Rewrite file_name to be relative-to-split-dir (just <id>.<ext>)
        n_imgs = 0
        for im in coco.get("images", []):
            orig = Path(im["file_name"])
            im["file_name"] = orig.name
            n_imgs += 1

        # Write the rewritten annotations
        (out_split / "_annotations.coco.json").write_text(json.dumps(coco))

        # Copy or symlink images.
        # On Windows, broken symlinks from earlier runs make Path.exists()
        # raise; we tear those down first via _safe_remove.
        src_img_split = img_dir / our_split
        n_copied = 0
        for src_img in src_img_split.iterdir():
            if not src_img.is_file():
                continue
            dst_img = out_split / src_img.name
            if _exists_safe(dst_img):
                continue
            _safe_remove(dst_img)
            if copy or sys.platform == "win32":
                # Windows symlinks need elevation; copy by default there
                shutil.copy2(src_img, dst_img)
            else:
                try:
                    dst_img.symlink_to(src_img.resolve())
                except OSError:
                    shutil.copy2(src_img, dst_img)
            n_copied += 1

        print(f"  {our_split:5s} -> {rf_split:5s}: "
              f"{n_imgs} images in json, {n_copied} files staged")

    print(f"Done. RF-DETR dataset_dir = {out_root}")
    return out_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["uicvd", "gengui", "vins"])
    ap.add_argument("--taxonomy", required=True,
                    help="e.g. gengui13, unified3, unified_ext")
    ap.add_argument("--copy", action="store_true",
                    help="Force file copies (default: symlinks on POSIX, copies on Windows)")
    args = ap.parse_args()
    convert(args.dataset, args.taxonomy, copy=args.copy)


if __name__ == "__main__":
    main()
