"""Prepare GenGUI: unzip -> normalize -> deterministic splits -> YOLO + COCO export.

Usage:
    python scripts/prepare_gengui.py [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _prepare_common import PREPARED_DIR, RAW_DIR, report_dataset, unzip_if_needed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip", type=Path, default=RAW_DIR / "GenGUI.zip")
    args = ap.parse_args()

    from uidet.data.adapters.gengui_csv import load_gengui
    from uidet.data.exporters.to_coco import export_coco
    from uidet.data.exporters.to_yolo import export_yolo
    from uidet.data.splits import load_or_create_splits

    extract_root = PREPARED_DIR / "gengui_raw"
    gen_root = unzip_if_needed(args.zip, extract_root, "GenGUI")

    print("Loading GenGUI...")
    ds = load_gengui(gen_root)
    print(f"Loaded {len(ds.images)} images, "
          f"{sum(len(im.boxes) for im in ds.images)} boxes (after ignore-mapping).")

    out_root = PREPARED_DIR / "gengui"
    splits_path = out_root / "splits" / f"splits_seed{args.seed}.json"
    splits = load_or_create_splits(ds.images, splits_path, seed=args.seed)
    print(f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # gengui13 is the GenGUI-only native 13-class taxonomy (Dicu et al. ICAART 2024).
    # Kept alongside the cross-dataset unified3 / unified_ext exports.
    for tax in ("unified3", "unified_ext", "gengui13"):
        ds_tax = ds.filter_for_taxonomy(tax)
        report_dataset(ds_tax, tax)
        out_dir = out_root / tax
        yaml_path = export_yolo(ds_tax, splits, tax, out_dir)
        coco_paths = export_coco(ds_tax, splits, tax, out_dir)
        print(f"  YOLO -> {yaml_path}")
        for split, p in coco_paths.items():
            print(f"  COCO {split} -> {p}")


if __name__ == "__main__":
    main()
