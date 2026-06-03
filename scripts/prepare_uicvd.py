"""Prepare UICVD: unzip -> normalize -> deterministic splits -> YOLO + COCO export.

Usage:
    python scripts/prepare_uicvd.py [--seed 42]

Outputs land in ``data/prepared/uicvd/{unified3,unified_ext}/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _prepare_common import PREPARED_DIR, RAW_DIR, report_dataset, unzip_if_needed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip", type=Path, default=RAW_DIR / "UICVD.zip")
    args = ap.parse_args()

    # Lazy imports so missing optional deps don't break --help
    from uidet.data.adapters.uicvd_csv import load_uicvd
    from uidet.data.exporters.to_coco import export_coco
    from uidet.data.exporters.to_yolo import export_yolo
    from uidet.data.splits import load_or_create_splits

    extract_root = PREPARED_DIR / "uicvd_raw"
    uicvd_root = unzip_if_needed(args.zip, extract_root, "UICVD")

    print("Loading UICVD...")
    ds = load_uicvd(uicvd_root)
    print(f"Loaded {len(ds.images)} images, "
          f"{sum(len(im.boxes) for im in ds.images)} boxes (after ignore-mapping).")

    out_root = PREPARED_DIR / "uicvd"
    splits_path = out_root / "splits" / f"splits_seed{args.seed}.json"
    splits = load_or_create_splits(ds.images, splits_path, seed=args.seed)
    print(f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    for tax in ("unified3", "unified_ext", "uicvd16"):
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
