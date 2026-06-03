"""Prepare VINS: unzip -> normalize -> deterministic splits -> YOLO + COCO export.

Usage:
    python scripts/prepare_vins.py [--seed 42] [--subsets Android Rico iphone uplabs Wireframes]

By default the five subsets (Android, Rico, iphone, uplabs, Wireframes) are
loaded together as one ``vins`` dataset, and per-subset prepared exports are
also produced under ``data/prepared/vins_<subset>/`` for slicing experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _prepare_common import PREPARED_DIR, RAW_DIR, report_dataset, unzip_if_needed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip", type=Path, default=RAW_DIR / "VINS Dataset.zip")
    ap.add_argument(
        "--subsets",
        nargs="+",
        default=["Android", "Rico", "iphone", "uplabs", "Wireframes"],
    )
    ap.add_argument(
        "--per-subset",
        action="store_true",
        help="Also export each subset on its own (in addition to the merged 'vins').",
    )
    args = ap.parse_args()

    from uidet.data.adapters.vins_pascal_voc import load_vins, load_vins_subset
    from uidet.data.exporters.to_coco import export_coco
    from uidet.data.exporters.to_yolo import export_yolo
    from uidet.data.splits import load_or_create_splits

    extract_root = PREPARED_DIR / "vins_raw"
    vins_root = unzip_if_needed(args.zip, extract_root, "All Dataset")

    # Verify the extract is actually complete. The earlier `unzip_if_needed`
    # only checks that "All Dataset" exists; a partial extract (e.g. interrupted
    # by a timeout) can leave it half-populated and silently shrink the dataset.
    incomplete: list[str] = []
    for sub in args.subsets:
        ann_dir = vins_root / sub / "Annotations"
        img_dir = vins_root / sub / "JPEGImages"
        n_ann = len(list(ann_dir.glob("*.xml"))) if ann_dir.exists() else 0
        n_img = len(list(img_dir.iterdir())) if img_dir.exists() else 0
        if n_ann == 0 or n_img == 0:
            incomplete.append(f"{sub} (ann={n_ann}, img={n_img})")
    if incomplete:
        raise RuntimeError(
            "VINS extract appears incomplete: " + ", ".join(incomplete) +
            f"\nDelete {extract_root} and re-run this script to force a clean unzip."
        )

    def export_one(ds, name: str) -> None:
        print(f"\n=== {name} ===")
        print(f"Loaded {len(ds.images)} images, "
              f"{sum(len(im.boxes) for im in ds.images)} boxes.")
        out_root = PREPARED_DIR / name
        splits_path = out_root / "splits" / f"splits_seed{args.seed}.json"
        splits = load_or_create_splits(ds.images, splits_path, seed=args.seed)
        print(f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

        for tax in ("unified3", "unified_ext", "vins13"):
            ds_tax = ds.filter_for_taxonomy(tax)
            report_dataset(ds_tax, tax)
            out_dir = out_root / tax
            yaml_path = export_yolo(ds_tax, splits, tax, out_dir)
            coco_paths = export_coco(ds_tax, splits, tax, out_dir)
            print(f"  YOLO -> {yaml_path}")
            for split, p in coco_paths.items():
                print(f"  COCO {split} -> {p}")

    print("Loading combined VINS...")
    ds_all = load_vins(vins_root, subsets=tuple(args.subsets))
    export_one(ds_all, "vins")

    if args.per_subset:
        for sub in args.subsets:
            ds_sub = load_vins_subset(vins_root, sub)
            export_one(ds_sub, f"vins_{sub.lower()}")


if __name__ == "__main__":
    main()
