"""Thesis-grade COCO evaluation.

    python -m uidet.eval_coco --config configs/experiments/yolov8n_uicvd_uicvd16.yaml --split test

Runs the trained model over every image in the chosen split's COCO ground-truth
JSON, runs pycocotools COCOeval, and saves:
    results_v2/<exp>/coco_eval_<split>/predictions.json
    results_v2/<exp>/coco_eval_<split>/metrics.json   <- per-class + s/m/l + headlines
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .models.base import build_detector
from .train import REPO_ROOT, get_class_names, resolve_data_paths
from .utils.io import load_yaml
from .utils.metrics import coco_evaluate, detections_to_coco, format_metrics_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="Detection confidence threshold (low for full PR curve)")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    ap.add_argument("--batch", type=int, default=8, help="Images per inference batch")
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config)
    name     = cfg_dict["name"]
    model    = cfg_dict["model"]
    dataset  = cfg_dict["dataset"]
    taxonomy = cfg_dict["taxonomy"]

    # Honour results_dir from YAML (default results_v2, same as train.py).
    results_dir = REPO_ROOT / cfg_dict.get("results_dir", "results_v2")

    data_yaml, *_ = resolve_data_paths(dataset, taxonomy)
    class_names = get_class_names(data_yaml)
    name_to_cat_id = {c: i + 1 for i, c in enumerate(class_names)}   # COCO is 1-based

    weights = args.weights or (results_dir / name / "weights" / "best.pt")
    if not weights.exists():
        raise FileNotFoundError(f"Trained weights not found: {weights}")

    gt_path = _split_to_gt_path(dataset, taxonomy, args.split)
    gt = json.loads(Path(gt_path).read_text())
    gt_root = Path(gt_path).parents[1]      # ...prepared/<dataset>/<taxonomy>/

    # Build (image_id, image_path) list in COCO id order
    items = [(im["id"], gt_root / im["file_name"]) for im in gt["images"]]
    print(f"Evaluating {model} on {dataset}/{taxonomy} {args.split}: "
          f"{len(items)} images, {len(gt['annotations'])} GT boxes")

    det = build_detector(model, num_classes=len(class_names), class_names=class_names)
    det.load(weights)

    predictions: list[dict] = []
    t0 = time.perf_counter()
    for i in range(0, len(items), args.batch):
        chunk = items[i:i + args.batch]
        paths = [p for _, p in chunk]
        ids   = [iid for iid, _ in chunk]
        batch_dets = det.predict_batch(paths, conf=args.conf, iou=args.iou)
        for image_id, dets in zip(ids, batch_dets):
            predictions.extend(detections_to_coco(dets, image_id, name_to_cat_id))
        if (i // args.batch) % 10 == 0:
            print(f"  {min(i + args.batch, len(items))}/{len(items)} images, "
                  f"{len(predictions)} predictions so far")
    dt  = time.perf_counter() - t0
    fps = len(items) / dt if dt > 0 else 0.0
    print(f"Inference done: {len(items)} images in {dt:.1f}s ({fps:.1f} img/s)")

    out_dir = results_dir / name / f"coco_eval_{args.split}"
    metrics = coco_evaluate(gt_path, predictions, out_dir)
    metrics["inference_seconds"] = dt
    metrics["inference_fps"]     = fps
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print()
    print(format_metrics_table(metrics))
    print(f"\nWrote {out_dir / 'metrics.json'}")
    print(f"Wrote {out_dir / 'predictions.json'}")


def _split_to_gt_path(dataset: str, taxonomy: str, split: str) -> Path:
    from .train import PREPARED_DIR
    p = PREPARED_DIR / dataset / taxonomy / "annotations" / f"{dataset}_{taxonomy}_{split}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


if __name__ == "__main__":
    main()
