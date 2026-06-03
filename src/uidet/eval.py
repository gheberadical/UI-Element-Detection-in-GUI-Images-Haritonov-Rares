"""Evaluation entry point.

    python -m uidet.eval --config configs/experiments/yolov8n_uicvd_unified3.yaml --split test

Loads the weights produced by training (results/<name>/weights/best.pt) and
runs the model's evaluate() on the requested split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models.base import TrainConfig, build_detector
from .train import RESULTS_DIR, get_class_names, resolve_data_paths
from .utils.io import dump_yaml, load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--weights", type=Path, default=None,
                    help="Override weights path (defaults to results/<name>/weights/best.pt)")
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config)
    name = cfg_dict["name"]
    model = cfg_dict["model"]
    dataset = cfg_dict["dataset"]
    taxonomy = cfg_dict["taxonomy"]

    data_yaml, val_json, test_json = resolve_data_paths(dataset, taxonomy)
    class_names = get_class_names(data_yaml)
    out_dir = RESULTS_DIR / name

    weights = args.weights or (out_dir / "weights" / "best.pt")
    if not weights.exists():
        raise FileNotFoundError(weights)

    cfg = TrainConfig(
        name=name, out_dir=out_dir, data_yaml=data_yaml,
        coco_val_json=val_json, coco_test_json=test_json,
        epochs=1, batch=int(cfg_dict.get("batch", 8)),
        imgsz=int(cfg_dict.get("imgsz", 640)), seed=int(cfg_dict.get("seed", 42)),
        device=str(cfg_dict.get("device", "0")),
    )

    detector = build_detector(model, num_classes=len(class_names), class_names=class_names)
    detector.load(weights)

    metrics = detector.evaluate(cfg, split=args.split)
    print(json.dumps(metrics, indent=2))
    dump_yaml({"split": args.split, "metrics": metrics},
              out_dir / f"metrics_{args.split}.yaml")


if __name__ == "__main__":
    main()
