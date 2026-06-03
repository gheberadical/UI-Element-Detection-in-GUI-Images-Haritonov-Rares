"""Single-image inference + visualization.

    python -m uidet.predict --config configs/experiments/yolov8n_uicvd_unified3.yaml \
                            --image path/to/screenshot.png --out predictions.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models.base import build_detector
from .train import RESULTS_DIR, get_class_names, resolve_data_paths
from .utils.io import load_yaml
from .utils.viz import draw_detections


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("prediction.png"))
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config)
    name = cfg_dict["name"]
    model = cfg_dict["model"]
    dataset = cfg_dict["dataset"]
    taxonomy = cfg_dict["taxonomy"]

    data_yaml, *_ = resolve_data_paths(dataset, taxonomy)
    class_names = get_class_names(data_yaml)

    weights = args.weights or (RESULTS_DIR / name / "weights" / "best.pt")
    detector = build_detector(model, num_classes=len(class_names), class_names=class_names)
    detector.load(weights)

    dets = detector.predict(args.image, conf=args.conf, iou=args.iou)
    print(f"{len(dets)} detections")
    draw_detections(args.image, dets, class_names, args.out)
    print(f"Wrote {args.out}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as fh:
            json.dump([d.__dict__ for d in dets], fh, indent=2)


if __name__ == "__main__":
    main()
