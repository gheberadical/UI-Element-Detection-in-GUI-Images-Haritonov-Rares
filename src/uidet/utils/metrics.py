"""COCO-style evaluation backed by pycocotools.

Why this exists in addition to Ultralytics' built-in `model.val()`:
  - We need per-class AP (Ultralytics gives it but as numpy arrays in result objects;
    here we surface them in a stable JSON file you can drop straight into the thesis).
  - We need mAP_small / mAP_medium / mAP_large -- Ultralytics doesn't expose those.
    Critical for GUI work because icons are mostly small and containers are large.
  - We need a single eval path that works for *any* detector (YOLO, Faster R-CNN,
    HF DETR, ...) so cross-model comparison tables are apples-to-apples.

A note on COCO's "small/medium/large" thresholds (area in pixels):
  - small  : area < 32^2  =  1024
  - medium : 32^2 <= area < 96^2  =  9216
  - large  : area >= 96^2

These are calibrated for COCO natural images at ~640x480. For UICVD (1920x940)
and VINS (~720x1280) they end up biased -- many "icons" land in 'small' even
when they're a comfortable click target. We keep COCO's defaults for parity
with prior work; the per-class table is what you should actually argue from.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..models.base import Detection


def detections_to_coco(
    detections: Iterable[Detection],
    image_id: int,
    name_to_cat_id: dict[str, int],
) -> list[dict]:
    """Convert a Detection list (single image) to COCO-results format."""
    out: list[dict] = []
    for d in detections:
        cat_id = name_to_cat_id.get(d.class_name)
        if cat_id is None:
            continue
        w = max(0.0, d.xmax - d.xmin)
        h = max(0.0, d.ymax - d.ymin)
        if w <= 0 or h <= 0:
            continue
        out.append({
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(d.xmin), float(d.ymin), float(w), float(h)],
            "score": float(d.score),
        })
    return out


# Headline COCO metric names (cocoEval.stats indices)
_STAT_NAMES = [
    "mAP",            # AP @ IoU=0.50:0.95 (primary metric)
    "mAP50",          # AP @ IoU=0.50
    "mAP75",          # AP @ IoU=0.75
    "mAP_small",      # AP for small objects (area < 32^2)
    "mAP_medium",     # AP for medium objects (32^2 <= area < 96^2)
    "mAP_large",      # AP for large objects (area >= 96^2)
    "AR_1",           # AR @ max=1
    "AR_10",          # AR @ max=10
    "AR_100",         # AR @ max=100
    "AR_small",
    "AR_medium",
    "AR_large",
]


def _per_class_ap(coco_eval: COCOeval, class_names: list[str]) -> dict[str, dict[str, float]]:
    """Pull per-class AP @ IoU=0.5:0.95 and AP @ IoU=0.5 out of COCOeval's
    precision tensor. Shape: [T=10, R=101, K=num_classes, A=4 areas, M=3 maxdets]."""
    precisions: np.ndarray = coco_eval.eval["precision"]
    out: dict[str, dict[str, float]] = {}
    for k, name in enumerate(class_names):
        # area range index 0 = "all"; max_dets index -1 = 100
        p_all = precisions[:, :, k, 0, -1]   # [T, R]
        p_50  = precisions[0, :, k, 0, -1]   # [R] at IoU=0.50
        valid_all = p_all[p_all > -1]
        valid_50  = p_50[p_50  > -1]
        out[name] = {
            "AP":   float(valid_all.mean()) if valid_all.size else float("nan"),
            "AP50": float(valid_50.mean())  if valid_50.size  else float("nan"),
        }
    return out


def coco_evaluate(
    gt_coco_path: Path,
    predictions: list[dict],
    save_dir: Path,
    *,
    iou_type: str = "bbox",
    quiet: bool = True,
) -> dict:
    """Run pycocotools COCOeval. Returns a metrics dict and saves predictions
    + a metrics.json into ``save_dir``."""
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_path = save_dir / "predictions.json"
    pred_path.write_text(json.dumps(predictions))

    coco_gt = COCO(str(gt_coco_path))
    if not predictions:
        # pycocotools blows up on empty preds; emit a zero-metrics record.
        metrics = {n: 0.0 for n in _STAT_NAMES}
        metrics["per_class"] = {}
        (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf if quiet else io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    class_names_in_order = [cat_id_to_name[i] for i in coco_gt.getCatIds()]

    metrics = {name: float(coco_eval.stats[i]) for i, name in enumerate(_STAT_NAMES)}
    metrics["per_class"] = _per_class_ap(coco_eval, class_names_in_order)
    metrics["num_predictions"] = len(predictions)
    metrics["num_gt_images"] = len(coco_gt.getImgIds())
    metrics["num_gt_annotations"] = len(coco_gt.getAnnIds())

    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if not quiet:
        print(buf.getvalue())
    return metrics


def format_metrics_table(metrics: dict) -> str:
    """Pretty one-screen summary suitable for printing to console."""
    lines = []
    lines.append(f"  Predictions: {metrics.get('num_predictions', '?')}  "
                 f"GT images: {metrics.get('num_gt_images', '?')}  "
                 f"GT boxes: {metrics.get('num_gt_annotations', '?')}")
    lines.append("  Overall:")
    for k in ("mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large"):
        v = metrics.get(k)
        if v is not None:
            lines.append(f"    {k:11s} {v:.4f}")
    pc = metrics.get("per_class") or {}
    if pc:
        lines.append("  Per-class AP @ [.50:.95] / AP @ .50:")
        width = max(len(c) for c in pc) + 2
        for name, m in sorted(pc.items()):
            lines.append(f"    {name:<{width}} {m['AP']:.4f}  /  {m['AP50']:.4f}")
    return "\n".join(lines)
