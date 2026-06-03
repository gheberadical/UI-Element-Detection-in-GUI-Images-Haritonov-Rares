"""Ultralytics-family detector wrapper.

Handles YOLOv8 / YOLOv9 / YOLO11 / YOLOv10 / YOLOv12 / YOLO26 and RT-DETR via the same
``ultralytics`` package -- they all share the same training API.
"""

from __future__ import annotations

from pathlib import Path

from .base import Detection, Detector, TrainConfig


# Mapping from "friendly" name to the Ultralytics weight name.
_ULTRA_WEIGHT_MAP = {
    # YOLOv8
    "yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt", "yolov8l": "yolov8l.pt", "yolov8x": "yolov8x.pt",
    # YOLOv9
    "yolov9t": "yolov9t.pt", "yolov9s": "yolov9s.pt",
    "yolov9m": "yolov9m.pt", "yolov9c": "yolov9c.pt", "yolov9e": "yolov9e.pt",
    # YOLOv10
    "yolov10n": "yolov10n.pt", "yolov10s": "yolov10s.pt",
    "yolov10m": "yolov10m.pt", "yolov10l": "yolov10l.pt", "yolov10x": "yolov10x.pt",
    # YOLO11
    "yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt", "yolo11l": "yolo11l.pt", "yolo11x": "yolo11x.pt",
    # YOLO12 (no 'v' prefix, same convention as YOLO11)
    "yolov12n": "yolo12n.pt", "yolov12s": "yolo12s.pt",
    "yolov12m": "yolo12m.pt", "yolov12l": "yolo12l.pt", "yolov12x": "yolo12x.pt",
    # YOLO26 (released Sep 2025; edge-optimised, NMS-free, STAL for small objects)
    "yolo26n": "yolo26n.pt", "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt", "yolo26l": "yolo26l.pt", "yolo26x": "yolo26x.pt",
    # RT-DETR
    "rtdetr_l": "rtdetr-l.pt", "rtdetr_x": "rtdetr-x.pt",
}


class UltralyticsDetector(Detector):
    def __init__(self, model_name: str, num_classes: int, class_names: list[str]):
        self.name = model_name
        self.num_classes = num_classes
        self.class_names = class_names
        self._weight = _ULTRA_WEIGHT_MAP.get(model_name)
        if self._weight is None:
            raise ValueError(
                f"Unknown Ultralytics model {model_name!r}. "
                f"Known: {sorted(_ULTRA_WEIGHT_MAP)}"
            )
        self._model = None  # type: ignore[assignment]

    # ---- training -------------------------------------------------------
    def train(self, cfg: TrainConfig) -> Path:
        from ultralytics import RTDETR, YOLO

        ModelCls = RTDETR if self.name.startswith("rtdetr") else YOLO
        model = ModelCls(self._weight)

        run_dir = cfg.out_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        model.train(
            data=str(cfg.data_yaml),
            epochs=cfg.epochs,
            batch=cfg.batch,
            imgsz=cfg.imgsz,
            lr0=cfg.lr0,
            seed=cfg.seed,
            device=cfg.device,
            workers=cfg.workers,
            amp=cfg.amp,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            **(cfg.extra or {}),
        )

        # Ultralytics writes weights under <project>/<name>/weights/best.pt
        weights = run_dir / "weights" / "best.pt"
        if not weights.exists():
            cands = sorted((run_dir / "weights").glob("*.pt"))
            if cands:
                weights = cands[-1]
        self._model = ModelCls(str(weights)) if weights.exists() else model
        return weights

    # ---- inference ------------------------------------------------------
    def load(self, weights: Path) -> None:
        from ultralytics import RTDETR, YOLO
        ModelCls = RTDETR if self.name.startswith("rtdetr") else YOLO
        self._model = ModelCls(str(weights))

    def _result_to_detections(self, r) -> list[Detection]:
        out: list[Detection] = []
        if r.boxes is None:
            return out
        xyxy   = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls    = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, scores, cls):
            cid = int(c)
            out.append(Detection(
                xmin=float(x1), ymin=float(y1), xmax=float(x2), ymax=float(y2),
                score=float(s), class_id=cid,
                class_name=self.class_names[cid] if cid < len(self.class_names) else str(cid),
            ))
        return out

    def predict(self, image_path: Path, conf: float = 0.25, iou: float = 0.45) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("Call .load(weights) or .train(cfg) before predict().")
        results = self._model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)
        return self._result_to_detections(results[0]) if results else []

    def predict_batch(
        self, image_paths: list[Path], conf: float = 0.001, iou: float = 0.6
    ) -> list[list[Detection]]:
        if self._model is None:
            raise RuntimeError("Call .load(weights) or .train(cfg) before predict_batch().")
        if not image_paths:
            return []
        results = self._model.predict(
            source=[str(p) for p in image_paths],
            conf=conf, iou=iou, verbose=False, stream=False,
        )
        return [self._result_to_detections(r) for r in results]

    # ---- evaluation -----------------------------------------------------
    def evaluate(self, cfg: TrainConfig, split: str = "test") -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Call .load(weights) or .train(cfg) before evaluate().")
        metrics = self._model.val(
            data=str(cfg.data_yaml),
            split=split,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            device=cfg.device,
            project=str(cfg.out_dir.parent),
            name=f"{cfg.out_dir.name}_eval_{split}",
            exist_ok=True,
            verbose=False,
        )
        rd = getattr(metrics, "results_dict", {}) or {}
        return {str(k): float(v) for k, v in rd.items() if isinstance(v, (int, float))}
