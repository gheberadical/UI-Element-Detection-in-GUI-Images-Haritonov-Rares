"""Detector abstract base class.

Every concrete model wrapper (YOLO, Faster R-CNN, RT-DETR via HuggingFace, ...)
implements this interface. Train/eval/predict CLIs only ever talk to this ABC,
so adding a new model is one new file in ``src/uidet/models/`` plus one factory
entry in :func:`build_detector`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    name: str                          # experiment id (e.g. "yolov8n_uicvd_unified3")
    out_dir: Path                      # results/<name>
    data_yaml: Path                    # path to YOLO data.yaml (also drives COCO eval)
    coco_val_json: Path | None = None  # optional ground-truth json for pycocotools eval
    coco_test_json: Path | None = None
    epochs: int = 100
    batch: int = 8
    imgsz: int = 640
    lr0: float = 0.01
    seed: int = 42
    device: str = "0"                  # "cpu" or CUDA index(es) "0", "0,1"
    workers: int = 4
    amp: bool = True
    extra: dict[str, Any] = field(default_factory=dict)   # model-specific overrides
    # WandB config -- all optional. If wandb_project is None, WandB is skipped.
    wandb_project: str | None = None   # e.g. "uidet-thesis"
    wandb_entity: str | None = None    # your wandb username/team; None = default
    wandb_tags: list[str] = field(default_factory=list)


@dataclass
class Detection:
    xmin: float; ymin: float; xmax: float; ymax: float
    score: float
    class_id: int
    class_name: str


class Detector(ABC):
    """Common interface for all detectors."""

    name: str = "detector"

    @abstractmethod
    def train(self, cfg: TrainConfig) -> Path:
        """Train and return path to the best weights file."""

    @abstractmethod
    def load(self, weights: Path) -> None:
        """Load trained weights for inference / evaluation."""

    @abstractmethod
    def predict(self, image_path: Path, conf: float = 0.25, iou: float = 0.45) -> list[Detection]:
        """Run inference on a single image."""

    def predict_batch(
        self, image_paths: list[Path], conf: float = 0.001, iou: float = 0.6
    ) -> list[list[Detection]]:
        """Inference over many images. Default = naive loop; subclasses can
        override with a faster batched call. Lower default ``conf`` because
        COCO eval wants the full PR curve, not a confidence-thresholded cut."""
        return [self.predict(p, conf=conf, iou=iou) for p in image_paths]

    @abstractmethod
    def evaluate(self, cfg: TrainConfig, split: str = "test") -> dict[str, float]:
        """Run the model's *built-in* val (cheap sanity numbers).

        For thesis-grade per-class / small-medium-large metrics use
        ``uidet.utils.metrics.coco_evaluate`` via ``uidet.eval_coco``."""


def build_detector(model_name: str, num_classes: int, class_names: list[str]) -> Detector:
    """Factory: pick a detector by name."""
    name = model_name.lower()

    # Ultralytics-family: YOLOv8/v9/v10/v11/v12/v26 + RT-DETR via same wrapper.
    ultralytics_prefixes = ("yolov8", "yolov9", "yolo11", "yolov10", "yolov12", "yolo26", "rtdetr")
    if name.startswith(ultralytics_prefixes):
        from .yolo_ultra import UltralyticsDetector
        return UltralyticsDetector(model_name, num_classes=num_classes, class_names=class_names)

    # Faster R-CNN family (torchvision):
    #   fasterrcnn_r50, fasterrcnn_r50_v2, fasterrcnn_mobilenet
    if name.startswith("fasterrcnn"):
        from .faster_rcnn_tv import FasterRCNNDetector
        return FasterRCNNDetector(model_name, num_classes=num_classes, class_names=class_names)

    # RF-DETR (Roboflow). prefix "rfdetr_" checked after Ultralytics list (no overlap).
    if name.startswith("rfdetr_"):
        from .rfdetr import RFDETRDetector
        return RFDETRDetector(model_name, num_classes=num_classes, class_names=class_names)

    if name.startswith("detr") or name.startswith("rtdetr_hf") or name.startswith("dfine"):
        from .detr_hf import HFDetrDetector
        return HFDetrDetector(model_name, num_classes=num_classes, class_names=class_names)

    if name.startswith("yoloworld"):
        from .yoloworld import YoloWorldDetector
        return YoloWorldDetector(model_name, num_classes=num_classes, class_names=class_names)

    raise ValueError(f"Unknown model_name {model_name!r}")
