"""Faster R-CNN wrapper backed by torchvision.

Supports three flavors via ``model_name``:
    fasterrcnn_r50           -> fasterrcnn_resnet50_fpn       (v1, classic)
    fasterrcnn_r50_v2        -> fasterrcnn_resnet50_fpn_v2    (v2, modern; default choice)
    fasterrcnn_mobilenet     -> fasterrcnn_mobilenet_v3_large_fpn (lighter; for 4 GB)

Memory notes (GTX 1650 Ti, 4 GB):
  - At default torchvision min_size/max_size of 800/1333 the v2 backbone OOMs.
    Configs default to min_size=480, max_size=800 which fits batch=2 with AMP.
  - mobilenet variant fits batch=4-8 even at default sizes; trades accuracy.

AMP note: AMP is force-disabled at runtime on Turing consumer GPUs (1650/1660/
Quadro T*) because RPN regression produces NaN losses under FP16 there.
Configs for those cards should set ``amp: false`` for clarity.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

from ._coco_dataset import CocoDetectionDataset, collate_detection
from .base import Detection, Detector, TrainConfig


_MODEL_BUILDERS = {
    "fasterrcnn_r50":        (fasterrcnn_resnet50_fpn,           FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
    "fasterrcnn_r50_v2":     (fasterrcnn_resnet50_fpn_v2,        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
    "fasterrcnn_mobilenet":  (fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
}


def _build_model(model_name: str, num_classes: int, min_size: int, max_size: int):
    builder, weights = _MODEL_BUILDERS[model_name]
    model = builder(weights=weights, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def _resolve_split_paths(data_yaml: Path) -> tuple[Path, Path, Path | None, Path | None, Path | None]:
    """Given a YOLO data.yaml, find the matching COCO json files we wrote.
    Returns (image_root, coco_dir, train_json, val_json, test_json)."""
    base = Path(data_yaml).parent.resolve()
    ann = base / "annotations"
    cands = {p.stem.split("_")[-1]: p for p in ann.glob("*.json")} if ann.exists() else {}
    return base, ann, cands.get("train"), cands.get("val"), cands.get("test")


class FasterRCNNDetector(Detector):
    def __init__(self, model_name: str, num_classes: int, class_names: list[str]):
        if model_name not in _MODEL_BUILDERS:
            raise ValueError(
                f"Unknown Faster R-CNN model {model_name!r}. "
                f"Known: {sorted(_MODEL_BUILDERS)}"
            )
        self.name = model_name
        self.num_classes = num_classes
        self.class_names = class_names
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        # label-to-name uses 1-based labels (0 = background)
        self._label_to_name = {i + 1: n for i, n in enumerate(class_names)}

    # ---- training -------------------------------------------------------
    def train(self, cfg: TrainConfig) -> Path:
        device_str = "cuda:" + cfg.device if cfg.device.isdigit() else cfg.device
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA unavailable; falling back to CPU (training will be slow).")
            device_str = "cpu"
        device = torch.device(device_str)

        extra = cfg.extra or {}
        min_size = int(extra.get("min_size", 480))
        max_size = int(extra.get("max_size", 800))
        accumulate = int(extra.get("accumulate", 1))
        weight_decay = float(extra.get("weight_decay", 5e-4))
        warmup_iters = int(extra.get("warmup_iters", 500))
        patience = int(extra.get("patience", 8))
        min_epochs = int(extra.get("min_epochs", 4))
        coco_eval_every = int(extra.get("coco_eval_every", 1))  # run COCO mAP every N epochs

        # AMP safety check: known-bad consumer Turing cards produce NaN losses
        # in F-RCNN's RPN regression under FP16.
        amp_requested = bool(cfg.amp) and device.type == "cuda"
        amp_enabled = amp_requested
        if amp_requested and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(device).lower()
            if any(bad in gpu_name for bad in ("1650", "1660", "quadro t1000", "quadro t2000")):
                print(f"  WARN: AMP requested on '{gpu_name}' which produces NaN losses "
                      f"in Faster R-CNN. Disabling AMP for stability.")
                amp_enabled = False

        image_root, _, train_json, val_json, _ = _resolve_split_paths(cfg.data_yaml)
        if train_json is None or val_json is None:
            raise FileNotFoundError(
                f"COCO json files not found under {image_root}. Run scripts/prepare_*.py first."
            )

        train_ds = CocoDetectionDataset(train_json, image_root)
        val_ds = CocoDetectionDataset(val_json, image_root)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch, shuffle=True,
            num_workers=cfg.workers, collate_fn=collate_detection,
            pin_memory=(device.type == "cuda"), persistent_workers=cfg.workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch, shuffle=False,
            num_workers=cfg.workers, collate_fn=collate_detection,
            pin_memory=(device.type == "cuda"), persistent_workers=cfg.workers > 0,
        )

        torch.manual_seed(cfg.seed)
        model = _build_model(self.name, num_classes=self.num_classes,
                             min_size=min_size, max_size=max_size).to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=cfg.lr0, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
        scaler = GradScaler("cuda", enabled=amp_enabled)

        out_dir = Path(cfg.out_dir)
        weights_dir = out_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "log.jsonl"
        log_fh = log_path.open("w")

        # results.csv — same column style as Ultralytics so _wandb_log_epochs_from_csv works.
        import csv as _csv
        csv_path = out_dir / "results.csv"
        csv_fh   = csv_path.open("w", newline="")
        csv_writer = _csv.DictWriter(csv_fh, fieldnames=[
            "epoch", "time",
            "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss",
            "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)", "lr/pg0",
        ])
        csv_writer.writeheader()

        # Whether we can run full pycocotools val mAP per epoch.
        _coco_val_json = Path(cfg.coco_val_json) if cfg.coco_val_json else None
        _has_coco_val  = (_coco_val_json is not None and _coco_val_json.exists())
        if not _has_coco_val:
            print("  NOTE: coco_val_json not set — metrics/mAP50(B) will be 0 in WandB.")

        best_map = -1.0
        best_path = weights_dir / "best.pt"
        best_epoch = 0
        epochs_without_improvement = 0

        # Attach to any WandB run already open (initialised by train.py before this call).
        try:
            import wandb as _wandb
            _has_wandb = _wandb.run is not None
        except ImportError:
            _has_wandb = False

        _train_wall_start = time.perf_counter()
        global_step = 0
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            running = {"total": 0.0, "box": 0.0, "cls": 0.0, "obj": 0.0, "n": 0}
            optimizer_stepped_this_epoch = False
            optim.zero_grad(set_to_none=True)
            for i, (images, targets) in enumerate(train_loader):
                images = [im.to(device, non_blocking=True) for im in images]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

                # Linear warmup over the first few hundred iters
                if global_step < warmup_iters:
                    lr_scale = (global_step + 1) / max(1, warmup_iters)
                    for pg in optim.param_groups:
                        pg["lr"] = cfg.lr0 * lr_scale

                with autocast("cuda", enabled=amp_enabled):
                    losses_dict = model(images, targets)
                    loss = sum(losses_dict.values())
                    loss = loss / accumulate
                if not torch.isfinite(loss):
                    print(f"  WARN: non-finite loss at epoch {epoch} step {i}; skipping batch.")
                    optim.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                if (i + 1) % accumulate == 0:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    optimizer_stepped_this_epoch = True
                # Track individual loss components (map to YOLO-style names).
                # box  = ROI box regression   (loss_box_reg)
                # cls  = ROI classification   (loss_classifier)
                # obj  = RPN objectness + RPN box reg  (closest to YOLO obj/dfl)
                _ld = {k: v.item() for k, v in losses_dict.items()}
                running["box"]   += _ld.get("loss_box_reg", 0.0)
                running["cls"]   += _ld.get("loss_classifier", 0.0)
                running["obj"]   += _ld.get("loss_objectness", 0.0) + _ld.get("loss_rpn_box_reg", 0.0)
                running["total"] += float(loss.item() * accumulate)
                running["n"] += 1
                global_step += 1

            if optimizer_stepped_this_epoch:
                sched.step()
            n = max(1, running["n"])
            mean_loss     = running["total"] / n
            mean_box_loss = running["box"]   / n
            mean_cls_loss = running["cls"]   / n
            mean_dfl_loss = running["obj"]   / n

            # Val losses (model in train mode, no_grad — torchvision returns loss dict).
            val_box_loss, val_cls_loss, val_dfl_loss = self._val_losses(
                model, val_loader, device, amp_enabled)

            # Precision + recall @IoU=0.5 (cheap, no pycocotools).
            val_precision, val_recall = self._cheap_val_metrics(
                model, val_loader, device, score_thresh=0.05)

            # Full COCO mAP on val set — only every coco_eval_every epochs to save time.
            if _has_coco_val and (epoch % coco_eval_every == 0 or epoch == 1):
                val_map5095, val_map50 = self._epoch_coco_mAP(
                    model, _coco_val_json, image_root, device)
            else:
                val_map5095, val_map50 = 0.0, 0.0

            took = time.perf_counter() - t0
            current_lr = optim.param_groups[0]["lr"]
            log_fh.write(json.dumps({
                "epoch": epoch, "loss": mean_loss,
                "train_box_loss": mean_box_loss, "train_cls_loss": mean_cls_loss,
                "train_dfl_loss": mean_dfl_loss,
                "val_box_loss": val_box_loss, "val_cls_loss": val_cls_loss,
                "val_dfl_loss": val_dfl_loss,
                "val_precision": val_precision, "val_recall50": val_recall,
                "val_mAP50": val_map50, "val_mAP5095": val_map5095,
                "lr": current_lr, "seconds": took,
            }) + "\n"); log_fh.flush()
            csv_writer.writerow({
                "epoch":                 epoch,
                "time":                  round(time.perf_counter() - _train_wall_start, 3),
                "train/box_loss":        round(mean_box_loss, 6),
                "train/cls_loss":        round(mean_cls_loss, 6),
                "train/dfl_loss":        round(mean_dfl_loss, 6),
                "val/box_loss":          round(val_box_loss, 6),
                "val/cls_loss":          round(val_cls_loss, 6),
                "val/dfl_loss":          round(val_dfl_loss, 6),
                "metrics/precision(B)":  round(val_precision, 6),
                "metrics/recall(B)":     round(val_recall, 6),
                "metrics/mAP50(B)":      round(val_map50, 6),
                "metrics/mAP50-95(B)":   round(val_map5095, 6),
                "lr/pg0":                round(current_lr, 8),
            }); csv_fh.flush()
            print(f"  epoch {epoch:3d}/{cfg.epochs}  "
                  f"loss={mean_loss:.4f}  mAP50={val_map50:.4f}  "
                  f"P={val_precision:.3f}  R={val_recall:.3f}  "
                  f"lr={current_lr:.2e}  {took:.1f}s")

            # Per-epoch WandB logging — metric names match Ultralytics/YOLO convention.
            if _has_wandb:
                try:
                    gpu_mem_mb = (
                        torch.cuda.max_memory_allocated(device) / 1e6
                        if device.type == "cuda" else 0.0
                    )
                    _wandb.log({
                        "epoch":                  epoch,
                        "train/box_loss":         mean_box_loss,
                        "train/cls_loss":         mean_cls_loss,
                        "train/dfl_loss":         mean_dfl_loss,
                        "val/box_loss":           val_box_loss,
                        "val/cls_loss":           val_cls_loss,
                        "val/dfl_loss":           val_dfl_loss,
                        "metrics/precision(B)":   val_precision,
                        "metrics/recall(B)":      val_recall,
                        "metrics/mAP50(B)":       val_map50,
                        "metrics/mAP50-95(B)":    val_map5095,
                        "lr/pg0":                 current_lr,
                        "perf/epoch_time_s":      took,
                        "perf/gpu_mem_peak_mb":   gpu_mem_mb,
                    }, step=epoch)
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception as _exc:
                    print(f"  WandB log failed (non-fatal): {_exc}")

            # Save checkpoints.
            torch.save({"model": model.state_dict(), "model_name": self.name,
                        "num_classes": self.num_classes, "class_names": self.class_names,
                        "min_size": min_size, "max_size": max_size},
                       weights_dir / "last.pt")
            # Use mAP50 for early stopping when available, fall back to recall.
            es_metric = val_map50 if _has_coco_val else val_recall
            if es_metric > best_map:
                best_map = es_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save({"model": model.state_dict(), "model_name": self.name,
                            "num_classes": self.num_classes, "class_names": self.class_names,
                            "min_size": min_size, "max_size": max_size},
                           best_path)
            else:
                epochs_without_improvement += 1

            if epoch >= min_epochs and epochs_without_improvement >= patience:
                metric_name = "mAP50" if _has_coco_val else "val_recall"
                print(f"  Early stop at epoch {epoch}: no {metric_name} improvement "
                      f"in {patience} epochs (best={best_map:.4f} at epoch {best_epoch}).")
                break

        log_fh.close()
        csv_fh.close()
        total_train_h = (time.perf_counter() - _train_wall_start) / 3600
        metric_label = "mAP50" if _has_coco_val else "val_recall"
        print(f"Training done. Best {metric_label}={best_map:.4f} at epoch {best_epoch}. "
              f"Total time: {total_train_h:.2f}h")

        # End-of-training WandB summary.
        if _has_wandb:
            try:
                param_count = sum(p.numel() for p in model.parameters())
                _wandb.log({
                    "train/best_recall":   best_map,
                    "train/best_epoch":    best_epoch,
                    "train/total_time_h":  total_train_h,
                    "model/params_M":      param_count / 1e6,
                })
            except Exception as _exc:
                print(f"  WandB end-of-training log failed (non-fatal): {_exc}")

        self._model = model
        self._device = device
        return best_path

    @torch.no_grad()
    def _val_losses(
        self, model, val_loader, device, amp_enabled: bool = False
    ) -> tuple[float, float, float]:
        """Compute val-set losses per component (box, cls, obj).

        Torchvision returns a loss dict when the model is in train mode and
        targets are supplied — we use no_grad so no memory is retained.
        Returns (val_box_loss, val_cls_loss, val_dfl_loss).
        """
        model.train()
        totals = {"box": 0.0, "cls": 0.0, "obj": 0.0, "n": 0}
        for images, targets in val_loader:
            images  = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                       for t in targets]
            with autocast("cuda", enabled=amp_enabled):
                ld = model(images, targets)
            totals["box"] += ld.get("loss_box_reg",    torch.tensor(0.0)).item()
            totals["cls"] += ld.get("loss_classifier", torch.tensor(0.0)).item()
            totals["obj"] += (ld.get("loss_objectness",   torch.tensor(0.0)).item() +
                              ld.get("loss_rpn_box_reg",  torch.tensor(0.0)).item())
            totals["n"]   += 1
        n = max(1, totals["n"])
        return totals["box"] / n, totals["cls"] / n, totals["obj"] / n

    @torch.no_grad()
    def _cheap_val_metrics(
        self, model, val_loader, device, score_thresh: float = 0.05
    ) -> tuple[float, float]:
        """Per-epoch precision and recall @IoU=0.5. NOT thesis metrics.

        Returns (precision, recall) — cheap proxy used for early stopping
        and as the metrics/recall(B) / metrics/precision(B) WandB curves.
        """
        model.eval()
        total_gt = 0
        total_pred = 0
        total_tp = 0
        for images, targets in val_loader:
            images = [im.to(device, non_blocking=True) for im in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                pred_boxes  = out["boxes"].detach().cpu()
                pred_scores = out["scores"].detach().cpu()
                gt_boxes    = tgt["boxes"]
                keep = pred_scores >= score_thresh
                total_pred += int(keep.sum().item())
                if gt_boxes.numel() == 0:
                    continue
                total_gt += gt_boxes.shape[0]
                if keep.sum() == 0:
                    continue
                kept = pred_boxes[keep]
                ious = _box_iou(gt_boxes, kept)
                if ious.numel() == 0:
                    continue
                total_tp += int((ious.max(dim=1).values >= 0.5).sum().item())
        recall    = total_tp / max(1, total_gt)
        precision = total_tp / max(1, total_pred)
        return precision, recall

    @torch.no_grad()
    def _epoch_coco_mAP(
        self,
        model,
        val_json: Path,
        image_root: Path,
        device,
        score_thresh: float = 0.001,
    ) -> tuple[float, float]:
        """Run pycocotools bbox mAP on the val set.

        Returns (mAP@.5:.95, mAP@.5).  Suppresses COCO init chatter.
        Falls back to (0.0, 0.0) if pycocotools is unavailable or eval fails.
        """
        try:
            import contextlib as _ctx
            import io as _io

            from PIL import Image as _PIL_Image
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            gt_data = json.loads(val_json.read_text())
            coco_gt = COCO()
            with _ctx.redirect_stdout(_io.StringIO()):
                coco_gt.dataset = gt_data
                coco_gt.createIndex()

            model.eval()
            preds = []
            images_info = gt_data["images"]
            BATCH = 4
            for i in range(0, len(images_info), BATCH):
                chunk = images_info[i:i + BATCH]
                tensors = [
                    to_tensor(_PIL_Image.open(image_root / inf["file_name"]).convert("RGB")).to(device)
                    for inf in chunk
                ]
                outputs = model(tensors)
                for inf, out in zip(chunk, outputs):
                    boxes  = out["boxes"].cpu().numpy()
                    scores = out["scores"].cpu().numpy()
                    labels = out["labels"].cpu().numpy()
                    for (x1, y1, x2, y2), s, lb in zip(boxes, scores, labels):
                        if s >= score_thresh:
                            preds.append({
                                "image_id":    inf["id"],
                                "category_id": int(lb),
                                "bbox":        [float(x1), float(y1),
                                                float(x2 - x1), float(y2 - y1)],
                                "score":       float(s),
                            })

            if not preds:
                return 0.0, 0.0

            with _ctx.redirect_stdout(_io.StringIO()):
                coco_dt = coco_gt.loadRes(preds)
                ev = COCOeval(coco_gt, coco_dt, "bbox")
                ev.evaluate()
                ev.accumulate()
                ev.summarize()

            # stats[0] = mAP@.5:.95,  stats[1] = mAP@.5
            return float(ev.stats[0]), float(ev.stats[1])

        except Exception as exc:
            print(f"  [val mAP] failed (non-fatal): {exc}")
            return 0.0, 0.0

    # ---- inference ------------------------------------------------------
    def load(self, weights: Path) -> None:
        ckpt = torch.load(str(weights), map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            min_size = ckpt.get("min_size", 480)
            max_size = ckpt.get("max_size", 800)
            model = _build_model(self.name, num_classes=self.num_classes,
                                 min_size=min_size, max_size=max_size)
            model.load_state_dict(ckpt["model"])
        else:
            model = _build_model(self.name, num_classes=self.num_classes, min_size=480, max_size=800)
            model.load_state_dict(ckpt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        self._model = model
        self._device = device

    def _ensure_loaded(self):
        if self._model is None or self._device is None:
            raise RuntimeError("Call .load(weights) or .train(cfg) before predict().")

    @torch.no_grad()
    def predict(self, image_path: Path, conf: float = 0.25, iou: float = 0.45) -> list[Detection]:
        del iou  # accepted for ABC parity; F-RCNN NMS is set at model build time
        self._ensure_loaded()
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        tensor = to_tensor(img).to(self._device)
        outputs = self._model([tensor])[0]
        return self._outputs_to_detections(outputs, conf)

    @torch.no_grad()
    def predict_batch(
        self, image_paths: list[Path], conf: float = 0.001, iou: float = 0.6
    ) -> list[list[Detection]]:
        del iou
        self._ensure_loaded()
        from PIL import Image
        results: list[list[Detection]] = []
        BATCH = 4
        for i in range(0, len(image_paths), BATCH):
            chunk = image_paths[i:i + BATCH]
            tensors = [to_tensor(Image.open(p).convert("RGB")).to(self._device) for p in chunk]
            outputs = self._model(tensors)
            for out in outputs:
                results.append(self._outputs_to_detections(out, conf))
        return results

    def _outputs_to_detections(self, out, conf: float) -> list[Detection]:
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        dets: list[Detection] = []
        for (x1, y1, x2, y2), s, lb in zip(boxes, scores, labels):
            if s < conf:
                continue
            name = self._label_to_name.get(int(lb), str(int(lb)))
            dets.append(Detection(
                xmin=float(x1), ymin=float(y1), xmax=float(x2), ymax=float(y2),
                score=float(s), class_id=int(lb) - 1,
                class_name=name,
            ))
        return dets

    # ---- evaluation -----------------------------------------------------
    def evaluate(self, cfg: TrainConfig, split: str = "test") -> dict[str, float]:
        """Cheap built-in val score for parity with the YOLO wrapper. For real
        thesis numbers run ``python -m uidet.eval_coco``."""
        self._ensure_loaded()
        image_root, _, _, val_json, test_json = _resolve_split_paths(cfg.data_yaml)
        json_path = test_json if split == "test" else val_json
        if json_path is None:
            return {}
        ds = CocoDetectionDataset(json_path, image_root)
        loader = DataLoader(ds, batch_size=cfg.batch, shuffle=False,
                            num_workers=cfg.workers, collate_fn=collate_detection)
        _, recall50 = self._cheap_val_metrics(self._model, loader, self._device)
        return {"recall@IoU0.5": recall50}


# ---------------------------------------------------------------------------
# Tiny IoU helper (no torchvision dependency)
# ---------------------------------------------------------------------------
def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: [N, 4], b: [M, 4] (xyxy). Returns [N, M] IoU matrix."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)
