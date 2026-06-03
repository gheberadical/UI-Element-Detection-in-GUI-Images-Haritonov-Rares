"""RF-DETR wrapper (Roboflow, ICLR 2026).

Built on a DINOv2 vision transformer backbone. As of mid-2026, the
``rfdetr`` Python package ships these variants:

    rfdetr_n        nano       (lightest)
    rfdetr_s        small
    rfdetr_m        medium    (paper SOTA, similar size to RT-DETR-L)
    rfdetr_b        base
    rfdetr_l        large      (heaviest)

We use medium as the default for our thesis comparison since it most closely
matches RT-DETR-L in parameter count, giving an apples-to-apples comparison.

Install:  pip install rfdetr supervision

Resolution note: RF-DETR's pretrained checkpoints are at specific resolutions
(e.g. 576x576 for Medium). Using a different resolution causes a tensor-shape
mismatch when loading pretrained weights, so we pass ``resolution=`` to the
constructor where possible.
"""

from __future__ import annotations

from pathlib import Path

from .base import Detection, Detector, TrainConfig


_VARIANT_MAP = {
    "rfdetr_n": "RFDETRNano",
    "rfdetr_s": "RFDETRSmall",
    "rfdetr_m": "RFDETRMedium",
    "rfdetr_b": "RFDETRBase",
    "rfdetr_l": "RFDETRLarge",
}


def _resolve_dataset_dir(data_yaml: Path) -> Path:
    """Map our data.yaml location to the RF-DETR-formatted version under
    data/prepared_rfdetr/<dataset>__<taxonomy>/. Run scripts/to_rfdetr_layout.py
    first if the directory doesn't exist yet."""
    tax_dir = Path(data_yaml).parent.resolve()
    taxonomy = tax_dir.name
    dataset = tax_dir.parent.name
    rfdetr_root = tax_dir.parents[2] / "prepared_rfdetr" / f"{dataset}__{taxonomy}"
    if not rfdetr_root.exists():
        raise FileNotFoundError(
            f"RF-DETR layout missing at {rfdetr_root}. Run:\n"
            f"    python scripts/to_rfdetr_layout.py --dataset {dataset} --taxonomy {taxonomy}"
        )
    return rfdetr_root


class RFDETRDetector(Detector):
    def __init__(self, model_name, num_classes, class_names):
        if model_name not in _VARIANT_MAP:
            raise ValueError(
                f"Unknown RF-DETR variant {model_name!r}. Known: {sorted(_VARIANT_MAP)}"
            )
        self.name = model_name
        self.num_classes = num_classes
        self.class_names = class_names
        self._model = None

    def _make_model(self, resolution=None):
        import rfdetr
        cls = getattr(rfdetr, _VARIANT_MAP[self.name])
        if resolution is not None:
            try:
                return cls(resolution=resolution)
            except TypeError:
                # Older rfdetr versions don't accept resolution in __init__.
                pass
        return cls()

    # ---- training -------------------------------------------------------
    def train(self, cfg):
        dataset_dir = _resolve_dataset_dir(cfg.data_yaml)
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        extra = cfg.extra or {}
        grad_accum = int(extra.get("grad_accum_steps", 4))
        resolution = extra.get("resolution")

        # Build the model at the requested resolution so its positional
        # embeddings match the pretrained checkpoint (avoids tensor-shape
        # mismatch when load_state_dict runs).
        model = self._make_model(resolution=resolution)

        kwargs = dict(
            dataset_dir=str(dataset_dir),
            epochs=cfg.epochs,
            batch_size=cfg.batch,
            grad_accum_steps=grad_accum,
            lr=cfg.lr0,
            output_dir=str(out_dir),
            # Forward the YAML's top-level amp flag so cfg.amp actually drives
            # behavior. The rfdetr package defaults amp=True, but a user
            # setting amp: false in the experiment YAML previously had no
            # effect because this kwarg was never forwarded.
            amp=cfg.amp,
        )
        # Optional kwargs the package may accept
        for k in ("resolution", "weight_decay", "early_stopping",
                  "early_stopping_patience", "use_ema"):
            if k in extra:
                kwargs[k] = extra[k]

        print(f"RF-DETR {self.name}: train(**{kwargs})")
        model.train(**kwargs)

        # RF-DETR saves best weights under output_dir; find the "best" pth.
        # Use rglob so we find checkpoints saved under sub-directories
        # (some rfdetr versions write to output_dir/<run>/ or output_dir/weights/).
        cands = sorted(out_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        best = next((p for p in cands if "best" in p.name.lower()), None) or (cands[0] if cands else None)
        if best is None:
            raise RuntimeError(f"No .pth weights found under {out_dir} after training")

        weights_dir = out_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        canonical = weights_dir / "best.pt"
        if not canonical.exists():
            try:
                canonical.symlink_to(best.resolve())
            except OSError:
                import shutil
                shutil.copy2(best, canonical)

        self._model = model
        return canonical

    # ---- inference ------------------------------------------------------
    def load(self, weights):
        import logging
        import warnings
        import rfdetr
        logging.getLogger("rf-detr").setLevel(logging.ERROR)
        logging.getLogger("rfdetr").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        warnings.filterwarnings("ignore", category=UserWarning, module="rfdetr")
        warnings.filterwarnings("ignore", message=".*use_return_dict.*")
        cls = getattr(rfdetr, _VARIANT_MAP[self.name])
        # Always tell rfdetr how many classes to expect so it builds the
        # detection head correctly before loading the checkpoint.  Without
        # this it defaults to 91 (COCO) and then applies a lossy adaptation
        # when it discovers the checkpoint only has self.num_classes classes,
        # producing near-zero mAP for most classes.
        try:
            self._model = cls(pretrain_weights=str(weights),
                              num_classes=self.num_classes)
        except TypeError:
            # Older rfdetr builds: construct first, then load.
            try:
                self._model = cls(num_classes=self.num_classes)
            except TypeError:
                self._model = cls()
            if hasattr(self._model, "load"):
                self._model.load(str(weights))
            else:
                raise RuntimeError(
                    "Could not load RF-DETR weights: package API doesn't accept "
                    "pretrain_weights / num_classes and exposes no .load() method."
                )
        if hasattr(self._model, "optimize_for_inference"):
            try:
                self._model.optimize_for_inference()
            except Exception:
                pass

    def _ensure_loaded(self):
        if self._model is None:
            raise RuntimeError("Call .load(weights) or .train(cfg) before predict().")

    @staticmethod
    def _load_rgb_tensor(path):
        """Load image as an explicit (3, H, W) float32 tensor in [0, 1].
        Handles RGBA / palette PNGs by forcing RGB conversion before tensorising.
        Passing a pre-built tensor bypasses rfdetr's internal image-loading code
        which may preserve alpha channels."""
        import torch
        import numpy as np
        from PIL import Image
        img = np.array(Image.open(str(path)).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

    def predict(self, image_path, conf=0.25, iou=0.45):
        del iou
        self._ensure_loaded()
        dets = self._model.predict(self._load_rgb_tensor(image_path), threshold=conf)
        return self._to_detections(dets)

    def predict_batch(self, image_paths, conf=0.001, iou=0.6):
        del iou
        self._ensure_loaded()
        out = []
        for p in image_paths:
            dets = self._model.predict(self._load_rgb_tensor(p), threshold=conf)
            out.append(self._to_detections(dets))
        return out

    def _to_detections(self, sv_dets):
        """Convert a supervision.Detections to our Detection list."""
        try:
            xyxy = sv_dets.xyxy
            scores = sv_dets.confidence
            classes = sv_dets.class_id
        except AttributeError:
            xyxy = sv_dets.get("xyxy") if hasattr(sv_dets, "get") else None
            scores = sv_dets.get("confidence") if hasattr(sv_dets, "get") else None
            classes = sv_dets.get("class_id") if hasattr(sv_dets, "get") else None
        out = []
        if xyxy is None:
            return out
        import numpy as np
        xyxy = np.asarray(xyxy)
        scores = np.asarray(scores) if scores is not None else np.ones(len(xyxy))
        classes = np.asarray(classes) if classes is not None else np.zeros(len(xyxy), dtype=int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, scores, classes):
            cid = int(c)
            # Fine-tuned RF-DETR models output 0-based class IDs (0 … n-1).
            # Do NOT subtract 1 — that was a wrong assumption carried over from
            # the COCO-pretrained model and caused all class assignments to be
            # off by one, giving near-zero mAP for every class except class 0.
            name = self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)
            out.append(Detection(
                xmin=float(x1), ymin=float(y1), xmax=float(x2), ymax=float(y2),
                score=float(s), class_id=cid, class_name=name,
            ))
        return out

    # ---- evaluation -----------------------------------------------------
    def evaluate(self, cfg, split="test"):
        """Cheap parity stub. The real per-class / s-m-l numbers come from
        ``python -m uidet.eval_coco`` which works through predict_batch."""
        return {}
