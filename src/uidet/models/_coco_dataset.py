"""Minimal COCO-format Dataset for torchvision detection models.

Used by the Faster R-CNN wrapper (and any future torchvision-style detector).
Reads our prepared COCO json + images and yields (image_tensor, target_dict)
in the format torchvision detection models expect:

    target = {
        "boxes":  Tensor[N, 4]   (xmin, ymin, xmax, ymax in *pixels*)
        "labels": Tensor[N]      (int64; **NOT zero-indexed** -- 0 is reserved
                                  for background, real classes start at 1)
        "image_id":  Tensor[1]
        "area":      Tensor[N]
        "iscrowd":   Tensor[N]
    }

Augmentation is intentionally minimal (no horizontal flip -- text reads L-to-R
so flipping corrupts the semantic). Just ToTensor, since FasterRCNN normalizes
internally.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDetectionDataset(Dataset):
    """Loads images + boxes from a COCO JSON written by our exporter."""

    def __init__(self, coco_json_path: Path, image_root: Path):
        self.image_root = Path(image_root)
        self.coco = COCO(str(coco_json_path))
        self.image_ids = sorted(self.coco.getImgIds())
        # Map COCO category id (1-based) to a contiguous 1..K (also 1-based,
        # since torchvision reserves 0 for background).
        cat_ids = sorted(self.coco.getCatIds())
        self.coco_cat_to_label = {c: i + 1 for i, c in enumerate(cat_ids)}
        self.label_to_coco_cat = {v: k for k, v in self.coco_cat_to_label.items()}
        self.cat_id_to_name = {c["id"]: c["name"] for c in self.coco.loadCats(cat_ids)}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        coco_image_id = self.image_ids[idx]
        info = self.coco.loadImgs(coco_image_id)[0]
        img_path = self.image_root / info["file_name"]
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=coco_image_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.coco_cat_to_label[a["category_id"]])
            areas.append(float(a.get("area", w * h)))
            iscrowd.append(int(a.get("iscrowd", 0)))

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            areas_t = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([coco_image_id]),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        # ToTensor: (H, W, 3) uint8 PIL -> (3, H, W) float32 [0,1]
        from torchvision.transforms.functional import to_tensor
        img_t = to_tensor(img)

        return img_t, target


def collate_detection(batch):
    """Variable-size detection batches need a custom collate -- you can't stack
    images of different sizes into a single tensor."""
    images, targets = list(zip(*batch))
    return list(images), list(targets)
