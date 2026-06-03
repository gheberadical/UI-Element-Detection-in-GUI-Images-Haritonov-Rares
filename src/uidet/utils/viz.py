"""Bounding-box visualization with consistent per-class colors."""

from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np


def class_colors(class_names: list[str]) -> dict[str, tuple[int, int, int]]:
    n = max(1, len(class_names))
    out = {}
    for i, name in enumerate(class_names):
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        out[name] = (int(b * 255), int(g * 255), int(r * 255))   # BGR for cv2
    return out


def draw_detections(image_path: Path, detections, class_names: list[str], out_path: Path) -> Path:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    palette = class_colors(class_names)

    for d in detections:
        color = palette.get(d.class_name, (0, 255, 0))
        x1, y1, x2, y2 = map(int, [d.xmin, d.ymin, d.xmax, d.ymax])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path
