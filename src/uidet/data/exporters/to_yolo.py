"""Export a :class:`UnifiedDataset` to Ultralytics YOLO format.

Layout produced:

    <out_dir>/
        images/{train,val,test}/<id>.<ext>     (symlinks on POSIX, copies on Windows)
        labels/{train,val,test}/<id>.txt
        data.yaml                              (path, train, val, test, names)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

from ..unified import UnifiedDataset

# On Windows, symlinks need elevation / Developer Mode and stat-walk badly under
# OneDrive; just copy. On POSIX symlinks are free.
_USE_SYMLINKS = sys.platform != "win32"


def _safe_remove(p: Path) -> None:
    """Delete ``p`` if anything is there, even broken/unreadable symlinks
    (Windows raises WinError 1920 from .exists() on dangling reparse points)."""
    try:
        p.unlink()
        return
    except FileNotFoundError:
        return
    except OSError:
        # could be a broken symlink whose stat fails; try lstat-based delete
        try:
            os.remove(p)
        except FileNotFoundError:
            return
        except OSError:
            pass


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _safe_remove(dst)
    if _USE_SYMLINKS:
        try:
            os.symlink(src, dst)
            return
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(src, dst)


def export_yolo(
    dataset: UnifiedDataset,
    splits: dict[str, list[str]],
    taxonomy: str,
    out_dir: Path,
) -> Path:
    """Write YOLO-format labels + a data.yaml. Returns path to data.yaml."""
    classes = dataset.classes_by_taxonomy.get(taxonomy)
    if not classes:
        raise ValueError(
            f"Taxonomy {taxonomy!r} not found in dataset.classes_by_taxonomy. "
            f"Available: {sorted(dataset.classes_by_taxonomy)}"
        )
    name_to_id = {c: i for i, c in enumerate(classes)}

    by_id = {im.image_id: im for im in dataset.images}

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in splits.items():
        for image_id in ids:
            img = by_id.get(image_id)
            if img is None:
                continue

            ext = img.image_path.suffix or ".png"
            img_dst = out_dir / "images" / split_name / f"{image_id}{ext}"
            _link_or_copy(img.image_path, img_dst)

            label_dst = out_dir / "labels" / split_name / f"{image_id}.txt"
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            for b in img.boxes:
                cname = b.label_for(taxonomy)
                if cname is None:
                    continue
                cid = name_to_id[cname]
                # convert to YOLO normalized cx, cy, w, h
                cx = (b.xmin + b.xmax) / 2.0 / img.width
                cy = (b.ymin + b.ymax) / 2.0 / img.height
                bw = b.width() / img.width
                bh = b.height() / img.height
                # Clamp to [0, 1] just in case
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            label_dst.write_text("\n".join(lines))

    data_yaml = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(classes)},
        "nc": len(classes),
    }
    yaml_path = out_dir / "data.yaml"
    with yaml_path.open("w") as fh:
        yaml.safe_dump(data_yaml, fh, sort_keys=False)
    return yaml_path
