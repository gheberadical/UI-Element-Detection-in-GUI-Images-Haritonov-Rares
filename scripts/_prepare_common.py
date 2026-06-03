"""Shared helpers for the prepare_*.py scripts.

Adds the ``src/`` directory to sys.path so the scripts can be run directly
without installing the package (`python scripts/prepare_uicvd.py` works).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
RAW_DIR = REPO_ROOT / "datasets"          # holds the input .zip files
PREPARED_DIR = REPO_ROOT / "data" / "prepared"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def unzip_if_needed(zip_path: Path, target_root: Path, marker_subdir: str) -> Path:
    """Unzip ``zip_path`` into ``target_root`` only if ``target_root/marker_subdir``
    does not yet exist. Returns the path to the extracted root folder."""
    extracted = target_root / marker_subdir
    if extracted.exists():
        return extracted
    target_root.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path.name} -> {target_root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_root)
    if not extracted.exists():
        raise RuntimeError(
            f"After unzipping {zip_path}, expected folder {extracted} not found. "
            f"Top-level entries: {[p.name for p in target_root.iterdir()][:10]}"
        )
    return extracted


def report_dataset(ds, taxonomy: str) -> None:
    """Print a per-class count breakdown for one taxonomy of one dataset.

    Uses the new taxonomy-keyed dicts on UnifiedDataset/UnifiedBox so any
    taxonomy registered in data/ontology/unified_ontology.yaml is reported
    correctly (not just unified3 / unified_ext).
    """
    n_imgs = len(ds.images)
    n_boxes = sum(im.num_annotations(taxonomy) for im in ds.images)
    classes = ds.classes_by_taxonomy.get(taxonomy)
    if not classes:
        print(f"  [{taxonomy}] WARN: no class list registered for this taxonomy")
        return
    per_class = {c: 0 for c in classes}
    for im in ds.images:
        for b in im.boxes:
            v = b.label_for(taxonomy)
            if v is not None:
                per_class[v] = per_class.get(v, 0) + 1
    print(f"  [{taxonomy}] images={n_imgs} boxes={n_boxes}")
    for c, n in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"    {c:18s} {n}")
