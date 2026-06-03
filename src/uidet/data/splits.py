"""Deterministic, seeded train/val/test splits.

The split for each dataset is computed once and persisted as a JSON file under
``data/prepared/<dataset>/splits/<split_name>.json`` so it cannot drift between
experiments. Re-running ``scripts/prepare_*.py`` regenerates the file
deterministically with the same seed; if the file already exists we just load
it (to make later code changes safe).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .unified import UnifiedImage


def stratify_key(img: UnifiedImage) -> str:
    """Group key for stratified-ish splitting. Uses the dominant native class.

    We don't do strict per-class stratification because GUI screenshots are
    multi-label by nature; the dominant-class heuristic still helps avoid the
    pathological case where rare classes only appear in test.
    """
    if not img.boxes:
        return "__empty__"
    counts: dict[str, int] = {}
    for b in img.boxes:
        counts[b.native_class] = counts.get(b.native_class, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def make_splits(
    images: list[UnifiedImage],
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict[str, list[str]]:
    """Return {'train': [image_id, ...], 'val': [...], 'test': [...]}.

    Stratified-ish on dominant native class. Test fraction = 1 - train - val.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be in (0, 1)")

    # Group by stratify key
    groups: dict[str, list[str]] = {}
    for img in images:
        groups.setdefault(stratify_key(img), []).append(img.image_id)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for key in sorted(groups.keys()):           # sorted -> deterministic
        ids = groups[key][:]
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(round(n * train_frac))) if n >= 3 else max(1, n - 1)
        n_val = int(round(n * val_frac)) if n >= 5 else 0
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train.extend(ids[:n_train])
        val.extend(ids[n_train:n_train + n_val])
        test.extend(ids[n_train + n_val:])

    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def load_or_create_splits(
    images: list[UnifiedImage],
    out_path: Path,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict[str, list[str]]:
    if out_path.exists():
        with out_path.open() as fh:
            return json.load(fh)
    splits = make_splits(images, seed=seed, train_frac=train_frac, val_frac=val_frac)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(splits, fh, indent=2)
    return splits
