"""Tiny IO helpers used across CLIs."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with Path(path).open() as fh:
        return yaml.safe_load(fh)


def dump_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False)
