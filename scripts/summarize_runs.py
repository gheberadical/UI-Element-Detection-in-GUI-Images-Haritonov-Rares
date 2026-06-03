"""Summarize every training run into one CSV + a printable table.

Walks ``results/`` and emits:
    results/_summary.csv     machine-readable, one row per training run
    results/_summary.txt     human-readable table (also printed to console)

Handles both detector backends transparently:
  - YOLO / RT-DETR (Ultralytics) -> reads results.csv
  - Faster R-CNN  (our wrapper)  -> reads log.jsonl

Also handles runs trained on Kaggle (no local summary.yaml) by falling back
to the matching configs/experiments/<name>.yaml.

Run from anywhere:
    python scripts/summarize_runs.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def hms(seconds):
    """Pretty-print a duration as HhMMmSSs."""
    if seconds is None:
        return "-"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s"


def summarize_ultralytics(run_dir):
    """Pull training stats from Ultralytics' results.csv."""
    out = dict(epochs_trained=None, train_seconds=None, best_epoch=None,
               val_peak_mAP50=None, val_peak_mAP=None)
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return out
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return out
    out["epochs_trained"] = len(rows)
    try:
        out["train_seconds"] = float(rows[-1].get("time", "") or 0)
    except ValueError:
        pass

    cols = rows[0].keys()
    map50_col = next((c for c in cols if "mAP50(B)" in c and "mAP50-95" not in c), None)
    map_col = next((c for c in cols if "mAP50-95(B)" in c), None)

    def _safe(r, c):
        try:
            return float(r[c])
        except (KeyError, ValueError, TypeError):
            return None

    if map_col:
        scored = [(int(r["epoch"]), _safe(r, map_col)) for r in rows]
        scored = [(e, v) for e, v in scored if v is not None]
        if scored:
            best_e, best_v = max(scored, key=lambda p: p[1])
            out["best_epoch"] = best_e
            out["val_peak_mAP"] = best_v
    if map50_col:
        vals = [_safe(r, map50_col) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            out["val_peak_mAP50"] = max(vals)
    return out


def summarize_frcnn(run_dir):
    """Pull training stats from our F-RCNN log.jsonl."""
    out = dict(epochs_trained=None, train_seconds=None, best_epoch=None,
               val_peak_recall50=None)
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        return out
    rows = []
    with log_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return out
    out["epochs_trained"] = len(rows)
    out["train_seconds"] = sum(float(r.get("seconds", 0) or 0) for r in rows)
    best = max(rows, key=lambda r: r.get("val_recall50", -1))
    out["best_epoch"] = best.get("epoch")
    out["val_peak_recall50"] = best.get("val_recall50")
    return out


def _load_config_fallback(run_name):
    """For Kaggle-trained runs, fall back to configs/experiments/<name>.yaml
    to get the model / dataset / taxonomy metadata."""
    cfg_path = REPO / "configs" / "experiments" / f"{run_name}.yaml"
    if not cfg_path.exists():
        return None
    return yaml.safe_load(cfg_path.read_text()) or {}


def collect_run(run_dir):
    """Return one row dict for a training run, or None for aux/eval-only dirs."""
    summary_path = run_dir / "summary.yaml"
    coco_path = run_dir / "coco_eval_test" / "metrics.json"

    cfg = {}
    if summary_path.exists():
        summary = yaml.safe_load(summary_path.read_text()) or {}
        cfg = summary.get("config", {}) or {}
    elif coco_path.exists():
        cfg = _load_config_fallback(run_dir.name) or {}
        if not cfg:
            return None
    else:
        return None

    row = {
        "name": run_dir.name,
        "model": cfg.get("model", ""),
        "dataset": cfg.get("dataset", ""),
        "taxonomy": cfg.get("taxonomy", ""),
        "epochs_trained": None,
        "train_seconds": None,
        "train_hms": "-",
        "best_epoch": None,
        "val_peak": None,
        "val_peak_metric": "",
        "test_mAP": None,
        "test_mAP50": None,
        "test_mAP75": None,
        "test_mAP_small": None,
        "test_mAP_medium": None,
        "test_mAP_large": None,
        "inference_fps": None,
    }

    if row["model"].startswith("fasterrcnn"):
        t = summarize_frcnn(run_dir)
        row.update(
            epochs_trained=t["epochs_trained"],
            train_seconds=t["train_seconds"],
            best_epoch=t["best_epoch"],
            val_peak=t["val_peak_recall50"],
            val_peak_metric="recall@IoU0.5",
        )
    else:
        t = summarize_ultralytics(run_dir)
        row.update(
            epochs_trained=t["epochs_trained"],
            train_seconds=t["train_seconds"],
            best_epoch=t["best_epoch"],
            val_peak=t["val_peak_mAP"],
            val_peak_metric="mAP@[.50:.95]",
        )

    if row["epochs_trained"] is None and coco_path.exists():
        row["train_hms"] = "(Kaggle)"
    else:
        row["train_hms"] = hms(row["train_seconds"])

    if coco_path.exists():
        m = json.loads(coco_path.read_text())
        for k in ("mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large"):
            row[f"test_{k}"] = m.get(k)
        row["inference_fps"] = m.get("inference_fps")

    return row


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_table(rows):
    columns = [
        ("name",        38),
        ("model",       18),
        ("dataset",     8),
        ("taxonomy",    11),
        ("epochs",      7),
        ("train",       12),
        ("val_peak",    9),
        ("test_mAP",    9),
        ("test_mAP50",  10),
        ("test_small",  10),
        ("test_med",    9),
        ("test_large",  10),
        ("fps",         6),
    ]
    header = "  ".join(f"{name:<{w}}" for name, w in columns)
    sep = "-" * len(header)
    out = [header, sep]
    for r in rows:
        v = {
            "name":       r["name"],
            "model":      r["model"],
            "dataset":    r["dataset"],
            "taxonomy":   r["taxonomy"],
            "epochs":     _fmt(r["epochs_trained"]),
            "train":      r["train_hms"],
            "val_peak":   _fmt(r["val_peak"]),
            "test_mAP":   _fmt(r["test_mAP"]),
            "test_mAP50": _fmt(r["test_mAP50"]),
            "test_small": _fmt(r["test_mAP_small"]),
            "test_med":   _fmt(r["test_mAP_medium"]),
            "test_large": _fmt(r["test_mAP_large"]),
            "fps":        _fmt(r["inference_fps"]),
        }
        out.append("  ".join(f"{str(v[name]):<{w}}" for name, w in columns))
    return "\n".join(out)


def main():
    if not RESULTS.exists():
        print(f"No results/ directory at {RESULTS}")
        return
    rows = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        r = collect_run(d)
        if r is not None:
            rows.append(r)
    if not rows:
        print("No training runs found.")
        return

    rows.sort(key=lambda r: (r["dataset"], r["taxonomy"], r["model"]))

    csv_path = RESULTS / "_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    table = render_table(rows)
    (RESULTS / "_summary.txt").write_text(table)

    print(table)
    print()
    print(f"Wrote {csv_path}")
    print(f"Wrote {RESULTS / '_summary.txt'}")
    print(f"Summarized {len(rows)} training runs.")


if __name__ == "__main__":
    main()
