"""Training entry point.

    python -m uidet.train --config configs/experiments/yolov8n_uicvd_unified3.yaml

The experiment YAML must contain (minimum):

    name: yolov8n_uicvd_unified3
    model: yolov8n
    dataset: uicvd                     # one of: uicvd, gengui, vins, vins_<subset>
    taxonomy: unified3                 # unified3 | unified_ext | uicvd16 | vins13 | gengui13
    epochs: 100
    batch: 8
    imgsz: 640
    lr0: 0.01
    seed: 42
    device: "0"
    workers: 4
    amp: true
    extra: {}                          # passed through to the model wrapper

    # Optional WandB config (omit or set wandb_project: null to disable)
    wandb_project: uidet-thesis
    wandb_entity: null                 # your wandb username; null = default
    wandb_tags: []                     # e.g. [local, nano, gengui]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .models.base import TrainConfig, build_detector
from .utils.io import dump_yaml, load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARED_DIR = REPO_ROOT / "data" / "prepared"
RESULTS_DIR = REPO_ROOT / "results_v2"


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------

def _wandb_init(cfg_dict: dict, cfg: TrainConfig, class_names: list[str]) -> object | None:
    """Initialise a WandB run and return the run object (or None if disabled/unavailable).

    Called BEFORE detector.train() so that Ultralytics' own WandB callback
    picks up the already-active run and logs per-epoch curves into it.
    F-RCNN checks ``wandb.run is not None`` inside its own training loop.
    """
    if not cfg.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("WandB not installed -- skipping. Run: pip install wandb --break-system-packages")
        return None

    model_str = cfg_dict["model"]
    dataset    = cfg_dict["dataset"]
    taxonomy   = cfg_dict["taxonomy"]

    # Derive automatic tags; merge with user-supplied ones; deduplicate.
    auto_tags = [model_str, dataset, taxonomy, platform.node()]
    all_tags  = list(dict.fromkeys(auto_tags + list(cfg.wandb_tags)))

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        name=cfg.name,
        group=f"{dataset}_{taxonomy}",   # group by dataset+taxonomy for easy comparison
        tags=all_tags,
        config={
            # experiment identity
            "model":       model_str,
            "dataset":     dataset,
            "taxonomy":    taxonomy,
            "num_classes": len(class_names),
            "class_names": class_names,
            # hyperparameters
            "epochs":     cfg.epochs,
            "batch_size": cfg.batch,
            "imgsz":      cfg.imgsz,
            "lr0":        cfg.lr0,
            "seed":       cfg.seed,
            "amp":        cfg.amp,
            # hardware / environment
            "device":   cfg.device,
            "workers":  cfg.workers,
            "hostname": platform.node(),
            "python":   sys.version,
            # model-specific extras (flattened)
            **{f"extra/{k}": v for k, v in (cfg.extra or {}).items()},
        },
        resume="allow",  # safe to re-run: resumes existing run with same name
    )
    return run


def _wandb_log_epochs_from_csv(out_dir: Path, run) -> None:
    """Read Ultralytics results.csv and log every epoch to WandB.

    Ultralytics' own WandB callback sometimes fails to fire; this is the
    reliable fallback that always works regardless of Ultralytics internals.
    """
    if run is None:
        return
    csv_path = out_dir / "results.csv"
    if not csv_path.exists():
        print("WandB: results.csv not found, skipping per-epoch logging.")
        return
    try:
        import csv as _csv
        import wandb
        with csv_path.open() as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        for row in rows:
            epoch = int(float(row["epoch"].strip()))
            log_dict: dict = {}
            for k, v in row.items():
                k = k.strip()
                if k == "epoch":
                    continue
                try:
                    log_dict[k] = float(v.strip())
                except (ValueError, TypeError, AttributeError):
                    pass
            wandb.log(log_dict, step=epoch)
        print(f"WandB: logged {len(rows)} epochs from results.csv")
    except Exception as exc:
        print(f"WandB epoch logging failed (non-fatal): {exc}")


def _wandb_log_summary(run, metrics: dict, split: str = "val") -> None:
    """Push final COCO eval metrics to WandB summary values (no extra log step)."""
    if run is None:
        return
    try:
        import wandb
        logged = {f"{split}/{k}": v for k, v in metrics.items()}
        for k, v in logged.items():
            run.summary[k] = v
    except Exception as exc:
        print(f"WandB summary logging failed (non-fatal): {exc}")


def _wandb_finish(run) -> None:
    if run is None:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data resolution helpers
# ---------------------------------------------------------------------------

def resolve_data_paths(dataset: str, taxonomy: str) -> tuple[Path, Path | None, Path | None]:
    base = PREPARED_DIR / dataset / taxonomy
    data_yaml = base / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Could not find {data_yaml}. Did you run scripts/prepare_{dataset.split('_')[0]}.py?"
        )
    val_json  = base / "annotations" / f"{dataset}_{taxonomy}_val.json"
    test_json = base / "annotations" / f"{dataset}_{taxonomy}_test.json"
    return data_yaml, val_json if val_json.exists() else None, test_json if test_json.exists() else None


def get_class_names(data_yaml: Path) -> list[str]:
    info  = load_yaml(data_yaml)
    names = info.get("names", {})
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config)
    name     = cfg_dict["name"]
    model    = cfg_dict["model"]
    dataset  = cfg_dict["dataset"]
    taxonomy = cfg_dict["taxonomy"]

    data_yaml, val_json, test_json = resolve_data_paths(dataset, taxonomy)
    class_names = get_class_names(data_yaml)

    results_dir = REPO_ROOT / cfg_dict.get("results_dir", "results_v2")
    out_dir = results_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        name=name,
        out_dir=out_dir,
        data_yaml=data_yaml,
        coco_val_json=val_json,
        coco_test_json=test_json,
        epochs=int(cfg_dict.get("epochs", 100)),
        batch=int(cfg_dict.get("batch", 8)),
        imgsz=int(cfg_dict.get("imgsz", 640)),
        lr0=float(cfg_dict.get("lr0", 0.01)),
        seed=int(cfg_dict.get("seed", 42)),
        device=str(cfg_dict.get("device", "0")),
        workers=int(cfg_dict.get("workers", 4)),
        amp=bool(cfg_dict.get("amp", True)),
        extra=dict(cfg_dict.get("extra", {})),
        # WandB -- all optional; skipped if wandb_project is absent / null
        wandb_project=cfg_dict.get("wandb_project") or None,
        wandb_entity=cfg_dict.get("wandb_entity") or None,
        wandb_tags=list(cfg_dict.get("wandb_tags", [])),
    )

    # Init WandB BEFORE training so Ultralytics' callback finds the active run.
    wandb_run = _wandb_init(cfg_dict, cfg, class_names)

    detector = build_detector(model, num_classes=len(class_names), class_names=class_names)
    print(f"Training {model} on {dataset}/{taxonomy} ({len(class_names)} classes) -> {out_dir}")

    weights = detector.train(cfg)
    print(f"Best weights: {weights}")

    # Log all per-epoch curves from results.csv (reliable regardless of Ultralytics internals).
    _wandb_log_epochs_from_csv(out_dir, wandb_run)

    metrics = detector.evaluate(cfg, split="val")
    print("Val metrics:", json.dumps(metrics, indent=2))

    # Push final COCO eval metrics as WandB summary values, then close the run.
    _wandb_log_summary(wandb_run, metrics, split="val")
    _wandb_finish(wandb_run)

    dump_yaml({"config": cfg_dict, "weights": str(weights), "val_metrics": metrics},
              out_dir / "summary.yaml")


if __name__ == "__main__":
    main()
