"""Gradio demo — UI Element Detection.

Run:
    python demo/app.py
"""

from __future__ import annotations

import colorsys
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2

warnings.filterwarnings("ignore", message=".*trace.*")
warnings.filterwarnings("ignore", message=".*Converting a tensor.*")
logging.getLogger("rf-detr").setLevel(logging.ERROR)
logging.getLogger("rfdetr").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Filter out Gradio's "To create a public link" message
class _GradioFilter(logging.Filter):
    def filter(self, record):
        return "public link" not in record.getMessage()

logging.getLogger("gradio").addFilter(_GradioFilter())

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import gradio as gr  # noqa: E402

from uidet.models.base import build_detector  # noqa: E402
from uidet.train import RESULTS_DIR  # noqa: E402

# Class names per taxonomy — hardcoded so the demo works without prepared datasets
_CLASS_NAMES: dict[str, list[str]] = {
    "uicvd16": [
        "Icon", "TextLabel", "MenuItem", "Row", "Button", "SubmenuItem",
        "NavigationItem", "DropdownItem", "InputField", "SectionTitle",
        "TitleBar", "Menu", "WorkingArea", "VerticalMenu", "NavigationMenu", "TableHeader",
    ],
    "gengui13": [
        "Text", "Icon", "Container", "MenuItem", "Button", "InputField",
        "TableColumn", "Row", "Menu", "WorkingArea", "Image", "Table", "Footer",
    ],
    "vins13": [
        "Text", "Icon", "Image", "TextButton", "UpperTaskBar", "EditText",
        "PageIndicator", "CheckedTextView", "Modal", "Toolbar", "Drawer", "Switch", "Card",
    ],
    "unified3": ["Button", "Icon", "Text"],
}

# ── Metadata ─────────────────────────────────────────────────────────────────

AUTHOR     = "Haritonov Rares-Costin"
UNIVERSITY = "Babeș-Bolyai University"

_TAXONOMIES = ["gengui13", "uicvd16", "vins13", "unified3"]
_DATASETS   = ["gengui", "uicvd", "vins"]

DATASET_TYPES = {
    "Desktop Application": "uicvd",
    "Web Application":     "gengui",
    "Mobile Application":  "vins",
}

DATASET_DESCRIPTIONS = {
    "Desktop Application": (
        "Trained on <strong>UICVD</strong> — a dataset of native desktop application "
        "screenshots collected for Robotic Process Automation research. "
        "Covers 16 fine-grained component classes such as menus, toolbars, input fields, "
        "title bars and navigation elements typical of desktop software."
    ),
    "Web Application": (
        "Trained on <strong>GenGUI</strong> — a dataset of web user interface images "
        "synthetically generated using ChatGPT (ICAART 2025). "
        "Covers 13 component classes found in dashboards, landing pages and web forms "
        "including containers, buttons, tables, footers and text elements."
    ),
    "Mobile Application": (
        "Trained on <strong>VINS</strong> — a dataset of high-fidelity iPhone and Android "
        "app screenshots introduced at CHI 2021. "
        "Covers 13 mobile UI component classes including toolbars, cards, switches, "
        "drawers, modals and text buttons across popular app categories."
    ),
}

MODEL_DISPLAY = {
    "rfdetr_m":          "RF-DETR Medium",
    "rtdetr_l":          "RT-DETR Large",
    "yolov12n":          "YOLOv12 Nano",
    "yolov8n":           "YOLOv8 Nano",
    "yolo11n":           "YOLO11 Nano",
    "yolo26n":           "YOLO26 Nano",
    "fasterrcnn_r50_v2": "Faster R-CNN ResNet-50 v2",
}

MODEL_DESCRIPTIONS = {
    "rfdetr_m":          "Transformer detector with a DINOv2 ViT backbone. Highest accuracy, slower inference.",
    "rtdetr_l":          "Real-Time DETR with a CNN-Transformer hybrid backbone (HGNetV2).",
    "yolov12n":          "Attention-based YOLO. Best accuracy among the YOLO family in this study.",
    "yolov8n":           "Reliable, battle-tested YOLO baseline with strong general performance.",
    "yolo11n":           "Latest standard YOLO release with incremental improvements over v8.",
    "yolo26n":           "Larger YOLO variant optimised for natural images. Underperforms on UI data.",
    "fasterrcnn_r50_v2": "Classic two-stage detector. Strong localization precision, lower throughput.",
}

EXPERIMENT_METRICS: dict[str, dict] = {
    "rfdetr_m_uicvd_uicvd16":           {"mAP": 0.8566, "fps": 4.68},
    "rfdetr_m_gengui_gengui13":          {"mAP": 0.8071, "fps": 4.34},
    "rfdetr_m_vins_vins13":              {"mAP": 0.8374, "fps": 15.64},
    "rtdetr_l_uicvd_uicvd16":            {"mAP": 0.4972, "fps": 8.94},
    "rtdetr_l_gengui_gengui13":          {"mAP": 0.7517, "fps": 6.48},
    "rtdetr_l_vins_vins13":              {"mAP": 0.8714, "fps": 22.87},
    "yolov12n_uicvd_uicvd16":            {"mAP": 0.7539, "fps": 13.72},
    "yolov12n_gengui_gengui13":          {"mAP": 0.8151, "fps": 11.68},
    "yolov12n_vins_vins13":              {"mAP": 0.8380, "fps": 17.56},
    "yolov8n_uicvd_uicvd16":             {"mAP": 0.7663, "fps": 7.04},
    "yolov8n_gengui_gengui13":           {"mAP": 0.7962, "fps": 7.61},
    "yolov8n_vins_vins13":               {"mAP": 0.8202, "fps": 26.68},
    "yolo11n_uicvd_uicvd16":             {"mAP": 0.7668, "fps": 8.21},
    "yolo11n_gengui_gengui13":           {"mAP": 0.7814, "fps": 7.64},
    "yolo11n_vins_vins13":               {"mAP": 0.8296, "fps": 20.88},
    "yolo26n_uicvd_uicvd16":             {"mAP": 0.5745, "fps": 4.74},
    "yolo26n_gengui_gengui13":           {"mAP": 0.6638, "fps": 9.06},
    "yolo26n_vins_vins13":               {"mAP": 0.7971, "fps": 22.44},
    "fasterrcnn_r50_v2_uicvd_uicvd16":   {"mAP": 0.7735, "fps": 7.16},
    "fasterrcnn_r50_v2_gengui_gengui13": {"mAP": 0.7313, "fps": 4.59},
    "fasterrcnn_r50_v2_vins_vins13":     {"mAP": 0.8009, "fps": 7.71},
}


def _rating(mAP: float) -> str:
    if mAP >= 0.85: return "Excellent"
    if mAP >= 0.77: return "Good"
    if mAP >= 0.65: return "Fair"
    return "Weak"


def _rating_color(mAP: float) -> str:
    if mAP >= 0.85: return "#22c55e"
    if mAP >= 0.77: return "#6366f1"
    if mAP >= 0.65: return "#f59e0b"
    return "#ef4444"


# ── Colors ───────────────────────────────────────────────────────────────────

def _class_palette(class_names: list[str]) -> dict[str, tuple[int, int, int]]:
    """Returns {class_name: (R, G, B)}."""
    n = max(1, len(class_names))
    return {
        name: tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 0.7, 0.95))
        for i, name in enumerate(class_names)
    }


def _rgb_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Drawing ──────────────────────────────────────────────────────────────────

def _palette_bgr(palette_rgb: dict) -> dict:
    return {k: (b, g, r) for k, (r, g, b) in palette_rgb.items()}


def draw_transparent(image_path: Path, dets, palette_rgb: dict, out_path: Path) -> Path:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    pal = _palette_bgr(palette_rgb)
    overlay = img.copy()
    for d in dets:
        color = pal.get(d.class_name, (0, 200, 0))
        cv2.rectangle(overlay, (int(d.xmin), int(d.ymin)), (int(d.xmax), int(d.ymax)), color, -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
    for d in dets:
        color = pal.get(d.class_name, (0, 200, 0))
        cv2.rectangle(img, (int(d.xmin), int(d.ymin)), (int(d.xmax), int(d.ymax)), color, 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path


def draw_highlighted(image_path: Path, dets, palette_rgb: dict, out_path: Path, cls: str) -> Path:
    """Draw with one class highlighted, all others dimmed."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    pal = _palette_bgr(palette_rgb)

    # Dim fill for non-highlighted
    overlay = img.copy()
    for d in dets:
        if d.class_name != cls:
            color = pal.get(d.class_name, (0, 200, 0))
            cv2.rectangle(overlay, (int(d.xmin), int(d.ymin)), (int(d.xmax), int(d.ymax)), color, -1)
    cv2.addWeighted(overlay, 0.04, img, 0.96, 0, img)

    # Dim borders for non-highlighted
    for d in dets:
        if d.class_name != cls:
            color = pal.get(d.class_name, (0, 200, 0))
            cv2.rectangle(img, (int(d.xmin), int(d.ymin)), (int(d.xmax), int(d.ymax)), color, 1)

    # Strong fill for highlighted class
    overlay2 = img.copy()
    for d in dets:
        if d.class_name == cls:
            color = pal.get(d.class_name, (0, 200, 0))
            cv2.rectangle(overlay2, (int(d.xmin), int(d.ymin)), (int(d.xmax), int(d.ymax)), color, -1)
    cv2.addWeighted(overlay2, 0.35, img, 0.65, 0, img)

    # Bold borders for highlighted class
    for d in dets:
        if d.class_name == cls:
            color = pal.get(d.class_name, (0, 200, 0))
            cv2.rectangle(img, (int(d.xmin), int(d.ymin)), (int(d.xmax), int(d.ymax)), color, 3)

    cv2.imwrite(str(out_path), img)
    return out_path


def class_chips_html(class_stats: dict, palette_rgb: dict, selected: str = "All") -> str:
    if not class_stats:
        return ""
    all_active = selected == "All"
    chips = []

    # "All" chip
    all_style = (
        "background:#6366f133;border:1px solid #6366f1;color:#e2e8f0;"
        if all_active else
        "background:#1e2538;border:1px solid #1e2538;color:#64748b;"
    )
    chips.append(
        f'<span style="display:inline-flex;align-items:center;gap:0.3rem;{all_style}'
        f'border-radius:5px;padding:0.2rem 0.65rem;font-size:0.78rem;font-weight:600;'
        f'white-space:nowrap;cursor:default;">All</span>'
    )

    for cls, scores in sorted(class_stats.items(), key=lambda x: -len(x[1])):
        r, g, b = palette_rgb.get(cls, (150, 150, 150))
        hex_col = _rgb_hex(r, g, b)
        count = len(scores)
        active = (selected == cls)
        bg = f"{hex_col}28" if not active else f"{hex_col}55"
        border = f"{hex_col}55" if not active else hex_col
        text_col = "#94a3b8" if not active else "#e2e8f0"
        chips.append(
            f'<span style="display:inline-flex;align-items:center;gap:0.35rem;'
            f'background:{bg};border:1px solid {border};color:{text_col};'
            f'border-radius:5px;padding:0.2rem 0.6rem;font-size:0.78rem;font-weight:500;'
            f'white-space:nowrap;cursor:default;">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{hex_col};'
            f'flex-shrink:0;opacity:{"1" if active else "0.6"};"></span>'
            f'{cls} <span style="font-weight:700;color:{hex_col};">{count}</span></span>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:0.35rem;padding:0.5rem 0 0.25rem;">'
        + "".join(chips) + "</div>"
    )


# ── Run discovery ────────────────────────────────────────────────────────────

def _parse_run_name(name: str) -> tuple[str, str, str] | None:
    for taxonomy in _TAXONOMIES:
        if name.endswith(f"_{taxonomy}"):
            rest = name[: -len(taxonomy) - 1]
            for dataset in _DATASETS:
                if f"_{dataset}_" in rest or rest.endswith(f"_{dataset}"):
                    idx = rest.rfind(f"_{dataset}")
                    model = rest[:idx]
                    return model, dataset, taxonomy
    return None


def find_runs() -> dict[str, Path]:
    runs: dict[str, Path] = {}
    if not RESULTS_DIR.exists():
        return runs
    for run_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        weights = run_dir / "weights" / "best.pt"
        if weights.exists() and _parse_run_name(run_dir.name) is not None:
            runs[run_dir.name] = run_dir
    return runs


def _model_choices(dataset_label: str, runs: dict) -> list[tuple[str, str]]:
    dataset_key = DATASET_TYPES.get(dataset_label, "uicvd")
    choices = []
    for run_name in runs:
        parsed = _parse_run_name(run_name)
        if not parsed:
            continue
        model_key, dataset, _ = parsed
        if dataset != dataset_key:
            continue
        m    = EXPERIMENT_METRICS.get(run_name, {})
        mAP  = m.get("mAP", 0.0)
        fps  = m.get("fps", 0.0)
        label = f"{MODEL_DISPLAY.get(model_key, model_key)}  —  {mAP:.1%}  {_rating(mAP)}  {fps:.0f} FPS"
        choices.append((label, run_name))
    choices.sort(key=lambda x: EXPERIMENT_METRICS.get(x[1], {}).get("mAP", 0), reverse=True)
    return choices


def _model_info_html(run_name: str | None) -> str:
    if not run_name:
        return ""
    parsed = _parse_run_name(run_name)
    if not parsed:
        return ""
    model_key, _, _ = parsed
    m      = EXPERIMENT_METRICS.get(run_name, {})
    mAP    = m.get("mAP", 0.0)
    fps    = m.get("fps", 0.0)
    desc   = MODEL_DESCRIPTIONS.get(model_key, "")
    color  = _rating_color(mAP)
    rating = _rating(mAP)
    return (
        f'<div style="border:1px solid #1e2538;border-radius:8px;padding:0.7rem 0.9rem;'
        f'margin-top:0.4rem;font-size:0.8rem;line-height:1.6;background:#0b0e18;">'
        f'<div style="display:flex;align-items:center;gap:0.9rem;margin-bottom:0.35rem;flex-wrap:wrap;">'
        f'<span style="color:#64748b;">mAP&ensp;<strong style="color:#e2e8f0;">{mAP:.1%}</strong></span>'
        f'<span style="color:#64748b;">Speed&ensp;<strong style="color:#e2e8f0;">{fps:.0f} FPS</strong></span>'
        f'<span style="background:{color}22;color:{color};border-radius:4px;'
        f'padding:0.1rem 0.45rem;font-weight:700;font-size:0.7rem;letter-spacing:0.04em;">{rating}</span>'
        f'</div>'
        f'<div style="color:#64748b;">{desc}</div>'
        f'</div>'
    )


# ── Inference ────────────────────────────────────────────────────────────────

_loaded: dict[str, tuple] = {}
_OUT_PATH = REPO_ROOT / "demo" / "_last_pred.png"

# Module-level state for highlight (single-user local demo)
_last_image_path: Path | None = None
_last_dets: list = []
_last_palette: dict = {}
_last_class_names: list = []
_last_class_stats: dict = {}


def _load(name: str):
    if name in _loaded:
        return _loaded[name]
    run_dir = find_runs()[name]
    model_key, dataset, taxonomy = _parse_run_name(name)
    class_names = _CLASS_NAMES[taxonomy]
    det = build_detector(model_key, num_classes=len(class_names), class_names=class_names)
    det.load(run_dir / "weights" / "best.pt")
    _loaded[name] = (det, class_names)
    return _loaded[name]


def predict(image, run_name, conf, iou):
    global _last_image_path, _last_dets, _last_palette, _last_class_names, _last_class_stats
    if image is None or not run_name:
        return None, "", [], gr.update(choices=["All"], value="All")

    det, class_names = _load(run_name)
    dets = sorted(det.predict(Path(image), conf=conf, iou=iou), key=lambda d: d.score, reverse=True)

    palette = _class_palette(class_names)
    draw_transparent(Path(image), dets, palette, _OUT_PATH)

    class_stats: dict[str, list[float]] = defaultdict(list)
    for d in dets:
        class_stats[d.class_name].append(d.score)

    # Save for highlight
    _last_image_path = Path(image)
    _last_dets       = dets
    _last_palette    = palette
    _last_class_names = class_names
    _last_class_stats = dict(class_stats)

    chips = class_chips_html(class_stats, palette, "All")
    rows  = [
        [cls, len(scores), f"{sum(scores)/len(scores):.4f}"]
        for cls, scores in sorted(class_stats.items(), key=lambda x: -len(x[1]))
    ]
    highlight_choices = ["All"] + sorted(class_stats.keys())

    return (
        str(_OUT_PATH),
        chips,
        rows,
        gr.update(choices=highlight_choices, value="All"),
    )


def on_highlight(cls_choice):
    if not _last_image_path or not _last_dets:
        return None, ""
    if cls_choice is None or cls_choice == "All":
        draw_transparent(_last_image_path, _last_dets, _last_palette, _OUT_PATH)
        chips = class_chips_html(_last_class_stats, _last_palette, "All")
    else:
        draw_highlighted(_last_image_path, _last_dets, _last_palette, _OUT_PATH, cls_choice)
        chips = class_chips_html(_last_class_stats, _last_palette, cls_choice)
    return str(_OUT_PATH), chips


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0b0e18 !important;
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
    max-width: 100% !important;
}

/* Header */
#app-header {
    background: #0f1117;
    border-bottom: 1px solid #1e2538;
    padding: 1.4rem 2rem;
    margin-bottom: 0;
}
#app-header h1 {
    font-size: 1.6rem; font-weight: 700; color: #f1f5f9;
    margin: 0 0 0.2rem; letter-spacing: -0.02em;
}
#app-header p { font-size: 0.8rem; color: #475569; margin: 0; }

/* Layout */
#main-row { padding: 1.25rem !important; gap: 1.25rem !important; align-items: flex-start !important; }

/* Sidebar */
#sidebar {
    background: #0f1117 !important;
    border: 1px solid #1e2538 !important;
    border-radius: 10px !important;
    padding: 1.25rem !important;
}

/* Image panels — clip loading overlay but allow Gradio fullscreen */
.img-panel {
    background: #0f1117 !important;
    border: 1px solid #1e2538 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Panels */
.panel {
    background: #0f1117 !important;
    border: 1px solid #1e2538 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
/* Bottom results panel */
.panel-results {
    background: #0f1117 !important;
    border: 1px solid #1e2538 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    position: relative !important;
    z-index: 0 !important;
}
/* Each block inside results panel is a positioning context so status trackers stay inside */
.panel-results .block { position: relative !important; }
/* Hide the loading spinner on the chips HTML block — it has no height so it bleeds out */
.panel-results .hide-container [data-testid="status-tracker"] { display: none !important; }

/* Section titles */
.section-title {
    display: block;
    font-size: 0.65rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    color: #334155 !important; padding-bottom: 0.4rem !important;
    border-bottom: 1px solid #1a1f2e !important;
    margin: 1.1rem 0 0.7rem !important;
}

/* Hide duplicate "Screenshot type" label inside the fieldset */
fieldset [data-testid="block-info"] { display: none !important; }

/* Radio container */
fieldset {
    background: #161b27 !important;
    border: 1px solid #1e2538 !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
    margin: 0 !important;
}

/* Radio labels */
label.svelte-19qdtil {
    display: flex !important;
    align-items: center !important;
    padding: 0.4rem 0.6rem !important;
    margin: 0.1rem 0.2rem !important;
    border-radius: 6px !important;
    cursor: pointer !important;
}
label.svelte-19qdtil:hover { background: #1e2538 !important; }

/* Radio label text */
label.svelte-19qdtil span {
    color: #cbd5e1 !important;
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}
input[type="radio"] { accent-color: #6366f1 !important; }

/* Inputs & select */
select, input:not([type="radio"]):not([type="range"]), textarea {
    background: #161b27 !important;
    border: 1px solid #1e2538 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-size: 0.875rem !important;
}

/* Dropdown wrapper */
.gr-dropdown { width: 100% !important; }
.gr-dropdown select { width: 100% !important; padding: 0.5rem 0.75rem !important; }

/* Sliders */
input[type=range] { accent-color: #6366f1; }
.info { color: #475569 !important; font-size: 0.75rem !important; }

/* Highlight dropdown */
#highlight-dd select {
    background: #161b27 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e2538 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}
#highlight-dd .label-wrap span {
    font-size: 0.65rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
    color: #475569 !important;
}

/* Upload button */
.img-panel .upload-button {
    width: 100% !important;
    margin-top: 0.5rem !important;
    background: #1e2538 !important;
    border: 1px solid #2d3748 !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
}
.img-panel .upload-button:hover { background: #2d3748 !important; color: #e2e8f0 !important; }

/* Detect button */
#detect-btn {
    background: #6366f1 !important; border: none !important;
    border-radius: 6px !important; color: #fff !important;
    font-size: 0.875rem !important; font-weight: 600 !important;
    padding: 0.65rem !important; width: 100% !important;
    margin-top: 1rem !important; transition: background 0.15s !important;
}
#detect-btn:hover { background: #4f46e5 !important; }

/* Dataframe */
.gr-dataframe table { border-collapse: collapse !important; width: 100% !important; }
.gr-dataframe th {
    background: #0b0e18 !important; color: #475569 !important;
    font-size: 0.68rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
    padding: 0.5rem 0.75rem !important; border-bottom: 1px solid #1e2538 !important;
}
.gr-dataframe td {
    color: #cbd5e1 !important; font-size: 0.825rem !important;
    padding: 0.45rem 0.75rem !important; border-bottom: 1px solid #161b27 !important;
}
.gr-dataframe tr:last-child td { border-bottom: none !important; }
.gr-dataframe tr:hover td { background: #1a1f2e !important; }

/* Hide share button */
button.icon-button[aria-label="Share"] { display: none !important; }

/* Hide Gradio chrome */
footer, .built-with { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0b0e18; }
::-webkit-scrollbar-thumb { background: #1e2538; border-radius: 99px; }
"""

# ── Layout ───────────────────────────────────────────────────────────────────

def main():
    runs = find_runs()
    if not runs:
        print("No trained runs found in results_v2/. Train a model first.")
        return

    default_dataset = "Desktop Application"
    default_choices = _model_choices(default_dataset, runs)
    default_run     = default_choices[0][1] if default_choices else None

    with gr.Blocks(title="UI Element Detection") as demo:

        gr.HTML(f"""
            <div id="app-header">
                <h1>UI Element Detection</h1>
                <p>Bachelor's Thesis &nbsp;·&nbsp; {AUTHOR} &nbsp;·&nbsp; {UNIVERSITY}</p>
            </div>
        """)

        with gr.Row(elem_id="main-row", equal_height=False):

            # ── Sidebar ──────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=420, elem_id="sidebar"):

                gr.HTML('<span class="section-title">Screenshot type</span>')
                dataset_radio = gr.Radio(
                    choices=list(DATASET_TYPES.keys()),
                    value=default_dataset,
                    label="Screenshot type",
                )
                dataset_desc = gr.HTML(
                    f'<p style="font-size:0.78rem;color:#475569;margin:0.5rem 0 0;line-height:1.55;">'
                    f'{DATASET_DESCRIPTIONS[default_dataset]}</p>'
                )

                gr.HTML('<span class="section-title">Model</span>')
                model_dd = gr.Dropdown(
                    choices=default_choices,
                    value=default_run,
                    label="Model",
                )
                model_info = gr.HTML(_model_info_html(default_run))

                gr.HTML('<span class="section-title">Parameters</span>')
                conf = gr.Slider(
                    0.05, 0.95, value=0.25, step=0.05,
                    label="Confidence threshold",
                    info="Higher → fewer but more certain detections.",
                )
                iou = gr.Slider(
                    0.1, 0.9, value=0.45, step=0.05,
                    label="IoU — NMS threshold",
                    info="Lower → suppress more overlapping boxes.",
                )

                btn = gr.Button("Run detection", variant="primary", elem_id="detect-btn")

            # ── Main area ────────────────────────────────────────────────────
            with gr.Column(scale=3):

                with gr.Row(equal_height=True):
                    with gr.Column(elem_classes="img-panel"):
                        inp = gr.Image(
                            type="filepath",
                            label="Input screenshot",
                            interactive=False,
                            height=460,
                        )
                        upload_btn = gr.UploadButton(
                            "Upload screenshot",
                            file_types=["image"],
                            size="sm",
                            variant="secondary",
                        )
                    with gr.Column(elem_classes="img-panel"):
                        out_img = gr.Image(
                            label="Detected elements",
                            height=460,
                            interactive=False,
                        )

                with gr.Column(elem_classes="panel-results"):
                    highlight_dd = gr.Dropdown(
                        choices=["All"],
                        value="All",
                        label="Highlight class",
                        info="Select a class to isolate its detections in the image.",
                        elem_id="highlight-dd",
                    )
                    out_chips = gr.HTML("")
                    out_tbl = gr.Dataframe(
                        headers=["Class", "Instances", "Avg. confidence"],
                        label="Detection summary",
                        value=[],
                        row_count=(1, "dynamic"),
                        column_count=(3, "fixed"),
                        wrap=False,
                    )

        # ── Handlers ────────────────────────────────────────────────────────

        def on_dataset_change(label):
            choices  = _model_choices(label, find_runs())
            best_run = choices[0][1] if choices else None
            return (
                gr.update(choices=choices, value=best_run),
                f'<p style="font-size:0.78rem;color:#475569;margin:0.5rem 0 0;line-height:1.55;">'
                f'{DATASET_DESCRIPTIONS.get(label, "")}</p>',
                _model_info_html(best_run),
            )

        upload_btn.upload(lambda f: f, [upload_btn], [inp])

        dataset_radio.change(on_dataset_change, [dataset_radio], [model_dd, dataset_desc, model_info])
        model_dd.change(lambda r: _model_info_html(r), [model_dd], [model_info])
        btn.click(predict, [inp, model_dd, conf, iou], [out_img, out_chips, out_tbl, highlight_dd])
        highlight_dd.change(on_highlight, [highlight_dd], [out_img, out_chips])

        demo.load(
            None,
            js="""() => {
                // Gradio sets overflow:hidden inline on fixed-height blocks,
                // which prevents its own fullscreen from working on the input image.
                function fixOverflow() {
                    document.querySelectorAll('.img-panel .block').forEach(el => {
                        el.style.overflow = 'visible';
                    });
                }
                fixOverflow();
                // Re-apply after Gradio re-renders (e.g. after image upload)
                new MutationObserver(fixOverflow).observe(
                    document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['style'] }
                );
            }"""
        )


    demo.launch(css=CSS, theme=gr.themes.Base(), share=False, inbrowser=True)


if __name__ == "__main__":
    main()
