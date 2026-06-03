"""Adapter for the VINS dataset (Pascal-VOC XML annotations).

Source layout (after unzipping ``datasets/VINS Dataset.zip``):

    All Dataset/
        Android/      Annotations/Android_*.xml      JPEGImages/Android_*.jpg
        Rico/         Annotations/Rico_*.xml         JPEGImages/Rico_*.jpg
        iphone/       Annotations/iphone_*.xml       JPEGImages/iphone_*.jpg
        uplabs/       Annotations/uplabs_*.xml       JPEGImages/uplabs_*.jpg
        Wireframes/   Annotations/Wireframes_*.xml   JPEGImages/Wireframes_*.jpg

Each subset can be loaded individually (``load_vins_subset``) or all five
together (``load_vins``). The combined dataset tags ``source_dataset`` per
image so downstream code can slice mobile / wireframes / etc.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from ..unified import UnifiedBox, UnifiedDataset, UnifiedImage
from .common import load_all_taxonomies, load_mapping, map_native_to_all

VINS_SUBSETS: tuple[str, ...] = ("Android", "Rico", "iphone", "uplabs", "Wireframes")


def _parse_xml(xml_path: Path, mapping: dict, source_tag: str, image_root: Path) -> UnifiedImage | None:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()

    fname_elem = root.find("filename")
    size_elem = root.find("size")
    if fname_elem is None or size_elem is None:
        return None
    filename = fname_elem.text or ""

    img_path = image_root / filename
    if not img_path.exists():
        stem = Path(filename).stem
        cand = list(image_root.glob(f"{stem}.*"))
        if not cand:
            return None
        img_path = cand[0]

    try:
        w = int(size_elem.findtext("width", "0") or 0)
        h = int(size_elem.findtext("height", "0") or 0)
    except ValueError:
        w = h = 0
    if w <= 0 or h <= 0:
        with Image.open(img_path) as im:
            w, h = im.size

    image_id = Path(filename).stem
    boxes: list[UnifiedBox] = []
    for obj in root.findall("object"):
        native = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if not native or bb is None:
            continue
        try:
            xmin = float(bb.findtext("xmin", "0") or 0)
            ymin = float(bb.findtext("ymin", "0") or 0)
            xmax = float(bb.findtext("xmax", "0") or 0)
            ymax = float(bb.findtext("ymax", "0") or 0)
        except ValueError:
            continue

        labels = map_native_to_all(native, mapping)
        if not any(v is not None for v in labels.values()):
            continue
        box = UnifiedBox(
            xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
            native_class=native, labels=labels,
        )
        if box.is_valid(w, h):
            boxes.append(box)

    return UnifiedImage(
        image_id=image_id,
        image_path=img_path,
        width=w, height=h,
        boxes=boxes,
        source_dataset=source_tag,
    )


def load_vins_subset(root: Path, subset: str) -> UnifiedDataset:
    """Load a single VINS sub-folder, e.g. ``Android``."""
    if subset not in VINS_SUBSETS:
        raise ValueError(f"Unknown VINS subset {subset!r}; expected one of {VINS_SUBSETS}")
    sub_root = root / subset
    ann_dir = sub_root / "Annotations"
    img_dir = sub_root / "JPEGImages"
    if not ann_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"VINS subset folders missing under {sub_root}")

    mapping = load_mapping("vins")
    taxonomies = load_all_taxonomies()
    src_tag = f"vins.{subset.lower()}"

    images = []
    for xml_path in sorted(ann_dir.glob("*.xml")):
        ui = _parse_xml(xml_path, mapping, src_tag, img_dir)
        if ui is not None:
            images.append(ui)

    return UnifiedDataset(
        name=f"vins_{subset.lower()}",
        images=images,
        classes_by_taxonomy=taxonomies,
    )


def load_vins(root: Path, subsets: tuple[str, ...] = VINS_SUBSETS) -> UnifiedDataset:
    """Load multiple VINS subsets concatenated."""
    taxonomies = load_all_taxonomies()
    all_images: list[UnifiedImage] = []
    for sub in subsets:
        ds = load_vins_subset(root, sub)
        all_images.extend(ds.images)
    return UnifiedDataset(
        name="vins",
        images=all_images,
        classes_by_taxonomy=taxonomies,
    )
