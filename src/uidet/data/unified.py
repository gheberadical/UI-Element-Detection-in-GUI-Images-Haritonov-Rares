"""Canonical in-memory representation of a labelled dataset.

Every dataset adapter (UICVD CSV, GenGUI CSV, VINS Pascal-VOC XML) loads its
source format and produces a list of :class:`UnifiedImage` objects. The
exporters then render that list into whatever a given detector expects (YOLO
text labels, COCO JSON, …).

Boxes carry both the original native class string and a *dictionary* of
labels keyed by taxonomy name. Adding a new taxonomy is now a YAML change
plus a per-adapter mapping; no code edits required.

Taxonomies currently in use:
    "unified3"     — cross-dataset 3 classes (Text/Button/Icon), KES2025/INNOCOMP2025 parity
    "unified_ext"  — cross-dataset ~12 classes, novel extension
    "gengui13"     — GenGUI native 13 classes (Dicu et al. ICAART 2024)
    …add more by editing data/ontology/unified_ontology.yaml + per-dataset mappings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UnifiedBox:
    """A single annotated bounding box in absolute pixel coordinates (xyxy).

    ``labels`` maps taxonomy name -> class name (or None if the box is
    ignored under that taxonomy). Adapters fill this dict by consulting
    the appropriate <dataset>_to_unified.yaml mapping.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    native_class: str
    labels: dict[str, str | None] = field(default_factory=dict)

    # ---- backward-compat shims (keep older code working) ------------
    @property
    def unified3(self) -> str | None:
        return self.labels.get("unified3")

    @property
    def unified_ext(self) -> str | None:
        return self.labels.get("unified_ext")

    def label_for(self, taxonomy: str) -> str | None:
        """Generic accessor; preferred over the named properties."""
        return self.labels.get(taxonomy)

    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    def area(self) -> float:
        return self.width() * self.height()

    def is_valid(self, image_w: int, image_h: int) -> bool:
        """Reject zero-area or out-of-image boxes."""
        if self.width() <= 1 or self.height() <= 1:
            return False
        if self.xmin < 0 or self.ymin < 0:
            return False
        if self.xmax > image_w or self.ymax > image_h:
            return False
        return True


@dataclass
class UnifiedImage:
    """A single labelled image."""

    image_id: str            # stable identifier (filename stem)
    image_path: Path         # absolute path on disk
    width: int
    height: int
    boxes: list[UnifiedBox] = field(default_factory=list)
    source_dataset: str = ""  # "uicvd" | "gengui" | "vins.android" | ...

    def num_annotations(self, taxonomy: str) -> int:
        return sum(1 for b in self.boxes if b.label_for(taxonomy) is not None)


@dataclass
class UnifiedDataset:
    """A collection of UnifiedImages plus the active class list per taxonomy."""

    name: str                                      # e.g. "uicvd"
    images: list[UnifiedImage]
    classes_by_taxonomy: dict[str, list[str]] = field(default_factory=dict)

    # ---- backward-compat shims --------------------------------------
    @property
    def classes_unified3(self) -> list[str]:
        return self.classes_by_taxonomy.get("unified3", [])

    @property
    def classes_unified_ext(self) -> list[str]:
        return self.classes_by_taxonomy.get("unified_ext", [])

    def class_id(self, taxonomy: str, name: str) -> int:
        classes = self.classes_by_taxonomy[taxonomy]
        return classes.index(name)

    def filter_for_taxonomy(self, taxonomy: str) -> "UnifiedDataset":
        """Return a copy with only boxes that have a non-None label in the chosen taxonomy."""
        new_images = []
        for img in self.images:
            kept = [b for b in img.boxes if b.label_for(taxonomy) is not None]
            new_images.append(
                UnifiedImage(
                    image_id=img.image_id,
                    image_path=img.image_path,
                    width=img.width,
                    height=img.height,
                    boxes=kept,
                    source_dataset=img.source_dataset,
                )
            )
        return UnifiedDataset(
            name=self.name,
            images=new_images,
            classes_by_taxonomy=dict(self.classes_by_taxonomy),
        )
