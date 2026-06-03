"""Shared utilities for dataset adapters."""

from __future__ import annotations

from pathlib import Path

import yaml

ONTOLOGY_DIR = Path(__file__).resolve().parents[4] / "data" / "ontology"


def load_all_taxonomies() -> dict[str, list[str]]:
    """Return {taxonomy_name: [classes_in_order]} for every taxonomy
    defined in data/ontology/unified_ontology.yaml."""
    with (ONTOLOGY_DIR / "unified_ontology.yaml").open() as fh:
        ont = yaml.safe_load(fh)
    out: dict[str, list[str]] = {}
    for tax_name, body in ont.items():
        if isinstance(body, dict) and "classes" in body:
            out[tax_name] = list(body["classes"])
    return out


def load_mapping(dataset_name: str) -> dict[str, dict[str, str]]:
    """Returns {taxonomy: {native_class: unified_class_or___ignore__}} dict
    from data/ontology/mappings/<dataset>_to_unified.yaml. Taxonomies absent
    from the file simply yield empty mappings (so unmapped classes drop to
    None for that taxonomy)."""
    path = ONTOLOGY_DIR / "mappings" / f"{dataset_name}_to_unified.yaml"
    with path.open() as fh:
        return yaml.safe_load(fh)


def map_native_to_all(
    native_class: str, mapping: dict[str, dict[str, str]]
) -> dict[str, str | None]:
    """Returns {taxonomy_name: label_or_None} for every taxonomy in ``mapping``.

    A label of None means the box is ignored for that taxonomy (either the
    native class isn't present in the mapping at all, or it explicitly maps
    to ``__ignore__``).
    """
    out: dict[str, str | None] = {}
    for tax_name, tax_map in mapping.items():
        v = tax_map.get(native_class)
        if v is None or v == "__ignore__":
            out[tax_name] = None
        else:
            out[tax_name] = v
    return out


# ---- legacy shims (kept so older code keeps importing) --------------
def load_unified_classes() -> tuple[list[str], list[str]]:
    """Legacy two-tuple accessor. Prefer ``load_all_taxonomies()``."""
    tax = load_all_taxonomies()
    return tax.get("unified3", []), tax.get("unified_ext", [])


def map_classes(
    native_class: str, mapping: dict[str, dict[str, str]]
) -> tuple[str | None, str | None]:
    """Legacy two-tuple accessor. Prefer ``map_native_to_all()``."""
    labels = map_native_to_all(native_class, mapping)
    return labels.get("unified3"), labels.get("unified_ext")
