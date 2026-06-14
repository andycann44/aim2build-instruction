"""Qty slot → significant island label binding with migration verdict overrides."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from clean.services.full_crop_mask_paths import (
    filter_significant_islands,
    find_full_mask_stem,
    master_islands_from_mask,
    raw_master_mask_path,
    sort_islands_for_slots,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BINDING_AUDIT = _REPO_ROOT / "debug" / "spatial_qty_island_binding_audit" / "binding_audit.json"
_VERDICTS = _REPO_ROOT / "debug" / "spatial_qty_island_binding_audit" / "migration_verdicts.json"


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def island_bbox_center(island: Dict[str, Any]) -> Tuple[float, float]:
    x, y, w, h = [float(v) for v in (island.get("bbox") or [0, 0, 0, 0])[:4]]
    return (x + w / 2.0, y + h / 2.0)


def qty_token_center(token: Dict[str, Any]) -> Tuple[float, float]:
    x = float(token.get("x", 0) or 0)
    y = float(token.get("y", 0) or 0)
    w = float(token.get("w", 0) or 0)
    h = float(token.get("h", 0) or 0)
    cx = token.get("cx")
    cy = token.get("cy")
    if cx is not None and cy is not None:
        return (float(cx), float(cy))
    return (x + w / 2.0, y + h / 2.0)


def resolve_nearest_significant_island_label(
    qty_centre: Tuple[float, float],
    significant_islands: List[Dict[str, Any]],
) -> Optional[int]:
    if not significant_islands:
        return None
    qx, qy = qty_centre
    best: Optional[Tuple[float, int]] = None
    for island in significant_islands:
        ix, iy = island_bbox_center(island)
        dist2 = (qx - ix) ** 2 + (qy - iy) ** 2
        label = int(island["label"])
        if best is None or dist2 < best[0]:
            best = (dist2, label)
    return best[1] if best else None


def significant_islands_for_stem(stem: str) -> List[Dict[str, Any]]:
    raw_path = raw_master_mask_path(stem)
    if not raw_path.is_file():
        return []
    return filter_significant_islands(
        sort_islands_for_slots(master_islands_from_mask(str(raw_path)))
    )


def order_island_label(
    significant_islands: List[Dict[str, Any]],
    slot_index: int,
) -> Optional[int]:
    if 0 <= int(slot_index) < len(significant_islands):
        return int(significant_islands[int(slot_index)]["label"])
    return None


@lru_cache(maxsize=1)
def _load_verdict_map() -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not _VERDICTS.is_file():
        return {}
    payload = json.loads(_VERDICTS.read_text(encoding="utf-8"))
    mapping: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for entry in payload.get("verdicts") or []:
        crop_id = str(entry.get("crop_id") or "")
        slot_index = _coerce_int(entry.get("slot_index"))
        if not crop_id or slot_index is None:
            continue
        mapping[(crop_id, int(slot_index))] = dict(entry)
    return mapping


@lru_cache(maxsize=1)
def _load_binding_current_map() -> Dict[Tuple[str, int], int]:
    if not _BINDING_AUDIT.is_file():
        return {}
    payload = json.loads(_BINDING_AUDIT.read_text(encoding="utf-8"))
    mapping: Dict[Tuple[str, int], int] = {}
    for crop in payload.get("crops") or []:
        crop_id = str(crop.get("crop_id") or "")
        for change in crop.get("changes") or []:
            slot_index = _coerce_int(change.get("slot_index"))
            current = _coerce_int(change.get("current_island"))
            if crop_id and slot_index is not None and current is not None:
                mapping[(crop_id, int(slot_index))] = int(current)
    return mapping


def resolve_island_label_for_slot(
    *,
    set_num: str,
    bag: int,
    crop_id: str,
    slot_index: int,
    slot_source: str,
    qty_token_boxes: Optional[List[Dict[str, Any]]] = None,
    preset_island_label: Optional[int] = None,
) -> Optional[int]:
    """Resolve mandatory island_label for a review slot.

    Rules:
    - island_fallback / supplement slots keep explicit preset label
    - qty slots: nearest significant island bbox centre
    - migration_verdicts KEEP_CURRENT → preserve order/current island
    - migration_verdicts MIGRATE_TO_PROPOSED → coord island
    - saved_label without qty token: order island on significant list
    """
    if preset_island_label is not None:
        return int(preset_island_label)

    stem = find_full_mask_stem(str(set_num), int(bag), str(crop_id))
    if not stem:
        return None
    sig = significant_islands_for_stem(stem)
    if not sig:
        return None

    slot_i = int(slot_index)
    verdict = _load_verdict_map().get((str(crop_id), slot_i))
    tokens = [dict(t) for t in list(qty_token_boxes or []) if isinstance(t, dict)]

    coord_label: Optional[int] = None
    if str(slot_source) == "qty" and 0 <= slot_i < len(tokens):
        coord_label = resolve_nearest_significant_island_label(qty_token_center(tokens[slot_i]), sig)
    elif str(slot_source) == "saved_label":
        return order_island_label(sig, slot_i)

    if coord_label is None and str(slot_source) != "saved_label":
        return order_island_label(sig, slot_i)

    if not verdict:
        return coord_label

    verdict_name = str(verdict.get("verdict") or "")
    if verdict_name == "KEEP_CURRENT":
        current = _load_binding_current_map().get((str(crop_id), slot_i))
        if current is not None:
            return int(current)
        return order_island_label(sig, slot_i)
    if verdict_name == "MIGRATE_TO_PROPOSED":
        return coord_label
    if verdict_name == "RELABEL_MANUAL":
        return coord_label or order_island_label(sig, slot_i)

    return coord_label
