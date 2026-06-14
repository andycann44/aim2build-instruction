import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote as _url_quote

from clean.services.full_crop_mask_paths import (
    ensure_island_slot_cutout_path,
    ensure_island_slot_mask_path,
    filter_significant_islands,
    find_full_mask_stem,
    master_islands_from_mask,
    raw_master_mask_path,
    sort_islands_for_slots,
    MAX_AUTO_ISLAND_SLOTS,
)
from clean.services.island_binding import order_island_label, significant_islands_for_stem


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_LABEL_DIR = _REPO_ROOT / "debug" / "training_labels"
_CROP_CACHE_DIR = _REPO_ROOT / "debug" / "crop_cache"
_STEP_SEG_DIR = _REPO_ROOT / "debug" / "ai_training" / "step_segmented_cutouts"
_PART_CUTOUT_DIR = _REPO_ROOT / "debug" / "ai_training" / "part_cutouts"
_SLOT_WINDOW_OVERLAY_DIR = _REPO_ROOT / "debug" / "ai_training" / "slot_window_overlays"


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _coerce_int_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    result: List[int] = []
    for item in value:
        parsed = _coerce_int(item)
        if parsed is not None:
            result.append(int(parsed))
    return result


def _coerce_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _label_store_path(set_num: str, bag: int) -> Path:
    safe_set = "".join(ch for ch in str(set_num or "").strip() if ch.isalnum() or ch in "-_") or "unknown"
    return _TRAINING_LABEL_DIR / f"{safe_set}_bag{int(bag)}.json"


def _crop_cache_path(set_num: str, bag: int) -> Path:
    safe_set = "".join(ch for ch in str(set_num or "").strip() if ch.isalnum() or ch in "-_") or "unknown"
    return _CROP_CACHE_DIR / f"{safe_set}_bag{int(bag)}.json"


def _empty_review_state(set_num: str, bag: int) -> Dict[str, Any]:
    return {
        "schema_version": "1.1",
        "set_num": str(set_num),
        "bag": int(bag),
        "created_at": _iso_now(),
        "source": {
            "route": "/debug/manual-match-review",
            "type": "bag_review",
            "crop_image_path_kind": "page_image_with_crop_box",
        },
        "crops": {},
    }


def load_review_state(set_num: str, bag: int) -> Dict[str, Any]:
    path = _label_store_path(set_num, int(bag))
    if not path.exists():
        return _empty_review_state(set_num, int(bag))
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_review_state(set_num, int(bag))
    if not isinstance(loaded, dict):
        return _empty_review_state(set_num, int(bag))
    loaded.setdefault("schema_version", "1.1")
    loaded.setdefault("set_num", str(set_num))
    loaded.setdefault("bag", int(bag))
    loaded.setdefault("crops", {})
    if not isinstance(loaded.get("crops"), dict):
        loaded["crops"] = {}
    return loaded


def save_review_state(set_num: str, bag: int, state: Dict[str, Any]) -> None:
    path = _label_store_path(set_num, int(bag))
    path.parent.mkdir(parents=True, exist_ok=True)
    state.setdefault("schema_version", "1.1")
    state["set_num"] = str(set_num)
    state["bag"] = int(bag)
    state.setdefault("crops", {})
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_crop_cache(set_num: str, bag: int) -> List[Dict[str, Any]]:
    path = _crop_cache_path(set_num, int(bag))
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(loaded, list):
        return []
    return [dict(item) for item in loaded if isinstance(item, dict)]


def _normalize_part_entry(data: Dict[str, Any]) -> Dict[str, Any]:
    color_id = _coerce_int(data.get("color_id"))
    selected_slot_index = _coerce_int(data.get("selected_slot_index"))
    qty = _coerce_int(data.get("qty"))
    return {
        "part_num": str(data.get("part_num") or "").strip(),
        "color_id": int(color_id or 0),
        "color_name": str(data.get("color_name") or "").strip(),
        "element_id": str(data.get("element_id") or "").strip(),
        "qty": qty,
        "qty_text": str(data.get("qty_text") or "").strip() or None,
        "selected_slot_index": selected_slot_index,
        "part_bbox": data.get("part_bbox"),
        "confidence": data.get("confidence"),
        "ai_snap_input_path": data.get("ai_snap_input_path"),
    }


def _build_qty_sequence(qty_numbers: Any, qty_text: Any) -> List[Dict[str, Any]]:
    numbers = _coerce_int_list(qty_numbers)
    texts = _coerce_str_list(qty_text)
    count = max(len(numbers), len(texts))
    result: List[Dict[str, Any]] = []
    for index in range(count):
        qty = numbers[index] if index < len(numbers) else None
        text = texts[index] if index < len(texts) else (f"{qty}x" if qty is not None else "")
        result.append({"qty": qty, "qty_text": text})
    return result


def _artifact_url(path_text: str) -> str:
    path = Path(str(path_text or "").strip())
    if not path.is_file():
        return ""
    return "/debug/ai-snap-artifact?path=" + _url_quote(str(path), safe="")


def _slot_assets(
    set_num: str,
    bag: int,
    crop_id: str,
    slot_index: int,
    *,
    crop_source: Optional[Dict[str, Any]] = None,
    island_label: Optional[int] = None,
) -> Dict[str, str]:
    step_masked_path = _STEP_SEG_DIR / f"{set_num}_bag{int(bag)}_{crop_id}_slot{int(slot_index)}_masked.png"
    part_cutout_hits = sorted(
        _PART_CUTOUT_DIR.glob(f"{set_num}_bag{int(bag)}_{crop_id}_slot{int(slot_index)}_*_cutout.png")
    )
    part_cutout_path = part_cutout_hits[-1] if part_cutout_hits else None
    overlay_hits = sorted(
        _SLOT_WINDOW_OVERLAY_DIR.glob(
            f"{set_num}_bag{int(bag)}_{crop_id}_slot{int(slot_index)}_*_slot_window_overlay.png"
        )
    )
    slot_window_overlay_path = overlay_hits[-1] if overlay_hits else None

    island_cutout_path_str = ""
    island_slot_mask_path_str = ""
    crop_source = crop_source or {}
    raw_box = list(crop_source.get("crop_box") or [])
    crop_box = [int(raw_box[i] or 0) for i in range(4)] if len(raw_box) >= 4 else []
    crop_image_path = str(crop_source.get("crop_image_path") or "").strip()
    if island_label is not None and len(crop_box) >= 4 and crop_image_path:
        stem = find_full_mask_stem(str(set_num), int(bag), str(crop_id))
        if stem:
            island_path = ensure_island_slot_cutout_path(
                stem=stem,
                slot_index=int(slot_index),
                crop_image_path=crop_image_path,
                crop_box=crop_box,
                island_label=island_label,
            )
            if island_path is not None and island_path.is_file():
                island_cutout_path_str = str(island_path)
            mask_path = ensure_island_slot_mask_path(
                stem=stem,
                slot_index=int(slot_index),
                crop_image_path=crop_image_path,
                crop_box=crop_box,
                island_label=island_label,
            )
            if mask_path is not None and mask_path.is_file():
                island_slot_mask_path_str = str(mask_path)

    step_masked_resolved = str(step_masked_path) if step_masked_path.is_file() else ""
    if not step_masked_resolved and island_slot_mask_path_str:
        step_masked_resolved = island_slot_mask_path_str

    return {
        "step_masked_path": step_masked_resolved,
        "step_masked_url": _artifact_url(step_masked_resolved) if step_masked_resolved else "",
        "part_cutout_path": str(part_cutout_path) if part_cutout_path and part_cutout_path.is_file() else "",
        "part_cutout_url": _artifact_url(str(part_cutout_path)) if part_cutout_path and part_cutout_path.is_file() else "",
        "island_cutout_path": island_cutout_path_str,
        "island_cutout_url": _artifact_url(island_cutout_path_str) if island_cutout_path_str else "",
        "island_slot_mask_path": island_slot_mask_path_str,
        "island_slot_mask_url": _artifact_url(island_slot_mask_path_str) if island_slot_mask_path_str else "",
        "slot_window_overlay_path": (
            str(slot_window_overlay_path) if slot_window_overlay_path and slot_window_overlay_path.is_file() else ""
        ),
        "slot_window_overlay_url": (
            _artifact_url(str(slot_window_overlay_path))
            if slot_window_overlay_path and slot_window_overlay_path.is_file()
            else ""
        ),
    }


def _crop_has_reviewed_labels(saved_crop: Dict[str, Any]) -> bool:
    if not isinstance(saved_crop, dict):
        return False
    for part in list(saved_crop.get("parts") or []):
        if not isinstance(part, dict):
            continue
        if str(part.get("part_num") or "").strip():
            return True
    if list(saved_crop.get("unknown_slots") or []):
        return True
    if list(saved_crop.get("ignored_slots") or []):
        return True
    return False


def _build_saved_label_slot_sequence(saved_crop: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Preserve existing reviewed slot indices when qty OCR is absent."""
    slot_indices: set[int] = set()
    for part_index, part in enumerate(list(saved_crop.get("parts") or [])):
        if not isinstance(part, dict):
            continue
        if not str(part.get("part_num") or "").strip():
            continue
        explicit_slot = _coerce_int(part.get("selected_slot_index"))
        resolved_slot = explicit_slot if explicit_slot is not None else int(part_index)
        if resolved_slot is not None and int(resolved_slot) >= 0:
            slot_indices.add(int(resolved_slot))
    for raw_index in list(saved_crop.get("unknown_slots") or []):
        parsed = _coerce_int(raw_index)
        if parsed is not None and int(parsed) >= 0:
            slot_indices.add(int(parsed))
    for raw_index in list(saved_crop.get("ignored_slots") or []):
        parsed = _coerce_int(raw_index)
        if parsed is not None and int(parsed) >= 0:
            slot_indices.add(int(parsed))
    if not slot_indices:
        return []
    count = max(slot_indices) + 1
    return [
        {"qty": None, "qty_text": "", "slot_source": "saved_label"}
        for _ in range(count)
    ]


def _build_island_fallback_slot_sequence(
    set_num: str,
    bag: int,
    crop_id: str,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Build review slots from significant raw_master_mask CCs when qty OCR is absent."""
    stem = find_full_mask_stem(str(set_num), int(bag), str(crop_id))
    if not stem:
        return [], "no_full_crop_mask"
    raw_path = raw_master_mask_path(stem)
    if not raw_path.is_file():
        return [], "no_raw_master_mask"
    raw_islands = sort_islands_for_slots(master_islands_from_mask(str(raw_path)))
    if not raw_islands:
        return [], "no_mask_islands"
    islands = filter_significant_islands(raw_islands)
    if not islands:
        return [], "no_significant_islands"
    if len(islands) > MAX_AUTO_ISLAND_SLOTS:
        return [], f"too_many_islands:{len(islands)}>{MAX_AUTO_ISLAND_SLOTS}"
    return [
        {
            "qty": None,
            "qty_text": "",
            "slot_source": "island_fallback",
            "island_label": int(island.get("label") or (index + 1)),
        }
        for index, island in enumerate(islands)
    ], None


def _supplement_slot_sequence_with_islands(
    set_num: str,
    bag: int,
    crop_id: str,
    slot_sequence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Append island-backed slots when mask geometry exceeds primary slot count.

    Primary qty/saved-label slots are preserved in order; supplemental slots use
    island_fallback geometry for indices beyond the primary sequence length.
    """
    if not slot_sequence:
        return slot_sequence
    island_slot_sequence, _island_audit_reason = _build_island_fallback_slot_sequence(
        set_num, int(bag), str(crop_id)
    )
    if not island_slot_sequence:
        return slot_sequence
    primary_count = len(slot_sequence)
    island_count = len(island_slot_sequence)
    if island_count <= primary_count or island_count > MAX_AUTO_ISLAND_SLOTS:
        return slot_sequence
    supplemented = list(slot_sequence)
    for index in range(primary_count, island_count):
        supplemented.append(dict(island_slot_sequence[index]))
    return supplemented


def _slot_display_text(slot_index: int, slot: Dict[str, Any]) -> str:
    label = slot.get("saved_label") if isinstance(slot.get("saved_label"), dict) else None
    if label and str(label.get("part_num") or "").strip():
        qty = str(label.get("qty_text") or slot.get("qty_text") or "").strip()
        return f"Slot {slot_index + 1}: {label.get('part_num')} / {label.get('color_id')} {qty}".strip()
    if bool(slot.get("unknown")):
        return f"Slot {slot_index + 1}: UNKNOWN"
    if bool(slot.get("ignored")):
        return f"Slot {slot_index + 1}: IGNORED"
    if str(slot.get("slot_source") or "") == "island_fallback":
        island_label = slot.get("island_label")
        if island_label is not None:
            return f"Slot {slot_index + 1}: island {int(island_label)}"
        return f"Slot {slot_index + 1}: island"
    return f"Slot {slot_index + 1}: needs review"


def _upsert_crop_record(
    state: Dict[str, Any],
    set_num: str,
    bag: int,
    crop_id: str,
    crop_source: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    crops = state.setdefault("crops", {})
    crop_source = crop_source or {}
    existing = crops.get(crop_id)
    if not isinstance(existing, dict):
        existing = {}
        crops[crop_id] = existing
    existing.setdefault("page", int(crop_source.get("page", 0) or 0))
    existing.setdefault("step", int(crop_source.get("step", 0) or 0))
    existing.setdefault("qty", _coerce_int_list(crop_source.get("qty_numbers", crop_source.get("qty", []))))
    existing.setdefault("qty_text", _coerce_str_list(crop_source.get("qty_text", [])))
    existing.setdefault("status", "good")
    existing.setdefault("crop_box", list(crop_source.get("crop_box", []) or []))
    existing.setdefault("crop_box_format", str(crop_source.get("crop_box_format") or "xywh"))
    existing.setdefault("crop_image_path", str(crop_source.get("crop_image_path") or ""))
    existing.setdefault("review_status", "unreviewed")
    existing.setdefault("parts", [])
    return existing


def _crop_source_by_id(set_num: str, bag: int) -> Dict[str, Dict[str, Any]]:
    return {
        str(crop.get("crop_id") or ""): crop
        for crop in _load_crop_cache(set_num, int(bag))
        if str(crop.get("crop_id") or "")
    }


def _find_crop_source(set_num: str, bag: int, crop_id: str) -> Dict[str, Any]:
    return _crop_source_by_id(set_num, int(bag)).get(str(crop_id), {})


def build_review_model(set_num: str, bag: int) -> Dict[str, Any]:
    state = load_review_state(set_num, int(bag))
    cached_crops = _load_crop_cache(set_num, int(bag))
    saved_crops = dict(state.get("crops") or {})
    review_crops: List[Dict[str, Any]] = []
    assigned_qty_by_key: Dict[str, int] = {}

    for saved_crop in saved_crops.values():
        if not isinstance(saved_crop, dict):
            continue
        for part_data in list(saved_crop.get("parts") or []):
            if not isinstance(part_data, dict):
                continue
            part = _normalize_part_entry(part_data)
            if not part["part_num"]:
                continue
            key = f"{part['part_num']}::{int(part['color_id'] or 0)}"
            assigned_qty_by_key[key] = assigned_qty_by_key.get(key, 0) + int(part.get("qty") or 1)

    for crop in cached_crops:
        crop_id = str(crop.get("crop_id") or "").strip()
        if not crop_id:
            continue
        saved_crop = dict(saved_crops.get(crop_id) or {})
        status = str(saved_crop.get("status") or "needs_adjust").strip().lower()
        if status == "hidden":
            continue

        saved_qty_text = _coerce_str_list(saved_crop.get("qty_text", []) or saved_crop.get("crop_qty_text", []))
        crop_qty_text = (
            _coerce_str_list(crop.get("candidate_detected_qty_text", []))
            or _coerce_str_list(crop.get("detected_qty_text", []))
            or saved_qty_text
            or _coerce_str_list(crop.get("qty_text", []))
        )
        crop_qty = (
            _coerce_int_list(crop.get("candidate_detected_qty_numbers", []))
            if crop_qty_text
            else (
                _coerce_int_list(saved_crop.get("qty", []) or saved_crop.get("crop_qty", []))
                or _coerce_int_list(crop.get("qty_numbers", []))
            )
        )
        slot_sequence = _build_qty_sequence(crop_qty, crop_qty_text)
        crop_slot_source = "qty"
        slot_audit_required = False
        slot_audit_reason = ""
        if not slot_sequence:
            if _crop_has_reviewed_labels(saved_crop):
                slot_sequence = _build_saved_label_slot_sequence(saved_crop)
                crop_slot_source = "saved_label"
            else:
                island_slot_sequence, island_audit_reason = _build_island_fallback_slot_sequence(
                    set_num, int(bag), crop_id
                )
                if island_slot_sequence:
                    slot_sequence = island_slot_sequence
                    crop_slot_source = "island_fallback"
                elif island_audit_reason:
                    slot_audit_required = True
                    slot_audit_reason = str(island_audit_reason)
        if slot_sequence:
            slot_sequence = _supplement_slot_sequence_with_islands(
                set_num, int(bag), crop_id, slot_sequence
            )

        saved_by_slot: Dict[int, Dict[str, Any]] = {}
        for part_index, saved_part in enumerate(list(saved_crop.get("parts", []) or [])):
            if not isinstance(saved_part, dict):
                continue
            normalized_part = _normalize_part_entry(saved_part)
            if not normalized_part["part_num"]:
                continue
            explicit_slot = _coerce_int(normalized_part.get("selected_slot_index"))
            resolved_slot = explicit_slot if explicit_slot is not None else part_index
            if resolved_slot is None or int(resolved_slot) < 0:
                continue
            saved_by_slot[int(resolved_slot)] = normalized_part

        unknown_slots = sorted(
            {
                int(item)
                for item in list(saved_crop.get("unknown_slots") or [])
                if _coerce_int(item) is not None and int(item) >= 0
            }
        )
        ignored_slots = sorted(
            {
                int(item)
                for item in list(saved_crop.get("ignored_slots") or [])
                if _coerce_int(item) is not None and int(item) >= 0
            }
        )

        slot_details: List[Dict[str, Any]] = []
        mask_stem = find_full_mask_stem(str(set_num), int(bag), crop_id)
        significant_islands = (
            significant_islands_for_stem(mask_stem) if mask_stem else []
        )
        for idx, slot in enumerate(slot_sequence):
            display_island_label = order_island_label(significant_islands, int(idx))
            slot_detail = {
                "slot_index": int(idx),
                "qty": slot.get("qty"),
                "qty_text": str(slot.get("qty_text") or slot.get("qty") or ""),
                "slot_source": str(slot.get("slot_source") or crop_slot_source),
                "island_label": display_island_label,
                "debug_island_label": display_island_label,
                "saved_label": saved_by_slot.get(int(idx)),
                "unknown": int(idx) in set(unknown_slots),
                "ignored": int(idx) in set(ignored_slots),
                **_slot_assets(
                    set_num,
                    int(bag),
                    crop_id,
                    int(idx),
                    crop_source=crop,
                    island_label=display_island_label,
                ),
            }
            slot_detail["display_text"] = _slot_display_text(int(idx), slot_detail)
            slot_details.append(slot_detail)

        review_crops.append(
            {
                "crop_id": crop_id,
                "page": int(crop.get("page", saved_crop.get("page", 0)) or 0),
                "step": int(crop.get("step", saved_crop.get("step", 0)) or 0),
                "crop_qty": list(crop_qty),
                "crop_qty_text": list(crop_qty_text),
                "crop_box": list(crop.get("crop_box", saved_crop.get("crop_box", [])) or []),
                "crop_box_format": str(crop.get("crop_box_format") or saved_crop.get("crop_box_format") or "xywh"),
                "crop_image_path": str(crop.get("crop_image_path") or saved_crop.get("crop_image_path") or ""),
                "original_data_uri": str(crop.get("data_uri") or ""),
                "qty_label": ", ".join(crop_qty_text) if crop_qty_text else "none",
                "slot_source": crop_slot_source,
                "slot_sequence": slot_sequence,
                "slot_audit_required": bool(slot_audit_required),
                "slot_audit_reason": str(slot_audit_reason or ""),
                "filled_slots": len([slot for slot in slot_details if slot.get("saved_label")]),
                "slots": slot_details,
                "unknown_slots": unknown_slots,
                "ignored_slots": ignored_slots,
            }
        )

    total = sum(len(crop.get("slots") or []) for crop in review_crops)
    confirmed = sum(1 for crop in review_crops for slot in crop.get("slots", []) if slot.get("saved_label"))
    unknown = sum(1 for crop in review_crops for slot in crop.get("slots", []) if slot.get("unknown"))
    ignored = sum(1 for crop in review_crops for slot in crop.get("slots", []) if slot.get("ignored"))
    complete = confirmed + unknown + ignored
    progress = {
        "total_slots": total,
        "confirmed_slots": confirmed,
        "unknown_slots": unknown,
        "ignored_slots": ignored,
        "needs_review_slots": max(0, total - complete),
        "percent_complete": int(round((complete / total) * 100)) if total else 0,
    }

    next_unreviewed = None
    for crop in review_crops:
        for slot in crop.get("slots", []):
            if not slot.get("saved_label") and not slot.get("unknown") and not slot.get("ignored"):
                next_unreviewed = {"crop_id": crop.get("crop_id"), "slot_index": slot.get("slot_index")}
                break
        if next_unreviewed:
            break
    progress["next_unreviewed"] = next_unreviewed

    return {
        "set_num": str(set_num),
        "bag": int(bag),
        "state": state,
        "crops": review_crops,
        "progress": progress,
        "assigned_qty_by_key": assigned_qty_by_key,
    }


def _slot_needs_review(slot: Dict[str, Any]) -> bool:
    return not slot.get("saved_label") and not slot.get("unknown") and not slot.get("ignored")


def find_next_unreviewed_slot(
    set_num: str,
    bag: int,
    *,
    from_crop_id: str = "",
    from_slot_index: Optional[int] = None,
    include_current: bool = False,
) -> Dict[str, Any]:
    """Return the next slot needing review using fresh label state from disk."""
    model = build_review_model(str(set_num), int(bag))
    flat: List[Dict[str, Any]] = []
    for crop in list(model.get("crops") or []):
        crop_id = str(crop.get("crop_id") or "")
        for slot in list(crop.get("slots") or []):
            flat.append(
                {
                    "crop_id": crop_id,
                    "slot_index": int(slot.get("slot_index", 0) or 0),
                    "slot": slot,
                }
            )

    if not flat:
        return {"found": False, "crop_id": None, "slot_index": None}

    crop_id_text = str(from_crop_id or "").strip()
    if crop_id_text:
        crop_items = [
            item for item in flat if str(item.get("crop_id") or "") == crop_id_text
        ]
        crop_start = -1
        if from_slot_index is not None:
            for index, item in enumerate(crop_items):
                if int(item.get("slot_index", -1)) == int(from_slot_index):
                    crop_start = index
                    break
        if include_current and crop_start >= 0:
            current = crop_items[crop_start]
            if _slot_needs_review(current.get("slot") or {}):
                return {
                    "found": True,
                    "crop_id": current.get("crop_id"),
                    "slot_index": current.get("slot_index"),
                }
        for step in range(1, len(crop_items) + 1):
            index = (
                (crop_start + step) % len(crop_items)
                if crop_start >= 0
                else (step - 1) % len(crop_items)
            )
            item = crop_items[index]
            if _slot_needs_review(item.get("slot") or {}):
                return {
                    "found": True,
                    "crop_id": item.get("crop_id"),
                    "slot_index": item.get("slot_index"),
                }

    start = -1
    if from_crop_id or from_slot_index is not None:
        for index, item in enumerate(flat):
            if str(item.get("crop_id") or "") == str(from_crop_id or "") and (
                from_slot_index is None or int(item.get("slot_index", -1)) == int(from_slot_index)
            ):
                start = index
                break

    if include_current and start >= 0:
        current = flat[start]
        if _slot_needs_review(current.get("slot") or {}):
            return {
                "found": True,
                "crop_id": current.get("crop_id"),
                "slot_index": current.get("slot_index"),
            }

    for step in range(1, len(flat) + 1):
        index = (start + step) % len(flat) if start >= 0 else (step - 1) % len(flat)
        item = flat[index]
        if _slot_needs_review(item.get("slot") or {}):
            return {
                "found": True,
                "crop_id": item.get("crop_id"),
                "slot_index": item.get("slot_index"),
            }

    return {"found": False, "crop_id": None, "slot_index": None}


def mark_slot_unknown(set_num: str, bag: int, crop_id: str, slot_index: int) -> Dict[str, Any]:
    state = load_review_state(set_num, int(bag))
    crop_record = _upsert_crop_record(state, set_num, int(bag), crop_id, _find_crop_source(set_num, int(bag), crop_id))
    unknown_slots = {
        int(item)
        for item in list(crop_record.get("unknown_slots") or [])
        if _coerce_int(item) is not None
    }
    unknown_slots.add(int(slot_index))
    ignored_slots = {
        int(item)
        for item in list(crop_record.get("ignored_slots") or [])
        if _coerce_int(item) is not None and int(item) != int(slot_index)
    }
    crop_record["unknown_slots"] = sorted(unknown_slots)
    crop_record["ignored_slots"] = sorted(ignored_slots)
    crop_record["annotated_at"] = _iso_now()
    save_review_state(set_num, int(bag), state)
    return crop_record


def mark_slot_ignored(set_num: str, bag: int, crop_id: str, slot_index: int) -> Dict[str, Any]:
    state = load_review_state(set_num, int(bag))
    crop_record = _upsert_crop_record(state, set_num, int(bag), crop_id, _find_crop_source(set_num, int(bag), crop_id))
    ignored_slots = {
        int(item)
        for item in list(crop_record.get("ignored_slots") or [])
        if _coerce_int(item) is not None
    }
    ignored_slots.add(int(slot_index))
    unknown_slots = {
        int(item)
        for item in list(crop_record.get("unknown_slots") or [])
        if _coerce_int(item) is not None and int(item) != int(slot_index)
    }
    crop_record["ignored_slots"] = sorted(ignored_slots)
    crop_record["unknown_slots"] = sorted(unknown_slots)
    crop_record["annotated_at"] = _iso_now()
    save_review_state(set_num, int(bag), state)
    return crop_record


def save_slot_label(
    set_num: str,
    bag: int,
    crop_id: str,
    slot_index: int,
    part_num: str,
    color_id: int,
    qty: Optional[int] = None,
) -> Dict[str, Any]:
    state = load_review_state(set_num, int(bag))
    crop_source = _find_crop_source(set_num, int(bag), crop_id)
    crop_record = _upsert_crop_record(state, set_num, int(bag), crop_id, crop_source)
    qty_sequence = _build_qty_sequence(crop_record.get("qty", []), crop_record.get("qty_text", []))
    slot = qty_sequence[int(slot_index)] if 0 <= int(slot_index) < len(qty_sequence) else {}
    resolved_qty = qty if qty is not None else _coerce_int(slot.get("qty"))
    part_entry = {
        "part_num": str(part_num or "").strip(),
        "color_id": int(color_id or 0),
        "color_name": "",
        "element_id": "",
        "qty": resolved_qty,
        "qty_text": str(slot.get("qty_text") or (f"{resolved_qty}x" if resolved_qty is not None else "") or "").strip() or None,
        "selected_slot_index": int(slot_index),
        "part_bbox": None,
        "confidence": None,
        "ai_snap_input_path": None,
    }
    parts = [part for part in list(crop_record.get("parts") or []) if isinstance(part, dict)]
    replaced = False
    for index, existing_part in enumerate(parts):
        if _coerce_int(existing_part.get("selected_slot_index")) == int(slot_index):
            parts[index] = part_entry
            replaced = True
            break
    if not replaced:
        parts.append(part_entry)
    crop_record["parts"] = parts
    crop_record["status"] = "good"
    crop_record["review_status"] = "reviewed"
    crop_record["unknown_slots"] = sorted(
        {
            int(item)
            for item in list(crop_record.get("unknown_slots") or [])
            if _coerce_int(item) is not None and int(item) != int(slot_index)
        }
    )
    crop_record["ignored_slots"] = sorted(
        {
            int(item)
            for item in list(crop_record.get("ignored_slots") or [])
            if _coerce_int(item) is not None and int(item) != int(slot_index)
        }
    )
    crop_record["annotated_at"] = _iso_now()
    save_review_state(set_num, int(bag), state)
    return crop_record
