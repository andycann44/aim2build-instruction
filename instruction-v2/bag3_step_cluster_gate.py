"""Bag 3 page-review gate: size-cluster filter for GLOBAL_STEP candidates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from bag4_misread_step_detection import (
    MIN_DOMINANT_STEP,
    SUBSTEP_VALUES,
    _infer_sequence_valid_steps,
    _load_bag_page_range,
    _load_crop_cache,
    _load_ocr_detections,
    dominant_step_number,
)
from bag_step_ocr_noise_detection import OCR_NOISE_CLASSIFICATION, OCR_NOISE_REASON

BAG = 3
GLOBAL_STEP_CLASSIFICATION = "GLOBAL_STEP"
UNCERTAIN_GATE_STATUS = "NEEDS_HUMAN_CONFIRMATION"

MIN_CLUSTER_WIDTH = 38
MIN_CLUSTER_HEIGHT = 25
MAX_CLUSTER_HEIGHT = 47
REQUIRED_CLUSTER_COMPONENTS = 2


def _best_entry_for_key(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    def rank(entry: Dict[str, Any]) -> Tuple[int, int, int]:
        primary = 1 if entry.get("is_primary_step") else 0
        box = entry.get("step_box") or []
        width = int(box[2] or 0) if len(box) >= 4 else 0
        components = int(entry.get("components") or 0)
        return (primary, components, width)

    return max(entries, key=rank)


def _entry_cluster_status(entry: Dict[str, Any], step: int) -> Tuple[str, List[str]]:
    box = entry.get("step_box") or []
    if len(box) < 4:
        return "uncertain", ["missing_box"]

    width = int(box[2] or 0)
    height = int(box[3] or 0)
    if entry.get("components") is None:
        return "uncertain", ["missing_component_count"]

    components = int(entry.get("components") or 0)
    reasons: List[str] = []

    if components == 1:
        reasons.append("single_component")
    if width < MIN_CLUSTER_WIDTH:
        reasons.append("width_below_cluster")
    if height < MIN_CLUSTER_HEIGHT or height > MAX_CLUSTER_HEIGHT:
        reasons.append("height_out_of_cluster_band")
    if step >= MIN_DOMINANT_STEP and components == 1:
        reasons.append("single_component_multi_digit_step")
    if step >= MIN_DOMINANT_STEP and len(str(step)) == 2 and components != REQUIRED_CLUSTER_COMPONENTS:
        reasons.append("component_digit_mismatch")

    if reasons:
        return "reject", reasons

    if (
        components == REQUIRED_CLUSTER_COMPONENTS
        and width >= MIN_CLUSTER_WIDTH
        and MIN_CLUSTER_HEIGHT <= height <= MAX_CLUSTER_HEIGHT
    ):
        return "pass", []

    return "uncertain", ["partial_cluster_match"]


def _bag_sequence_context(crops: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = sorted({int(crop.get("step") or 0) for crop in crops if int(crop.get("step") or 0) > 0})
    crop_steps = set(steps)
    return {
        "min_step": steps[0] if steps else None,
        "max_step": steps[-1] if steps else None,
        "crop_steps": crop_steps,
    }


def _sequence_plausible(
    page: int,
    step: int,
    ctx: Dict[str, Any],
    crops: List[Dict[str, Any]],
    page_start: int,
    page_end: int,
    primary_steps_on_page: List[Dict[str, Any]],
) -> Tuple[Optional[bool], str]:
    step = int(step)
    crop_steps = ctx["crop_steps"]
    if step in crop_steps:
        return True, "in_crop_cache"

    sorted_steps = sorted(crop_steps)
    for left, right in zip(sorted_steps, sorted_steps[1:]):
        if left < step < right:
            return True, "sequence_gap_candidate"

    inferred = _infer_sequence_valid_steps(page, page_start, page_end, crops, BAG)
    if step in inferred:
        return True, "sequence_inferred_for_page"

    for entry in primary_steps_on_page:
        if int(entry.get("step") or 0) == step and entry.get("is_primary_step"):
            return True, "primary_step_map_anchor"

    seq_min = ctx.get("min_step")
    seq_max = ctx.get("max_step")
    if seq_min is not None and seq_max is not None and seq_min <= step <= seq_max:
        return True, "within_bag_step_span"

    if seq_max is not None and seq_max >= MIN_DOMINANT_STEP and step in SUBSTEP_VALUES:
        return False, "single_digit_late_sequence"

    if seq_max is not None and step > seq_max + 3:
        return False, "above_bag_sequence"
    if seq_min is not None and step < seq_min - 3:
        return False, "below_bag_sequence"

    return None, "sequence_uncertain"


def _collect_detection_keys(ocr_by_page: Dict[int, List[Dict[str, Any]]], page_start: int, page_end: int) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for page in range(page_start, page_end + 1):
        for entry in ocr_by_page.get(page, []) or []:
            step = int(entry.get("step") or 0)
            if not step:
                continue
            grouped.setdefault((page, step), []).append(entry)
    return grouped


def audit_bag3_step_cluster_gate(
    *,
    substep_keys: Optional[Set[Tuple[int, int]]] = None,
    misread_keys: Optional[Set[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    page_start, page_end = _load_bag_page_range(BAG)
    crops = _load_crop_cache(BAG, page_start, page_end)
    ocr_by_page = _load_ocr_detections(BAG)
    ctx = _bag_sequence_context(crops)
    grouped = _collect_detection_keys(ocr_by_page, page_start, page_end)

    substep_keys = set(substep_keys or [])
    misread_keys = set(misread_keys or [])

    rows: List[Dict[str, Any]] = []
    global_step_keys: Set[Tuple[int, int]] = set()
    ocr_noise_keys: Set[Tuple[int, int]] = set()
    uncertain_keys: Set[Tuple[int, int]] = set()

    for (page, step), entries in sorted(grouped.items()):
        best = _best_entry_for_key(entries)
        cluster_status, cluster_reasons = _entry_cluster_status(best, step)
        primary_steps = [entry for entry in ocr_by_page.get(page, []) or [] if entry.get("is_primary_step")]
        page_crops = [crop for crop in crops if int(crop.get("page") or 0) == page]
        seq_ok, seq_reason = _sequence_plausible(page, step, ctx, crops, page_start, page_end, primary_steps)
        dominant = dominant_step_number(page, page_crops, primary_steps)

        if (page, step) in misread_keys:
            classification = "MISREAD_STEP_NUMBER"
            gate_status = "suppressed"
            suppress = True
        elif (page, step) in substep_keys and cluster_status != "reject" and seq_ok is not False:
            classification = "SUBSTEP_NUMBER"
            gate_status = "suppressed"
            suppress = True
        elif seq_ok is False:
            classification = OCR_NOISE_CLASSIFICATION
            gate_status = "suppressed"
            suppress = True
            cluster_reasons = cluster_reasons + [seq_reason]
        elif cluster_status == "reject":
            classification = OCR_NOISE_CLASSIFICATION
            gate_status = "suppressed"
            suppress = True
        elif cluster_status == "pass" and seq_ok is True:
            classification = GLOBAL_STEP_CLASSIFICATION
            gate_status = "accepted"
            suppress = False
        elif cluster_status == "pass" and seq_ok is None:
            classification = GLOBAL_STEP_CLASSIFICATION
            gate_status = UNCERTAIN_GATE_STATUS
            suppress = False
        elif cluster_status == "uncertain" or seq_ok is None:
            classification = GLOBAL_STEP_CLASSIFICATION
            gate_status = UNCERTAIN_GATE_STATUS
            suppress = False
        else:
            classification = OCR_NOISE_CLASSIFICATION
            gate_status = "suppressed"
            suppress = True

        row = {
            "page": page,
            "ocr_value": step,
            "classification": classification,
            "gate_status": gate_status,
            "match_reason": (
                OCR_NOISE_REASON
                if classification == OCR_NOISE_CLASSIFICATION
                else (seq_reason if seq_ok is not None else "cluster_or_sequence_uncertain")
            ),
            "cluster_status": cluster_status,
            "cluster_reasons": cluster_reasons,
            "sequence_plausible": seq_ok,
            "sequence_reason": seq_reason,
            "dominant_step_on_page": dominant,
            "source": best.get("source"),
            "raw_text": best.get("raw_text"),
            "step_box_xywh": list(best.get("step_box") or []),
            "components": best.get("components"),
            "width": int((best.get("step_box") or [0, 0, 0, 0])[2] or 0),
            "height": int((best.get("step_box") or [0, 0, 0, 0])[3] or 0),
            "suppressed": suppress,
        }
        rows.append(row)

        if classification == GLOBAL_STEP_CLASSIFICATION and gate_status == "accepted":
            global_step_keys.add((page, step))
        elif suppress and classification == OCR_NOISE_CLASSIFICATION:
            ocr_noise_keys.add((page, step))
        elif gate_status == UNCERTAIN_GATE_STATUS:
            uncertain_keys.add((page, step))

    return {
        "bag": BAG,
        "page_start": page_start,
        "page_end": page_end,
        "gate": "bag3_step_cluster",
        "cluster_rules": {
            "required_components": REQUIRED_CLUSTER_COMPONENTS,
            "min_width_px": MIN_CLUSTER_WIDTH,
            "min_height_px": MIN_CLUSTER_HEIGHT,
            "max_height_px": MAX_CLUSTER_HEIGHT,
        },
        "classification": OCR_NOISE_CLASSIFICATION,
        "match_reason": OCR_NOISE_REASON,
        "row_count": len(rows),
        "global_step_count": len(global_step_keys),
        "ocr_noise_count": len(ocr_noise_keys),
        "uncertain_count": len(uncertain_keys),
        "ocr_noise_keys": [f"p{page:03d}_s{step}" for page, step in sorted(ocr_noise_keys)],
        "global_step_keys": [f"p{page:03d}_s{step}" for page, step in sorted(global_step_keys)],
        "uncertain_keys": [f"p{page:03d}_s{step}" for page, step in sorted(uncertain_keys)],
        "rows": rows,
        "ocr_noise_keys_set": ocr_noise_keys,
        "global_step_keys_set": global_step_keys,
        "uncertain_keys_set": uncertain_keys,
    }


def cluster_gate_step_keys(
    *,
    substep_keys: Optional[Set[Tuple[int, int]]] = None,
    misread_keys: Optional[Set[Tuple[int, int]]] = None,
) -> Set[Tuple[int, int]]:
    payload = audit_bag3_step_cluster_gate(substep_keys=substep_keys, misread_keys=misread_keys)
    return set(payload["ocr_noise_keys_set"])
