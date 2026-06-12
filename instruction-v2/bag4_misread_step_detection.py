"""Detect OCR step numbers that are misreads or substeps on build pages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from paths import INDEXES_DIR, ROOT_DIR


PROJECT_ROOT = ROOT_DIR.parent
CROP_CACHE_DIR = PROJECT_ROOT / "debug" / "crop_cache"
STEP_MAP_PATH = INDEXES_DIR / "05_step_map.json"
STEP_PROBE_PATH = ROOT_DIR / "debug" / "sequence_completeness" / "05_step_probe.json"
BAG_MAP_PATH = INDEXES_DIR / "04_bag_map.json"

MISREAD_CLASSIFICATION = "MISREAD_STEP_NUMBER"
SUBSTEP_CLASSIFICATION = "SUBSTEP_NUMBER"
SUBSTEP_VALUES = set(range(1, 10))
MIN_DOMINANT_STEP = 10
OCR_DIGIT_SWAPS = {"7": "2", "2": "7", "8": "3", "3": "8", "5": "6", "6": "5", "1": "7", "0": "8"}
PROBE_FALSE_DIGITS = {1, 4, 7}
KNOWN_CROP_ID_CORRECTIONS = {
    "p77_s173_c1": 123,
    "p70_s11_c1": 111,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _box_xywh(box: Any) -> List[int]:
    if isinstance(box, dict):
        return [int(box.get("x", 0) or 0), int(box.get("y", 0) or 0), int(box.get("w", 0) or 0), int(box.get("h", 0) or 0)]
    if isinstance(box, list) and len(box) >= 4:
        return [int(box[i] or 0) for i in range(4)]
    return []


def _load_bag_page_range(bag: int) -> Tuple[int, int]:
    payload = _load_json(BAG_MAP_PATH)
    for entry in payload.get("bags", []) or []:
        if int(entry.get("bag") or 0) == bag:
            return int(entry.get("start_page") or 1), int(entry.get("end_page") or 10_000)
    return 58, 80


def _ocr_variants(step: int) -> Set[int]:
    text = str(int(step))
    variants = {int(text)}
    for idx, ch in enumerate(text):
        swap = OCR_DIGIT_SWAPS.get(ch)
        if not swap:
            continue
        variants.add(int(text[:idx] + swap + text[idx + 1 :]))
    return variants


def _load_crop_cache(bag: int, page_start: int, page_end: int) -> List[Dict[str, Any]]:
    cache_path = CROP_CACHE_DIR / f"70618_bag{int(bag)}.json"
    if not cache_path.exists():
        return []
    crops = []
    for crop in _load_json(cache_path):
        if not isinstance(crop, dict):
            continue
        page = int(crop.get("page") or 0)
        step = int(crop.get("step") or 0)
        if page_start <= page <= page_end and step > 0:
            crops.append(dict(crop))
    return crops


def _load_ocr_detections(bag: int) -> Dict[int, List[Dict[str, Any]]]:
    by_page: Dict[int, List[Dict[str, Any]]] = {}

    def add(page: int, step: int, step_box: Any, source: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not page or not step:
            return
        payload = {
            "page": page,
            "step": int(step),
            "step_box": _box_xywh(step_box),
            "source": source,
            **(extra or {}),
        }
        by_page.setdefault(page, []).append(payload)

    for entry in _load_json(STEP_MAP_PATH).get("steps", []) or []:
        if int(entry.get("bag") or 0) != bag:
            continue
        page = int(entry.get("page") or 0)
        step = int(entry.get("step_number") or 0)
        signals = entry.get("signals") or {}
        add(
            page,
            step,
            entry.get("step_box"),
            "step_map",
            {
                "raw_text": signals.get("step_number_raw_text"),
                "components": signals.get("components"),
                "is_primary_step": True,
            },
        )
        for alt in signals.get("step_number_rejected_reads", []) or []:
            if alt.get("value") is None:
                continue
            add(
                page,
                int(alt["value"]),
                entry.get("step_box"),
                "step_map_rejected_read",
                {
                    "raw_text": alt.get("raw_text"),
                    "parent_step": step,
                    "reason": alt.get("reason"),
                },
            )

    for page_entry in _load_json(STEP_MAP_PATH).get("pages", []) or []:
        if int(page_entry.get("bag") or 0) != bag:
            continue
        page = int(page_entry.get("page") or 0)
        for candidate in page_entry.get("full_page_audit_candidates", []) or []:
            if candidate.get("rejection_reason"):
                continue
            step = int(candidate.get("step_number") or 0)
            signals = candidate.get("signals") or {}
            add(
                page,
                step,
                candidate.get("step_box"),
                "full_page_audit",
                {
                    "raw_text": signals.get("step_number_raw_text"),
                    "components": signals.get("components"),
                    "is_primary_step": False,
                },
            )
            for alt in signals.get("step_number_rejected_reads", []) or []:
                if alt.get("value") is None:
                    continue
                add(
                    page,
                    int(alt["value"]),
                    candidate.get("step_box"),
                    "full_page_audit_rejected_read",
                    {
                        "raw_text": alt.get("raw_text"),
                        "parent_step": step,
                        "reason": alt.get("reason"),
                    },
                )

    if STEP_PROBE_PATH.exists():
        for entry in _load_json(STEP_PROBE_PATH).get("steps", []) or []:
            if int(entry.get("bag") or 0) != bag:
                continue
            add(
                int(entry.get("page") or 0),
                int(entry.get("step_number") or 0),
                entry.get("step_box"),
                "step_probe",
                {"is_primary_step": False},
            )

    return by_page


def _valid_steps_for_page(
    page: int,
    crops: List[Dict[str, Any]],
    primary_steps: List[Dict[str, Any]],
    inferred_steps: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    valid: Dict[int, Dict[str, Any]] = {}
    for crop in crops:
        if int(crop.get("page") or 0) != page:
            continue
        step = int(crop.get("step") or 0)
        if step > 0:
            valid[step] = {"step": step, "source": "crop_cache", "crop_id": crop.get("crop_id")}

    for entry in primary_steps:
        step = int(entry.get("step") or 0)
        if step <= 0 or step in valid:
            continue
        components = int(entry.get("components") or 0)
        if components >= 2:
            valid[step] = {"step": step, "source": "step_map_primary", "raw_text": entry.get("raw_text")}
        elif entry.get("source") == "step_map" and step not in PROBE_FALSE_DIGITS:
            valid[step] = {"step": step, "source": "step_map_primary", "raw_text": entry.get("raw_text")}

    for step in inferred_steps or []:
        if step not in valid:
            valid[step] = {"step": step, "source": "sequence_inferred"}

    if (inferred_steps or any(int(item["step"]) >= 100 for item in valid.values())) and valid:
        for step in list(valid):
            if step in PROBE_FALSE_DIGITS and valid[step].get("source") != "crop_cache":
                del valid[step]

    return [valid[step] for step in sorted(valid)]


def _next_step_anchor_after_page(page: int, bag: int) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int]] = None
    for entry in _load_json(STEP_MAP_PATH).get("steps", []) or []:
        if int(entry.get("bag") or 0) != bag:
            continue
        p = int(entry.get("page") or 0)
        s = int(entry.get("step_number") or 0)
        if p <= page or not s:
            continue
        if best is None or (p, s) < best:
            best = (p, s)
    if best:
        return best
    for entry in _load_json(STEP_MAP_PATH).get("steps", []) or []:
        p = int(entry.get("page") or 0)
        s = int(entry.get("step_number") or 0)
        if p <= page or not s:
            continue
        if best is None or (p, s) < best:
            best = (p, s)
    return best


def _infer_sequence_valid_steps(
    page: int,
    page_start: int,
    page_end: int,
    crops: List[Dict[str, Any]],
    bag: int,
) -> List[int]:
    by_page: Dict[int, List[int]] = {}
    for crop in crops:
        p = int(crop.get("page") or 0)
        s = int(crop.get("step") or 0)
        if p and s:
            by_page.setdefault(p, []).append(s)

    if page in by_page:
        return []

    pages_with_steps = sorted(by_page)
    prev_page = max((p for p in pages_with_steps if p < page), default=None)
    if prev_page is None:
        return []

    prev_step = max(by_page[prev_page])
    next_page = min((p for p in pages_with_steps if p > page), default=None)
    next_step: Optional[int] = None
    if next_page is not None:
        next_step = min(by_page[next_page])
    else:
        external = _next_step_anchor_after_page(page, bag)
        if external:
            next_page, next_step = external

    if next_page is None or next_step is None:
        return []

    inferred: List[int] = []
    for p in range(prev_page + 1, next_page):
        if p != page:
            continue
        offset = p - prev_page
        candidate = prev_step + offset
        if prev_step < candidate < next_step:
            inferred.append(candidate)
    return inferred


def _box_distance(a: List[int], b: List[int]) -> float:
    if len(a) < 4 or len(b) < 4:
        return float("inf")
    ax, ay = a[0] + a[2] // 2, a[1] + a[3] // 2
    bx, by = b[0] + b[2] // 2, b[1] + b[3] // 2
    return abs(ax - bx) + abs(ay - by) * 1.5


def _main_build_steps_on_page(
    page: int,
    crops: List[Dict[str, Any]],
    primary_steps: List[Dict[str, Any]],
) -> Set[int]:
    main_steps: Set[int] = set()
    for crop in crops:
        if int(crop.get("page") or 0) != page:
            continue
        step = int(crop.get("step") or 0)
        if step >= MIN_DOMINANT_STEP:
            main_steps.add(step)
    for entry in primary_steps:
        step = int(entry.get("step") or 0)
        components = int(entry.get("components") or 0)
        if step >= MIN_DOMINANT_STEP or components >= 2:
            main_steps.add(step)
    return main_steps


def dominant_step_number(
    page: int,
    crops: List[Dict[str, Any]],
    primary_steps: Optional[List[Dict[str, Any]]] = None,
) -> Optional[int]:
    """Return the dominant build-step number N on a page, if any."""
    main_steps = _main_build_steps_on_page(page, crops, primary_steps or [])
    if not main_steps:
        return None
    return max(main_steps)


def _match_substep(
    ocr_step: int,
    dominant_step: Optional[int],
) -> Optional[str]:
    if dominant_step is None or dominant_step < MIN_DOMINANT_STEP:
        return None
    if ocr_step not in SUBSTEP_VALUES:
        return None
    if ocr_step == dominant_step:
        return None
    return "substep_of_dominant_page_step"


def _match_misread(
    ocr_step: int,
    ocr_entry: Dict[str, Any],
    valid_steps: List[Dict[str, Any]],
    inferred_steps: List[int],
) -> Optional[Tuple[int, str]]:
    valid_values = {int(item["step"]) for item in valid_steps}
    valid_values.update(inferred_steps)
    if ocr_step in valid_values:
        return None

    all_valid = sorted(valid_values)
    if not all_valid:
        return None

    ocr_text = str(ocr_step)
    candidates: List[Tuple[int, str, float]] = []

    for valid in all_valid:
        valid_text = str(valid)
        parent = int(ocr_entry.get("parent_step") or 0)
        if ocr_entry.get("source", "").endswith("rejected_read") and parent == valid:
            candidates.append((valid, "rejected_read_of_valid_anchor", 0.99))
        if valid_text.startswith(ocr_text) and len(ocr_text) < len(valid_text):
            candidates.append((valid, "prefix_fragment_of_valid_step", 0.95))
        if abs(valid - ocr_step) <= 100:
            if ocr_step in _ocr_variants(valid) or valid in _ocr_variants(ocr_step):
                candidates.append((valid, "digit_swap_variant_of_valid_step", 0.92))

    components = int(ocr_entry.get("components") or 0)
    is_probe_like = (
        ocr_step in PROBE_FALSE_DIGITS
        and (components <= 1 or ocr_entry.get("source") in {"step_probe", "full_page_audit"})
        and not ocr_entry.get("is_primary_step")
    )
    if is_probe_like and all(v >= 20 for v in all_valid):
        if len(all_valid) == 1:
            candidates.append((all_valid[0], "spurious_probe_digit_single_valid_step", 0.88))
        elif ocr_text and any(str(v).startswith(ocr_text) for v in all_valid):
            for valid in all_valid:
                if str(valid).startswith(ocr_text):
                    candidates.append((valid, "prefix_fragment_of_valid_step", 0.90))
        else:
            candidates.append((min(all_valid), "spurious_probe_digit_on_multi_step_page", 0.85))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[2], abs(item[0] - ocr_step)))
    return candidates[0][0], candidates[0][1]


def audit_bag_substep_numbers(bag: int = 4, ocr_noise_keys: Optional[Set[Tuple[int, int]]] = None) -> Dict[str, Any]:
    page_start, page_end = _load_bag_page_range(bag)
    crops = _load_crop_cache(bag, page_start, page_end)
    ocr_by_page = _load_ocr_detections(bag)
    if ocr_noise_keys is None:
        from bag_step_ocr_noise_detection import ocr_noise_step_keys

        ocr_noise_keys = ocr_noise_step_keys(bag, ocr_by_page=ocr_by_page)

    rows: List[Dict[str, Any]] = []
    substep_keys: Set[Tuple[int, int]] = set()
    seen_report: Set[Tuple[int, int, int]] = set()

    for page in range(page_start, page_end + 1):
        page_ocr = ocr_by_page.get(page, [])
        primary_steps = [entry for entry in page_ocr if entry.get("is_primary_step")]
        page_crops = [crop for crop in crops if int(crop.get("page") or 0) == page]
        dominant = dominant_step_number(page, page_crops, primary_steps)
        if dominant is None:
            continue

        for entry in page_ocr:
            ocr_step = int(entry["step"])
            if (page, ocr_step) in ocr_noise_keys:
                continue
            reason = _match_substep(ocr_step, dominant)
            if not reason:
                continue
            substep_keys.add((page, ocr_step))
            report_key = (page, ocr_step, dominant)
            if report_key in seen_report:
                continue
            seen_report.add(report_key)
            rows.append(
                {
                    "page": page,
                    "ocr_value": ocr_step,
                    "parent_step": dominant,
                    "classification": SUBSTEP_CLASSIFICATION,
                    "match_reason": reason,
                    "source": entry.get("source"),
                    "raw_text": entry.get("raw_text"),
                    "dominant_step_on_page": dominant,
                }
            )

    rows.sort(key=lambda row: (int(row["page"]), int(row["ocr_value"]), int(row["parent_step"])))
    return {
        "bag": bag,
        "page_start": page_start,
        "page_end": page_end,
        "classification": SUBSTEP_CLASSIFICATION,
        "row_count": len(rows),
        "substep_keys": [f"p{page:03d}_s{step}" for page, step in sorted(substep_keys)],
        "rows": rows,
    }


def audit_bag_misread_steps(bag: int = 4, ocr_noise_keys: Optional[Set[Tuple[int, int]]] = None) -> Dict[str, Any]:
    page_start, page_end = _load_bag_page_range(bag)
    crops = _load_crop_cache(bag, page_start, page_end)
    ocr_by_page = _load_ocr_detections(bag)
    if ocr_noise_keys is None:
        from bag_step_ocr_noise_detection import ocr_noise_step_keys

        ocr_noise_keys = ocr_noise_step_keys(bag, ocr_by_page=ocr_by_page)
    substep_keys = substep_step_keys(bag, crops=crops, ocr_by_page=ocr_by_page, ocr_noise_keys=ocr_noise_keys)

    rows: List[Dict[str, Any]] = []
    misread_keys: Set[Tuple[int, int]] = set()
    seen_report: Set[Tuple[int, int, int]] = set()

    for page in range(page_start, page_end + 1):
        page_ocr = ocr_by_page.get(page, [])
        primary_steps = [entry for entry in page_ocr if entry.get("is_primary_step")]
        page_crops = [crop for crop in crops if int(crop.get("page") or 0) == page]
        inferred_steps = _infer_sequence_valid_steps(page, page_start, page_end, crops, bag)
        valid_steps = _valid_steps_for_page(page, crops, primary_steps, inferred_steps)

        for entry in page_ocr:
            ocr_step = int(entry["step"])
            if (page, ocr_step) in ocr_noise_keys or (page, ocr_step) in substep_keys:
                continue
            match = _match_misread(ocr_step, entry, valid_steps, [])
            if not match:
                continue
            corrected, reason = match
            misread_keys.add((page, ocr_step))
            report_key = (page, ocr_step, corrected)
            if report_key in seen_report:
                continue
            seen_report.add(report_key)
            rows.append(
                {
                    "page": page,
                    "ocr_value": ocr_step,
                    "corrected_value": corrected,
                    "classification": MISREAD_CLASSIFICATION,
                    "match_reason": reason,
                    "source": entry.get("source"),
                    "raw_text": entry.get("raw_text"),
                    "valid_steps_on_page": [int(item["step"]) for item in valid_steps],
                }
            )

    if int(bag) == 4:
        for crop_id, corrected in KNOWN_CROP_ID_CORRECTIONS.items():
            match = re.search(r"p(\d+)_s(\d+)_", crop_id)
            if not match:
                continue
            page = int(match.group(1))
            ocr_value = int(match.group(2))
            key = (page, ocr_value, corrected)
            if key in seen_report:
                continue
            seen_report.add(key)
            rows.append(
                {
                    "page": page,
                    "ocr_value": ocr_value,
                    "corrected_value": corrected,
                    "classification": MISREAD_CLASSIFICATION,
                    "match_reason": "known_crop_id_correction",
                    "source": "crop_cache_history",
                    "crop_id": crop_id,
                }
            )
            misread_keys.add((page, ocr_value))

    rows.sort(key=lambda row: (int(row["page"]), int(row["ocr_value"]), int(row["corrected_value"])))
    return {
        "bag": bag,
        "page_start": page_start,
        "page_end": page_end,
        "classification": MISREAD_CLASSIFICATION,
        "row_count": len(rows),
        "misread_keys": [f"p{page:03d}_s{step}" for page, step in sorted(misread_keys)],
        "rows": rows,
    }


def substep_step_keys(
    bag: int = 4,
    *,
    crops: Optional[List[Dict[str, Any]]] = None,
    ocr_by_page: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ocr_noise_keys: Optional[Set[Tuple[int, int]]] = None,
) -> Set[Tuple[int, int]]:
    if crops is not None and ocr_by_page is not None:
        page_start, page_end = _load_bag_page_range(bag)
        if ocr_noise_keys is None:
            from bag_step_ocr_noise_detection import ocr_noise_step_keys

            ocr_noise_keys = ocr_noise_step_keys(bag, ocr_by_page=ocr_by_page)
        keys: Set[Tuple[int, int]] = set()
        for page in range(page_start, page_end + 1):
            page_ocr = ocr_by_page.get(page, [])
            primary_steps = [entry for entry in page_ocr if entry.get("is_primary_step")]
            page_crops = [crop for crop in crops if int(crop.get("page") or 0) == page]
            dominant = dominant_step_number(page, page_crops, primary_steps)
            for entry in page_ocr:
                ocr_step = int(entry["step"])
                if (page, ocr_step) in ocr_noise_keys:
                    continue
                if _match_substep(ocr_step, dominant):
                    keys.add((page, ocr_step))
        return keys

    payload = audit_bag_substep_numbers(bag)
    keys: Set[Tuple[int, int]] = set()
    for token in payload.get("substep_keys", []):
        match = re.search(r"p(\d+)_s(\d+)$", str(token))
        if match:
            keys.add((int(match.group(1)), int(match.group(2))))
    return keys


def suppressed_step_keys(bag: int = 4) -> Set[Tuple[int, int]]:
    from bag_step_ocr_noise_detection import ocr_noise_step_keys

    ocr_by_page = _load_ocr_detections(bag)
    return (
        misread_step_keys(bag)
        | substep_step_keys(bag)
        | ocr_noise_step_keys(bag, ocr_by_page=ocr_by_page)
    )


def misread_step_keys(bag: int = 4) -> Set[Tuple[int, int]]:
    payload = audit_bag_misread_steps(bag)
    keys: Set[Tuple[int, int]] = set()
    for token in payload.get("misread_keys", []):
        match = re.search(r"p(\d+)_s(\d+)$", str(token))
        if match:
            keys.add((int(match.group(1)), int(match.group(2))))
    return keys


def is_misread_step(page: int, step: int, bag: int = 4, cache: Optional[Set[Tuple[int, int]]] = None) -> bool:
    keys = cache if cache is not None else misread_step_keys(bag)
    return (int(page), int(step)) in keys


def is_substep_number(page: int, step: int, bag: int = 4, cache: Optional[Set[Tuple[int, int]]] = None) -> bool:
    keys = cache if cache is not None else substep_step_keys(bag)
    return (int(page), int(step)) in keys


def is_suppressed_step_number(page: int, step: int, bag: int = 4, cache: Optional[Set[Tuple[int, int]]] = None) -> bool:
    keys = cache if cache is not None else suppressed_step_keys(bag)
    return (int(page), int(step)) in keys
