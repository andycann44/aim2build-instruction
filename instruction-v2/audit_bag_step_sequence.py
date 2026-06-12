"""
Bag/page step sequence sanity audit for crop_cache entries.

Flags crops whose step number breaks local page and bag sequence order.
Does not modify crop_cache.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from paths import INDEXES_DIR, ROOT_DIR
from bag4_misread_step_detection import substep_step_keys
from bag_step_ocr_noise_detection import ocr_noise_step_keys


PROJECT_ROOT = ROOT_DIR.parent
CROP_CACHE_DIR = PROJECT_ROOT / "debug" / "crop_cache"
STEP_MAP_PATH = INDEXES_DIR / "05_step_map.json"
STEP_PROBE_PATH = ROOT_DIR / "debug" / "sequence_completeness" / "05_step_probe.json"
BAG_MAP_PATH = INDEXES_DIR / "04_bag_map.json"
OUT_DIR = ROOT_DIR / "debug" / "bag_step_sequence_audit"

SET_NUM_DEFAULT = "70618"
BAG_DEFAULT = 4
SUSPECT_STATUS = "SUSPECT_STEP_ID"
MAX_ADJACENT_PAGE_STEP_JUMP = 8
OCR_DIGIT_SWAPS = {"7": "2", "2": "7", "8": "3", "3": "8", "5": "6", "6": "5", "1": "7", "0": "8"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_step_from_crop_id(crop_id: str) -> Optional[int]:
    match = re.search(r"_s(\d+)_", str(crop_id or ""))
    return int(match.group(1)) if match else None


def _parse_crop_index(crop_id: str) -> Optional[int]:
    match = re.search(r"_c(\d+)$", str(crop_id or ""))
    return int(match.group(1)) if match else None


def _box_xywh(box: Any) -> List[int]:
    if isinstance(box, dict):
        return [int(box.get("x", 0) or 0), int(box.get("y", 0) or 0), int(box.get("w", 0) or 0), int(box.get("h", 0) or 0)]
    if isinstance(box, list) and len(box) >= 4:
        return [int(box[i] or 0) for i in range(4)]
    return []


def _load_bag_page_range(bag: int) -> Tuple[int, int]:
    payload = _load_json(BAG_MAP_PATH)
    for entry in payload.get("bags", []) or []:
        if int(entry.get("bag") or 0) == int(bag):
            return int(entry.get("start_page") or 1), int(entry.get("end_page") or 10_000)
    return 1, 10_000


def _load_detected_steps_by_page(bag: int) -> Dict[int, List[Dict[str, Any]]]:
    by_page: Dict[int, Dict[int, Dict[str, Any]]] = {}
    substep_keys = substep_step_keys(bag)
    ocr_noise_keys = ocr_noise_step_keys(bag)

    def add(page: int, step: int, step_box: Any, source: str) -> None:
        if not page or not step:
            return
        if (page, step) in substep_keys or (page, step) in ocr_noise_keys:
            return
        by_page.setdefault(page, {})
        existing = by_page[page].get(step)
        payload = {"step": step, "step_box": _box_xywh(step_box), "sources": [source]}
        if existing is None:
            by_page[page][step] = payload
        elif source not in existing["sources"]:
            existing["sources"].append(source)
            if not existing.get("step_box") and payload["step_box"]:
                existing["step_box"] = payload["step_box"]

    step_map = _load_json(STEP_MAP_PATH)
    for entry in step_map.get("steps", []) or []:
        if int(entry.get("bag") or 0) != bag:
            continue
        add(int(entry.get("page") or 0), int(entry.get("step_number") or 0), entry.get("step_box"), "05_step_map.json")

    for page_entry in step_map.get("pages", []) or []:
        if int(page_entry.get("bag") or 0) != bag:
            continue
        page = int(page_entry.get("page") or 0)
        for candidate in page_entry.get("full_page_audit_candidates", []) or []:
            if candidate.get("rejection_reason"):
                continue
            add(page, int(candidate.get("step_number") or 0), candidate.get("step_box"), "full_page_audit")

    if STEP_PROBE_PATH.exists():
        for entry in _load_json(STEP_PROBE_PATH).get("steps", []) or []:
            if int(entry.get("bag") or 0) != bag:
                continue
            add(int(entry.get("page") or 0), int(entry.get("step_number") or 0), entry.get("step_box"), "05_step_probe.json")

    return {page: [by_page[page][step] for step in sorted(by_page[page])] for page in sorted(by_page)}


def _sort_crops_visual(crops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(item: Dict[str, Any]) -> Tuple[int, int, int, str]:
        box = _box_xywh(item.get("crop_box"))
        y = box[1] if len(box) >= 2 else 0
        x = box[0] if len(box) >= 1 else 0
        return (int(item.get("page") or 0), y, x, str(item.get("crop_id") or ""))

    return sorted(crops, key=key)


def _ocr_variants(step: int) -> Set[int]:
    text = str(int(step))
    variants = {int(text)}
    for idx, ch in enumerate(text):
        swap = OCR_DIGIT_SWAPS.get(ch)
        if not swap:
            continue
        candidate = int(text[:idx] + swap + text[idx + 1 :])
        variants.add(candidate)
    return variants


def _nearest_step_box(
    crop_box: List[int],
    detected_steps: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if len(crop_box) < 4 or not detected_steps:
        return None
    cx = crop_box[0] + crop_box[2] // 2
    cy = crop_box[1] + crop_box[3] // 2
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for entry in detected_steps:
        box = entry.get("step_box") or []
        if len(box) < 4:
            continue
        sx = box[0] + box[2] // 2
        sy = box[1] + box[3] // 2
        score = abs(cx - sx) + abs(cy - sy) * 1.5
        if best is None or score < best[0]:
            best = (score, entry)
    return best[1] if best else None


def _propose_step(
    crop: Dict[str, Any],
    prev_step: Optional[int],
    next_step: Optional[int],
    detected_steps: List[Dict[str, Any]],
    same_page_crops: List[Dict[str, Any]],
) -> Tuple[Optional[int], float, str]:
    current = int(crop.get("step") or 0)
    crop_id_step = _parse_step_from_crop_id(str(crop.get("crop_id") or ""))

    if (
        crop_id_step is not None
        and crop_id_step != current
        and (prev_step is None or current > prev_step)
        and (next_step is None or current < next_step)
    ):
        return current, 0.93, "record_step_fits_sequence"

    if prev_step is not None and next_step is not None and next_step - prev_step == 2:
        gap_step = prev_step + 1
        if gap_step < next_step:
            confidence = 0.95
            if gap_step in _ocr_variants(current):
                confidence = min(0.99, confidence + 0.03)
            return gap_step, confidence, "sequence_gap_fill"

    if prev_step is not None and next_step is not None and prev_step < next_step:
        span = next_step - prev_step
        if len(same_page_crops) == 1 and span <= MAX_ADJACENT_PAGE_STEP_JUMP + 4:
            interpolated = prev_step + max(1, span // 2)
            if prev_step < interpolated < next_step:
                return interpolated, 0.82, "interpolate_between_neighbour_pages"

    nearest = _nearest_step_box(_box_xywh(crop.get("crop_box")), detected_steps)
    if nearest:
        candidate = int(nearest.get("step") or 0)
        if candidate and (prev_step is None or candidate > prev_step) and (next_step is None or candidate < next_step):
            return candidate, 0.72, "nearest_detected_step_box"

    for variant in sorted(_ocr_variants(current)):
        if variant == current:
            continue
        if prev_step is not None and variant <= prev_step:
            continue
        if next_step is not None and variant >= next_step:
            continue
        return variant, 0.68, "ocr_variant_in_sequence_window"

    if prev_step is not None and (next_step is None or prev_step + 1 < next_step):
        return prev_step + 1, 0.55, "fallback_prev_plus_one"

    return None, 0.0, "unresolved"


def _same_page_crops(crops: List[Dict[str, Any]], page: int) -> List[Dict[str, Any]]:
    return _sort_crops_visual([c for c in crops if int(c.get("page") or 0) == page])


def _trusted_neighbour_steps(crops: List[Dict[str, Any]], page: int) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    prev_page = max((int(c.get("page") or 0) for c in crops if int(c.get("page") or 0) < page), default=None)
    next_page = min((int(c.get("page") or 0) for c in crops if int(c.get("page") or 0) > page), default=None)
    prev_step = max(
        (int(c.get("step") or 0) for c in crops if int(c.get("page") or 0) == prev_page),
        default=None,
    ) if prev_page is not None else None
    next_step = min(
        (int(c.get("step") or 0) for c in crops if int(c.get("page") or 0) == next_page),
        default=None,
    ) if next_page is not None else None
    return prev_page, prev_step, next_page, next_step


def _filter_detected_for_page(
    detected_steps: List[Dict[str, Any]],
    prev_step: Optional[int],
    next_step: Optional[int],
    page_crop_steps: List[int],
) -> List[Dict[str, Any]]:
    if not detected_steps:
        return []
    anchor = page_crop_steps[:] if page_crop_steps else []
    if prev_step is not None:
        anchor.append(prev_step)
    if next_step is not None:
        anchor.append(next_step)
    if not anchor:
        return detected_steps
    lo = min(anchor) - 12
    hi = max(anchor) + 12
    return [item for item in detected_steps if lo <= int(item.get("step") or 0) <= hi]


def _detect_suspect(
    crop: Dict[str, Any],
    crops: List[Dict[str, Any]],
    detected_by_page: Dict[int, List[Dict[str, Any]]],
    upstream_suspect_pages: Set[int],
) -> Tuple[bool, List[str]]:
    page = int(crop.get("page") or 0)
    current = int(crop.get("step") or 0)
    crop_id_step = _parse_step_from_crop_id(str(crop.get("crop_id") or ""))
    reasons: List[str] = []

    prev_page, prev_step, next_page, next_step = _trusted_neighbour_steps(crops, page)
    if prev_page in upstream_suspect_pages:
        prev_page = None
        prev_step = None

    page_crop_steps = [int(c.get("step") or 0) for c in crops if int(c.get("page") or 0) == page]
    detected_steps = _filter_detected_for_page(
        detected_by_page.get(page, []),
        prev_step,
        next_step,
        page_crop_steps,
    )
    detected_numbers = [int(item.get("step") or 0) for item in detected_steps]

    if next_step is not None and current >= next_step:
        reasons.append(f"step {current} is not less than next page min step {next_step} (page {next_page})")

    if prev_step is not None and current <= prev_step and page > (prev_page or page):
        reasons.append(f"step {current} is not greater than previous page max step {prev_step} (page {prev_page})")

    if prev_page is not None and prev_step is not None and page - prev_page <= 2:
        jump = current - prev_step
        if jump > MAX_ADJACENT_PAGE_STEP_JUMP:
            reasons.append(
                f"step jump {jump} from page {prev_page} step {prev_step} exceeds adjacent-page limit {MAX_ADJACENT_PAGE_STEP_JUMP}"
            )

    if prev_step is not None and next_step is not None and next_step - prev_step == 2 and current != prev_step + 1:
        reasons.append(f"only missing sequence step between {prev_step} and {next_step} is {prev_step + 1}")

    if crop_id_step is not None and crop_id_step != current:
        if (prev_step is None or crop_id_step <= prev_step) or (next_step is not None and crop_id_step >= next_step):
            reasons.append(f"crop_id parses step {crop_id_step} which also breaks sequence (record step {current})")

    if (
        detected_numbers
        and current not in detected_numbers
        and len(page_crop_steps) == 1
        and prev_step is not None
        and next_step is not None
        and next_step - prev_step == 2
        and prev_step + 1 not in detected_numbers
    ):
        reasons.append(f"detected page steps {detected_numbers} disagree with expected local sequence")

    return bool(reasons), reasons


def audit_bag_step_sequence(set_num: str, bag: int) -> Dict[str, Any]:
    cache_path = CROP_CACHE_DIR / f"{set_num}_bag{bag}.json"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    start_page, end_page = _load_bag_page_range(bag)
    crops = [dict(item) for item in _load_json(cache_path) if isinstance(item, dict)]
    crops = [c for c in crops if start_page <= int(c.get("page") or 0) <= end_page]
    crops_sorted = _sort_crops_visual(crops)
    detected_by_page = _load_detected_steps_by_page(bag)

    suspects: List[Dict[str, Any]] = []
    upstream_suspect_pages: Set[int] = set()
    audit_order = sorted(crops_sorted, key=lambda item: (int(item.get("page") or 0), int(item.get("step") or 0)))

    for crop in audit_order:
        is_suspect, reasons = _detect_suspect(crop, crops_sorted, detected_by_page, upstream_suspect_pages)
        if not is_suspect:
            continue
        upstream_suspect_pages.add(int(crop.get("page") or 0))

        page = int(crop.get("page") or 0)
        current = int(crop.get("step") or 0)
        prev_page, prev_step, next_page, next_step = _trusted_neighbour_steps(crops_sorted, page)
        if prev_page in upstream_suspect_pages:
            prev_page, prev_step = None, None
        page_crop_steps = [int(c.get("step") or 0) for c in crops_sorted if int(c.get("page") or 0) == page]
        filtered_detected = _filter_detected_for_page(
            detected_by_page.get(page, []),
            prev_step,
            next_step,
            page_crop_steps,
        )
        proposed_step, confidence, proposal_method = _propose_step(
            crop,
            prev_step,
            next_step,
            filtered_detected,
            _same_page_crops(crops_sorted, page),
        )
        crop_index = _parse_crop_index(str(crop.get("crop_id") or "")) or 1
        proposed_crop_id = f"p{page}_s{int(proposed_step)}_c{crop_index}" if proposed_step else None

        suspects.append(
            {
                "status": SUSPECT_STATUS,
                "crop_id": str(crop.get("crop_id") or ""),
                "page": page,
                "current_step": current,
                "crop_id_parsed_step": _parse_step_from_crop_id(str(crop.get("crop_id") or "")),
                "reason_suspect": "; ".join(reasons),
                "proposed_step": proposed_step,
                "proposed_crop_id": proposed_crop_id,
                "confidence": round(confidence, 4) if proposed_step else 0.0,
                "proposal_method": proposal_method,
                "context": {
                    "prev_page": prev_page,
                    "prev_step": prev_step,
                    "next_page": next_page,
                    "next_step": next_step,
                    "detected_page_steps": [int(item.get("step") or 0) for item in filtered_detected],
                },
                "crop_box": _box_xywh(crop.get("crop_box")),
            }
        )

    payload = {
        "name": "bag_step_sequence_audit",
        "set_num": set_num,
        "bag": bag,
        "page_range": [start_page, end_page],
        "crop_cache_path": str(cache_path),
        "suspect_count": len(suspects),
        "suspects": suspects,
        "rules": {
            "status": SUSPECT_STATUS,
            "max_adjacent_page_step_jump": MAX_ADJACENT_PAGE_STEP_JUMP,
            "does_not_modify_crop_cache": True,
        },
    }
    return payload


def _print_table(suspects: List[Dict[str, Any]]) -> None:
    header = ["crop_id", "page", "current_step", "reason_suspect", "proposed_step", "proposed_crop_id", "confidence"]
    print("\t".join(header))
    for row in suspects:
        print(
            "\t".join(
                [
                    str(row.get("crop_id") or ""),
                    str(row.get("page") or ""),
                    str(row.get("current_step") or ""),
                    str(row.get("reason_suspect") or ""),
                    str(row.get("proposed_step") if row.get("proposed_step") is not None else ""),
                    str(row.get("proposed_crop_id") or ""),
                    str(row.get("confidence") or ""),
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit crop_cache step IDs against bag/page sequence.")
    parser.add_argument("--set-num", default=SET_NUM_DEFAULT)
    parser.add_argument("--bag", type=int, default=BAG_DEFAULT)
    args = parser.parse_args()

    payload = audit_bag_step_sequence(str(args.set_num), int(args.bag))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.set_num}_bag{int(args.bag)}.json"
    _write_json(out_path, payload)
    _print_table(payload.get("suspects", []) or [])
    print(str(out_path))
    print(f"suspect_count={payload.get('suspect_count', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
