"""
Export V2 callout/qty manifests to V1 crop_cache format for mask-review.

Reads instruction-v2 indexes and writes debug/crop_cache/{set_num}_bag{bag}.json
as a flat JSON list compatible with mask_review._load_bag_crops().
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paths import INDEXES_DIR, ROOT_DIR


PROJECT_ROOT = ROOT_DIR.parent
CROP_CACHE_DIR = PROJECT_ROOT / "debug" / "crop_cache"
TRAINING_LABELS_DIR = PROJECT_ROOT / "debug" / "training_labels"
SET_NUM_DEFAULT = "70618"
BAG_DEFAULT = 4
PAGE_RUN_ID = "70618_01"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_entries(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        entries = payload.get("entries")
        if isinstance(entries, list):
            return [dict(item) for item in entries if isinstance(item, dict)]
    return []


def _box_list(value: Any) -> List[int]:
    if isinstance(value, dict):
        return [
            int(value.get("x", 0) or 0),
            int(value.get("y", 0) or 0),
            int(value.get("w", 0) or 0),
            int(value.get("h", 0) or 0),
        ]
    if isinstance(value, list) and len(value) >= 4:
        return [int(value[i] or 0) for i in range(4)]
    return []


def _boxes_close(a: List[int], b: List[int], tolerance: int = 8) -> bool:
    if len(a) < 4 or len(b) < 4:
        return False
    return all(abs(int(a[i]) - int(b[i])) <= tolerance for i in range(4))


def _parse_crop_index(crop_id: str) -> Optional[int]:
    match = re.search(r"_c(\d+)$", str(crop_id or "").strip())
    return int(match.group(1)) if match else None


def _bag04_qty_key(crop_image_path: str) -> str:
    match = re.search(
        r"bag_04_page_(\d+)_step_(\d+)_(\d+)",
        str(crop_image_path or ""),
    )
    if not match:
        return ""
    page, step, seq = match.groups()
    return f"bag_04_page_{int(page):03d}_step_{int(step)}_{seq}"


def _page_image_path(set_num: str, page: int) -> str:
    return str(
        PROJECT_ROOT
        / "debug"
        / str(set_num)
        / PAGE_RUN_ID
        / "pages"
        / f"page_{int(page):03d}.png"
    )


def _qty_label(qty_text: List[str]) -> str:
    cleaned = [str(item).strip() for item in qty_text if str(item or "").strip()]
    return ", ".join(cleaned) if cleaned else "none"


def _coerce_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _coerce_int_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    result: List[int] = []
    for item in value:
        try:
            if item is None or item == "":
                continue
            result.append(int(item))
        except Exception:
            continue
    return result


def _load_bag_page_range(bag: int) -> Tuple[int, int]:
    path = INDEXES_DIR / "04_bag_map.json"
    if not path.exists():
        return 1, 10_000
    payload = _load_json(path)
    for entry in payload.get("bags", []) or []:
        if int(entry.get("bag") or 0) == int(bag):
            return int(entry.get("start_page") or 1), int(entry.get("end_page") or 10_000)
    return 1, 10_000


def _load_training_labels(set_num: str, bag: int) -> Dict[str, Dict[str, Any]]:
    path = TRAINING_LABELS_DIR / f"{set_num}_bag{int(bag)}.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    crops = payload.get("crops", {}) if isinstance(payload, dict) else {}
    if not isinstance(crops, dict):
        return {}
    return {str(crop_id): dict(record) for crop_id, record in crops.items() if isinstance(record, dict)}


def _build_qty_indexes(
    qty_entries: List[Dict[str, Any]],
    bag: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[int, int], List[Dict[str, Any]]]]:
    by_crop_id: Dict[str, Dict[str, Any]] = {}
    by_page_step: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for entry in qty_entries:
        if int(entry.get("bag") or 0) != int(bag):
            continue
        crop_id = str(entry.get("crop_id") or "").strip()
        if crop_id:
            by_crop_id[crop_id] = entry
        page = int(entry.get("page") or 0)
        step = int(entry.get("step") or 0)
        by_page_step.setdefault((page, step), []).append(entry)
    return by_crop_id, by_page_step


def _resolve_crop_id(
    callout_entry: Dict[str, Any],
    training_labels: Dict[str, Dict[str, Any]],
) -> str:
    existing = str(callout_entry.get("crop_id") or "").strip()
    if existing:
        return existing

    page = int(callout_entry.get("page") or 0)
    step = int(callout_entry.get("step") or 0)
    crop_box = _box_list(callout_entry.get("callout_crop_box"))

    candidates = [
        (crop_id, record)
        for crop_id, record in training_labels.items()
        if int(record.get("page") or 0) == page and int(record.get("step") or 0) == step
    ]
    if len(candidates) == 1:
        return candidates[0][0]
    if len(candidates) > 1 and crop_box:
        for crop_id, record in candidates:
            if _boxes_close(crop_box, _box_list(record.get("crop_box"))):
                return crop_id

    crop_index = callout_entry.get("crop_index")
    if crop_index is None:
        crop_index = 1
    return f"p{page}_s{step}_c{int(crop_index)}"


def _resolve_qty_entry(
    crop_id: str,
    callout_entry: Dict[str, Any],
    qty_by_crop_id: Dict[str, Dict[str, Any]],
    qty_by_page_step: Dict[Tuple[int, int], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    if crop_id in qty_by_crop_id:
        return qty_by_crop_id[crop_id]

    bag04_key = _bag04_qty_key(str(callout_entry.get("crop_image_path") or ""))
    if bag04_key and bag04_key in qty_by_crop_id:
        return qty_by_crop_id[bag04_key]

    page = int(callout_entry.get("page") or 0)
    step = int(callout_entry.get("step") or 0)
    page_step_hits = qty_by_page_step.get((page, step), [])
    if len(page_step_hits) == 1:
        return page_step_hits[0]

    crop_box = _box_list(callout_entry.get("callout_crop_box"))
    for entry in page_step_hits:
        detected = entry.get("detected_callout_crop_box")
        if isinstance(detected, dict) and _boxes_close(crop_box, _box_list(detected)):
            return entry
    return page_step_hits[0] if page_step_hits else {}


def _build_crop_record(
    crop_id: str,
    page: int,
    step: int,
    crop_box: List[int],
    crop_image_path: str,
    qty_entry: Dict[str, Any],
    training_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    qty_text = _coerce_str_list(qty_entry.get("qty_text"))
    qty_numbers = _coerce_int_list(qty_entry.get("qty_numbers"))
    if training_record:
        qty_text = qty_text or _coerce_str_list(training_record.get("qty_text"))
        qty_numbers = qty_numbers or _coerce_int_list(training_record.get("qty", training_record.get("qty_numbers")))

    qty_token_boxes = [
        dict(item)
        for item in list(qty_entry.get("qty_token_boxes", []) or [])
        if isinstance(item, dict)
    ]
    if not qty_token_boxes and training_record:
        qty_token_boxes = [
            dict(item)
            for item in list(training_record.get("qty_token_boxes", []) or [])
            if isinstance(item, dict)
        ]

    return {
        "crop_id": crop_id,
        "page": int(page),
        "step": int(step),
        "crop_box": crop_box,
        "crop_box_format": "xywh",
        "crop_image_path": crop_image_path,
        "qty_text": qty_text,
        "qty_numbers": qty_numbers,
        "qty_token_boxes": qty_token_boxes,
        "qty_label": _qty_label(qty_text),
        "source": "v2_export",
    }


def build_crop_cache(
    set_num: str,
    bag: int,
    *,
    backfill_training_labels: bool = True,
) -> List[Dict[str, Any]]:
    callout_entries = _manifest_entries(_load_json(INDEXES_DIR / "06_callout_crop_box_map.json"))
    qty_entries = _manifest_entries(_load_json(INDEXES_DIR / "07_qty_ocr_map.json"))
    training_labels = _load_training_labels(set_num, bag)
    qty_by_crop_id, qty_by_page_step = _build_qty_indexes(qty_entries, bag)
    start_page, end_page = _load_bag_page_range(bag)

    output_by_id: Dict[str, Dict[str, Any]] = {}

    for entry in callout_entries:
        if int(entry.get("bag") or 0) != int(bag):
            continue
        page = int(entry.get("page") or 0)
        if page < start_page or page > end_page:
            continue

        crop_id = _resolve_crop_id(entry, training_labels)
        crop_box = _box_list(entry.get("callout_crop_box"))
        if not crop_box:
            continue

        qty_entry = _resolve_qty_entry(crop_id, entry, qty_by_crop_id, qty_by_page_step)
        training_record = training_labels.get(crop_id)
        record = _build_crop_record(
            crop_id=crop_id,
            page=page,
            step=int(entry.get("step") or 0),
            crop_box=crop_box,
            crop_image_path=_page_image_path(set_num, page),
            qty_entry=qty_entry,
            training_record=training_record,
        )
        output_by_id[crop_id] = record

    if backfill_training_labels:
        for crop_id, training_record in training_labels.items():
            if crop_id in output_by_id:
                continue
            page = int(training_record.get("page") or 0)
            step = int(training_record.get("step") or 0)
            if page < start_page or page > end_page:
                continue
            crop_box = _box_list(training_record.get("crop_box"))
            if not crop_box:
                continue
            qty_entry = qty_by_crop_id.get(crop_id, {})
            output_by_id[crop_id] = _build_crop_record(
                crop_id=crop_id,
                page=page,
                step=step,
                crop_box=crop_box,
                crop_image_path=str(training_record.get("crop_image_path") or _page_image_path(set_num, page)),
                qty_entry=qty_entry,
                training_record=training_record,
            )

    crops = list(output_by_id.values())
    crops.sort(
        key=lambda item: (
            int(item.get("page") or 0),
            int(item.get("step") or 0),
            int(_parse_crop_index(str(item.get("crop_id") or "")) or 0),
            str(item.get("crop_id") or ""),
        )
    )
    return crops


def _write_atomic(path: Path, crops: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(crops, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _verify_output(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"crop cache not written: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected list crop cache: {path}")
    invalid = [
        str(item.get("crop_id") or "")
        for item in payload
        if not isinstance(item, dict) or not str(item.get("crop_id") or "").strip()
    ]
    if invalid:
        raise RuntimeError(f"Invalid crop cache entries (missing crop_id): {invalid[:5]}")
    return len(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export V2 manifests to V1 crop_cache JSON.")
    parser.add_argument("--set-num", default=SET_NUM_DEFAULT)
    parser.add_argument("--bag", type=int, default=BAG_DEFAULT)
    parser.add_argument(
        "--no-training-label-backfill",
        action="store_true",
        help="Do not add training_labels crops missing from V2 callout map.",
    )
    args = parser.parse_args()

    set_num = str(args.set_num).strip() or SET_NUM_DEFAULT
    bag = int(args.bag)
    out_path = CROP_CACHE_DIR / f"{set_num}_bag{bag}.json"

    crops = build_crop_cache(
        set_num,
        bag,
        backfill_training_labels=not args.no_training_label_backfill,
    )
    _write_atomic(out_path, crops)
    count = _verify_output(out_path)

    print(str(out_path))
    print(f"bag={bag} crop_count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
