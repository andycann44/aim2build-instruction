"""Promote Bag 4 V1-recovered callout crops into debug/crop_cache."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from paths import ROOT_DIR


PROJECT_ROOT = ROOT_DIR.parent
CROP_CACHE_PATH = PROJECT_ROOT / "debug" / "crop_cache" / "70618_bag4.json"
RECOVERY_INDEX_PATH = ROOT_DIR / "debug" / "bag4_v1_callout_recovery" / "index.json"
SET_NUM = "70618"
PAGE_RUN_ID = "70618_01"

PROMOTE_STEPS = {110, 112, 113, 115, 116, 117, 119, 126, 131, 84, 86}
EXCLUDE_STEPS = {4, 7, 102}
# Page-aware V1 recovery targets for verified REAL_CALLOUT promotions.
VERIFIED_REAL_CALLOUT_TARGETS: Dict[int, int] = {
    84: 61,
    86: 62,
}
STEP_111_LINK = {
    "crop_id": "p70_s111_c1",
    "page": 70,
    "step": 111,
    "ocr_step": 11,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _page_image_path(page: int) -> str:
    return str(
        PROJECT_ROOT
        / "debug"
        / SET_NUM
        / PAGE_RUN_ID
        / "pages"
        / f"page_{int(page):03d}.png"
    )


def _parse_crop_index(crop_id: str) -> Optional[int]:
    match = re.search(r"_c(\d+)$", str(crop_id or "").strip())
    return int(match.group(1)) if match else None


def _write_atomic(path: Path, crops: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(crops, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _load_recovery_rows() -> Dict[int, Dict[str, Any]]:
    payload = _load_json(RECOVERY_INDEX_PATH)
    rows: Dict[int, Dict[str, Any]] = {}
    for row in payload.get("results", []) or []:
        step = int(row.get("step") or 0)
        page = int(row.get("page") or 0)
        if not step or not row.get("crop_box"):
            continue
        expected_page = VERIFIED_REAL_CALLOUT_TARGETS.get(step)
        if expected_page is not None and page != expected_page:
            continue
        if step in rows and expected_page is None:
            continue
        rows[step] = dict(row)
    return rows


def _build_promoted_record(row: Dict[str, Any]) -> Dict[str, Any]:
    page = int(row.get("page") or 0)
    step = int(row.get("step") or 0)
    crop_id = str(row.get("candidate_crop_id") or f"p{page}_s{step}_c1")
    return {
        "crop_id": crop_id,
        "page": page,
        "step": step,
        "crop_box": [int(v) for v in list(row.get("crop_box") or [])[:4]],
        "crop_box_format": "xywh",
        "crop_image_path": _page_image_path(page),
        "qty_text": [],
        "qty_numbers": [],
        "qty_token_boxes": [],
        "qty_label": "none",
        "source": "bag4_v1_recovery",
        "recovery_confidence": row.get("confidence"),
        "recovery_method": row.get("method"),
    }


def promote_bag4_recovered_crops() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not CROP_CACHE_PATH.exists():
        raise FileNotFoundError(CROP_CACHE_PATH)
    if not RECOVERY_INDEX_PATH.exists():
        raise FileNotFoundError(RECOVERY_INDEX_PATH)

    crops = [dict(item) for item in _load_json(CROP_CACHE_PATH) if isinstance(item, dict)]
    recovery_rows = _load_recovery_rows()

    by_id: Dict[str, Dict[str, Any]] = {}
    existing_steps: Set[int] = set()
    report: Dict[str, Any] = {
        "promoted_steps": [],
        "already_promoted_steps": [],
        "linked_steps": [],
        "skipped_steps": sorted(EXCLUDE_STEPS),
        "updated_crop_ids": [],
        "added_crop_ids": [],
    }

    for crop in crops:
        crop_id = str(crop.get("crop_id") or "").strip()
        step = int(crop.get("step") or 0)
        if step in EXCLUDE_STEPS:
            continue
        if crop_id == STEP_111_LINK["crop_id"]:
            report["linked_steps"].append({"step": 111, "crop_id": crop_id})
        by_id[crop_id] = crop
        if step:
            existing_steps.add(step)

    steps_needing_recovery = sorted(step for step in PROMOTE_STEPS if step not in existing_steps)
    missing_recovery = sorted(step for step in steps_needing_recovery if step not in recovery_rows)
    if missing_recovery:
        raise RuntimeError(f"Missing V1 recovery rows for steps: {missing_recovery}")

    for step in sorted(PROMOTE_STEPS):
        if step in existing_steps:
            report["already_promoted_steps"].append(step)
            continue
        row = recovery_rows[step]
        record = _build_promoted_record(row)
        crop_id = str(record["crop_id"])
        if crop_id in by_id:
            existing = by_id[crop_id]
            if int(existing.get("step") or 0) == step:
                report["already_promoted_steps"].append(step)
                continue
            raise RuntimeError(
                f"Refusing to overwrite existing crop_id {crop_id} "
                f"(existing step {existing.get('step')}, recovery step {step})"
            )
        by_id[crop_id] = record
        report["promoted_steps"].append(step)
        report["added_crop_ids"].append(crop_id)

    if STEP_111_LINK["crop_id"] not in by_id:
        raise RuntimeError(f"Missing link target crop: {STEP_111_LINK['crop_id']}")

    promoted = list(by_id.values())
    promoted.sort(
        key=lambda item: (
            int(item.get("page") or 0),
            int(item.get("step") or 0),
            int(_parse_crop_index(str(item.get("crop_id") or "")) or 0),
            str(item.get("crop_id") or ""),
        )
    )

    crop_ids = [str(item.get("crop_id") or "") for item in promoted]
    duplicates = sorted({crop_id for crop_id in crop_ids if crop_ids.count(crop_id) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate crop_id values after promotion: {duplicates}")

    _write_atomic(CROP_CACHE_PATH, promoted)
    report["final_crop_count"] = len(promoted)
    report["duplicate_crop_ids"] = duplicates
    report["crop_cache_path"] = str(CROP_CACHE_PATH)
    return promoted, report


def main() -> int:
    promoted, report = promote_bag4_recovered_crops()
    print(str(CROP_CACHE_PATH))
    print(f"final_crop_count={report['final_crop_count']}")
    print(f"promoted_steps={report['promoted_steps']}")
    print(f"linked_steps={report['linked_steps']}")
    print(f"duplicate_crop_ids={report['duplicate_crop_ids']}")
    print(f"already_promoted_steps={report['already_promoted_steps']}")
    print(f"added={len(report['added_crop_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
