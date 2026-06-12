"""Apply sequence ID corrections from bag_step_sequence_audit to crop_cache."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paths import ROOT_DIR


PROJECT_ROOT = ROOT_DIR.parent
CROP_CACHE_PATH = PROJECT_ROOT / "debug" / "crop_cache" / "70618_bag4.json"
AUDIT_PATH = ROOT_DIR / "debug" / "bag_step_sequence_audit" / "70618_bag4.json"

SEQUENCE_CORRECTIONS = [
    {
        "from_crop_id": "p77_s173_c1",
        "to_crop_id": "p77_s123_c1",
        "step": 123,
    },
    {
        "from_crop_id": "p70_s11_c1",
        "to_crop_id": "p70_s111_c1",
        "step": 111,
    },
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_crop_index(crop_id: str) -> Optional[int]:
    match = re.search(r"_c(\d+)$", str(crop_id or "").strip())
    return int(match.group(1)) if match else None


def _write_atomic(path: Path, crops: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(crops, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def apply_sequence_corrections() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not CROP_CACHE_PATH.exists():
        raise FileNotFoundError(CROP_CACHE_PATH)
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(AUDIT_PATH)

    audit = _load_json(AUDIT_PATH)
    audit_by_from = {str(item.get("crop_id") or ""): item for item in audit.get("suspects", []) or []}
    crops = [dict(item) for item in _load_json(CROP_CACHE_PATH) if isinstance(item, dict)]

    report: Dict[str, Any] = {
        "audit_path": str(AUDIT_PATH),
        "corrections_applied": [],
        "missing_crop_ids": [],
    }

    by_id: Dict[str, Dict[str, Any]] = {}
    for crop in crops:
        crop_id = str(crop.get("crop_id") or "").strip()
        if not crop_id:
            continue

        correction = next((item for item in SEQUENCE_CORRECTIONS if item["from_crop_id"] == crop_id), None)
        if correction:
            to_crop_id = str(correction["to_crop_id"])
            if to_crop_id in by_id:
                raise RuntimeError(f"Refusing to create duplicate crop_id: {to_crop_id}")

            audit_row = audit_by_from.get(crop_id, {})
            updated = dict(crop)
            updated["crop_id"] = to_crop_id
            updated["step"] = int(correction["step"])
            updated["sequence_correction"] = {
                "from_crop_id": crop_id,
                "from_step": int(crop.get("step") or 0),
                "to_crop_id": to_crop_id,
                "to_step": int(correction["step"]),
                "audit_status": audit_row.get("status"),
                "audit_confidence": audit_row.get("confidence"),
                "audit_reason": audit_row.get("reason_suspect"),
            }
            if updated.get("step_link") and crop_id == "p70_s11_c1":
                step_link = dict(updated["step_link"])
                step_link["canonical_crop_id"] = to_crop_id
                updated["step_link"] = step_link

            report["corrections_applied"].append(
                {
                    "from_crop_id": crop_id,
                    "to_crop_id": to_crop_id,
                    "step": int(correction["step"]),
                }
            )
            by_id[to_crop_id] = updated
            continue

        if crop_id in by_id:
            raise RuntimeError(f"Duplicate crop_id in source cache: {crop_id}")
        by_id[crop_id] = crop

    for correction in SEQUENCE_CORRECTIONS:
        if correction["from_crop_id"] not in {item["from_crop_id"] for item in report["corrections_applied"]}:
            report["missing_crop_ids"].append(correction["from_crop_id"])

    if report["missing_crop_ids"]:
        raise RuntimeError(f"Missing crops for correction: {report['missing_crop_ids']}")

    corrected = list(by_id.values())
    corrected.sort(
        key=lambda item: (
            int(item.get("page") or 0),
            int(item.get("step") or 0),
            int(_parse_crop_index(str(item.get("crop_id") or "")) or 0),
            str(item.get("crop_id") or ""),
        )
    )

    crop_ids = [str(item.get("crop_id") or "") for item in corrected]
    duplicates = sorted({crop_id for crop_id in crop_ids if crop_ids.count(crop_id) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate crop_id values after correction: {duplicates}")

    _write_atomic(CROP_CACHE_PATH, corrected)
    report["final_crop_count"] = len(corrected)
    report["duplicate_crop_ids"] = duplicates
    report["crop_cache_path"] = str(CROP_CACHE_PATH)
    return corrected, report


def main() -> int:
    _crops, report = apply_sequence_corrections()
    print(str(CROP_CACHE_PATH))
    print(f"final_crop_count={report['final_crop_count']}")
    print(f"corrections_applied={report['corrections_applied']}")
    print(f"duplicate_crop_ids={report['duplicate_crop_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
