"""Read-only audit: significant mask islands vs review-slot exposure.

Flags genuine review-slot exposure mismatches after human Bag 4 signoff (2026-06-12).

Naive comparison (total review slots vs significant islands) over-reports OVER-EXPOSED
when crops contain ignored slots or historical saved-label slot indices. This audit uses
refined exposure counts and suppresses explainable OVER gaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from paths import ROOT_DIR

PROJECT_ROOT = ROOT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean.services.bag_review_service import build_review_model  # noqa: E402
from clean.services.full_crop_mask_paths import (  # noqa: E402
    filter_significant_islands,
    find_full_mask_stem,
    master_islands_from_mask,
    raw_master_mask_path,
    sort_islands_for_slots,
)


def _mask_island_counts(set_num: str, bag: int, crop_id: str) -> tuple[int, int]:
    stem = find_full_mask_stem(str(set_num), int(bag), str(crop_id))
    if not stem:
        return 0, 0
    raw_path = raw_master_mask_path(stem)
    if not raw_path.is_file():
        return 0, 0
    raw_islands = sort_islands_for_slots(master_islands_from_mask(str(raw_path)))
    significant = filter_significant_islands(raw_islands)
    return len(significant), len(raw_islands)


def _exposure_slot_count(crop: Dict[str, Any]) -> int:
    """Active review slots = non-ignored slots shown in manual-match-review."""
    slots = list(crop.get("slots") or [])
    return sum(1 for slot in slots if not slot.get("ignored"))


def classify_slot_exposure(
    *,
    significant_islands: int,
    raw_islands: int,
    total_review_slots: int,
    exposure_slots: int,
    ignored_slot_count: int,
    slot_source: str,
    qty_text: List[str],
) -> Dict[str, Any]:
    """Return naive vs refined exception classification for one crop."""
    naive_gap = int(significant_islands) - int(total_review_slots)
    exposure_gap = int(significant_islands) - int(exposure_slots)

    under_exposed = exposure_gap > 0
    over_suppressed_reason: Optional[str] = None
    over_exposed = False

    if int(exposure_slots) > int(significant_islands):
        if ignored_slot_count > 0 and int(exposure_slots) <= int(significant_islands):
            over_suppressed_reason = "ignored_slots"
        elif str(slot_source) == "qty" and len(list(qty_text or [])) == int(exposure_slots):
            over_suppressed_reason = "qty_authoritative"
        elif int(raw_islands) >= int(exposure_slots):
            over_suppressed_reason = "raw_islands_cover_exposure"
        else:
            over_exposed = True

    naive_under = naive_gap > 0
    naive_over = naive_gap < 0

    exception = under_exposed or over_exposed
    category: Optional[str] = None
    if under_exposed:
        category = "UNDER-EXPOSED"
    elif over_exposed:
        category = "OVER-EXPOSED"

    return {
        "significant_islands": int(significant_islands),
        "raw_islands": int(raw_islands),
        "total_review_slots": int(total_review_slots),
        "exposure_slots": int(exposure_slots),
        "ignored_slot_count": int(ignored_slot_count),
        "naive_gap": int(naive_gap),
        "exposure_gap": int(exposure_gap),
        "naive_under_exposed": bool(naive_under),
        "naive_over_exposed": bool(naive_over),
        "under_exposed": bool(under_exposed),
        "over_exposed": bool(over_exposed),
        "over_suppressed_reason": over_suppressed_reason,
        "exception": bool(exception),
        "category": category,
    }


def audit_bag_slot_exposure(set_num: str, bag: int) -> Dict[str, Any]:
    review_model = build_review_model(str(set_num), int(bag))
    rows: List[Dict[str, Any]] = []

    for crop in sorted(list(review_model.get("crops") or []), key=lambda item: str(item.get("crop_id") or "")):
        crop_id = str(crop.get("crop_id") or "").strip()
        if not crop_id:
            continue
        significant_islands, raw_islands = _mask_island_counts(set_num, bag, crop_id)
        total_review_slots = len(list(crop.get("slots") or []))
        exposure_slots = _exposure_slot_count(crop)
        ignored_slots = list(crop.get("ignored_slots") or [])
        qty_text = list(crop.get("crop_qty_text") or crop.get("qty_text") or [])
        slot_source = str(crop.get("slot_source") or "")

        classification = classify_slot_exposure(
            significant_islands=significant_islands,
            raw_islands=raw_islands,
            total_review_slots=total_review_slots,
            exposure_slots=exposure_slots,
            ignored_slot_count=len(ignored_slots),
            slot_source=slot_source,
            qty_text=qty_text,
        )
        rows.append(
            {
                "crop_id": crop_id,
                "slot_source": slot_source,
                "qty_text": qty_text,
                "saved_labels_count": int(crop.get("filled_slots") or 0),
                **classification,
            }
        )

    exceptions = [row for row in rows if row.get("exception")]
    naive_exceptions = [
        row
        for row in rows
        if row.get("naive_under_exposed") or row.get("naive_over_exposed")
    ]
    suppressed_over = [
        row
        for row in rows
        if row.get("naive_over_exposed") and not row.get("over_exposed")
    ]

    return {
        "set_num": str(set_num),
        "bag": int(bag),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "audit_version": "refined_exposure_v1",
        "rules": {
            "under_exposed": "significant_islands > exposure_slots (non-ignored)",
            "over_exposed": "exposure_slots > significant_islands, unless suppressed by ignored_slots, qty_authoritative, or raw_islands_cover_exposure",
            "exposure_slots": "count of review slots where ignored=false",
        },
        "summary": {
            "review_crops": len(rows),
            "exceptions": len(exceptions),
            "under_exposed": sum(1 for row in exceptions if row.get("category") == "UNDER-EXPOSED"),
            "over_exposed": sum(1 for row in exceptions if row.get("category") == "OVER-EXPOSED"),
            "naive_exceptions": len(naive_exceptions),
            "suppressed_over_exposed": len(suppressed_over),
        },
        "exceptions": exceptions,
        "suppressed_over_exposed": suppressed_over,
        "all": rows,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit mask islands vs review-slot exposure.")
    parser.add_argument("--set-num", default="70618")
    parser.add_argument("--bag", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "debug" / "bag4_slot_exposure_audit.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    payload = audit_bag_slot_exposure(str(args.set_num), int(args.bag))
    _write_json(Path(args.out), payload)

    summary = payload["summary"]
    print(
        f"Bag {args.bag}: {summary['exceptions']} exception(s) "
        f"({summary['under_exposed']} under, {summary['over_exposed']} over) "
        f"from {summary['review_crops']} crops; "
        f"{summary['suppressed_over_exposed']} naive over-exposed suppressed"
    )
    for row in payload["exceptions"]:
        print(
            f"  {row['category']}: {row['crop_id']} "
            f"(sig={row['significant_islands']} exposure={row['exposure_slots']} "
            f"total={row['total_review_slots']} gap={row['exposure_gap']})"
        )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
