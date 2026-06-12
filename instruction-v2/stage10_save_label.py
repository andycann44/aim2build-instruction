"""
V2 final-save bridge: write a confirmed slot label to debug/training_labels/.

Reads V2 manifests (08_part_segmentation_map, 09_match_manifest) and writes
to the V1 training label format via bag_review_service.save_slot_label().

No stages 0-9 are touched. bag_review_service is the sole write path.

Usage:
  # Dry-run — prints exactly what would be written, no file changes:
  python stage10_save_label.py \\
      --crop-id bag_01_page_012_step_12_002 \\
      --segment-index 1 \\
      --accept-top \\
      --dry-run

  # Accept the top candidate and write:
  python stage10_save_label.py \\
      --crop-id bag_01_page_012_step_12_002 \\
      --segment-index 1 \\
      --accept-top

  # Specify part explicitly:
  python stage10_save_label.py \\
      --crop-id bag_01_page_012_step_12_002 \\
      --segment-index 1 \\
      --part-num 3020 \\
      --color-id 484 \\
      [--qty 6]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

_V2_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _V2_DIR.parent
_INDEXES_DIR = _V2_DIR / "indexes"

sys.path.insert(0, str(_REPO_ROOT))
from clean.services.bag_review_service import save_slot_label


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_set_context() -> dict:
    return _load_json(_INDEXES_DIR / "00_set_context.json")


def _load_match_manifest() -> list[dict]:
    data = _load_json(_INDEXES_DIR / "09_match_manifest.json")
    return data.get("entries", []) if isinstance(data, dict) else list(data)


def _load_seg_map() -> list[dict]:
    data = _load_json(_INDEXES_DIR / "08_part_segmentation_map.json")
    return data.get("entries", []) if isinstance(data, dict) else list(data)


def _load_qty_ocr_map() -> dict[str, dict]:
    """Return a dict keyed by V1 crop_id for qty lookup."""
    data = _load_json(_INDEXES_DIR / "07_qty_ocr_map.json")
    entries = data.get("entries", []) if isinstance(data, dict) else list(data)
    return {str(e.get("crop_id") or ""): e for e in entries if e.get("crop_id")}


# ---------------------------------------------------------------------------
# crop_id conversion
# ---------------------------------------------------------------------------

_V2_CROP_PATTERN = re.compile(
    r"^bag_(\d+)_page_(\d+)_step_(\w+)_(\d+)$"
)


def v2_crop_id_to_v1(v2_id: str) -> str:
    """Convert 'bag_01_page_012_step_12_002' → 'p12_s12_c2'.

    For crops whose step is recorded as 'unknown' this returns a best-effort
    ID; those crops cannot be saved with a meaningful V1 key.
    """
    m = _V2_CROP_PATTERN.match(v2_id)
    if not m:
        # Already V1 format or unrecognised — return as-is.
        return v2_id
    page = int(m.group(2))
    step_raw = m.group(3)
    idx = int(m.group(4))
    if not step_raw.isdigit():
        raise ValueError(
            f"Cannot convert V2 crop_id '{v2_id}': step is '{step_raw}', "
            "no V1 crop_id equivalent exists for unknown-step crops."
        )
    step = int(step_raw)
    return f"p{page}_s{step}_c{idx}"


def _normalize_set_num(raw: str) -> str:
    """Strip trailing '-N' suffixes so '70618-1' becomes '70618'."""
    return re.sub(r"-\d+$", "", str(raw or "").strip()) or raw


# ---------------------------------------------------------------------------
# Core bridge function
# ---------------------------------------------------------------------------

def v2_accept_match(
    v2_crop_id: str,
    segment_index: int,
    part_num: str,
    color_id: int,
    qty: Optional[int] = None,
) -> dict:
    """Save a confirmed part label for a V2 segment into training_labels/.

    Args:
        v2_crop_id:     V2 crop_id (e.g. 'bag_01_page_012_step_12_002')
                        or a V1 crop_id (e.g. 'p12_s12_c2') — both accepted.
        segment_index:  1-based segment index from V2 manifests.
        part_num:       LEGO part number string (e.g. '3020').
        color_id:       BrickLink color id (e.g. 484).
        qty:            Optional quantity override; if None, derived from
                        the qty_sequence stored in the crop record.

    Returns:
        The updated crop record dict from bag_review_service.
    """
    context = _load_set_context()
    set_num = _normalize_set_num(str(context.get("set_num") or ""))
    if not set_num:
        raise ValueError("set_num missing from 00_set_context.json")

    # Derive bag from v2_crop_id if possible; fall back to context.
    bag: Optional[int] = None
    m = _V2_CROP_PATTERN.match(v2_crop_id)
    if m:
        bag = int(m.group(1))
    if bag is None:
        ctx_bag = context.get("bag")
        if ctx_bag is not None:
            bag = int(ctx_bag)
    if bag is None:
        raise ValueError(
            "Cannot determine bag from crop_id or 00_set_context.json. "
            "Pass a V2 crop_id of the form 'bag_NN_page_...'."
        )

    v1_crop_id = v2_crop_id_to_v1(v2_crop_id)

    # segment_index in V2 is 1-based; V1 slot_index is 0-based.
    slot_index = int(segment_index) - 1
    if slot_index < 0:
        raise ValueError(f"segment_index must be >= 1, got {segment_index}")

    crop_record = save_slot_label(
        set_num=set_num,
        bag=bag,
        crop_id=v1_crop_id,
        slot_index=slot_index,
        part_num=str(part_num).strip(),
        color_id=int(color_id),
        qty=qty,
    )
    return crop_record


def _find_match_entry(entries: list[dict], v2_crop_id: str, segment_index: int) -> Optional[dict]:
    for e in entries:
        if e.get("crop_id") == v2_crop_id and e.get("segment_index") == segment_index:
            return e
    return None


def _label_path(set_num: str, bag: int) -> Path:
    safe = "".join(ch for ch in str(set_num).strip() if ch.isalnum() or ch in "-_") or "unknown"
    return _REPO_ROOT / "debug" / "training_labels" / f"{safe}_bag{bag}.json"


# ---------------------------------------------------------------------------
# Dry-run preview
# ---------------------------------------------------------------------------

def resolve_preview(
    v2_crop_id: str,
    segment_index: int,
    part_num: str,
    color_id: int,
    color_name: str = "",
    qty: Optional[int] = None,
) -> dict:
    """Resolve all fields that would be written without touching any file.

    Returns a plain dict with every field that the write path would use,
    plus the target file path. Safe to call repeatedly.
    """
    context = _load_set_context()
    set_num = _normalize_set_num(str(context.get("set_num") or ""))

    bag: Optional[int] = None
    m = _V2_CROP_PATTERN.match(v2_crop_id)
    if m:
        bag = int(m.group(1))
    if bag is None:
        ctx_bag = context.get("bag")
        if ctx_bag is not None:
            bag = int(ctx_bag)
    if bag is None:
        raise ValueError("Cannot determine bag from crop_id or 00_set_context.json.")

    v1_crop_id = v2_crop_id_to_v1(v2_crop_id)
    slot_index = int(segment_index) - 1

    # Page and step from the match manifest entry.
    match_entries = _load_match_manifest()
    match_entry = _find_match_entry(match_entries, v2_crop_id, segment_index)
    page = int((match_entry or {}).get("page") or (m.group(2) if m else 0))
    step_raw = (match_entry or {}).get("step")
    step = int(step_raw) if step_raw is not None else (int(m.group(3)) if m and m.group(3).isdigit() else 0)

    # Qty from match entry qty_numbers, then 07_qty_ocr_map, then None.
    qty_numbers: list[int] = list((match_entry or {}).get("qty_numbers") or [])
    qty_ocr = _load_qty_ocr_map().get(v1_crop_id, {})
    if not qty_numbers:
        qty_numbers = list(qty_ocr.get("qty_numbers") or [])
    qty_texts: list[str] = list(qty_ocr.get("qty_text") or [])

    resolved_qty = qty
    resolved_qty_text: Optional[str] = None
    if slot_index < len(qty_numbers):
        if resolved_qty is None:
            resolved_qty = qty_numbers[slot_index]
        if slot_index < len(qty_texts):
            resolved_qty_text = qty_texts[slot_index]
        else:
            resolved_qty_text = f"{resolved_qty}x" if resolved_qty is not None else None

    return {
        "v2_crop_id": v2_crop_id,
        "v1_crop_id": v1_crop_id,
        "set_num": set_num,
        "bag": bag,
        "page": page,
        "step": step,
        "segment_index": segment_index,
        "v1_slot_index": slot_index,
        "qty": resolved_qty,
        "qty_text": resolved_qty_text,
        "part_num": part_num,
        "color_id": color_id,
        "color_name": color_name,
        "target_file": str(_label_path(set_num, bag)),
    }


def print_preview(preview: dict) -> None:
    w = 18
    print("--- DRY RUN (no files written) ---")
    print(f"  {'v2_crop_id':<{w}} {preview['v2_crop_id']}")
    print(f"  {'v1_crop_id':<{w}} {preview['v1_crop_id']}")
    print(f"  {'set_num':<{w}} {preview['set_num']}")
    print(f"  {'bag':<{w}} {preview['bag']}")
    print(f"  {'page':<{w}} {preview['page']}")
    print(f"  {'step':<{w}} {preview['step']}")
    print(f"  {'segment_index':<{w}} {preview['segment_index']}")
    print(f"  {'v1_slot_index':<{w}} {preview['v1_slot_index']}")
    print(f"  {'qty':<{w}} {preview['qty']}")
    print(f"  {'qty_text':<{w}} {preview['qty_text']}")
    print(f"  {'part_num':<{w}} {preview['part_num']}")
    print(f"  {'color_id':<{w}} {preview['color_id']}")
    print(f"  {'color_name':<{w}} {preview['color_name'] or '(not stored)'}")
    print(f"  {'target_file':<{w}} {preview['target_file']}")
    print("--- end dry run ---")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write a confirmed V2 match into debug/training_labels/ (V1 format)."
    )
    p.add_argument("--crop-id", required=True,
                   help="V2 crop_id, e.g. 'bag_01_page_012_step_12_002'")
    p.add_argument("--segment-index", required=True, type=int,
                   help="1-based segment index from V2 manifests")
    p.add_argument("--accept-top", action="store_true",
                   help="Use the top candidate from 09_match_manifest automatically")
    p.add_argument("--part-num",
                   help="LEGO part number (required unless --accept-top)")
    p.add_argument("--color-id", type=int,
                   help="BrickLink color id (required unless --accept-top)")
    p.add_argument("--qty", type=int, default=None,
                   help="Quantity override (optional)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written; do not modify any file")
    return p.parse_args()


def _resolve_part_from_args(args: argparse.Namespace) -> tuple[str, int, str]:
    """Return (part_num, color_id, color_name) from args, loading manifest if --accept-top."""
    part_num = args.part_num
    color_id = args.color_id
    color_name = ""

    if args.accept_top:
        entries = _load_match_manifest()
        entry = _find_match_entry(entries, args.crop_id, args.segment_index)
        if not entry:
            print(
                f"ERROR: No entry in 09_match_manifest for crop_id={args.crop_id!r} "
                f"segment_index={args.segment_index}",
                file=sys.stderr,
            )
            sys.exit(1)
        candidates = entry.get("top_candidates") or []
        if not candidates:
            print(
                f"ERROR: No top_candidates in match entry for {args.crop_id!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        top = candidates[0]
        part_num = str(top.get("part_num") or "").strip()
        color_id = int(top.get("color_id") or 0)
        color_name = str(top.get("color_name") or "")

    if not part_num:
        print("ERROR: --part-num is required (or use --accept-top)", file=sys.stderr)
        sys.exit(1)
    if color_id is None:
        print("ERROR: --color-id is required (or use --accept-top)", file=sys.stderr)
        sys.exit(1)

    return part_num, color_id, color_name


def main() -> None:
    args = _parse_args()
    part_num, color_id, color_name = _resolve_part_from_args(args)

    if args.dry_run:
        preview = resolve_preview(
            v2_crop_id=args.crop_id,
            segment_index=args.segment_index,
            part_num=part_num,
            color_id=color_id,
            color_name=color_name,
            qty=args.qty,
        )
        print_preview(preview)
        return

    crop_record = v2_accept_match(
        v2_crop_id=args.crop_id,
        segment_index=args.segment_index,
        part_num=part_num,
        color_id=color_id,
        qty=args.qty,
    )

    v1_id = v2_crop_id_to_v1(args.crop_id)
    print(f"Saved: crop_id={v1_id!r} slot_index={args.segment_index - 1} "
          f"part_num={part_num!r} color_id={color_id}")
    print(json.dumps(crop_record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
