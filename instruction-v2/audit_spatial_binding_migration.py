"""Read-only migration audit: reviewed slots affected by spatial qty-island rebinding.

Generates per-crop contact sheets for human truth review before any migration.
Does not modify labels, training data, or production code paths.
"""

from __future__ import annotations

import json
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from PIL import Image, ImageDraw, ImageFont

from paths import ROOT_DIR

PROJECT_ROOT = ROOT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean.services.bag_review_service import _load_crop_cache, load_review_state  # noqa: E402
from clean.services.full_crop_mask_paths import (  # noqa: E402
    ensure_island_slot_cutout_path,
    ensure_island_slot_mask_path,
    find_full_mask_stem,
    master_island_overlay_path,
    raw_master_mask_path,
)

BINDING_AUDIT = PROJECT_ROOT / "debug" / "spatial_qty_island_binding_audit" / "binding_audit.json"
OUT_ROOT = PROJECT_ROOT / "debug" / "spatial_qty_island_binding_audit" / "migration"
SET_NUM = "70618"
CLIP_URL = "http://127.0.0.1:8000/debug/buildability-clip-suggest"


def _font(size: int = 13) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _clip_top5(bag: int, crop_id: str, slot_index: int, island_cutout_path: str) -> List[Dict[str, Any]]:
    payload = json.dumps(
        {
            "set_num": SET_NUM,
            "bag": int(bag),
            "crop_id": crop_id,
            "slot_index": int(slot_index),
            "island_slot_cutout_path": island_cutout_path,
            "step_masked_path": "",
            "part_cutout_path": "",
        }
    ).encode()
    req = urllib.request.Request(CLIP_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        return [{"error": str(exc)}]
    rows = list(data.get("ranked_candidates") or [])[:5]
    return rows


def _format_top5(rows: List[Dict[str, Any]], saved_part: str) -> List[str]:
    lines: List[str] = []
    if not rows:
        return ["  (no candidates)"]
    if rows and rows[0].get("error"):
        return [f"  CLIP error: {rows[0]['error']}"]
    for index, row in enumerate(rows, start=1):
        part = f"{row.get('part_num')}/{row.get('color_id')}"
        score = row.get("score") or row.get("clip_score") or "?"
        mark = " <-- SAVED" if saved_part and part.replace(" ", "") in saved_part.replace(" ", "") else ""
        lines.append(f"  {index}. {part} score={score}{mark}")
    return lines


def _text_panel(title: str, lines: List[str], width: int = 420) -> Image.Image:
    font = _font(12)
    line_h = 15
    height = 28 + line_h * (len(lines) + 1)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((8, 6), title, fill=(0, 0, 0), font=_font(13))
    y = 26
    for line in lines:
        for chunk in textwrap.wrap(line, width=52) or [""]:
            draw.text((8, y), chunk, fill=(20, 20, 20), font=font)
            y += line_h
    return img


def _load_crop_rgb(cache_crop: Dict[str, Any]) -> Optional[Image.Image]:
    box = list(cache_crop.get("crop_box") or [])
    page_path = str(cache_crop.get("crop_image_path") or "")
    if len(box) < 4 or not page_path:
        return None
    page = cv2.imread(page_path)
    if page is None:
        return None
    cx, cy, cw, ch = [int(v) for v in box[:4]]
    crop = page[cy : cy + ch, cx : cx + cw]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def _overlay_islands(
    crop_rgb: Image.Image,
    stem: str,
    current_label: int,
    proposed_label: int,
) -> Image.Image:
    overlay_path = master_island_overlay_path(stem)
    if overlay_path.is_file():
        base = Image.open(overlay_path).convert("RGB")
    else:
        base = crop_rgb.copy()
    draw = ImageDraw.Draw(base)
    font = _font(14)
    # legend only — full overlay already colored
    draw.rectangle([4, 4, 180, 44], fill=(255, 255, 255))
    draw.text((8, 8), f"current I{current_label}", fill=(255, 0, 0), font=font)
    draw.text((8, 24), f"proposed I{proposed_label}", fill=(0, 128, 0), font=font)
    return base


def _compose_slot_sheet(
    crop_id: str,
    slot_index: int,
    saved_label: str,
    current_label: int,
    proposed_label: int,
    current_cutout: Optional[Path],
    proposed_cutout: Optional[Path],
    current_top5: List[Dict[str, Any]],
    proposed_top5: List[Dict[str, Any]],
    crop_rgb: Image.Image,
    overlay: Image.Image,
) -> Image.Image:
    panels: List[tuple[str, Image.Image]] = [
        ("original crop", crop_rgb),
        ("island overlay", overlay),
    ]
    if current_cutout and current_cutout.is_file():
        panels.append((f"CURRENT island {current_label}", Image.open(current_cutout).convert("RGBA")))
    if proposed_cutout and proposed_cutout.is_file():
        panels.append((f"PROPOSED island {proposed_label}", Image.open(proposed_cutout).convert("RGBA")))

    cur_lines = [f"slot {slot_index}", f"saved: {saved_label}", "CURRENT top5:"] + _format_top5(current_top5, saved_label)
    prop_lines = ["PROPOSED top5:"] + _format_top5(proposed_top5, saved_label)
    panels.append(("CLIP current", _text_panel("Current island CLIP", cur_lines, 400)))
    panels.append(("CLIP proposed", _text_panel("Proposed island CLIP", prop_lines, 400)))

    row_h = max(p.size[1] for _, p in panels) + 24
    row_w = sum(p.size[0] for _, p in panels) + 8 * len(panels)
    sheet = Image.new("RGB", (row_w, row_h + 28), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), f"{crop_id} slot {slot_index} | saved {saved_label} | I{current_label} -> I{proposed_label}", fill=(0, 0, 0), font=_font(13))
    x = 4
    y = 28
    for label, im in panels:
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (240, 240, 240))
            bg.paste(im, mask=im.split()[3])
            im = bg
        header = Image.new("RGB", (im.size[0], 20), (230, 230, 230))
        ImageDraw.Draw(header).text((2, 3), label, fill=(0, 0, 0), font=_font(11))
        cell = Image.new("RGB", (im.size[0], 20 + im.size[1]))
        cell.paste(header, (0, 0))
        cell.paste(im, (0, 20))
        sheet.paste(cell, (x, y))
        x += cell.size[0] + 4
    return sheet


def run_migration_audit() -> Dict[str, Any]:
    binding = json.loads(BINDING_AUDIT.read_text(encoding="utf-8"))
    affected_crops = [
        crop
        for crop in binding.get("crops") or []
        if int(crop.get("reviewed_slots_affected") or 0) > 0
    ]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    migration_rows: List[Dict[str, Any]] = []

    for crop_row in sorted(affected_crops, key=lambda item: (int(item["bag"]), str(item["crop_id"]))):
        bag = int(crop_row["bag"])
        crop_id = str(crop_row["crop_id"])
        cache = next(c for c in _load_crop_cache(SET_NUM, bag) if c["crop_id"] == crop_id)
        stem = find_full_mask_stem(SET_NUM, bag, crop_id)
        if not stem:
            continue
        crop_rgb = _load_crop_rgb(cache)
        if crop_rgb is None:
            continue
        crop_dir = OUT_ROOT / f"bag{bag}_{crop_id}"
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_rgb.save(crop_dir / "00_original_crop.png")

        slot_rows: List[Dict[str, Any]] = []
        reviewed_changes = [c for c in crop_row.get("changes") or [] if c.get("reviewed")]

        for change in reviewed_changes:
            slot_index = int(change["slot_index"])
            current_label = int(change["current_island"])
            proposed_label = int(change["proposed_island"])
            saved_label = str(change.get("review_kind") or "")

            current_cutout = ensure_island_slot_cutout_path(
                stem=stem,
                slot_index=slot_index,
                crop_image_path=str(cache["crop_image_path"]),
                crop_box=list(cache["crop_box"]),
                island_label=current_label,
            )
            proposed_cutout = ensure_island_slot_cutout_path(
                stem=stem,
                slot_index=slot_index,
                crop_image_path=str(cache["crop_image_path"]),
                crop_box=list(cache["crop_box"]),
                island_label=proposed_label,
            )
            ensure_island_slot_mask_path(
                stem=stem,
                slot_index=slot_index,
                crop_image_path=str(cache["crop_image_path"]),
                crop_box=list(cache["crop_box"]),
                island_label=current_label,
            )
            ensure_island_slot_mask_path(
                stem=stem,
                slot_index=slot_index,
                crop_image_path=str(cache["crop_image_path"]),
                crop_box=list(cache["crop_box"]),
                island_label=proposed_label,
            )

            current_top5 = _clip_top5(bag, crop_id, slot_index, str(current_cutout or ""))
            proposed_top5 = _clip_top5(bag, crop_id, slot_index, str(proposed_cutout or ""))

            overlay = _overlay_islands(crop_rgb, stem, current_label, proposed_label)
            sheet = _compose_slot_sheet(
                crop_id,
                slot_index,
                saved_label,
                current_label,
                proposed_label,
                current_cutout,
                proposed_cutout,
                current_top5,
                proposed_top5,
                crop_rgb,
                overlay,
            )
            sheet_path = crop_dir / f"slot{slot_index}_migration.png"
            sheet.save(sheet_path)

            saved_part = saved_label.replace("label:", "") if saved_label.startswith("label:") else saved_label

            def _rank_of(part_key: str, rows: List[Dict[str, Any]]) -> Optional[int]:
                for index, row in enumerate(rows[:5], start=1):
                    key = f"{row.get('part_num')}/{row.get('color_id')}"
                    if key == part_key:
                        return index
                return None

            slot_rows.append(
                {
                    "slot_index": slot_index,
                    "saved_label": saved_part,
                    "current_island": current_label,
                    "proposed_island": proposed_label,
                    "current_cutout": str(current_cutout or ""),
                    "proposed_cutout": str(proposed_cutout or ""),
                    "current_top5": [
                        {
                            "part_num": r.get("part_num"),
                            "color_id": r.get("color_id"),
                            "score": r.get("score") or r.get("clip_score"),
                        }
                        for r in current_top5[:5]
                        if not r.get("error")
                    ],
                    "proposed_top5": [
                        {
                            "part_num": r.get("part_num"),
                            "color_id": r.get("color_id"),
                            "score": r.get("score") or r.get("clip_score"),
                        }
                        for r in proposed_top5[:5]
                        if not r.get("error")
                    ],
                    "saved_rank_current": _rank_of(saved_part, current_top5),
                    "saved_rank_proposed": _rank_of(saved_part, proposed_top5),
                    "contact_sheet": str(sheet_path.relative_to(PROJECT_ROOT)),
                }
            )

        migration_rows.append(
            {
                "bag": bag,
                "crop_id": crop_id,
                "reviewed_slots_affected": len(slot_rows),
                "slots": slot_rows,
                "crop_dir": str(crop_dir.relative_to(PROJECT_ROOT)),
            }
        )

    payload = {
        "set_num": SET_NUM,
        "audit": "spatial_qty_island_binding_migration",
        "purpose": "Human truth review before label migration",
        "affected_crops": len(migration_rows),
        "affected_reviewed_slots": sum(len(c["slots"]) for c in migration_rows),
        "crops": migration_rows,
    }
    (OUT_ROOT / "migration_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run_migration_audit()
    print(
        f"Migration audit: {result['affected_crops']} crops, "
        f"{result['affected_reviewed_slots']} reviewed slots"
    )
    for crop in result["crops"]:
        print(f"  bag{crop['bag']} {crop['crop_id']}: {crop['reviewed_slots_affected']} slots -> {crop['crop_dir']}")
