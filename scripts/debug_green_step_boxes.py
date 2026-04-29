#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean.services import debug_service, step_detector_service  # noqa: E402


def _numeric_box_summary(token: dict) -> dict:
    return {
        "text": str(token.get("text", "")),
        "value": int(token.get("value", 0) or 0),
        "x": int(token.get("x", 0) or 0),
        "y": int(token.get("y", 0) or 0),
        "w": int(token.get("w", 0) or 0),
        "h": int(token.get("h", 0) or 0),
        "area": int(token.get("area", 0) or 0),
        "score": float(token.get("score", 0.0) or 0.0),
    }


def analyze_page(set_num: str, page: int, save_overlays: bool, overlay_dir: Path) -> dict:
    image_path = debug_service.resolve_page_image_path(set_num, page)
    if image_path is None:
        return {
            "page": int(page),
            "image_found": False,
            "green_box_count": 0,
            "detected_numbers": [],
            "green_boxes": [],
            "overlay_path": None,
            "note": "page image not found",
        }

    result = step_detector_service.detect_steps(set_num, page)
    numeric_tokens = result.get("numeric_tokens", []) or []
    green_boxes = [_numeric_box_summary(token) for token in numeric_tokens]
    overlay_path = None

    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_bytes = step_detector_service.build_step_overlay(set_num, page)
        overlay_path = overlay_dir / f"page_{int(page):03d}.png"
        overlay_path.write_bytes(overlay_bytes)

    return {
        "page": int(page),
        "image_found": True,
        "image_path": str(image_path),
        "green_box_count": len(green_boxes),
        "detected_numbers": [box["value"] for box in green_boxes],
        "green_boxes": green_boxes,
        "page_number_tokens": result.get("page_number_tokens", []),
        "main_step": result.get("main_step", {}),
        "main_steps": result.get("main_steps", []),
        "sub_steps": result.get("sub_steps", []),
        "step_candidate_count": len(result.get("step_candidates", []) or []),
        "overlay_path": str(overlay_path) if overlay_path else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump green OCR/step-number boxes for rendered instruction pages.")
    parser.add_argument("--set-num", default="70618")
    parser.add_argument("--start-page", type=int, default=7)
    parser.add_argument("--end-page", type=int, default=38)
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()

    set_num = str(args.set_num)
    start_page = int(args.start_page)
    end_page = int(args.end_page)

    debug_root = PROJECT_ROOT / "debug" / set_num
    overlay_dir = debug_root / "green_step_boxes"
    output_path = debug_root / f"green_step_boxes_{start_page}_{end_page}.json"

    pages = []
    for page in range(start_page, end_page + 1):
        row = analyze_page(set_num, page, save_overlays=not args.no_overlays, overlay_dir=overlay_dir)
        pages.append(row)
        print(
            json.dumps(
                {
                    "page": row["page"],
                    "green_box_count": row["green_box_count"],
                    "detected_numbers": row["detected_numbers"],
                    "green_boxes": [
                        {k: box[k] for k in ("text", "value", "x", "y", "w", "h")}
                        for box in row["green_boxes"]
                    ],
                }
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "set_num": set_num,
                "page_range": [start_page, end_page],
                "overlay_dir": str(overlay_dir) if not args.no_overlays else None,
                "pages": pages,
            },
            indent=2,
        )
    )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
