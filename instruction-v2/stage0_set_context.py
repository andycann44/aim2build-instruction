import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from paths import INDEXES_DIR


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clean.services.instruction_buildability_source import load_instruction_set_parts  # noqa: E402


OUT_PATH = INDEXES_DIR / "00_set_context.json"


def _part_sort_key(part: Dict[str, Any]) -> tuple:
    return (str(part.get("part_num") or ""), int(part.get("color_id") or 0))


def _normalize_parts(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for part in sorted(parts, key=_part_sort_key):
        normalized.append(
            {
                "part_num": str(part.get("part_num") or ""),
                "color_id": part.get("color_id"),
                "color_name": part.get("color_name"),
                "rgb": part.get("rgb"),
                "element_id": part.get("element_id"),
                "set_required_qty": int(part.get("set_required_qty") or part.get("qty") or 0),
                "qty": int(part.get("qty") or part.get("set_required_qty") or 0),
                "img_url": part.get("img_url"),
                "needs_image": bool(part.get("needs_image")),
                "shape_key": part.get("shape_key"),
                "color_key": part.get("color_key"),
            }
        )
    return normalized


def build_set_context(set_num: str) -> Dict[str, Any]:
    catalog_payload = load_instruction_set_parts(set_num)
    parts = _normalize_parts(catalog_payload.get("parts") or [])
    return {
        "stage": 0,
        "name": "catalog_set_parts",
        "set_num": catalog_payload.get("set_num") or set_num,
        "inventory_id": catalog_payload.get("inventory_id"),
        "source": {
            "function": "clean.services.instruction_buildability_source.load_instruction_set_parts",
        },
        "counts": {
            "part_color_rows": len(parts),
            "total_required_qty": sum(int(part.get("set_required_qty") or 0) for part in parts),
            "missing_image_rows": sum(1 for part in parts if part.get("needs_image")),
        },
        "parts": parts,
    }


def write_set_context(set_num: str) -> Path:
    payload = build_set_context(set_num)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return OUT_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="instruction-v2 stage 0: catalog set parts")
    parser.add_argument("--set-num", required=True, help="LEGO set number, for example 70618")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_path = write_set_context(args.set_num)
    print(str(out_path))


if __name__ == "__main__":
    main()
