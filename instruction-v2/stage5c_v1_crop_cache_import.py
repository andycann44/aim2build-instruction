import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from paths import INDEXES_DIR, ROOT_DIR


PROJECT_ROOT = ROOT_DIR.parent
V1_CROP_CACHE_DIR = PROJECT_ROOT / "debug" / "crop_cache"
OUT_PATH = INDEXES_DIR / "05c_v1_crop_cache_import.json"
SET_NUM = "70618"
BAGS = (1, 2)


def _parse_crop_index(crop_id: str) -> Optional[int]:
    match = re.search(r"_c(\d+)$", str(crop_id or "").strip())
    return int(match.group(1)) if match else None


def _load_bag_cache(set_num: str, bag: int) -> List[Dict[str, Any]]:
    path = V1_CROP_CACHE_DIR / f"{set_num}_bag{int(bag)}.json"
    if not path.exists():
        raise FileNotFoundError(f"V1 crop cache not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected list crop cache: {path}")

    entries: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        crop_id = str(item.get("crop_id") or "").strip()
        if not crop_id:
            continue
        entries.append(
            {
                "set_num": str(set_num),
                "bag": int(bag),
                "page": int(item.get("page") or 0),
                "step_number": int(item.get("step") or 0),
                "crop_index": _parse_crop_index(crop_id),
                "crop_id": crop_id,
                "stem": crop_id,
                "crop_box": item.get("crop_box"),
                "crop_box_format": item.get("crop_box_format") or "xywh",
                "crop_image_path": item.get("crop_image_path"),
                "qty_text": list(item.get("qty_text") or []),
                "qty_numbers": list(item.get("qty_numbers") or []),
                "v1_source": item.get("source"),
                "source": "v1_crop_cache",
            }
        )
    return entries


def build_manifest() -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    input_files: List[str] = []
    for bag in BAGS:
        input_path = V1_CROP_CACHE_DIR / f"{SET_NUM}_bag{int(bag)}.json"
        input_files.append(str(input_path))
        entries.extend(_load_bag_cache(SET_NUM, int(bag)))

    entries.sort(
        key=lambda item: (
            int(item.get("bag") or 0),
            int(item.get("page") or 0),
            int(item.get("step_number") or 0),
            int(item.get("crop_index") or 0),
            str(item.get("crop_id") or ""),
        )
    )

    return {
        "stage": "05c",
        "name": "v1_crop_cache_import",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "set_num": SET_NUM,
        "source": "v1_crop_cache",
        "read_only": True,
        "input_files": input_files,
        "entry_count": len(entries),
        "entries": entries,
    }


def main() -> int:
    payload = build_manifest()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(str(OUT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
