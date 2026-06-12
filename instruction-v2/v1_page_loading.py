from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path("/Users/olly/aim2build-instruction")
DEBUG_ROOT = PROJECT_ROOT / "debug"
ROOT_DIR = Path(__file__).resolve().parent
INDEXES_DIR = ROOT_DIR / "indexes"


def _find_latest_pages_dir_for_set(set_num: str):
    """
    Reuse the old idea:
    look under debug/<set_num>/*/pages and pick the most recently updated one.
    """
    set_root = DEBUG_ROOT / str(set_num)
    if not set_root.exists():
        return None

    candidates = []
    for pages_dir in set_root.glob("*/pages"):
        try:
            page_files = list(pages_dir.glob("page_*.png"))
        except OSError:
            continue
        if not page_files:
            continue
        latest_mtime = max((p.stat().st_mtime_ns for p in page_files), default=0)
        candidates.append((latest_mtime, pages_dir))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def resolve_page_image_path(set_num: str, page: int):
    """
    Prefer a clean path lookup by set_num.
    If later needed, this can be swapped to call old legacy helpers.
    """
    pages_dir = _find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        return None

    path = pages_dir / f"page_{int(page):03d}.png"
    if not path.exists():
        return None
    return path


def v2_page_image_path(page: int) -> Optional[Path]:
    index_path = INDEXES_DIR / "02_page_index.json"
    if not index_path.exists():
        return None

    import json

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in payload.get("pages", []) or []:
        if int(entry.get("page") or 0) == int(page):
            image_path = entry.get("image_path")
            if not image_path:
                return None
            path = Path(image_path)
            return path if path.is_absolute() else ROOT_DIR / path
    return None


def compare_v1_v2_page_loading(set_num: str, page: int) -> Dict[str, Any]:
    v1_path = resolve_page_image_path(set_num, page)
    v2_path = v2_page_image_path(page)
    return {
        "set_num": str(set_num),
        "page": int(page),
        "v1_path": str(v1_path) if v1_path else None,
        "v1_exists": bool(v1_path and v1_path.exists()),
        "v2_path": str(v2_path) if v2_path else None,
        "v2_exists": bool(v2_path and v2_path.exists()),
        "same_path": bool(v1_path and v2_path and v1_path.resolve() == v2_path.resolve()),
    }
