from pathlib import Path

try:
    import app as legacy_app
except Exception:
    legacy_app = None


PROJECT_ROOT = Path("/Users/olly/aim2build-instruction")
DEBUG_ROOT = PROJECT_ROOT / "debug"


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
