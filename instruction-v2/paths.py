from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
PDFS_DIR = ROOT_DIR / "pdfs"
PAGES_DIR = ROOT_DIR / "pages"
INDEXES_DIR = ROOT_DIR / "indexes"


def ensure_phase1_dirs() -> None:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
