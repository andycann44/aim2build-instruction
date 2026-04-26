from pathlib import Path

from lego_reader.downloader import DownloadError, SetNotFoundError, download_set_pdfs
from lego_reader.pdf_reader import read_pdf_pages
from lego_reader.utils import normalize_set_num

try:
    import app as legacy_app
except Exception:
    legacy_app = None


PROJECT_ROOT = Path("/Users/olly/aim2build-instruction")
INSTRUCTIONS_ROOT = PROJECT_ROOT / "instructions"
DEBUG_ROOT = PROJECT_ROOT / "debug"


def _require_legacy_attr(name: str):
    if legacy_app is None:
        raise RuntimeError("legacy app import failed")
    value = getattr(legacy_app, name, None)
    if value is None:
        raise RuntimeError(f"legacy app missing required attribute: {name}")
    return value


def switch_active_pages_dir(pages_dir: Path):
    switch_fn = _require_legacy_attr("_switch_detector_pages_dir")
    switch_fn(Path(pages_dir))


def load_set_from_number(raw_set_num: str):
    """
    Clean wrapper around the old working flow:
    normalize -> download -> render pages -> switch active pages dir
    """
    try:
        normalized_set_num = normalize_set_num(raw_set_num)
    except Exception as e:
        return {
            "ok": False,
            "stage": "normalize_set_num",
            "error": str(e),
            "raw_set_num": raw_set_num,
        }

    try:
        download_result = download_set_pdfs(
            set_num=normalized_set_num,
            instructions_dir=INSTRUCTIONS_ROOT,
        )
    except SetNotFoundError as e:
        return {
            "ok": False,
            "stage": "download_set_pdfs",
            "error": str(e),
            "set_num": normalized_set_num,
        }
    except DownloadError as e:
        return {
            "ok": False,
            "stage": "download_set_pdfs",
            "error": str(e),
            "set_num": normalized_set_num,
        }
    except Exception as e:
        return {
            "ok": False,
            "stage": "download_set_pdfs",
            "error": str(e),
            "set_num": normalized_set_num,
        }

    pdfs = list(getattr(download_result, "pdfs", []) or [])
    if not pdfs:
        return {
            "ok": False,
            "stage": "download_set_pdfs",
            "error": "no PDFs returned",
            "set_num": normalized_set_num,
        }

    downloaded_pdf = pdfs[0]
    pdf_path = Path(downloaded_pdf.local_path)

    per_pdf_debug_dir = DEBUG_ROOT / normalized_set_num / pdf_path.stem
    pages_dir = per_pdf_debug_dir / "pages"

    rendered_now = False
    page_count = 0

    try:
        existing_pages = sorted(pages_dir.glob("page_*.png")) if pages_dir.exists() else []
        if not existing_pages:
            read_pdf_pages(
                pdf_path=pdf_path,
                debug=True,
                debug_dir=per_pdf_debug_dir,
            )
            rendered_now = True

        final_pages = sorted(pages_dir.glob("page_*.png")) if pages_dir.exists() else []
        page_count = len(final_pages)

        if page_count == 0:
            return {
                "ok": False,
                "stage": "read_pdf_pages",
                "error": "pages were not rendered",
                "set_num": normalized_set_num,
                "pdf_path": str(pdf_path),
                "pages_dir": str(pages_dir),
            }

    except Exception as e:
        return {
            "ok": False,
            "stage": "read_pdf_pages",
            "error": str(e),
            "set_num": normalized_set_num,
            "pdf_path": str(pdf_path),
            "pages_dir": str(pages_dir),
        }

    try:
        switch_active_pages_dir(pages_dir)
    except Exception as e:
        return {
            "ok": False,
            "stage": "_switch_detector_pages_dir",
            "error": str(e),
            "set_num": normalized_set_num,
            "pages_dir": str(pages_dir),
        }

    return {
        "ok": True,
        "stage": "ready",
        "set_num": normalized_set_num,
        "pdf_path": str(pdf_path),
        "pdf_stem": pdf_path.stem,
        "pages_dir": str(pages_dir),
        "page_count": page_count,
        "rendered_now": rendered_now,
    }