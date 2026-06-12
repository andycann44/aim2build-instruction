import argparse
import hashlib
import json
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from paths import INDEXES_DIR, PAGES_DIR, PDFS_DIR, ROOT_DIR, ensure_phase1_dirs

try:
    from lego_reader.pdf_reader import read_pdf_pages
except ModuleNotFoundError:
    read_pdf_pages = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_pdf_into_store(source_pdf: Path) -> Path:
    if not source_pdf.exists():
        raise RuntimeError(f"PDF not found: {source_pdf}")

    candidate = PDFS_DIR / source_pdf.name
    if not candidate.exists():
        shutil.copy2(source_pdf, candidate)
        return candidate

    if _sha256(candidate) == _sha256(source_pdf):
        return candidate

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    deduped = PDFS_DIR / f"{source_pdf.stem}_{stamp}{source_pdf.suffix.lower()}"
    shutil.copy2(source_pdf, deduped)
    return deduped


def _render_pdf_pages_with_node(saved_pdf: Path, run_pages_root: Path) -> List[Path]:
    node_bin = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
    node_modules = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")
    renderer = ROOT_DIR / "render_pdf_pages.mjs"

    if not node_bin.exists() or not node_modules.exists():
        raise RuntimeError("No PDF renderer is available. Missing bundled Node runtime/dependencies.")

    result = subprocess.run(
        [
            str(node_bin),
            str(renderer),
            "--pdf",
            str(saved_pdf),
            "--out-dir",
            str(run_pages_root),
            "--node-modules",
            str(node_modules),
            "--scale",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    json.loads(result.stdout)
    rendered = sorted(run_pages_root.glob("page_*.png"))
    if not rendered:
        raise RuntimeError(f"No rendered pages were produced for {saved_pdf.name}")
    return rendered


def _render_pdf_pages(saved_pdf: Path, run_id: str) -> List[Path]:
    run_pages_root = PAGES_DIR / run_id
    if run_pages_root.exists():
        shutil.rmtree(run_pages_root)
    run_pages_root.mkdir(parents=True, exist_ok=True)

    if read_pdf_pages is None:
        return _render_pdf_pages_with_node(saved_pdf=saved_pdf, run_pages_root=run_pages_root)

    render_dir = run_pages_root / "_render"
    render_dir.mkdir(parents=True, exist_ok=True)

    read_pdf_pages(pdf_path=saved_pdf, debug=True, debug_dir=render_dir)

    rendered_pages_dir = render_dir / "pages"
    if not rendered_pages_dir.exists():
        rendered_pages_dir = render_dir

    rendered = sorted(rendered_pages_dir.glob("page_*.png"))
    if not rendered:
        raise RuntimeError(f"No rendered pages were produced for {saved_pdf.name}")

    for page_path in rendered:
        target = run_pages_root / page_path.name
        if target.exists():
            target.unlink()
        shutil.move(str(page_path), str(target))

    shutil.rmtree(render_dir, ignore_errors=True)
    return sorted(run_pages_root.glob("page_*.png"))


def _write_manifests(
    run_id: str,
    source_pdf: Path,
    saved_pdf: Path,
    page_files: List[Path],
) -> Dict[str, Path]:
    page_entries: List[Dict[str, Any]] = []
    for idx, page_path in enumerate(page_files, start=1):
        page_entries.append(
            {
                "page": int(idx),
                "file_name": page_path.name,
                "image_path": str(page_path.relative_to(ROOT_DIR)),
                "sha256": _sha256(page_path),
            }
        )

    created_at_utc = _utc_now_iso()
    pdf_payload = {
        "stage": 1,
        "name": "pdf_render",
        "created_at_utc": created_at_utc,
        "run_id": str(run_id),
        "pdf": {
            "source_path": str(source_pdf.resolve()),
            "saved_path": str(saved_pdf.relative_to(ROOT_DIR)),
            "file_name": saved_pdf.name,
            "sha256": _sha256(saved_pdf),
        },
    }
    page_payload = {
        "stage": 1,
        "name": "page_index",
        "created_at_utc": created_at_utc,
        "run_id": str(run_id),
        "pdf_manifest": "indexes/01_pdf_manifest.json",
        "page_count": len(page_entries),
        "pages": page_entries,
    }

    pdf_manifest_path = INDEXES_DIR / "01_pdf_manifest.json"
    page_index_path = INDEXES_DIR / "02_page_index.json"
    pdf_manifest_path.write_text(json.dumps(pdf_payload, indent=2) + "\n", encoding="utf-8")
    page_index_path.write_text(json.dumps(page_payload, indent=2) + "\n", encoding="utf-8")
    return {
        "pdf_manifest": pdf_manifest_path,
        "page_index": page_index_path,
    }


def run_phase1(pdf_path: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    ensure_phase1_dirs()
    source_pdf = Path(pdf_path).expanduser().resolve()
    saved_pdf = _copy_pdf_into_store(source_pdf)

    if run_id:
        resolved_run_id = str(run_id)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        resolved_run_id = f"{saved_pdf.stem}_{stamp}"

    page_files = _render_pdf_pages(saved_pdf=saved_pdf, run_id=resolved_run_id)
    manifest_paths = _write_manifests(
        run_id=resolved_run_id,
        source_pdf=source_pdf,
        saved_pdf=saved_pdf,
        page_files=page_files,
    )

    return {
        "ok": True,
        "phase": "phase1",
        "run_id": resolved_run_id,
        "saved_pdf": str(saved_pdf.relative_to(ROOT_DIR)),
        "pages_dir": str((PAGES_DIR / resolved_run_id).relative_to(ROOT_DIR)),
        "page_count": len(page_files),
        "pdf_manifest": str(manifest_paths["pdf_manifest"].relative_to(ROOT_DIR)),
        "page_index": str(manifest_paths["page_index"].relative_to(ROOT_DIR)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="instruction-v2 phase1 runner")
    parser.add_argument("--pdf", required=True, help="Path to source instruction PDF")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id for instruction-v2/pages/<run-id>/",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = run_phase1(pdf_path=args.pdf, run_id=args.run_id)
    print(json.dumps(result, indent=2))
