from __future__ import annotations

import logging
from pathlib import Path

import fitz

from .models import PageData
from .utils import ensure_dir


LOG = logging.getLogger("lego_reader.pdf_reader")


def _render_page(page: fitz.Page, destination: Path) -> Path:
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pix.save(destination.as_posix())
    return destination


def read_pdf_pages(pdf_path: Path, debug: bool, debug_dir: Path) -> list[PageData]:
    document = fitz.open(pdf_path)
    pages: list[PageData] = []

    render_dir = ensure_dir(debug_dir / "pages") if debug else debug_dir / "pages"

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        text = page.get_text("text").strip()
        image_path = None
        if debug:
            image_path = _render_page(page, render_dir / f"page_{page_index + 1:03d}.png")
        pages.append(
            PageData(
                page_index=page_index,
                page_number=page_index + 1,
                text=text,
                width=float(page.rect.width),
                height=float(page.rect.height),
                word_count=len(page.get_text("words")),
                drawing_count=len(page.get_drawings()),
                image_count=len(page.get_images(full=True)),
                image_path=image_path,
            )
        )

    LOG.info("Loaded %s pages from %s", len(pages), pdf_path.name)
    return pages
