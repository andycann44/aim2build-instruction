from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .models import DownloadResult, DownloadedPdf
from .utils import ensure_dir


LOG = logging.getLogger("lego_reader.downloader")
REQUEST_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (compatible; lego-reader/1.0)"
SITE_CANDIDATES = (
    "https://www.lego.com/en-gb/service/buildinginstructions/{set_num}",
    "https://www.lego.com/en-us/service/buildinginstructions/{set_num}",
    "https://www.lego.com/en-gb/service/building-instructions/{set_num}",
    "https://www.lego.com/en-us/service/building-instructions/{set_num}",
)
PDF_URL_RE = re.compile(r'https?://[^"\'\s>]+\.pdf(?:\?[^"\'\s>]*)?', re.IGNORECASE)
PDF_PATH_RE = re.compile(r'/cdn/[^"\'\s>]+\.pdf(?:\?[^"\'\s>]*)?', re.IGNORECASE)


class DownloadError(RuntimeError):
    """Raised when the LEGO instruction page or PDFs cannot be downloaded."""


@dataclass
class SetNotFoundError(DownloadError):
    set_num: str
    source_url: str

    def __str__(self) -> str:
        return f"Official LEGO building instructions were not found for set {self.set_num}."


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _looks_like_instruction_page(response: requests.Response, set_num: str) -> bool:
    if not response.ok:
        return False
    text = response.text
    markers = (
        "Select the instructions you want",
        "Building Instructions - Download",
        f"#{set_num}",
        f"{set_num} pieces",
        set_num,
    )
    return any(marker in text for marker in markers)


def _find_best_source_page(session: requests.Session, set_num: str) -> tuple[str, str]:
    last_url = SITE_CANDIDATES[0].format(set_num=set_num)
    for template in SITE_CANDIDATES:
        url = template.format(set_num=set_num)
        LOG.info("Trying source page %s", url)
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as error:
            LOG.warning("Failed to fetch %s: %s", url, error)
            last_url = url
            continue
        last_url = response.url
        LOG.info("Fetched %s -> %s (status=%s)", url, response.url, response.status_code)
        if _looks_like_instruction_page(response, set_num):
            return response.text, response.url
    raise SetNotFoundError(set_num=set_num, source_url=last_url)


def _extract_pdf_links_from_anchors(html: str, base_url: str) -> tuple[str, list[tuple[str, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "LEGO Building Instructions"
    pdf_entries: list[tuple[str, str]] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = urljoin(base_url, anchor["href"])
        if ".pdf" not in href.lower():
            continue
        parsed = urlparse(href)
        if "lego.com" not in (parsed.netloc or ""):
            continue
        normalized = href.split("?", 1)[0]
        if normalized in seen:
            continue
        seen.add(normalized)
        link_text = anchor.get_text(" ", strip=True) or Path(parsed.path).name
        pdf_entries.append((link_text, normalized))

    return title, pdf_entries


def _extract_pdf_links_from_text(html: str, base_url: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    for match in PDF_URL_RE.findall(html):
        normalized = match.split("?", 1)[0]
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append((Path(urlparse(normalized).path).name, normalized))

    for match in PDF_PATH_RE.findall(html):
        resolved = urljoin(base_url, match)
        normalized = resolved.split("?", 1)[0]
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append((Path(urlparse(normalized).path).name, normalized))

    return candidates


def _extract_pdf_links(html: str, base_url: str) -> tuple[str, list[tuple[str, str]]]:
    title, anchor_entries = _extract_pdf_links_from_anchors(html, base_url)
    LOG.info("Found %s PDF links from anchors", len(anchor_entries))
    if anchor_entries:
        return title, anchor_entries

    fallback_entries = _extract_pdf_links_from_text(html, base_url)
    LOG.info("Found %s PDF links from raw HTML fallback", len(fallback_entries))
    return title, fallback_entries


def _download_file(session: requests.Session, source_url: str, destination: Path) -> None:
    try:
        with session.get(source_url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not source_url.lower().endswith(".pdf"):
                raise DownloadError(
                    f"Expected PDF from {source_url}, got {content_type or 'unknown content type'}."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        file_obj.write(chunk)
    except requests.RequestException as error:
        raise DownloadError(f"Failed to download PDF {source_url}: {error}") from error


def download_set_pdfs(set_num: str, instructions_dir: Path) -> DownloadResult:
    session = _session()
    html, source_url = _find_best_source_page(session, set_num)
    LOG.info("Using source page %s", source_url)
    page_title, entries = _extract_pdf_links(html, source_url)

    if not entries:
        raise SetNotFoundError(set_num=set_num, source_url=source_url)

    LOG.info("Preparing to download %s PDF(s) for set %s", len(entries), set_num)
    set_dir = ensure_dir(instructions_dir / set_num)
    downloaded: list[DownloadedPdf] = []
    for index, (title, pdf_url) in enumerate(entries, start=1):
        local_path = set_dir / f"{set_num}_{index:02d}.pdf"
        if not local_path.exists() or local_path.stat().st_size == 0:
            LOG.info("Downloading PDF %s -> %s", pdf_url, local_path)
            _download_file(session, pdf_url, local_path)
        else:
            LOG.info("Reusing existing PDF %s", local_path)
        downloaded.append(
            DownloadedPdf(
                index=index,
                title=title,
                source_url=pdf_url,
                local_path=local_path,
            )
        )

    return DownloadResult(
        set_num=set_num,
        source_url=source_url,
        page_title=page_title,
        pdfs=downloaded,
    )
