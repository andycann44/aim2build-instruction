import html
import json
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

from paths import ROOT_DIR


GAP_REVIEW_PATH = ROOT_DIR / "indexes" / "03b_bag_gap_review.json"
PAGE_INDEX_PATH = ROOT_DIR / "indexes" / "02_page_index.json"
OUT_DIR = ROOT_DIR / "debug" / "bag_gap_review"
OUT_PATH = OUT_DIR / "contact_sheet.html"
MAX_WINDOW_THUMBNAILS = 12


def esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def file_url(path_value: Any) -> str:
    if not isinstance(path_value, str) or not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return "file://" + quote(str(path.resolve()))


def sample_pages(start_page: int, end_page: int, max_count: int = MAX_WINDOW_THUMBNAILS) -> List[int]:
    pages = list(range(start_page, end_page + 1))
    if len(pages) <= max_count:
        return pages
    indexes = []
    for index in range(max_count):
        raw = round(index * (len(pages) - 1) / max(1, max_count - 1))
        indexes.append(raw)
    return [pages[index] for index in sorted(set(indexes))]


def page_card(page: int, page_by_number: Dict[int, Dict[str, Any]], candidate_by_page: Dict[int, Dict[str, Any]]) -> str:
    page_entry = page_by_number.get(page, {})
    candidate = candidate_by_page.get(page)
    candidate_badge = ""
    if candidate:
        candidate_badge = f"<div class=\"badge\">candidate score {esc(candidate.get('score'))}</div>"
    image_path = page_entry.get("image_path")
    return f"""
    <a class="page-card {'candidate' if candidate else ''}" href="{file_url(image_path)}" target="_blank">
      <img src="{file_url(image_path)}" alt="page {page}" loading="lazy" />
      <div class="page-label">Page {page}</div>
      {candidate_badge}
    </a>
    """


def candidate_pages_html(candidates: List[Dict[str, Any]], page_by_number: Dict[int, Dict[str, Any]]) -> str:
    if not candidates:
        return '<p class="muted">No detected candidate pages inside this gap window.</p>'
    candidate_by_page = {int(item.get("page")): item for item in candidates if item.get("page") is not None}
    return '<div class="thumb-grid">' + "\n".join(
        page_card(page, page_by_number, candidate_by_page)
        for page in sorted(candidate_by_page)
    ) + "</div>"


def sampled_window_pages_html(window: Dict[str, Any], page_by_number: Dict[int, Dict[str, Any]], candidates: List[Dict[str, Any]]) -> str:
    bounds = window.get("window") or {}
    start_page = int(bounds.get("start_page") or 0)
    end_page = int(bounds.get("end_page") or 0)
    if not start_page or not end_page:
        return '<p class="muted">No window pages available.</p>'
    candidate_by_page = {int(item.get("page")): item for item in candidates if item.get("page") is not None}
    return '<div class="thumb-grid">' + "\n".join(
        page_card(page, page_by_number, candidate_by_page)
        for page in sample_pages(start_page, end_page)
    ) + "</div>"


def known_starts_text(gap: Dict[str, Any]) -> str:
    between = gap.get("between_known_bags") or []
    left_bag = between[0] if len(between) > 0 else ""
    right_bag = between[1] if len(between) > 1 else ""
    return (
        f"Bag {esc(left_bag)} starts page {esc(gap.get('previous_known_start_page'))}; "
        f"Bag {esc(right_bag)} starts page {esc(gap.get('next_known_start_page'))}"
    )


def gap_section(gap: Dict[str, Any], page_by_number: Dict[int, Dict[str, Any]]) -> str:
    candidates = list(gap.get("candidate_pages") or [])
    window = gap.get("window") or {}
    notes = "".join(f"<li>{esc(note)}</li>" for note in gap.get("notes") or [])
    return f"""
    <section class="gap-card status-{esc(gap.get('status'))}">
      <header>
        <div>
          <h2>{esc(gap.get('window_id'))}</h2>
          <p class="meta">{known_starts_text(gap)}</p>
        </div>
        <div class="status-box">
          <div>Status: <strong>{esc(gap.get('status'))}</strong></div>
          <div>Confidence: <strong>{esc(gap.get('confidence'))}</strong></div>
        </div>
      </header>
      <div class="summary">
        Window pages {esc(window.get('start_page'))}-{esc(window.get('end_page'))}
        ({esc(window.get('page_span'))} pages)
      </div>
      <h3>Candidate Pages Inside Gap</h3>
      {candidate_pages_html(candidates, page_by_number)}
      <h3>Window Page Scan</h3>
      {sampled_window_pages_html(gap, page_by_number, candidates)}
      <ul class="notes">{notes}</ul>
    </section>
    """


def build_contact_sheet() -> Path:
    if not GAP_REVIEW_PATH.exists():
        raise RuntimeError(f"Missing gap review manifest: {GAP_REVIEW_PATH}")
    if not PAGE_INDEX_PATH.exists():
        raise RuntimeError(f"Missing page index manifest: {PAGE_INDEX_PATH}")

    gap_review = json.loads(GAP_REVIEW_PATH.read_text(encoding="utf-8"))
    page_index = json.loads(PAGE_INDEX_PATH.read_text(encoding="utf-8"))
    page_by_number = {int(item["page"]): item for item in page_index.get("pages", [])}
    pending_windows = [gap for gap in gap_review.get("gap_windows", []) if gap.get("status") == "pending"]

    sections = "\n".join(gap_section(gap, page_by_number) for gap in pending_windows)
    if not sections:
        sections = '<section class="gap-card"><p>No pending gap windows.</p></section>'

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Instruction V2 Bag Gap Review</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f8;
      color: #1f2933;
    }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
    }}
    h2 {{
      margin: 0;
      font-size: 18px;
    }}
    h3 {{
      margin: 18px 0 10px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #52606d;
    }}
    .meta {{
      margin: 6px 0 0;
      color: #52606d;
    }}
    .top-meta {{
      margin-bottom: 18px;
      color: #52606d;
      font-size: 14px;
    }}
    .gap-card {{
      margin-bottom: 24px;
      padding: 18px;
      background: white;
      border: 1px solid #d9e2ec;
      border-left: 5px solid #d97706;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
    }}
    .status-box {{
      min-width: 180px;
      padding: 10px;
      background: #fffbea;
      border: 1px solid #f0c36d;
      font-size: 14px;
    }}
    .summary {{
      margin-top: 14px;
      font-weight: 700;
    }}
    .thumb-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
      gap: 12px;
    }}
    .page-card {{
      display: block;
      padding: 8px;
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: inherit;
      text-decoration: none;
    }}
    .page-card.candidate {{
      border-color: #0f7b42;
      background: #f0fff4;
    }}
    img {{
      display: block;
      width: 100%;
      height: 150px;
      object-fit: contain;
      background: white;
      border: 1px solid #e4e7eb;
    }}
    .page-label {{
      margin-top: 6px;
      font-weight: 700;
      font-size: 13px;
    }}
    .badge {{
      margin-top: 4px;
      color: #0f7b42;
      font-size: 12px;
    }}
    .muted {{
      color: #66788a;
      font-style: italic;
    }}
    .notes {{
      margin: 16px 0 0;
      padding-left: 20px;
      color: #52606d;
    }}
  </style>
</head>
<body>
  <h1>Instruction V2 Bag Gap Review</h1>
  <div class="top-meta">
    Source: indexes/03b_bag_gap_review.json · pending windows: {esc(len(pending_windows))}
  </div>
  {sections}
</body>
</html>
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html_doc, encoding="utf-8")
    return OUT_PATH


def main() -> None:
    print(build_contact_sheet())


if __name__ == "__main__":
    main()
