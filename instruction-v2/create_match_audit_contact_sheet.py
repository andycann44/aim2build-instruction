import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from paths import ROOT_DIR


AUDIT_PATH = ROOT_DIR / "indexes" / "10_match_audit.json"
OUT_DIR = ROOT_DIR / "debug" / "match_audit"
OUT_PATH = OUT_DIR / "contact_sheet.html"


def file_url(path_value: Any) -> str:
    if not isinstance(path_value, str) or not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return "file://" + quote(str(path.resolve()))


def esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def row_html(item: dict[str, Any]) -> str:
    part_num = item.get("top_candidate_part_num")
    color_id = item.get("top_candidate_color_id")
    score = item.get("top_candidate_score")
    bucket = item.get("reason_bucket")
    catalog_url = item.get("catalog_img_url") or ""
    manual = item.get("manual_config") or {}
    force_part_num = manual.get("force_part_num")
    force_color_id = manual.get("force_color_id")
    status = manual.get("status") or "pending"
    notes = manual.get("notes") or ""
    rejected_parts = manual.get("reject_candidate_part_nums") or []
    rejected_colors = manual.get("reject_candidate_color_ids") or []
    override_text = "none"
    if force_part_num or force_color_id is not None:
        override_text = f"{force_part_num or ''} / {force_color_id if force_color_id is not None else ''}"

    return f"""
    <tr class="bucket-{esc(bucket)} status-{esc(status)}">
      <td class="bucket">{esc(bucket)}</td>
      <td>
        <img src="{file_url(item.get("cutout_path"))}" alt="cutout {esc(item.get("crop_id"))}" />
        <div class="path">{esc(item.get("cutout_path"))}</div>
      </td>
      <td>
        <img src="{file_url(item.get("overlay_path"))}" alt="overlay {esc(item.get("crop_id"))}" />
        <div class="path">{esc(item.get("overlay_path"))}</div>
      </td>
      <td>
        <img src="{esc(catalog_url)}" alt="catalog {esc(part_num)} {esc(color_id)}" />
        <div class="path">{esc(catalog_url)}</div>
      </td>
      <td>
        <div class="label">Current top candidate</div>
        <div><strong>{esc(part_num)}</strong> / {esc(color_id)}</div>
        <div>score: <strong>{esc(score)}</strong></div>
        <div>expected in set: {esc(item.get("expected_in_set"))}</div>
        <hr />
        <div class="label">Manual config</div>
        <div>override: <strong>{esc(override_text)}</strong></div>
        <div>status: <strong>{esc(status)}</strong></div>
        <div>reject parts: {esc(", ".join(str(value) for value in rejected_parts))}</div>
        <div>reject colors: {esc(", ".join(str(value) for value in rejected_colors))}</div>
        <div class="notes">notes: {esc(notes)}</div>
      </td>
      <td>
        <div>bag {esc(item.get("bag"))}, page {esc(item.get("page"))}, step {esc(item.get("step"))}</div>
        <div>{esc(item.get("crop_id"))}</div>
        <div>segment {esc(item.get("segment_index"))}</div>
      </td>
    </tr>
    """


def build_contact_sheet() -> Path:
    if not AUDIT_PATH.exists():
        raise RuntimeError(f"Missing match audit manifest: {AUDIT_PATH}")

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    entries = audit.get("entries") or []
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("10_match_audit.json has no entries")

    rows = "\n".join(row_html(item) for item in entries)
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Instruction V2 Match Audit</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f6f7;
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
    .meta {{
      margin-bottom: 18px;
      color: #52606d;
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid #d9e2ec;
    }}
    th, td {{
      padding: 10px;
      border-bottom: 1px solid #d9e2ec;
      vertical-align: top;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #e9eef3;
      z-index: 1;
    }}
    img {{
      display: block;
      width: 120px;
      max-height: 120px;
      object-fit: contain;
      border: 1px solid #cbd5e1;
      background: white;
    }}
    .path {{
      max-width: 240px;
      margin-top: 6px;
      overflow-wrap: anywhere;
      color: #52606d;
      font-size: 11px;
      line-height: 1.25;
    }}
    .bucket {{
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 12px;
    }}
    .bucket-highest .bucket {{ color: #0f7b42; }}
    .bucket-lowest .bucket {{ color: #a61b1b; }}
    .bucket-random .bucket {{ color: #375a7f; }}
    .label {{
      margin-top: 2px;
      color: #52606d;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    hr {{
      border: 0;
      border-top: 1px solid #d9e2ec;
      margin: 8px 0;
    }}
    .notes {{
      margin-top: 4px;
      max-width: 260px;
      overflow-wrap: anywhere;
    }}
    .status-accepted {{
      background: #f0fff4;
    }}
    .status-rejected {{
      background: #fff5f5;
    }}
    .status-needs_ai_check {{
      background: #fffbea;
    }}
  </style>
</head>
<body>
  <h1>Instruction V2 Match Audit</h1>
  <div class="meta">
    Source: indexes/10_match_audit.json · entries: {esc(len(entries))}
  </div>
  <table>
    <thead>
      <tr>
        <th>Bucket</th>
        <th>Cutout</th>
        <th>Overlay</th>
        <th>Catalog</th>
        <th>Candidate + Manual Config</th>
        <th>Segment</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(content, encoding="utf-8")
    return OUT_PATH


def main() -> None:
    print(build_contact_sheet())


if __name__ == "__main__":
    main()
