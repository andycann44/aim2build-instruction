import hashlib
import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from paths import ROOT_DIR


SEGMENTATION_PATH = ROOT_DIR / "indexes" / "08_part_segmentation_map.json"
OUT_DIR = ROOT_DIR / "debug" / "part_segmentation_audit"
OUT_PATH = OUT_DIR / "contact_sheet.html"
AUDIT_SIZE = 10


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


def deterministic_key(item: dict[str, Any]) -> str:
    seed = "|".join(
        str(item.get(key, ""))
        for key in ("bag", "page", "step", "crop_id", "segment_index")
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def pick_audit_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    highest = sorted(entries, key=lambda item: item.get("confidence") or 0, reverse=True)[
        :AUDIT_SIZE
    ]
    lowest = sorted(entries, key=lambda item: item.get("confidence") or 0)[:AUDIT_SIZE]
    randomish = sorted(entries, key=deterministic_key)[:AUDIT_SIZE]

    picked: list[dict[str, Any]] = []
    for bucket, bucket_entries in (
        ("highest", highest),
        ("lowest", lowest),
        ("random", randomish),
    ):
        for item in bucket_entries:
            picked.append({**item, "reason_bucket": bucket})
    return picked


def row_html(item: dict[str, Any]) -> str:
    bucket = item.get("reason_bucket")
    confidence = item.get("confidence")
    return f"""
    <tr class="bucket-{esc(bucket)}">
      <td class="bucket">{esc(bucket)}</td>
      <td>
        <img src="{file_url(item.get("cutout_path"))}" alt="cutout {esc(item.get("crop_id"))}" />
        <div class="path">{esc(item.get("cutout_path"))}</div>
      </td>
      <td>
        <img src="{file_url(item.get("mask_path"))}" alt="mask {esc(item.get("crop_id"))}" />
        <div class="path">{esc(item.get("mask_path"))}</div>
      </td>
      <td>
        <img src="{file_url(item.get("overlay_path"))}" alt="overlay {esc(item.get("crop_id"))}" />
        <div class="path">{esc(item.get("overlay_path"))}</div>
      </td>
      <td>
        <div><strong>bag</strong> {esc(item.get("bag"))}</div>
        <div><strong>page</strong> {esc(item.get("page"))}</div>
        <div><strong>step</strong> {esc(item.get("step"))}</div>
        <div><strong>crop</strong> {esc(item.get("crop_id"))}</div>
        <div><strong>segment</strong> {esc(item.get("segment_index"))}</div>
        <div><strong>confidence</strong> {esc(confidence)}</div>
      </td>
    </tr>
    """


def build_contact_sheet() -> Path:
    if not SEGMENTATION_PATH.exists():
        raise RuntimeError(f"Missing segmentation manifest: {SEGMENTATION_PATH}")

    payload = json.loads(SEGMENTATION_PATH.read_text(encoding="utf-8"))
    entries = payload.get("entries") or []
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("08_part_segmentation_map.json has no entries")

    audit_entries = pick_audit_entries(entries)
    rows = "\n".join(row_html(item) for item in audit_entries)
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Instruction V2 Part Segmentation Audit</title>
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
      width: 140px;
      max-height: 140px;
      object-fit: contain;
      border: 1px solid #cbd5e1;
      background: white;
    }}
    .path {{
      max-width: 260px;
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
  </style>
</head>
<body>
  <h1>Instruction V2 Part Segmentation Audit</h1>
  <div class="meta">
    Source: indexes/08_part_segmentation_map.json · total segments: {esc(len(entries))} · audit rows: {esc(len(audit_entries))}
  </div>
  <table>
    <thead>
      <tr>
        <th>Bucket</th>
        <th>Cutout</th>
        <th>Mask</th>
        <th>Overlay</th>
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
