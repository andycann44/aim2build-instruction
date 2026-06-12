"""Read-only audit of accepted step-map detections: dimensions and cluster stability."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paths import INDEXES_DIR, ROOT_DIR


OUTLIER_BAND_RATIO = 0.30


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bag_page_range(bag_map: Dict[str, Any], bag: int) -> Tuple[int, int]:
    for entry in bag_map.get("bags", []):
        if int(entry.get("bag", 0)) == bag:
            return int(entry["start_page"]), int(entry["end_page"])
    raise ValueError(f"bag {bag} not found in bag map")


def _median_band(values: List[float], ratio: float = OUTLIER_BAND_RATIO) -> Dict[str, float]:
    if not values:
        return {"median": 0.0, "low": 0.0, "high": 0.0}
    med = float(statistics.median(values))
    return {"median": med, "low": med * (1.0 - ratio), "high": med * (1.0 + ratio)}


def _is_outside_band(value: float, band: Dict[str, float]) -> bool:
    return value < band["low"] or value > band["high"]


def collect_accepted_steps(step_map: Dict[str, Any], bag: int, start_page: int, end_page: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in step_map.get("steps", []):
        if int(step.get("bag", 0)) != bag:
            continue
        page = int(step.get("page", 0))
        if page < start_page or page > end_page:
            continue
        if step.get("rejection_reason"):
            continue
        box = step.get("step_box") or {}
        signals = step.get("signals") or {}
        rows.append(
            {
                "page": page,
                "value": step.get("step_number"),
                "bbox": {
                    "x": int(box.get("x", 0)),
                    "y": int(box.get("y", 0)),
                    "w": int(box.get("w", 0)),
                    "h": int(box.get("h", 0)),
                },
                "width": int(box.get("w", 0)),
                "height": int(box.get("h", 0)),
                "component_count": int(signals.get("components") or 0),
                "source": step.get("source") or "step_map_primary",
                "step_index": step.get("step_index"),
            }
        )
    rows.sort(key=lambda row: (row["page"], row.get("step_index") or 0, row["value"] or 0))
    return rows


def _step_identity(page: int, value: Any, bbox: Dict[str, int]) -> Tuple[int, Any, int, int, int, int]:
    return (page, value, bbox["x"], bbox["y"], bbox["w"], bbox["h"])


def collect_rejected_audit_candidates(
    step_map: Dict[str, Any],
    bag: int,
    start_page: int,
    end_page: int,
    accepted: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Full-page audit candidates that never entered the accepted steps array."""
    accepted_keys = {
        _step_identity(row["page"], row.get("value"), row["bbox"]) for row in accepted
    }
    rows: List[Dict[str, Any]] = []
    for page_entry in step_map.get("pages", []):
        if int(page_entry.get("bag", 0)) != bag:
            continue
        page = int(page_entry.get("page", 0))
        if page < start_page or page > end_page:
            continue
        for candidate in page_entry.get("full_page_audit_candidates") or []:
            box = candidate.get("step_box") or {}
            signals = candidate.get("signals") or {}
            bbox = {
                "x": int(box.get("x", 0)),
                "y": int(box.get("y", 0)),
                "w": int(box.get("w", 0)),
                "h": int(box.get("h", 0)),
            }
            identity = _step_identity(page, candidate.get("step_number"), bbox)
            if identity in accepted_keys:
                continue
            rows.append(
                {
                    "page": page,
                    "value": candidate.get("step_number"),
                    "bbox": bbox,
                    "width": bbox["w"],
                    "height": bbox["h"],
                    "component_count": int(signals.get("components") or 0),
                    "source": candidate.get("source") or "full_page_step_audit_candidate",
                }
            )
    rows.sort(key=lambda row: (row["page"], row["value"] or 0))
    return rows


def annotate_outliers(rows: List[Dict[str, Any]], bands: Dict[str, Dict[str, float]]) -> None:
    for row in rows:
        width_outlier = _is_outside_band(float(row["width"]), bands["width"])
        height_outlier = _is_outside_band(float(row["height"]), bands["height"])
        component_outlier = _is_outside_band(float(row["component_count"]), bands["components"])
        row["outlier"] = {
            "width": width_outlier,
            "height": height_outlier,
            "component_count": component_outlier,
            "any": width_outlier or height_outlier or component_outlier,
        }


def audit_bag_step_number_statistics(bag: int, set_num: str = "70618") -> Dict[str, Any]:
    bag_map = _load_json(INDEXES_DIR / "04_bag_map.json")
    step_map = _load_json(INDEXES_DIR / "05_step_map.json")
    start_page, end_page = _bag_page_range(bag_map, bag)

    accepted = collect_accepted_steps(step_map, bag, start_page, end_page)
    rejected_audit = collect_rejected_audit_candidates(
        step_map, bag, start_page, end_page, accepted
    )

    heights = [float(row["height"]) for row in accepted]
    widths = [float(row["width"]) for row in accepted]
    components = [float(row["component_count"]) for row in accepted]

    bands = {
        "height": _median_band(heights),
        "width": _median_band(widths),
        "components": _median_band(components),
    }
    annotate_outliers(accepted, bands)

    outlier_rows = [row for row in accepted if row["outlier"]["any"]]
    in_band_rows = [row for row in accepted if not row["outlier"]["any"]]
    narrow_width_rows = [
        row for row in accepted if float(row["width"]) < bands["width"]["median"] - 10.0
    ]

    return {
        "set_num": set_num,
        "bag": bag,
        "page_range": {"start": start_page, "end": end_page},
        "definition": {
            "accepted": "05_step_map.json steps[] with matching bag, in-range page, rejection_reason null",
            "excluded": "full_page_audit_candidates on pages[] that never entered steps[]",
            "outlier_rule": f"dimension outside median ±{int(OUTLIER_BAND_RATIO * 100)}%",
        },
        "summary": {
            "accepted_count": len(accepted),
            "in_band_count": len(in_band_rows),
            "outlier_count": len(outlier_rows),
            "rejected_audit_candidate_count": len(rejected_audit),
            "narrow_width_watch_count": len(narrow_width_rows),
            "median_height_px": bands["height"]["median"],
            "median_width_px": bands["width"]["median"],
            "median_component_count": bands["components"]["median"],
            "height_band_px": {"low": bands["height"]["low"], "high": bands["height"]["high"]},
            "width_band_px": {"low": bands["width"]["low"], "high": bands["width"]["high"]},
            "component_band": {"low": bands["components"]["low"], "high": bands["components"]["high"]},
        },
        "bands": bands,
        "accepted_rows": accepted,
        "outlier_rows": outlier_rows,
        "narrow_width_watch_rows": narrow_width_rows,
        "rejected_audit_candidates": rejected_audit,
    }


def _write_tsv(path: Path, rows: List[Dict[str, Any]], include_outlier: bool) -> None:
    header = [
        "page",
        "value",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "height",
        "width",
        "component_count",
        "source",
    ]
    if include_outlier:
        header.append("outlier")
    lines = ["\t".join(header)]
    for row in rows:
        bbox = row["bbox"]
        cells = [
            str(row["page"]),
            str(row.get("value") or ""),
            str(bbox["x"]),
            str(bbox["y"]),
            str(bbox["w"]),
            str(bbox["h"]),
            str(row["height"]),
            str(row["width"]),
            str(row["component_count"]),
            str(row.get("source") or ""),
        ]
        if include_outlier:
            outlier = row.get("outlier") or {}
            flags = []
            if outlier.get("width"):
                flags.append("width")
            if outlier.get("height"):
                flags.append("height")
            if outlier.get("component_count"):
                flags.append("components")
            cells.append("|".join(flags) if flags else "")
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    summary = payload["summary"]
    bands = payload["bands"]
    lines = [
        f"# Bag {payload['bag']} step-number statistics audit",
        "",
        f"Set `{payload['set_num']}`. Pages {payload['page_range']['start']}–{payload['page_range']['end']}.",
        "",
        "## Definition",
        "",
        f"- **Accepted:** {payload['definition']['accepted']}",
        f"- **Excluded from this table:** {payload['definition']['excluded']}",
        f"- **Outlier rule:** {payload['definition']['outlier_rule']}",
        "",
        "## Median dimensions (accepted only)",
        "",
        f"| Metric | Median | Low band | High band |",
        f"| --- | ---: | ---: | ---: |",
        f"| Height (px) | {summary['median_height_px']:.1f} | {summary['height_band_px']['low']:.1f} | {summary['height_band_px']['high']:.1f} |",
        f"| Width (px) | {summary['median_width_px']:.1f} | {summary['width_band_px']['low']:.1f} | {summary['width_band_px']['high']:.1f} |",
        f"| Component count | {summary['median_component_count']:.1f} | {summary['component_band']['low']:.1f} | {summary['component_band']['high']:.1f} |",
        "",
        "## Cluster summary",
        "",
        f"- Accepted detections: **{summary['accepted_count']}**",
        f"- Within ±{int(OUTLIER_BAND_RATIO * 100)}% band on all dimensions: **{summary['in_band_count']}** ({100 * summary['in_band_count'] / max(summary['accepted_count'], 1):.1f}%)",
        f"- Outliers on at least one dimension: **{summary['outlier_count']}**",
        f"- Rejected full-page audit candidates (not accepted): **{summary['rejected_audit_candidate_count']}**",
        "",
        "## Accepted step detections",
        "",
        "| Page | Value | Bbox (x,y,w,h) | Height | Width | Components | Source | Outlier |",
        "| ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]

    for row in payload["accepted_rows"]:
        bbox = row["bbox"]
        outlier = row.get("outlier") or {}
        flags = []
        if outlier.get("width"):
            flags.append("width")
        if outlier.get("height"):
            flags.append("height")
        if outlier.get("component_count"):
            flags.append("components")
        flag_text = ", ".join(flags) if flags else "—"
        mark = "**" if outlier.get("any") else ""
        end = "**" if outlier.get("any") else ""
        lines.append(
            f"| {row['page']} | {row.get('value') or ''} | "
            f"({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}) | "
            f"{mark}{row['height']}{end} | {mark}{row['width']}{end} | "
            f"{mark}{row['component_count']}{end} | {row.get('source') or ''} | {flag_text} |"
        )

    narrow = payload.get("narrow_width_watch_rows") or []
    if narrow:
        lines.extend(
            [
                "",
                "## Narrow-width watch (accepted, width ≥10 px below median)",
                "",
                "Not ±30% outliers, but materially narrower than the 54 px cluster mode.",
                "",
                "| Page | Value | Bbox (x,y,w,h) | Width |",
                "| ---: | ---: | --- | ---: |",
            ]
        )
        for row in narrow:
            bbox = row["bbox"]
            lines.append(
                f"| {row['page']} | {row.get('value') or ''} | "
                f"({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}) | {row['width']} |"
            )

    if payload["rejected_audit_candidates"]:
        lines.extend(
            [
                "",
                "## Rejected audit candidates (contrast — not accepted)",
                "",
                "These never entered `steps[]`. Compare dimensions to the accepted cluster above.",
                "",
                "| Page | Value | Bbox (x,y,w,h) | Height | Width | Components |",
                "| ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in payload["rejected_audit_candidates"]:
            bbox = row["bbox"]
            lines.append(
                f"| {row['page']} | {row.get('value') or ''} | "
                f"({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}) | "
                f"{row['height']} | {row['width']} | {row['component_count']} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Valid LEGO step numbers in Bag 3 form a tight visual cluster: height ~35–37 px, "
            "width ~51–55 px, and component count 2. Narrow boxes (39–41 px) appear on a few "
            "two-digit steps but remain above the noise floor seen on page 48 audit candidates "
            "(14–28 px wide, 1 component). A pre-OCR size gate using median ±30% would retain "
            "most accepted steps while excluding typical OCR-noise regions.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(path: Path, payload: Dict[str, Any]) -> None:
    summary = payload["summary"]
    rows_html = []
    for row in payload["accepted_rows"]:
        bbox = row["bbox"]
        outlier = row.get("outlier") or {}
        cls = "outlier" if outlier.get("any") else ""
        flags = []
        if outlier.get("width"):
            flags.append("width")
        if outlier.get("height"):
            flags.append("height")
        if outlier.get("component_count"):
            flags.append("components")
        rows_html.append(
            f"<tr class='{cls}'>"
            f"<td>{row['page']}</td>"
            f"<td>{row.get('value') or ''}</td>"
            f"<td>({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']})</td>"
            f"<td>{row['height']}</td>"
            f"<td>{row['width']}</td>"
            f"<td>{row['component_count']}</td>"
            f"<td>{row.get('source') or ''}</td>"
            f"<td>{', '.join(flags) if flags else '—'}</td>"
            f"</tr>"
        )

    contrast_html = []
    for row in payload.get("rejected_audit_candidates") or []:
        bbox = row["bbox"]
        contrast_html.append(
            f"<tr class='noise'>"
            f"<td>{row['page']}</td>"
            f"<td>{row.get('value') or ''}</td>"
            f"<td>({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']})</td>"
            f"<td>{row['height']}</td>"
            f"<td>{row['width']}</td>"
            f"<td>{row['component_count']}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bag {payload['bag']} step-number statistics</title>
  <style>
    body {{ font: 14px/1.4 system-ui, sans-serif; margin: 24px; color: #111; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .stats {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fafafa; }}
    .card strong {{ display: block; font-size: 22px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f3f3; }}
    tr.outlier {{ background: #fff4cc; }}
    tr.noise {{ background: #fde8e8; }}
    .note {{ max-width: 900px; color: #333; }}
  </style>
</head>
<body>
  <h1>Bag {payload['bag']} step-number statistics audit</h1>
  <p>Set {payload['set_num']}, pages {payload['page_range']['start']}–{payload['page_range']['end']}.</p>
  <p class="note">Accepted = <code>05_step_map.json</code> <code>steps[]</code> with no rejection reason.
  Outliers = any dimension outside median ±{int(OUTLIER_BAND_RATIO * 100)}%.</p>

  <div class="stats">
    <div class="card">Median height<strong>{summary['median_height_px']:.0f}px</strong>
      band {summary['height_band_px']['low']:.1f}–{summary['height_band_px']['high']:.1f}</div>
    <div class="card">Median width<strong>{summary['median_width_px']:.0f}px</strong>
      band {summary['width_band_px']['low']:.1f}–{summary['width_band_px']['high']:.1f}</div>
    <div class="card">Median components<strong>{summary['median_component_count']:.0f}</strong>
      band {summary['component_band']['low']:.1f}–{summary['component_band']['high']:.1f}</div>
  </div>

  <p><strong>{summary['accepted_count']}</strong> accepted,
     <strong>{summary['in_band_count']}</strong> in band,
     <strong>{summary['outlier_count']}</strong> outliers.</p>

  <h2>Accepted step detections</h2>
  <table>
    <thead>
      <tr>
        <th>Page</th><th>Value</th><th>Bbox</th><th>Height</th><th>Width</th>
        <th>Components</th><th>Source</th><th>Outlier</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>

  <h2>Rejected audit candidates (contrast)</h2>
  <table>
    <thead>
      <tr><th>Page</th><th>Value</th><th>Bbox</th><th>Height</th><th>Width</th><th>Components</th></tr>
    </thead>
    <tbody>
      {''.join(contrast_html) if contrast_html else '<tr><td colspan="6">None</td></tr>'}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit accepted step-number visual cluster statistics.")
    parser.add_argument("--bag", type=int, default=3)
    parser.add_argument("--set-num", default="70618")
    args = parser.parse_args(argv)

    payload = audit_bag_step_number_statistics(bag=int(args.bag), set_num=str(args.set_num))
    out_dir = ROOT_DIR / "debug" / f"bag{args.bag}_step_number_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.set_num}_bag{args.bag}"
    json_path = out_dir / f"{prefix}_step_number_statistics.json"
    tsv_path = out_dir / f"{prefix}_step_number_statistics.tsv"
    md_path = ROOT_DIR / f"BAG{args.bag}_STEP_NUMBER_STATISTICS_AUDIT.md"
    html_path = out_dir / "report.html"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_tsv(tsv_path, payload["accepted_rows"], include_outlier=True)
    _write_markdown(md_path, payload)
    _write_html(html_path, payload)

    summary = payload["summary"]
    print(f"accepted={summary['accepted_count']} in_band={summary['in_band_count']} outliers={summary['outlier_count']}")
    print(f"median_height={summary['median_height_px']:.1f}px median_width={summary['median_width_px']:.1f}px")
    print(str(json_path))
    print(str(tsv_path))
    print(str(md_path))
    print(str(html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
