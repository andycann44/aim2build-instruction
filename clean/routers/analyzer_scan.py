from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from clean.services import analyzer_scan_service

router = APIRouter()


@router.get("/api/scan-set-with-analyzer")
def api_scan_set_with_analyzer(
    set_num: str = Query(...),
    include_all: bool = Query(False),
):
    return analyzer_scan_service.scan_set_with_analyzer(set_num, include_all=include_all)


@router.get("/debug/scan-set-with-analyzer", response_class=HTMLResponse)
def debug_scan_set_with_analyzer(
    set_num: str = Query(...),
    include_all: bool = Query(False),
):
    result = analyzer_scan_service.scan_set_with_analyzer(set_num, include_all=include_all)

    if not result.get("ok"):
        return HTMLResponse(
            f"""
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8" />
              <title>Analyzer Scan</title>
              <style>
                body {{ font-family: Arial, sans-serif; background: #f3f3f3; margin: 16px; }}
                .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; max-width: 900px; margin: 0 auto; }}
              </style>
            </head>
            <body>
              <div class="card">
                <h2>Analyzer Scan Error</h2>
                <p><strong>Set:</strong> {set_num}</p>
                <p><strong>Error:</strong> {result.get("error", "unknown error")}</p>
                <p><a href="/">Back to home</a></p>
              </div>
            </body>
            </html>
            """,
            status_code=400,
        )

    rows = sorted(result.get("rows", []), key=lambda item: int(item.get("page", 0) or 0))
    pages_dir = result.get("pages_dir", "")
    page_count = result.get("page_count", 0)
    shortlist_page_count = result.get("shortlist_page_count", 0)
    skipped_page_count = result.get("skipped_page_count", 0)
    analyzed_page_count = result.get("analyzed_page_count", 0)
    cache_hits = result.get("cache_hits", 0)
    row_count = result.get("row_count", 0)

    mode_label = "FULL" if include_all else "SHORTLIST"
    toggle_include_all = "false" if include_all else "true"
    toggle_label = "Switch to Full Scan" if not include_all else "Switch to Shortlist Scan"
    display_limit = 40
    hidden_count = 0
    if not include_all and len(rows) > display_limit:
        hidden_count = len(rows) - display_limit
        rows = rows[:display_limit]

    rows_html = []

    for row in rows:
        page = int(row.get("page", 0) or 0)
        summary = row.get("analyzer_summary", {}) or {}
        skipped = bool(row.get("skipped"))
        precheck_kind = row.get("precheck_kind", "other")
        skip_reason = row.get("skip_reason", "")
        confidence = float(summary.get("confidence", 0.0) or 0.0)
        bag_number = summary.get("bag_number")
        panel_found = bool(summary.get("panel_found"))
        shell_found = bool(summary.get("shell_found"))
        grey_bag_found = bool(summary.get("grey_bag_found"))
        number_found = bool(summary.get("number_found"))
        cache_hit = bool(row.get("cache_hit"))
        panel_source = summary.get("panel_source") or "-"

        if confidence >= 0.85:
            confidence_label = "HIGH"
            confidence_class = "high"
        elif confidence >= 0.60:
            confidence_label = "MED"
            confidence_class = "med"
        else:
            confidence_label = "LOW"
            confidence_class = "low"

        save_form = ""
        if not skipped:
            save_form = f"""
            <form method="post" action="/api/debug/save-bag-truth" class="save-form">
              <input type="number" name="bag_number" min="1" placeholder="Bag #" required>
              <input type="hidden" name="set_num" value="{set_num}">
              <input type="hidden" name="start_page" value="{page}">
              <input type="hidden" name="redirect_to" value="/debug/scan-set-with-analyzer?set_num={set_num}&include_all={'true' if include_all else 'false'}">
              <button type="submit">Save bag start</button>
            </form>
            """

        card_class = "row-card skipped" if skipped else f"row-card {confidence_class}"
        summary_text = " | ".join(
            [
                f"panel: {'yes' if panel_found else 'no'}",
                f"shell: {'yes' if shell_found else 'no'}",
                f"grey bag: {'yes' if grey_bag_found else 'no'}",
                f"number: {'yes' if number_found else 'no'}",
            ]
        )

        raw_analyze_link = f"/api/analyze-gap-page?set_num={set_num}&bag_number={bag_number if bag_number is not None else 1}&page={page}"
        image_link = f"/api/debug/page-image?set_num={set_num}&page={page}"

        rows_html.append(
            f"""
            <div class="{card_class}">
              <div class="thumb-col">
                <a href="{image_link}" target="_blank">
                  <img
                    class="thumb lazy-thumb"
                    src="data:image/gif;base64,R0lGODlhAQABAAAAACw="
                    data-src="{image_link}"
                    loading="lazy"
                    alt="Page {page}"
                  >
                </a>
              </div>
              <div class="meta-col">
                <div class="meta-top">
                  <div>
                    <h3>Page {page}</h3>
                    <div class="muted">Precheck: {precheck_kind}{(' | skipped: ' + skip_reason) if skip_reason else ''}</div>
                    <div class="muted">Panel source: {panel_source}</div>
                    <div class="muted">{summary_text}</div>
                  </div>
                  <div class="score-box {confidence_class}">
                    <div class="score-label">{confidence_label}</div>
                    <div class="score-value">{confidence:.2f}</div>
                    <div class="muted score-bag">bag: {bag_number if bag_number is not None else '-'}</div>
                  </div>
                </div>
                <div class="meta-actions">
                  <a class="btn" href="{raw_analyze_link}" target="_blank">Raw analyze</a>
                  <a class="btn" href="{image_link}" target="_blank">Open image</a>
                  <span class="muted">cache: {'yes' if cache_hit else 'no'}</span>
                </div>
                {save_form}
              </div>
            </div>
            """
        )

    show_more_block = ""
    if hidden_count:
        show_more_block = f"""
        <div class="card">
          <p><strong>{hidden_count}</strong> more rows are hidden in shortlist mode to keep this page responsive.</p>
          <a class="btn alt" href="/debug/scan-set-with-analyzer?set_num={set_num}&include_all=true">Show full scan</a>
        </div>
        """

    rows_block = "\n".join(rows_html) if rows_html else "<div class='card'><p>No rows to show.</p></div>"

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Analyzer Scan</title>
          <style>
            body {{ font-family: Arial, sans-serif; background: #f3f3f3; margin: 16px; color: #111; }}
            .wrap {{ max-width: 1200px; margin: 0 auto; }}
            .card {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin-bottom: 16px; }}
            .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }}
            .btn {{ display: inline-block; padding: 8px 12px; background: #222; color: #fff; text-decoration: none; border-radius: 6px; border: none; cursor: pointer; }}
            .btn.alt {{ background: #666; }}
            .stats {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; margin-top: 12px; }}
            .stat {{ background: #fafafa; border: 1px solid #e3e3e3; border-radius: 8px; padding: 10px; }}
            .stat .k {{ font-size: 12px; color: #666; }}
            .stat .v {{ font-size: 20px; font-weight: bold; margin-top: 4px; }}
            .row-card {{ display: grid; grid-template-columns: 240px 1fr; gap: 14px; background: #fff; border: 2px solid #ddd; border-radius: 10px; padding: 12px; margin-bottom: 14px; }}
            .row-card.high {{ border-color: #2e8b57; }}
            .row-card.med {{ border-color: #c79200; }}
            .row-card.low {{ border-color: #999; }}
            .row-card.skipped {{ opacity: 0.8; border-style: dashed; }}
            .thumb {{ width: 100%; max-width: 240px; border: 1px solid #ccc; border-radius: 6px; background: #fff; }}
            .lazy-thumb {{ min-height: 160px; object-fit: contain; }}
            .meta-top {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
            .meta-col h3 {{ margin: 0 0 8px 0; }}
            .muted {{ color: #666; font-size: 13px; margin-top: 4px; }}
            .score-box {{ min-width: 110px; text-align: center; border-radius: 8px; padding: 10px; color: #fff; }}
            .score-box.high {{ background: #2e8b57; }}
            .score-box.med {{ background: #c79200; }}
            .score-box.low {{ background: #666; }}
            .score-label {{ font-size: 12px; font-weight: bold; letter-spacing: 0.5px; }}
            .score-value {{ font-size: 28px; font-weight: bold; margin-top: 4px; }}
            .score-bag {{ color: rgba(255,255,255,0.9); }}
            .meta-actions {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 12px; }}
            .save-form {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }}
            .save-form input {{ padding: 8px; border: 1px solid #bbb; border-radius: 6px; width: 120px; }}
            .save-form button {{ padding: 8px 12px; border: none; border-radius: 6px; background: #005bbb; color: white; cursor: pointer; }}
            @media (max-width: 900px) {{
              .row-card {{ grid-template-columns: 1fr; }}
              .stats {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
              .thumb {{ max-width: 100%; }}
            }}
          </style>
        </head>
        <body>
          <div class="wrap">
            <div class="card">
              <h2>Analyzer Scan</h2>
              <p><strong>Set:</strong> {set_num}</p>
              <p><strong>Mode:</strong> {mode_label}</p>
              <p><strong>Pages dir:</strong> {pages_dir}</p>
              <div class="toolbar">
                <a class="btn" href="/">Back to home</a>
                <a class="btn alt" href="/debug/scan-set-with-analyzer?set_num={set_num}&include_all={toggle_include_all}">{toggle_label}</a>
                <a class="btn alt" href="/api/scan-set-with-analyzer?set_num={set_num}&include_all={'true' if include_all else 'false'}" target="_blank">Raw JSON</a>
              </div>
              <div class="stats">
                <div class="stat"><div class="k">page count</div><div class="v">{page_count}</div></div>
                <div class="stat"><div class="k">shortlist</div><div class="v">{shortlist_page_count}</div></div>
                <div class="stat"><div class="k">skipped</div><div class="v">{skipped_page_count}</div></div>
                <div class="stat"><div class="k">analyzed</div><div class="v">{analyzed_page_count}</div></div>
                <div class="stat"><div class="k">cache hits</div><div class="v">{cache_hits}</div></div>
                <div class="stat"><div class="k">shown</div><div class="v">{row_count}</div></div>
              </div>
            </div>
            {rows_block}
            {show_more_block}
          </div>
          <script>
            (() => {{
              const loadImage = (img) => {{
                const src = img.dataset.src;
                if (!src) return;
                img.src = src;
                img.removeAttribute("data-src");
              }};

              const images = Array.from(document.querySelectorAll("img.lazy-thumb[data-src]"));
              if (!("IntersectionObserver" in window)) {{
                images.forEach(loadImage);
                return;
              }}

              const observer = new IntersectionObserver((entries) => {{
                entries.forEach((entry) => {{
                  if (!entry.isIntersecting) return;
                  loadImage(entry.target);
                  observer.unobserve(entry.target);
                }});
              }}, {{
                rootMargin: "400px 0px",
                threshold: 0.01,
              }});

              images.forEach((img) => observer.observe(img));
            }})();
          </script>
        </body>
        </html>
        """
    )
