from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from clean.services import set_scan_service

router = APIRouter()


@router.get("/api/scan-set-candidates")
def api_scan_set_candidates(
    set_num: str = Query(...),
    include_all: bool = Query(True),
):
    return set_scan_service.scan_set_pages(set_num, include_all=include_all)


@router.get("/debug/scan-set-candidates", response_class=HTMLResponse)
def debug_scan_set_candidates(
    set_num: str = Query(...),
    include_all: bool = Query(True),
):
    result = set_scan_service.scan_set_pages(set_num, include_all=include_all)

    if not result.get("ok"):
        return HTMLResponse(
            f"""
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8" />
              <title>Scan set candidates</title>
              <style>
                body {{ font-family: Arial, sans-serif; margin: 16px; background: #f3f3f3; }}
                .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
              </style>
            </head>
            <body>
              <div class="card">
                <h2>Scan set candidates</h2>
                <p><strong>Set:</strong> {set_num}</p>
                <p><strong>Error:</strong> {result.get("error", "Unknown error")}</p>
              </div>
            </body>
            </html>
            """
        )

    rows_html = []
    for row in result["rows"]:
        page = row["page"]
        img_url = f"/api/debug/page-image?set_num={set_num}&page={page}"
        precheck_url = f"/api/page-precheck?set_num={set_num}&page={page}"

        rows_html.append(
            f"""
            <tr>
              <td>{page}</td>
              <td>{row.get("page_kind", "-")}</td>
              <td>{row.get("numeric_token_count", 0)}</td>
              <td>{row.get("word_count", 0)}</td>
              <td>{row.get("line_count", 0)}</td>
              <td>{row.get("bright_ratio", 0.0)}</td>
              <td>{row.get("edge_density", 0.0)}</td>
              <td>{row.get("large_box_count", 0)}</td>
              <td>{row.get("largest_box_area_ratio", 0.0)}</td>
              <td>
                <a href="{img_url}" target="_blank">View image</a>
                &nbsp;|&nbsp;
                <a href="{precheck_url}" target="_blank">Precheck JSON</a>
              </td>
            </tr>
            """
        )

    rows_block = "\n".join(rows_html) if rows_html else "<tr><td colspan='10'>No rows</td></tr>"

    include_all_toggle = (
        f"/debug/scan-set-candidates?set_num={set_num}&include_all=false"
        if include_all
        else f"/debug/scan-set-candidates?set_num={set_num}&include_all=true"
    )
    include_all_label = "Show only candidates/unknown" if include_all else "Show all pages"

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Scan set candidates</title>
          <style>
            body {{
              font-family: Arial, sans-serif;
              margin: 16px;
              background: #f3f3f3;
            }}
            .card {{
              background: #fff;
              border: 1px solid #ddd;
              border-radius: 8px;
              padding: 12px;
              margin-bottom: 16px;
            }}
            table {{
              width: 100%;
              border-collapse: collapse;
              background: #fff;
            }}
            th, td {{
              border: 1px solid #ddd;
              padding: 8px;
              text-align: left;
              vertical-align: top;
            }}
            th {{
              background: #f7f7f7;
            }}
            .btn {{
              display:inline-block;
              padding:8px 12px;
              background:#222;
              color:#fff;
              text-decoration:none;
              border-radius:6px;
            }}
          </style>
        </head>
        <body>
          <div class="card">
            <h2>Rendered page scan for set {set_num}</h2>
            <p><strong>Pages dir:</strong> {result.get("pages_dir", "-")}</p>
            <p><strong>Total rendered pages:</strong> {result.get("page_count", 0)}</p>
            <p><strong>Rows shown:</strong> {result.get("row_count", 0)}</p>
            <p>
              <a class="btn" href="/api/scan-set-candidates?set_num={set_num}&include_all={'true' if include_all else 'false'}" target="_blank">Raw JSON</a>
              &nbsp;
              <a class="btn" href="{include_all_toggle}">{include_all_label}</a>
            </p>
          </div>

          <table>
            <thead>
              <tr>
                <th>Page</th>
                <th>Kind</th>
                <th>Numeric</th>
                <th>Words</th>
                <th>Lines</th>
                <th>Bright</th>
                <th>Edges</th>
                <th>Boxes</th>
                <th>Largest box</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {rows_block}
            </tbody>
          </table>
        </body>
        </html>
        """
    )
