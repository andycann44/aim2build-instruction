import time
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse

from clean.services import auto_confirm_service, workflow_service, debug_service
from clean.services.page_analyzer import analyze_page, configure_pages_dir
router = APIRouter()


def _list_rendered_pages(pages_dir):
    pages = []
    for path in sorted(pages_dir.glob("page_*.png")):
        stem = path.stem
        try:
            page_num = int(stem.replace("page_", ""))
        except ValueError:
            continue
        pages.append(page_num)
    return pages


@router.get("/api/analyze-page-direct")
def analyze_page_direct(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
):
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)

    if pages_dir is None:
        return {
            "ok": False,
            "set_num": set_num,
            "page": int(page),
            "error": "no rendered pages found for set",
        }

    configure_pages_dir(str(pages_dir))
    result = analyze_page(int(page), include_image=False)
    return result


@router.get("/api/bag-starts-scan")
def bag_starts_scan(
    set_num: str = Query(...),
    start: Optional[int] = Query(None, ge=1),
    end: Optional[int] = Query(None, ge=1),
    limit: int = Query(10, ge=1),
    expected_next_bag: Optional[int] = Query(None, ge=1),
):
    started_at = time.monotonic()
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)

    if pages_dir is None:
        return {
            "ok": False,
            "set_num": set_num,
            "error": "no rendered pages found for set",
            "pages_dir": None,
            "page_count": 0,
            "scanned_pages_count": 0,
            "next_start_page": None,
            "remaining_pages": 0,
            "timed_out": False,
            "elapsed_seconds_total": 0.0,
            "page_timings": [],
            "slow_pages": [],
            "skipped_rows": [],
            "bag_starts": [],
            "bag_ranges": [],
        }

    pages = _list_rendered_pages(pages_dir)
    if start is not None:
        pages = [page for page in pages if int(page) >= int(start)]
    if end is not None:
        pages = [page for page in pages if int(page) <= int(end)]

    total_pages = len(pages)

    configure_pages_dir(str(pages_dir))

    bag_starts = []
    skipped_rows = []
    slow_pages = []
    page_timings = []
    scanned_pages_count = 0
    last_scanned_page = None
    for page in pages:
        if scanned_pages_count >= limit:
            break

        scanned_pages_count += 1
        last_scanned_page = int(page)
        page_started_at = time.monotonic()
        result = analyze_page(int(page), include_image=False)
        elapsed_seconds = round(time.monotonic() - page_started_at, 3)
        page_timings.append(
            {
                "page": int(page),
                "elapsed_seconds": elapsed_seconds,
            }
        )
        slow_page = elapsed_seconds > 8.0
        if elapsed_seconds > 8.0:
            slow_row = {
                "page": int(page),
                "slow_page": True,
                "elapsed_seconds": elapsed_seconds,
            }
            slow_pages.append(slow_row)
        if result.get("bag_number") is None:
            continue
        if bool(result.get("multi_step_green_boxes")):
            continue
        if not bool(result.get("number_box_found")):
            continue
        if result.get("panel_source") != "strict_top_left":
            continue
        number_box_area = result.get("number_box_area")
        if number_box_area is not None and int(number_box_area) <= 40000:
            continue

        bag_starts.append(
            {
                "page": int(result.get("page", page) or page),
                "bag_number": result.get("bag_number"),
                "panel_found": bool(result.get("panel_found")),
                "panel_source": result.get("panel_source"),
                "number_box_found": bool(result.get("number_box_found")),
                "multi_step_green_boxes": bool(result.get("multi_step_green_boxes")),
                "page_kind": result.get("page_kind", "other"),
                "confidence": float(result.get("confidence", 0.0) or 0.0),
                "ocr_raw": result.get("ocr_raw", "") or "",
                "slow_page": slow_page,
            }
        )

    bag_starts.sort(key=lambda item: int(item["page"]))
    if expected_next_bag is not None:
        filtered_bag_starts = []
        current_expected_bag = int(expected_next_bag)
        for item in bag_starts:
            candidate_bag_number = item.get("bag_number")
            if candidate_bag_number == current_expected_bag:
                filtered_bag_starts.append(item)
                current_expected_bag += 1
                continue

            skipped_rows.append(
                {
                    "page": int(item["page"]),
                    "skipped": True,
                    "skip_reason": "unexpected_bag_number",
                    "expected_bag_number": current_expected_bag,
                    "candidate_bag_number": candidate_bag_number,
                    "slow_page": bool(item.get("slow_page")),
                }
            )

        bag_starts = filtered_bag_starts

    kept_bag_starts = []
    previous_kept_bag_number = None
    for item in bag_starts:
        candidate_bag_number = item.get("bag_number")
        if previous_kept_bag_number is None:
            kept_bag_starts.append(item)
            previous_kept_bag_number = candidate_bag_number
            continue

        if candidate_bag_number == previous_kept_bag_number + 1:
            kept_bag_starts.append(item)
            previous_kept_bag_number = candidate_bag_number
            continue

        skipped_rows.append(
            {
                "page": int(item["page"]),
                "skipped": True,
                "skip_reason": "bag_number_jump",
                "previous_bag_number": previous_kept_bag_number,
                "candidate_bag_number": candidate_bag_number,
                "slow_page": bool(item.get("slow_page")),
            }
        )

    bag_starts = kept_bag_starts

    next_start_page = None
    remaining_pages = 0
    if last_scanned_page is not None:
        remaining_page_list = [page for page in pages if int(page) > int(last_scanned_page)]
        next_start_page = (
            int(remaining_page_list[0]) if remaining_page_list else None
        )
        remaining_pages = len(remaining_page_list)

    bag_ranges = []
    if scanned_pages_count > 0 and last_scanned_page is not None:
        last_page = int(last_scanned_page)
        for index, item in enumerate(bag_starts):
            start_page = int(item["page"])
            next_bag_start_page = (
                int(bag_starts[index + 1]["page"])
                if index + 1 < len(bag_starts)
                else None
            )
            end_page = (
                int(next_bag_start_page) - 1
                if next_bag_start_page is not None
                else last_page
            )
            bag_ranges.append(
                {
                    "bag_number": item["bag_number"],
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )

    return {
        "ok": True,
        "set_num": set_num,
        "pages_dir": str(pages_dir),
        "page_count": total_pages,
        "scanned_pages_count": scanned_pages_count,
        "next_start_page": next_start_page,
        "remaining_pages": remaining_pages,
        "timed_out": False,
        "elapsed_seconds_total": round(time.monotonic() - started_at, 3),
        "page_timings": page_timings,
        "slow_pages": slow_pages,
        "skipped_rows": skipped_rows,
        "bag_starts": bag_starts,
        "bag_ranges": bag_ranges,
    }


@router.get("/api/set-workflow")
def set_workflow(set_num: str = Query(...)):
    return workflow_service.get_set_workflow(set_num)


@router.post("/api/auto-confirm-top-candidates")
def auto_confirm_top_candidates(
    set_num: Optional[str] = Query(None),
    set_num_form: Optional[str] = Form(None, alias="set_num"),
    redirect_to: Optional[str] = Form(None),
):
    selected_set_num = set_num_form or set_num
    if not selected_set_num:
        raise HTTPException(status_code=400, detail="set_num is required")

    summary = auto_confirm_service.auto_confirm_top_candidates(selected_set_num)
    if redirect_to:
        return RedirectResponse(url=redirect_to, status_code=303)
    return summary


@router.get("/debug/set-overview", response_class=HTMLResponse)
def debug_set_overview(set_num: str = Query(...)):
    workflow = workflow_service.get_set_workflow(set_num)
    mode = workflow.get("mode")

    if mode == "fresh_precheck":
        result = workflow.get("analyzer_scan", {})
        if not result.get("ok"):
            return HTMLResponse(
                f"""
                <!doctype html>
                <html>
                <head>
                  <meta charset="utf-8" />
                  <title>Set overview</title>
                  <style>
                    body {{ font-family: Arial, sans-serif; margin: 16px; background: #f3f3f3; }}
                    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
                  </style>
                </head>
                <body>
                  <div class="card">
                    <h2>Set overview for {set_num}</h2>
                    <p><strong>Mode:</strong> fresh_precheck</p>
                    <p><strong>Source:</strong> legacy analyzer scan with lightweight precheck skips</p>
                    <p><strong>Error:</strong> {result.get("error", "Unknown error")}</p>
                  </div>
                </body>
                </html>
                """
            )

        rows_html = []
        for row in result.get("rows", []):
            page = int(row["page"])
            image_url = f"/api/debug/page-image?set_num={set_num}&page={page}"
            precheck_url = f"/api/page-precheck?set_num={set_num}&page={page}"
            analyze_url = f"/api/analyze-page-direct?set_num={set_num}&page={page}"
            summary = row.get("analyzer_summary", {})
            save_form = f"""
            <form method="post" action="/api/debug/save-bag-truth" style="display:inline-flex; gap:6px; align-items:center; margin:0;">
              <input type="hidden" name="set_num" value="{set_num}">
              <input type="hidden" name="start_page" value="{page}">
              <input type="hidden" name="redirect_to" value="/debug/set-overview?set_num={set_num}">
              <input type="number" name="bag_number" min="1" placeholder="Bag #" value="{'' if summary.get('bag_number') is None else summary.get('bag_number')}" required style="width:84px;">
              <button type="submit">Save as truth</button>
            </form>
            """

            rows_html.append(
                f"""
                <tr>
                  <td>{page}</td>
                  <td>{row.get("precheck_kind", "-")}</td>
                  <td>{'Y' if summary.get("panel_found") else ''}</td>
                  <td>{'Y' if summary.get("shell_found") else ''}</td>
                  <td>{'Y' if summary.get("grey_bag_found") else ''}</td>
                  <td>{'Y' if summary.get("number_found") else ''}</td>
                  <td>{'' if summary.get("bag_number") is None else summary.get("bag_number")}</td>
                  <td>{float(summary.get("confidence", 0.0) or 0.0):.2f}</td>
                  <td>
                    <a href="{image_url}" target="_blank">View image</a>
                    &nbsp;|&nbsp;
                    <a href="{precheck_url}" target="_blank">Precheck JSON</a>
                    &nbsp;|&nbsp;
                    <a href="{analyze_url}" target="_blank">Analyze direct</a>
                    &nbsp;|&nbsp;
                    {save_form}
                  </td>
                </tr>
                """
            )

        rows_block = (
            "\n".join(rows_html)
            if rows_html
            else "<tr><td colspan='9'>No analyzed rows</td></tr>"
        )

        return HTMLResponse(
            f"""
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8" />
              <title>Set overview</title>
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
                button {{
                  padding: 6px 10px;
                  cursor: pointer;
                }}
              </style>
            </head>
            <body>
              <div class="card">
                <h2>Set overview for {set_num}</h2>
                <p><strong>Mode:</strong> fresh_precheck</p>
                <p>This view skips only obvious junk pages, then ranks candidates using the legacy analyzer.</p>
                <p><strong>Pages dir:</strong> {result.get("pages_dir", "-")}</p>
                <p><strong>Total rendered pages:</strong> {result.get("page_count", 0)}</p>
                <p><strong>Rows shown:</strong> {result.get("row_count", 0)}</p>
                <form method="post" action="/api/auto-confirm-top-candidates" style="margin: 12px 0;">
                  <input type="hidden" name="set_num" value="{set_num}">
                  <input type="hidden" name="redirect_to" value="/debug/set-overview?set_num={set_num}">
                  <button type="submit">Auto-confirm strong candidates</button>
                </form>
                <p>
                  <a class="btn" href="/api/set-workflow?set_num={set_num}" target="_blank">Workflow JSON</a>
                  &nbsp;
                  <a class="btn" href="/api/scan-set-with-analyzer?set_num={set_num}&include_all=false" target="_blank">Raw analyzer JSON</a>
                  &nbsp;
                  <a class="btn" href="/debug/scan-set-with-analyzer?set_num={set_num}&include_all=false">Open analyzer table</a>
                </p>
              </div>

              <table>
                <thead>
                  <tr>
                    <th>Page</th>
                    <th>Precheck kind</th>
                    <th>Panel</th>
                    <th>Shell</th>
                    <th>Grey bag</th>
                    <th>Number</th>
                    <th>Bag number</th>
                    <th>Confidence</th>
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

    sequence = workflow.get("sequence", {})
    confirmed_rows = sequence.get("confirmed", [])
    missing_numbers = sequence.get("missing_numbers", [])
    missing_windows = sequence.get("missing_bag_windows", [])

    confirmed_rows_html = []
    for row in confirmed_rows:
        page = int(row.get("start_page", row.get("page", 0)))
        bag_number = int(row["bag_number"])
        image_url = f"/api/debug/page-image?set_num={set_num}&page={page}"
        confirmed_rows_html.append(
            f"""
            <tr>
              <td>{bag_number}</td>
              <td>{page}</td>
              <td>{float(row.get("confidence", 0.0) or 0.0):.2f}</td>
              <td>{row.get("source", "-")}</td>
              <td><a href="{image_url}" target="_blank">View image</a></td>
            </tr>
            """
        )

    missing_rows_html = []
    for window in missing_windows:
        bag_number = int(window["bag_number"])
        gap_url = f"/debug/gap-table?set_num={set_num}&bag_number={bag_number}"
        missing_rows_html.append(
            f"""
            <tr>
              <td>{bag_number}</td>
              <td>{window.get("previous_confirmed_bag")}</td>
              <td>{window.get("previous_confirmed_page")}</td>
              <td>{window.get("next_confirmed_bag")}</td>
              <td>{window.get("next_confirmed_page")}</td>
              <td><a href="{gap_url}">Open gap table</a></td>
            </tr>
            """
        )

    confirmed_block = (
        "\n".join(confirmed_rows_html)
        if confirmed_rows_html
        else "<tr><td colspan='5'>No confirmed rows</td></tr>"
    )
    missing_block = (
        "\n".join(missing_rows_html)
        if missing_rows_html
        else "<tr><td colspan='6'>No missing bag windows</td></tr>"
    )

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Set overview</title>
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
              margin-bottom: 16px;
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
            <h2>Set overview for {set_num}</h2>
            <p><strong>Mode:</strong> truth_guided</p>
            <p><strong>Missing numbers:</strong> {missing_numbers}</p>
            <p>
              <a class="btn" href="/api/set-workflow?set_num={set_num}" target="_blank">Workflow JSON</a>
              &nbsp;
              <a class="btn" href="/api/sequence-scan?set_num={set_num}" target="_blank">Sequence JSON</a>
            </p>
          </div>

          <div class="card">
            <h3>Confirmed rows</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Bag</th>
                <th>Page</th>
                <th>Confidence</th>
                <th>Source</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {confirmed_block}
            </tbody>
          </table>

          <div class="card">
            <h3>Missing bag windows</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Bag</th>
                <th>Prev bag</th>
                <th>Prev page</th>
                <th>Next bag</th>
                <th>Next page</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {missing_block}
            </tbody>
          </table>
        </body>
        </html>
        """
    )
