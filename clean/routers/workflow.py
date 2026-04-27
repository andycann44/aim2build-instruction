import json
from html import escape
import time
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse

from clean.services import (
    auto_confirm_service,
    bag_truth_store,
    workflow_service,
    debug_service,
    step_sequence_bag_service,
)
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


def _build_bag_ranges(bag_starts, last_page):
    bag_ranges = []
    if not bag_starts or last_page is None:
        return bag_ranges

    sorted_bag_starts = sorted(bag_starts, key=lambda item: int(item["page"]))
    last_page = int(last_page)

    for index, item in enumerate(sorted_bag_starts):
        start_page = int(item["page"])
        next_bag_start_page = (
            int(sorted_bag_starts[index + 1]["page"])
            if index + 1 < len(sorted_bag_starts)
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

    return bag_ranges


def _boxes_overlap(box_a, box_b):
    if not box_a or not box_b:
        return False

    ax, ay, aw, ah = [int(v) for v in box_a]
    bx, by, bw, bh = [int(v) for v in box_b]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False

    a_right = ax + aw
    a_bottom = ay + ah
    b_right = bx + bw
    b_bottom = by + bh

    return not (
        a_right <= bx
        or b_right <= ax
        or a_bottom <= by
        or b_bottom <= ay
    )


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
    pages_to_scan = []
    for page in pages:
        if len(pages_to_scan) >= limit:
            break
        pages_to_scan.append(int(page))

    configure_pages_dir(str(pages_dir))
    step_role_map = step_sequence_bag_service.build_step_sequence_prepass_for_pages(
        set_num=set_num,
        pages=pages_to_scan,
    )
    step_role_by_page = {
        int(item.get("page", 0) or 0): item for item in (step_role_map or [])
    }

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
        number_box_area = result.get("number_box_area")
        if number_box_area is not None:
            area = int(number_box_area)
            if expected_next_bag is not None:
                if area <= 20000:
                    continue
            else:
                if area <= 40000:
                    continue
        panel_box = result.get("panel_box")
        number_box = result.get("number_box")
        step_role_row = step_role_by_page.get(int(page))
        detected_step_boxes = list(
            (step_role_row or {}).get("detected_step_boxes", []) or []
        )
        main_step_boxes = [
            item.get("box")
            for item in detected_step_boxes
            if item.get("step_group") == "main_steps"
        ]
        contaminating_step_box = None
        for main_step_box in main_step_boxes:
            if _boxes_overlap(main_step_box, number_box):
                contaminating_step_box = main_step_box
                break
        if contaminating_step_box is not None:
            skipped_rows.append(
                {
                    "page": int(page),
                    "skipped": True,
                    "skip_reason": "step_box_contaminates_bag_number",
                    "candidate_bag_number": result.get("bag_number"),
                    "number_box": number_box,
                    "panel_box": panel_box,
                    "main_step_box": contaminating_step_box,
                    "slow_page": slow_page,
                }
            )
            continue
        accept_by_card = bool(result.get("bag_start_card_found"))
        accept_by_strict_pattern = (
            result.get("panel_source") == "strict_top_left"
            and bool(result.get("number_box_found"))
        )
        if not (accept_by_card or accept_by_strict_pattern):
            skipped_rows.append(
                {
                    "page": int(page),
                    "skipped": True,
                    "skip_reason": "no_valid_bag_start_pattern",
                    "candidate_bag_number": result.get("bag_number"),
                    "panel_source": result.get("panel_source"),
                    "bag_start_card_found": bool(result.get("bag_start_card_found")),
                    "bag_start_card_score": float(
                        result.get("bag_start_card_score", 0.0) or 0.0
                    ),
                    "bag_start_card_reasons": list(
                        result.get("bag_start_card_reasons", []) or []
                    ),
                    "slow_page": slow_page,
                }
            )
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
                "bag_start_card_found": bool(result.get("bag_start_card_found")),
                "bag_start_card_score": float(
                    result.get("bag_start_card_score", 0.0) or 0.0
                ),
                "bag_start_card_box": result.get("bag_start_card_box"),
                "bag_start_card_reasons": list(
                    result.get("bag_start_card_reasons", []) or []
                ),
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
        bag_ranges = _build_bag_ranges(bag_starts, int(last_scanned_page))
    save_summary = bag_truth_store.save_many_bag_starts(
        set_num=set_num,
        bag_starts=bag_starts,
        source="detector",
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
        "step_role_map": step_role_map,
        "skipped_rows": skipped_rows,
        "bag_starts": bag_starts,
        "bag_ranges": bag_ranges,
        "saved_truth": save_summary.get("saved_truth", []),
        "conflicts": save_summary.get("conflicts", []),
    }


@router.get("/api/full-bag-scan")
def full_bag_scan(
    set_num: str = Query(...),
    start: Optional[int] = Query(None, ge=1),
    end: Optional[int] = Query(None, ge=1),
    chunk_size: int = Query(5, ge=1),
    max_chunks: int = Query(3, ge=1),
    expected_next_bag: Optional[int] = Query(None, ge=1),
):
    started_at = time.monotonic()
    current_start = int(start) if start is not None else 1
    current_expected_bag = (
        int(expected_next_bag) if expected_next_bag is not None else None
    )
    chunks_processed = 0
    pages_scanned = 0
    last_scanned_page = None
    all_bag_starts = []
    all_skipped_rows = []
    all_step_role_map = []
    step_role_pages_seen = set()

    while True:
        if chunks_processed >= max_chunks:
            break

        chunk = bag_starts_scan(
            set_num=set_num,
            start=current_start,
            end=end,
            limit=chunk_size,
            expected_next_bag=current_expected_bag,
        )
        chunks_processed += 1

        if not chunk.get("ok"):
            return {
                "ok": False,
                "set_num": set_num,
                "error": chunk.get("error", "bag-starts-scan failed"),
                "bag_starts": [],
                "bag_ranges": [],
                "skipped_rows": [],
                "pages_scanned": pages_scanned,
                "chunks_processed": chunks_processed,
                "total_time_seconds": round(time.monotonic() - started_at, 3),
            }

        chunk_bag_starts = list(chunk.get("bag_starts", []) or [])
        chunk_skipped_rows = list(chunk.get("skipped_rows", []) or [])
        accepted_chunk_bag_starts = []
        if current_expected_bag is not None:
            next_expected_bag = int(current_expected_bag)
            for item in chunk_bag_starts:
                candidate_bag_number = item.get("bag_number")
                if candidate_bag_number == next_expected_bag:
                    accepted_chunk_bag_starts.append(item)
                    next_expected_bag += 1
                    continue

                chunk_skipped_rows.append(
                    {
                        "page": int(item.get("page", 0) or 0),
                        "skipped": True,
                        "skip_reason": "unexpected_bag_number",
                        "expected_bag_number": next_expected_bag,
                        "candidate_bag_number": candidate_bag_number,
                        "slow_page": bool(item.get("slow_page")),
                    }
                )
        else:
            accepted_chunk_bag_starts = chunk_bag_starts

        all_bag_starts.extend(accepted_chunk_bag_starts)
        if current_expected_bag is not None:
            if accepted_chunk_bag_starts:
                current_expected_bag = next_expected_bag
        elif accepted_chunk_bag_starts:
            current_expected_bag = (
                int(accepted_chunk_bag_starts[-1]["bag_number"]) + 1
            )
        all_skipped_rows.extend(chunk_skipped_rows)
        for row in chunk.get("step_role_map", []) or []:
            page = int(row.get("page", 0) or 0)
            if page in step_role_pages_seen:
                continue
            step_role_pages_seen.add(page)
            all_step_role_map.append(row)
        pages_scanned += int(chunk.get("scanned_pages_count", 0) or 0)

        chunk_next_start_page = chunk.get("next_start_page")
        if chunk.get("scanned_pages_count", 0):
            if chunk_next_start_page is not None:
                last_scanned_page = int(chunk_next_start_page) - 1
            else:
                page_count = chunk.get("page_count")
                if page_count is not None:
                    last_scanned_page = int(page_count)

        if chunk_next_start_page is None:
            break

        current_start = int(chunk_next_start_page)

    all_bag_starts.sort(key=lambda item: int(item["page"]))
    bag_ranges = _build_bag_ranges(all_bag_starts, last_scanned_page)
    save_summary = bag_truth_store.save_many_bag_starts(
        set_num=set_num,
        bag_starts=all_bag_starts,
        source="detector",
    )

    return {
        "ok": True,
        "set_num": set_num,
        "bag_starts": all_bag_starts,
        "bag_ranges": bag_ranges,
        "step_role_map": sorted(
            all_step_role_map,
            key=lambda item: int(item.get("page", 0) or 0),
        ),
        "skipped_rows": all_skipped_rows,
        "saved_truth": save_summary.get("saved_truth", []),
        "conflicts": save_summary.get("conflicts", []),
        "pages_scanned": pages_scanned,
        "chunks_processed": chunks_processed,
        "total_time_seconds": round(time.monotonic() - started_at, 3),
    }


@router.get("/api/bag-truth")
def api_bag_truth(
    set_num: str = Query(...),
):
    return {
        "set_num": str(set_num),
        "saved_truth": bag_truth_store.get_bag_truth(set_num),
        "conflicts": bag_truth_store.get_conflicts(set_num),
    }


@router.get("/debug/full-bag-scan", response_class=HTMLResponse)
def debug_full_bag_scan(
    set_num: str = Query(...),
    start: Optional[int] = Query(None, ge=1),
    end: Optional[int] = Query(None, ge=1),
    chunk_size: int = Query(5, ge=1),
    max_chunks: int = Query(3, ge=1),
    expected_next_bag: Optional[int] = Query(None, ge=1),
):
    payload = full_bag_scan(
        set_num=set_num,
        start=start,
        end=end,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
        expected_next_bag=expected_next_bag,
    )

    saved_path = None
    save_error = None
    try:
        out_path = debug_service.DEBUG_ROOT / str(set_num) / "full_bag_scan.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        saved_path = str(out_path)
    except Exception as exc:
        save_error = str(exc)

    raw_json_url = (
        f"/api/full-bag-scan?set_num={escape(str(set_num))}"
        f"{'' if start is None else f'&start={int(start)}'}"
        f"{'' if end is None else f'&end={int(end)}'}"
        f"&chunk_size={int(chunk_size)}"
        f"&max_chunks={int(max_chunks)}"
        f"{'' if expected_next_bag is None else f'&expected_next_bag={int(expected_next_bag)}'}"
    )

    def _rows(items):
        if not items:
            return "<tr><td colspan='6'>None</td></tr>"
        rows = []
        for item in items:
            rows.append(
                "<tr>"
                f"<td>{escape(str(item.get('page', '-')))}</td>"
                f"<td>{escape(str(item.get('bag_number', '-')))}</td>"
                f"<td>{escape(str(item.get('start_page', '-')))}</td>"
                f"<td>{escape(str(item.get('end_page', '-')))}</td>"
                f"<td>{escape(str(item.get('skip_reason', '-')))}</td>"
                f"<td><pre>{escape(json.dumps(item, indent=2))}</pre></td>"
                "</tr>"
            )
        return "\n".join(rows)

    bag_starts_rows = _rows(payload.get("bag_starts", []))
    bag_ranges_rows = _rows(payload.get("bag_ranges", []))
    skipped_rows = _rows(payload.get("skipped_rows", []))
    slow_pages_rows = _rows(payload.get("slow_pages", [])) if "slow_pages" in payload else "<tr><td colspan='6'>None</td></tr>"
    step_role_rows = (
        "\n".join(
            [
                "<tr>"
                f"<td>{escape(str(item.get('page', '-')))}</td>"
                f"<td>{escape(str(item.get('step_sequence_role', '-')))}</td>"
                f"<td>{escape(str(item.get('main_step_min', '-')))}</td>"
                f"<td>{escape(str(item.get('main_step_max', '-')))}</td>"
                f"<td>{escape(str(item.get('previous_max_step', '-')))}</td>"
                f"<td>{escape(str(item.get('step_drop', '-')))}</td>"
                f"<td>{escape(str(item.get('bag_start_allowed', '-')))}</td>"
                f"<td>{escape(str(item.get('step_role_reason', '-')))}</td>"
                "</tr>"
                for item in (payload.get('step_role_map', []) or [])
            ]
        )
        if (payload.get("step_role_map", []) or [])
        else "<tr><td colspan='8'>None</td></tr>"
    )
    page_timings_json = escape(json.dumps(payload.get("page_timings", []), indent=2))

    saved_block = (
        f"<p><strong>Saved debug JSON:</strong> <code>{escape(saved_path)}</code></p>"
        if saved_path
        else (
            f"<p><strong>Save error:</strong> {escape(save_error)}</p>"
            if save_error
            else ""
        )
    )

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Full Bag Scan</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 16px; background: #f3f3f3; }}
            .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 16px; }}
            .btn {{ display:inline-block; padding:8px 12px; background:#222; color:#fff; text-decoration:none; border-radius:6px; margin-right:8px; }}
            table {{ width: 100%; border-collapse: collapse; background: #fff; margin-bottom: 16px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
            th {{ background: #f7f7f7; }}
            pre {{ white-space: pre-wrap; margin: 0; }}
            code {{ white-space: pre-wrap; }}
          </style>
        </head>
        <body>
          <div class="card">
            <h2>Full Bag Scan for {escape(str(set_num))}</h2>
            <p><strong>Params:</strong> start={escape(str(start))}, end={escape(str(end))}, chunk_size={int(chunk_size)}, max_chunks={int(max_chunks)}, expected_next_bag={escape(str(expected_next_bag))}</p>
            <p><strong>Pages scanned:</strong> {int(payload.get("pages_scanned", 0) or 0)} | <strong>Chunks processed:</strong> {int(payload.get("chunks_processed", 0) or 0)} | <strong>Total seconds:</strong> {escape(str(payload.get("total_time_seconds", 0.0)))}</p>
            {saved_block}
            <p>
              <a class="btn" href="{raw_json_url}" target="_blank" download>Download JSON</a>
              <a class="btn" href="{raw_json_url}" target="_blank">Open Raw JSON</a>
            </p>
          </div>

          <div class="card"><h3>Accepted Bag Starts</h3></div>
          <table>
            <thead><tr><th>Page</th><th>Bag</th><th>Start</th><th>End</th><th>Status</th><th>Data</th></tr></thead>
            <tbody>{bag_starts_rows}</tbody>
          </table>

          <div class="card"><h3>Bag Ranges</h3></div>
          <table>
            <thead><tr><th>Page</th><th>Bag</th><th>Start</th><th>End</th><th>Status</th><th>Data</th></tr></thead>
            <tbody>{bag_ranges_rows}</tbody>
          </table>

          <div class="card"><h3>Skipped Rows</h3></div>
          <table>
            <thead><tr><th>Page</th><th>Bag</th><th>Start</th><th>End</th><th>Status</th><th>Data</th></tr></thead>
            <tbody>{skipped_rows}</tbody>
          </table>

          <div class="card"><h3>Slow Pages</h3></div>
          <table>
            <thead><tr><th>Page</th><th>Bag</th><th>Start</th><th>End</th><th>Status</th><th>Data</th></tr></thead>
            <tbody>{slow_pages_rows}</tbody>
          </table>

          <div class="card"><h3>Step Sequence Prepass</h3></div>
          <table>
            <thead><tr><th>Page</th><th>Role</th><th>Main Min</th><th>Main Max</th><th>Prev Max</th><th>Step Drop</th><th>Allowed</th><th>Reason</th></tr></thead>
            <tbody>{step_role_rows}</tbody>
          </table>

          <div class="card">
            <h3>Page Timings</h3>
            <pre>{page_timings_json}</pre>
          </div>
        </body>
        </html>
        """
    )


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
