import re
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pytesseract
from pytesseract import Output
from fastapi import APIRouter, HTTPException, Query, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, Response

from clean.services import (
    analyzer_scan_service,
    bag_truth_store,
    gap_scan_service,
    truth_service,
    debug_service,
    precheck_service,
    page_analyzer,
)

router = APIRouter()


def _strong_structure_from_result(result: Dict[str, Any]) -> bool:
    return bool(
        result.get("strong_structure")
        or (
            bool(result.get("panel_found"))
            and bool(result.get("shell_found"))
            and bool(result.get("grey_bag_found"))
        )
    )


def _debug_focus_ocr(img, box):
    crop_img = page_analyzer.crop(img, box)
    if crop_img is None or crop_img.size == 0:
        return {"box": box, "crop_ok": False}

    raw_psm8 = pytesseract.image_to_string(
        crop_img,
        config="--psm 8 -c tessedit_char_whitelist=0123456789",
    ).strip()
    val, raw, score = page_analyzer.read_bag_number_with_score(crop_img)
    region_candidate = page_analyzer._find_best_ocr_number_in_region(
        img,
        box,
        single_digit_min_score=48.0,
        reject_step_strips=True,
    )
    return {
        "box": box,
        "crop_ok": True,
        "ocr_raw_psm8": raw_psm8,
        "direct_read": {"value": val, "raw": raw, "score": score},
        "region_candidate": region_candidate,
        "accepted": region_candidate is not None,
    }


def _classify_ocr_token(
    text: str,
    x: int,
    y: int,
    page_width: int,
    page_height: int,
) -> Tuple[str, Optional[str]]:
    if re.match(r"^\d+\s*x$", text, flags=re.IGNORECASE):
        return "x_marker", None
    if re.match(r"^\d+$", text):
        if x < page_width * 0.18 and y > page_height * 0.88:
            return "page_number", "left"
        if x > page_width * 0.82 and y > page_height * 0.88:
            return "page_number", "right"
        return "number", None
    return "other", None


def _extract_ocr_tokens(
    img,
    page: int,
    min_conf: float = 0.0,
) -> Tuple[int, int, str, List[int], List[str], List[Dict[str, Any]]]:
    page_height, page_width = img.shape[:2]
    raw_text = pytesseract.image_to_string(img, config="--psm 6") or ""
    numbers = [int(token) for token in re.findall(r"\b\d+\b", raw_text)]
    x_markers = re.findall(r"\b\d+\s*x\b", raw_text, flags=re.IGNORECASE)
    data = pytesseract.image_to_data(img, config="--psm 6", output_type=Output.DICT)

    tokens: List[Dict[str, Any]] = []
    token_count = len(data.get("text", []))
    for i in range(token_count):
        text = (data.get("text", [""])[i] or "").strip()
        if not text:
            continue

        try:
            conf = float(data.get("conf", ["-1"])[i])
        except Exception:
            conf = -1.0

        if conf <= float(min_conf):
            continue

        x = int(data.get("left", [0])[i] or 0)
        y = int(data.get("top", [0])[i] or 0)
        w = int(data.get("width", [0])[i] or 0)
        h = int(data.get("height", [0])[i] or 0)
        kind, page_number_side = _classify_ocr_token(text, x, y, page_width, page_height)

        token: Dict[str, Any] = {
            "text": text,
            "conf": conf,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": x + (w // 2),
            "cy": y + (h // 2),
            "area": w * h,
            "kind": kind,
        }
        if kind == "page_number":
            token["page_number_side"] = page_number_side
            token["page_number_match"] = int(text) == int(page)

        tokens.append(token)

    return page_width, page_height, raw_text, numbers, x_markers, tokens


def _should_join_number_tokens(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    if left.get("kind") != "number" or right.get("kind") != "number":
        return False

    h1 = int(left.get("h", 0) or 0)
    h2 = int(right.get("h", 0) or 0)
    if h1 <= 0 or h2 <= 0:
        return False

    max_h = max(h1, h2)
    cy_diff = abs(int(left.get("cy", 0) or 0) - int(right.get("cy", 0) or 0))
    if cy_diff > max_h * 0.45:
        return False

    height_ratio = float(min(h1, h2)) / float(max_h)
    if height_ratio < 0.65 or height_ratio > 1.55:
        return False

    left_right_edge = int(left.get("x", 0) or 0) + int(left.get("w", 0) or 0)
    right_left_edge = int(right.get("x", 0) or 0)
    gap = right_left_edge - left_right_edge
    if gap < -2 or gap > max_h * 0.75:
        return False

    return True


def _build_joined_numbers(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    number_tokens = [token for token in tokens if token.get("kind") == "number"]
    number_tokens.sort(key=lambda token: (int(token.get("y", 0) or 0), int(token.get("x", 0) or 0)))

    used = [False] * len(number_tokens)
    joined_numbers: List[Dict[str, Any]] = []

    for i, token in enumerate(number_tokens):
        if used[i]:
            continue

        group = [token]
        used[i] = True
        current = token

        for j in range(i + 1, len(number_tokens)):
            if used[j]:
                continue
            candidate = number_tokens[j]
            if _should_join_number_tokens(current, candidate):
                group.append(candidate)
                used[j] = True
                current = candidate
            elif int(candidate.get("x", 0) or 0) > int(current.get("x", 0) or 0) + int(current.get("w", 0) or 0) + max(int(current.get("h", 0) or 0), int(candidate.get("h", 0) or 0)):
                break

        if len(group) < 2:
            continue

        xs = [int(item.get("x", 0) or 0) for item in group]
        ys = [int(item.get("y", 0) or 0) for item in group]
        rights = [int(item.get("x", 0) or 0) + int(item.get("w", 0) or 0) for item in group]
        bottoms = [int(item.get("y", 0) or 0) + int(item.get("h", 0) or 0) for item in group]
        min_x = min(xs)
        min_y = min(ys)
        max_right = max(rights)
        max_bottom = max(bottoms)
        width = max_right - min_x
        height = max_bottom - min_y

        joined_numbers.append(
            {
                "text": "".join(str(item.get("text", "")) for item in group),
                "parts": [str(item.get("text", "")) for item in group],
                "x": min_x,
                "y": min_y,
                "w": width,
                "h": height,
                "cx": min_x + (width // 2),
                "cy": min_y + (height // 2),
                "area": width * height,
            }
        )

    joined_numbers.sort(key=lambda item: (int(item.get("y", 0) or 0), int(item.get("x", 0) or 0)))
    return joined_numbers


def _parse_pages_param(pages: str) -> List[int]:
    values: List[int] = []
    for chunk in (pages or "").split(","):
        raw = chunk.strip()
        if not raw:
            continue
        try:
            page = int(raw)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid page value: {raw}",
            ) from exc
        if page < 1:
            raise HTTPException(status_code=400, detail="Pages must be >= 1")
        values.append(page)

    if not values:
        raise HTTPException(status_code=400, detail="At least one page is required")

    return values


_GREEN_STEP_DEBUG_CACHE: Dict[str, Dict[int, Dict[str, Any]]] = {}


def _get_green_step_debug_row(set_num: str, page: int) -> Optional[Dict[str, Any]]:
    cache = _GREEN_STEP_DEBUG_CACHE.get(str(set_num))
    if cache is None:
        cache = {}
        path = debug_service.DEBUG_ROOT / str(set_num) / "green_step_boxes_7_38.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text())
                cache = {
                    int(item.get("page", 0) or 0): item
                    for item in (payload.get("pages", []) or [])
                }
            except Exception:
                cache = {}
        _GREEN_STEP_DEBUG_CACHE[str(set_num)] = cache
    return cache.get(int(page))


@router.get("/debug/number-box-sizes")
def debug_number_box_sizes(
    set_num: str = Query(...),
    start: int = Query(..., ge=1),
    end: int = Query(..., ge=1),
):
    if int(end) < int(start):
        raise HTTPException(status_code=400, detail="end must be >= start")

    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        raise HTTPException(status_code=404, detail="No rendered pages found for set")

    page_analyzer.configure_pages_dir(str(pages_dir))

    pages = []
    for page in range(int(start), int(end) + 1):
        image_path = debug_service.resolve_page_image_path(set_num, page)
        if image_path is None:
            pages.append(
                {
                    "page": int(page),
                    "image_found": False,
                    "bag_number": None,
                    "number_box_found": False,
                    "number_box": None,
                    "number_box_width": None,
                    "number_box_height": None,
                    "number_box_area": None,
                    "panel_found": False,
                    "panel_source": None,
                    "strong_structure": False,
                    "page_kind": None,
                    "reasons": [],
                }
            )
            continue

        result = page_analyzer.analyze_page(int(page), include_image=False)
        precheck = precheck_service.get_page_precheck(set_num, int(page))
        green_step_row = _get_green_step_debug_row(set_num, int(page))
        number_box = result.get("number_box")

        green_box_count = None
        step_grid_number_count = None
        multi_step_green_boxes = None
        if green_step_row is not None:
            green_box_count = int(green_step_row.get("green_box_count", 0) or 0)
            step_grid_number_count = int(
                green_step_row.get("step_candidate_count", 0) or 0
            )
            multi_step_green_boxes = bool(green_box_count >= 2)

        pages.append(
            {
                "page": int(page),
                "image_found": True,
                "bag_number": result.get("bag_number"),
                "number_box_found": bool(result.get("number_box_found")),
                "number_box": number_box,
                "number_box_x": result.get("number_box_x"),
                "number_box_y": result.get("number_box_y"),
                "number_box_width": result.get("number_box_width"),
                "number_box_height": result.get("number_box_height"),
                "number_box_area": result.get("number_box_area"),
                "panel_found": bool(result.get("panel_found")),
                "panel_box": result.get("panel_box"),
                "panel_source": result.get("panel_source"),
                "strong_structure": _strong_structure_from_result(result),
                "page_kind": precheck.get("page_kind", "other"),
                "step_grid_number_count": step_grid_number_count,
                "green_box_count": green_box_count,
                "multi_step_green_boxes": multi_step_green_boxes,
                "reasons": result.get("reasons", []),
            }
        )

    return {
        "set_num": str(set_num),
        "start": int(start),
        "end": int(end),
        "pages": pages,
    }


@router.get("/debug/number-box-visual")
def debug_number_box_visual(
    set_num: str = Query(...),
    pages: str = Query(...),
):
    page_list = _parse_pages_param(pages)
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        raise HTTPException(status_code=404, detail="No rendered pages found for set")

    page_analyzer.configure_pages_dir(str(pages_dir))

    output_dir = debug_service.DEBUG_ROOT / str(set_num) / "number_visual"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []

    for page in page_list:
        image_path = debug_service.resolve_page_image_path(set_num, page)
        if image_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"Page image not found for page {page}",
            )

        img = cv2.imread(str(image_path))
        if img is None:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load page image for page {page}",
            )

        result = page_analyzer.analyze_page(int(page), include_image=False)
        panel_box = result.get("panel_box")
        number_box = result.get("number_box")
        bag_number = result.get("bag_number")

        if isinstance(panel_box, list) and len(panel_box) == 4:
            px, py, pw, ph = [int(value) for value in panel_box]
            cv2.rectangle(img, (px, py), (px + pw, py + ph), (255, 0, 0), 3)

        number_box_area = None
        if isinstance(number_box, list) and len(number_box) == 4:
            nx, ny, nw, nh = [int(value) for value in number_box]
            number_box_area = int(nw * nh)
            cv2.rectangle(img, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 3)

        label = f"p{page} | bag={bag_number} | area={number_box_area}"
        cv2.putText(
            img,
            label,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        output_path = output_dir / f"page_{int(page):03d}.png"
        ok = cv2.imwrite(str(output_path), img)
        if not ok:
            raise HTTPException(
                status_code=500,
                detail=f"Could not save overlay for page {page}",
            )
        saved_files.append(str(output_path))

    return {
        "set_num": str(set_num),
        "saved_files": saved_files,
    }




@router.get("/debug/page-numbers")
def debug_page_numbers(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
):
    path = debug_service.resolve_page_image_path(set_num, page)
    if path is None:
        raise HTTPException(status_code=404, detail="Page image not found")

    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not load page image")

    page_width, page_height, raw_text, numbers, x_markers, tokens = _extract_ocr_tokens(
        img,
        page,
        min_conf=0.0,
    )
    joined_numbers = _build_joined_numbers(tokens)

    return {
        "set_num": str(set_num),
        "page": int(page),
        "page_width": int(page_width),
        "page_height": int(page_height),
        "numbers": numbers,
        "x_markers": x_markers,
        "raw_text": raw_text,
        "tokens": tokens,
        "joined_numbers": joined_numbers,
    }


@router.get("/debug/page-numbers-overlay")
def debug_page_numbers_overlay(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
    min_conf: int = Query(0),
):
    path = debug_service.resolve_page_image_path(set_num, page)
    if path is None:
        raise HTTPException(status_code=404, detail="Page image not found")

    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not load page image")

    page_width, page_height, _raw_text, _numbers, _x_markers, tokens = _extract_ocr_tokens(
        img,
        page,
        min_conf=float(min_conf),
    )
    joined_numbers = _build_joined_numbers(tokens)

    for token in tokens:
        text = str(token.get("text", ""))
        x = int(token.get("x", 0) or 0)
        y = int(token.get("y", 0) or 0)
        w = int(token.get("w", 0) or 0)
        h = int(token.get("h", 0) or 0)
        kind = str(token.get("kind", "other"))
        page_number_side = token.get("page_number_side")

        if kind == "page_number":
            color = (255, 0, 0)
            label_text = f"page_number:{text}:{page_number_side}"
        elif kind == "number":
            color = (0, 200, 0)
            label_text = text
        elif kind == "x_marker":
            color = (0, 165, 255)
            label_text = text
        else:
            color = (140, 140, 140)
            label_text = text

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        label_y = y - 6 if y > 18 else y + h + 14
        cv2.putText(
            img,
            label_text,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    for joined in joined_numbers:
        x = int(joined.get("x", 0) or 0)
        y = int(joined.get("y", 0) or 0)
        w = int(joined.get("w", 0) or 0)
        h = int(joined.get("h", 0) or 0)
        text = str(joined.get("text", ""))
        color = (180, 0, 180)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label_y = y - 6 if y > 18 else y + h + 14
        cv2.putText(
            img,
            f"joined_number:{text}",
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode overlay image")

    return Response(content=encoded.tobytes(), media_type="image/png")


@router.get("/api/debug/page-analyzer-focus")
def debug_page_analyzer_focus(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
):
    if str(set_num) != "70618" or int(page) != 81:
        raise HTTPException(status_code=403, detail="debug limited to set 70618 page 81")

    path = debug_service.resolve_page_image_path(set_num, page)
    if path is None:
        raise HTTPException(status_code=404, detail="Page image not found")

    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    page_analyzer.configure_pages_dir(str(pages_dir))
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not load page image")

    cache_key = (str(set_num), int(page))
    cached_row_before = analyzer_scan_service._ANALYZER_SCAN_CACHE.get(cache_key)
    fresh_result = page_analyzer.analyze_page(int(page), include_image=False)
    panel, panel_source, panel_score = page_analyzer.find_bag_intro_panel_with_debug(img)

    focus_debug = []
    if panel is not None:
        rx, ry, rw, rh = panel
        focus_boxes = [
            (rx + int(rw * 0.22), ry + int(rh * 0.18), int(rw * 0.38), int(rh * 0.56)),
            (rx + int(rw * 0.26), ry + int(rh * 0.22), int(rw * 0.28), int(rh * 0.48)),
            (rx + int(rw * 0.30), ry + int(rh * 0.28), int(rw * 0.20), int(rh * 0.42)),
        ]
        if panel_source == "strict_top_left":
            focus_boxes.append(
                (rx + int(rw * 0.40), ry + int(rh * 0.28), int(rw * 0.16), int(rh * 0.34))
            )
        focus_debug = [_debug_focus_ocr(img, box) for box in focus_boxes]

    why_false = []
    if not fresh_result.get("panel_found"):
        why_false.append("strict_top_left_panel_not_found")
    if not fresh_result.get("number_box_found"):
        why_false.append("no_focus_box_returned_an_accepted_number_candidate")
    if not any(item.get("region_candidate") for item in focus_debug):
        why_false.append("all_focus_boxes_rejected_by_region_ocr")
    if fresh_result.get("bag_number") is None:
        why_false.append("bag_number_never_promoted_into_final_result")

    return {
        "set_num": str(set_num),
        "page": int(page),
        "image_path": str(path),
        "precheck": precheck_service.get_page_precheck(set_num, page),
        "cached_row_before": cached_row_before,
        "fresh_result": fresh_result,
        "strict_top_left_debug": {
            "panel_found": panel is not None,
            "panel_source": panel_source,
            "panel_score": panel_score,
            "panel_box": panel,
            "focus_boxes_tested": focus_debug,
        },
        "why_number_box_found_false": why_false,
    }


@router.get("/debug/gap-table", response_class=HTMLResponse)
def debug_gap_table(
    set_num: str = Query(...),
    bag_number: int = Query(..., ge=1),
):
    gap = gap_scan_service.scan_gap_for_bag(set_num, bag_number)

    status = gap.get("status")
    if status == "already_confirmed":
        confirmed_page = truth_service.get_confirmed_page_for_bag(set_num, bag_number)
        return HTMLResponse(
            f"""
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8" />
              <title>Gap table</title>
              <style>
                body {{ font-family: Arial, sans-serif; margin: 16px; background: #f3f3f3; }}
                .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
              </style>
            </head>
            <body>
              <div class="card">
                <h2>Bag {bag_number} for set {set_num}</h2>
                <p><strong>Status:</strong> already confirmed</p>
                <p><strong>Confirmed page:</strong> {confirmed_page}</p>
              </div>
            </body>
            </html>
            """
        )

    if status != "ok":
        raise HTTPException(status_code=400, detail=gap)

    window = gap["window"]
    start_page = gap["scan_start_page"]
    end_page = gap["scan_end_page"]
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)

    rows_html = []

    for page in gap["pages_considered"]:
        analyze_url = f"/api/analyze-gap-page?set_num={set_num}&bag_number={bag_number}&page={page}"
        img_url = f"/api/debug/page-image?set_num={set_num}&page={page}"

        precheck = precheck_service.get_page_precheck(set_num, page)
        page_kind = precheck.get("page_kind", "-")
        bag_score = precheck.get("bag_start_score", 0.0)

        try:
            bag_score_label = f"{float(bag_score):.2f}"
        except Exception:
            bag_score_label = str(bag_score)

        probe = {"analysis_available": False, "confidence": 0.0}
        if pages_dir is not None:
            try:
                page_analyzer.configure_pages_dir(str(pages_dir))
                result = page_analyzer.analyze_page(page, include_image=False)
                probe = {
                    "analysis_available": True,
                    "confidence": float(result.get("confidence", 0.0) or 0.0),
                    "panel_found": bool(result.get("panel_found")),
                    "shell_found": bool(result.get("shell_found")),
                    "grey_bag_found": bool(result.get("grey_bag_found")),
                    "number_found": bool(result.get("number_found")),
                    "bag_number": result.get("bag_number"),
                }
            except Exception:
                pass

        confidence_label = "-"
        if probe.get("analysis_available"):
            conf = probe.get("confidence")
            try:
                if conf is not None and conf >= 0.85:
                    confidence_label = "HIGH"
                elif conf is not None and conf >= 0.60:
                    confidence_label = "MED"
                elif conf is not None:
                    confidence_label = "LOW"
            except Exception:
                confidence_label = "-"

        save_form = f"""
        <form method="post" action="/api/debug/save-bag-truth" style="display:inline-block; margin:0;">
          <input type="hidden" name="set_num" value="{set_num}">
          <input type="hidden" name="bag_number" value="{bag_number}">
          <input type="hidden" name="start_page" value="{page}">
          <input type="hidden" name="redirect_to" value="/debug/gap-table?set_num={set_num}&bag_number={bag_number}">
          <button type="submit">Save as truth</button>
        </form>
        """

        rows_html.append(
            f"""
        <tr>
          <td>{page}</td>
          <td>{page_kind}</td>
          <td>{bag_score_label}</td>
          <td>{confidence_label}</td>
          <td>
            <a href="{img_url}" target="_blank">View image</a>
            &nbsp;|&nbsp;
            <a href="{analyze_url}" target="_blank">Analyze page</a>
            &nbsp;|&nbsp;
            {save_form}
          </td>
        </tr>
        """
        )

    rows_block = (
        "\n".join(rows_html)
        if rows_html
        else "<tr><td colspan='5'>No candidate pages</td></tr>"
    )

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Gap table</title>
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
            button {{
              padding: 6px 10px;
              cursor: pointer;
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
            <h2>Gap table for bag {bag_number} (set {set_num})</h2>
            <p><strong>Previous confirmed:</strong> bag {window.get("previous_confirmed_bag")} page {window.get("previous_confirmed_page")}</p>
            <p><strong>Next confirmed:</strong> bag {window.get("next_confirmed_bag")} page {window.get("next_confirmed_page")}</p>
            <p><strong>Scan window:</strong> {start_page} to {end_page}</p>
            <p><a class="btn" href="/api/gap-scan?set_num={set_num}&bag_number={bag_number}" target="_blank">Raw gap JSON</a></p>
          </div>

          <table>
            <thead>
              <tr>
                <th>Page</th>
                <th>Kind</th>
                <th>Bag score</th>
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


@router.get("/api/debug/page-image")
def debug_page_image(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
):
    path = debug_service.resolve_page_image_path(set_num, page)
    if path is None:
        raise HTTPException(status_code=404, detail="Page image not found")
    return FileResponse(str(path))


@router.get("/debug/page-image")
def debug_page_image_alias(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
):
    return debug_page_image(set_num=set_num, page=page)


@router.get("/debug/bag-truth-visual", response_class=HTMLResponse)
def debug_bag_truth_visual(
    set_num: str = Query(...),
):
    saved_truth = bag_truth_store.get_bag_truth(set_num)
    cards: List[str] = []

    for row in saved_truth:
        bag_number = int(row.get("bag_number", 0) or 0)
        start_page = int(row.get("start_page", 0) or 0)
        confidence = row.get("confidence")
        source = str(row.get("source", "") or "")
        image_url = (
            f"/debug/page-image?set_num={escape(str(set_num))}&page={int(start_page)}"
        )
        analyze_url = (
            f"/api/analyze-page-direct?set_num={escape(str(set_num))}&page={int(start_page)}"
        )
        if confidence is None:
            confidence_label = "n/a"
        else:
            confidence_label = f"{float(confidence):.2f}"

        source_badge_class = (
            "badge badge-green"
            if "card" in source.lower()
            else "badge badge-slate"
        )

        cards.append(
            f"""
            <article class="tile">
              <a class="thumb-wrap" href="{image_url}" target="_blank">
                <img class="thumb" src="{image_url}" alt="Bag {bag_number} page {start_page}" loading="lazy" />
              </a>
              <div class="tile-body">
                <h3>Bag {bag_number}</h3>
                <p class="meta">Page {start_page}</p>
                <div class="badges">
                  <span class="badge badge-green">confidence {escape(confidence_label)}</span>
                  <span class="{source_badge_class}">{escape(source or 'unknown source')}</span>
                </div>
                <p class="actions">
                  <a href="{image_url}" target="_blank">Open image</a>
                  <span class="sep">|</span>
                  <a href="{analyze_url}" target="_blank">Analyze</a>
                </p>
              </div>
            </article>
            """
        )

    cards_block = (
        "\n".join(cards)
        if cards
        else "<div class='empty'>No saved bag truth found for this set.</div>"
    )

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Bag Truth Visual Gallery</title>
          <style>
            :root {{
              --bg: #f3f5f2;
              --panel: #ffffff;
              --border: #d9dfd4;
              --text: #17301f;
              --muted: #5f7467;
              --green: #2f8f55;
              --green-soft: #e7f6ec;
              --slate: #e9ecef;
              --shadow: 0 12px 30px rgba(25, 47, 33, 0.08);
            }}
            * {{ box-sizing: border-box; }}
            body {{
              margin: 0;
              padding: 24px;
              background: linear-gradient(180deg, #eef4ee 0%, var(--bg) 100%);
              color: var(--text);
              font-family: Arial, sans-serif;
            }}
            .shell {{
              max-width: 1280px;
              margin: 0 auto;
            }}
            .hero {{
              background: var(--panel);
              border: 1px solid var(--border);
              border-radius: 16px;
              padding: 18px 20px;
              margin-bottom: 18px;
              box-shadow: var(--shadow);
            }}
            .hero h1 {{
              margin: 0 0 8px;
              font-size: 28px;
            }}
            .hero p {{
              margin: 4px 0;
              color: var(--muted);
            }}
            .hero a {{
              color: var(--green);
              text-decoration: none;
            }}
            .grid {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
              gap: 16px;
            }}
            .tile {{
              background: var(--panel);
              border: 1px solid var(--border);
              border-radius: 16px;
              overflow: hidden;
              box-shadow: var(--shadow);
            }}
            .thumb-wrap {{
              display: block;
              background: #dbe7db;
              aspect-ratio: 1 / 1.25;
            }}
            .thumb {{
              width: 100%;
              height: 100%;
              object-fit: cover;
              display: block;
            }}
            .tile-body {{
              padding: 14px;
            }}
            .tile-body h3 {{
              margin: 0 0 6px;
              font-size: 20px;
            }}
            .meta {{
              margin: 0 0 10px;
              color: var(--muted);
            }}
            .badges {{
              display: flex;
              flex-wrap: wrap;
              gap: 8px;
              margin-bottom: 10px;
            }}
            .badge {{
              display: inline-block;
              border-radius: 999px;
              padding: 5px 10px;
              font-size: 12px;
              font-weight: 700;
            }}
            .badge-green {{
              background: var(--green-soft);
              color: var(--green);
            }}
            .badge-slate {{
              background: var(--slate);
              color: #44515a;
            }}
            .actions {{
              margin: 0;
              font-size: 14px;
            }}
            .actions a {{
              color: var(--green);
              text-decoration: none;
            }}
            .sep {{
              color: #90a091;
              margin: 0 6px;
            }}
            .empty {{
              background: var(--panel);
              border: 1px dashed var(--border);
              border-radius: 16px;
              padding: 30px;
              color: var(--muted);
            }}
          </style>
        </head>
        <body>
          <div class="shell">
            <section class="hero">
              <h1>Bag Truth Visual Gallery</h1>
              <p><strong>Set:</strong> {escape(str(set_num))}</p>
              <p><strong>Saved bags:</strong> {len(saved_truth)}</p>
              <p><a href="/api/bag-truth?set_num={escape(str(set_num))}" target="_blank">Open raw bag truth JSON</a></p>
            </section>
            <section class="grid">
              {cards_block}
            </section>
          </div>
        </body>
        </html>
        """
    )


@router.post("/api/debug/save-bag-truth")
def debug_save_bag_truth(
    set_num: str = Form(...),
    bag_number: int = Form(...),
    start_page: int = Form(...),
    redirect_to: str = Form(...),
):
    truth_service.save_confirmed_bag_truth(
        set_num=set_num,
        bag_number=int(bag_number),
        start_page=int(start_page),
    )
    return RedirectResponse(url=redirect_to, status_code=303)
