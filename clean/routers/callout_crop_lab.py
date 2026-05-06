"""
Diagnostic-only route: /debug/callout-crop-lab

Read-only. Does NOT modify any data, UI, training, or exports.
Shows, per page/step, the full callout-crop detection pipeline:
  1. Original page image with step box overlay
  2. Detected callout rect (edge-detection pass)
  3. Final proposed crop with blob boxes + OCR token boxes
  4. Computed qty slot list
"""
import base64
import math
import os
import re
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from clean.routers.debug import (
    _contact_sheet_step_boxes_from_detected,
    _encode_debug_image_data_uri,
    _extract_qty_tokens_from_image,
    _require_openai_vision_client_debug,
    _resolve_bag_page_range,
    _response_text_to_json_debug,
)
from clean.routers.instruction_debug import (
    _detect_callout_rect_by_edges,
    _extract_detected_qty_details_from_crop,
)
from clean.services import debug_service, step_detector_service

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers (lab-only, no side effects)
# ---------------------------------------------------------------------------

def _lab_encode(img, max_width: int = 600) -> str:
    """Encode a cv2 image as a data-URI JPEG for inline HTML display."""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _draw_rect(img, x: int, y: int, w: int, h: int, color: Tuple[int, int, int], thickness: int = 2) -> None:
    if w > 0 and h > 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def _lab_get_blobs(crop_img) -> List[Tuple[int, int, int, int]]:
    """Return surviving blob bounding boxes (bx, by, bw, bh) from the same mask
    used in _extract_per_blob_qty_slots / _estimate_visible_part_count_from_crop."""
    try:
        if crop_img is None or crop_img.size == 0:
            return []
        height, width = crop_img.shape[:2]
        roi = crop_img
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # detect anything darker than background
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            8,
        )
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float(max(1, roi.shape[0] * roi.shape[1]))
        blobs: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 20 or area > roi_area * 0.25:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < 3 or bh < 3:
                continue
            if bw > width * 0.70 and bh < 18:
                continue
            blobs.append((bx, by, bw, bh))
        blobs.sort(key=lambda b: b[0])
        return blobs
    except Exception:
        return []


def _build_step_region(
    page_width: int,
    page_height: int,
    sb: Dict[str, Any],
) -> Optional[Tuple[int, int, int, int]]:
    """Replicate the padded region logic from _build_instruction_callout_crops."""
    try:
        x = int(sb.get("x", 0) or 0)
        y = int(sb.get("y", 0) or 0)
        w = int(sb.get("w", 0) or 0)
        h = int(sb.get("h", 0) or 0)
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    pad_left = max(18, int(w * 0.8))
    pad_above = max(75, int(h * 5.0))
    pad_right = max(220, int(w * 8.0))
    x1 = max(0, x - pad_left)
    y1 = max(0, y - pad_above)
    x2 = min(page_width, x + w + pad_right)
    y2 = min(page_height, y)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _img_tag(data_uri: str, label: str = "", max_width: int = 600) -> str:
    if not data_uri:
        return f"<div class='missing'>no image: {escape(label)}</div>"
    return (
        f"<figure>"
        f"<img src='{data_uri}' style='max-width:{max_width}px;border:1px solid #555'>"
        f"<figcaption>{escape(label)}</figcaption>"
        f"</figure>"
    )


def _qty_badge(qty_text: List[str], qty_numbers: List[int]) -> str:
    if not qty_text:
        return "<span class='badge empty'>no qty detected</span>"
    parts = []
    for t, n in zip(qty_text, qty_numbers):
        parts.append(f"<span class='badge'>{escape(str(t))} ({n})</span>")
    return " ".join(parts)


def _lab_openai_crop_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "box_ok": {"type": "boolean"},
            "part_count": {"type": "integer"},
            "qty_labels": {
                "type": "array",
                "items": {"type": "string"},
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["box_ok", "part_count", "qty_labels", "issues"],
    }


def _lab_openai_crop_prompt(set_num: str, page: int, step_num: int) -> str:
    return (
        "You are checking a debug crop from a LEGO instruction callout. "
        "Return JSON only. "
        "Decide whether the crop box is usable, count visible part images, list visible quantity labels like 1x or 2x, "
        "and list any issues such as crop too loose, includes yellow substep, top border cut, missing parts, or OCR-only text. "
        "Be conservative and do not invent hidden parts or labels. "
        "Set number: %s. Page: %d. Step: %d."
    ) % (set_num, int(page), int(step_num))


def _lab_analyze_crop_with_openai(crop_img, set_num: str, page: int, step_num: int) -> Dict[str, Any]:
    if crop_img is None or crop_img.size == 0:
        return {"box_ok": False, "part_count": 0, "qty_labels": [], "issues": ["empty crop"]}

    client = _require_openai_vision_client_debug()
    ok, buf = cv2.imencode(".png", crop_img)
    if not ok:
        raise RuntimeError("Could not encode crop image for OpenAI request.")

    response = client.responses.create(
        model=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1"),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _lab_openai_crop_prompt(set_num, int(page), int(step_num)),
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii"),
                        "detail": "high",
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "lego_callout_crop_debug",
                "strict": True,
                "schema": _lab_openai_crop_schema(),
            }
        },
    )
    payload = _response_text_to_json_debug(response)
    return {
        "box_ok": bool(payload.get("box_ok", False)),
        "part_count": int(payload.get("part_count", 0) or 0),
        "qty_labels": [str(v) for v in (payload.get("qty_labels", []) or []) if str(v)],
        "issues": [str(v) for v in (payload.get("issues", []) or []) if str(v)],
    }


def _lab_render_ai_result(ai_result: Optional[Dict[str, Any]], ai_status: str) -> str:
    if ai_status == "disabled":
        return "<p><b>OpenAI:</b> OpenAI disabled</p>"
    if ai_status.startswith("error:"):
        return f"<p><b>OpenAI:</b> {escape(ai_status[6:])}</p>"
    if not ai_result:
        return ""
    qty_labels = ", ".join(escape(str(v)) for v in (ai_result.get("qty_labels", []) or [])) or "none"
    issues = ", ".join(escape(str(v)) for v in (ai_result.get("issues", []) or [])) or "none"
    return (
        f"<p><b>OpenAI:</b> box_ok={bool(ai_result.get('box_ok', False))} "
        f"| part_count={int(ai_result.get('part_count', 0) or 0)}</p>"
        f"<p><b>OpenAI qty labels:</b> {qty_labels}</p>"
        f"<p><b>OpenAI issues:</b> {issues}</p>"
    )


# ---------------------------------------------------------------------------
# Main route
# ---------------------------------------------------------------------------

@router.get("/debug/callout-crop-lab", response_class=HTMLResponse)
def callout_crop_lab(
    set_num: str = Query(...),
    bag: int = Query(...),
    ai: int = Query(0),
):
    rendered_pages, start_page, end_page = _resolve_bag_page_range(str(set_num), int(bag))
    ai_enabled = int(ai or 0) == 1
    openai_available = bool(os.getenv("OPENAI_API_KEY"))
    ai_cache: Dict[str, Dict[str, Any]] = {}

    sections: List[str] = []

    for page in rendered_pages:
        if int(page) < int(start_page) or int(page) > int(end_page):
            continue

        image_path = debug_service.resolve_page_image_path(str(set_num), int(page))
        if image_path is None:
            sections.append(f"<section><h2>Page {page}</h2><p class='warn'>image not found</p></section>")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            sections.append(f"<section><h2>Page {page}</h2><p class='warn'>cv2 could not read image</p></section>")
            continue

        page_height, page_width = img.shape[:2]
        detected = step_detector_service.detect_steps(str(set_num), int(page))
        step_boxes = _contact_sheet_step_boxes_from_detected(detected)

        if not step_boxes:
            sections.append(f"<section><h2>Page {page}</h2><p class='warn'>no step boxes detected</p></section>")
            continue

        # Page overview with all step boxes drawn
        page_overview = img.copy()
        for sb in step_boxes:
            sx = int(sb.get("x", 0) or 0)
            sy = int(sb.get("y", 0) or 0)
            sw = int(sb.get("w", 0) or 0)
            sh = int(sb.get("h", 0) or 0)
            sn = int(sb.get("step_number", 0) or 0)
            _draw_rect(page_overview, sx, sy, sw, sh, (40, 200, 40), 3)
            cv2.putText(page_overview, f"S{sn}", (sx, max(0, sy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 200, 40), 2)
        page_uri = _lab_encode(page_overview, max_width=700)

        step_cards: List[str] = []
        for sb in step_boxes:
            step_num = int(sb.get("step_number", 0) or 0)
            sx = int(sb.get("x", 0) or 0)
            sy = int(sb.get("y", 0) or 0)
            sw = int(sb.get("w", 0) or 0)
            sh = int(sb.get("h", 0) or 0)

            region = _build_step_region(page_width, page_height, sb)
            if region is None:
                step_cards.append(
                    f"<div class='step-card'><h3>Step {step_num}</h3>"
                    f"<p class='warn'>Could not build step region</p></div>"
                )
                continue

            x1, y1, x2, y2 = region

            # --- edge-detection pass ---
            edge_box = _detect_callout_rect_by_edges(
                img, [x1, y1, x2, y2], int(sb.get("y", 0) or 0), page_width, page_height
            )

            # Draw: step region (grey), step box (green), callout rect (blue)
            region_overlay = img.copy()
            _draw_rect(region_overlay, x1, y1, x2 - x1, y2 - y1, (160, 160, 160), 1)
            _draw_rect(region_overlay, sx, sy, sw, sh, (40, 200, 40), 3)
            cv2.putText(region_overlay, f"S{step_num} box", (sx, max(0, sy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 200, 40), 2)

            callout_status = "none"
            final_crop = None
            final_coords: Optional[Tuple[int, int, int, int]] = None

            if edge_box is not None:
                eax, eay, eaw, eah = edge_box
                _draw_rect(region_overlay, eax, eay, eaw, eah, (255, 100, 0), 3)
                cv2.putText(region_overlay, "callout", (eax, max(0, eay - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                callout_status = f"detected x={eax} y={eay} w={eaw} h={eah}"
                ebx = min(page_width, eax + eaw)
                eby = min(page_height, eay + eah)
                if ebx > eax and eby > eay:
                    final_crop = img[eay:eby, eax:ebx]
                    final_coords = (eax, eay, eaw, eah)
            else:
                callout_status = "not detected (edge pass)"

            region_uri = _lab_encode(region_overlay, max_width=700)

            # --- crop detail: blobs + tokens ---
            crop_detail_html = ""
            qty_html = "<span class='badge empty'>no crop</span>"
            qty_source = "local"
            if final_crop is not None and final_crop.size > 0:
                crop_vis = final_crop.copy()
                blobs = _lab_get_blobs(final_crop)
                tokens = _extract_qty_tokens_from_image(final_crop)
                ai_html = ""

                # Draw blobs in yellow
                for bx, by, bw, bh in blobs:
                    _draw_rect(crop_vis, bx, by, bw, bh, (0, 220, 220), 2)

                # Draw OCR token boxes in magenta
                for tok in tokens:
                    tx = int(tok.get("x", 0))
                    ty = int(tok.get("y", 0))
                    tw = int(tok.get("w", 0))
                    th = int(tok.get("h", 0))
                    text = str(tok.get("text", ""))
                    _draw_rect(crop_vis, tx, ty, tw, th, (220, 0, 220), 2)
                    cv2.putText(crop_vis, text, (tx, max(0, ty - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 0, 220), 1)

                crop_uri = _lab_encode(crop_vis, max_width=420)
                blob_summary = (
                    f"{len(blobs)} blob(s): " +
                    ", ".join(f"[{bx},{by},{bw},{bh}]" for bx, by, bw, bh in blobs)
                    if blobs else "0 blobs"
                )
                token_summary = (
                    f"{len(tokens)} token(s): " +
                    ", ".join(escape(str(t.get("text", ""))) for t in tokens)
                    if tokens else "0 tokens"
                )

                qty_result = _extract_detected_qty_details_from_crop(final_crop)

                if ai_enabled:
                    if not openai_available:
                        ai_html = _lab_render_ai_result(None, "disabled")
                    else:
                        crop_id = (
                            f"{page}:{step_num}:{final_coords[0]}:{final_coords[1]}:"
                            f"{final_coords[2]}:{final_coords[3]}"
                            if final_coords
                            else f"{page}:{step_num}:none"
                        )
                        if crop_id not in ai_cache:
                            try:
                                ai_cache[crop_id] = _lab_analyze_crop_with_openai(
                                    final_crop,
                                    str(set_num),
                                    int(page),
                                    int(step_num),
                                )
                            except Exception as exc:
                                ai_html = _lab_render_ai_result(None, f"error:{str(exc)}")
                        if not ai_html:
                            ai_html = _lab_render_ai_result(ai_cache.get(crop_id), "ok")
                        ai_result = ai_cache.get(crop_id) or {}
                        if ai_result.get("box_ok") is True:
                            qty_labels = list(ai_result.get("qty_labels") or [])
                            qty_numbers: List[int] = []
                            for label in qty_labels:
                                match = re.match(r"^\s*(\d+)\s*x\s*$", str(label), flags=re.IGNORECASE)
                                if match:
                                    qty_numbers.append(int(match.group(1)))
                            qty_result = {
                                "detected_qty_text": qty_labels,
                                "detected_qty_numbers": qty_numbers,
                                "source": "openai",
                            }
                            qty_source = "openai"

                qty_html = _qty_badge(
                    qty_result.get("detected_qty_text", []),
                    qty_result.get("detected_qty_numbers", []),
                )

                crop_detail_html = f"""
                <div class='crop-detail'>
                  {_img_tag(crop_uri, "crop (cyan=blob, magenta=OCR token)", 420)}
                  <p><b>Blobs:</b> {escape(blob_summary)}</p>
                  <p><b>OCR tokens:</b> {escape(token_summary)}</p>
                  {ai_html}
                </div>"""
            else:
                crop_detail_html = "<p class='warn'>no final crop</p>"

            fcoords_str = (
                f"x={final_coords[0]} y={final_coords[1]} w={final_coords[2]} h={final_coords[3]}"
                if final_coords else "—"
            )

            step_cards.append(f"""
            <div class='step-card'>
              <h3>Step {step_num} &nbsp;<small>step box: x={sx} y={sy} w={sw} h={sh}</small></h3>
              <div class='row'>
                {_img_tag(region_uri, f"page region — callout: {callout_status}", 700)}
                {crop_detail_html}
              </div>
              <p><b>Final crop coords:</b> {escape(fcoords_str)}</p>
              <p><b>Computed qty slots:</b> {qty_html} <small>(source: {escape(qty_source)})</small></p>
            </div>""")

        sections.append(f"""
        <section>
          <h2>Page {page}</h2>
          {_img_tag(page_uri, f"page {page} — all step boxes (green)", 700)}
          {''.join(step_cards)}
        </section>""")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Callout Crop Lab — {escape(str(set_num))} bag {bag}</title>
<style>
  body {{ font-family: monospace; background: #1a1a1a; color: #ddd; padding: 16px; }}
  h1 {{ color: #8cf; }}
  h2 {{ color: #fc8; border-top: 1px solid #444; padding-top: 8px; }}
  h3 {{ color: #aef; margin-bottom: 4px; }}
  small {{ color: #888; font-size: 0.85em; }}
  section {{ margin-bottom: 32px; }}
  .step-card {{ background: #252525; border: 1px solid #444; border-radius: 4px;
                padding: 12px; margin: 8px 0; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }}
  figure {{ margin: 0; }}
  figcaption {{ font-size: 0.8em; color: #aaa; max-width: 700px; }}
  .badge {{ background: #2a4a2a; border: 1px solid #4a8; border-radius: 3px;
            padding: 2px 6px; margin-right: 4px; font-size: 0.9em; }}
  .badge.empty {{ background: #3a2a2a; border-color: #844; color: #a88; }}
  .warn {{ color: #f88; }}
  .missing {{ color: #888; font-style: italic; }}
  .crop-detail {{ display: flex; flex-direction: column; gap: 6px; }}
</style>
</head>
<body>
<h1>Callout Crop Lab — set {escape(str(set_num))} bag {bag}</h1>
<p>Pages {start_page}–{end_page} &nbsp;|&nbsp;
   <b>Legend:</b>
   <span style='color:#28c828'>■ step box</span> &nbsp;
   <span style='color:#8080ff'>■ callout rect</span> &nbsp;
   <span style='color:#00dcdc'>■ blob</span> &nbsp;
   <span style='color:#dc00dc'>■ OCR token</span>
</p>
{''.join(sections)}
</body>
</html>"""

    return HTMLResponse(content=html)
