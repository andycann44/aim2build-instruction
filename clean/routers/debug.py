import base64
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from fastapi import APIRouter, HTTPException, Query, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, Response

from clean.services import (
    analyzer_scan_service,
    bag_truth_store,
    inventory_scan_service,
    gap_scan_service,
    step_detector_service,
    step_sequence_bag_service,
    truth_service,
    debug_service,
    precheck_service,
    page_analyzer,
)

router = APIRouter()


def _list_rendered_pages(pages_dir: Path) -> List[int]:
    pages: List[int] = []
    for path in sorted(pages_dir.glob("page_*.png")):
        try:
            pages.append(int(path.stem.replace("page_", "")))
        except ValueError:
            continue
    return pages


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
_PART_IMAGE_CACHE: Dict[str, Optional[Any]] = {}
_PART_IMAGE_FEATURE_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
_OPENAI_CALLOUT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")


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


def _normalize_contact_sheet_box(
    item: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    raw_box = item.get("box")
    if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
        x = int(raw_box[0] or 0)
        y = int(raw_box[1] or 0)
        w = int(raw_box[2] or 0)
        h = int(raw_box[3] or 0)
    else:
        x = int(item.get("x", 0) or 0)
        y = int(item.get("y", 0) or 0)
        w = int(item.get("w", 0) or 0)
        h = int(item.get("h", 0) or 0)

    if w <= 0 or h <= 0:
        return None

    step_number = int(
        item.get("step_number", item.get("value", item.get("step", 0))) or 0
    )
    source = str(item.get("source", item.get("step_group", "unknown")) or "unknown")
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "step_number": step_number,
        "source": source,
        "label": str(item.get("text", "") or ""),
    }


def _clamp_contact_sheet_box(
    box: List[int],
    page_width: int,
    page_height: int,
) -> Optional[List[int]]:
    x = max(0, int(box[0] or 0))
    y = max(0, int(box[1] or 0))
    w = max(0, int(box[2] or 0))
    h = max(0, int(box[3] or 0))
    if w <= 0 or h <= 0:
        return None

    if x >= int(page_width) or y >= int(page_height):
        return None

    w = min(w, int(page_width) - x)
    h = min(h, int(page_height) - y)
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def _encode_contact_sheet_crop(
    img,
    box: List[int],
    max_edge: int = 320,
) -> Optional[str]:
    crop_img = page_analyzer.crop(img, box)
    if crop_img is None or crop_img.size == 0:
        return None

    height, width = crop_img.shape[:2]
    longest_edge = max(int(width), int(height), 1)
    if longest_edge > int(max_edge):
        scale = float(max_edge) / float(longest_edge)
        crop_img = cv2.resize(
            crop_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        crop_img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 84],
    )
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _contact_sheet_step_boxes_from_green_row(
    green_row: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    for item in ((green_row or {}).get("main_steps", []) or []):
        normalized = _normalize_contact_sheet_box(item)
        if normalized is None:
            continue
        normalized["source"] = "green_step_debug"
        boxes.append(normalized)
    return sorted(boxes, key=lambda item: (item["y"], item["x"], item["step_number"]))


def _contact_sheet_step_boxes_from_detected(
    detected: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    for item in ((detected or {}).get("classified_step_boxes", []) or []):
        if str(item.get("step_group", "")) != "main_steps":
            continue
        normalized = _normalize_contact_sheet_box(item)
        if normalized is None:
            continue
        boxes.append(normalized)
    return sorted(boxes, key=lambda item: (item["y"], item["x"], item["step_number"]))


def _build_step_anchor_tiles(
    img,
    page_width: int,
    page_height: int,
    step_boxes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    tiles: List[Dict[str, Any]] = []
    for item in step_boxes:
        x = int(item.get("x", 0) or 0)
        y = int(item.get("y", 0) or 0)
        w = int(item.get("w", 0) or 0)
        h = int(item.get("h", 0) or 0)
        step_number = int(item.get("step_number", 0) or 0)
        crop_box = _clamp_contact_sheet_box(
            [
                max(0, x - max(28, w * 4)),
                max(0, y - max(24, h * 2)),
                max(150, w + max(56, w * 8)),
                max(120, h + max(48, h * 4)),
            ],
            page_width=page_width,
            page_height=page_height,
        )
        if crop_box is None:
            continue

        data_uri = _encode_contact_sheet_crop(img, crop_box, max_edge=260)
        if data_uri is None:
            continue

        tiles.append(
            {
                "step_number": step_number,
                "label": (
                    f"Step {step_number}" if step_number > 0 else "Unknown step"
                ),
                "coords": crop_box,
                "source": str(item.get("source", "step_debug") or "step_debug"),
                "data_uri": data_uri,
            }
        )
    return tiles


def _box_iou_debug(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, aw, ah = [int(value or 0) for value in box_a[:4]]
    bx1, by1, bw, bh = [int(value or 0) for value in box_b[:4]]
    ax2 = ax1 + max(0, aw)
    ay2 = ay1 + max(0, ah)
    bx2 = bx1 + max(0, bw)
    by2 = by1 + max(0, bh)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(1, max(0, aw) * max(0, ah))
    area_b = max(1, max(0, bw) * max(0, bh))
    union = float(area_a + area_b - inter_area)
    if union <= 0:
        return 0.0
    return float(inter_area) / union


def _build_step_panel_specs(
    page_width: int,
    page_height: int,
    step_boxes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    panel_specs: List[Dict[str, Any]] = []

    if step_boxes:
        ordered = sorted(step_boxes, key=lambda item: (item["y"], item["x"]))
        for idx, item in enumerate(ordered):
            step_number = int(item.get("step_number", 0) or 0)
            y = int(item.get("y", 0) or 0)
            h = int(item.get("h", 0) or 0)
            next_y = (
                int(ordered[idx + 1].get("y", 0) or 0)
                if idx + 1 < len(ordered)
                else int(page_height)
            )

            panel_top = max(0, y - max(44, h * 3))
            panel_bottom = min(
                int(page_height),
                max(panel_top + 160, next_y - max(24, h // 2)),
            )
            panel_height = max(140, panel_bottom - panel_top)
            panel_box = _clamp_contact_sheet_box(
                [
                    0,
                    panel_top,
                    max(200, int(page_width * 0.50)),
                    panel_height,
                ],
                page_width=page_width,
                page_height=page_height,
            )
            if panel_box is None:
                continue
            panel_specs.append(
                {
                    "step_number": step_number,
                    "label": f"step_panel_{step_number or '?'}",
                    "coords": panel_box,
                }
            )
        return panel_specs

    fallback_panels = [
        [0, 0, int(page_width * 0.58), int(page_height * 0.28)],
        [0, int(page_height * 0.18), int(page_width * 0.58), int(page_height * 0.28)],
        [0, int(page_height * 0.36), int(page_width * 0.58), int(page_height * 0.22)],
    ]
    for idx, raw_box in enumerate(fallback_panels, start=1):
        panel_box = _clamp_contact_sheet_box(
            raw_box,
            page_width=page_width,
            page_height=page_height,
        )
        if panel_box is None:
            continue
        panel_specs.append(
            {
                "step_number": 0,
                "label": f"fallback_panel_{idx}",
                "coords": panel_box,
            }
        )
    return panel_specs


def _build_legacy_material_candidate_specs(
    page_width: int,
    page_height: int,
    step_boxes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidate_specs: List[Dict[str, Any]] = []

    if step_boxes:
        ordered = sorted(step_boxes, key=lambda item: (item["y"], item["x"]))
        for item in ordered:
            step_number = int(item.get("step_number", 0) or 0)
            y = int(item.get("y", 0) or 0)
            h = int(item.get("h", 0) or 0)
            panel_top = max(0, y - max(44, h * 3))
            wide_box = _clamp_contact_sheet_box(
                [
                    0,
                    panel_top,
                    max(120, int(page_width * 0.36)),
                    max(96, int(page_height * 0.16)),
                ],
                page_width=page_width,
                page_height=page_height,
            )
            tight_box = _clamp_contact_sheet_box(
                [
                    0,
                    panel_top,
                    max(100, int(page_width * 0.24)),
                    max(84, int(page_height * 0.12)),
                ],
                page_width=page_width,
                page_height=page_height,
            )
            if wide_box is not None:
                candidate_specs.append(
                    {
                        "step_number": step_number,
                        "label": f"legacy_top_left_candidate_wide_{step_number or '?'}",
                        "coords": wide_box,
                        "candidate_origin": "legacy_top_left_candidate",
                    }
                )
            if tight_box is not None:
                candidate_specs.append(
                    {
                        "step_number": step_number,
                        "label": f"legacy_top_left_candidate_tight_{step_number or '?'}",
                        "coords": tight_box,
                        "candidate_origin": "legacy_top_left_candidate",
                    }
                )

    generic_boxes = [
        [0, 0, int(page_width * 0.34), int(page_height * 0.14)],
        [0, int(page_height * 0.16), int(page_width * 0.30), int(page_height * 0.14)],
        [0, int(page_height * 0.32), int(page_width * 0.30), int(page_height * 0.14)],
        [0, int(page_height * 0.56), int(page_width * 0.30), int(page_height * 0.14)],
    ]
    for idx, raw_box in enumerate(generic_boxes, start=1):
        box = _clamp_contact_sheet_box(
            raw_box,
            page_width=page_width,
            page_height=page_height,
        )
        if box is None:
            continue
        candidate_specs.append(
            {
                "step_number": 0,
                "label": f"legacy_top_left_candidate_{idx}",
                "coords": box,
                "candidate_origin": "legacy_top_left_candidate",
            }
        )
    return candidate_specs


def _ocr_large_step_anchors(
    img,
    page_width: int,
    page_height: int,
) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anchors: List[Dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = int(w * h)
        if h < 20 or h > 70 or w < 6 or w > 52:
            continue
        if area < 120 or area > 2500:
            continue
        if y >= int(page_height * 0.92):
            continue
        if y > int(page_height * 0.88) and (
            x < int(page_width * 0.18) or x > int(page_width * 0.82)
        ):
            continue

        crop = gray[
            max(0, y - 4) : min(gray.shape[0], y + h + 4),
            max(0, x - 4) : min(gray.shape[1], x + w + 4),
        ]
        if crop.size == 0:
            continue
        up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        text = (
            pytesseract.image_to_string(
                up,
                config="--psm 10 -c tessedit_char_whitelist=0123456789",
            )
            or ""
        ).strip()
        if not re.match(r"^\d{1,2}$", text):
            continue

        value = int(text)
        if value <= 0 or value > 99:
            continue

        candidate = {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "step_number": int(value),
            "source": "ocr_anchor",
            "label": text,
        }

        duplicate = False
        for existing in anchors:
            if _box_iou_debug(
                [candidate["x"], candidate["y"], candidate["w"], candidate["h"]],
                [existing["x"], existing["y"], existing["w"], existing["h"]],
            ) >= 0.35:
                duplicate = True
                break
        if duplicate:
            continue
        anchors.append(candidate)

    return sorted(anchors, key=lambda item: (item["y"], item["x"], item["step_number"]))


def _find_step_search_anchors(
    img,
    page_width: int,
    page_height: int,
    step_boxes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    anchors = sorted(
        [dict(item) for item in (step_boxes or [])],
        key=lambda item: (item["y"], item["x"], item["step_number"]),
    )
    ocr_anchors = _ocr_large_step_anchors(
        img,
        page_width=page_width,
        page_height=page_height,
    )
    if not anchors:
        return ocr_anchors

    for candidate in ocr_anchors:
        duplicate = False
        for existing in anchors:
            if _box_iou_debug(
                [candidate["x"], candidate["y"], candidate["w"], candidate["h"]],
                [existing["x"], existing["y"], existing["w"], existing["h"]],
            ) >= 0.35:
                duplicate = True
                break
        if not duplicate:
            anchors.append(candidate)
    return sorted(anchors, key=lambda item: (item["y"], item["x"], item["step_number"]))


def _build_callout_search_regions(
    page_width: int,
    page_height: int,
    step_anchors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    regions: List[Dict[str, Any]] = []
    for anchor in step_anchors:
        x = int(anchor.get("x", 0) or 0)
        y = int(anchor.get("y", 0) or 0)
        search_box = _clamp_contact_sheet_box(
            [
                x - 220,
                y - 260,
                640,
                340,
            ],
            page_width=page_width,
            page_height=page_height,
        )
        if search_box is None:
            continue
        regions.append(
            {
                "step_number": int(anchor.get("step_number", 0) or 0),
                "coords": search_box,
                "anchor": anchor,
                "search_region": "step_anchor_zone",
            }
        )
    return regions


def _detect_callout_box_candidate_specs(
    img,
    page_width: int,
    page_height: int,
    step_boxes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_specs: List[Dict[str, Any]] = []
    rejected_specs: List[Dict[str, Any]] = []
    step_anchors = _find_step_search_anchors(
        img,
        page_width=page_width,
        page_height=page_height,
        step_boxes=step_boxes,
    )
    search_regions = _build_callout_search_regions(
        page_width=page_width,
        page_height=page_height,
        step_anchors=step_anchors,
    )

    def _append_rejection(
        region: Dict[str, Any],
        coords: List[int],
        reason: str,
    ) -> None:
        if len(coords) < 4:
            return
        abs_box = _clamp_contact_sheet_box(
            coords,
            page_width=page_width,
            page_height=page_height,
        )
        if abs_box is None:
            return
        rejected_specs.append(
            {
                "coords": abs_box,
                "reason": str(reason or "rejected"),
                "step_number": int(region.get("step_number", 0) or 0),
                "search_zone": list(region.get("coords", []) or []),
                "step_anchor": dict(region.get("anchor", {}) or {}),
                "candidate_origin": "rejected_callout_candidate",
            }
        )

    for region in search_regions:
        region_coords = list(region.get("coords", []) or [])
        if len(region_coords) < 4:
            continue
        px, py, pw, ph = [int(value or 0) for value in region_coords[:4]]
        anchor = dict(region.get("anchor", {}) or {})
        step_x = int(anchor.get("x", 0) or 0)
        step_y = int(anchor.get("y", 0) or 0)
        step_w = int(anchor.get("w", 0) or 0)
        step_h = int(anchor.get("h", 0) or 0)
        step_center_x = step_x + (step_w // 2)
        step_top_y = step_y
        step_bottom_y = step_y + step_h
        step_left_x = step_x
        region_img = page_analyzer.crop(img, region_coords)
        if region_img is None or region_img.size == 0:
            continue
        qty_tokens = _extract_qty_tokens_from_image(region_img)
        if not qty_tokens:
            continue

        hsv = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        hue = hsv[:, :, 0]
        search_area = float(max(1, pw * ph))

        def _validate_local_candidate(
            local_box: List[int],
            max_area_ratio: float = 0.35,
        ) -> Optional[Dict[str, Any]]:
            x, y, w, h = [int(value or 0) for value in local_box[:4]]
            if w <= 0 or h <= 0:
                return None

            if y > int(ph * 0.62):
                return None

            box_area = float(w * h)
            area_ratio = box_area / search_area
            if area_ratio < 0.02 or area_ratio > float(max_area_ratio):
                return None

            aspect_ratio = w / float(max(h, 1))
            if aspect_ratio < 1.2 or aspect_ratio > 4.8:
                return None

            if w < max(50, int(pw * 0.10)) or h < max(24, int(ph * 0.06)):
                return None

            box_gray = gray[y : y + h, x : x + w]
            box_saturation = saturation[y : y + h, x : x + w]
            box_value = value[y : y + h, x : x + w]
            box_hue = hue[y : y + h, x : x + w]
            if box_gray.size == 0 or box_saturation.size == 0 or box_value.size == 0:
                return None

            pad_x = max(2, int(w * 0.12))
            pad_y = max(2, int(h * 0.16))
            inner = box_gray[pad_y : max(pad_y + 1, h - pad_y), pad_x : max(pad_x + 1, w - pad_x)]
            inner_sat = box_saturation[pad_y : max(pad_y + 1, h - pad_y), pad_x : max(pad_x + 1, w - pad_x)]
            inner_val = box_value[pad_y : max(pad_y + 1, h - pad_y), pad_x : max(pad_x + 1, w - pad_x)]
            inner_hue = box_hue[pad_y : max(pad_y + 1, h - pad_y), pad_x : max(pad_x + 1, w - pad_x)]
            if inner.size == 0 or inner_sat.size == 0 or inner_val.size == 0:
                return None

            border_mask = np.ones((h, w), dtype=np.uint8)
            border_mask[pad_y : max(pad_y + 1, h - pad_y), pad_x : max(pad_x + 1, w - pad_x)] = 0
            border_pixels = box_gray[border_mask == 1]
            inner_edges = cv2.Canny(inner, 60, 150)
            mean_saturation = float(inner_sat.mean()) if inner_sat.size else 255.0
            mean_value = float(inner_val.mean()) if inner_val.size else 0.0
            bright_ratio = float(
                np.count_nonzero((inner_val >= 168) & (inner_sat <= 120))
            ) / float(max(1, inner_val.size))
            pale_blue_ratio = float(
                np.count_nonzero(
                    (
                        (inner_hue >= 82)
                        & (inner_hue <= 132)
                        & (inner_val >= 150)
                        & (inner_sat <= 140)
                    )
                    | ((inner_val >= 198) & (inner_sat <= 45))
                )
            ) / float(max(1, inner_val.size))
            blue_mean = float(region_img[y : y + h, x : x + w, 0].mean())
            red_mean = float(region_img[y : y + h, x : x + w, 2].mean())
            border_mean = float(border_pixels.mean()) if border_pixels.size else mean_value
            edge_density = float(np.count_nonzero(inner_edges)) / float(max(1, inner_edges.size))
            if mean_value < 148.0:
                return None
            if mean_saturation > 118.0:
                return None
            if bright_ratio < 0.22:
                return None
            if pale_blue_ratio < 0.42:
                return None
            if blue_mean < red_mean - 6.0:
                return None
            if border_mean > mean_value - 4.0:
                return None
            if edge_density > 0.26:
                return None

            abs_box = _clamp_contact_sheet_box(
                [px + x, py + y, w, h],
                page_width=page_width,
                page_height=page_height,
            )
            if abs_box is None:
                return None

            if abs_box[3] > int(page_height * 0.18) or abs_box[2] > int(page_width * 0.42):
                return None
            if abs_box[2] <= abs_box[3]:
                return None
            if abs_box[0] > (step_x + 420) or (abs_box[0] + abs_box[2]) < (step_x - 220):
                return None
            if abs_box[1] > (step_y + 90) or (abs_box[1] + abs_box[3]) < (step_y - 280):
                return None
            if _box_iou_debug(
                abs_box,
                [
                    max(0, step_x - 40),
                    max(0, step_y - 20),
                    max(120, step_w + 200),
                    max(110, step_h + 120),
                ],
            ) > 0.52:
                return None

            return {
                "local_box": [x, y, w, h],
                "abs_box": abs_box,
                "score": (
                    pale_blue_ratio * 2.0
                    + bright_ratio * 1.5
                    + max(0.0, mean_value - border_mean) / 32.0
                    - edge_density
                ),
            }

        def _expand_to_full_callout_candidate(
            seed_payload: Dict[str, Any],
            token_cx: int,
            token_cy: int,
        ) -> Dict[str, Any]:
            seed_local_box = list(seed_payload.get("local_box", []) or [])
            if len(seed_local_box) < 4:
                return seed_payload

            sx, sy, sw, sh = [int(value or 0) for value in seed_local_box[:4]]
            proposal_box = _clamp_contact_sheet_box(
                [
                    max(0, sx - max(70, sw // 2)),
                    max(0, sy - max(56, sh // 2)),
                    max(sw + max(150, sw), 180),
                    max(sh + max(110, sh), 110),
                ],
                page_width=pw,
                page_height=ph,
            )
            if proposal_box is None:
                return seed_payload

            proposal_img = page_analyzer.crop(region_img, proposal_box)
            if proposal_img is None or proposal_img.size == 0:
                return seed_payload

            proposal_gray = cv2.cvtColor(proposal_img, cv2.COLOR_BGR2GRAY)
            proposal_edges = cv2.Canny(
                cv2.GaussianBlur(proposal_gray, (5, 5), 0),
                28,
                96,
            )
            proposal_edges = cv2.dilate(
                proposal_edges,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
            contours, _ = cv2.findContours(
                proposal_edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            ox, oy, _, _ = proposal_box
            seed_area = float(max(1, sw * sh))
            seed_center_x = sx + (sw // 2)
            seed_center_y = sy + (sh // 2)
            best_payload = dict(seed_payload)
            best_score = float(seed_payload.get("score", 0.0) or 0.0)
            contour_boxes: List[Dict[str, Any]] = []

            for contour in contours:
                cx, cy, cw, ch = cv2.boundingRect(contour)
                local_box = [ox + cx, oy + cy, cw, ch]
                if cw <= 0 or ch <= 0:
                    continue
                if cw < sw or ch < sh:
                    continue
                if (cw * ch) <= (seed_area * 1.12):
                    continue
                if not (
                    local_box[0] <= token_cx <= local_box[0] + local_box[2]
                    and local_box[1] <= token_cy <= local_box[1] + local_box[3]
                ):
                    continue
                if not (
                    local_box[0] <= seed_center_x <= local_box[0] + local_box[2]
                    and local_box[1] <= seed_center_y <= local_box[1] + local_box[3]
                ):
                    continue
                if local_box[0] > sx + 8 or local_box[1] > sy + 8:
                    continue
                if (local_box[0] + local_box[2]) < (sx + sw - 8):
                    continue
                if (local_box[1] + local_box[3]) < (sy + sh - 8):
                    continue

                contour_area = float(cv2.contourArea(contour))
                box_area = float(max(1, cw * ch))
                fill_ratio = contour_area / box_area
                aspect_ratio = cw / float(max(ch, 1))
                if fill_ratio < 0.01 or fill_ratio > 0.20:
                    continue

                contour_boxes.append(
                    {
                        "local_box": local_box,
                        "fill_ratio": fill_ratio,
                        "aspect_ratio": aspect_ratio,
                    }
                )

                if aspect_ratio < 1.05 or aspect_ratio > 4.8:
                    continue

                validated = _validate_local_candidate(
                    local_box,
                    max_area_ratio=0.70,
                )
                if validated is None:
                    continue

                validated["inner_local_box"] = seed_local_box
                validated["inner_abs_box"] = list(seed_payload.get("abs_box", []) or [])
                score = float(validated.get("score", 0.0) or 0.0)
                score += min(1.2, box_area / max(1.0, seed_area * 2.0))
                score -= fill_ratio * 2.0
                if score > best_score:
                    best_payload = validated
                    best_score = score

            contour_boxes = sorted(
                contour_boxes,
                key=lambda item: int((item.get("local_box") or [0])[0] or 0),
            )
            for start_idx in range(len(contour_boxes)):
                merged_box = list(contour_boxes[start_idx].get("local_box", []) or [])
                if len(merged_box) < 4:
                    continue
                for end_idx in range(start_idx, min(len(contour_boxes), start_idx + 3)):
                    current_box = list(contour_boxes[end_idx].get("local_box", []) or [])
                    if len(current_box) < 4:
                        continue
                    if end_idx > start_idx:
                        mx, my, mw, mh = merged_box
                        cx, cy, cw, ch = current_box
                        gap = cx - (mx + mw)
                        y_overlap = max(0, min(my + mh, cy + ch) - max(my, cy))
                        if gap > max(42, sw // 2):
                            break
                        if y_overlap < int(min(mh, ch) * 0.35):
                            continue
                        x1 = min(mx, cx)
                        y1 = min(my, cy)
                        x2 = max(mx + mw, cx + cw)
                        y2 = max(my + mh, cy + ch)
                        merged_box = [x1, y1, x2 - x1, y2 - y1]

                    if not (
                        merged_box[0] <= token_cx <= merged_box[0] + merged_box[2]
                        and merged_box[1] <= token_cy <= merged_box[1] + merged_box[3]
                    ):
                        continue
                    if not (
                        merged_box[0] <= seed_center_x <= merged_box[0] + merged_box[2]
                        and merged_box[1] <= seed_center_y <= merged_box[1] + merged_box[3]
                    ):
                        continue
                    validated = _validate_local_candidate(
                        merged_box,
                        max_area_ratio=0.72,
                    )
                    if validated is None:
                        continue
                    validated["inner_local_box"] = seed_local_box
                    validated["inner_abs_box"] = list(seed_payload.get("abs_box", []) or [])
                    score = float(validated.get("score", 0.0) or 0.0)
                    score += min(
                        1.5,
                        (float(merged_box[2] * merged_box[3]) / max(1.0, seed_area * 2.0)),
                    )
                    if score > best_score:
                        best_payload = validated
                        best_score = score

            cluster_tokens: List[Dict[str, Any]] = []
            for nearby_token in qty_tokens:
                nearby_cx = int(nearby_token.get("cx", 0) or 0)
                nearby_cy = int(nearby_token.get("cy", 0) or 0)
                if abs(nearby_cy - token_cy) > max(84, sh):
                    continue
                if nearby_cx < (sx - 24):
                    continue
                if nearby_cx > (sx + sw + max(140, sw)):
                    continue
                cluster_tokens.append(nearby_token)

            if cluster_tokens:
                cluster_x1 = sx
                cluster_y1 = sy
                cluster_x2 = sx + sw
                cluster_y2 = sy + sh
                cluster_outside = False
                for nearby_token in cluster_tokens:
                    tx0 = int(nearby_token.get("x", 0) or 0)
                    ty0 = int(nearby_token.get("y", 0) or 0)
                    tw0 = int(nearby_token.get("w", 0) or 0)
                    th0 = int(nearby_token.get("h", 0) or 0)
                    if (
                        tx0 < sx
                        or ty0 < sy
                        or (tx0 + tw0) > (sx + sw)
                        or (ty0 + th0) > (sy + sh)
                    ):
                        cluster_outside = True
                    cluster_x1 = min(cluster_x1, max(0, tx0 - max(24, tw0 * 2)))
                    cluster_y1 = min(cluster_y1, max(0, ty0 - max(18, th0 * 2)))
                    cluster_x2 = max(cluster_x2, tx0 + tw0 + max(28, tw0 * 3))
                    cluster_y2 = max(cluster_y2, ty0 + th0 + max(20, th0 * 2))

                if cluster_outside:
                    cluster_box = _clamp_contact_sheet_box(
                        [
                            cluster_x1,
                            cluster_y1,
                            cluster_x2 - cluster_x1,
                            cluster_y2 - cluster_y1,
                        ],
                        page_width=pw,
                        page_height=ph,
                    )
                    if cluster_box is not None:
                        validated = _validate_local_candidate(
                            cluster_box,
                            max_area_ratio=0.72,
                        )
                        if validated is not None:
                            validated["inner_local_box"] = seed_local_box
                            validated["inner_abs_box"] = list(seed_payload.get("abs_box", []) or [])
                            score = float(validated.get("score", 0.0) or 0.0)
                            score += 0.8 + (0.25 * len(cluster_tokens))
                            if score > best_score:
                                best_payload = validated
                                best_score = score

            if "inner_local_box" not in best_payload:
                best_payload["inner_local_box"] = seed_local_box
                best_payload["inner_abs_box"] = list(seed_payload.get("abs_box", []) or [])
            return best_payload

        seen_qty_tokens: set[Tuple[str, int, int]] = set()
        for token in qty_tokens:
            normalized_token = re.sub(r"\s+", "", str(token.get("text", "") or "").lower())
            token_key = (
                normalized_token,
                int((int(token.get("x", 0) or 0)) // 4),
                int((int(token.get("y", 0) or 0)) // 4),
            )
            if token_key in seen_qty_tokens:
                continue
            seen_qty_tokens.add(token_key)

            tx = int(token.get("x", 0) or 0)
            ty = int(token.get("y", 0) or 0)
            tw = int(token.get("w", 0) or 0)
            th = int(token.get("h", 0) or 0)
            token_cx = int(token.get("cx", tx + (tw // 2)) or 0)
            token_cy = int(token.get("cy", ty + (th // 2)) or 0)

            local_best: Optional[Dict[str, Any]] = None
            used_fallback_seed = False
            proposal_specs = [
                [
                    max(0, tx - max(165, tw * 10)),
                    max(0, ty - max(96, th * 8)),
                    max(150, tw * 12),
                    max(108, th * 10),
                ],
                [
                    max(0, tx - max(132, tw * 8)),
                    max(0, ty - max(84, th * 7)),
                    max(128, tw * 10),
                    max(96, th * 9),
                ],
            ]
            for raw_proposal in proposal_specs:
                proposal_box = _clamp_contact_sheet_box(
                    raw_proposal,
                    page_width=pw,
                    page_height=ph,
                )
                if proposal_box is None:
                    continue
                proposal_img = page_analyzer.crop(region_img, proposal_box)
                if proposal_img is None or proposal_img.size == 0:
                    continue
                proposal_gray = cv2.cvtColor(proposal_img, cv2.COLOR_BGR2GRAY)
                proposal_edges = cv2.Canny(
                    cv2.GaussianBlur(proposal_gray, (5, 5), 0),
                    40,
                    120,
                )
                proposal_edges = cv2.dilate(
                    proposal_edges,
                    np.ones((3, 3), np.uint8),
                    iterations=1,
                )
                contours, _ = cv2.findContours(
                    proposal_edges,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                ox, oy, _, _ = proposal_box
                for contour in contours:
                    cx, cy, cw, ch = cv2.boundingRect(contour)
                    local_box = [ox + cx, oy + cy, cw, ch]
                    if not (
                        local_box[0] <= token_cx <= local_box[0] + local_box[2]
                        and local_box[1] <= token_cy <= local_box[1] + local_box[3]
                    ):
                        continue
                    validated = _validate_local_candidate(local_box)
                    if validated is None:
                        continue
                    if local_best is None or float(validated["score"]) > float(local_best["score"]):
                        local_best = validated
                        used_fallback_seed = False

            if local_best is None:
                local_step_x = max(0, min(pw - 1, step_x - px))
                local_step_y = max(0, min(ph - 1, step_y - py))
                fallback_specs = [
                    [
                        max(0, local_step_x - max(34, tw * 2)),
                        max(0, local_step_y - max(172, th * 13)),
                        max(190, tw * 15),
                        max(130, th * 11),
                    ],
                    [
                        max(0, tx - max(168, tw * 11)),
                        max(0, ty - max(98, th * 8)),
                        max(170, tw * 13),
                        max(112, th * 10),
                    ],
                ]
                for fallback_spec in fallback_specs:
                    fallback_box = _clamp_contact_sheet_box(
                        fallback_spec,
                        page_width=pw,
                        page_height=ph,
                    )
                    if fallback_box is None:
                        continue
                    fallback_validated = _validate_local_candidate(fallback_box)
                    if fallback_validated is None:
                        continue
                    fallback_abs_box = list(fallback_validated.get("abs_box", []) or [])
                    if len(fallback_abs_box) < 4:
                        continue
                    fallback_bottom = int(fallback_abs_box[1]) + int(fallback_abs_box[3])
                    fallback_left = int(fallback_abs_box[0])
                    bottom_left_distance = abs(int(step_left_x) - fallback_left) + abs(int(step_bottom_y) - fallback_bottom)
                    fallback_score = float(fallback_validated.get("score", 0.0) or 0.0) - (float(bottom_left_distance) / 640.0)
                    if local_best is None or fallback_score > float(local_best.get("score", 0.0) or 0.0):
                        local_best = fallback_validated
                        local_best["score"] = fallback_score
                        used_fallback_seed = True

            if local_best is None:
                _append_rejection(
                    region,
                    [px + max(0, tx - 40), py + max(0, ty - 30), max(60, tw + 80), max(40, th + 50)],
                    "no_valid_seed_box",
                )
                continue

            local_best = _expand_to_full_callout_candidate(
                local_best,
                token_cx=token_cx,
                token_cy=token_cy,
            )

            abs_box = list(local_best["abs_box"])
            candidate_qty_texts = [
                str(token.get("text", ""))
                for token in qty_tokens
                if (
                    int(token.get("cx", 0) or 0) >= int(local_best["local_box"][0])
                    and int(token.get("cx", 0) or 0) <= int(local_best["local_box"][0] + local_best["local_box"][2])
                    and int(token.get("cy", 0) or 0) >= int(local_best["local_box"][1])
                    and int(token.get("cy", 0) or 0) <= int(local_best["local_box"][1] + local_best["local_box"][3])
                )
            ]
            crop_img = page_analyzer.crop(img, abs_box)
            qty_payload = _extract_crop_qty_status(crop_img)
            if qty_payload.get("qty") is not None:
                candidate_qty_texts.append(f"x{int(qty_payload['qty'])}")
            candidate_qty_texts = sorted(
                {
                    re.sub(r"\s+", "", str(text or "").lower())
                    for text in candidate_qty_texts
                    if str(text or "").strip()
                }
            )
            if not candidate_qty_texts:
                _append_rejection(region, abs_box, "no_qty_text_in_box")
                continue

            if abs_box[2] <= abs_box[3]:
                _append_rejection(region, abs_box, "width_not_greater_than_height")
                continue

            box_bottom = abs_box[1] + abs_box[3]
            box_left = abs_box[0]
            box_center_x = abs_box[0] + (abs_box[2] // 2)
            vertical_gap = abs(step_top_y - box_bottom)
            horizontal_gap = abs(step_center_x - box_center_x)
            bottom_left_distance = abs(step_left_x - box_left) + abs(step_bottom_y - box_bottom)
            above_step = box_bottom <= (step_top_y + 26)
            if not above_step and abs_box[1] > (step_top_y + 90):
                _append_rejection(region, abs_box, "too_far_below_step")
                continue

            candidate_score = float(local_best.get("score", 0.0) or 0.0)
            candidate_score += (1.2 if above_step else 0.0)
            candidate_score -= (vertical_gap / 120.0)
            candidate_score -= (horizontal_gap / 480.0)
            candidate_score -= (bottom_left_distance / 720.0)

            candidate_specs.append(
                {
                    "step_number": int(region.get("step_number", 0) or 0),
                    "label": "callout_box_candidate",
                    "coords": abs_box,
                    "candidate_origin": "callout_box_candidate",
                    "qty_text": candidate_qty_texts,
                    "search_region": str(region.get("search_region", "")),
                    "candidate_score": candidate_score,
                    "inner_detected_region": list(local_best.get("inner_abs_box", []) or []),
                    "step_anchor": anchor,
                    "search_zone": region_coords,
                    "above_step_preferred": above_step,
                    "fallback_seed_used": bool(used_fallback_seed),
                    "bottom_left_distance": float(bottom_left_distance),
                    "distance_score": vertical_gap + (horizontal_gap * 0.35),
                }
            )

    best_candidates: List[Dict[str, Any]] = []
    fallback_candidates: List[Dict[str, Any]] = []
    used_fallback_steps: set = set()
    for region in search_regions:
        region_step = int(region.get("step_number", 0) or 0)
        region_anchor = dict(region.get("anchor", {}) or {})
        matching = [
            item
            for item in candidate_specs
            if int(item.get("step_number", 0) or 0) == region_step
            and int((item.get("step_anchor", {}) or {}).get("x", -1) or -1)
            == int(region_anchor.get("x", -2) or -2)
            and int((item.get("step_anchor", {}) or {}).get("y", -1) or -1)
            == int(region_anchor.get("y", -2) or -2)
        ]
        if matching:
            matching = sorted(
                matching,
                key=lambda item: (
                    0 if bool(item.get("above_step_preferred")) else 1,
                    float(item.get("bottom_left_distance", 99999.0) or 99999.0),
                    float(item.get("distance_score", 99999.0) or 99999.0),
                    -float(item.get("candidate_score", 0.0) or 0.0),
                ),
            )
            best_candidates.append(matching[0])
            for rejected in matching[1:]:
                _append_rejection(region, list(rejected.get("coords", []) or []), "not_nearest_to_step")
        else:
            # --- Fallback: scan a generous region (or full page) for blue callout box near this step anchor ---
            px, py, pw, ph = [int(value or 0) for value in list(region.get("coords", []) or [0,0,0,0])[:4]]
            anchor = dict(region.get("anchor", {}) or {})
            step_x = int(anchor.get("x", 0) or 0)
            step_y = int(anchor.get("y", 0) or 0)
            step_w = int(anchor.get("w", 0) or 0)
            step_h = int(anchor.get("h", 0) or 0)
            # Use a generous region around the step anchor, or fallback to the whole page if needed
            fallback_box = _clamp_contact_sheet_box(
                [
                    max(0, step_x - 260),
                    max(0, step_y - 320),
                    max(340, step_w + 520),
                    max(220, step_h + 340),
                ],
                page_width=page_width,
                page_height=page_height,
            )
            if fallback_box is None:
                fallback_box = [0, 0, page_width, page_height]
            region_img = page_analyzer.crop(img, fallback_box)
            if region_img is not None and region_img.size > 0:
                hsv = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
                # Find blue-ish regions with pale fill and dark border
                mask_blue = cv2.inRange(hsv, (82, 20, 120), (132, 140, 255))
                mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
                edges = cv2.Canny(gray, 40, 120)
                contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                fallback_found = False
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w < 40 or h < 24:
                        continue
                    # Check border darkness
                    border = gray[y:y+h, x:x+w]
                    if border.size == 0:
                        continue
                    mean_val = float(border.mean())
                    if mean_val > 210.0:
                        continue
                    # Check proximity to step anchor
                    abs_x = fallback_box[0] + x
                    abs_y = fallback_box[1] + y
                    center_dist = abs((abs_x + w//2) - (step_x + step_w//2)) + abs((abs_y + h//2) - (step_y + step_h//2))
                    # Prefer above/beside
                    above = (abs_y + h) <= (step_y + 26)
                    if not above and abs_y > (step_y + 90):
                        continue
                    # Extract qty tokens in this region
                    crop_img = page_analyzer.crop(img, [abs_x, abs_y, w, h])
                    qty_payload = _extract_crop_qty_status(crop_img)
                    qty_tokens = _extract_qty_tokens_from_image(crop_img)
                    qty_texts = [re.sub(r"\\s+", "", str(token.get("text", "")).lower()) for token in qty_tokens if str(token.get("text", "")).strip()]
                    if qty_payload.get("qty") is not None:
                        qty_texts.append(f"x{int(qty_payload['qty'])}")
                    qty_texts = sorted(set([t for t in qty_texts if t]))
                    if not qty_texts:
                        continue
                    candidate_score = 1.0
                    candidate_score += 1.2 if above else 0.0
                    candidate_score -= (center_dist / 480.0)
                    fallback_candidates.append({
                        "step_number": region_step,
                        "label": "callout_box_fallback",
                        "coords": [abs_x, abs_y, w, h],
                        "candidate_origin": "callout_box_fallback",
                        "qty_text": qty_texts,
                        "search_region": "fallback_step_anchor_zone",
                        "candidate_score": candidate_score,
                        "step_anchor": anchor,
                        "search_zone": fallback_box,
                        "above_step_preferred": above,
                        "fallback_seed_used": True,
                        "bottom_left_distance": center_dist,
                        "distance_score": center_dist,
                    })
                    fallback_found = True
                if fallback_found:
                    used_fallback_steps.add(region_step)
    # For each step with no candidate, pick best fallback if available
    for region in search_regions:
        region_step = int(region.get("step_number", 0) or 0)
        if region_step in used_fallback_steps:
            matching = [c for c in fallback_candidates if int(c.get("step_number", 0) or 0) == region_step]
            if matching:
                matching = sorted(
                    matching,
                    key=lambda item: (
                        0 if bool(item.get("above_step_preferred")) else 1,
                        float(item.get("bottom_left_distance", 99999.0) or 99999.0),
                        float(item.get("distance_score", 99999.0) or 99999.0),
                        -float(item.get("candidate_score", 0.0) or 0.0),
                    ),
                )
                best_candidates.append(matching[0])

    accepted_candidates = sorted(
        _dedupe_callout_candidates(best_candidates),
        key=lambda item: (
            int((item.get("coords") or [0, 0])[1] or 0),
            int((item.get("coords") or [0, 0])[0] or 0),
        ),
    )
    deduped_rejected: List[Dict[str, Any]] = []
    seen_rejected: set[Tuple[int, int, int, int, str]] = set()
    for item in rejected_specs:
        coords = list(item.get("coords", []) or [])
        if len(coords) < 4:
            continue
        key = (
            int(coords[0]),
            int(coords[1]),
            int(coords[2]),
            int(coords[3]),
            str(item.get("reason", "")),
        )
        if key in seen_rejected:
            continue
        seen_rejected.add(key)
        deduped_rejected.append(item)

    return {
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": deduped_rejected,
        "search_regions": search_regions,
        "step_anchors": step_anchors,
    }


def _dedupe_callout_candidates(candidate_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for candidate in candidate_specs:
        coords = list(candidate.get("coords", []) or [])
        if len(coords) < 4:
            continue
        step_number = int(candidate.get("step_number", 0) or 0)
        x, y, w, h = [int(value or 0) for value in coords[:4]]
        cx = x + (w // 2)
        cy = y + (h // 2)
        score = float(candidate.get("candidate_score", 0.0) or 0.0)
        replaced = False
        for idx, existing in enumerate(deduped):
            existing_coords = list(existing.get("coords", []) or [])
            if len(existing_coords) < 4:
                continue
            ex, ey, ew, eh = [int(value or 0) for value in existing_coords[:4]]
            ecx = ex + (ew // 2)
            ecy = ey + (eh // 2)
            same_step_band = (
                step_number > 0
                and step_number == int(existing.get("step_number", 0) or 0)
                and abs(cy - ecy) <= 120
                and abs(cx - ecx) <= 160
            )
            overlapping = _box_iou_debug(coords, existing_coords) >= 0.25
            if not same_step_band and not overlapping:
                continue
            existing_score = float(existing.get("candidate_score", 0.0) or 0.0)
            existing_qty_count = len(list(existing.get("qty_text", []) or []))
            new_qty_count = len(list(candidate.get("qty_text", []) or []))
            if score > existing_score or (
                abs(score - existing_score) < 1e-6 and new_qty_count >= existing_qty_count
            ):
                deduped[idx] = candidate
            replaced = True
            break
        if not replaced:
            deduped.append(candidate)
    return deduped


def _build_material_crop_candidates(
    img,
    page_width: int,
    page_height: int,
    step_boxes: List[Dict[str, Any]],
    include_minifig: bool = False,
    detection_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    detection_payload = detection_payload or _detect_callout_box_candidate_specs(
        img,
        page_width=page_width,
        page_height=page_height,
        step_boxes=step_boxes,
    )
    callout_candidate_specs = list(
        detection_payload.get("accepted_candidates", []) or []
    )
    candidate_specs: List[Dict[str, Any]] = list(callout_candidate_specs)

    min_known_visual_step = min(
        [
            int(item.get("step_number", 0) or 0)
            for item in step_boxes
            if int(item.get("step_number", 0) or 0) > 0
            and str(item.get("source", "")) == "visual"
        ]
        or [0]
    )

    def _classify_candidate(candidate: Dict[str, Any], coords: List[int]) -> Tuple[str, Optional[str]]:
        x, y, w, h = [int(value or 0) for value in coords[:4]]
        candidate_step_number = int(candidate.get("step_number", 0) or 0)
        lower_half = y >= int(page_height * 0.5)
        mid_lower_band = y >= int(page_height * 0.38)
        top_anchor_near_header = bool(
            step_boxes
            and min(int(item.get("y", 0) or 0) for item in step_boxes) <= int(page_height * 0.18)
        )
        has_clear_top_left_region = x <= int(page_width * 0.08) and w <= int(page_width * 0.42)

        if min_known_visual_step > 0 and 0 < candidate_step_number < min_known_visual_step:
            return "excluded_minifig_step", "pre_main_step_special_callout"
        if lower_half and x <= int(page_width * 0.22):
            return "excluded_minifig_step", "lower_half_step_panel"
        if top_anchor_near_header and mid_lower_band and not has_clear_top_left_region:
            if x <= int(page_width * 0.24):
                return "excluded_minifig_step", "no_clear_top_left_callout_region"
        if top_anchor_near_header and y >= int(page_height * 0.42) and x <= int(page_width * 0.24):
            return "excluded_minifig_step", "lower_character_assembly_band"
        return "normal_candidate", None

    tiles: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int, int]] = set()
    for candidate in candidate_specs:
        coords = list(candidate.get("coords", []) or [])
        if len(coords) < 4:
            continue
        key = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
        if key in seen:
            continue
        seen.add(key)

        data_uri = _encode_contact_sheet_crop(img, coords, max_edge=240)
        if data_uri is None:
            continue

        candidate_type, exclusion_reason = _classify_candidate(candidate, coords)
        crop_img = page_analyzer.crop(img, coords)
        match_enabled = bool(
            str(candidate.get("candidate_origin", "callout_box_candidate"))
            == "callout_box_candidate"
            and (candidate_type == "normal_candidate" or include_minifig)
        )
        detected_qty_payload = (
            _extract_detected_qty_details_from_crop(crop_img)
            if match_enabled
            else {
                "detected_qty_text": [],
                "detected_qty_numbers": [],
                "detected_qty_raw_text": "",
            }
        )
        x = int(coords[0] or 0)
        y = int(coords[1] or 0)
        w = int(coords[2] or 0)
        h = int(coords[3] or 0)
        x2 = x + max(0, w)
        y2 = y + max(0, h)
        tiles.append(
            {
                "step_number": int(candidate.get("step_number", 0) or 0),
                "label": str(candidate.get("label", "Crop candidate") or "Crop candidate"),
                "coords": coords,
                "coords_xywh": [x, y, w, h],
                "coords_xyxy": [x, y, x2, y2],
                "coords_label": (
                    f"x={x}, y={y}, w={w}, h={h} | "
                    f"x1={x}, y1={y}, x2={x2}, y2={y2}"
                ),
                "data_uri": data_uri,
                "candidate_origin": str(
                    candidate.get("candidate_origin", "callout_box_candidate")
                    or "callout_box_candidate"
                ),
                "candidate_type": candidate_type,
                "exclusion_reason": exclusion_reason,
                "qty_text": list(candidate.get("qty_text", []) or []),
                "detected_qty_text": list(
                    detected_qty_payload.get("detected_qty_text", []) or []
                ),
                "detected_qty_numbers": list(
                    detected_qty_payload.get("detected_qty_numbers", []) or []
                ),
                "detected_qty_raw_text": str(
                    detected_qty_payload.get("detected_qty_raw_text", "") or ""
                ),
                "inner_detected_region": list(
                    candidate.get("inner_detected_region", []) or []
                ),
                "match_enabled": match_enabled,
            }
        )
    return tiles


def _format_debug_coords(coords: List[int]) -> str:
    if len(coords) < 4:
        return "coords unavailable"
    x = int(coords[0] or 0)
    y = int(coords[1] or 0)
    w = int(coords[2] or 0)
    h = int(coords[3] or 0)
    x2 = x + max(0, w)
    y2 = y + max(0, h)
    return f"x={x}, y={y}, w={w}, h={h} | x1={x}, y1={y}, x2={x2}, y2={y2}"


def _extract_detected_qty_details_from_crop(crop_img) -> Dict[str, Any]:
    if crop_img is None or crop_img.size == 0:
        return {
            "detected_qty_text": [],
            "detected_qty_numbers": [],
            "detected_qty_raw_text": "",
        }

    height, width = crop_img.shape[:2]
    region_specs = [
        ("full", crop_img),
        ("right_half", crop_img[:, max(0, width // 2) : width]),
        ("lower_half", crop_img[max(0, height // 2) : height, :]),
        (
            "lower_right",
            crop_img[max(0, height // 3) : height, max(0, width // 2) : width],
        ),
        ("right_band", crop_img[:, max(0, int(width * 0.65)) : width]),
    ]

    normalized_texts: List[str] = []
    raw_texts: List[str] = []
    for _, region_img in region_specs:
        if region_img is None or region_img.size == 0:
            continue
        token_matches = _extract_qty_tokens_from_image(region_img)
        for token in token_matches:
            text = re.sub(r"\s+", "", str(token.get("text", "") or "").lower())
            if re.match(r"^\d+x$", text) or re.match(r"^x\d+$", text):
                normalized_texts.append(text)

        qty_status = _extract_crop_qty_status(region_img)
        raw_text = str(qty_status.get("raw_text", "") or "")
        if raw_text:
            raw_texts.append(raw_text)
        raw_matches = re.findall(r"(?i)\b(?:x\s*\d{1,2}|\d{1,2}\s*x)\b", raw_text)
        for match in raw_matches:
            normalized_texts.append(re.sub(r"\s+", "", str(match).lower()))

    # Count occurrences per unique text using bounding-box tokens only.
    # The raw-text regex can match the same label multiple times from a single
    # OCR string (e.g. "8x 8x" → 2 regex hits from one label), so using it for
    # counting would inflate duplicates. Positional tokens are the ground truth.
    from collections import Counter as _Counter
    full_texts: List[str] = []
    for token in (_extract_qty_tokens_from_image(crop_img) or []):
        t = re.sub(r"\s+", "", str(token.get("text", "") or "").lower())
        if re.match(r"^\d+x$", t) or re.match(r"^x\d+$", t):
            full_texts.append(t)
    full_counts = _Counter(full_texts)

    detected_qty_text: List[str] = []
    detected_qty_numbers: List[int] = []
    # Use only positional tokens detected by OCR on the full crop image.
    # Each entry in full_texts corresponds to one visually detected label.
    for text in full_texts:
        number_match = re.search(r"(\d{1,2})", text)
        qty_val = int(number_match.group(1)) if number_match else None
        detected_qty_text.append(text)
        if qty_val is not None:
            detected_qty_numbers.append(qty_val)

    return {
        "detected_qty_text": detected_qty_text,
        "detected_qty_numbers": detected_qty_numbers,
        "detected_qty_raw_text": " | ".join(
            text for text in raw_texts if str(text or "").strip()
        ),
    }


def _encode_debug_image_data_uri(
    img,
    max_width: int = 1400,
) -> str:
    render_img = img.copy()
    height, width = render_img.shape[:2]
    if width > int(max_width):
        scale = float(max_width) / float(max(width, 1))
        render_img = cv2.resize(
            render_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        render_img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 88],
    )
    if not ok:
        raise RuntimeError("Could not encode debug image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _draw_step_boxes_overlay(
    img,
    step_boxes: List[Dict[str, Any]],
) -> None:
    for item in step_boxes:
        x = int(item.get("x", 0) or 0)
        y = int(item.get("y", 0) or 0)
        w = int(item.get("w", 0) or 0)
        h = int(item.get("h", 0) or 0)
        step_number = int(item.get("step_number", 0) or 0)
        if w <= 0 or h <= 0:
            continue

        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 190, 70), 2)
        label = f"step {step_number}" if step_number > 0 else "step ?"
        label_y = max(18, y - 8)
        cv2.putText(
            img,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 70, 20),
            2,
            cv2.LINE_AA,
        )


def _draw_grid_overlay(
    img,
    spacing: int = 100,
) -> None:
    page_height, page_width = img.shape[:2]
    spacing = max(40, int(spacing))
    grid_color = (185, 120, 35)
    text_color = (120, 70, 20)

    for x in range(0, page_width, spacing):
        cv2.line(img, (x, 0), (x, page_height), grid_color, 1)
        cv2.putText(
            img,
            f"x{x}",
            (x + 4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

    for y in range(0, page_height, spacing):
        cv2.line(img, (0, y), (page_width, y), grid_color, 1)
        cv2.putText(
            img,
            f"y{y}",
            (6, max(18, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )


def _draw_candidate_boxes_overlay(
    img,
    candidates: List[Dict[str, Any]],
    include_minifig: bool = False,
) -> None:
    def _draw_dashed_rect(
        target_img,
        box: List[int],
        color: Tuple[int, int, int],
        thickness: int = 1,
        dash: int = 10,
        gap: int = 6,
    ) -> None:
        if len(box) < 4:
            return
        x, y, w, h = [int(value or 0) for value in box[:4]]
        if w <= 0 or h <= 0:
            return
        x2 = x + w
        y2 = y + h
        for start in range(x, x2, dash + gap):
            end = min(start + dash, x2)
            cv2.line(target_img, (start, y), (end, y), color, thickness)
            cv2.line(target_img, (start, y2), (end, y2), color, thickness)
        for start in range(y, y2, dash + gap):
            end = min(start + dash, y2)
            cv2.line(target_img, (x, start), (x, end), color, thickness)
            cv2.line(target_img, (x2, start), (x2, end), color, thickness)

    for idx, item in enumerate(candidates, start=1):
        coords = list(item.get("coords", []) or [])
        if len(coords) < 4:
            continue
        x, y, w, h = [int(value or 0) for value in coords[:4]]
        if w <= 0 or h <= 0:
            continue
        candidate_origin = str(
            item.get("candidate_origin", "callout_box_candidate")
            or "callout_box_candidate"
        )
        candidate_type = str(item.get("candidate_type", "normal_candidate") or "normal_candidate")
        is_excluded = candidate_type == "excluded_minifig_step" and not include_minifig
        if is_excluded:
            color = (150, 150, 150)
            text_color = (95, 95, 95)
        elif candidate_origin == "callout_box_candidate":
            color = (40, 185, 60)
            text_color = (30, 110, 40)
        else:
            color = (215, 140, 40)
            text_color = (140, 80, 15)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        inner_region = list(item.get("inner_detected_region", []) or [])
        if len(inner_region) >= 4:
            _draw_dashed_rect(img, inner_region, (40, 150, 220), thickness=1)
        cv2.putText(
            img,
            f"c{idx}",
            (x + 4, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            2,
            cv2.LINE_AA,
        )


def _draw_step_anchor_debug_overlay(
    img,
    step_anchors: List[Dict[str, Any]],
) -> None:
    for anchor in step_anchors:
        x = int(anchor.get("x", 0) or 0)
        y = int(anchor.get("y", 0) or 0)
        w = int(anchor.get("w", 0) or 0)
        h = int(anchor.get("h", 0) or 0)
        step_number = int(anchor.get("step_number", 0) or 0)
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (180, 60, 190), 2)
        cv2.putText(
            img,
            f"s{step_number or '?'}",
            (x + 3, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (110, 35, 120),
            2,
            cv2.LINE_AA,
        )


def _draw_search_zone_overlay(
    img,
    search_regions: List[Dict[str, Any]],
) -> None:
    for idx, region in enumerate(search_regions, start=1):
        coords = list(region.get("coords", []) or [])
        if len(coords) < 4:
            continue
        x, y, w, h = [int(value or 0) for value in coords[:4]]
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 210, 40), 1)
        cv2.putText(
            img,
            f"z{idx}",
            (x + 3, min(y + 16, y + h - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (145, 150, 25),
            1,
            cv2.LINE_AA,
        )


def _draw_rejected_callout_overlay(
    img,
    rejected_candidates: List[Dict[str, Any]],
) -> None:
    for idx, item in enumerate(rejected_candidates, start=1):
        coords = list(item.get("coords", []) or [])
        if len(coords) < 4:
            continue
        x, y, w, h = [int(value or 0) for value in coords[:4]]
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (135, 135, 135), 1)
        cv2.putText(
            img,
            f"r{idx}",
            (x + 3, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )


def _coerce_optional_debug_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(value)
    except Exception:
        return None


def _load_set_part_image_library(set_num: str) -> Dict[str, Any]:
    catalog_db_path = (
        debug_service.DEBUG_ROOT / "server_catalog" / "lego_catalog.db"
    )
    if not catalog_db_path.exists():
        raise HTTPException(status_code=404, detail="Catalog DB not found")

    db_uri = f"file:{catalog_db_path}?mode=ro"
    try:
        conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not open catalog DB") from exc

    try:
        inventory_set_num = str(set_num).strip()
        if "-" not in inventory_set_num:
            inventory_set_num = f"{inventory_set_num}-1"

        inventory_row = conn.execute(
            """
            SELECT inventory_id, set_num, version
            FROM inventories
            WHERE set_num = ?
            ORDER BY version DESC, inventory_id DESC
            LIMIT 1
            """,
            (inventory_set_num,),
        ).fetchone()
        if inventory_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Inventory not found for set {inventory_set_num}",
            )

        requirement_rows = conn.execute(
            """
            SELECT
              part_num,
              color_id,
              SUM(quantity) AS quantity
            FROM inventory_parts
            WHERE inventory_id = ?
              AND COALESCE(is_spare, 0) = 0
            GROUP BY part_num, color_id
            HAVING COALESCE(SUM(quantity), 0) > 0
            ORDER BY quantity DESC, part_num ASC, color_id ASC
            """,
            (int(inventory_row["inventory_id"]),),
        ).fetchall()

        unique_pairs = [
            (str(row["part_num"] or ""), int(row["color_id"] or 0))
            for row in requirement_rows
            if str(row["part_num"] or "")
        ]
        element_map: Dict[Tuple[str, int], Any] = {}
        img_map: Dict[Tuple[str, int], str] = {}
        color_map: Dict[int, Dict[str, Any]] = {}

        for row in conn.execute(
            "SELECT color_id, name, rgb, is_trans FROM colors"
        ).fetchall():
            color_id = int(row["color_id"] or 0)
            color_map[color_id] = {
                "color_name": str(row["name"] or ""),
                "color_rgb": str(row["rgb"] or ""),
                "color_is_trans": bool(int(row["is_trans"] or 0)),
            }

        if unique_pairs:
            chunk_size = 200
            for start in range(0, len(unique_pairs), chunk_size):
                chunk = unique_pairs[start : start + chunk_size]
                where = " OR ".join(["(part_num = ? AND color_id = ?)"] * len(chunk))
                params: List[Any] = []
                for part_num, color_id in chunk:
                    params.extend([part_num, color_id])

                img_rows = conn.execute(
                    (
                        "SELECT part_num, color_id, img_url "
                        "FROM element_images "
                        f"WHERE {where}"
                    ),
                    params,
                ).fetchall()
                for row in img_rows:
                    if row["img_url"]:
                        img_map[(str(row["part_num"]), int(row["color_id"]))] = str(row["img_url"])

                element_rows = conn.execute(
                    (
                        "SELECT part_num, color_id, MIN(element_id) AS element_id "
                        "FROM elements "
                        f"WHERE {where} "
                        "GROUP BY part_num, color_id"
                    ),
                    params,
                ).fetchall()
                for row in element_rows:
                    element_map[(str(row["part_num"]), int(row["color_id"]))] = row["element_id"]
    finally:
        conn.close()

    items: List[Dict[str, Any]] = []
    with_images = 0
    for row in requirement_rows:
        pair = (str(row["part_num"] or ""), int(row["color_id"] or 0))
        color_payload = color_map.get(pair[1], {})
        item = {
            "part_num": pair[0],
            "color_id": pair[1],
            "element_id": (
                str(element_map.get(pair))
                if element_map.get(pair) not in (None, "")
                else None
            ),
            "qty": int(row["quantity"] or 0),
            "img_url": (
                str(img_map.get(pair))
                if img_map.get(pair) not in (None, "")
                else None
            ),
            "color_name": str(color_payload.get("color_name", "") or ""),
            "color_rgb": str(color_payload.get("color_rgb", "") or ""),
            "color_is_trans": bool(color_payload.get("color_is_trans", False)),
        }
        if item["img_url"]:
            with_images += 1
        items.append(item)

    return {
        "set_num": str(set_num),
        "inventory_set_num": str(inventory_row["set_num"]),
        "inventory_id": int(inventory_row["inventory_id"]),
        "version": int(inventory_row["version"] or 0),
        "items": items,
        "tile_count": len(items),
        "with_images": with_images,
    }


def _require_openai_vision_client_debug():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI Python SDK is not installed. Install `openai` in this environment first."
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required in the environment.")

    return OpenAI(api_key=api_key)


def _response_text_to_json_debug(response: Any) -> Any:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return json.loads(output_text)

    try:
        response_dict = response.model_dump()
    except Exception:
        response_dict = None

    if isinstance(response_dict, dict):
        for item in response_dict.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return json.loads(content["text"])

    raise RuntimeError("Could not extract JSON text from OpenAI response.")


def _openai_callout_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "boxes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "step": {"type": ["integer", "null"]},
                        "box": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "qty_text": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["step", "box", "qty_text", "confidence"],
                },
            }
        },
        "required": ["boxes"],
    }


def _openai_callout_prompt(set_num: str, page: int) -> str:
    return (
        "Find every LEGO parts-needed callout box on this instruction page. "
        "The target boxes are pale or light-blue rounded rectangles with a grey outline, "
        "containing loose part images and quantity labels such as 1x, 2x, or 3x. "
        "They are normally above or next to a large step number. "
        "Do not return main build areas, arrows, minifig assembly panels, or page decorations. "
        "Return JSON only as an object with a top-level `boxes` array. "
        "Coordinates must be pixel coordinates in the original page image as [x1, y1, x2, y2]. "
        "Set number: %s. Page: %d."
    ) % (set_num, int(page))


def _normalize_openai_callout_boxes(
    payload: Any,
    page_width: int,
    page_height: int,
) -> List[Dict[str, Any]]:
    rows = payload.get("boxes", []) if isinstance(payload, dict) else payload
    normalized: List[Dict[str, Any]] = []
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        raw_box = list(item.get("box", []) or [])
        if len(raw_box) < 4:
            continue
        x1 = int(raw_box[0] or 0)
        y1 = int(raw_box[1] or 0)
        x2 = int(raw_box[2] or 0)
        y2 = int(raw_box[3] or 0)
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        box = _clamp_contact_sheet_box(
            [left, top, right - left, bottom - top],
            page_width=page_width,
            page_height=page_height,
        )
        if box is None:
            continue
        normalized.append(
            {
                "step": _coerce_optional_debug_int(item.get("step")),
                "coords": box,
                "qty_text": [str(value) for value in (item.get("qty_text", []) or []) if str(value)],
                "confidence": float(item.get("confidence", 0.0) or 0.0),
            }
        )
    return normalized


def _detect_openai_callout_boxes(
    set_num: str,
    page: int,
    image_path: Path,
    page_width: int,
    page_height: int,
    model: str,
) -> List[Dict[str, Any]]:
    client = _require_openai_vision_client_debug()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _openai_callout_prompt(set_num, int(page)),
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii"),
                        "detail": "high",
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "lego_callout_boxes",
                "strict": True,
                "schema": _openai_callout_schema(),
            }
        },
    )
    payload = _response_text_to_json_debug(response)
    return _normalize_openai_callout_boxes(payload, page_width=page_width, page_height=page_height)


def _resolve_bag_page_range(set_num: str, bag: int) -> Tuple[List[int], int, int]:
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        raise HTTPException(status_code=404, detail="No rendered pages found for set")

    rendered_pages = _list_rendered_pages(pages_dir)
    if not rendered_pages:
        raise HTTPException(status_code=404, detail="No rendered pages found in pages dir")

    bag_truth = sorted(
        list(bag_truth_store.get_bag_truth(set_num) or []),
        key=lambda item: (
            int(item.get("bag_number", 0) or 0),
            int(item.get("start_page", 0) or 0),
        ),
    )
    bag_row = next(
        (
            item
            for item in bag_truth
            if int(item.get("bag_number", 0) or 0) == int(bag)
        ),
        None,
    )
    if bag_row is None:
        raise HTTPException(status_code=404, detail=f"Bag {bag} not found in saved bag truth")

    start_page = int(bag_row.get("start_page", 0) or 0)
    next_bag_row = next(
        (
            item
            for item in bag_truth
            if int(item.get("bag_number", 0) or 0) > int(bag)
        ),
        None,
    )
    end_page = (
        int(next_bag_row.get("start_page", 0) or 0) - 1
        if next_bag_row is not None
        else int(rendered_pages[-1])
    )
    if end_page < start_page:
        end_page = start_page
    return rendered_pages, start_page, end_page


def _extract_crop_qty_status(crop_img) -> Dict[str, Any]:
    if crop_img is None or crop_img.size == 0:
        return {
            "qty": None,
            "status": "empty_crop",
            "raw_text": "",
        }

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    thresholded = cv2.adaptiveThreshold(
        enlarged,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        6,
    )

    raw_texts: List[str] = []
    ocr_images = [
        (crop_img, "--psm 6 -c tessedit_char_whitelist=0123456789xX"),
        (enlarged, "--psm 11 -c tessedit_char_whitelist=0123456789xX"),
        (thresholded, "--psm 11 -c tessedit_char_whitelist=0123456789xX"),
    ]
    for source_img, config in ocr_images:
        try:
            text = pytesseract.image_to_string(source_img, config=config) or ""
        except Exception:
            text = ""
        normalized_text = " ".join((text or "").replace("\n", " ").split())
        if normalized_text:
            raw_texts.append(normalized_text)

    normalized = " | ".join(raw_texts)
    match = re.search(r"(?i)\bx\s*(\d{1,2})\b", normalized)
    if match is None:
        match = re.search(r"(?i)\b(\d{1,2})\s*x\b", normalized)

    qty = int(match.group(1)) if match else None
    status = "qty_found" if qty is not None else "no_qty_marker"
    return {
        "qty": qty,
        "status": status,
        "raw_text": normalized,
    }


def _extract_qty_tokens_from_image(img) -> List[Dict[str, Any]]:
    if img is None or img.size == 0:
        return []

    tokens: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int, int, int, int]] = set()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    thresholded = cv2.adaptiveThreshold(
        enlarged,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        6,
    )
    image_variants = [
        (img, 1.0, "--psm 6 -c tessedit_char_whitelist=0123456789xX"),
        (enlarged, 0.5, "--psm 11 -c tessedit_char_whitelist=0123456789xX"),
        (thresholded, 0.5, "--psm 11 -c tessedit_char_whitelist=0123456789xX"),
    ]
    for variant_img, scale_back, config in image_variants:
        data = pytesseract.image_to_data(
            variant_img,
            config=config,
            output_type=Output.DICT,
        )
        count = len(data.get("text", []))
        for idx in range(count):
            text = (data.get("text", [""])[idx] or "").strip()
            if not text:
                continue
            normalized = re.sub(r"\s+", "", text.lower())
            if not (
                re.match(r"^\d+x$", normalized)
                or re.match(r"^x\d+$", normalized)
            ):
                continue
            x = int((int(data.get("left", [0])[idx] or 0)) * scale_back)
            y = int((int(data.get("top", [0])[idx] or 0)) * scale_back)
            w = int((int(data.get("width", [0])[idx] or 0)) * scale_back)
            h = int((int(data.get("height", [0])[idx] or 0)) * scale_back)
            if w <= 0 or h <= 0:
                continue
            key = (
                normalized,
                int(x // 4),
                int(y // 4),
                int(w // 4),
                int(h // 4),
            )
            if key in seen:
                continue
            seen.add(key)
            tokens.append(
                {
                    "text": normalized,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "cx": x + (w // 2),
                    "cy": y + (h // 2),
                }
            )
    return tokens


def _download_part_image(url: str):
    cached = _PART_IMAGE_CACHE.get(url)
    if url in _PART_IMAGE_CACHE:
        return cached

    try:
        with urllib.request.urlopen(url, timeout=6) as response:
            data = response.read()
    except Exception:
        _PART_IMAGE_CACHE[url] = None
        return None

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    _PART_IMAGE_CACHE[url] = img
    return img


def _average_hash_bits(gray_img, hash_size: int = 8) -> int:
    resized = cv2.resize(gray_img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = float(resized.mean())
    bits = 0
    for value in resized.flatten():
        bits = (bits << 1) | (1 if float(value) >= avg else 0)
    return bits


def _hex_to_rgb_tuple(rgb_hex: str) -> Optional[Tuple[int, int, int]]:
    value = str(rgb_hex or "").strip().lstrip("#")
    if len(value) != 6:
        return None
    try:
        return (
            int(value[0:2], 16),
            int(value[2:4], 16),
            int(value[4:6], 16),
        )
    except Exception:
        return None


def _estimate_crop_part_color_payload(crop_img) -> Dict[str, Any]:
    if crop_img is None or crop_img.size == 0:
        return {
            "crop_rgb": None,
            "tight_box": None,
            "selected_pixel_count": 0,
            "selected_fraction": 0.0,
            "debug_data_uri": None,
            "tight_crop_data_uri": None,
        }

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    page_h, page_w = crop_img.shape[:2]

    # Focus on likely part pixels while dropping pale callout background, white paper, black text, and dark shadow.
    foreground_mask = (
        ((sat >= 42) & (val >= 52))
        | ((val >= 68) & (val <= 188))
    )
    pale_blue_background_mask = (
        (hue >= 82)
        & (hue <= 132)
        & (sat <= 140)
        & (val >= 150)
    )
    near_white_mask = (sat <= 36) & (val >= 205)
    black_text_mask = (val <= 58) & (sat <= 95)
    dark_shadow_mask = (val <= 82) & (sat <= 120)
    foreground_mask = (
        foreground_mask
        & ~pale_blue_background_mask
        & ~near_white_mask
        & ~black_text_mask
        & ~dark_shadow_mask
    )

    foreground_u8 = (foreground_mask.astype(np.uint8) * 255)
    foreground_u8 = cv2.morphologyEx(
        foreground_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    foreground_u8 = cv2.morphologyEx(
        foreground_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    tight_box: Optional[List[int]] = None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_u8, 8)
    best_label = 0
    best_score = -1.0
    crop_cx = page_w / 2.0
    crop_cy = page_h / 2.0
    for label_idx in range(1, int(num_labels)):
        x, y, w, h, area = [int(value) for value in stats[label_idx]]
        if area < 20 or w <= 0 or h <= 0:
            continue
        region_mask = labels == label_idx
        region_sat = sat[region_mask]
        region_val = val[region_mask]
        if region_sat.size == 0 or region_val.size == 0:
            continue
        mean_sat = float(region_sat.mean())
        mean_val = float(region_val.mean())
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        center_penalty = ((abs(cx - crop_cx) / max(1.0, page_w)) * 0.45) + ((abs(cy - crop_cy) / max(1.0, page_h)) * 0.25)
        score = float(area) * (0.65 + (mean_sat / 255.0) * 0.35 + (mean_val / 255.0) * 0.15) - (center_penalty * 120.0)
        if score > best_score:
            best_score = score
            best_label = label_idx
            tight_box = [x, y, w, h]

    if tight_box is not None:
        x, y, w, h = [int(value or 0) for value in tight_box[:4]]
        pad_x = max(3, int(w * 0.06))
        pad_y = max(3, int(h * 0.08))
        tight_box = _clamp_contact_sheet_box(
            [x - pad_x, y - pad_y, w + (pad_x * 2), h + (pad_y * 2)],
            page_width=page_w,
            page_height=page_h,
        )
    if tight_box is None:
        tight_box = [0, 0, page_w, page_h]

    tx, ty, tw, th = [int(value or 0) for value in tight_box[:4]]
    tight_img = crop_img[ty : ty + th, tx : tx + tw]
    tight_hsv = hsv[ty : ty + th, tx : tx + tw]
    tight_mask = foreground_u8[ty : ty + th, tx : tx + tw] > 0
    if tight_img.size == 0:
        tight_img = crop_img
        tight_hsv = hsv
        tight_mask = foreground_u8 > 0
        tight_box = [0, 0, page_w, page_h]

    tight_bgr_pixels = tight_img[tight_mask]
    tight_sat = tight_hsv[:, :, 1][tight_mask]
    tight_val = tight_hsv[:, :, 2][tight_mask]

    if tight_bgr_pixels.size == 0:
        fallback_mask = ~(pale_blue_background_mask | near_white_mask | black_text_mask | dark_shadow_mask)
        tight_bgr_pixels = crop_img[fallback_mask]
        if tight_bgr_pixels.size == 0:
            tight_bgr_pixels = crop_img.reshape(-1, 3)

    sampled = tight_bgr_pixels.reshape(-1, 3).astype(np.float32)
    if sampled.shape[0] > 1200:
        step = max(1, sampled.shape[0] // 1200)
        sampled = sampled[::step]

    representative_rgb: Optional[Tuple[int, int, int]] = None
    if sampled.shape[0] >= 6:
        k = int(min(3, sampled.shape[0]))
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            18,
            0.8,
        )
        try:
            _compactness, label_arr, centers = cv2.kmeans(
                sampled,
                k,
                None,
                criteria,
                4,
                cv2.KMEANS_PP_CENTERS,
            )
            labels_flat = label_arr.reshape(-1)
            best_cluster_idx = 0
            best_cluster_score = -1.0
            for idx in range(k):
                cluster_pixels = sampled[labels_flat == idx]
                if cluster_pixels.size == 0:
                    continue
                center = centers[idx]
                center_rgb = (float(center[2]), float(center[1]), float(center[0]))
                cluster_size = float(cluster_pixels.shape[0])
                cluster_score = cluster_size
                cluster_score *= 0.75 + (max(center_rgb) / 255.0) * 0.15
                if cluster_score > best_cluster_score:
                    best_cluster_score = cluster_score
                    best_cluster_idx = idx
            cluster_pixels = sampled[labels_flat == best_cluster_idx]
            if cluster_pixels.size > 0:
                median_bgr = np.median(cluster_pixels, axis=0)
                representative_rgb = (
                    int(round(float(median_bgr[2]))),
                    int(round(float(median_bgr[1]))),
                    int(round(float(median_bgr[0]))),
                )
        except Exception:
            representative_rgb = None

    if representative_rgb is None and sampled.size > 0:
        median_bgr = np.median(sampled, axis=0)
        representative_rgb = (
            int(round(float(median_bgr[2]))),
            int(round(float(median_bgr[1]))),
            int(round(float(median_bgr[0]))),
        )

    debug_preview = crop_img.copy()
    preview_mask = np.zeros((page_h, page_w), dtype=np.uint8)
    preview_mask[ty : ty + th, tx : tx + tw] = (
        (tight_mask.astype(np.uint8) * 255)
        if tight_mask.shape[:2] == (th, tw)
        else 0
    )
    debug_preview[preview_mask == 0] = (245, 245, 245)
    if tight_box is not None:
        cv2.rectangle(
            debug_preview,
            (tx, ty),
            (tx + tw, ty + th),
            (40, 180, 60),
            2,
        )
    debug_data_uri = _encode_contact_sheet_crop(
        debug_preview,
        [0, 0, page_w, page_h],
        max_edge=220,
    )
    tight_crop_data_uri = _encode_contact_sheet_crop(
        crop_img,
        list(tight_box or [0, 0, page_w, page_h]),
        max_edge=220,
    )

    selected_pixel_count = int(np.count_nonzero(preview_mask))
    selected_fraction = float(selected_pixel_count) / float(max(1, page_h * page_w))
    return {
        "crop_rgb": representative_rgb,
        "tight_box": list(tight_box or [0, 0, page_w, page_h]),
        "selected_pixel_count": selected_pixel_count,
        "selected_fraction": selected_fraction,
        "debug_data_uri": debug_data_uri,
        "tight_crop_data_uri": tight_crop_data_uri,
    }


def _estimate_crop_part_rgb(crop_img) -> Optional[Tuple[int, int, int]]:
    return _estimate_crop_part_color_payload(crop_img).get("crop_rgb")


def _color_distance_rgb(
    rgb_a: Optional[Tuple[int, int, int]],
    rgb_b: Optional[Tuple[int, int, int]],
) -> Optional[float]:
    if rgb_a is None or rgb_b is None:
        return None
    diff = np.array(rgb_a, dtype=np.float32) - np.array(rgb_b, dtype=np.float32)
    return float(np.linalg.norm(diff))


def _nearest_catalog_color_payload(
    crop_rgb: Optional[Tuple[int, int, int]],
    library_items: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if crop_rgb is None:
        return None

    color_candidates: Dict[int, Dict[str, Any]] = {}
    for item in library_items:
        color_id = int(item.get("color_id", 0) or 0)
        if color_id in color_candidates:
            continue
        rgb_tuple = _hex_to_rgb_tuple(str(item.get("color_rgb") or ""))
        if rgb_tuple is None:
            continue
        distance = _color_distance_rgb(crop_rgb, rgb_tuple)
        if distance is None:
            continue
        color_candidates[color_id] = {
            "color_id": color_id,
            "color_name": str(item.get("color_name") or ""),
            "color_rgb": str(item.get("color_rgb") or ""),
            "distance": distance,
        }
    if not color_candidates:
        return None
    return sorted(
        color_candidates.values(),
        key=lambda item: float(item.get("distance", 99999.0) or 99999.0),
    )[0]


def _compute_match_features(img) -> Optional[Dict[str, Any]]:
    if img is None or img.size == 0:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    standardized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    return {
        "gray64": standardized,
        "hash": _average_hash_bits(standardized),
    }


def _get_library_item_features(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    img_url = str(item.get("img_url") or "")
    if not img_url:
        return None
    if img_url in _PART_IMAGE_FEATURE_CACHE:
        return _PART_IMAGE_FEATURE_CACHE[img_url]

    img = _download_part_image(img_url)
    features = _compute_match_features(img)
    _PART_IMAGE_FEATURE_CACHE[img_url] = features
    return features


def _score_crop_against_library(
    crop_img,
    library_items: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    crop_features = _compute_match_features(crop_img)
    crop_rgb = _estimate_crop_part_rgb(crop_img)
    nearest_color = _nearest_catalog_color_payload(crop_rgb, library_items)
    if crop_features is None and crop_rgb is None:
        return []

    crop_hash = int(crop_features["hash"]) if crop_features is not None else 0
    crop_gray = crop_features["gray64"].astype(np.float32) if crop_features is not None else None
    scored: List[Dict[str, Any]] = []
    for item in library_items:
        candidate_rgb = _hex_to_rgb_tuple(str(item.get("color_rgb") or ""))
        color_distance = _color_distance_rgb(crop_rgb, candidate_rgb)
        if color_distance is None:
            color_score = 0.0
        else:
            color_score = max(0.0, 1.0 - (color_distance / 441.6729559300637))

        image_score: Optional[float] = None
        features = _get_library_item_features(item)
        if features is not None and crop_gray is not None:
            lib_hash = int(features["hash"])
            lib_gray = features["gray64"].astype(np.float32)
            xor_bits = crop_hash ^ lib_hash
            hamming = int(xor_bits.bit_count()) if hasattr(xor_bits, "bit_count") else bin(xor_bits).count("1")
            hash_similarity = 1.0 - (float(hamming) / 64.0)
            diff = cv2.absdiff(crop_gray, lib_gray)
            mse = float(np.mean(np.square(diff))) / float(255.0 * 255.0)
            mse_similarity = max(0.0, 1.0 - mse)
            image_score = (hash_similarity * 0.62) + (mse_similarity * 0.38)

        if image_score is None:
            combined_score = color_score
        else:
            combined_score = (color_score * 0.45) + (image_score * 0.55)

        scored.append(
            {
                "score": combined_score,
                "combined_score": combined_score,
                "color_score": color_score,
                "image_score": image_score,
                "part_num": str(item.get("part_num") or ""),
                "color_id": int(item.get("color_id", 0) or 0),
                "element_id": item.get("element_id"),
                "qty": int(item.get("qty", 0) or 0),
                "img_url": item.get("img_url"),
                "color_name": str(item.get("color_name") or ""),
                "color_rgb": str(item.get("color_rgb") or ""),
                "crop_rgb": crop_rgb,
                "nearest_crop_color_id": (
                    int(nearest_color.get("color_id", 0) or 0)
                    if nearest_color is not None
                    else None
                ),
                "nearest_crop_color_name": (
                    str(nearest_color.get("color_name") or "")
                    if nearest_color is not None
                    else None
                ),
                "nearest_crop_color_rgb": (
                    str(nearest_color.get("color_rgb") or "")
                    if nearest_color is not None
                    else None
                ),
            }
        )

    scored.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return scored[: max(1, int(top_k))]


def _score_crop_against_library_color_only(
    crop_img,
    library_items: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    color_payload = _estimate_crop_part_color_payload(crop_img)
    crop_rgb = color_payload.get("crop_rgb")
    nearest_color = _nearest_catalog_color_payload(crop_rgb, library_items)
    if crop_rgb is None:
        return {
            "crop_rgb": None,
            "tight_box": color_payload.get("tight_box"),
            "selected_pixel_count": int(color_payload.get("selected_pixel_count", 0) or 0),
            "selected_fraction": float(color_payload.get("selected_fraction", 0.0) or 0.0),
            "debug_data_uri": color_payload.get("debug_data_uri"),
            "tight_crop_data_uri": color_payload.get("tight_crop_data_uri"),
            "nearest_color": nearest_color,
            "matches": [],
        }

    nearest_color_id = (
        int(nearest_color.get("color_id", 0) or 0)
        if nearest_color is not None
        else None
    )
    scored: List[Dict[str, Any]] = []
    for item in library_items:
        candidate_rgb = _hex_to_rgb_tuple(str(item.get("color_rgb") or ""))
        color_distance = _color_distance_rgb(crop_rgb, candidate_rgb)
        if color_distance is None:
            color_score = 0.0
            color_distance = 99999.0
        else:
            color_score = max(0.0, 1.0 - (color_distance / 441.6729559300637))

        color_id = int(item.get("color_id", 0) or 0)
        exact_color_match = nearest_color_id is not None and color_id == nearest_color_id
        proximity_bonus = 0.08 if exact_color_match else 0.0
        combined_score = min(1.0, color_score + proximity_bonus)

        scored.append(
            {
                "score": combined_score,
                "combined_score": combined_score,
                "color_score": color_score,
                "image_score": None,
                "part_num": str(item.get("part_num") or ""),
                "color_id": color_id,
                "color_name": str(item.get("color_name") or ""),
                "color_rgb": str(item.get("color_rgb") or ""),
                "element_id": item.get("element_id"),
                "qty": int(item.get("qty", 0) or 0),
                "img_url": item.get("img_url"),
                "crop_rgb": crop_rgb,
                "nearest_crop_color_id": nearest_color_id,
                "nearest_crop_color_name": (
                    str(nearest_color.get("color_name") or "")
                    if nearest_color is not None
                    else None
                ),
                "nearest_crop_color_rgb": (
                    str(nearest_color.get("color_rgb") or "")
                    if nearest_color is not None
                    else None
                ),
                "exact_color_match": bool(exact_color_match),
                "color_distance": float(color_distance),
            }
        )

    scored.sort(
        key=lambda item: (
            0 if bool(item.get("exact_color_match")) else 1,
            -float(item.get("color_score", 0.0) or 0.0),
            float(item.get("color_distance", 99999.0) or 99999.0),
            str(item.get("part_num") or ""),
        )
    )
    return {
        "crop_rgb": crop_rgb,
        "tight_box": color_payload.get("tight_box"),
        "selected_pixel_count": int(color_payload.get("selected_pixel_count", 0) or 0),
        "selected_fraction": float(color_payload.get("selected_fraction", 0.0) or 0.0),
        "debug_data_uri": color_payload.get("debug_data_uri"),
        "tight_crop_data_uri": color_payload.get("tight_crop_data_uri"),
        "nearest_color": nearest_color,
        "matches": scored[: max(1, int(top_k))],
    }


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


@router.get("/debug/set-part-image-library", response_class=HTMLResponse)
def debug_set_part_image_library(
    set_num: str = Query(...),
):
    payload = _load_set_part_image_library(str(set_num))
    items = list(payload.get("items", []) or [])

    tiles_html = (
        "\n".join(
            f"""
            <article class="tile">
              <div class="thumb-wrap">
                {f'<img src="{escape(str(item.get("img_url") or ""))}" alt="{escape(str(item.get("part_num") or ""))}" loading="lazy" />' if item.get("img_url") else '<div class="thumb-missing">No image URL</div>'}
              </div>
              <div class="meta">
                <p><strong>part_num:</strong> {escape(str(item.get("part_num") or ""))}</p>
                <p><strong>color_id:</strong> {int(item.get("color_id", 0) or 0)}</p>
                <p><strong>element_id:</strong> {escape(str(item.get("element_id") or "n/a"))}</p>
                <p><strong>qty:</strong> {int(item.get("qty", 0) or 0)}</p>
              </div>
            </article>
            """
            for item in items
        )
        if items
        else "<div class='empty'>No catalog-backed inventory rows found for this set.</div>"
    )

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Set Part Image Library: {escape(str(set_num))}</title>
          <style>
            body {{
              margin: 0;
              padding: 24px;
              background: linear-gradient(180deg, #eef4f8 0%, #f7fafc 100%);
              color: #17301f;
              font-family: Arial, sans-serif;
            }}
            .shell {{
              max-width: 1700px;
              margin: 0 auto;
            }}
            .hero {{
              background: #fff;
              border: 1px solid #d7e0e9;
              border-radius: 16px;
              padding: 18px 20px;
              margin-bottom: 18px;
              box-shadow: 0 10px 28px rgba(20, 42, 58, 0.08);
            }}
            .hero h1 {{
              margin: 0 0 8px;
              font-size: 30px;
            }}
            .hero p {{
              margin: 6px 0;
            }}
            .grid {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
              gap: 14px;
            }}
            .tile {{
              background: #fff;
              border: 1px solid #d7e0e9;
              border-radius: 16px;
              overflow: hidden;
              box-shadow: 0 10px 28px rgba(20, 42, 58, 0.08);
            }}
            .thumb-wrap {{
              height: 170px;
              display: flex;
              align-items: center;
              justify-content: center;
              background: linear-gradient(180deg, #f3f7fa 0%, #eaf1f5 100%);
              padding: 12px;
            }}
            .thumb-wrap img {{
              display: block;
              max-width: 100%;
              max-height: 100%;
              object-fit: contain;
            }}
            .thumb-missing {{
              color: #788994;
              font-size: 13px;
            }}
            .meta {{
              padding: 12px 14px 14px;
            }}
            .meta p {{
              margin: 5px 0;
              font-size: 13px;
              line-height: 1.35;
              word-break: break-word;
            }}
            .empty {{
              background: #fff;
              border: 1px dashed #c8d6e4;
              border-radius: 16px;
              padding: 28px;
              color: #586b78;
            }}
          </style>
        </head>
        <body>
          <div class="shell">
            <section class="hero">
              <h1>Set Part Image Library</h1>
              <p><strong>Set:</strong> {escape(str(payload.get("set_num") or set_num))}</p>
              <p><strong>Inventory:</strong> {escape(str(payload.get("inventory_set_num") or ""))}</p>
              <p><strong>Inventory ID:</strong> {int(payload.get("inventory_id", 0) or 0)}</p>
              <p><strong>Version:</strong> {int(payload.get("version", 0) or 0)}</p>
              <p><strong>Unique tiles:</strong> {int(payload.get("tile_count", 0) or 0)}</p>
              <p><strong>Tiles with images:</strong> {int(payload.get("with_images", 0) or 0)}</p>
            </section>
            <section class="grid">
              {tiles_html}
            </section>
          </div>
        </body>
        </html>
        """
    )


@router.get("/debug/openai-callout-boxes", response_class=HTMLResponse)
def debug_openai_callout_boxes(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
    model: str = Query(_OPENAI_CALLOUT_MODEL),
):
    path = debug_service.resolve_page_image_path(str(set_num), int(page))
    if path is None:
        raise HTTPException(status_code=404, detail="Page image not found")

    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not load page image")

    page_height, page_width = img.shape[:2]
    detected = step_detector_service.detect_steps(str(set_num), int(page))
    step_boxes = _contact_sheet_step_boxes_from_detected(detected)
    local_candidates = _build_material_crop_candidates(
        img,
        page_width=page_width,
        page_height=page_height,
        step_boxes=step_boxes,
        include_minifig=False,
    )

    openai_error: Optional[str] = None
    openai_boxes: List[Dict[str, Any]] = []
    try:
        openai_boxes = _detect_openai_callout_boxes(
            set_num=str(set_num),
            page=int(page),
            image_path=path,
            page_width=page_width,
            page_height=page_height,
            model=str(model or _OPENAI_CALLOUT_MODEL),
        )
    except Exception as exc:
        openai_error = str(exc)

    overlay = img.copy()
    for idx, item in enumerate(local_candidates, start=1):
        coords = list(item.get("coords", []) or [])
        if len(coords) < 4:
            continue
        x, y, w, h = [int(value or 0) for value in coords[:4]]
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 165, 255), 2)
        local_label = f"L{idx} {str(item.get('candidate_origin', 'local'))}"
        cv2.putText(
            overlay,
            local_label,
            (x + 4, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 110, 180),
            2,
            cv2.LINE_AA,
        )

    for idx, item in enumerate(openai_boxes, start=1):
        coords = list(item.get("coords", []) or [])
        if len(coords) < 4:
            continue
        x, y, w, h = [int(value or 0) for value in coords[:4]]
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (40, 225, 70), 3)
        qty_text = ",".join(str(v) for v in (item.get("qty_text", []) or []))
        step_label = "?" if item.get("step") in (None, "") else str(int(item.get("step") or 0))
        label = f"O{idx} s{step_label} c{float(item.get('confidence', 0.0) or 0.0):.2f}"
        if qty_text:
            label += f" {qty_text}"
        cv2.putText(
            overlay,
            label,
            (x + 4, max(22, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 140, 30),
            2,
            cv2.LINE_AA,
        )

    overlay_data_uri = _encode_debug_image_data_uri(overlay)
    original_data_uri = _encode_debug_image_data_uri(img)

    openai_tiles_html = (
        "\n".join(
            f"""
            <figure class="crop-tile openai">
              <img src="{_encode_contact_sheet_crop(img, list(item.get('coords', []) or []), max_edge=320) or ''}" alt="OpenAI crop" loading="lazy" />
              <figcaption>
                <strong>step:</strong> {escape('' if item.get('step') is None else str(int(item.get('step') or 0)))}<br/>
                <strong>conf:</strong> {float(item.get('confidence', 0.0) or 0.0):.3f}<br/>
                <strong>qty_text:</strong> {escape(', '.join(str(v) for v in (item.get('qty_text', []) or [])) or 'n/a')}<br/>
                <strong>box:</strong> {escape(f"{item['coords'][0]},{item['coords'][1]},{item['coords'][2]},{item['coords'][3]}")}
              </figcaption>
            </figure>
            """
            for item in openai_boxes
            if (_encode_contact_sheet_crop(img, list(item.get("coords", []) or []), max_edge=320) is not None)
        )
        if openai_boxes
        else "<p class='crop-empty'>No OpenAI callout boxes returned.</p>"
    )

    local_tiles_html = (
        "\n".join(
            f"""
            <figure class="crop-tile local">
              <img src="{_encode_contact_sheet_crop(img, list(item.get('coords', []) or []), max_edge=320) or ''}" alt="Local crop" loading="lazy" />
              <figcaption>
                <strong>{escape(str(item.get('candidate_origin', 'local')))}</strong><br/>
                {escape(str(item.get('candidate_type', 'normal_candidate')))}<br/>
                {escape(f"{item['coords'][0]},{item['coords'][1]},{item['coords'][2]},{item['coords'][3]}")}
              </figcaption>
            </figure>
            """
            for item in local_candidates
            if (_encode_contact_sheet_crop(img, list(item.get("coords", []) or []), max_edge=320) is not None)
        )
        if local_candidates
        else "<p class='crop-empty'>No local detector boxes available for this page.</p>"
    )

    error_block = ""
    if openai_error:
        error_block = f"""
        <section class="error-card">
          <h2>OpenAI Vision Unavailable</h2>
          <p>{escape(openai_error)}</p>
          <p>Set `OPENAI_API_KEY` and ensure the `openai` Python SDK is installed to enable this debug page.</p>
        </section>
        """

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>OpenAI Callout Boxes: {escape(str(set_num))} page {int(page)}</title>
          <style>
            body {{
              margin: 0;
              padding: 24px;
              background: linear-gradient(180deg, #eef4f8 0%, #f7fafc 100%);
              color: #17301f;
              font-family: Arial, sans-serif;
            }}
            .shell {{
              max-width: 1800px;
              margin: 0 auto;
            }}
            .hero, .panel, .error-card {{
              background: #fff;
              border: 1px solid #d7e0e9;
              border-radius: 16px;
              box-shadow: 0 10px 28px rgba(20, 42, 58, 0.08);
            }}
            .hero, .panel, .error-card {{
              padding: 16px 18px;
              margin-bottom: 18px;
            }}
            .hero h1, .panel h2, .error-card h2 {{
              margin: 0 0 8px;
            }}
            .hero p, .panel p, .error-card p {{
              margin: 6px 0;
            }}
            .viewer-grid {{
              display: grid;
              grid-template-columns: 1fr 1fr;
              gap: 16px;
              margin-bottom: 18px;
            }}
            .viewer-grid img {{
              display: block;
              width: 100%;
              height: auto;
              border-radius: 12px;
              background: #eef4f8;
            }}
            .crop-sections {{
              display: grid;
              grid-template-columns: 1fr 1fr;
              gap: 16px;
            }}
            .crop-grid {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
              gap: 10px;
            }}
            .crop-tile {{
              margin: 0;
              border: 1px solid #d8e3eb;
              border-radius: 12px;
              overflow: hidden;
              background: #f8fbfd;
            }}
            .crop-tile.openai {{
              border-color: #bde9c4;
              background: #f3fcf4;
            }}
            .crop-tile.local {{
              border-color: #ead0a8;
              background: #fff8ee;
            }}
            .crop-tile img {{
              display: block;
              width: 100%;
              height: 160px;
              object-fit: contain;
              background: linear-gradient(180deg, #eef4f8 0%, #f8fafc 100%);
            }}
            .crop-tile figcaption {{
              padding: 8px;
              font-size: 12px;
              line-height: 1.35;
              color: #3e5563;
              word-break: break-word;
            }}
            .crop-empty {{
              margin: 0;
              padding: 10px 12px;
              border: 1px dashed #c8d6e4;
              border-radius: 12px;
              color: #697d8a;
              background: #fbfdff;
              font-size: 13px;
            }}
            .legend {{
              font-size: 13px;
              color: #5c6f7b;
            }}
          </style>
        </head>
        <body>
          <div class="shell">
            <section class="hero">
              <h1>OpenAI Callout Boxes</h1>
              <p><strong>Set:</strong> {escape(str(set_num))}</p>
              <p><strong>Page:</strong> {int(page)}</p>
              <p><strong>Model:</strong> {escape(str(model or _OPENAI_CALLOUT_MODEL))}</p>
              <p><strong>OpenAI boxes:</strong> {len(openai_boxes)} | <strong>Local boxes:</strong> {len(local_candidates)}</p>
              <p class="legend">Bright green = OpenAI vision callout boxes. Amber = current local detector boxes.</p>
            </section>

            {error_block}

            <section class="viewer-grid">
              <article class="panel">
                <h2>Original Page</h2>
                <img src="{original_data_uri}" alt="Original page {int(page)}" />
              </article>
              <article class="panel">
                <h2>Overlay Comparison</h2>
                <img src="{overlay_data_uri}" alt="Overlay page {int(page)}" />
              </article>
            </section>

            <section class="crop-sections">
              <article class="panel">
                <h2>OpenAI Crops</h2>
                <div class="crop-grid">
                  {openai_tiles_html}
                </div>
              </article>
              <article class="panel">
                <h2>Local Detector Crops</h2>
                <div class="crop-grid">
                  {local_tiles_html}
                </div>
              </article>
            </section>
          </div>
        </body>
        </html>
        """
    )


@router.get("/debug/bag-parts-lab", response_class=HTMLResponse)
def debug_bag_parts_lab(
    set_num: str = Query(...),
    bag: int = Query(..., ge=1),
    include_minifig: int = Query(0),
    top_k: int = Query(5, ge=1, le=10),
):
    include_minifig = _coerce_optional_debug_int(include_minifig) or 0
    top_k = max(1, min(10, _coerce_optional_debug_int(top_k) or 5))
    rendered_pages, start_page, end_page = _resolve_bag_page_range(str(set_num), int(bag))
    library_payload = _load_set_part_image_library(str(set_num))
    library_items = list(library_payload.get("items", []) or [])

    page_blocks: List[str] = []
    crop_count = 0
    matchable_crop_count = 0
    matched_crop_count = 0

    for page in rendered_pages:
        if int(page) < start_page or int(page) > end_page:
            continue

        image_path = debug_service.resolve_page_image_path(str(set_num), int(page))
        if image_path is None:
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        page_height, page_width = img.shape[:2]
        detected = step_detector_service.detect_steps(str(set_num), int(page))
        step_boxes = _contact_sheet_step_boxes_from_detected(detected)
        crop_candidates = _build_material_crop_candidates(
            img,
            page_width=page_width,
            page_height=page_height,
            step_boxes=step_boxes,
            include_minifig=bool(int(include_minifig or 0)),
        )
        callout_candidates = [
            item
            for item in crop_candidates
            if str(item.get("candidate_origin", "")) == "callout_box_candidate"
        ]
        if not bool(int(include_minifig or 0)):
            callout_candidates = [
                item
                for item in callout_candidates
                if str(item.get("candidate_type", "")) != "excluded_minifig_step"
            ]

        if not callout_candidates:
            continue

        crop_count += len(callout_candidates)
        page_crop_cards: List[str] = []
        for idx, candidate in enumerate(callout_candidates, start=1):
            coords = list(candidate.get("coords", []) or [])
            crop_img = page_analyzer.crop(img, coords)
            crop_data_uri = _encode_contact_sheet_crop(img, coords, max_edge=360)
            qty_payload = _extract_crop_qty_status(crop_img)
            color_match_payload = _score_crop_against_library_color_only(
                crop_img,
                library_items,
                top_k=top_k,
            )
            matches = list(color_match_payload.get("matches", []) or [])
            if crop_img is not None and crop_img.size != 0:
                matchable_crop_count += 1
            if matches:
                matched_crop_count += 1

            nearest_color_payload = dict(color_match_payload.get("nearest_color", {}) or {})
            nearest_color_name = str(
                nearest_color_payload.get("name")
                or nearest_color_payload.get("color_name")
                or "n/a"
            )
            nearest_color_rgb = str(
                nearest_color_payload.get("rgb")
                or nearest_color_payload.get("color_rgb")
                or ""
            )
            estimated_swatch = (
                f"rgb{tuple(color_match_payload.get('crop_rgb'))}"
                if color_match_payload.get("crop_rgb") is not None
                else "transparent"
            )
            nearest_swatch = (
                f"#{nearest_color_rgb}"
                if nearest_color_rgb
                else "transparent"
            )

            matches_html = (
                "\n".join(
                    f"""
                    <article class="match-card">
                      <div class="match-thumb">
                        {f'<img src="{escape(str(match.get("img_url") or ""))}" alt="{escape(str(match.get("part_num") or ""))}" loading="lazy" />' if match.get("img_url") else '<div class="crop-missing">No image</div>'}
                      </div>
                      <div class="match-meta">
                        <p><strong>combined_score:</strong> {float(match.get("combined_score", 0.0) or 0.0):.3f}</p>
                        <p><strong>color_score:</strong> {float(match.get("color_score", 0.0) or 0.0):.3f}</p>
                        <p><strong>image_score:</strong> unavailable</p>
                        <p><strong>part_num:</strong> {escape(str(match.get("part_num") or ""))}</p>
                        <p><strong>color_id:</strong> {int(match.get("color_id", 0) or 0)}</p>
                        <p><strong>color:</strong> {escape(str(match.get("color_name") or "n/a"))} {escape(f"#{str(match.get('color_rgb') or '')}" if match.get("color_rgb") else "")}</p>
                        <p><strong>element_id:</strong> {escape(str(match.get("element_id") or "n/a"))}</p>
                        <p><strong>qty:</strong> {int(match.get("qty", 0) or 0)}</p>
                      </div>
                    </article>
                    """
                    for match in matches
                )
                if matches
                else "<p class='match-empty'>No local catalog candidates available for this crop.</p>"
            )

            page_crop_cards.append(
                f"""
                <article class="crop-card">
                  <div class="crop-head">
                    <h4>Page {int(page)} Crop {idx}</h4>
                    <p><strong>type:</strong> {escape(str(candidate.get("candidate_type", "normal_candidate")))} | <strong>coords:</strong> {escape(str(candidate.get("coords_label", _format_debug_coords(coords))))}</p>
                    <p><strong>qty status:</strong> {escape(str(qty_payload.get("status", "")))} | <strong>qty:</strong> {escape("" if qty_payload.get("qty") is None else f"x{int(qty_payload.get('qty'))}")}</p>
                    <p><strong>qty OCR:</strong> {escape(str(qty_payload.get("raw_text", "")) or "n/a")}</p>
                    <p><strong>estimated crop RGB:</strong> {escape(str(tuple(color_match_payload.get("crop_rgb"))) if color_match_payload.get("crop_rgb") is not None else "n/a")}</p>
                    <p><strong>nearest catalog colour:</strong> {escape(nearest_color_name)} {escape(f"#{nearest_color_rgb}" if nearest_color_rgb else "")}</p>
                    <p><strong>tight object box:</strong> {escape(_format_debug_coords(list(color_match_payload.get("tight_box", []) or [])))}</p>
                    <p><strong>pixels used:</strong> {int(color_match_payload.get("selected_pixel_count", 0) or 0)} ({float(color_match_payload.get("selected_fraction", 0.0) or 0.0) * 100.0:.1f}%)</p>
                    <div class="color-debug-row">
                      <div class="swatch-group">
                        <span class="swatch-label">Estimated</span>
                        <span class="swatch" style="background:{escape(estimated_swatch)};"></span>
                      </div>
                      <div class="swatch-group">
                        <span class="swatch-label">Nearest catalog</span>
                        <span class="swatch" style="background:{escape(nearest_swatch)};"></span>
                      </div>
                    </div>
                  </div>
                  <div class="crop-body">
                    <div class="crop-preview">
                      {f'<img src="{crop_data_uri}" alt="Crop preview" loading="lazy" />' if crop_data_uri else '<div class="crop-missing">Crop unavailable</div>'}
                      <div class="color-debug-previews">
                        {f'<img src="{escape(str(color_match_payload.get("tight_crop_data_uri") or ""))}" alt="Tight object crop" loading="lazy" />' if color_match_payload.get("tight_crop_data_uri") else '<div class="crop-missing">No tight crop</div>'}
                        {f'<img src="{escape(str(color_match_payload.get("debug_data_uri") or ""))}" alt="Pixels used for colour estimate" loading="lazy" />' if color_match_payload.get("debug_data_uri") else '<div class="crop-missing">No pixel mask preview</div>'}
                      </div>
                    </div>
                    <div class="matches-grid">
                      {matches_html}
                    </div>
                  </div>
                </article>
                """
            )

        if page_crop_cards:
            page_blocks.append(
                f"""
                <section class="page-group">
                  <div class="page-group-head">
                    <h3>Page {int(page)}</h3>
                    <p><a href="/debug/step-part-lab?set_num={escape(str(set_num))}&page={int(page)}{'&include_minifig=1' if bool(int(include_minifig or 0)) else ''}" target="_blank">Open step-part lab</a></p>
                  </div>
                  <div class="page-crops">
                    {' '.join(page_crop_cards)}
                  </div>
                </section>
                """
            )

    body_html = (
        "\n".join(page_blocks)
        if page_blocks
        else "<div class='empty'>No callout_box_candidate crops found for this bag with the current debug filters.</div>"
    )

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Bag Parts Lab: {escape(str(set_num))} bag {int(bag)}</title>
          <style>
            body {{
              margin: 0;
              padding: 24px;
              background: linear-gradient(180deg, #eef4f8 0%, #f7fafc 100%);
              color: #17301f;
              font-family: Arial, sans-serif;
            }}
            .shell {{
              max-width: 1800px;
              margin: 0 auto;
            }}
            .hero, .page-group, .crop-card {{
              background: #fff;
              border: 1px solid #d7e0e9;
              border-radius: 16px;
              box-shadow: 0 10px 28px rgba(20, 42, 58, 0.08);
            }}
            .hero {{
              padding: 18px 20px;
              margin-bottom: 18px;
            }}
            .hero h1, .page-group-head h3, .crop-head h4 {{
              margin: 0 0 8px;
            }}
            .hero p, .page-group-head p, .crop-head p, .match-meta p {{
              margin: 6px 0;
            }}
            .page-group {{
              padding: 16px;
              margin-bottom: 16px;
            }}
            .page-group-head {{
              margin-bottom: 12px;
            }}
            .page-group-head a {{
              color: #1d6fa5;
              text-decoration: none;
            }}
            .page-crops {{
              display: grid;
              gap: 14px;
            }}
            .crop-card {{
              padding: 14px;
            }}
            .crop-body {{
              display: grid;
              grid-template-columns: 280px 1fr;
              gap: 16px;
              align-items: start;
            }}
            .crop-preview {{
              min-height: 200px;
              border: 1px solid #d9e5ec;
              border-radius: 12px;
              background: linear-gradient(180deg, #f5f9fc 0%, #edf3f7 100%);
              display: flex;
              align-items: center;
              justify-content: center;
              padding: 12px;
            }}
            .crop-preview img {{
              display: block;
              max-width: 100%;
              max-height: 240px;
              object-fit: contain;
            }}
            .color-debug-row {{
              display: flex;
              gap: 12px;
              flex-wrap: wrap;
              margin-top: 8px;
            }}
            .swatch-group {{
              display: inline-flex;
              align-items: center;
              gap: 8px;
              font-size: 12px;
              color: #4d6371;
            }}
            .swatch-label {{
              white-space: nowrap;
            }}
            .swatch {{
              display: inline-block;
              width: 22px;
              height: 22px;
              border-radius: 6px;
              border: 1px solid #bccddb;
              box-shadow: inset 0 0 0 1px rgba(255,255,255,0.35);
            }}
            .color-debug-previews {{
              margin-top: 12px;
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 8px;
            }}
            .color-debug-previews img {{
              display: block;
              width: 100%;
              max-height: 120px;
              object-fit: contain;
              border: 1px solid #d9e5ec;
              border-radius: 8px;
              background: linear-gradient(180deg, #f5f9fc 0%, #edf3f7 100%);
            }}
            .crop-missing, .match-empty, .empty {{
              color: #6c7f8c;
              font-size: 13px;
            }}
            .matches-grid {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
              gap: 10px;
            }}
            .match-card {{
              border: 1px solid #d7e0e9;
              border-radius: 12px;
              overflow: hidden;
              background: #fbfdff;
            }}
            .match-thumb {{
              height: 140px;
              background: linear-gradient(180deg, #f2f7fb 0%, #e9f0f5 100%);
              display: flex;
              align-items: center;
              justify-content: center;
              padding: 10px;
            }}
            .match-thumb img {{
              display: block;
              max-width: 100%;
              max-height: 100%;
              object-fit: contain;
            }}
            .match-meta {{
              padding: 10px;
              font-size: 13px;
              line-height: 1.35;
              word-break: break-word;
            }}
            .empty {{
              background: #fff;
              border: 1px dashed #c8d6e4;
              border-radius: 16px;
              padding: 28px;
            }}
          </style>
        </head>
        <body>
          <div class="shell">
            <section class="hero">
              <h1>Bag Parts Lab</h1>
              <p><strong>Set:</strong> {escape(str(set_num))}</p>
              <p><strong>Bag:</strong> {int(bag)}</p>
              <p><strong>Bag page range:</strong> {int(start_page)}-{int(end_page)}</p>
              <p><strong>Set library tiles:</strong> {int(library_payload.get("tile_count", 0) or 0)} | <strong>with images:</strong> {int(library_payload.get("with_images", 0) or 0)}</p>
              <p><strong>Callout crops shown:</strong> {crop_count} | <strong>matchable crops:</strong> {matchable_crop_count} | <strong>crops with preview matches:</strong> {matched_crop_count}</p>
              <p><strong>Top candidates per crop:</strong> {top_k} | <strong>Include minifig override:</strong> {"on" if bool(int(include_minifig or 0)) else "off"}</p>
              <p>Preview only. This pass uses local Rebrickable catalog colours from the set inventory first. Image shape matching is intentionally disabled here.</p>
            </section>
            {body_html}
          </div>
        </body>
        </html>
        """
    )


@router.get("/debug/step-part-lab", response_class=HTMLResponse)
def debug_step_part_lab(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
    x1: Optional[int] = Query(None),
    y1: Optional[int] = Query(None),
    x2: Optional[int] = Query(None),
    y2: Optional[int] = Query(None),
    include_minifig: int = Query(0),
):
    path = debug_service.resolve_page_image_path(set_num, page)
    if path is None:
        raise HTTPException(status_code=404, detail="Page image not found")

    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not load page image")

    page_height, page_width = img.shape[:2]
    include_minifig = _coerce_optional_debug_int(include_minifig) or 0
    detected = step_detector_service.detect_steps(str(set_num), int(page))
    step_boxes = _contact_sheet_step_boxes_from_detected(detected)
    callout_detection = _detect_callout_box_candidate_specs(
        img,
        page_width=page_width,
        page_height=page_height,
        step_boxes=step_boxes,
    )
    crop_candidates = _build_material_crop_candidates(
        img,
        page_width=page_width,
        page_height=page_height,
        step_boxes=step_boxes,
        include_minifig=bool(int(include_minifig or 0)),
        detection_payload=callout_detection,
    )

    x1 = _coerce_optional_debug_int(x1)
    y1 = _coerce_optional_debug_int(y1)
    x2 = _coerce_optional_debug_int(x2)
    y2 = _coerce_optional_debug_int(y2)
    selection_values = [x1, y1, x2, y2]
    has_selection = any(value is not None for value in selection_values)
    if has_selection and not all(value is not None for value in selection_values):
        raise HTTPException(
            status_code=400,
            detail="Provide x1, y1, x2, y2 together",
        )

    selected_box: Optional[List[int]] = None
    selected_crop_data_uri: Optional[str] = None
    selected_summary = "No manual crop selected."
    if has_selection:
        raw_x1 = int(x1 or 0)
        raw_y1 = int(y1 or 0)
        raw_x2 = int(x2 or 0)
        raw_y2 = int(y2 or 0)
        left = min(raw_x1, raw_x2)
        top = min(raw_y1, raw_y2)
        right = max(raw_x1, raw_x2)
        bottom = max(raw_y1, raw_y2)
        selected_box = _clamp_contact_sheet_box(
            [left, top, right - left, bottom - top],
            page_width=page_width,
            page_height=page_height,
        )
        if selected_box is None:
            raise HTTPException(status_code=400, detail="Manual crop rectangle is outside the page")
        selected_crop_data_uri = _encode_contact_sheet_crop(img, selected_box, max_edge=600)
        selected_summary = (
            f"manual_crop x={selected_box[0]}, y={selected_box[1]}, "
            f"w={selected_box[2]}, h={selected_box[3]}"
        )

    original_data_uri = _encode_debug_image_data_uri(img)

    step_overlay = img.copy()
    _draw_step_boxes_overlay(step_overlay, step_boxes)
    step_overlay_data_uri = _encode_debug_image_data_uri(step_overlay)

    grid_overlay = img.copy()
    _draw_grid_overlay(grid_overlay, spacing=100)
    _draw_step_anchor_debug_overlay(
        grid_overlay,
        list(callout_detection.get("step_anchors", []) or []),
    )
    _draw_search_zone_overlay(
        grid_overlay,
        list(callout_detection.get("search_regions", []) or []),
    )
    _draw_rejected_callout_overlay(
        grid_overlay,
        list(callout_detection.get("rejected_candidates", []) or []),
    )
    _draw_candidate_boxes_overlay(
        grid_overlay,
        crop_candidates,
        include_minifig=bool(int(include_minifig or 0)),
    )
    if selected_box is not None:
        x = int(selected_box[0])
        y = int(selected_box[1])
        w = int(selected_box[2])
        h = int(selected_box[3])
        cv2.rectangle(grid_overlay, (x, y), (x + w, y + h), (170, 40, 170), 3)
        cv2.putText(
            grid_overlay,
            "manual_crop",
            (x + 4, max(18, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (120, 25, 120),
            2,
            cv2.LINE_AA,
        )
    grid_overlay_data_uri = _encode_debug_image_data_uri(grid_overlay)

    candidate_tiles_html = (
        "\n".join(
            f"""
            <figure class="crop-tile candidate{" callout" if str(tile.get("candidate_origin", "")) == "callout_box_candidate" else ""}{" excluded" if str(tile.get("candidate_type", "")) == "excluded_minifig_step" and not bool(int(include_minifig or 0)) else ""}">
              <img src="{tile["data_uri"]}" alt="{escape(tile["label"])}" loading="lazy" />
              <figcaption>
                <strong>C{idx}</strong> {escape(str(tile.get("candidate_origin", tile["label"])))}<br/>
                {escape(str(tile.get("candidate_type", "normal_candidate")))}<br/>
                {escape(str(tile.get("exclusion_reason", "")) or "")}<br/>
                full_callout_box: {escape(str(tile.get("coords_label", _format_debug_coords(tile["coords"]))))}<br/>
                inner_detected_region: {escape(_format_debug_coords(list(tile.get("inner_detected_region", []) or [])))}<br/>
                qty: {escape(", ".join(str(value) for value in list(tile.get("detected_qty_text", []) or [])) or "none")}<br/>
                qty nums: {escape(", ".join(str(int(value)) for value in list(tile.get("detected_qty_numbers", []) or [])) or "none")}<br/>
              </figcaption>
            </figure>
            """
            for idx, tile in enumerate(crop_candidates, start=1)
        )
        if crop_candidates
        else "<p class='crop-empty'>No crop candidates generated for this page.</p>"
    )

    step_box_rows = (
        "\n".join(
            f"""
            <tr>
              <td>{int(item.get("step_number", 0) or 0) if int(item.get("step_number", 0) or 0) > 0 else "?"}</td>
              <td>{escape(str(item.get("source", "unknown")) or "unknown")}</td>
              <td>{int(item.get("x", 0) or 0)}</td>
              <td>{int(item.get("y", 0) or 0)}</td>
              <td>{int(item.get("w", 0) or 0)}</td>
              <td>{int(item.get("h", 0) or 0)}</td>
            </tr>
            """
            for item in step_boxes
        )
        if step_boxes
        else "<tr><td colspan='6'>No detected main step boxes.</td></tr>"
    )

    selected_crop_html = ""
    if selected_crop_data_uri is not None and selected_box is not None:
        selected_crop_html = f"""
        <section class="panel manual-panel">
          <h2>Manual Crop</h2>
          <p>{escape(selected_summary)}</p>
          <img src="{selected_crop_data_uri}" alt="Manual selected crop" />
        </section>
        """

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Step Part Lab: {escape(str(set_num))} page {int(page)}</title>
          <style>
            body {{
              margin: 0;
              padding: 24px;
              background: #f4f7fa;
              color: #17301f;
              font-family: Arial, sans-serif;
            }}
            .shell {{
              max-width: 1800px;
              margin: 0 auto;
            }}
            .hero, .panel {{
              background: #fff;
              border: 1px solid #d7e0e9;
              border-radius: 16px;
              box-shadow: 0 10px 28px rgba(20, 42, 58, 0.08);
            }}
            .hero {{
              padding: 18px 20px;
              margin-bottom: 18px;
            }}
            .hero h1 {{
              margin: 0 0 8px;
              font-size: 30px;
            }}
            .hero p {{
              margin: 6px 0;
            }}
            .image-grid {{
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
              gap: 16px;
              margin-bottom: 18px;
            }}
            .panel {{
              padding: 16px;
            }}
            .panel h2 {{
              margin: 0 0 10px;
              font-size: 20px;
            }}
            .panel img {{
              display: block;
              width: 100%;
              height: auto;
              border-radius: 12px;
              background: #eef4f8;
            }}
            .legend {{
              font-size: 13px;
              color: #5c6f7b;
            }}
            .tables {{
              display: grid;
              grid-template-columns: minmax(300px, 1fr) minmax(400px, 1.2fr);
              gap: 16px;
              align-items: start;
            }}
            .crop-grid {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
              gap: 10px;
            }}
            .crop-tile {{
              margin: 0;
              border: 1px solid #d8e3eb;
              border-radius: 12px;
              overflow: hidden;
              background: #f8fbfd;
            }}
            .crop-tile.candidate {{
              border-color: #e0d3c1;
              background: #fffaf2;
            }}
            .crop-tile.candidate.callout {{
              border-color: #bfe0c5;
              background: #f4fbf5;
            }}
            .crop-tile.candidate.excluded {{
              border-color: #c9c9c9;
              background: #f1f1f1;
              opacity: 0.72;
            }}
            .crop-tile img {{
              display: block;
              width: 100%;
              height: 130px;
              object-fit: contain;
              background: linear-gradient(180deg, #eef4f8 0%, #f8fafc 100%);
            }}
            .crop-tile figcaption {{
              padding: 8px;
              font-size: 12px;
              line-height: 1.35;
              color: #3e5563;
              word-break: break-word;
            }}
            .crop-empty {{
              margin: 0;
              padding: 10px 12px;
              border: 1px dashed #c8d6e4;
              border-radius: 12px;
              color: #697d8a;
              background: #fbfdff;
              font-size: 13px;
            }}
            table {{
              width: 100%;
              border-collapse: collapse;
              font-size: 13px;
            }}
            th, td {{
              padding: 8px 10px;
              border-bottom: 1px solid #e5edf3;
              text-align: left;
              vertical-align: top;
            }}
            th {{
              background: #f7fafc;
            }}
            code {{
              background: #f0f4f7;
              padding: 2px 6px;
              border-radius: 6px;
            }}
            .manual-panel {{
              margin-top: 16px;
            }}
          </style>
        </head>
        <body>
          <div class="shell">
            <section class="hero">
              <h1>Step Part Lab</h1>
              <p><strong>Set:</strong> {escape(str(set_num))}</p>
              <p><strong>Page:</strong> {int(page)}</p>
              <p><strong>Detected main step boxes:</strong> {len(step_boxes)}</p>
              <p><strong>Current crop candidates:</strong> {len(crop_candidates)}</p>
              <p><strong>Manual crop:</strong> {escape(selected_summary)}</p>
              <p><strong>Include minifig override:</strong> {"on" if bool(int(include_minifig or 0)) else "off"}</p>
              <p class="legend">Purple = step number bbox. Cyan = step-anchored search zone. Green = chosen full_callout_box. Blue dashed = inner_detected_region seed. Grey = rejected callout candidate or excluded candidate. Purple outline = manual_crop from URL params. Grid labels show page coordinates.</p>
            </section>

            <section class="image-grid">
              <article class="panel">
                <h2>Original Full Page</h2>
                <img src="{original_data_uri}" alt="Original page {int(page)}" />
              </article>
              <article class="panel">
                <h2>Detected Step Boxes</h2>
                <img src="{step_overlay_data_uri}" alt="Detected step boxes on page {int(page)}" />
              </article>
              <article class="panel">
                <h2>Grid Overlay + Crop Candidates</h2>
                <img src="{grid_overlay_data_uri}" alt="Grid overlay for page {int(page)}" />
              </article>
            </section>

            <section class="tables">
              <article class="panel">
                <h2>Current Crop Candidates</h2>
                <div class="crop-grid">
                  {candidate_tiles_html}
                </div>
              </article>
              <article class="panel">
                <h2>Detected Main Step Boxes</h2>
                <table>
                  <thead>
                    <tr>
                      <th>Step</th>
                      <th>Source</th>
                      <th>X</th>
                      <th>Y</th>
                      <th>W</th>
                      <th>H</th>
                    </tr>
                  </thead>
                  <tbody>
                    {step_box_rows}
                  </tbody>
                </table>
                <p><strong>Manual crop URL format:</strong> <code>/debug/step-part-lab?set_num={escape(str(set_num))}&page={int(page)}&x1=0&y1=0&x2=400&y2=300</code></p>
                <p><strong>Include minifig URL:</strong> <code>/debug/step-part-lab?set_num={escape(str(set_num))}&page={int(page)}&include_minifig=1</code></p>
              </article>
            </section>

            {selected_crop_html}
          </div>
        </body>
        </html>
        """
    )


@router.get("/debug/bag-step-contact-sheet", response_class=HTMLResponse)
def debug_bag_step_contact_sheet(
    set_num: str = Query(...),
    bag: int = Query(..., ge=1),
    include_minifig: int = Query(0),
):
    include_minifig = _coerce_optional_debug_int(include_minifig) or 0
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        raise HTTPException(status_code=404, detail="No rendered pages found for set")

    rendered_pages = _list_rendered_pages(pages_dir)
    if not rendered_pages:
        raise HTTPException(status_code=404, detail="No rendered pages found in pages dir")

    bag_truth = sorted(
        list(bag_truth_store.get_bag_truth(set_num) or []),
        key=lambda item: (
            int(item.get("bag_number", 0) or 0),
            int(item.get("start_page", 0) or 0),
        ),
    )
    bag_row = next(
        (
            item
            for item in bag_truth
            if int(item.get("bag_number", 0) or 0) == int(bag)
        ),
        None,
    )
    if bag_row is None:
        raise HTTPException(status_code=404, detail=f"Bag {bag} not found in saved bag truth")

    start_page = int(bag_row.get("start_page", 0) or 0)
    next_bag_row = next(
        (
            item
            for item in bag_truth
            if int(item.get("bag_number", 0) or 0) > int(bag)
        ),
        None,
    )
    end_page = (
        int(next_bag_row.get("start_page", 0) or 0) - 1
        if next_bag_row is not None
        else int(rendered_pages[-1])
    )
    if end_page < start_page:
        end_page = start_page

    page_cards: List[str] = []
    all_visible_steps: List[int] = []
    for page in rendered_pages:
        if int(page) < start_page or int(page) > end_page:
            continue

        detected: Optional[Dict[str, Any]] = None
        green_step_row = _get_green_step_debug_row(str(set_num), int(page))
        known_step_boxes = _contact_sheet_step_boxes_from_green_row(green_step_row)
        cached_steps = step_sequence_bag_service.get_cached_page_main_steps(
            str(set_num),
            int(page),
        )
        if cached_steps is not None:
            visible_steps = list(cached_steps)
        else:
            if green_step_row and (green_step_row.get("main_steps") or []):
                visible_steps = sorted(
                    {
                        int(item.get("value", 0) or 0)
                        for item in (green_step_row.get("main_steps", []) or [])
                        if int(item.get("value", 0) or 0) > 0
                    }
                )
            else:
                detected = step_detector_service.detect_steps(str(set_num), int(page))
                visible_steps = sorted(
                    {
                        int(item.get("value", 0) or 0)
                        for item in (detected.get("main_steps", []) or [])
                        if int(item.get("value", 0) or 0) > 0
                    }
                )

        if not known_step_boxes:
            if detected is None:
                detected = step_detector_service.detect_steps(str(set_num), int(page))
            known_step_boxes = _contact_sheet_step_boxes_from_detected(detected)

        all_visible_steps.extend(visible_steps)
        image_url = f"/debug/page-image?set_num={escape(str(set_num))}&page={int(page)}"
        steps_label = (
            ", ".join(str(int(value)) for value in visible_steps)
            if visible_steps
            else "unknown"
        )

        step_tiles_html = "<p class='crop-empty'>No known step anchor boxes yet.</p>"
        candidate_tiles_html = "<p class='crop-empty'>No crop candidates generated for this page.</p>"

        image_path = debug_service.resolve_page_image_path(str(set_num), int(page))
        if image_path is not None:
            img = cv2.imread(str(image_path))
            if img is not None:
                page_height, page_width = img.shape[:2]
                step_anchor_tiles = _build_step_anchor_tiles(
                    img,
                    page_width=page_width,
                    page_height=page_height,
                    step_boxes=known_step_boxes,
                )
                material_candidate_tiles = _build_material_crop_candidates(
                    img,
                    page_width=page_width,
                    page_height=page_height,
                    step_boxes=known_step_boxes,
                    include_minifig=bool(int(include_minifig or 0)),
                )

                if step_anchor_tiles:
                    step_tiles_html = "\n".join(
                        f"""
                        <figure class="crop-tile">
                          <img src="{tile["data_uri"]}" alt="{escape(tile["label"])}" loading="lazy" />
                          <figcaption>
                            <strong>{escape(tile["label"])}</strong><br/>
                            {escape(str(tile["source"]))}<br/>
                            {escape(_format_debug_coords(tile["coords"]))}
                          </figcaption>
                        </figure>
                        """
                        for tile in step_anchor_tiles
                    )

                if material_candidate_tiles:
                    candidate_tiles_html = "\n".join(
                        f"""
                        <figure class="crop-tile candidate{" callout" if str(tile.get("candidate_origin", "")) == "callout_box_candidate" else ""}{" excluded" if str(tile.get("candidate_type", "")) == "excluded_minifig_step" and not bool(int(include_minifig or 0)) else ""}">
                          <img src="{tile["data_uri"]}" alt="{escape(tile["label"])}" loading="lazy" />
                          <figcaption>
                            <strong>{escape(str(tile.get("candidate_origin", tile["label"])))}</strong><br/>
                            {escape(str(tile.get("candidate_type", "normal_candidate")))}<br/>
                            {escape(str(tile.get("exclusion_reason", "")) or "")}<br/>
                            full_callout_box: {escape(str(tile.get("coords_label", _format_debug_coords(tile["coords"]))))}<br/>
                            inner_detected_region: {escape(_format_debug_coords(list(tile.get("inner_detected_region", []) or [])))}<br/>
                            qty: {escape(", ".join(str(value) for value in list(tile.get("detected_qty_text", []) or [])) or "none")}<br/>
                            {escape(f"page {int(page)} @ {str(tile.get('coords_label', _format_debug_coords(tile['coords'])))}")}
                          </figcaption>
                        </figure>
                        """
                        for tile in material_candidate_tiles
                    )

        page_cards.append(
            f"""
            <article class="page-card">
              <div class="page-thumb-wrap">
                <a href="{image_url}" target="_blank">
                  <img class="page-thumb" src="{image_url}" alt="Bag {int(bag)} page {int(page)}" loading="lazy" />
                </a>
              </div>
              <div class="page-meta">
                <h3>Page {int(page)}</h3>
                <p><strong>Visible main steps:</strong> {escape(steps_label)}</p>
                <p><strong>Step number if known:</strong> {escape(steps_label)}</p>
                <p><a href="{image_url}" target="_blank">Open original page</a></p>
              </div>
              <section class="crop-section">
                <h4>Known Step Crops</h4>
                <div class="crop-grid">
                  {step_tiles_html}
                </div>
              </section>
              <section class="crop-section">
                <h4>Crop Candidates</h4>
                <div class="crop-grid">
                  {candidate_tiles_html}
                </div>
              </section>
            </article>
            """
        )

    visible_step_min = int(min(all_visible_steps)) if all_visible_steps else None
    visible_step_max = int(max(all_visible_steps)) if all_visible_steps else None
    cards_block = (
        "\n".join(page_cards)
        if page_cards
        else "<div class='empty'>No rendered bag pages found for this bag range.</div>"
    )

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Bag {int(bag)} Step Contact Sheet</title>
          <style>
            body {{
              margin: 0;
              padding: 24px;
              background: linear-gradient(180deg, #edf4fb 0%, #f6f8fb 100%);
              color: #17301f;
              font-family: Arial, sans-serif;
            }}
            .shell {{
              max-width: 1600px;
              margin: 0 auto;
            }}
            .hero {{
              background: #ffffff;
              border: 1px solid #d7e0e9;
              border-radius: 16px;
              padding: 18px 20px;
              margin-bottom: 18px;
              box-shadow: 0 10px 28px rgba(20, 42, 58, 0.08);
            }}
            .hero h1 {{
              margin: 0 0 8px;
              font-size: 30px;
            }}
            .hero p {{
              margin: 4px 0;
            }}
            .grid {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
              gap: 16px;
            }}
            .page-card {{
              background: #ffffff;
              border: 1px solid #d7e0e9;
              border-radius: 16px;
              overflow: hidden;
              box-shadow: 0 10px 28px rgba(20, 42, 58, 0.08);
            }}
            .page-thumb-wrap {{
              background: #dfeaf3;
              aspect-ratio: 1 / 1.35;
            }}
            .page-thumb {{
              width: 100%;
              height: 100%;
              object-fit: contain;
              display: block;
            }}
            .page-meta {{
              padding: 14px;
            }}
            .page-meta h3 {{
              margin: 0 0 8px;
              font-size: 20px;
            }}
            .page-meta p {{
              margin: 6px 0;
              line-height: 1.4;
            }}
            .page-meta a {{
              color: #1d6fa5;
              text-decoration: none;
            }}
            .crop-section {{
              padding: 0 14px 14px;
            }}
            .crop-section h4 {{
              margin: 0 0 10px;
              font-size: 14px;
              letter-spacing: 0.02em;
              text-transform: uppercase;
              color: #48606f;
            }}
            .crop-grid {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
              gap: 10px;
            }}
            .crop-tile {{
              margin: 0;
              border: 1px solid #d8e3eb;
              border-radius: 12px;
              overflow: hidden;
              background: #f8fbfd;
            }}
            .crop-tile.candidate {{
              border-color: #e0d3c1;
              background: #fffaf2;
            }}
            .crop-tile.candidate.callout {{
              border-color: #bfe0c5;
              background: #f4fbf5;
            }}
            .crop-tile.candidate.excluded {{
              border-color: #c9c9c9;
              background: #f1f1f1;
              opacity: 0.72;
            }}
            .crop-tile img {{
              display: block;
              width: 100%;
              height: 118px;
              object-fit: contain;
              background: linear-gradient(180deg, #eef4f8 0%, #f8fafc 100%);
            }}
            .crop-tile figcaption {{
              padding: 8px;
              font-size: 12px;
              line-height: 1.35;
              color: #3e5563;
              word-break: break-word;
            }}
            .crop-empty {{
              margin: 0;
              padding: 10px 12px;
              border: 1px dashed #c8d6e4;
              border-radius: 12px;
              color: #697d8a;
              background: #fbfdff;
              font-size: 13px;
            }}
            .empty {{
              background: #ffffff;
              border: 1px dashed #c8d6e4;
              border-radius: 16px;
              padding: 30px;
              color: #586b78;
            }}
          </style>
        </head>
        <body>
          <div class="shell">
            <section class="hero">
              <h1>Bag {int(bag)} Step Contact Sheet</h1>
              <p><strong>Set:</strong> {escape(str(set_num))}</p>
              <p><strong>Bag page range:</strong> {int(start_page)}-{int(end_page)}</p>
              <p><strong>Visible step range:</strong> {escape("unknown" if visible_step_min is None or visible_step_max is None else f"{visible_step_min}-{visible_step_max}")}</p>
              <p><strong>Pages shown:</strong> {len(page_cards)}</p>
              <p><strong>Include minifig override:</strong> {"on" if bool(int(include_minifig or 0)) else "off"}</p>
            </section>
            <section class="grid">
              {cards_block}
            </section>
          </div>
        </body>
        </html>
        """
    )


@router.get("/debug/inventory-overlay", response_class=HTMLResponse)
def debug_inventory_overlay(
    set_num: str = Query(...),
    page: int = Query(..., ge=1),
):
    path = debug_service.resolve_page_image_path(set_num, page)
    if path is None:
        raise HTTPException(status_code=404, detail="Page image not found")

    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not load page image")

    scan_payload = inventory_scan_service.scan_instruction_inventory(
        set_num=str(set_num),
        start=int(page),
        end=int(page),
    )
    items = [
        item
        for item in (scan_payload.get("items", []) or [])
        if int(item.get("page", 0) or 0) == int(page)
    ]
    ocr_candidates = [
        item
        for item in (scan_payload.get("ocr_candidates", []) or [])
        if int(item.get("page", 0) or 0) == int(page)
    ]
    candidate_by_key = {
        (
            str(item.get("element_id", "")),
            int((item.get("position") or {}).get("x", 0) or 0),
            int((item.get("position") or {}).get("y", 0) or 0),
        ): item
        for item in ocr_candidates
    }

    for item in items:
        box = item.get("position") or {}
        x = int(box.get("x", 0) or 0)
        y = int(box.get("y", 0) or 0)
        w = int(box.get("w", 0) or 0)
        h = int(box.get("h", 0) or 0)
        if w > 0 and h > 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (40, 180, 40), 2)

        qty_box = item.get("qty_position") or {}
        qx = int(qty_box.get("x", 0) or 0)
        qy = int(qty_box.get("y", 0) or 0)
        qw = int(qty_box.get("w", 0) or 0)
        qh = int(qty_box.get("h", 0) or 0)
        if qw > 0 and qh > 0:
            cv2.rectangle(img, (qx, qy), (qx + qw, qy + qh), (255, 0, 0), 2)

        qty_label = item.get("qty")
        candidate = candidate_by_key.get(
            (
                str(item.get("element_id", "")),
                int(x),
                int(y),
            )
        )
        conf = None if candidate is None else candidate.get("confidence")
        conf_label = "" if conf is None or float(conf) < 0.0 else f" c{float(conf):.0f}"
        reason_label = ""
        if candidate is not None:
            reason_label = " accepted"
        label = (
            f"{item.get('element_id')} (x{int(qty_label)}){conf_label}{reason_label}"
            if qty_label is not None
            else f"{item.get('element_id')}{conf_label}{reason_label}"
        )
        label_x = x if w > 0 else qx
        label_base_y = y if h > 0 else qy
        label_y = label_base_y - 8 if label_base_y > 20 else label_base_y + max(h, qh, 16) + 16
        cv2.putText(
            img,
            label,
            (int(label_x), int(label_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            label,
            (int(label_x), int(label_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for candidate in ocr_candidates:
        if bool(candidate.get("accepted")):
            continue

        box = candidate.get("position") or {}
        x = int(box.get("x", 0) or 0)
        y = int(box.get("y", 0) or 0)
        w = int(box.get("w", 0) or 0)
        h = int(box.get("h", 0) or 0)
        if w > 0 and h > 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)

        conf = candidate.get("confidence")
        conf_label = "?"
        try:
            conf_label = f"{float(conf):.0f}"
        except Exception:
            pass
        reason = str(candidate.get("rejection_reason") or "rejected")
        label = f"{candidate.get('element_id')} c{conf_label} {reason}"
        label_y = y - 8 if y > 20 else y + h + 16
        cv2.putText(
            img,
            label,
            (int(x), int(label_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            label,
            (int(x), int(label_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 245, 220),
            1,
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode overlay image")

    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    rows_html = []
    for item in items:
        box = item.get("position") or {}
        qty = item.get("qty")
        rows_html.append(
            f"""
            <tr>
              <td>{int(item.get("page", 0) or 0)}</td>
              <td>{escape(str(item.get("element_id", "")))}</td>
              <td>{escape("" if qty is None else f"x{int(qty)}")}</td>
              <td>{int(box.get("x", 0) or 0)},{int(box.get("y", 0) or 0)},{int(box.get("w", 0) or 0)},{int(box.get("h", 0) or 0)}</td>
              <td>{escape(str(item.get("matched_catalog_part_num") or ""))}</td>
              <td>{escape(str(item.get("matched_catalog_color_id") or ""))}</td>
              <td>{escape(str(item.get("matched_catalog_qty") or ""))}</td>
            </tr>
            """
        )

    rows_block = (
        "\n".join(rows_html)
        if rows_html
        else "<tr><td colspan='7'>No inventory items detected on this page.</td></tr>"
    )
    candidate_rows_html = []
    for candidate in ocr_candidates:
        box = candidate.get("position") or {}
        nearby_qty_candidates = list(candidate.get("nearby_qty_candidates", []) or [])
        nearby_qty_label = ", ".join(
            str(item.get("text", ""))
            + (
                ""
                if item.get("qty") is None
                else f"={int(item.get('qty'))}"
            )
            for item in nearby_qty_candidates
        )
        conf = candidate.get("confidence")
        conf_label = ""
        try:
            conf_label = f"{float(conf):.1f}"
        except Exception:
            conf_label = ""
        candidate_rows_html.append(
            f"""
            <tr>
              <td>{int(candidate.get("page", 0) or 0)}</td>
              <td>{escape(str(candidate.get("element_id", "")))}</td>
              <td>{escape(conf_label)}</td>
              <td>{escape("yes" if candidate.get("accepted") else "no")}</td>
              <td>{escape(str(candidate.get("rejection_reason") or ""))}</td>
              <td>{escape(nearby_qty_label)}</td>
              <td>{int(box.get("x", 0) or 0)},{int(box.get("y", 0) or 0)},{int(box.get("w", 0) or 0)},{int(box.get("h", 0) or 0)}</td>
            </tr>
            """
        )
    candidate_rows_block = (
        "\n".join(candidate_rows_html)
        if candidate_rows_html
        else "<tr><td colspan='7'>No 6-7 digit OCR candidates found on this page.</td></tr>"
    )

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Inventory Overlay</title>
          <style>
            body {{
              margin: 0;
              padding: 20px;
              background: #eef3f8;
              color: #17301f;
              font-family: Arial, sans-serif;
            }}
            .shell {{
              max-width: 1500px;
              margin: 0 auto;
            }}
            .card {{
              background: #ffffff;
              border: 1px solid #d7e0e9;
              border-radius: 14px;
              padding: 16px 18px;
              box-shadow: 0 10px 28px rgba(20, 42, 58, 0.08);
              margin-bottom: 18px;
            }}
            h1 {{
              margin: 0 0 8px;
              font-size: 28px;
            }}
            p {{
              margin: 4px 0;
            }}
            .overlay-img {{
              width: 100%;
              height: auto;
              display: block;
              border-radius: 12px;
              background: #dce7ef;
            }}
            table {{
              width: 100%;
              border-collapse: collapse;
              font-size: 14px;
              background: #ffffff;
            }}
            th, td {{
              border: 1px solid #d7e0e9;
              padding: 8px 10px;
              text-align: left;
              vertical-align: top;
            }}
            th {{
              background: #f3f7fb;
            }}
            .legend {{
              display: flex;
              gap: 16px;
              flex-wrap: wrap;
            }}
            .legend span {{
              display: inline-flex;
              align-items: center;
              gap: 8px;
            }}
            .swatch {{
              width: 14px;
              height: 14px;
              border-radius: 3px;
              display: inline-block;
            }}
          </style>
        </head>
        <body>
          <div class="shell">
            <section class="card">
              <h1>Inventory Overlay</h1>
              <p><strong>Set:</strong> {escape(str(set_num))}</p>
              <p><strong>Page:</strong> {int(page)}</p>
              <p><strong>Detected items:</strong> {len(items)}</p>
              <p><strong>6-7 digit OCR candidates:</strong> {len(ocr_candidates)}</p>
              <p><strong>Catalog join available:</strong> {"yes" if scan_payload.get("catalog_join_available") else "no"}</p>
              <p><strong>JSON mapping available:</strong> {"yes" if scan_payload.get("catalog_json_fallback_available") else "no"}</p>
              <div class="legend">
                <span><i class="swatch" style="background:#28b428;"></i> element ID box</span>
                <span><i class="swatch" style="background:#0055ff;"></i> quantity box</span>
                <span><i class="swatch" style="background:#ffa500;"></i> rejected 6-7 digit OCR candidate</span>
              </div>
            </section>
            <section class="card">
              <img class="overlay-img" src="data:image/png;base64,{image_b64}" alt="Inventory overlay for page {int(page)}" />
            </section>
            <section class="card">
              <table>
                <thead>
                  <tr>
                    <th>Page</th>
                    <th>Element ID</th>
                    <th>Qty</th>
                    <th>Position</th>
                    <th>Catalog part_num</th>
                    <th>Catalog color_id</th>
                    <th>Catalog qty</th>
                  </tr>
                </thead>
                <tbody>
                  {rows_block}
                </tbody>
              </table>
            </section>
            <section class="card">
              <table>
                <thead>
                  <tr>
                    <th>Page</th>
                    <th>OCR element ID</th>
                    <th>Confidence</th>
                    <th>Accepted</th>
                    <th>Rejection reason</th>
                    <th>Nearby qty candidates</th>
                    <th>Position</th>
                  </tr>
                </thead>
                <tbody>
                  {candidate_rows_block}
                </tbody>
              </table>
            </section>
          </div>
        </body>
        </html>
        """
    )


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
                <span class="thumb-preview" aria-hidden="true">
                  <img class="thumb-preview-img" src="{image_url}" alt="" loading="lazy" />
                </span>
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
              overflow: visible;
            }}
            .tile {{
              background: var(--panel);
              border: 1px solid var(--border);
              border-radius: 16px;
              overflow: visible;
              box-shadow: var(--shadow);
            }}
            .thumb-wrap {{
              display: block;
              background: #dbe7db;
              aspect-ratio: 1 / 1.25;
              position: relative;
              overflow: visible;
            }}
            .thumb {{
              width: 100%;
              height: 100%;
              object-fit: cover;
              display: block;
            }}
            .thumb-preview {{
              position: fixed;
              inset: 24px;
              display: flex;
              align-items: center;
              justify-content: center;
              padding: 24px;
              background: rgba(23, 48, 31, 0.44);
              opacity: 0;
              visibility: hidden;
              pointer-events: none;
              z-index: 99999;
              transition:
                opacity 0.15s ease,
                visibility 0s linear 1s;
            }}
            .thumb-preview-img {{
              display: block;
              width: auto;
              height: auto;
              max-width: min(92vw, 1200px);
              max-height: 92vh;
              object-fit: contain;
              border-radius: 14px;
              box-shadow: 0 20px 80px rgba(10, 24, 15, 0.45);
              background: #ffffff;
            }}
            .thumb-wrap:hover .thumb-preview,
            .thumb-wrap:focus-visible .thumb-preview {{
              opacity: 1;
              visibility: visible;
              transition:
                opacity 0.15s ease 1s,
                visibility 0s;
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


@router.get("/debug/page-range-thumbs", response_class=HTMLResponse)
def debug_page_range_thumbs(
    set_num: str = Query(...),
    start: int = Query(..., ge=1),
    end: int = Query(..., ge=1),
):
    if int(end) < int(start):
        raise HTTPException(status_code=400, detail="end must be >= start")

    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        raise HTTPException(status_code=404, detail="no rendered pages found for set")

    rendered_pages = [
        int(page)
        for page in _list_rendered_pages(pages_dir)
        if int(start) <= int(page) <= int(end)
    ]

    cards: List[str] = []
    for page in rendered_pages:
        image_url = f"/debug/page-image?set_num={escape(str(set_num))}&page={int(page)}"
        analyze_url = (
            f"/api/analyze-page-direct?set_num={escape(str(set_num))}&page={int(page)}"
        )
        cards.append(
            f"""
            <article class="tile">
              <a class="thumb-wrap" href="{image_url}" target="_blank">
                <img class="thumb" src="{image_url}" alt="Page {int(page)}" loading="lazy" />
                <span class="thumb-preview" aria-hidden="true">
                  <img class="thumb-preview-img" src="{image_url}" alt="" loading="lazy" />
                </span>
              </a>
              <div class="tile-body">
                <h3>Page {int(page)}</h3>
                <p class="actions">
                  <a href="{image_url}" target="_blank">Open image</a>
                  <span class="sep">|</span>
                  <a href="{analyze_url}" target="_blank">Analyze JSON</a>
                </p>
              </div>
            </article>
            """
        )

    cards_block = (
        "\n".join(cards)
        if cards
        else "<div class='empty'>No rendered pages found in this range.</div>"
    )

    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Page Range Thumbnails</title>
          <style>
            :root {{
              --bg: #f4f6f3;
              --panel: #ffffff;
              --border: #d7ddd5;
              --text: #1d2d22;
              --muted: #64756a;
              --accent: #2f8f55;
              --shadow: 0 10px 28px rgba(25, 47, 33, 0.08);
            }}
            * {{ box-sizing: border-box; }}
            body {{
              margin: 0;
              padding: 24px;
              background: linear-gradient(180deg, #eef3ed 0%, var(--bg) 100%);
              color: var(--text);
              font-family: Arial, sans-serif;
            }}
            .shell {{
              max-width: 1380px;
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
              color: var(--accent);
              text-decoration: none;
            }}
            .grid {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
              gap: 16px;
              overflow: visible;
            }}
            .tile {{
              background: var(--panel);
              border: 1px solid var(--border);
              border-radius: 16px;
              overflow: visible;
              box-shadow: var(--shadow);
            }}
            .thumb-wrap {{
              display: block;
              background: #dce7dc;
              aspect-ratio: 1 / 1.25;
              position: relative;
              overflow: visible;
            }}
            .thumb {{
              display: block;
              width: 100%;
              height: 100%;
              object-fit: cover;
            }}
            .thumb-preview {{
              position: fixed;
              inset: 24px;
              display: flex;
              align-items: center;
              justify-content: center;
              padding: 24px;
              background: rgba(29, 45, 34, 0.44);
              opacity: 0;
              visibility: hidden;
              pointer-events: none;
              z-index: 99999;
              transition:
                opacity 0.15s ease,
                visibility 0s linear 1s;
            }}
            .thumb-preview-img {{
              display: block;
              width: auto;
              height: auto;
              max-width: min(92vw, 1200px);
              max-height: 92vh;
              object-fit: contain;
              border-radius: 14px;
              box-shadow: 0 20px 80px rgba(10, 24, 15, 0.45);
              background: #ffffff;
            }}
            .thumb-wrap:hover .thumb-preview,
            .thumb-wrap:focus-visible .thumb-preview {{
              opacity: 1;
              visibility: visible;
              transition:
                opacity 0.15s ease 1s,
                visibility 0s;
            }}
            .tile-body {{
              padding: 12px 14px 14px;
            }}
            .tile-body h3 {{
              margin: 0 0 8px;
              font-size: 19px;
            }}
            .actions {{
              margin: 0;
              font-size: 14px;
            }}
            .actions a {{
              color: var(--accent);
              text-decoration: none;
            }}
            .sep {{
              color: #93a095;
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
              <h1>Page Range Thumbnails</h1>
              <p><strong>Set:</strong> {escape(str(set_num))}</p>
              <p><strong>Range:</strong> pages {int(start)}-{int(end)}</p>
              <p><strong>Pages shown:</strong> {len(rendered_pages)}</p>
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
