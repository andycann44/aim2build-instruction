import copy
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractError
from fastapi import HTTPException

# Extracted from app.py.
# This module is detector-only. No routes, no HTML, no workflow.

BASE = ""
ANALYZE_PAGE_CACHE = {}
ANALYZE_PAGE_CACHE_LOCK = threading.Lock()
_OCR_STATE = threading.local()


def configure_pages_dir(pages_dir):
    global BASE
    BASE = Path(pages_dir).as_posix()
    with ANALYZE_PAGE_CACHE_LOCK:
        ANALYZE_PAGE_CACHE.clear()


def _clone_cached_value(value):
    return copy.deepcopy(value)


def _reset_ocr_error_flag():
    _OCR_STATE.ocr_error = False


def _mark_ocr_error():
    _OCR_STATE.ocr_error = True


def _get_ocr_error():
    return bool(getattr(_OCR_STATE, "ocr_error", False))


def _safe_tesseract_image_to_string(*args, **kwargs):
    try:
        return pytesseract.image_to_string(*args, **kwargs)
    except TesseractError:
        _mark_ocr_error()
        return ""


def _empty_ocr_data_dict():
    return {
        "text": [],
        "conf": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
        "block_num": [],
        "par_num": [],
    }


def _safe_tesseract_image_to_data(*args, **kwargs):
    try:
        return pytesseract.image_to_data(*args, **kwargs)
    except TesseractError:
        _mark_ocr_error()
        return _empty_ocr_data_dict()

def _path_cache_key(path: str):
    try:
        stat = os.stat(path)
        return (path, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (path, None, None)

def page_path(page: int) -> str:
    return os.path.join(BASE, f"page_{page:03d}.png")

def draw_box(img, box, color, t=3):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, t)

def crop(img, box):
    x, y, w, h = box
    return img[y : y + h, x : x + w]


def detect_step_grid(img):
    if img is None or img.size == 0:
        return 0, 0, False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_h, page_w = gray.shape[:2]
    page_area = float(max(1, page_h * page_w))
    step_grid_number_count = 0

    for contour in cnts:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w <= 0 or h <= 0:
            continue

        area_ratio = float(w * h) / page_area
        aspect_ratio = float(w) / float(max(h, 1))

        if area_ratio < 0.01 or area_ratio > 0.10:
            continue
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue

        step_grid_number_count += 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        np.array((99, 25, 245), dtype=np.uint8),
        np.array((105, 60, 255), dtype=np.uint8),
    )
    green_mask = cv2.morphologyEx(
        green_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    green_mask = cv2.dilate(green_mask, np.ones((3, 3), np.uint8), iterations=1)

    green_ratio = float(np.mean(green_mask > 0))
    green_box_count = 0

    if green_ratio >= 0.82:
        line_segments = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=120,
            minLineLength=max(120, int(gray.shape[0] * 0.60)),
            maxLineGap=10,
        )

        internal_vertical_xs = []
        page_h, page_w = gray.shape[:2]
        min_x = int(page_w * 0.12)
        max_x = int(page_w * 0.88)
        min_vertical_length = int(page_h * 0.75)

        if line_segments is not None:
            for segment in line_segments[:, 0, :]:
                x1, y1, x2, y2 = [int(v) for v in segment]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx > 10 or dy < min_vertical_length:
                    continue

                x_mid = int(round((x1 + x2) * 0.5))
                if x_mid < min_x or x_mid > max_x:
                    continue

                internal_vertical_xs.append(x_mid)

        if internal_vertical_xs:
            internal_vertical_xs.sort()
            vertical_bands = 1
            last_x = internal_vertical_xs[0]
            for x_mid in internal_vertical_xs[1:]:
                if abs(x_mid - last_x) > 20:
                    vertical_bands += 1
                    last_x = x_mid
            if vertical_bands >= 2:
                green_box_count = vertical_bands + 2

    multi_step_green_boxes = bool(
        green_box_count >= 4 or step_grid_number_count >= 4
    )
    return int(step_grid_number_count), int(green_box_count), multi_step_green_boxes

def _find_bag_intro_panel_once(
    img,
    *,
    search_w_ratio,
    search_h_ratio,
    min_area_ratio,
    max_area_ratio,
    max_x_ratio,
    max_y_ratio,
    min_roi_mean,
    min_mid_edge_density,
    position_penalty=0.0,
):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if float(np.mean(gray)) < 150:
        return None, None

    sx1, sy1 = 0, 0
    sx2, sy2 = int(w * search_w_ratio), int(h * search_h_ratio)
    search = gray[sy1:sy2, sx1:sx2]

    th = cv2.threshold(search, 228, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1e9
    page_area = float(w * h)

    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        x += sx1
        y += sy1

        area = bw * bh
        area_ratio = area / page_area
        wh_ratio = bw / float(max(bh, 1))

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if wh_ratio < 1.2 or wh_ratio > 5.5:
            continue
        if x > int(w * max_x_ratio):
            continue
        if y > int(h * max_y_ratio):
            continue

        roi = gray[y : y + bh, x : x + bw]
        if roi.size == 0:
            continue

        roi_mean = float(np.mean(roi))
        roi_std = float(np.std(roi))
        if roi_mean < min_roi_mean:
            continue

        left = roi[:, : max(1, int(bw * 0.30))]
        mid = roi[:, int(bw * 0.30) : max(int(bw * 0.48), int(bw * 0.30) + 1)]
        right = roi[:, max(int(bw * 0.48), int(bw * 0.30) + 1) :]

        if left.size == 0 or mid.size == 0 or right.size == 0:
            continue

        left_mean = float(np.mean(left))
        right_mean = float(np.mean(right))
        left_std = float(np.std(left))
        right_std = float(np.std(right))

        if left_mean > 242:
            continue
        if left_std < 6:
            continue
        if right_std < 6:
            continue

        mid_edges = cv2.Canny(mid, 30, 120)
        mid_edge_density = float(np.mean(mid_edges > 0))
        if mid_edge_density < min_mid_edge_density:
            continue

        contrast_gap = right_mean - left_mean

        score = 0.0
        score += left_std * 1.8
        score += right_std * 0.8
        score += mid_edge_density * 1600.0
        score += contrast_gap * 1.2
        score += roi_std * 0.6
        score -= abs(area_ratio - 0.18) * 300.0
        score -= (x / max(1.0, w)) * position_penalty
        score -= (y / max(1.0, h)) * (position_penalty * 0.5)

        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)

    if best is None:
        return None, None

    return best, float(best_score)

def find_bag_intro_panel_with_debug(img):
    strict_panel, strict_score = _find_bag_intro_panel_once(
        img,
        search_w_ratio=0.75,
        search_h_ratio=0.48,
        min_area_ratio=0.06,
        max_area_ratio=0.36,
        max_x_ratio=0.10,
        max_y_ratio=0.12,
        min_roi_mean=190.0,
        min_mid_edge_density=0.004,
        position_penalty=0.0,
    )
    if strict_panel is not None:
        return strict_panel, "strict_top_left", strict_score

    relaxed_panel, relaxed_score = _find_bag_intro_panel_once(
        img,
        search_w_ratio=0.90,
        search_h_ratio=0.60,
        min_area_ratio=0.05,
        max_area_ratio=0.36,
        max_x_ratio=0.38,
        max_y_ratio=0.34,
        min_roi_mean=185.0,
        min_mid_edge_density=0.0035,
        position_penalty=120.0,
    )
    if relaxed_panel is not None:
        return relaxed_panel, "relaxed_top_band", relaxed_score

    return None, "none", None

def _box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / float(union_area)

def _border_center_stats(gray_roi):
    if gray_roi is None or gray_roi.size == 0:
        return None

    h, w = gray_roi.shape[:2]
    border = np.concatenate(
        [
            gray_roi[: max(1, h // 10), :].ravel(),
            gray_roi[-max(1, h // 10) :, :].ravel(),
            gray_roi[:, : max(1, w // 10)].ravel(),
            gray_roi[:, -max(1, w // 10) :].ravel(),
        ]
    )
    center = gray_roi[
        max(1, int(h * 0.18)) : max(h - 1, int(h * 0.82)),
        max(1, int(w * 0.18)) : max(w - 1, int(w * 0.82)),
    ]
    if border.size == 0 or center.size == 0:
        return None

    return {
        "border_mean": float(np.mean(border)),
        "center_mean": float(np.mean(center)),
        "std": float(np.std(gray_roi)),
    }

def find_rectangle_candidates(panel_img):
    if panel_img is None or panel_img.size == 0:
        return []

    ph, pw = panel_img.shape[:2]
    gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY)

    sx1, sy1 = 0, 0
    sx2, sy2 = int(pw * 0.42), int(ph * 0.95)
    search = gray[sy1:sy2, sx1:sx2]
    if search.size == 0:
        return []

    dark_mask = cv2.threshold(search, 205, 255, cv2.THRESH_BINARY_INV)[1]
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    edge_mask = cv2.Canny(search, 35, 130)
    candidate_mask = cv2.bitwise_or(dark_mask, edge_mask)
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )

    cnts, _ = cv2.findContours(
        candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    area_total = float(search.shape[0] * search.shape[1])
    candidates = []

    for contour in cnts:
        x, y, bw, bh = cv2.boundingRect(contour)
        area_ratio = (bw * bh) / max(1.0, area_total)
        wh_ratio = bw / float(max(bh, 1))

        if area_ratio < 0.10 or area_ratio > 0.80:
            continue
        if wh_ratio < 0.75 or wh_ratio > 1.80:
            continue

        roi = search[y : y + bh, x : x + bw]
        if roi.size == 0:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) < 4 or len(approx) > 10:
            continue

        stats = _border_center_stats(roi)
        if stats is None:
            continue

        border_mean = stats["border_mean"]
        center_mean = stats["center_mean"]
        roi_std = stats["std"]
        if border_mean < 175:
            continue
        if center_mean > border_mean - 4:
            continue
        if roi_std < 12:
            continue

        edge_density = float(np.mean(cv2.Canny(roi, 30, 120) > 0))
        if edge_density < 0.01:
            continue

        score = 0.0
        score += max(0.0, border_mean - center_mean) / 45.0
        score += roi_std / 45.0
        score += edge_density * 5.0
        score += max(0.0, 1.2 - abs(wh_ratio - 1.1))
        score -= abs(area_ratio - 0.38) * 2.0
        score = max(0.0, min(1.0, score / 3.2))

        candidates.append(
            {
                "box": (x + sx1, y + sy1, bw, bh),
                "score": float(score),
            }
        )

    deduped = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if any(
            _box_iou(candidate["box"], existing["box"]) > 0.70 for existing in deduped
        ):
            continue
        deduped.append(candidate)

    return deduped[:8]

def _find_square_box_inside_candidate(candidate_img):
    if candidate_img is None or candidate_img.size == 0:
        return None, None

    ch, cw = candidate_img.shape[:2]
    gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)

    search_x1, search_y1 = 0, 0
    search_x2, search_y2 = int(cw * 0.88), ch
    search = gray[search_y1:search_y2, search_x1:search_x2]
    if search.size == 0:
        return None, None

    mask = cv2.threshold(search, 205, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_area = float(ch * cw)
    best = None
    best_score = None

    for contour in cnts:
        x, y, bw, bh = cv2.boundingRect(contour)
        area_ratio = (bw * bh) / max(1.0, candidate_area)
        wh_ratio = bw / float(max(bh, 1))

        if area_ratio < 0.18 or area_ratio > 0.90:
            continue
        if wh_ratio < 0.75 or wh_ratio > 1.35:
            continue
        if bh < ch * 0.45:
            continue
        if x > int(cw * 0.32):
            continue

        roi = search[y : y + bh, x : x + bw]
        if roi.size == 0:
            continue

        stats = _border_center_stats(roi)
        if stats is None:
            continue

        border_mean = stats["border_mean"]
        center_mean = stats["center_mean"]
        roi_std = stats["std"]
        if border_mean < 170:
            continue
        if center_mean > border_mean - 4:
            continue
        if roi_std < 12:
            continue

        edge_density = float(np.mean(cv2.Canny(roi, 30, 120) > 0))
        if edge_density < 0.01:
            continue

        score = 0.0
        score += max(0.0, border_mean - center_mean) / 45.0
        score += roi_std / 45.0
        score += edge_density * 5.0
        score += max(0.0, 1.2 - abs(wh_ratio - 1.0))
        score -= abs(area_ratio - 0.45) * 2.0
        score = max(0.0, min(1.0, score / 3.0))

        if best_score is None or score > best_score:
            best_score = score
            best = (x + search_x1, y + search_y1, bw, bh)

    return best, (float(best_score) if best_score is not None else None)

def _score_side_tail(candidate_img, square_box, side):
    if candidate_img is None or candidate_img.size == 0 or square_box is None:
        return 0.0

    gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape[:2]
    sx, sy, sw, sh = square_box

    y1 = max(0, sy + int(sh * 0.15))
    y2 = min(ch, sy + int(sh * 0.85))
    if y2 <= y1:
        return 0.0

    if side == "right":
        x1 = max(0, sx + sw - int(sw * 0.08))
        x2 = cw
        band_x1 = max(0, sx + sw - int(sw * 0.05))
        band_x2 = min(cw, sx + sw + max(2, int(sw * 0.18)))
    else:
        x1 = 0
        x2 = min(cw, sx + int(sw * 0.08))
        band_x1 = max(0, sx - max(2, int(sw * 0.18)))
        band_x2 = min(cw, sx + int(sw * 0.05))

    if x2 - x1 < max(8, int(cw * 0.12)):
        return 0.0

    tail_roi = gray[y1:y2, x1:x2]
    if tail_roi.size == 0:
        return 0.0

    tail_mask = cv2.threshold(tail_roi, 210, 255, cv2.THRESH_BINARY_INV)[1]
    tail_mask = cv2.morphologyEx(tail_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    tail_mask = cv2.morphologyEx(tail_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    edge_density = float(np.mean(cv2.Canny(tail_roi, 30, 120) > 0))
    dark_density = float(np.mean(tail_mask > 0))

    cnts, _ = cv2.findContours(tail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area_ratio = 0.0
    tail_area = float(tail_roi.shape[0] * tail_roi.shape[1])
    for contour in cnts:
        _, _, bw, bh = cv2.boundingRect(contour)
        largest_area_ratio = max(largest_area_ratio, (bw * bh) / max(1.0, tail_area))

    connector_band = gray[y1:y2, band_x1:band_x2]
    connector_dark_density = 0.0
    if connector_band.size != 0:
        connector_mask = cv2.threshold(connector_band, 210, 255, cv2.THRESH_BINARY_INV)[
            1
        ]
        connector_dark_density = float(np.mean(connector_mask > 0))

    score = 0.0
    score += edge_density * 6.0
    score += dark_density * 1.5
    score += largest_area_ratio * 2.0
    score += connector_dark_density * 2.5
    return max(0.0, min(1.0, score / 2.2))

def _find_side_tail(candidate_img, square_box):
    right_score = _score_side_tail(candidate_img, square_box, "right")
    left_score = _score_side_tail(candidate_img, square_box, "left")

    if right_score >= left_score and right_score >= 0.22:
        return "right", right_score
    if left_score > right_score and left_score >= 0.22:
        return "left", left_score
    return None, max(right_score, left_score)

def _find_structured_shell_candidate_with_debug(panel_img):
    rectangle_candidates = find_rectangle_candidates(panel_img)
    if not rectangle_candidates:
        return None, None, "shape_first"

    best_partial_score = rectangle_candidates[0]["score"]
    best_candidate = None

    for rectangle_candidate in rectangle_candidates:
        candidate_box = rectangle_candidate["box"]
        candidate_img = crop(panel_img, candidate_box)
        if candidate_img is None or candidate_img.size == 0:
            continue

        square_box, square_score = _find_square_box_inside_candidate(candidate_img)
        if square_box is None or square_score is None:
            continue

        tail_side, tail_score = _find_side_tail(candidate_img, square_box)
        if tail_side is None:
            continue

        square_img = crop(candidate_img, square_box)
        bag_rel, bag_score = find_grey_bag_blob(square_img)
        if bag_rel is None:
            bag_rel, bag_score = find_grey_bag_blob(candidate_img)
        if bag_rel is None:
            continue

        bag_quality = max(0.0, min(1.0, (bag_score + 40.0) / 80.0))
        combined_score = (
            (rectangle_candidate["score"] * 0.35)
            + (square_score * 0.25)
            + (tail_score * 0.20)
            + (bag_quality * 0.20)
        )
        combined_score = max(0.0, min(0.99, combined_score))

        candidate = {
            "box": candidate_box,
            "score": float(combined_score),
            "method": f"shape_first_{tail_side}",
        }
        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate

    if best_candidate is None:
        return None, float(best_partial_score), "shape_first"

    return best_candidate["box"], best_candidate["score"], best_candidate["method"]

def find_square_bag_box_with_debug(panel_img):
    if panel_img is None or panel_img.size == 0:
        return None, None, "shape_first"

    return _find_structured_shell_candidate_with_debug(panel_img)

def find_grey_bag_blob(square_img):
    if square_img is None or square_img.size == 0:
        return None, 0.0

    h, w = square_img.shape[:2]
    gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)

    pad = max(2, int(min(w, h) * 0.06))
    inner = gray[pad : h - pad, pad : w - pad]
    if inner.size == 0:
        return None, 0.0

    mask = cv2.inRange(inner, 150, 235)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1e9
    area_total = float(inner.shape[0] * inner.shape[1])

    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area_ratio = (bw * bh) / max(1.0, area_total)
        wh_ratio = bw / float(max(bh, 1))

        if area_ratio < 0.18 or area_ratio > 0.80:
            continue
        if wh_ratio < 0.50 or wh_ratio > 1.30:
            continue

        roi = inner[y : y + bh, x : x + bw]
        if roi.size == 0:
            continue

        mean = float(np.mean(roi))
        std = float(np.std(roi))

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        dx = abs(cx - inner.shape[1] / 2.0) / max(1.0, inner.shape[1] / 2.0)
        dy = abs(cy - inner.shape[0] / 2.0) / max(1.0, inner.shape[0] / 2.0)
        center_penalty = (dx + dy) * 40.0

        score = 0.0
        score += std * 1.6
        score -= abs(mean - 195.0) * 0.8
        score -= abs(area_ratio - 0.42) * 180.0
        score -= abs(wh_ratio - 0.85) * 80.0
        score -= center_penalty

        if score > best_score:
            best_score = score
            best = (x + pad, y + pad, bw, bh)

    if best is None:
        return None, 0.0

    return best, float(best_score)

def _append_number_candidate(candidates, seen, box, img_w, img_h):
    x, y, w, h = [int(round(v)) for v in box]
    x = max(0, x)
    y = max(0, y)
    w = min(max(0, w), img_w - x)
    h = min(max(0, h), img_h - y)

    if w < 8 or h < 12:
        return

    key = (x, y, w, h)
    if key in seen:
        return

    seen.add(key)
    candidates.append(key)

def _extract_digit_like_boxes(
    gray_img,
    *,
    min_area_ratio=0.01,
    max_area_ratio=0.35,
    min_height_ratio=0.28,
    min_width_ratio=0.05,
):
    if gray_img is None or gray_img.size == 0:
        return []

    h, w = gray_img.shape[:2]
    roi_area = float(h * w)
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    masks = [
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY_INV)[1],
    ]

    boxes = []
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            area_ratio = (bw * bh) / max(1.0, roi_area)

            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            if bh < h * min_height_ratio:
                continue
            if bw < w * min_width_ratio:
                continue

            boxes.append((x, y, bw, bh))

    deduped = []
    for box in sorted(boxes, key=lambda item: (item[0], item[1], -(item[2] * item[3]))):
        if any(
            abs(box[0] - other[0]) <= 4
            and abs(box[1] - other[1]) <= 4
            and abs(box[2] - other[2]) <= 4
            and abs(box[3] - other[3]) <= 4
            for other in deduped
        ):
            continue
        deduped.append(box)

    return deduped[:8]

def _group_adjacent_digit_boxes(boxes, max_gap):
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda item: (item[0], item[1]))
    groups = []
    current_group = [ordered[0]]

    for box in ordered[1:]:
        prev = current_group[-1]
        gap = box[0] - (prev[0] + prev[2])
        overlap = min(prev[1] + prev[3], box[1] + box[3]) - max(prev[1], box[1])
        overlap_needed = min(prev[3], box[3]) * 0.30

        if gap <= max_gap and overlap > overlap_needed:
            current_group.append(box)
        else:
            groups.append(current_group)
            current_group = [box]

    groups.append(current_group)
    return groups

def _group_union_box(group):
    gx1 = min(item[0] for item in group)
    gy1 = min(item[1] for item in group)
    gx2 = max(item[0] + item[2] for item in group)
    gy2 = max(item[1] + item[3] for item in group)
    return gx1, gy1, gx2 - gx1, gy2 - gy1

def find_number_crop(bag_img):
    if bag_img is None or bag_img.size == 0:
        return []

    h, w = bag_img.shape[:2]
    candidates = []
    seen = set()

    gray = cv2.cvtColor(bag_img, cv2.COLOR_BGR2GRAY)

    roi_x1 = int(w * 0.15)
    roi_x2 = int(w * 0.85)
    roi_y1 = int(h * 0.05)
    roi_y2 = int(h * 0.72)
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

    contour_boxes = []
    has_multi_digit_group = False
    if roi.size != 0:
        roi_boxes = _extract_digit_like_boxes(
            roi,
            min_area_ratio=0.01,
            max_area_ratio=0.35,
            min_height_ratio=0.28,
            min_width_ratio=0.05,
        )
        contour_boxes = [(roi_x1 + x, roi_y1 + y, bw, bh) for x, y, bw, bh in roi_boxes]

        groups = _group_adjacent_digit_boxes(
            contour_boxes,
            max_gap=max(10, int(roi.shape[1] * 0.10)),
        )

        for group in groups:
            if len(group) < 2:
                continue

            has_multi_digit_group = True
            gx1, gy1, gw, gh = _group_union_box(group)

            tight_pad_x = max(8, int(gw * 0.18))
            tight_pad_y = max(6, int(gh * 0.16))
            _append_number_candidate(
                candidates,
                seen,
                (
                    gx1 - tight_pad_x,
                    gy1 - tight_pad_y,
                    gw + (2 * tight_pad_x),
                    gh + (2 * tight_pad_y),
                ),
                w,
                h,
            )

            generous_pad_x = max(10, int(gw * 0.32))
            generous_pad_y = max(8, int(gh * 0.24))
            _append_number_candidate(
                candidates,
                seen,
                (
                    gx1 - generous_pad_x,
                    gy1 - generous_pad_y,
                    gw + (2 * generous_pad_x),
                    gh + (2 * generous_pad_y),
                ),
                w,
                h,
            )

        if not has_multi_digit_group:
            for x, y, bw, bh in contour_boxes:
                pad_x = max(10, int(bw * 1.00))
                pad_y = max(8, int(bh * 0.35))
                _append_number_candidate(
                    candidates,
                    seen,
                    (x - pad_x, y - pad_y, bw + (2 * pad_x), bh + (2 * pad_y)),
                    w,
                    h,
                )

                generous_pad_x = max(12, int(bw * 1.20))
                generous_pad_y = max(10, int(bh * 0.45))
                _append_number_candidate(
                    candidates,
                    seen,
                    (
                        x - generous_pad_x,
                        y - generous_pad_y,
                        bw + (2 * generous_pad_x),
                        bh + (2 * generous_pad_y),
                    ),
                    w,
                    h,
                )

    if has_multi_digit_group:
        fallback_boxes = [
            (int(w * 0.24), int(h * 0.16), int(w * 0.54), int(h * 0.56)),
            (int(w * 0.20), int(h * 0.12), int(w * 0.60), int(h * 0.62)),
        ]
    else:
        fallback_boxes = [
            (int(w * 0.28), int(h * 0.16), int(w * 0.40), int(h * 0.54)),
            (int(w * 0.24), int(h * 0.12), int(w * 0.46), int(h * 0.60)),
        ]

    for box in fallback_boxes:
        _append_number_candidate(candidates, seen, box, w, h)

    return candidates

def _ocr_hits_from_images(images, configs):
    hits = []
    for img_try in images:
        for cfg in configs:
            raw = _safe_tesseract_image_to_string(img_try, config=cfg).strip()
            digits = "".join(ch for ch in raw if ch.isdigit())
            if len(digits) == 0 or len(digits) > 2:
                continue
            val = int(digits)
            if 1 <= val <= 99:
                hits.append((val, raw))
    return hits

def read_bag_number(number_img):
    if number_img is None or number_img.size == 0:
        return None, ""

    gray = cv2.cvtColor(number_img, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (3, 3), 0)

    h, w = up.shape
    cx1 = int(w * 0.25)
    cx2 = int(w * 0.75)
    cy1 = int(h * 0.20)
    cy2 = int(h * 0.80)
    up = up[cy1:cy2, cx1:cx2]

    up = cv2.GaussianBlur(up, (3, 3), 0)

    full_variants = []

    v1 = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    full_variants.append(v1)

    v2 = cv2.threshold(up, 170, 255, cv2.THRESH_BINARY_INV)[1]
    full_variants.append(v2)

    v3 = cv2.GaussianBlur(up, (3, 3), 0)
    v3 = cv2.threshold(v3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    full_variants.append(v3)

    cleaned = []
    for th in full_variants:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0
        H, W = th.shape[:2]

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < (W * H) * 0.008:
                continue
            if h < H * 0.35:
                continue
            if w < W * 0.10:
                continue
            if area > best_area:
                best_area = area
                best = (x, y, w, h)

        if best is None:
            cleaned.append(th)
            continue

        x, y, w, h = best
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        roi = th[y1:y2, x1:x2]
        canvas = np.zeros(
            (max(roi.shape[0] + 20, 120), max(roi.shape[1] + 20, 120)), dtype=np.uint8
        )
        oy = (canvas.shape[0] - roi.shape[0]) // 2
        ox = (canvas.shape[1] - roi.shape[1]) // 2
        canvas[oy : oy + roi.shape[0], ox : ox + roi.shape[1]] = roi
        cleaned.append(canvas)

    configs = [
        "--psm 10 -c tessedit_char_whitelist=0123456789",
        "--psm 8 -c tessedit_char_whitelist=0123456789",
        "--psm 13 -c tessedit_char_whitelist=0123456789",
    ]

    hits = _ocr_hits_from_images(cleaned, configs)
    if not hits:
        hits = _ocr_hits_from_images(full_variants, configs)

    if not hits:
        # fallback: try OCR on raw upscaled image directly
        raw = _safe_tesseract_image_to_string(
            up, config="--psm 8 -c tessedit_char_whitelist=0123456789"
        ).strip()

        digits = "".join(ch for ch in raw if ch.isdigit())

        if digits:
            return int(digits), raw

        return None, ""

    counts = {}
    raw_map = {}
    for val, raw in hits:
        counts[val] = counts.get(val, 0) + 1
        raw_map[val] = raw

    best_val = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    # --- sanity correction for 2 vs 9 ---
    h, w = gray.shape[:2]
    bottom_half = gray[int(h * 0.55) :, :]

    bottom_dark_ratio = float((bottom_half < 120).sum()) / max(1, bottom_half.size)

    # If OCR says 2 but bottom has strong dark pixels, it's likely a 9.
    if best_val == 2 and bottom_dark_ratio > 0.12:
        return 9, raw_map[best_val]

    return best_val, raw_map[best_val]

def _estimate_number_structure(gray_img):
    if gray_img is None or gray_img.size == 0:
        return {
            "has_adjacent_pair": False,
            "pair_aspect_ratio": 0.0,
            "crop_aspect_ratio": 0.0,
        }

    h, w = gray_img.shape[:2]
    roi_x1 = int(w * 0.10)
    roi_x2 = int(w * 0.90)
    roi_y1 = int(h * 0.08)
    roi_y2 = int(h * 0.92)
    roi = gray_img[roi_y1:roi_y2, roi_x1:roi_x2]

    boxes = _extract_digit_like_boxes(
        roi,
        min_area_ratio=0.006,
        max_area_ratio=0.45,
        min_height_ratio=0.24,
        min_width_ratio=0.03,
    )
    groups = (
        _group_adjacent_digit_boxes(
            boxes, max_gap=max(6, int(max(1, roi.shape[1]) * 0.12))
        )
        if boxes
        else []
    )
    pair_groups = [group for group in groups if len(group) >= 2]

    pair_aspect_ratio = 0.0
    if pair_groups:
        best_group = max(
            pair_groups, key=lambda group: (_group_union_box(group)[2], len(group))
        )
        _, _, gw, gh = _group_union_box(best_group)
        pair_aspect_ratio = gw / float(max(gh, 1))

    return {
        "has_adjacent_pair": bool(pair_groups),
        "pair_aspect_ratio": pair_aspect_ratio,
        "crop_aspect_ratio": w / float(max(h, 1)),
    }

def read_bag_number_with_score(number_img):
    if number_img is None or number_img.size == 0:
        return None, "", -1.0

    votes = {}

    def add_vote(val, raw, conf_score):
        if val is None or val < 1 or val > 24:
            return

        entry = votes.setdefault(
            val,
            {
                "count": 0,
                "best_conf": -1.0,
                "raw": raw or str(val),
            },
        )
        entry["count"] += 1
        if conf_score > entry["best_conf"]:
            entry["best_conf"] = conf_score
            entry["raw"] = raw or str(val)

    base_val, base_raw = read_bag_number(number_img)
    if base_val is not None and 1 <= base_val <= 24:
        add_vote(base_val, base_raw, 25.0)

    gray = cv2.cvtColor(number_img, cv2.COLOR_BGR2GRAY)
    structure = _estimate_number_structure(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    variants = [
        blurred,
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]
    configs = [
        "--psm 13 -c tessedit_char_whitelist=0123456789",
        "--psm 10 -c tessedit_char_whitelist=0123456789",
        "--psm 8 -c tessedit_char_whitelist=0123456789",
        "--psm 7 -c tessedit_char_whitelist=0123456789",
    ]

    for variant in variants:
        up = cv2.resize(variant, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        for cfg in configs:
            data = _safe_tesseract_image_to_data(
                up,
                config=cfg,
                output_type=pytesseract.Output.DICT,
            )

            for raw_text, conf_text in zip(data.get("text", []), data.get("conf", [])):
                digits = "".join(ch for ch in raw_text if ch.isdigit())
                if len(digits) == 0 or len(digits) > 2:
                    continue

                try:
                    val = int(digits)
                except ValueError:
                    continue

                try:
                    conf_score = float(conf_text)
                except (TypeError, ValueError):
                    conf_score = -1.0

                if conf_score < 0.0:
                    continue

                add_vote(val, digits, conf_score)

    if not votes:
        return None, "", -1.0

    def total_vote_score(val, meta):
        digit_len = len(meta["raw"])
        total = (
            meta["best_conf"] + ((meta["count"] - 1) * 4.0) + ((digit_len - 1) * 40.0)
        )

        if structure["has_adjacent_pair"]:
            if digit_len >= 2:
                total += 48.0
                if structure["pair_aspect_ratio"] >= 0.85:
                    total += 18.0
            else:
                total -= 34.0
                if structure["crop_aspect_ratio"] >= 0.65:
                    total -= 8.0
        elif digit_len >= 2 and structure["crop_aspect_ratio"] < 0.58:
            total -= 12.0

        return total

    best_val, best_meta = max(
        votes.items(),
        key=lambda item: (
            total_vote_score(item[0], item[1]),
            item[1]["best_conf"],
            len(item[1]["raw"]),
            item[1]["count"],
        ),
    )
    best_score = total_vote_score(best_val, best_meta)
    return best_val, best_meta["raw"], float(best_score)

def compute_confidence(
    panel_found,
    shell_found,
    grey_bag_found,
    number_box_found,
    bag_number,
    shell_score,
    grey_bag_score,
):

    score = 0.0

    # Stage 1: panel
    if panel_found:
        score += 0.15

    # Stage 2: shell
    if shell_found:
        score += 0.20

        if shell_score is not None:
            score += min(shell_score, 1.0) * 0.15

    # Stage 3: grey bag
    if grey_bag_found:
        score += 0.25

        if grey_bag_score is not None:
            norm = max(0.0, min(1.0, (grey_bag_score + 40) / 80))
            score += norm * 0.15

    # Stage 4: number crop
    if number_box_found:
        score += 0.10

    # Stage 5: OCR validity
    if bag_number is not None:
        score += 0.40
    else:
        score -= 0.30

    return round(min(score, 1.0), 3)

def _box_area_ratio(box, img_shape):
    if box is None or img_shape is None:
        return 0.0

    h, w = img_shape[:2]
    if h <= 0 or w <= 0:
        return 0.0

    _, _, bw, bh = box
    return float(max(0, bw) * max(0, bh)) / float(w * h)

def _bag_region_quality_flags(img_shape, shell_box, grey_bag_box):
    shell_area_ratio = _box_area_ratio(shell_box, img_shape)
    grey_bag_area_ratio = _box_area_ratio(grey_bag_box, img_shape)

    tiny_bag_reference = False
    weak_bag_reference = False
    bag_region_quality = 1.0

    if grey_bag_box is not None:
        tiny_bag_reference = (
            grey_bag_area_ratio < 0.006
            or (shell_box is not None and shell_area_ratio < 0.025)
        )
        weak_bag_reference = (
            not tiny_bag_reference
            and (
                grey_bag_area_ratio < 0.030
                or (shell_box is not None and shell_area_ratio < 0.080)
            )
        )

        if tiny_bag_reference:
            bag_region_quality = 0.25
        elif weak_bag_reference:
            bag_region_quality = 0.55

    return {
        "shell_area_ratio": round(shell_area_ratio, 5),
        "grey_bag_area_ratio": round(grey_bag_area_ratio, 5),
        "tiny_bag_reference": bool(tiny_bag_reference),
        "weak_bag_reference": bool(weak_bag_reference),
        "bag_region_quality": round(float(bag_region_quality), 3),
    }

def _safe_ocr_text(img, psm=6):
    if img is None or img.size == 0:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    txt = _safe_tesseract_image_to_string(up, config=f"--psm {psm}")
    return txt or ""

def _region_contains_xn_marker(region_img):
    txt = _safe_ocr_text(region_img, psm=6).lower().replace(" ", "")
    for i in range(1, 10):
        if f"x{i}" in txt:
            return True
    return False

def _find_intro_regions(page_img):
    if page_img is None or page_img.size == 0:
        return []

    page_h, page_w = page_img.shape[:2]
    page_area = float(max(1, page_h * page_w))

    gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)

    # Search upper 75% of page. Some real bag starts sit lower than the top band.
    search_h = int(round(page_h * 0.75))
    search = gray[:search_h, :]

    # Bright cards on gray page background.
    mask = cv2.threshold(search, 190, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        if w <= 0 or h <= 0:
            continue

        area_ratio = (w * h) / page_area
        if area_ratio < 0.03:
            continue

        # Reject tiny thin strips and tiny boxes.
        if w < page_w * 0.12 or h < page_h * 0.08:
            continue

        aspect_ratio = w / float(max(h, 1))
        if aspect_ratio < 1.2 or aspect_ratio > 3.8:
            continue

        region_img = page_img[y : y + h, x : x + w]
        if region_img is None or region_img.size == 0:
            continue

        # Simplicity: intro cards tend to be cleaner / lower edge density.
        region_gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(region_gray, 60, 140)
        edge_density = float(np.count_nonzero(edges)) / float(max(1, edges.size))

        # Reject obvious quantity/part callout boxes like x1, x2, x6.
        if _region_contains_xn_marker(region_img):
            region_type = "part_callout"
            type_penalty = 0.30
        else:
            region_type = "intro_candidate"
            type_penalty = 0.0

        # Soft preferences:
        # - bigger is better
        # - higher on page is better, but not absolute
        # - left/mid-left cards are more likely than right-side cards
        # - cleaner region is better
        top_bias = max(0.0, 1.0 - (y / float(max(1, page_h))))
        left_bias = max(0.0, 1.0 - (x / float(max(1, page_w))))
        simplicity = max(0.0, 1.0 - min(1.0, edge_density / 0.12))
        shape_bias = max(0.0, 1.0 - (abs(aspect_ratio - 2.0) / 1.4))
        right_penalty = max(0.0, ((x + w) / float(max(1, page_w))) - 0.72)

        score = (
            (area_ratio * 100.0)
            + (top_bias * 16.0)
            + (left_bias * 20.0)
            + (simplicity * 16.0)
            + (shape_bias * 8.0)
            - (right_penalty * 24.0)
            - (type_penalty * 100.0)
        )

        candidates.append(
            {
                "box": (int(x), int(y), int(w), int(h)),
                "score": float(score),
                "type": region_type,
                "area_ratio": float(area_ratio),
                "edge_density": float(edge_density),
                "top_bias": float(top_bias),
                "left_bias": float(left_bias),
                "shape_bias": float(shape_bias),
            }
        )

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    return candidates[:8]

def _find_best_ocr_number_in_region(
    page_img,
    region_box,
    *,
    single_digit_min_score,
    reject_step_strips=True,
):
    region_img = crop(page_img, region_box)
    if region_img is None or region_img.size == 0:
        return None

    best_candidate = None
    single_digit_candidates = []

    for number_rel in find_number_crop(region_img):
        px, py, _, _ = region_box
        nx, ny, nw, nh = number_rel
        region_w = float(max(1, region_box[2]))
        region_h = float(max(1, region_box[3]))
        if nw > (region_w * 0.35) or nh > (region_h * 0.35):
            continue
        number_abs_try = (px + nx, py + ny, nw, nh)
        number_img = crop(page_img, number_abs_try)
        if number_img is None or number_img.size == 0:
            continue

        if _region_contains_xn_marker(number_img):
            continue

        val, raw, ocr_score = read_bag_number_with_score(number_img)
        if val is None:
            continue

        if len(str(val)) == 1 and float(ocr_score) < float(single_digit_min_score):
            continue

        candidate = {
            "score": float(ocr_score),
            "number": val,
            "raw": raw,
            "box": number_abs_try,
        }

        if len(str(val)) == 1:
            single_digit_candidates.append(candidate)

        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate

    if reject_step_strips and len(single_digit_candidates) >= 3:
        ys = [
            item["box"][1] + (item["box"][3] * 0.5) for item in single_digit_candidates
        ]
        xs = [
            item["box"][0] + (item["box"][2] * 0.5) for item in single_digit_candidates
        ]
        region_w = float(max(1, region_box[2]))
        region_h = float(max(1, region_box[3]))
        is_horizontal_strip = (max(ys) - min(ys)) <= (region_h * 0.22)
        is_vertical_strip = (max(xs) - min(xs)) <= (region_w * 0.18)
        if is_horizontal_strip or is_vertical_strip:
            return None

    return best_candidate

def _run_panel_bag_number_fallback(img, panel, panel_source, intro_region_candidates):
    if panel is None:
        return None, None

    candidate_regions = []
    if panel_source == "intro_region" and intro_region_candidates:
        candidate_regions.extend(
            [
                tuple(item["box"])
                for item in intro_region_candidates[:4]
                if item.get("type") != "part_callout"
            ]
        )
    else:
        candidate_regions.append(tuple(panel))

    seen_regions = set()
    best_candidate = None
    best_region_box = None

    for region_box in candidate_regions:
        if region_box in seen_regions:
            continue
        seen_regions.add(region_box)

        region_candidate = _find_best_ocr_number_in_region(
            img,
            region_box,
            single_digit_min_score=60.0,
            reject_step_strips=True,
        )
        if (
            region_candidate is not None
            and (
                best_candidate is None
                or region_candidate["score"] > best_candidate["score"]
            )
        ):
            best_candidate = region_candidate
            best_region_box = region_box

        rx, ry, rw, rh = region_box
        focus_boxes = [
            (
                rx + int(rw * 0.22),
                ry + int(rh * 0.18),
                int(rw * 0.38),
                int(rh * 0.56),
            ),
            (
                rx + int(rw * 0.26),
                ry + int(rh * 0.22),
                int(rw * 0.28),
                int(rh * 0.48),
            ),
            (
                rx + int(rw * 0.30),
                ry + int(rh * 0.28),
                int(rw * 0.20),
                int(rh * 0.42),
            ),
        ]

        strict_direct_focus_boxes = []
        if panel_source == "strict_top_left":
            strict_direct_focus_boxes.append(
                (
                    rx + int(rw * 0.04),
                    ry + int(rh * 0.166),
                    int(rw * 0.20),
                    int(rh * 0.42),
                )
            )
            strict_direct_focus_boxes.append(
                (
                    rx + int(rw * 0.12),
                    ry + int(rh * 0.18),
                    int(rw * 0.17),
                    int(rh * 0.34),
                )
            )
            focus_boxes.extend(strict_direct_focus_boxes)

        # Some valid bag-start panels place the leaflet/number block farther right
        # than the older anchor examples. Add one extra right-biased probe for the
        # strict panel path so shifted bag numbers still get a tight OCR crop.
        if panel_source == "strict_top_left":
            strict_direct_focus_boxes.append(
                (
                    rx + int(rw * 0.40),
                    ry + int(rh * 0.28),
                    int(rw * 0.16),
                    int(rh * 0.34),
                )
            )
            focus_boxes.append(strict_direct_focus_boxes[-1])

        for focus_box in focus_boxes:
            fx, fy, fw, fh = focus_box
            if fw <= 0 or fh <= 0:
                continue

            focus_candidate = _find_best_ocr_number_in_region(
                img,
                focus_box,
                single_digit_min_score=48.0,
                reject_step_strips=True,
            )
            if (
                focus_candidate is None
                and panel_source == "strict_top_left"
                and focus_box in strict_direct_focus_boxes
            ):
                val, raw, score = read_bag_number_with_score(crop(img, focus_box))
                if val is not None and float(score) >= 48.0:
                    focus_candidate = {
                        "score": float(score),
                        "number": val,
                        "raw": raw,
                        "box": focus_box,
                    }
            if (
                focus_candidate is not None
                and (
                    best_candidate is None
                    or focus_candidate["score"] > best_candidate["score"]
                )
            ):
                best_candidate = focus_candidate
                best_region_box = region_box

    return best_candidate, best_region_box

def _is_effectively_full_page_region(region_box, page_shape):
    if region_box is None or page_shape is None:
        return False
    page_h, page_w = page_shape[:2]
    x, y, w, h = region_box
    if page_w <= 0 or page_h <= 0:
        return False
    return (
        x <= int(page_w * 0.02)
        and y <= int(page_h * 0.02)
        and w >= int(page_w * 0.95)
        and h >= int(page_h * 0.95)
    )

def _cluster_band_count(values, gap_threshold):
    if not values:
        return 0

    ordered = sorted(values)
    bands = 1
    prev = ordered[0]

    for value in ordered[1:]:
        if abs(value - prev) > gap_threshold:
            bands += 1
        prev = value

    return bands

def _collect_page_bag_number_candidates(page_img, max_candidates=32):
    if page_img is None or page_img.size == 0:
        return []

    page_h, page_w = page_img.shape[:2]
    candidates = []

    def add_candidate(box, number, raw, score):
        duplicate_idx = None
        for idx, existing in enumerate(candidates):
            if _box_iou(existing["box"], box) >= 0.35:
                duplicate_idx = idx
                break

        candidate = {
            "box": box,
            "number": int(number),
            "raw": raw,
            "score": float(score),
        }

        if duplicate_idx is None:
            candidates.append(candidate)
            return

        if float(score) > float(candidates[duplicate_idx]["score"]):
            candidates[duplicate_idx] = candidate

    for number_rel in find_number_crop(page_img)[:18]:
        x, y, w, h = number_rel

        if w > (page_w * 0.18) or h > (page_h * 0.18):
            continue
        if w < max(18, int(page_w * 0.012)) or h < max(24, int(page_h * 0.025)):
            continue

        number_img = crop(page_img, number_rel)
        if number_img is None or number_img.size == 0:
            continue
        if _region_contains_xn_marker(number_img):
            continue

        val, raw, ocr_score = read_bag_number_with_score(number_img)
        if val is None or val < 1 or val > 24:
            continue
        if len(str(val)) > 2:
            continue
        if float(ocr_score) < 65.0:
            continue

        add_candidate(number_rel, val, raw, ocr_score)

    gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1600.0 / float(max(page_w, 1)))
    gray_for_ocr = gray
    if scale != 1.0:
        gray_for_ocr = cv2.resize(
            gray,
            (max(1, int(round(page_w * scale))), max(1, int(round(page_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    ocr_data = _safe_tesseract_image_to_data(
        gray_for_ocr,
        config="--psm 11 -c tessedit_char_whitelist=0123456789",
        output_type=pytesseract.Output.DICT,
    )

    for i, raw_text in enumerate(ocr_data.get("text", [])):
        token = (raw_text or "").strip()
        if not token or not token.isdigit():
            continue
        if len(token) > 2:
            continue

        try:
            val = int(token)
            conf = float(ocr_data["conf"][i])
        except (TypeError, ValueError, KeyError):
            continue

        if val < 1 or val > 24 or conf < 45.0:
            continue

        x = int(round(int(ocr_data["left"][i]) / max(scale, 1e-6)))
        y = int(round(int(ocr_data["top"][i]) / max(scale, 1e-6)))
        w = int(round(int(ocr_data["width"][i]) / max(scale, 1e-6)))
        h = int(round(int(ocr_data["height"][i]) / max(scale, 1e-6)))
        if w <= 0 or h <= 0:
            continue
        if w > (page_w * 0.18) or h > (page_h * 0.18):
            continue
        if w < max(18, int(page_w * 0.012)) or h < max(24, int(page_h * 0.025)):
            continue

        box = (x, y, w, h)
        number_img = crop(page_img, box)
        if number_img is None or number_img.size == 0:
            continue
        if _region_contains_xn_marker(number_img):
            continue

        add_candidate(box, val, token, conf)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:max_candidates]

def _is_multi_bag_overview_page(page_img):
    if page_img is None or page_img.size == 0:
        return False

    page_h, page_w = page_img.shape[:2]
    overview_candidates = _collect_page_bag_number_candidates(page_img)

    if len(overview_candidates) < 2:
        return False

    unique_numbers = sorted({item["number"] for item in overview_candidates})
    if len(unique_numbers) < 2:
        return False

    x_centers = [item["box"][0] + (item["box"][2] * 0.5) for item in overview_candidates]
    y_centers = [item["box"][1] + (item["box"][3] * 0.5) for item in overview_candidates]

    x_spread = max(x_centers) - min(x_centers)
    y_spread = max(y_centers) - min(y_centers)
    x_bands = _cluster_band_count(x_centers, page_w * 0.12)
    y_bands = _cluster_band_count(y_centers, page_h * 0.12)

    if (
        len(unique_numbers) >= 2
        and len(overview_candidates) >= 4
        and x_spread >= (page_w * 0.35)
        and y_spread >= (page_h * 0.10)
        and (x_bands >= 2 or y_bands >= 2)
    ):
        return True

    return (
        len(overview_candidates) >= 4
        and len(unique_numbers) >= 4
        and
        x_spread >= (page_w * 0.45)
        and y_spread >= (page_h * 0.15)
        and x_bands >= 2
        and y_bands >= 2
    )

def _looks_like_text_token(token):
    cleaned = "".join(ch for ch in (token or "") if ch.isalpha())
    return len(cleaned) >= 2

def _is_text_heavy_page(page_img):
    if page_img is None or page_img.size == 0:
        return False

    page_h, page_w = page_img.shape[:2]
    gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)

    scale = min(1.0, 1400.0 / float(max(page_w, 1)))
    if scale != 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(round(page_w * scale))), max(1, int(round(page_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    ocr_data = _safe_tesseract_image_to_data(
        gray,
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )

    word_boxes = []
    paragraph_groups = {}
    total_word_area = 0.0
    scaled_h, scaled_w = gray.shape[:2]

    for i, raw_text in enumerate(ocr_data.get("text", [])):
        text = (raw_text or "").strip()
        if not _looks_like_text_token(text):
            continue

        try:
            conf = float(ocr_data["conf"][i])
        except (ValueError, TypeError, KeyError):
            conf = -1.0
        if conf < 35.0:
            continue

        x = int(ocr_data["left"][i])
        y = int(ocr_data["top"][i])
        w = int(ocr_data["width"][i])
        h = int(ocr_data["height"][i])
        if w <= 0 or h <= 0:
            continue

        word_boxes.append((x, y, w, h))
        total_word_area += float(w * h)

        block_num = int(ocr_data.get("block_num", [0] * len(ocr_data["text"]))[i])
        par_num = int(ocr_data.get("par_num", [0] * len(ocr_data["text"]))[i])
        key = (block_num, par_num)
        paragraph_groups.setdefault(key, []).append((x, y, w, h))

    if not word_boxes:
        return False

    page_area = float(max(1, scaled_w * scaled_h))
    text_coverage = total_word_area / page_area
    word_count = len(word_boxes)

    paragraph_like = False
    for boxes in paragraph_groups.values():
        if len(boxes) < 10:
            continue

        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[0] + box[2] for box in boxes)
        y2 = max(box[1] + box[3] for box in boxes)
        union_w = x2 - x1
        union_h = y2 - y1

        if union_w >= (scaled_w * 0.22) and union_h >= (scaled_h * 0.12):
            paragraph_like = True
            break

    return (
        (paragraph_like and word_count >= 14 and text_coverage >= 0.03)
        or (word_count >= 26 and text_coverage >= 0.04)
    )

def _is_build_step_page(page_img):
    if page_img is None or page_img.size == 0:
        return False

    page_h, page_w = page_img.shape[:2]
    gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)

    mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    callout_candidates = []
    page_area = float(max(1, page_h * page_w))

    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        area_ratio = (w * h) / page_area
        aspect_ratio = w / float(max(h, 1))
        if area_ratio < 0.004 or area_ratio > 0.055:
            continue
        if w > (page_w * 0.28) or h > (page_h * 0.20):
            continue
        if w < max(40, int(page_w * 0.04)) or h < max(30, int(page_h * 0.035)):
            continue
        if aspect_ratio < 0.7 or aspect_ratio > 3.4:
            continue

        region_img = page_img[y : y + h, x : x + w]
        if region_img is None or region_img.size == 0:
            continue
        if not _region_contains_xn_marker(region_img):
            continue

        region_gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(region_gray, 60, 140)
        edge_density = float(np.count_nonzero(edges)) / float(max(1, edges.size))
        if edge_density < 0.01:
            continue

        duplicate = False
        for existing in callout_candidates:
            if _box_iou(existing["box"], (x, y, w, h)) >= 0.35:
                duplicate = True
                break
        if duplicate:
            continue

        callout_candidates.append(
            {
                "box": (x, y, w, h),
                "edge_density": edge_density,
            }
        )

    if len(callout_candidates) < 3:
        return False

    x_centers = [item["box"][0] + (item["box"][2] * 0.5) for item in callout_candidates]
    y_centers = [item["box"][1] + (item["box"][3] * 0.5) for item in callout_candidates]
    x_bands = _cluster_band_count(x_centers, page_w * 0.12)
    y_bands = _cluster_band_count(y_centers, page_h * 0.10)
    x_spread = max(x_centers) - min(x_centers)
    y_spread = max(y_centers) - min(y_centers)

    return (
        len(callout_candidates) >= 4
        or (
            len(callout_candidates) >= 3
            and x_spread >= (page_w * 0.25)
            and y_spread >= (page_h * 0.12)
            and (x_bands >= 2 or y_bands >= 2)
        )
    )

def analyze_page(page: int, include_image: bool = True):
    _reset_ocr_error_flag()
    path = page_path(page)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Page not found")

    cache_key = None
    if not include_image:
        cache_key = _path_cache_key(path)
        with ANALYZE_PAGE_CACHE_LOCK:
            cached = ANALYZE_PAGE_CACHE.get(cache_key)
        if cached is not None:
            return _clone_cached_value(cached)

    img = cv2.imread(path)
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read image")

    vis = img.copy() if include_image else None
    step_grid_number_count, green_box_count, multi_step_green_boxes = detect_step_grid(img)
    text_heavy_page = _is_text_heavy_page(img)
    build_step_page = _is_build_step_page(img)
    overview_page = _is_multi_bag_overview_page(img)
    if False:
        if build_step_page:
            rejection_source = "build_step_page"
        elif text_heavy_page:
            rejection_source = "text_heavy_page"
        elif overview_page:
            rejection_source = "overview_page"
        else:
            rejection_source = "rejected_early"
        if include_image:
            cv2.putText(
                vis,
                (
                    "BUILD-STEP / NON-START"
                    if rejection_source == "build_step_page"
                    else "TEXT-HEAVY / NON-START"
                    if rejection_source == "text_heavy_page"
                    else "REJECTED EARLY"
                ),
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        result = {
            "page": page,
            "page_kind": "other",
            "panel_found": False,
            "panel_box": None,
            "panel_source": rejection_source,
            "panel_score": None,
            "shell_found": False,
            "shell_box": None,
            "shell_score": None,
            "shell_method": "none",
            "grey_bag_found": False,
            "grey_bag_box": None,
            "grey_bag_score": None,
            "number_box_found": False,
            "number_box": None,
            "bag_number": None,
            "ocr_raw": "",
            "intro_region_candidates": [],
            "intro_region_top_box": None,
            "ocr_error": _get_ocr_error(),
            "confidence": 0.0,
        }

        if include_image:
            ok, buf = cv2.imencode(".png", vis)
            if not ok:
                raise HTTPException(status_code=500, detail="Encode failed")
            result["image_bytes"] = buf.tobytes()
        elif cache_key is not None:
            with ANALYZE_PAGE_CACHE_LOCK:
                ANALYZE_PAGE_CACHE[cache_key] = _clone_cached_value(result)

        return result

    if overview_page:
        intro_region_candidates = []
        panel = None
        panel_source = "overview_page"
        panel_score = None
        legacy_panel_found = False
    else:
        intro_region_candidates = _find_intro_regions(img)

        panel, panel_source, panel_score = find_bag_intro_panel_with_debug(img)
        legacy_panel_found = panel is not None
        # STEP 1 fallback: if the legacy panel detector finds nothing, use the best
        # intro-region candidate as a simple top-level card candidate.
        if panel is None and intro_region_candidates:
            top_box = intro_region_candidates[0]["box"]
            if top_box is not None:
                panel = tuple(top_box)
                panel_source = "intro_region"
                panel_score = float(intro_region_candidates[0].get("score", 0.0))

    square_abs = None
    shell_score = None
    shell_method = "none"
    bag_abs = None
    bag_score = None
    number_abs = None
    bag_number = None
    ocr_raw = ""

    # --------------------------------------------------
    # PANEL → SHELL → BAG → NUMBER
    # --------------------------------------------------
    if panel is not None:
        panel_img = crop(img, panel)

        square_rel, shell_score, shell_method = find_square_bag_box_with_debug(
            panel_img
        )

        if square_rel is not None:
            px, py, _, _ = panel
            sx, sy, sw, sh = square_rel

            square_abs = (px + sx, py + sy, sw, sh)

            if include_image:
                draw_box(vis, square_abs, (255, 0, 0), 3)

            square_img = crop(img, square_abs)

            bag_rel, bag_score = find_grey_bag_blob(square_img)

            if bag_rel is not None:
                bx, by, _, _ = square_abs
                gx, gy, gw, gh = bag_rel
                bag_abs = (bx + gx, by + gy, gw, gh)

                if include_image:
                    draw_box(vis, bag_abs, (0, 200, 255), 2)

                bag_img = crop(img, bag_abs)

                best_candidate = None
                best_multi_digit_candidate = None

                for number_rel in find_number_crop(bag_img):
                    gx2, gy2, _, _ = bag_abs
                    nx, ny, nw, nh = number_rel

                    number_abs_try = (gx2 + nx, gy2 + ny, nw, nh)
                    number_img = crop(img, number_abs_try)

                    val, raw, ocr_score = read_bag_number_with_score(number_img)
                    if val is None:
                        continue

                    candidate = {
                        "score": ocr_score,
                        "number": val,
                        "raw": raw,
                        "box": number_abs_try,
                    }

                    if (
                        best_candidate is None
                        or candidate["score"] > best_candidate["score"]
                    ):
                        best_candidate = candidate

                    if len(str(val)) >= 2:
                        if (
                            best_multi_digit_candidate is None
                            or candidate["score"] > best_multi_digit_candidate["score"]
                        ):
                            best_multi_digit_candidate = candidate

                chosen_candidate = best_candidate
                if (
                    best_candidate is not None
                    and best_multi_digit_candidate is not None
                    and best_multi_digit_candidate["score"] >= 90.0
                    and best_multi_digit_candidate["score"] + 40.0
                    >= best_candidate["score"]
                ):
                    chosen_candidate = best_multi_digit_candidate

                if chosen_candidate is not None:
                    bag_number = chosen_candidate["number"]
                    ocr_raw = chosen_candidate["raw"]
                    number_abs = chosen_candidate["box"]

                    if include_image and number_abs is not None:
                        draw_box(vis, number_abs, (0, 0, 255), 2)

    # STEP 1 OCR fallback: if we found a panel/intro region but the old shell/bag
    # chain produced no number, try reading numbers directly from the panel region.
    if panel is not None and bag_number is None:
        best_panel_candidate = None
        best_region_box = None
        if not (
            panel_source == "intro_region"
            and _is_effectively_full_page_region(panel, img.shape)
            and bag_abs is None
        ):
            best_panel_candidate, best_region_box = _run_panel_bag_number_fallback(
                img,
                tuple(panel),
                panel_source,
                intro_region_candidates,
            )

        if best_panel_candidate is not None:
            if best_region_box is not None:
                panel = best_region_box
            if panel_source == "intro_region" and best_region_box is not None:
                for region in intro_region_candidates[:4]:
                    if tuple(region["box"]) == tuple(best_region_box):
                        panel_score = float(region.get("score", panel_score or 0.0))
                        break
            bag_number = best_panel_candidate["number"]
            ocr_raw = best_panel_candidate["raw"]
            number_abs = best_panel_candidate["box"]
            if include_image and number_abs is not None:
                draw_box(vis, number_abs, (0, 0, 255), 2)

    # No-shell panel rescue:
    # Some real bag-start pages have a valid intro panel but the shell detector
    # misses the expected square/bag layout. In that case, trust a stronger OCR
    # read inside the panel region itself.
    if (
        panel is not None
        and bag_number is None
        and square_abs is None
        and panel_source in {"intro_region", "strict_top_left"}
    ):
        rescue_candidate, rescue_region_box = _run_panel_bag_number_fallback(
            img,
            tuple(panel),
            panel_source,
            intro_region_candidates,
        )
        if rescue_candidate is not None:
            if rescue_region_box is not None:
                panel = rescue_region_box
            bag_number = rescue_candidate["number"]
            ocr_raw = rescue_candidate["raw"]
            number_abs = rescue_candidate["box"]
            if include_image and number_abs is not None:
                draw_box(vis, number_abs, (0, 0, 255), 2)
                
    # Full-page fallback: some valid bag-start pages have no intro panel/card at
    # all, so only after the panel/introduction paths fail, scan the whole page
    # with stricter OCR acceptance.
    if not overview_page and not legacy_panel_found and bag_number is None:
        page_h, page_w = img.shape[:2]
        full_page_candidate = _find_best_ocr_number_in_region(
            img,
            (0, 0, page_w, page_h),
            single_digit_min_score=65.0,
            reject_step_strips=True,
        )
        if full_page_candidate is not None:
            panel = None
            panel_source = "full_page_fallback"
            panel_score = None
            square_abs = None
            shell_score = None
            shell_method = "none"
            bag_abs = None
            bag_score = None
            bag_number = full_page_candidate["number"]
            ocr_raw = full_page_candidate["raw"]
            number_abs = full_page_candidate["box"]
            if include_image and number_abs is not None:
                draw_box(vis, number_abs, (0, 0, 255), 2)

    # --------------------------------------------------
    # FLAGS
    # --------------------------------------------------
    panel_found = panel is not None
    shell_found = square_abs is not None
    grey_bag_found = bag_abs is not None
    number_box_found = number_abs is not None
    bag_region_flags = _bag_region_quality_flags(img.shape, square_abs, bag_abs)

    confidence = compute_confidence(
        panel_found,
        shell_found,
        grey_bag_found,
        number_box_found,
        bag_number,
        shell_score,
        bag_score,
    )
    if bag_region_flags["tiny_bag_reference"]:
        confidence = min(confidence, 0.45)
    elif bag_region_flags["weak_bag_reference"]:
        confidence = min(confidence, 0.59)

    # --------------------------------------------------
    # DRAW LABELS
    # --------------------------------------------------
    if include_image:
        if panel_found:
            draw_box(vis, panel, (0, 255, 0), 4)
            panel_label = (
                "BAG PANEL"
                if panel_source == "strict_top_left"
                else f"BAG PANEL {panel_source}"
            )
            cv2.putText(
                vis,
                panel_label,
                (panel[0], max(30, panel[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
        elif panel_source == "full_page_fallback":
            cv2.putText(
                vis,
                "FULL PAGE FALLBACK",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
        elif panel_source == "overview_page":
            cv2.putText(
                vis,
                "OVERVIEW / NON-START",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                vis,
                "NO PANEL",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        if shell_found:
            shell_label = (
                f"SHELL {shell_method} {shell_score:.2f}"
                if shell_score is not None
                else f"SHELL {shell_method}"
            )
            cv2.putText(
                vis,
                shell_label,
                (square_abs[0], square_abs[1] + square_abs[3] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
        elif panel_found:
            txt = (
                f"NO SHELL ({shell_score:.2f})"
                if shell_score is not None
                else "NO SHELL"
            )
            cv2.putText(
                vis,
                txt,
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        if grey_bag_found:
            cv2.putText(
                vis,
                f"GREY BAG {bag_score:.1f}",
                (bag_abs[0], max(30, bag_abs[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
        elif shell_found:
            cv2.putText(
                vis,
                "NO GREY BAG",
                (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 200, 255),
                2,
            )

        label = f"BAG {bag_number}" if bag_number is not None else "BAG ?"
        if panel_found or panel_source == "full_page_fallback":
            cv2.putText(
                vis,
                f"{label}  CONF {confidence:.2f}",
                (30, img.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

    structural_rejection_reasons = []
    page_kind = "other"
    no_bag_structure_number_only = bool(
        number_box_found and not shell_found and not grey_bag_found
    )
    if multi_step_green_boxes:
        structural_rejection_reasons.append("multi_step_green_boxes")
        page_kind = (
            "no_bag_structure_number_only"
            if no_bag_structure_number_only
            else "multi_step_green_boxes"
        )

    result = {
        "page": page,
        "page_kind": page_kind,
        "panel_found": panel_found,
        "panel_box": list(panel) if panel_found else None,
        "panel_source": panel_source,
        "panel_score": panel_score,
        "shell_found": shell_found,
        "shell_box": list(square_abs) if shell_found else None,
        "shell_score": shell_score,
        "shell_method": shell_method,
        "grey_bag_found": grey_bag_found,
        "grey_bag_box": list(bag_abs) if grey_bag_found else None,
        "grey_bag_score": bag_score,
        "shell_area_ratio": bag_region_flags["shell_area_ratio"],
        "grey_bag_area_ratio": bag_region_flags["grey_bag_area_ratio"],
        "tiny_bag_reference": bag_region_flags["tiny_bag_reference"],
        "weak_bag_reference": bag_region_flags["weak_bag_reference"],
        "bag_region_quality": bag_region_flags["bag_region_quality"],
        "number_box_found": number_box_found,
        "number_found": number_box_found,
        "number_box": list(number_abs) if number_box_found else None,
        "number_box_x": int(number_abs[0]) if number_box_found else None,
        "number_box_y": int(number_abs[1]) if number_box_found else None,
        "number_box_width": int(number_abs[2]) if number_box_found else None,
        "number_box_height": int(number_abs[3]) if number_box_found else None,
        "number_box_area": (
            int(number_abs[2]) * int(number_abs[3]) if number_box_found else None
        ),
        "bag_number": bag_number,
        "ocr_raw": ocr_raw,
        "intro_region_candidates": intro_region_candidates,
        "intro_region_top_box": (
            list(intro_region_candidates[0]["box"]) if intro_region_candidates else None
        ),
        "text_heavy_page": text_heavy_page,
        "build_step_page": build_step_page,
        "overview_page": overview_page,
        "step_grid_number_count": step_grid_number_count,
        "green_box_count": green_box_count,
        "multi_step_green_boxes": multi_step_green_boxes,
        "ocr_error": _get_ocr_error(),
        "rejection_reason": (
            structural_rejection_reasons[0] if structural_rejection_reasons else None
        ),
        "rejection_reasons": structural_rejection_reasons,
        "confidence": confidence,
    }

    if include_image:
        ok, buf = cv2.imencode(".png", vis)
        if not ok:
            raise HTTPException(status_code=500, detail="Encode failed")
        result["image_bytes"] = buf.tobytes()
    elif cache_key is not None:
        with ANALYZE_PAGE_CACHE_LOCK:
            ANALYZE_PAGE_CACHE[cache_key] = _clone_cached_value(result)

    return result
