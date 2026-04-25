from __future__ import annotations

import json
import logging
import re
import shutil
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import fitz

from .models import BagCandidate, PageData, PdfBagAnalysis
from .utils import ensure_dir


LOG = logging.getLogger("lego_reader.bag_detector")
NUMBER_RE = re.compile(r"\b(\d{1,3})\b")
BAG_WORD_RE = re.compile(r"\bbag\b", re.IGNORECASE)
QUANTITY_RE = re.compile(r"\d\s*x|x\s*\d|\d+x\d+x?", re.IGNORECASE)
MAX_BAG_NUMBER = 50
MISSING_BAG_TOP_K = 3
REVIEWED_BAG_MIN_START_LIKE_CONFIDENCE = 0.70
BAG_BADGE_REGION_WIDTH_RATIO = 0.68
BAG_BADGE_REGION_HEIGHT_RATIO = 0.32
BAG_BADGE_SAMPLE_STEP = 4
BAG_BADGE_BRIGHT_THRESHOLD = 245
BAG_BADGE_DARK_THRESHOLD = 180
INTRO_PANEL_SEARCH_WIDTH_RATIO = 0.72
INTRO_PANEL_SEARCH_HEIGHT_RATIO = 0.45
INTRO_PANEL_MIN_AREA_RATIO = 0.08
INTRO_PANEL_MAX_AREA_RATIO = 0.34
INTRO_PANEL_MIN_ASPECT = 1.4
INTRO_PANEL_MAX_ASPECT = 5.0
INTRO_PANEL_BRIGHT_THRESHOLD = 236
INTRO_PANEL_MIN_PAGE_MEAN = 165
INTRO_PANEL_MIN_ROI_MEAN = 215
INTRO_PANEL_LEFT_MAX_MEAN = 235
INTRO_PANEL_LEFT_MIN_STD = 12
INTRO_PANEL_RIGHT_MIN_STD = 10
INTRO_PANEL_CANNY_LOW = 40
INTRO_PANEL_CANNY_HIGH = 140
INTRO_PANEL_MIN_EDGE_DENSITY = 0.015
INTRO_PANEL_MIN_CONTRAST_GAP = 8
INTRO_PANEL_LEFT_STD_SCALE = 30.0
INTRO_PANEL_LEFT_STD_MAX_SCORE = 0.20
INTRO_PANEL_RIGHT_STD_SCALE = 25.0
INTRO_PANEL_RIGHT_STD_MAX_SCORE = 0.15
INTRO_PANEL_EDGE_DENSITY_SCALE = 0.05
INTRO_PANEL_EDGE_DENSITY_MAX_SCORE = 0.20
INTRO_PANEL_CONTRAST_SCALE = 30.0
INTRO_PANEL_CONTRAST_MAX_SCORE = 0.20
INTRO_PANEL_AREA_IDEAL_RATIO = 0.18
INTRO_PANEL_AREA_WEIGHT = 0.15
INTRO_PANEL_MAX_SCORE = 0.90
INTRO_PANEL_BOOST_THRESHOLD = 0.40
INTRO_PANEL_START_LIKE_MIN = 0.40
INTRO_PANEL_BOOST_MULTIPLIER = 0.20
INTRO_PANEL_MAX_BOOST = 0.10
MULTI_STEP_MIN_PANELS = 4
MULTI_STEP_PANEL_MIN_AREA_RATIO = 0.04
MULTI_STEP_PANEL_MAX_AREA_RATIO = 0.38
MULTI_STEP_SIZE_TOLERANCE = 0.55
MULTI_STEP_X_SPREAD_THRESHOLD = 0.40
MULTI_STEP_Y_SPREAD_THRESHOLD = 0.20
MULTI_STEP_CANNY_LOW = 30
MULTI_STEP_CANNY_HIGH = 100
MULTI_STEP_POLY_EPSILON = 0.03
MULTI_STEP_MIN_VERTICES = 4
MULTI_STEP_MAX_VERTICES = 8
MULTI_STEP_MIN_ASPECT = 0.4
MULTI_STEP_MAX_ASPECT = 4.0
FEATURE_SCALES = {
    "word_count": 50.0,
    "drawing_count": 150.0,
    "image_count": 100.0,
    "dark_pixel_ratio": 0.25,
    "start_like_score": 1.0,
}
FEATURE_WEIGHTS = {
    "word_count": 0.22,
    "drawing_count": 0.18,
    "image_count": 0.18,
    "dark_pixel_ratio": 0.24,
    "start_like_score": 0.18,
}


def _extract_candidate_numbers(page: PageData) -> list[int]:
    raw_numbers = [int(match.group(1)) for match in NUMBER_RE.finditer(page.text or "")]
    counts = Counter(raw_numbers)
    candidates: list[int] = []
    for number in sorted(counts):
        if number < 1 or number > MAX_BAG_NUMBER:
            continue
        if number == page.page_number and counts[number] == 1:
            continue
        candidates.append(number)
    return candidates


def _longest_run(numbers: list[int]) -> int:
    if not numbers:
        return 0
    longest = 1
    current = 1
    for previous, current_num in zip(numbers, numbers[1:]):
        if current_num == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _dark_pixel_ratio(document: fitz.Document, page_index: int) -> float:
    page = document.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), colorspace=fitz.csGRAY, alpha=False)
    samples = pix.samples
    if not samples:
        return 1.0
    dark_pixels = sum(1 for value in samples if value < 200)
    return dark_pixels / len(samples)


def _region_ratio(grid: list[list[int]], x0: int, y0: int, x1: int, y1: int, predicate: Any) -> float:
    if not grid or not grid[0]:
        return 0.0

    height = len(grid)
    width = len(grid[0])
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 < x0 or y1 < y0:
        return 0.0

    total = 0
    matched = 0
    for y in range(y0, y1 + 1):
        row = grid[y]
        for x in range(x0, x1 + 1):
            total += 1
            if predicate(row[x]):
                matched += 1
    if total == 0:
        return 0.0
    return matched / total


def _detect_bag_icon_shape(page: PageData) -> tuple[float, list[str]]:
    if page.image_path is None or not page.image_path.exists():
        return 0.0, ["no rendered page image available for bag icon detection"]

    try:
        import cv2
    except ImportError:
        return 0.0, ["opencv unavailable"]

    image = cv2.imread(page.image_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0, ["image load failed"]

    roi_height = max(1, int(image.shape[0] * 0.40))
    roi_width = max(1, int(image.shape[1] * 0.60))
    roi = image[:roi_height, :roi_width]

    edges = cv2.Canny(roi, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_rect_area = max(12000, int(roi_height * roi_width * 0.015))
    min_arrow_area = max(1500, int(roi_height * roi_width * 0.0015))

    best_score = 0.0
    best_reasons: list[str] = []
    saw_small_box = False
    bounding_boxes: list[tuple[tuple[int, int, int, int], Any]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append(((x, y, w, h), contour))

    for (x, y, w, h), contour in bounding_boxes:
        area = w * h
        if area < min_rect_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        aspect_ratio = w / float(max(h, 1))
        if not (1.6 <= aspect_ratio <= 5.5):
            saw_small_box = True
            continue

        frame = roi[y : y + h, x : x + w]
        if frame.size == 0:
            continue

        outer_white_ratio = float((frame >= 235).mean())
        if outer_white_ratio < 0.55:
            continue

        reasons = ["outer white frame found"]
        score = 0.40

        inner_x0 = x + max(1, int(w * 0.04))
        inner_x1 = x + max(2, int(w * 0.28))
        inner_y0 = y + max(1, int(h * 0.10))
        inner_y1 = y + max(2, int(h * 0.84))
        inner = roi[inner_y0:inner_y1, inner_x0:inner_x1]
        if inner.size == 0:
            continue

        inner_nonwhite_ratio = float((inner < 245).mean())
        inner_dark_ratio = float((inner < 190).mean())
        if not (0.22 <= inner_nonwhite_ratio <= 0.82 and 0.05 <= inner_dark_ratio <= 0.28):
            saw_small_box = True
            continue

        score += 0.25
        reasons.append("inner bag-like region found")

        search_x0 = max(0, x + int(w * 0.18))
        search_y0 = max(0, y + int(h * 0.18))
        search_x1 = min(roi_width, x + w + int(w * 0.90))
        search_y1 = min(roi_height, y + h + int(h * 0.55))

        arrow_found = False
        for (ax, ay, aw, ah), other_contour in bounding_boxes:
            if other_contour is contour:
                continue
            if aw * ah < min_arrow_area:
                continue
            if ax < search_x0 or ay < search_y0 or ax + aw > search_x1 or ay + ah > search_y1:
                continue

            other_perimeter = cv2.arcLength(other_contour, True)
            if other_perimeter <= 0:
                continue

            other_approx = cv2.approxPolyDP(other_contour, 0.03 * other_perimeter, True)
            if 3 <= len(other_approx) <= 8:
                arrow_found = True
                break

        if arrow_found:
            score += 0.20
            reasons.append("arrow/connector-like contour found nearby")

        if score > best_score:
            best_score = min(score, 0.95)
            best_reasons = reasons

    if best_score <= 0.0:
        if saw_small_box:
            return 0.0, ["small sticker-like box rejected"]
        return 0.0, ["no bag icon shape found in restricted region"]
    return best_score, best_reasons


def _detect_intro_panel(page: PageData) -> tuple[float, list[str]]:
    """Detect the bag-start intro panel in the top-left of the page.

    A true bag-start page contains a wide bright panel anchored in the
    top-left corner.  The panel has three internal zones:
      - left  zone: the bag card / leaflet image (darker, textured)
      - middle zone: red arrow connecting card to model (edge-dense)
      - right zone:  partial model preview (brighter, textured)

    Returns (score, reasons).  score is in the range [0.0, INTRO_PANEL_MAX_SCORE].
    A score of 0.0 means no intro panel was found; higher values indicate
    increasing confidence that the detected bright region is a bag-start panel.
    """
    if page.image_path is None or not page.image_path.exists():
        return 0.0, ["no rendered page image available for intro panel detection"]

    try:
        import cv2
        import numpy as np
    except ImportError:
        return 0.0, ["opencv unavailable"]

    image = cv2.imread(page.image_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0, ["image load failed"]

    h, w = image.shape[:2]
    page_mean = float(np.mean(image))
    if page_mean < INTRO_PANEL_MIN_PAGE_MEAN:
        return 0.0, ["page too dark for bag-start intro panel"]

    sx2 = int(w * INTRO_PANEL_SEARCH_WIDTH_RATIO)
    sy2 = int(h * INTRO_PANEL_SEARCH_HEIGHT_RATIO)
    search = image[:sy2, :sx2]

    _, th = cv2.threshold(search, INTRO_PANEL_BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_area = float(w * h)
    best_score = 0.0
    best_reasons: list[str] = ["no intro panel found in top-left region"]

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        area_ratio = area / page_area
        wh_ratio = bw / float(max(bh, 1))

        if not (INTRO_PANEL_MIN_AREA_RATIO <= area_ratio <= INTRO_PANEL_MAX_AREA_RATIO):
            continue
        if not (INTRO_PANEL_MIN_ASPECT <= wh_ratio <= INTRO_PANEL_MAX_ASPECT):
            continue
        if x > int(w * 0.08) or y > int(h * 0.10):
            continue

        roi = image[y: y + bh, x: x + bw]
        if roi.size == 0:
            continue

        roi_mean = float(np.mean(roi))
        if roi_mean < INTRO_PANEL_MIN_ROI_MEAN:
            continue

        left = roi[:, : int(bw * 0.28)]
        mid = roi[:, int(bw * 0.28): int(bw * 0.45)]
        right = roi[:, int(bw * 0.45):]

        if left.size == 0 or mid.size == 0 or right.size == 0:
            continue

        left_mean = float(np.mean(left))
        right_mean = float(np.mean(right))
        left_std = float(np.std(left))
        right_std = float(np.std(right))

        # Left (bag card) must be darker/textured, not blank white
        if left_mean > INTRO_PANEL_LEFT_MAX_MEAN or left_std < INTRO_PANEL_LEFT_MIN_STD:
            continue
        # Right (model preview) must have some texture
        if right_std < INTRO_PANEL_RIGHT_MIN_STD:
            continue

        # Middle (arrow zone) must have edge structure
        mid_edges = cv2.Canny(mid, INTRO_PANEL_CANNY_LOW, INTRO_PANEL_CANNY_HIGH)
        mid_edge_density = float(np.mean(mid_edges > 0))
        if mid_edge_density < INTRO_PANEL_MIN_EDGE_DENSITY:
            continue

        # Card (left) darker than model background (right)
        contrast_gap = right_mean - left_mean
        if contrast_gap < INTRO_PANEL_MIN_CONTRAST_GAP:
            continue

        score = 0.0
        reasons: list[str] = [f"intro panel at ({x},{y}) size {bw}x{bh}"]
        score += min(left_std / INTRO_PANEL_LEFT_STD_SCALE, INTRO_PANEL_LEFT_STD_MAX_SCORE)
        reasons.append(f"bag-card zone texture std={left_std:.1f}")
        score += min(right_std / INTRO_PANEL_RIGHT_STD_SCALE, INTRO_PANEL_RIGHT_STD_MAX_SCORE)
        reasons.append(f"model-preview zone texture std={right_std:.1f}")
        score += min(mid_edge_density / INTRO_PANEL_EDGE_DENSITY_SCALE, INTRO_PANEL_EDGE_DENSITY_MAX_SCORE)
        reasons.append(f"arrow zone edge density={mid_edge_density:.3f}")
        score += min(contrast_gap / INTRO_PANEL_CONTRAST_SCALE, INTRO_PANEL_CONTRAST_MAX_SCORE)
        reasons.append(f"left-right contrast gap={contrast_gap:.1f}")
        area_closeness = max(0.0, 1.0 - abs(area_ratio - INTRO_PANEL_AREA_IDEAL_RATIO) / INTRO_PANEL_AREA_IDEAL_RATIO)
        score += area_closeness * INTRO_PANEL_AREA_WEIGHT
        reasons.append(f"panel area ratio={area_ratio:.3f}")
        score = min(score, INTRO_PANEL_MAX_SCORE)

        if score > best_score:
            best_score = score
            best_reasons = reasons

    return best_score, best_reasons


def _detect_multi_step_grid(page: PageData) -> tuple[bool, list[str]]:
    """Detect whether a page is a multi-step build-instruction grid.

    Multi-step grid pages (e.g., a 2×2 or 2×3 layout of build steps) are
    NOT bag-start pages.  They are identified by:
      - four or more rectangular panels of similar size
      - panels spread across both the horizontal and vertical extents of the
        page (not confined to the top-left corner)

    Returns (True, reasons) when the page should be rejected as a grid page.
    Returns (False, reasons) when the page does not match the multi-step grid
    pattern and should proceed to further bag-start evaluation.
    """
    if page.image_path is None or not page.image_path.exists():
        return False, ["no rendered page image available for grid detection"]

    try:
        import cv2
    except ImportError:
        return False, ["opencv unavailable"]

    image = cv2.imread(page.image_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False, ["image load failed"]

    h, w = image.shape[:2]
    page_area = float(h * w)
    min_panel_area = page_area * MULTI_STEP_PANEL_MIN_AREA_RATIO
    max_panel_area = page_area * MULTI_STEP_PANEL_MAX_AREA_RATIO

    edges = cv2.Canny(image, MULTI_STEP_CANNY_LOW, MULTI_STEP_CANNY_HIGH)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_panels: list[tuple[int, int, int, int]] = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(c, MULTI_STEP_POLY_EPSILON * perimeter, True)
        if len(approx) < MULTI_STEP_MIN_VERTICES or len(approx) > MULTI_STEP_MAX_VERTICES:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if not (min_panel_area <= area <= max_panel_area):
            continue
        aspect = bw / float(max(bh, 1))
        if not (MULTI_STEP_MIN_ASPECT <= aspect <= MULTI_STEP_MAX_ASPECT):
            continue
        rect_panels.append((x, y, bw, bh))

    if len(rect_panels) < MULTI_STEP_MIN_PANELS:
        return False, [
            f"only {len(rect_panels)} panel-sized rectangles found "
            f"(need {MULTI_STEP_MIN_PANELS})"
        ]

    areas = sorted(bw * bh for _, _, bw, bh in rect_panels)
    median_area = areas[len(areas) // 2]
    similar_panels = [
        (x, y, bw, bh)
        for x, y, bw, bh in rect_panels
        if MULTI_STEP_SIZE_TOLERANCE * median_area
        <= bw * bh
        <= (1.0 / MULTI_STEP_SIZE_TOLERANCE) * median_area
    ]

    if len(similar_panels) < MULTI_STEP_MIN_PANELS:
        return False, [
            f"panels not uniform enough for multi-step grid "
            f"({len(similar_panels)} similar of {len(rect_panels)} total)"
        ]

    x_centers = [x + bw // 2 for x, y, bw, bh in similar_panels]
    y_centers = [y + bh // 2 for x, y, bw, bh in similar_panels]
    x_spread = (max(x_centers) - min(x_centers)) / float(max(w, 1))
    y_spread = (max(y_centers) - min(y_centers)) / float(max(h, 1))

    if x_spread >= MULTI_STEP_X_SPREAD_THRESHOLD and y_spread >= MULTI_STEP_Y_SPREAD_THRESHOLD:
        return True, [
            f"multi-step grid: {len(similar_panels)} similar panels, "
            f"x-spread={x_spread:.2f}, y-spread={y_spread:.2f}"
        ]

    return False, [
        f"panels not in full-page grid "
        f"(x-spread={x_spread:.2f}, y-spread={y_spread:.2f})"
    ]


def _detect_bag_badge(page: PageData) -> tuple[float, list[str]]:
    if page.image_path is None or not page.image_path.exists():
        return 0.0, ["no rendered page image available for bag badge detection"]

    try:
        pix = fitz.Pixmap(page.image_path.as_posix())
    except RuntimeError:
        return 0.0, ["could not read rendered page image for bag badge detection"]

    roi_width = max(1, int(pix.width * BAG_BADGE_REGION_WIDTH_RATIO))
    roi_height = max(1, int(pix.height * BAG_BADGE_REGION_HEIGHT_RATIO))
    step = BAG_BADGE_SAMPLE_STEP
    channels = pix.n
    samples = pix.samples

    grid: list[list[int]] = []
    bright_points: list[tuple[int, int]] = []
    bright_count = 0
    total = 0

    for y in range(0, roi_height, step):
        row: list[int] = []
        row_base = y * pix.width * channels
        for x in range(0, roi_width, step):
            sample_index = row_base + x * channels
            if channels >= 3:
                gray = (samples[sample_index] + samples[sample_index + 1] + samples[sample_index + 2]) // 3
            else:
                gray = samples[sample_index]
            row.append(gray)
            total += 1
            if gray >= BAG_BADGE_BRIGHT_THRESHOLD:
                bright_count += 1
                bright_points.append((len(row) - 1, len(grid)))
        grid.append(row)

    if not grid or not grid[0] or not bright_points or total == 0:
        return 0.0, ["top-left outer rectangle not found"]

    sampled_height = len(grid)
    sampled_width = len(grid[0])
    xs = [point[0] for point in bright_points]
    ys = [point[1] for point in bright_points]
    x0 = min(xs)
    x1 = max(xs)
    y0 = min(ys)
    y1 = max(ys)
    box_width = x1 - x0 + 1
    box_height = y1 - y0 + 1
    bright_ratio = bright_count / total
    width_ratio = box_width / sampled_width
    height_ratio = box_height / sampled_height
    top_ratio = y0 / sampled_height
    left_ratio = x0 / sampled_width

    reasons: list[str] = []
    score = 0.0

    outer_rectangle_found = (
        bright_ratio >= 0.20
        and width_ratio >= 0.55
        and height_ratio >= 0.45
        and top_ratio <= 0.15
        and left_ratio <= 0.12
    )
    if not outer_rectangle_found:
        if bright_ratio >= 0.08 and (width_ratio < 0.55 or height_ratio < 0.45):
            return 0.0, ["small sticker-like box rejected"]
        return 0.0, ["top-left outer rectangle not found"]

    score += 0.45
    reasons.append("outer white frame found")

    inner_x0 = x0 + max(1, int(box_width * 0.03))
    inner_x1 = x0 + max(2, int(box_width * 0.26))
    inner_y0 = y0 + max(1, int(box_height * 0.08))
    inner_y1 = y0 + max(2, int(box_height * 0.82))
    inner_bright_ratio = _region_ratio(
        grid,
        inner_x0,
        inner_y0,
        inner_x1,
        inner_y1,
        lambda value: value >= BAG_BADGE_BRIGHT_THRESHOLD,
    )
    inner_dark_density = _region_ratio(
        grid,
        inner_x0,
        inner_y0,
        inner_x1,
        inner_y1,
        lambda value: value < BAG_BADGE_DARK_THRESHOLD,
    )
    inner_nonwhite_ratio = _region_ratio(
        grid,
        inner_x0,
        inner_y0,
        inner_x1,
        inner_y1,
        lambda value: value < BAG_BADGE_BRIGHT_THRESHOLD,
    )
    inner_bag_like_found = (
        0.22 <= inner_nonwhite_ratio <= 0.82
        and 0.05 <= inner_dark_density <= 0.20
        and 0.20 <= inner_bright_ratio <= 0.60
    )
    if inner_bag_like_found:
        score += 0.25
        reasons.append("inner bag-like region found")
        score += 0.10
        reasons.append("large centered bag number region found")
    else:
        reasons.append("small sticker-like box rejected")

    connector_x0 = x0 + max(1, int(box_width * 0.25))
    connector_x1 = x0 + max(2, int(box_width * 0.38))
    connector_y0 = y0 + max(1, int(box_height * 0.28))
    connector_y1 = y0 + max(2, int(box_height * 0.68))
    connector_dark_density = _region_ratio(
        grid,
        connector_x0,
        connector_y0,
        connector_x1,
        connector_y1,
        lambda value: value < BAG_BADGE_DARK_THRESHOLD,
    )
    connector_bright_ratio = _region_ratio(
        grid,
        connector_x0,
        connector_y0,
        connector_x1,
        connector_y1,
        lambda value: value >= 235,
    )
    if inner_bag_like_found and 0.02 <= connector_dark_density <= 0.18 and connector_bright_ratio >= 0.70:
        score += 0.15
        reasons.append("possible connector shape near badge")

    return min(score, 0.95), reasons


def _overview_score(page: PageData, numbers: list[int], dark_ratio: float) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    run_length = _longest_run(numbers)

    if len(numbers) >= 4:
        score += 0.25
        reasons.append(f"contains {len(numbers)} bag-like numbers")
    if run_length >= 4:
        score += 0.25
        reasons.append(f"contains ascending run of {run_length}")
    if page.word_count < 25:
        score += 0.15
        reasons.append(f"low word count ({page.word_count})")
    if 20 <= page.drawing_count <= 150:
        score += 0.1
        reasons.append(f"moderate drawing count ({page.drawing_count})")
    if dark_ratio < 0.16:
        score += 0.15
        reasons.append(f"visually sparse page (dark ratio {dark_ratio:.3f})")
    if page.image_count > 15:
        score += 0.1
        reasons.append(f"image count boost ({page.image_count})")
    if QUANTITY_RE.search(page.text or ""):
        score -= 0.45
        reasons.append("contains quantity markers typical of step pages")
    else:
        score += 0.1
        reasons.append("no quantity markers")

    return max(0.0, min(score, 0.99)), reasons


def _bag_start_like_score(page: PageData, dark_ratio: float) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if QUANTITY_RE.search(page.text or ""):
        return 0.0, ["rejected: contains quantity markers"]
    if page.word_count > 50:
        return 0.0, ["rejected: too many words for a bag-start page"]

    score = 0.0
    if page.word_count < 20:
        score += 0.25
        reasons.append(f"low word count ({page.word_count})")
    if 20 <= page.drawing_count <= 150:
        score += 0.2
        reasons.append(f"moderate drawing count ({page.drawing_count})")
    if dark_ratio < 0.11:
        score += 0.2
        reasons.append(f"visually sparse page (dark ratio {dark_ratio:.3f})")
    if page.image_count > 15:
        score += 0.15
        reasons.append(f"image count boost ({page.image_count})")
    if BAG_WORD_RE.search(page.text or ""):
        score += 0.15
        reasons.append("contains explicit bag label")
    if not QUANTITY_RE.search(page.text or ""):
        score += 0.05
        reasons.append("no quantity markers")

    return max(0.0, min(score, 0.99)), reasons


def _resolve_bag_number(page: PageData, numbers: list[int]) -> tuple[int | None, list[str]]:
    reasons: list[str] = []
    remaining = sorted({number for number in numbers if number != page.page_number})
    if len(remaining) == 1:
        reasons.append(f"resolved bag number {remaining[0]} after dropping page number {page.page_number}")
        return remaining[0], reasons
    if len(numbers) == 1:
        reasons.append(f"single extracted bag number {numbers[0]}")
        return numbers[0], reasons
    reasons.append("no unambiguous bag number extracted")
    return None, reasons


def _candidate_stem(page_number: int) -> str:
    return f"candidate_page_{page_number:03d}"


def _candidate_json_path(debug_dir: Path, page_number: int) -> Path:
    return debug_dir / f"{_candidate_stem(page_number)}.json"


def _normalize_reviewed_expected_bags(raw_value: object) -> list[int]:
    if not isinstance(raw_value, list):
        return []

    reviewed_numbers: set[int] = set()
    for item in raw_value:
        number: int | None = None
        if isinstance(item, int):
            number = item
        elif isinstance(item, str) and item.strip().isdigit():
            number = int(item.strip())

        if number is None:
            continue
        if 1 <= number <= MAX_BAG_NUMBER:
            reviewed_numbers.add(number)

    return sorted(reviewed_numbers)


def _load_reviewed_expected_bags(candidate_json_path: Path) -> list[int]:
    if not candidate_json_path.exists():
        return []

    try:
        payload = json.loads(candidate_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    return _normalize_reviewed_expected_bags(payload.get("reviewed_expected_bags"))


def _load_reviewed_bag(candidate_json_path: Path) -> int | None:
    if not candidate_json_path.exists():
        return None

    try:
        payload = json.loads(candidate_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if payload.get("classification") != "bag_start":
        return None

    raw_value = payload.get("reviewed_bag")
    if isinstance(raw_value, int):
        bag = raw_value
    elif isinstance(raw_value, str) and raw_value.strip().isdigit():
        bag = int(raw_value.strip())
    else:
        return None

    if 1 <= bag <= MAX_BAG_NUMBER:
        return bag
    return None


def _feature_vector(page: PageData, dark_ratio: float, start_like_score: float) -> dict[str, float]:
    return {
        "word_count": float(page.word_count),
        "drawing_count": float(page.drawing_count),
        "image_count": float(page.image_count),
        "dark_pixel_ratio": float(dark_ratio),
        "start_like_score": float(start_like_score),
    }


def _page_similarity(
    candidate_page: PageData,
    candidate_dark_ratio: float,
    candidate_start_like_score: float,
    reference_page: PageData,
    reference_dark_ratio: float,
    reference_start_like_score: float,
) -> float:
    candidate_vector = _feature_vector(candidate_page, candidate_dark_ratio, candidate_start_like_score)
    reference_vector = _feature_vector(reference_page, reference_dark_ratio, reference_start_like_score)
    total = 0.0
    for key, weight in FEATURE_WEIGHTS.items():
        scale = FEATURE_SCALES[key]
        diff = abs(candidate_vector[key] - reference_vector[key])
        component = max(0.0, 1.0 - min(diff / scale, 1.0))
        total += component * weight
    return round(total, 4)


def _pages_per_bag_step(confirmed_bags: list[BagCandidate]) -> float:
    ordered = sorted(confirmed_bags, key=lambda bag: bag.bag)
    if len(ordered) < 2:
        return 18.0

    steps: list[float] = []
    for previous, current in zip(ordered, ordered[1:]):
        bag_delta = current.bag - previous.bag
        if bag_delta <= 0:
            continue
        steps.append((current.start_page - previous.start_page) / bag_delta)

    if not steps:
        return 18.0
    return float(median(steps))


def _neighboring_confirmed_bags(target_bag: int, confirmed_bags: list[BagCandidate]) -> tuple[BagCandidate | None, BagCandidate | None]:
    lower: BagCandidate | None = None
    upper: BagCandidate | None = None
    for bag in sorted(confirmed_bags, key=lambda item: item.bag):
        if bag.bag < target_bag:
            lower = bag
            continue
        if bag.bag > target_bag:
            upper = bag
            break
    return lower, upper


def _estimated_transition_page(target_bag: int, confirmed_bags: list[BagCandidate], default_step: float) -> float | None:
    if not confirmed_bags:
        return None

    lower, upper = _neighboring_confirmed_bags(target_bag, confirmed_bags)
    if lower and upper and upper.bag != lower.bag:
        ratio = (target_bag - lower.bag) / (upper.bag - lower.bag)
        return lower.start_page + ratio * (upper.start_page - lower.start_page)
    if lower:
        return lower.start_page + (target_bag - lower.bag) * default_step
    if upper:
        return upper.start_page - (upper.bag - target_bag) * default_step
    return None


def _transition_score(page_number: int, estimated_page: float | None, page_step: float) -> float:
    if estimated_page is None:
        return 0.0
    window = max(18.0, page_step * 1.75)
    distance = abs(page_number - estimated_page)
    return round(max(0.0, 1.0 - min(distance / window, 1.0)), 4)


def _build_bag_intervals(confirmed_bags: list[BagCandidate]) -> list[dict[str, int]]:
    if len(confirmed_bags) < 2:
        return []

    sorted_bags = sorted(confirmed_bags, key=lambda bag: bag.start_page)
    intervals: list[dict[str, int]] = []

    for current, nxt in zip(sorted_bags, sorted_bags[1:]):
        if current.bag is None or nxt.bag is None:
            continue
        if current.bag + 1 > nxt.bag - 1:
            continue

        intervals.append(
            {
                "bag_min": current.bag + 1,
                "bag_max": nxt.bag - 1,
                "page_min": current.start_page,
                "page_max": nxt.start_page,
            }
        )

    return intervals


def _candidate_within_interval(candidate: dict[str, Any], missing_bag: int, intervals: list[dict[str, int]]) -> bool:
    page_number = candidate.get("page_number")
    if not isinstance(page_number, int):
        return True

    for interval in intervals:
        if interval["bag_min"] <= missing_bag <= interval["bag_max"]:
            return interval["page_min"] <= page_number <= interval["page_max"]

    return True


def _final_missing_candidate_score(candidate: dict[str, Any]) -> float:
    return candidate["score"] + (candidate.get("bag_badge_score", 0.0) * 0.15)


def _missing_bag_candidate_entry(
    missing_bag: int,
    estimated_page: float | None,
    page: PageData,
    numbers: list[int],
    dark_ratio: float,
    start_like_score: float,
    confirmed_profiles: list[dict[str, Any]],
    overview_page_numbers: set[int],
    confirmed_start_pages: set[int],
    page_step: float,
) -> dict[str, Any] | None:
    if page.page_number in overview_page_numbers or page.page_number in confirmed_start_pages:
        return None
    if QUANTITY_RE.search(page.text or ""):
        return None
    if page.word_count > 35:
        return None

    sparse_layout = (
        page.word_count < 20
        and 10 <= page.drawing_count <= 180
        and (page.image_count > 15 or dark_ratio < 0.16)
    )
    if start_like_score < 0.45 and not sparse_layout:
        return None

    transition = _transition_score(page.page_number, estimated_page, page_step)
    if estimated_page is not None and transition < 0.15:
        return None

    similarity = 0.0
    if confirmed_profiles:
        similarity = max(
            _page_similarity(
                page,
                dark_ratio,
                start_like_score,
                profile["page"],
                profile["dark_ratio"],
                profile["start_like_score"],
            )
            for profile in confirmed_profiles
        )

    bag_badge_score, _ = _detect_bag_badge(page)
    bag_icon_score, _ = _detect_bag_icon_shape(page)

    score = 0.45 * similarity + 0.35 * transition + 0.2 * start_like_score

    if bag_badge_score >= 0.85:
        score += 0.10
    elif bag_badge_score >= 0.60:
        score += 0.05

    if bag_icon_score >= 0.80:
        score += 0.10
    reasons = [
        f"similarity to confirmed bag starts {similarity:.3f}",
        f"bag-start-like layout score {start_like_score:.3f}",
    ]
    if estimated_page is not None:
        reasons.append(
            f"near estimated transition page {estimated_page:.1f} (transition score {transition:.3f})"
        )
    else:
        reasons.append("no transition estimate available")
    if sparse_layout:
        reasons.append("sparse part-overview layout")
    reasons.append(f"badge score {bag_badge_score:.3f}")
    if missing_bag in numbers:
        score += 0.08
        reasons.append(f"page text includes missing bag number {missing_bag}")

    score = min(score, 0.99)
    return {
        "bag": missing_bag,
        "page_number": page.page_number,
        "estimated_page": round(estimated_page, 1) if estimated_page is not None else None,
        "score": round(score, 3),
        "similarity_to_confirmed_start": round(similarity, 3),
        "transition_score": round(transition, 3),
        "start_like_score": round(start_like_score, 3),
        "bag_badge_score": round(bag_badge_score, 3),
        "detected_numbers": numbers,
        "reasons": reasons,
        "image_path": page.image_path.as_posix() if page.image_path else None,
    }


def _build_missing_bag_review_groups(
    missing_bags: list[int],
    pages: list[PageData],
    page_numbers: dict[int, list[int]],
    page_dark_ratios: dict[int, float],
    page_start_like_scores: dict[int, float],
    confirmed_bags: list[BagCandidate],
    overview_pages: list[int],
) -> list[dict[str, Any]]:
    confirmed_profiles: list[dict[str, Any]] = []
    page_lookup = {page.page_number: page for page in pages}
    for bag in sorted(confirmed_bags, key=lambda item: item.bag):
        page = page_lookup.get(bag.start_page)
        if page is None:
            continue
        confirmed_profiles.append(
            {
                "bag": bag.bag,
                "page": page,
                "dark_ratio": page_dark_ratios.get(page.page_number, 1.0),
                "start_like_score": page_start_like_scores.get(page.page_number, 0.0),
            }
        )

    page_step = _pages_per_bag_step(confirmed_bags)
    overview_page_numbers = set(overview_pages)
    confirmed_start_pages = {bag.start_page for bag in confirmed_bags}
    intervals = _build_bag_intervals(confirmed_bags)
    groups: list[dict[str, Any]] = []

    for missing_bag in missing_bags:
        estimated_page = _estimated_transition_page(missing_bag, confirmed_bags, page_step)
        candidates: list[dict[str, Any]] = []
        for page in pages:
            candidate = _missing_bag_candidate_entry(
                missing_bag=missing_bag,
                estimated_page=estimated_page,
                page=page,
                numbers=page_numbers.get(page.page_number, []),
                dark_ratio=page_dark_ratios.get(page.page_number, 1.0),
                start_like_score=page_start_like_scores.get(page.page_number, 0.0),
                confirmed_profiles=confirmed_profiles,
                overview_page_numbers=overview_page_numbers,
                confirmed_start_pages=confirmed_start_pages,
                page_step=page_step,
            )
            if candidate is not None:
                candidates.append(candidate)

        filtered_candidates = [
            candidate for candidate in candidates if _candidate_within_interval(candidate, missing_bag, intervals)
        ]
        if filtered_candidates:
            candidates = filtered_candidates

        candidates.sort(
            key=lambda item: (
                -_final_missing_candidate_score(item),
                abs(item["page_number"] - item["estimated_page"]) if item["estimated_page"] is not None else item["page_number"],
                item["page_number"],
            )
        )
        groups.append(
            {
                "bag": missing_bag,
                "estimated_page": round(estimated_page, 1) if estimated_page is not None else None,
                "candidates": candidates[:MISSING_BAG_TOP_K],
            }
        )

    return groups


def _save_candidate_debug(
    debug_dir: Path,
    page: PageData,
    detected_numbers: list[int],
    classification: str,
    accepted: bool,
    confidence: float,
    reasons: list[str],
    dark_ratio: float,
) -> None:
    ensure_dir(debug_dir)
    stem = _candidate_stem(page.page_number)
    candidate_json_path = _candidate_json_path(debug_dir, page.page_number)
    reviewed_expected_bags = _load_reviewed_expected_bags(candidate_json_path)
    reviewed_bag = _load_reviewed_bag(candidate_json_path)
    snippet_image_path: Path | None = None
    if page.image_path and page.image_path.exists():
        snippet_image_path = debug_dir / f"{stem}{page.image_path.suffix or '.png'}"
        shutil.copy2(page.image_path, snippet_image_path)

    bag_badge_score, bag_badge_reasons = _detect_bag_badge(page)
    bag_icon_score, bag_icon_reasons = _detect_bag_icon_shape(page)
    intro_panel_score, intro_panel_reasons = _detect_intro_panel(page)
    multi_step_grid, multi_step_reasons = _detect_multi_step_grid(page)

    payload = {
        "page_number": page.page_number,
        "detected_numbers": detected_numbers,
        "classification": classification,
        "accepted": accepted,
        "confidence": round(confidence, 3),
        "reasons": reasons,
        "dark_pixel_ratio": round(dark_ratio, 4),
        "bag_badge_score": round(bag_badge_score, 3),
        "bag_badge_reasons": bag_badge_reasons,
        "bag_icon_score": round(bag_icon_score, 3),
        "bag_icon_reasons": bag_icon_reasons,
        "intro_panel_score": round(intro_panel_score, 3),
        "intro_panel_reasons": intro_panel_reasons,
        "multi_step_grid": multi_step_grid,
        "multi_step_grid_reasons": multi_step_reasons,
        "text_excerpt": page.text[:500],
        "word_count": page.word_count,
        "drawing_count": page.drawing_count,
        "image_count": page.image_count,
        "image_path": snippet_image_path.as_posix() if snippet_image_path else None,
        "source_image_path": page.image_path.as_posix() if page.image_path else None,
    }
    if reviewed_bag is not None:
        payload["reviewed_bag"] = reviewed_bag
    if classification == "overview" or reviewed_expected_bags:
        payload["reviewed_expected_bags"] = reviewed_expected_bags
    candidate_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_pdf_for_bags(pdf_path: Path, pages: list[PageData], debug: bool, debug_dir: Path) -> PdfBagAnalysis:
    document = fitz.open(pdf_path)
    overview_pages: list[int] = []
    overview_numbers: set[int] = set()
    bag_start_like_pages: list[int] = []
    bags: list[BagCandidate] = []
    uncertain_pages: list[int] = []
    seen_single_bags: set[int] = set()
    expected_bags: list[int] = []
    accepted_overviews: list[dict[str, Any]] = []
    confirmed_bag_start_review: list[dict[str, Any]] = []
    page_numbers: dict[int, list[int]] = {}
    page_dark_ratios: dict[int, float] = {}
    page_start_like_scores: dict[int, float] = {}

    for page in pages:
        candidate_json_path = _candidate_json_path(debug_dir / "candidates", page.page_number)
        reviewed_bag = _load_reviewed_bag(candidate_json_path)
        numbers = _extract_candidate_numbers(page)
        dark_ratio = _dark_pixel_ratio(document, page.page_index)
        start_like_confidence, start_like_reasons = _bag_start_like_score(page, dark_ratio)
        bag_badge_score, bag_badge_reasons = _detect_bag_badge(page)
        bag_icon_score, _ = _detect_bag_icon_shape(page)
        intro_panel_score, intro_panel_reasons = _detect_intro_panel(page)
        multi_step_grid, multi_step_reasons = _detect_multi_step_grid(page)

        LOG.debug(
            "Page %s signals: start_like=%.3f badge=%.3f icon=%.3f intro_panel=%.3f multi_step_grid=%s",
            page.page_number,
            start_like_confidence,
            bag_badge_score,
            bag_icon_score,
            intro_panel_score,
            multi_step_grid,
        )

        # Hard rejection: multi-step grid pages are not bag-start pages.
        # A reviewed bag label overrides this guard to allow manual corrections.
        if multi_step_grid and reviewed_bag is None:
            LOG.info(
                "Rejected page %s as multi-step grid in %s: %s",
                page.page_number,
                pdf_path.name,
                "; ".join(multi_step_reasons),
            )
            if debug:
                _save_candidate_debug(
                    debug_dir / "candidates",
                    page,
                    numbers,
                    "rejected-grid",
                    False,
                    0.0,
                    multi_step_reasons,
                    dark_ratio,
                )
            continue

        if (
            bag_badge_score >= 0.90
            and bag_icon_score >= 0.80
            and start_like_confidence >= 0.65
        ):
            start_like_confidence = min(0.99, start_like_confidence + 0.03)
            start_like_reasons = start_like_reasons + [
                f"strong bag header signals observed (badge {bag_badge_score:.3f}, icon {bag_icon_score:.3f})"
            ]

        # Positive boost: a detected intro panel (bag card + arrow + model zones)
        # provides strong direct evidence that this is a bag-start page.
        # The start_like_confidence guard prevents the intro panel from rescuing
        # pages that fail all text/layout checks (e.g., dark promo pages that
        # happen to have a bright rectangle in the top-left corner).
        if intro_panel_score >= INTRO_PANEL_BOOST_THRESHOLD and start_like_confidence >= INTRO_PANEL_START_LIKE_MIN:
            boost = min(intro_panel_score * INTRO_PANEL_BOOST_MULTIPLIER, INTRO_PANEL_MAX_BOOST)
            start_like_confidence = min(0.99, start_like_confidence + boost)
            start_like_reasons = start_like_reasons + [
                f"intro panel detected (score {intro_panel_score:.3f}): "
                + "; ".join(intro_panel_reasons[:2])
            ]

        page_numbers[page.page_number] = numbers
        page_dark_ratios[page.page_number] = dark_ratio
        page_start_like_scores[page.page_number] = start_like_confidence

        if reviewed_bag is not None:
            if start_like_confidence < REVIEWED_BAG_MIN_START_LIKE_CONFIDENCE:
                reviewed_reasons = [
                    (
                        f"rejected manual bag-start review for bag {reviewed_bag}: "
                        f"start-like confidence {start_like_confidence:.3f} "
                        f"below {REVIEWED_BAG_MIN_START_LIKE_CONFIDENCE:.2f}"
                    )
                ] + start_like_reasons
                LOG.warning(
                    "Rejected reviewed bag-start page %s for bag %s in %s: start-like confidence %.3f below %.2f",
                    page.page_number,
                    reviewed_bag,
                    pdf_path.name,
                    start_like_confidence,
                    REVIEWED_BAG_MIN_START_LIKE_CONFIDENCE,
                )
                if debug:
                    _save_candidate_debug(
                        debug_dir / "candidates",
                        page,
                        numbers,
                        "bag_start",
                        False,
                        start_like_confidence,
                        reviewed_reasons,
                        dark_ratio,
                    )
                continue
            reviewed_reasons = [f"manually reviewed bag start for bag {reviewed_bag}"]
            if reviewed_bag not in seen_single_bags:
                seen_single_bags.add(reviewed_bag)
                bag_start_like_pages.append(page.page_number)
                bags.append(
                    BagCandidate(
                        bag=reviewed_bag,
                        start_page=page.page_number,
                        confidence=1.0,
                        page_index=page.page_index,
                        reasons=reviewed_reasons,
                    )
                )
                confirmed_bag_start_review.append(
                    {
                        "bag": reviewed_bag,
                        "page_number": page.page_number,
                        "confidence": 1.0,
                        "detected_numbers": numbers,
                        "reasons": reviewed_reasons,
                        "image_path": page.image_path.as_posix() if page.image_path else None,
                    }
                )
                LOG.info(
                    "Using reviewed bag-start page %s for bag %s in %s",
                    page.page_number,
                    reviewed_bag,
                    pdf_path.name,
                )
            if debug:
                _save_candidate_debug(
                    debug_dir / "candidates",
                    page,
                    numbers,
                    "bag_start",
                    True,
                    1.0,
                    reviewed_reasons,
                    dark_ratio,
                )
            continue

        if numbers:
            reviewed_expected_bags = _load_reviewed_expected_bags(candidate_json_path)
            overview_confidence, overview_reasons = _overview_score(page, numbers, dark_ratio)
            if overview_confidence >= 0.75 and len(numbers) >= 8 and page.image_count >= 40:
                overview_pages.append(page.page_number)
                overview_numbers.update(numbers)
                if reviewed_expected_bags:
                    overview_numbers.update(reviewed_expected_bags)
                    LOG.info(
                        "Applied reviewed overview numbers %s from page %s in %s",
                        reviewed_expected_bags,
                        page.page_number,
                        pdf_path.name,
                    )
                expected_bags = sorted(set(overview_numbers))
                accepted_overviews.append(
                    {
                        "page_number": page.page_number,
                        "confidence": round(overview_confidence, 3),
                        "detected_numbers": numbers,
                        "reviewed_expected_bags": reviewed_expected_bags,
                        "resolved_expected_bags": sorted(set(numbers) | set(reviewed_expected_bags)),
                        "reasons": overview_reasons + ["accepted with tightened overview gate"],
                        "image_path": page.image_path.as_posix() if page.image_path else None,
                    }
                )
                LOG.info(
                    "Accepted overview page %s in %s with confidence %.3f (%s)",
                    page.page_number,
                    pdf_path.name,
                    overview_confidence,
                    "; ".join(overview_reasons),
                )
                if debug:
                    _save_candidate_debug(
                        debug_dir / "candidates",
                        page,
                        numbers,
                        "overview",
                        True,
                        overview_confidence,
                        overview_reasons + ["accepted with tightened overview gate"],
                        dark_ratio,
                    )
                continue
            elif debug and overview_confidence >= 0.45:
                uncertain_pages.append(page.page_number)
                _save_candidate_debug(
                    debug_dir / "candidates",
                    page,
                    numbers,
                    "overview",
                    False,
                    overview_confidence,
                    overview_reasons + ["rejected by tightened overview gate"],
                    dark_ratio,
                )

        if start_like_confidence >= 0.7:
            bag_start_like_pages.append(page.page_number)
            if numbers:
                bag_num, number_reasons = _resolve_bag_number(page, numbers)
                combined_reasons = start_like_reasons + number_reasons
                if bag_num is not None and start_like_confidence >= 0.8 and bag_num not in seen_single_bags:
                    seen_single_bags.add(bag_num)
                    bags.append(
                        BagCandidate(
                            bag=bag_num,
                            start_page=page.page_number,
                            confidence=round(start_like_confidence, 3),
                            page_index=page.page_index,
                            reasons=combined_reasons,
                        )
                    )
                    confirmed_bag_start_review.append(
                        {
                            "bag": bag_num,
                            "page_number": page.page_number,
                            "confidence": round(start_like_confidence, 3),
                            "detected_numbers": numbers,
                            "reasons": combined_reasons,
                            "image_path": page.image_path.as_posix() if page.image_path else None,
                        }
                    )
                    LOG.info(
                        "Accepted numbered bag-start page %s for bag %s in %s with confidence %.3f (%s)",
                        page.page_number,
                        bag_num,
                        pdf_path.name,
                        start_like_confidence,
                        "; ".join(combined_reasons),
                    )
                    if debug:
                        _save_candidate_debug(
                            debug_dir / "candidates",
                            page,
                            numbers,
                            "single-divider",
                            True,
                            start_like_confidence,
                            combined_reasons,
                            dark_ratio,
                        )
                    continue
            LOG.info(
                "Accepted bag-start-like page %s in %s with confidence %.3f (%s)",
                page.page_number,
                pdf_path.name,
                start_like_confidence,
                "; ".join(start_like_reasons),
            )
            if debug:
                _save_candidate_debug(
                    debug_dir / "candidates",
                    page,
                    numbers,
                    "bag-start-like",
                    True,
                    start_like_confidence,
                    start_like_reasons,
                    dark_ratio,
                )
        elif debug and start_like_confidence >= 0.45:
            uncertain_pages.append(page.page_number)
            _save_candidate_debug(
                debug_dir / "candidates",
                page,
                numbers,
                "bag-start-like",
                False,
                start_like_confidence,
                start_like_reasons,
                dark_ratio,
            )

    ordered_bags = sorted(bags, key=lambda bag: (bag.start_page, bag.bag))
    ordered_start_like_pages = sorted(set(bag_start_like_pages))
    deduped_uncertain = sorted(set(uncertain_pages))
    detected_bags = sorted(set(bag.bag for bag in ordered_bags if bag.bag is not None))
    missing_bags = sorted(set(expected_bags) - set(detected_bags))
    bag_count_estimate = len(expected_bags) if expected_bags else max((bag.bag for bag in ordered_bags), default=0)
    missing_bag_review_groups = _build_missing_bag_review_groups(
        missing_bags=missing_bags,
        pages=pages,
        page_numbers=page_numbers,
        page_dark_ratios=page_dark_ratios,
        page_start_like_scores=page_start_like_scores,
        confirmed_bags=ordered_bags,
        overview_pages=overview_pages,
    )
    review_groups = {
        "confirmed_bag_start": sorted(confirmed_bag_start_review, key=lambda item: (item["bag"], item["page_number"])),
        "overview": sorted(accepted_overviews, key=lambda item: item["page_number"]),
        "missing_bag_candidates": missing_bag_review_groups,
    }

    LOG.info(
        "Expected bags: %s | Detected: %s | Missing: %s",
        expected_bags,
        detected_bags,
        missing_bags,
    )
    LOG.info(
        "Prepared review groups in %s: %s confirmed bag starts, %s overview pages, %s missing bag candidate groups",
        pdf_path.name,
        len(review_groups["confirmed_bag_start"]),
        len(review_groups["overview"]),
        len(review_groups["missing_bag_candidates"]),
    )
    LOG.info(
        "Estimated %s total bag(s) in %s from overview pages %s; confirmed start pages: %s; start-like pages: %s; uncertain pages: %s",
        bag_count_estimate,
        pdf_path.name,
        overview_pages,
        [bag.start_page for bag in ordered_bags],
        ordered_start_like_pages,
        deduped_uncertain,
    )
    return PdfBagAnalysis(
        file=pdf_path.as_posix(),
        bag_count_estimate=bag_count_estimate,
        bags=ordered_bags,
        overview_pages=overview_pages,
        bag_start_like_pages=ordered_start_like_pages,
        uncertain_pages=deduped_uncertain,
        expected_bags=expected_bags,
        detected_bags=detected_bags,
        missing_bags=missing_bags,
        review_groups=review_groups,
    )
