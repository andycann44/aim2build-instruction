import re
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pytesseract
from pytesseract import Output

from clean.services import debug_service


def _classify_numeric_token(
    text: str,
    x: int,
    y: int,
    page_width: int,
    page_height: int,
) -> Tuple[bool, Optional[str]]:
    if not re.match(r"^\d+$", text):
        return False, None

    if x < page_width * 0.18 and y > page_height * 0.88:
        return True, "left"
    if x > page_width * 0.82 and y > page_height * 0.88:
        return True, "right"
    return False, None


def _extract_numeric_tokens(
    img,
    page: int,
) -> Tuple[int, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    page_height, page_width = img.shape[:2]
    data = pytesseract.image_to_data(img, config="--psm 6", output_type=Output.DICT)

    numeric_tokens: List[Dict[str, Any]] = []
    page_number_tokens: List[Dict[str, Any]] = []

    token_count = len(data.get("text", []))
    for i in range(token_count):
        text = (data.get("text", [""])[i] or "").strip()
        if not text or not re.match(r"^\d+$", text):
            continue

        try:
            conf = float(data.get("conf", ["-1"])[i])
        except Exception:
            conf = -1.0

        if conf <= 0:
            continue

        x = int(data.get("left", [0])[i] or 0)
        y = int(data.get("top", [0])[i] or 0)
        w = int(data.get("width", [0])[i] or 0)
        h = int(data.get("height", [0])[i] or 0)
        is_page_number, side = _classify_numeric_token(text, x, y, page_width, page_height)

        token: Dict[str, Any] = {
            "text": text,
            "value": int(text),
            "conf": conf,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": x + (w // 2),
            "cy": y + (h // 2),
            "area": w * h,
            "source": "ocr",
            "reasons": ["ocr_numeric_token"],
        }

        if is_page_number:
            token["kind"] = "page_number"
            token["page_number_side"] = side or "unknown"
            token["page_number_match"] = int(text) == int(page)
            page_number_tokens.append(token)
        else:
            token["kind"] = "number"
            numeric_tokens.append(token)

    return page_width, page_height, numeric_tokens, page_number_tokens


def _should_join_number_tokens(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
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
    sorted_tokens = sorted(tokens, key=lambda token: (int(token.get("x", 0) or 0), int(token.get("y", 0) or 0)))
    joined_numbers: List[Dict[str, Any]] = []

    i = 0
    while i < len(sorted_tokens):
        current = sorted_tokens[i]
        group = [current]
        j = i + 1

        while j < len(sorted_tokens):
            candidate = sorted_tokens[j]
            if _should_join_number_tokens(group[-1], candidate):
                group.append(candidate)
                j += 1
                continue
            break

        if len(group) >= 2:
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
            text = "".join(str(item.get("text", "")) for item in group)
            joined_numbers.append(
                {
                    "text": text,
                    "value": int(text),
                    "parts": [str(item.get("text", "")) for item in group],
                    "x": min_x,
                    "y": min_y,
                    "w": width,
                    "h": height,
                    "cx": min_x + (width // 2),
                    "cy": min_y + (height // 2),
                    "area": width * height,
                    "source": "joined_ocr",
                    "score": 0.0,
                    "reasons": ["joined_adjacent_ocr_digits"],
                }
            )
            i = j
            continue

        i += 1

    joined_numbers.sort(key=lambda item: (int(item.get("y", 0) or 0), int(item.get("x", 0) or 0)))
    return joined_numbers


def _run_visual_number_ocr(gray_crop) -> str:
    if gray_crop.size == 0:
        return ""
    _, th = cv2.threshold(gray_crop, 90, 255, cv2.THRESH_BINARY_INV)
    up = cv2.resize(th, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(
        up,
        config="--psm 13 -c tessedit_char_whitelist=0123456789",
    )
    return (text or "").strip()


def _visual_candidates_from_image(img) -> List[Dict[str, Any]]:
    page_height, page_width = img.shape[:2]
    left_limit = int(page_width * 0.22)
    top_limit = int(page_height * 0.88)

    roi = img[0:top_limit, 0:left_limit]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: List[Dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = int(w * h)
        if h < 24 or h > 60:
            continue
        if w < 8 or w > 40:
            continue
        if area < 250 or area > 2000:
            continue
        if x > int(page_width * 0.12):
            continue

        components.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "cx": int(x + (w // 2)),
                "cy": int(y + (h // 2)),
                "area": int(area),
            }
        )

    components.sort(key=lambda item: (int(item["y"]), int(item["x"])))

    groups: List[List[Dict[str, Any]]] = []
    used = [False] * len(components)
    for i, component in enumerate(components):
        if used[i]:
            continue
        group = [component]
        used[i] = True

        for j in range(i + 1, len(components)):
            if used[j]:
                continue
            candidate = components[j]
            if _should_join_number_tokens(group[-1], candidate):
                group.append(candidate)
                used[j] = True
                continue
            break

        groups.append(group)

    visual_candidates: List[Dict[str, Any]] = []
    for group in groups:
        xs = [int(item["x"]) for item in group]
        ys = [int(item["y"]) for item in group]
        rights = [int(item["x"]) + int(item["w"]) for item in group]
        bottoms = [int(item["y"]) + int(item["h"]) for item in group]
        min_x = min(xs)
        min_y = min(ys)
        max_right = max(rights)
        max_bottom = max(bottoms)
        width = max_right - min_x
        height = max_bottom - min_y

        crop = gray[max(0, min_y - 6):min(gray.shape[0], max_bottom + 6), max(0, min_x - 6):min(gray.shape[1], max_right + 6)]
        text = _run_visual_number_ocr(crop)
        if not text or not re.match(r"^\d+$", text):
            continue

        reasons = ["visual_left_side_component_group", "visual_ocr_digits"]
        if len(group) >= 2:
            reasons.append("visual_joined_components")

        visual_candidates.append(
            {
                "text": text,
                "value": int(text),
                "x": int(min_x),
                "y": int(min_y),
                "w": int(width),
                "h": int(height),
                "cx": int(min_x + (width // 2)),
                "cy": int(min_y + (height // 2)),
                "area": int(width * height),
                "source": "visual",
                "score": 0.0,
                "reasons": reasons,
            }
        )

    visual_candidates.sort(key=lambda item: (int(item["y"]), int(item["x"])))
    return visual_candidates


def _candidate_score(
    candidate: Dict[str, Any],
    median_numeric_height: float,
    median_numeric_area: float,
    page_width: int,
    page_height: int,
) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = list(candidate.get("reasons", []))
    height = float(candidate.get("h", 0) or 0)
    area = float(candidate.get("area", 0) or 0)
    x = float(candidate.get("x", 0) or 0)
    y = float(candidate.get("y", 0) or 0)

    if median_numeric_height > 0:
        height_ratio = height / median_numeric_height
        score += min(3.0, height_ratio)
        if height_ratio >= 1.15:
            reasons.append("large_height")

    if median_numeric_area > 0:
        area_ratio = area / median_numeric_area
        score += min(3.0, area_ratio)
        if area_ratio >= 1.4:
            reasons.append("large_area")

    left_bonus = max(0.0, 2.0 - (x / max(1.0, page_width * 0.16)))
    if left_bonus > 0:
        score += left_bonus
        reasons.append("left_side_position")

    upper_bonus = max(0.0, 1.25 - (y / max(1.0, page_height * 0.88)))
    if upper_bonus > 0:
        score += upper_bonus

    if candidate.get("source") == "visual":
        score += 3.0
        reasons.append("visual_primary")
    elif candidate.get("source") == "joined_ocr":
        score += 1.0
    else:
        score -= 0.5

    return round(score, 3), reasons


def _is_valid_step_region(
    candidate: Dict[str, Any],
    page_width: int,
    page_height: int,
    median_numeric_height: float,
    median_numeric_area: float,
) -> bool:
    x = int(candidate.get("x", 0) or 0)
    y = int(candidate.get("y", 0) or 0)
    h = int(candidate.get("h", 0) or 0)
    area = int(candidate.get("area", 0) or 0)
    value = int(candidate.get("value", 0) or 0)

    if value <= 0:
        return False
    if x >= page_width * 0.45:
        return False
    if y >= page_height * 0.88:
        return False

    if candidate.get("source") == "visual":
        return True

    if median_numeric_height > 0 and h >= median_numeric_height * 1.15:
        return True
    if median_numeric_area > 0 and area >= median_numeric_area * 1.4:
        return True
    return False


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = sorted(
        candidates,
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            0 if item.get("source") == "visual" else 1,
            int(item.get("y", 0) or 0),
            int(item.get("x", 0) or 0),
        ),
    )
    kept: List[Dict[str, Any]] = []
    for candidate in candidates:
        duplicate = False
        for existing in kept:
            if abs(int(candidate.get("cy", 0) or 0) - int(existing.get("cy", 0) or 0)) > max(int(candidate.get("h", 0) or 0), int(existing.get("h", 0) or 0)) * 0.8:
                continue
            if abs(int(candidate.get("cx", 0) or 0) - int(existing.get("cx", 0) or 0)) > max(int(candidate.get("w", 0) or 0), int(existing.get("w", 0) or 0)) * 1.2:
                continue
            duplicate = True
            break
        if not duplicate:
            kept.append(candidate)
    kept.sort(key=lambda item: (int(item.get("y", 0) or 0), int(item.get("x", 0) or 0)))
    return kept


def _rank_step_candidate(candidate: Dict[str, Any]) -> Tuple[float, float, int, int]:
    return (
        -float(candidate.get("area", 0) or 0),
        -float(candidate.get("score", 0.0) or 0.0),
        int(candidate.get("y", 0) or 0),
        int(candidate.get("x", 0) or 0),
    )


def _summarize_step_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "value": int(candidate.get("value", 0) or 0),
        "text": str(candidate.get("text", "")),
        "x": int(candidate.get("x", 0) or 0),
        "y": int(candidate.get("y", 0) or 0),
        "w": int(candidate.get("w", 0) or 0),
        "h": int(candidate.get("h", 0) or 0),
        "score": float(candidate.get("score", 0.0) or 0.0),
    }


def _classify_main_and_sub_steps(
    step_candidates: List[Dict[str, Any]],
    page_width: int,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[int],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    if not step_candidates:
        return {}, [], [], [], []

    sorted_candidates = sorted(
        step_candidates,
        key=lambda item: (int(item.get("y", 0) or 0), int(item.get("x", 0) or 0)),
    )
    leftmost_x = min(int(item.get("x", 0) or 0) for item in sorted_candidates)
    left_band_margin = max(60, int(page_width * 0.04))
    left_band = [
        item for item in sorted_candidates
        if int(item.get("x", 0) or 0) <= leftmost_x + left_band_margin
    ]
    if not left_band:
        left_band = list(sorted_candidates)

    band_max_area = max(float(item.get("area", 0) or 0) for item in left_band)
    band_max_score = max(float(item.get("score", 0.0) or 0.0) for item in left_band)

    main_steps_raw: List[Dict[str, Any]] = []
    for item in left_band:
        area = float(item.get("area", 0) or 0)
        score = float(item.get("score", 0.0) or 0.0)
        if area >= band_max_area * 0.55 or score >= band_max_score - 0.75:
            main_steps_raw.append(item)

    if not main_steps_raw:
        main_steps_raw = [sorted(left_band, key=_rank_step_candidate)[0]]

    main_steps_raw = sorted(
        main_steps_raw,
        key=lambda item: (int(item.get("y", 0) or 0), int(item.get("x", 0) or 0)),
    )
    main_step = _summarize_step_candidate(
        sorted(main_steps_raw, key=_rank_step_candidate)[0]
    )
    main_steps = [_summarize_step_candidate(item) for item in main_steps_raw]

    main_ids = {
        (
            int(item.get("x", 0) or 0),
            int(item.get("y", 0) or 0),
            int(item.get("w", 0) or 0),
            int(item.get("h", 0) or 0),
            str(item.get("text", "")),
        )
        for item in main_steps_raw
    }

    sub_step_values: List[int] = []
    sub_step_candidates_raw: List[Dict[str, Any]] = []
    for item in sorted_candidates:
        item_id = (
            int(item.get("x", 0) or 0),
            int(item.get("y", 0) or 0),
            int(item.get("w", 0) or 0),
            int(item.get("h", 0) or 0),
            str(item.get("text", "")),
        )
        if item_id in main_ids:
            continue
        if int(item.get("x", 0) or 0) > int(page_width * 0.35):
            continue
        value = int(item.get("value", 0) or 0)
        if value not in sub_step_values:
            sub_step_values.append(value)
        sub_step_candidates_raw.append(item)

    return main_step, main_steps, sub_step_values, main_steps_raw, sub_step_candidates_raw


def _build_classified_step_boxes(
    step_candidates: List[Dict[str, Any]],
    main_steps_raw: List[Dict[str, Any]],
    sub_step_candidates_raw: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    main_ids = {
        (
            int(item.get("x", 0) or 0),
            int(item.get("y", 0) or 0),
            int(item.get("w", 0) or 0),
            int(item.get("h", 0) or 0),
            str(item.get("text", "")),
        )
        for item in main_steps_raw
    }
    sub_ids = {
        (
            int(item.get("x", 0) or 0),
            int(item.get("y", 0) or 0),
            int(item.get("w", 0) or 0),
            int(item.get("h", 0) or 0),
            str(item.get("text", "")),
        )
        for item in sub_step_candidates_raw
    }

    classified: List[Dict[str, Any]] = []
    for item in step_candidates:
        item_id = (
            int(item.get("x", 0) or 0),
            int(item.get("y", 0) or 0),
            int(item.get("w", 0) or 0),
            int(item.get("h", 0) or 0),
            str(item.get("text", "")),
        )

        step_group = "other_detected_step"
        if item_id in main_ids:
            step_group = "main_steps"
        elif item_id in sub_ids:
            step_group = "sub_steps"

        classified.append(
            {
                "step_number": int(item.get("value", 0) or 0),
                "text": str(item.get("text", "")),
                "box": [
                    int(item.get("x", 0) or 0),
                    int(item.get("y", 0) or 0),
                    int(item.get("w", 0) or 0),
                    int(item.get("h", 0) or 0),
                ],
                "x": int(item.get("x", 0) or 0),
                "y": int(item.get("y", 0) or 0),
                "w": int(item.get("w", 0) or 0),
                "h": int(item.get("h", 0) or 0),
                "confidence": float(item.get("score", 0.0) or 0.0),
                "source": str(item.get("source", "")),
                "step_group": step_group,
                "reasons": list(item.get("reasons", []) or []),
            }
        )

    return classified


def detect_steps(set_num: str, page: int) -> Dict[str, Any]:
    image_path = debug_service.resolve_page_image_path(set_num, page)
    if image_path is None:
        raise RuntimeError("Page image not found")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError("Could not load page image")

    page_width, page_height, numeric_tokens, page_number_tokens = _extract_numeric_tokens(img, page)
    joined_numbers = _build_joined_numbers(numeric_tokens)
    visual_candidates = _visual_candidates_from_image(img)

    size_seed = list(numeric_tokens) + list(visual_candidates)
    height_values = [int(token.get("h", 0) or 0) for token in size_seed if int(token.get("h", 0) or 0) > 0]
    area_values = [int(token.get("area", 0) or 0) for token in size_seed if int(token.get("area", 0) or 0) > 0]
    median_numeric_height = float(median(height_values)) if height_values else 0.0
    median_numeric_area = float(median(area_values)) if area_values else 0.0

    candidates: List[Dict[str, Any]] = []
    for candidate in list(numeric_tokens) + list(joined_numbers) + list(visual_candidates):
        if not _is_valid_step_region(
            candidate,
            page_width,
            page_height,
            median_numeric_height,
            median_numeric_area,
        ):
            continue

        scored = dict(candidate)
        score, reasons = _candidate_score(
            candidate,
            median_numeric_height,
            median_numeric_area,
            page_width,
            page_height,
        )
        scored["score"] = score
        scored["reasons"] = reasons
        candidates.append(scored)

    step_candidates = _dedupe_candidates(candidates)
    main_step, main_steps, sub_steps, main_steps_raw, sub_step_candidates_raw = _classify_main_and_sub_steps(
        step_candidates,
        page_width,
    )
    classified_step_boxes = _build_classified_step_boxes(
        step_candidates,
        main_steps_raw,
        sub_step_candidates_raw,
    )

    return {
        "set_num": str(set_num),
        "page": int(page),
        "page_width": int(page_width),
        "page_height": int(page_height),
        "page_number_tokens": page_number_tokens,
        "step_candidates": step_candidates,
        "classified_step_boxes": classified_step_boxes,
        "main_step": main_step,
        "main_steps": main_steps,
        "sub_steps": sub_steps,
        "numeric_tokens": numeric_tokens,
        "joined_numbers": joined_numbers,
        "visual_candidates": visual_candidates,
    }


def build_step_overlay(set_num: str, page: int) -> bytes:
    image_path = debug_service.resolve_page_image_path(set_num, page)
    if image_path is None:
        raise RuntimeError("Page image not found")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError("Could not load page image")

    result = detect_steps(set_num, page)

    for token in result.get("numeric_tokens", []):
        x = int(token.get("x", 0) or 0)
        y = int(token.get("y", 0) or 0)
        w = int(token.get("w", 0) or 0)
        h = int(token.get("h", 0) or 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 1)

    for token in result.get("page_number_tokens", []):
        x = int(token.get("x", 0) or 0)
        y = int(token.get("y", 0) or 0)
        w = int(token.get("w", 0) or 0)
        h = int(token.get("h", 0) or 0)
        text = str(token.get("text", ""))
        side = str(token.get("page_number_side", "unknown"))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            img,
            "page:%s:%s" % (text, side),
            (x, y - 6 if y > 18 else y + h + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    for item in result.get("joined_numbers", []):
        x = int(item.get("x", 0) or 0)
        y = int(item.get("y", 0) or 0)
        w = int(item.get("w", 0) or 0)
        h = int(item.get("h", 0) or 0)
        text = str(item.get("text", ""))
        cv2.rectangle(img, (x, y), (x + w, y + h), (180, 0, 180), 1)
        cv2.putText(
            img,
            "joined:%s" % text,
            (x, y - 6 if y > 18 else y + h + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 0, 180),
            1,
            cv2.LINE_AA,
        )

    for item in result.get("visual_candidates", []):
        x = int(item.get("x", 0) or 0)
        y = int(item.get("y", 0) or 0)
        w = int(item.get("w", 0) or 0)
        h = int(item.get("h", 0) or 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 255), 1)

    for item in result.get("step_candidates", []):
        x = int(item.get("x", 0) or 0)
        y = int(item.get("y", 0) or 0)
        w = int(item.get("w", 0) or 0)
        h = int(item.get("h", 0) or 0)
        text = str(item.get("text", ""))
        box_color = (0, 255, 255)
        box_thickness = 5
        label_text = "STEP %s" % text
        font_scale = 1.2

        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, box_thickness)
        (label_width, label_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            2,
        )
        label_x = x
        label_bottom_y = y - 8
        if label_bottom_y - label_height - baseline < 0:
            label_bottom_y = y + h + label_height + 8
        label_top_y = label_bottom_y - label_height - baseline
        cv2.rectangle(
            img,
            (label_x - 4, label_top_y - 4),
            (label_x + label_width + 4, label_bottom_y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img,
            label_text,
            (label_x, label_bottom_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            box_color,
            2,
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Could not encode overlay image")
    return encoded.tobytes()
