from pathlib import Path
import re

import cv2
import pytesseract
import numpy as np

from clean.services import debug_service


_ALPHA_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+[A-Za-z]?\b")


def _normalize_text_lines(text: str):
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line:
            lines.append(line)
    return lines


def _count_alpha_words(text: str):
    return len(_ALPHA_WORD_RE.findall(text or ""))


def _count_long_alpha_words(text: str, min_len: int = 5):
    return sum(1 for w in _ALPHA_WORD_RE.findall(text or "") if len(w) >= min_len)


def _count_numeric_tokens(text: str):
    return len(_NUMERIC_TOKEN_RE.findall(text or ""))


def _ocr_page_text(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "ocr_text": "",
            "word_count": 0,
            "line_count": 0,
            "long_word_count": 0,
            "numeric_token_count": 0,
            "ocr_ok": False,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    max_w = 1400
    if w > max_w:
        scale = max_w / float(w)
        gray = cv2.resize(
            gray,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh, config="--psm 6")
    lines = _normalize_text_lines(text)

    return {
        "ocr_text": text,
        "word_count": _count_alpha_words(text),
        "line_count": len(lines),
        "long_word_count": _count_long_alpha_words(text),
        "numeric_token_count": _count_numeric_tokens(text),
        "ocr_ok": True,
    }


def _layout_signals(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "image_ok": False,
            "bright_ratio": 0.0,
            "edge_density": 0.0,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    max_w = 1400
    if w > max_w:
        scale = max_w / float(w)
        gray = cv2.resize(
            gray,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    bright_ratio = float((gray > 210).mean())

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    return {
        "image_ok": True,
        "bright_ratio": bright_ratio,
        "edge_density": edge_density,
    }


def _panel_signals(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "large_box_count": 0,
            "largest_box_area_ratio": 0.0,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    max_w = 1400
    if w > max_w:
        scale = max_w / float(w)
        gray = cv2.resize(
            gray,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 80, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_area = float(h * w)
    large_box_count = 0
    largest_box_area_ratio = 0.0

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = float(bw * bh)
        if area <= 0:
            continue

        area_ratio = area / page_area
        if area_ratio >= 0.03:
            large_box_count += 1

        if area_ratio > largest_box_area_ratio:
            largest_box_area_ratio = area_ratio

    return {
        "large_box_count": int(large_box_count),
        "largest_box_area_ratio": round(largest_box_area_ratio, 4),
    }


def text_heavy_precheck(set_num: str, page: int):
    """
    Conservative skip filter only.

    Returns page_kind in:
    - cover_page
    - intro_or_legal
    - parts_page
    - other
    """
    image_path = debug_service.resolve_page_image_path(set_num, page)
    if image_path is None:
        return {
            "page": int(page),
            "image_found": False,
            "text_heavy_page": False,
            "hard_skip": False,
            "skip_reason": "page_image_not_found",
            "word_count": 0,
            "line_count": 0,
            "long_word_count": 0,
            "bright_ratio": 0.0,
            "edge_density": 0.0,
            "is_cover_page": False,
            "is_likely_build_page": False,
            "is_possible_bag_candidate": False,
            "page_kind": "other",
            "bag_start_score": 0.0,
            "bag_start_label": "disabled",
            "numeric_token_count": 0,
            "large_box_count": 0,
            "largest_box_area_ratio": 0.0,
        }

    ocr = _ocr_page_text(image_path)
    layout = _layout_signals(image_path)
    panel = _panel_signals(image_path)

    word_count = int(ocr["word_count"])
    line_count = int(ocr["line_count"])
    long_word_count = int(ocr["long_word_count"])
    numeric_token_count = int(ocr.get("numeric_token_count", 0))
    bright_ratio = float(layout["bright_ratio"])
    edge_density = float(layout["edge_density"])
    large_box_count = int(panel["large_box_count"])
    largest_box_area_ratio = float(panel["largest_box_area_ratio"])

    is_cover_page = (
        int(page) <= 2
        and word_count <= 20
        and line_count <= 8
        and bright_ratio < 0.30
        and edge_density >= 0.06
    )

    text_heavy_page = (
        word_count >= 35
        or line_count >= 9
        or (word_count >= 24 and line_count >= 6)
        or (word_count >= 22 and long_word_count >= 12)
    )

    is_intro_or_legal_page = (
        not is_cover_page
        and text_heavy_page
        and largest_box_area_ratio < 0.35
    )

    is_parts_page = (
        not is_cover_page
        and not is_intro_or_legal_page
        and largest_box_area_ratio < 0.20
        and (
            numeric_token_count >= 20
            or (numeric_token_count >= 12 and large_box_count >= 8)
            or (numeric_token_count >= 10 and line_count >= 10 and large_box_count >= 6)
        )
    )

    if is_cover_page:
        page_kind = "cover_page"
        hard_skip = True
        skip_reason = "cover_page"
    elif is_intro_or_legal_page:
        page_kind = "intro_or_legal"
        hard_skip = True
        skip_reason = "intro_or_legal"
    elif is_parts_page:
        page_kind = "parts_page"
        hard_skip = True
        skip_reason = "parts_page"
    else:
        page_kind = "other"
        hard_skip = False
        skip_reason = ""

    return {
        "page": int(page),
        "image_found": True,
        "text_heavy_page": bool(text_heavy_page),
        "hard_skip": bool(hard_skip),
        "skip_reason": skip_reason,
        "word_count": word_count,
        "line_count": line_count,
        "long_word_count": long_word_count,
        "bright_ratio": round(bright_ratio, 4),
        "edge_density": round(edge_density, 4),
        "is_cover_page": bool(is_cover_page),
        "is_likely_build_page": False,
        "is_possible_bag_candidate": page_kind == "other",
        "page_kind": page_kind,
        "bag_start_score": 0.0,
        "bag_start_label": "disabled",
        "numeric_token_count": numeric_token_count,
        "large_box_count": large_box_count,
        "largest_box_area_ratio": largest_box_area_ratio,
    }

def get_page_precheck(set_num: str, page: int):
    return text_heavy_precheck(set_num, page)
