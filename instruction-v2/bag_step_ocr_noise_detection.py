"""Reject OCR step detections whose region lacks LEGO step typography."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import pytesseract
from pytesseract import Output

from bag4_misread_step_detection import (
    _box_xywh,
    _load_bag_page_range,
    _load_crop_cache,
    _load_ocr_detections,
)
from paths import ROOT_DIR


PROJECT_ROOT = ROOT_DIR.parent
PAGES_DIR = PROJECT_ROOT / "debug" / "70618" / "70618_01" / "pages"

OCR_NOISE_CLASSIFICATION = "FALSE_STEP"
OCR_NOISE_REASON = "OCR_NOISE"
MIN_TYPOGRAPHY_CONF = 60.0
PAGE_IMAGE_PAD = 8


def _page_image_path(page: int) -> Path:
    return PAGES_DIR / f"page_{int(page):03d}.png"


@lru_cache(maxsize=64)
def _load_page_image(page: int):
    path = _page_image_path(page)
    if not path.is_file():
        return None
    return cv2.imread(str(path))


def _ocr_variant_reads(gray: Any, tight: Any) -> List[Dict[str, Any]]:
    reads: List[Dict[str, Any]] = []
    th, tw = tight.shape[:2]
    gh, gw = gray.shape[:2]
    variants: List[Tuple[str, Any]] = [
        (
            "tesseract_step_box_tight_threshold_6x",
            cv2.resize(
                cv2.threshold(tight, 135, 255, cv2.THRESH_BINARY)[1],
                None,
                fx=6,
                fy=6,
                interpolation=cv2.INTER_CUBIC,
            ),
        ),
        (
            "tesseract_step_box_tight_threshold_invert_6x",
            cv2.resize(
                cv2.threshold(tight, 105, 255, cv2.THRESH_BINARY_INV)[1],
                None,
                fx=6,
                fy=6,
                interpolation=cv2.INTER_CUBIC,
            ),
        ),
        (
            "tesseract_step_box_threshold_5x",
            cv2.resize(
                cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)[1],
                None,
                fx=5,
                fy=5,
                interpolation=cv2.INTER_CUBIC,
            ),
        ),
        (
            "tesseract_step_box_threshold_invert_5x",
            cv2.resize(
                cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY_INV)[1],
                None,
                fx=5,
                fy=5,
                interpolation=cv2.INTER_CUBIC,
            ),
        ),
        (
            "tesseract_step_box_gray_5x",
            cv2.resize(
                cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX),
                None,
                fx=5,
                fy=5,
                interpolation=cv2.INTER_CUBIC,
            ),
        ),
    ]
    if th <= 0 or tw <= 0:
        variants = variants[2:]

    for source, image in variants:
        if image is None or getattr(image, "size", 0) == 0:
            continue
        data = pytesseract.image_to_data(
            image,
            config="--psm 13 -c tessedit_char_whitelist=0123456789",
            output_type=Output.DICT,
        )
        for idx in range(len(data.get("text", []))):
            text = (data["text"][idx] or "").strip()
            if not text or not re.match(r"^\d+$", text):
                continue
            try:
                conf = float(data["conf"][idx])
            except Exception:
                conf = -1.0
            if conf <= 0:
                continue
            reads.append({"text": text, "value": int(text), "conf": conf, "source": source})
    return reads


def region_contains_lego_step_numeral(
    img: Any,
    box: List[int],
    expected_step: int,
) -> Tuple[bool, Dict[str, Any]]:
    if img is None or len(box) < 4:
        return False, {"reason": "missing_image_or_box"}

    x, y, w, h = [int(box[i] or 0) for i in range(4)]
    if w <= 0 or h <= 0:
        return False, {"reason": "empty_box"}

    page_h, page_w = img.shape[:2]
    left = max(0, x - PAGE_IMAGE_PAD)
    top = max(0, y - PAGE_IMAGE_PAD)
    right = min(page_w, x + w + PAGE_IMAGE_PAD)
    bottom = min(page_h, y + h + PAGE_IMAGE_PAD)
    crop = img[top:bottom, left:right]
    if crop.size == 0:
        return False, {"reason": "empty_crop"}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    tight = gray[max(0, y - top) : max(0, y - top) + h, max(0, x - left) : max(0, x - left) + w]
    reads = _ocr_variant_reads(gray, tight)
    expected_text = str(int(expected_step))
    matches = [read for read in reads if read["text"] == expected_text and read["conf"] >= MIN_TYPOGRAPHY_CONF]
    if matches:
        best = max(matches, key=lambda item: float(item["conf"]))
        return True, {
            "reason": "lego_step_typography_confirmed",
            "matching_read": best,
            "all_reads": reads,
        }

    return False, {
        "reason": OCR_NOISE_REASON,
        "expected_step": int(expected_step),
        "all_reads": reads,
    }


def audit_bag_ocr_noise(bag: int = 4) -> Dict[str, Any]:
    page_start, page_end = _load_bag_page_range(bag)
    ocr_by_page = _load_ocr_detections(bag)

    passed_keys: Set[Tuple[int, int]] = set()
    failed_rows: List[Dict[str, Any]] = []
    seen_failures: Set[Tuple[int, int, int, int, int, int]] = set()

    for page in range(page_start, page_end + 1):
        img = _load_page_image(page)
        for entry in ocr_by_page.get(page, []):
            step = int(entry.get("step") or 0)
            box = _box_xywh(entry.get("step_box"))
            if not step or len(box) < 4:
                continue

            ok, evidence = region_contains_lego_step_numeral(img, box, step)
            key = (page, step)
            if ok:
                passed_keys.add(key)
                continue

            dedupe_key = (page, step, box[0], box[1], box[2], box[3])
            if dedupe_key in seen_failures:
                continue
            seen_failures.add(dedupe_key)
            failed_rows.append(
                {
                    "page": page,
                    "ocr_value": step,
                    "classification": OCR_NOISE_CLASSIFICATION,
                    "match_reason": OCR_NOISE_REASON,
                    "source": entry.get("source"),
                    "raw_text": entry.get("raw_text"),
                    "step_box_xywh": box,
                    "typography_evidence": evidence,
                }
            )

    noise_keys = {
        (int(row["page"]), int(row["ocr_value"]))
        for row in failed_rows
        if (int(row["page"]), int(row["ocr_value"])) not in passed_keys
    }
    rows = [
        row
        for row in failed_rows
        if (int(row["page"]), int(row["ocr_value"])) in noise_keys
    ]
    rows.sort(key=lambda row: (int(row["page"]), int(row["ocr_value"]), int(row["step_box_xywh"][1])))

    return {
        "bag": bag,
        "page_start": page_start,
        "page_end": page_end,
        "classification": OCR_NOISE_CLASSIFICATION,
        "match_reason": OCR_NOISE_REASON,
        "row_count": len(rows),
        "ocr_noise_keys": [f"p{page:03d}_s{step}" for page, step in sorted(noise_keys)],
        "rows": rows,
    }


def ocr_noise_step_keys(
    bag: int = 4,
    *,
    ocr_by_page: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> Set[Tuple[int, int]]:
    if ocr_by_page is not None:
        page_start, page_end = _load_bag_page_range(bag)
        passed: Set[Tuple[int, int]] = set()
        failed: Set[Tuple[int, int]] = set()
        for page in range(page_start, page_end + 1):
            img = _load_page_image(page)
            for entry in ocr_by_page.get(page, []):
                step = int(entry.get("step") or 0)
                box = _box_xywh(entry.get("step_box"))
                if not step or len(box) < 4:
                    continue
                ok, _ = region_contains_lego_step_numeral(img, box, step)
                key = (page, step)
                if ok:
                    passed.add(key)
                else:
                    failed.add(key)
        return failed - passed

    payload = audit_bag_ocr_noise(bag)
    keys: Set[Tuple[int, int]] = set()
    for token in payload.get("ocr_noise_keys", []):
        match = re.search(r"p(\d+)_s(\d+)$", str(token))
        if match:
            keys.add((int(match.group(1)), int(match.group(2))))
    return keys


def is_ocr_noise_step(page: int, step: int, bag: int = 4, cache: Optional[Set[Tuple[int, int]]] = None) -> bool:
    keys = cache if cache is not None else ocr_noise_step_keys(bag)
    return (int(page), int(step)) in keys
