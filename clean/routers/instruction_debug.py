import base64
from html import escape
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from clean.routers.debug import (
    _build_material_crop_candidates,
    _contact_sheet_step_boxes_from_detected,
    _encode_contact_sheet_crop,
    _encode_debug_image_data_uri,
    _extract_detected_qty_details_from_crop,
    _extract_qty_tokens_from_image,
    _resolve_bag_page_range,
    _response_text_to_json_debug,
)
from clean.services.azure_openai_service import (
    _crop_ai_model_name as service_crop_ai_model_name,
    _dedupe_candidate_rows as service_dedupe_candidate_rows,
    _require_crop_ai_client as service_require_crop_ai_client,
    rank_crop_candidates,
)
from clean.services.part_candidate_service import get_part_candidates_for_crop
from clean.services import debug_service, step_detector_service
from clean.services.instruction_buildability_source import load_instruction_set_parts

router = APIRouter()


def _coerce_label_filename(set_num: str, bag: int) -> str:
    safe_set = "".join(ch for ch in str(set_num or "").strip() if ch.isalnum() or ch in "-_")
    if not safe_set:
        safe_set = "unknown"
    safe_bag = max(1, int(bag or 1))
    return f"{safe_set}_bag{safe_bag}.json"


def _label_store_path(set_num: str, bag: int) -> Path:
    return Path("/Users/olly/aim2build-instruction/debug/training_labels") / _coerce_label_filename(
        set_num,
        bag,
    )


def _training_export_path(set_num: str, bag: int) -> Path:
    return Path("/Users/olly/aim2build-instruction/debug/training_data") / _coerce_label_filename(
        set_num,
        bag,
    )


def _missing_images_report_path(set_num: str) -> Path:
    safe_set = "".join(ch for ch in str(set_num or "").strip() if ch.isalnum() or ch in "-_") or "unknown"
    return Path("/Users/olly/aim2build-instruction/debug/missing_images") / f"{safe_set}.json"


def _remaining_parts_cache_path(set_num: str, bag: int) -> Path:
    return Path("/Users/olly/aim2build-instruction/debug/remaining_parts") / _coerce_label_filename(
        set_num,
        bag,
    )


def _bag_inspector_db_path() -> Path:
    return Path("/Users/olly/aim2build-instruction/bag_inspector.db")


def _connect_bag_inspector_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_bag_inspector_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_bag_inventory_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bag_review_status (
            set_num TEXT NOT NULL,
            bag_number INTEGER NOT NULL,
            status TEXT NOT NULL,
            reviewed_at TEXT,
            notes TEXT,
            PRIMARY KEY (set_num, bag_number)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS set_bag_parts (
            set_num TEXT NOT NULL,
            bag_number INTEGER NOT NULL,
            part_num TEXT NOT NULL,
            color_id INTEGER NOT NULL,
            qty_total INTEGER NOT NULL,
            verified_examples INTEGER NOT NULL,
            updated_at TEXT,
            PRIMARY KEY (set_num, bag_number, part_num, color_id)
        )
        """
    )
    conn.commit()


def _get_bag_review_status(set_num: str, bag_number: int) -> Dict[str, Any]:
    conn = _connect_bag_inspector_db()
    try:
        _ensure_bag_inventory_tables(conn)
        row = conn.execute(
            """
            SELECT status, reviewed_at, notes
            FROM bag_review_status
            WHERE set_num = ? AND bag_number = ?
            """,
            (str(set_num or "").strip(), int(bag_number or 1)),
        ).fetchone()
        return {
            "set_num": str(set_num or "").strip(),
            "bag_number": int(bag_number or 1),
            "status": str((row["status"] if row else "") or "").strip(),
            "reviewed_at": str((row["reviewed_at"] if row else "") or "").strip(),
            "notes": str((row["notes"] if row else "") or "").strip(),
        }
    finally:
        conn.close()


def _set_bag_review_status(set_num: str, bag_number: int, status: str, notes: Any = None) -> Dict[str, Any]:
    conn = _connect_bag_inspector_db()
    try:
        _ensure_bag_inventory_tables(conn)
        reviewed_at = _iso_now()
        conn.execute(
            """
            INSERT INTO bag_review_status (set_num, bag_number, status, reviewed_at, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(set_num, bag_number) DO UPDATE SET
              status = excluded.status,
              reviewed_at = excluded.reviewed_at,
              notes = excluded.notes
            """,
            (
                str(set_num or "").strip(),
                int(bag_number or 1),
                str(status or "").strip(),
                reviewed_at,
                str(notes or "").strip(),
            ),
        )
        conn.commit()
        return _get_bag_review_status(set_num, bag_number)
    finally:
        conn.close()


def _catalog_db_path() -> Path:
    return Path("/Users/olly/aim2build-instruction/debug/server_catalog/lego_catalog.db")


VALID_CROP_STATUSES = {"good", "bad", "needs_adjust", "hidden"}
VALID_REVIEW_STATUSES = {"unreviewed", "reviewed", "needs_review"}


def _empty_label_store(set_num: str, bag: int) -> Dict[str, Any]:
    return {
        "schema_version": "1.1",
        "set_num": str(set_num or "").strip() or "70618",
        "bag": max(1, int(bag or 1)),
        "created_at": _iso_now(),
        "source": {
            "route": "/debug/instruction-buildability",
            "type": "debug_training_ui",
            "crop_image_path_kind": "page_image_with_crop_box",
        },
        "crops": {},
    }


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_int_list(values: Any) -> List[int]:
    if values in (None, ""):
        source_values: List[Any] = []
    elif isinstance(values, (list, tuple)):
        source_values = list(values)
    else:
        source_values = [values]

    out: List[int] = []
    for value in source_values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _coerce_str_list(values: Any) -> List[str]:
    if values in (None, ""):
        source_values: List[Any] = []
    elif isinstance(values, (list, tuple)):
        source_values = list(values)
    else:
        source_values = [values]

    out: List[str] = []
    for value in source_values:
        text = str(value or "").strip()
        if text:
            out.append(text)
    return out


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_box_list(value: Any) -> Optional[List[int]]:
    if not isinstance(value, (list, tuple)):
        return None
    out: List[int] = []
    for item in list(value)[:4]:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return None
    return out if len(out) == 4 else None


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _normalize_rgb_hex(value: Any) -> Optional[str]:
    text = str(value or "").strip().replace("#", "").replace("0x", "").replace("0X", "").upper()
    if not re.fullmatch(r"[0-9A-F]{6}", text):
        return None
    return text


def _load_catalog_colors_for_ids(color_ids: List[int]) -> List[Dict[str, Any]]:
    normalized_ids = sorted({int(color_id) for color_id in color_ids if _coerce_int(color_id) is not None})
    if not normalized_ids:
        return []

    db_path = _catalog_db_path()
    if not db_path.exists():
        return []

    placeholders = ",".join("?" for _ in normalized_ids)
    db_uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT color_id, name, rgb
            FROM colors
            WHERE color_id IN ({placeholders})
            """,
            normalized_ids,
        ).fetchall()
    finally:
        conn.close()

    lego_colors: List[Dict[str, Any]] = []
    for row in rows:
        color_id = _coerce_int(row["color_id"])
        rgb_hex = _normalize_rgb_hex(row["rgb"])
        if color_id is None or not rgb_hex:
            continue
        lego_colors.append(
            {
                "color_id": int(color_id),
                "color_name": str(row["name"] or f"color {int(color_id)}"),
                "rgb": rgb_hex,
            }
        )
    return lego_colors


def _lookup_element_image_url(part_num: str, color_id: int) -> Optional[str]:
    db_path = _catalog_db_path()
    if not db_path.exists():
        return None
    db_uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT img_url
            FROM element_images
            WHERE part_num = ? AND color_id = ?
            LIMIT 1
            """,
            (str(part_num or "").strip(), int(color_id or 0)),
        ).fetchone()
    finally:
        conn.close()
    img_url = str((row["img_url"] if row else "") or "").strip()
    return img_url or None


def _lookup_color_name_for_part_image(color_id: int) -> str:
    db_path = _catalog_db_path()
    if not db_path.exists():
        return ""
    db_uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT name
            FROM colors
            WHERE color_id = ?
            LIMIT 1
            """,
            (int(color_id or 0),),
        ).fetchone()
    finally:
        conn.close()
    return str((row["name"] if row else "") or "").strip()


def _canonical_element_r2_url(part_num: str, color_id: int) -> str:
    part_key = str(part_num or "").strip()
    return f"https://img.aim2build.co.uk/static/element_images/{part_key}/{part_key}__{int(color_id or 0)}.jpg"


def _extract_qty_from_text(value: Any) -> Optional[int]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return _coerce_int(digits)


def _safe_crop_bounds(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Optional[List[int]]:
    x1 = max(0, min(int(x1), int(width)))
    y1 = max(0, min(int(y1), int(height)))
    x2 = max(0, min(int(x2), int(width)))
    y2 = max(0, min(int(y2), int(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _yellow_ratio_bgr(crop_img: Any) -> float:
    try:
        if crop_img is None or getattr(crop_img, "size", 0) == 0:
            return 0.0
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        mask = (
            (hsv[:, :, 0] >= 15)
            & (hsv[:, :, 0] <= 45)
            & (hsv[:, :, 1] > 35)
            & (hsv[:, :, 2] > 120)
        )
        return float(mask.mean())
    except Exception:
        return 0.0


def _estimate_local_page_bg_colour(roi: Any) -> Dict[str, Any]:
    try:
        if roi is None or getattr(roi, "size", 0) == 0:
            return {"bgr": [0, 0, 0], "hsv": [0, 0, 0]}
        h, w = roi.shape[:2]
        if h <= 0 or w <= 0:
            return {"bgr": [0, 0, 0], "hsv": [0, 0, 0]}
        top_h = max(1, int(round(h * 0.08)))
        side_w = max(1, int(round(w * 0.08)))
        samples = np.concatenate(
            [
                roi[:top_h, :, :].reshape(-1, 3),
                roi[:, :side_w, :].reshape(-1, 3),
                roi[:, max(0, w - side_w):, :].reshape(-1, 3),
            ],
            axis=0,
        )
        samples_hsv = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        keep = (
            (samples_hsv[:, 2] >= 80)
            & ~(
                (samples_hsv[:, 0] >= 15)
                & (samples_hsv[:, 0] <= 45)
                & (samples_hsv[:, 1] > 25)
                & (samples_hsv[:, 2] > 120)
            )
        )
        if np.any(keep):
            bgr_source = samples[keep]
            hsv_source = samples_hsv[keep]
        else:
            bgr_source = roi.reshape(-1, 3)
            hsv_source = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        bgr_med = np.median(bgr_source, axis=0)
        hsv_med = np.median(hsv_source, axis=0)
        return {
            "bgr": [int(v) for v in bgr_med.tolist()],
            "hsv": [int(v) for v in hsv_med.tolist()],
        }
    except Exception:
        return {"bgr": [0, 0, 0], "hsv": [0, 0, 0]}


def _build_callout_fill_mask(roi: Any, bg_colour: Dict[str, Any]) -> Any:
    try:
        if roi is None or getattr(roi, "size", 0) == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bg_bgr = list(bg_colour.get("bgr", [0, 0, 0]) or [0, 0, 0])
        bg_hsv = list(bg_colour.get("hsv", [0, 0, 0]) or [0, 0, 0])
        if len(bg_bgr) < 3:
            bg_bgr = [0, 0, 0]
        if len(bg_hsv) < 3:
            bg_hsv = [0, 0, 0]
        diff = (
            np.abs(roi[:, :, 0].astype(np.int16) - int(bg_bgr[0]))
            + np.abs(roi[:, :, 1].astype(np.int16) - int(bg_bgr[1]))
            + np.abs(roi[:, :, 2].astype(np.int16) - int(bg_bgr[2]))
        )
        strict_mask = (
            (((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 115)) | ((hsv[:, :, 0] >= 80) & (hsv[:, :, 1] < 90)))
            & (hsv[:, :, 1] >= int(bg_hsv[1]) + 8)
            & (hsv[:, :, 2] >= max(0, int(bg_hsv[2]) - 28))
            & (diff > 18)
        ).astype(np.uint8) * 255
        strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        strict_contours, _ = cv2.findContours(strict_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if strict_contours:
            return strict_mask
        blue_mask = (
            (((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 115)) | ((hsv[:, :, 0] >= 80) & (hsv[:, :, 1] < 90)))
            & (hsv[:, :, 1] >= 25)
            & (hsv[:, :, 2] >= 120)
        )
        bg_distance_mask = diff > 55
        final_mask = (blue_mask | bg_distance_mask).astype(np.uint8) * 255
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        return final_mask
    except Exception:
        return np.zeros((0, 0), dtype=np.uint8)


def _detect_callout_rect_by_color_fill(
    img: Any,
    search_box: List[int],
    step_y: int,
    page_width: int,
    page_height: int,
) -> Optional[List[int]]:
    try:
        bounds = _safe_crop_bounds(search_box[0], search_box[1], search_box[2], search_box[3], page_width, page_height)
        if bounds is None:
            return None
        sx1, sy1, sx2, sy2 = bounds
        roi = img[sy1:sy2, sx1:sx2]
        if roi is None or roi.size == 0:
            return None
        bg_colour = _estimate_local_page_bg_colour(roi)
        mask = _build_callout_fill_mask(roi, bg_colour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_cx = int((search_box[0] + search_box[2]) / 2.0)
        best: Optional[List[int]] = None
        best_score = 10**9
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 90 or h <= 35:
                continue
            aspect = float(w) / float(max(h, 1))
            if aspect < 1.2 or aspect > 5.0:
                continue
            if aspect > 4.2:
                continue
            rect_area = float(max(1, w * h))
            fill_ratio = float(cv2.contourArea(contour)) / rect_area
            if fill_ratio <= 0.65:
                continue
            ax = max(0, sx1 + x - 2)
            ay = max(0, sy1 + y - 24)
            bx = min(page_width, sx1 + x + w + 2)
            by = min(page_height, sy1 + y + h + 8)
            by = min(by, int(step_y) - 8)
            if by <= ay:
                continue
            if abs(((ax + bx) // 2) - roi_cx) > 400:
                continue
            crop = img[ay:by, ax:bx]
            if crop is None or crop.size == 0:
                continue
            if _yellow_ratio_bgr(crop) > 0.18:
                continue
            score = abs(int(step_y) - by) - (h * 0.15)
            if score < best_score:
                best_score = score
                best = [ax, ay, bx - ax, by - ay]
        return best
    except Exception:
        return None


def _detect_callout_rect_by_edges(
    img: Any,
    search_box: List[int],
    step_y: int,
    page_width: int,
    page_height: int,
) -> Optional[List[int]]:
    """Find a blue callout rectangle in a local search box using horizontal border lines.

    Returns [x, y, w, h] in page coordinates. This is debug-only and deliberately
    conservative: if unsure it returns None so the existing fallback can run.
    """
    try:
        bounds = _safe_crop_bounds(search_box[0], search_box[1], search_box[2], search_box[3], page_width, page_height)
        if bounds is None:
            return None
        sx1, sy1, sx2, sy2 = bounds
        roi = img[sy1:sy2, sx1:sx2]
        if roi is None or roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 35, 110)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=35,
            minLineLength=70,
            maxLineGap=14,
        )
        if lines is None:
            return None

        groups: List[Dict[str, int]] = []
        for raw in lines:
            x1, y1, x2, y2 = [int(v) for v in raw[0]]
            if abs(y1 - y2) > 5:
                continue
            length = abs(x2 - x1)
            if length < 70:
                continue
            y = int(round((y1 + y2) / 2.0))
            lx = min(x1, x2)
            rx = max(x1, x2)
            matched = False
            for group in groups:
                if abs(group["y"] - y) <= 7:
                    group["y"] = int(round((group["y"] + y) / 2.0))
                    group["x1"] = min(group["x1"], lx)
                    group["x2"] = max(group["x2"], rx)
                    matched = True
                    break
            if not matched:
                groups.append({"y": y, "x1": lx, "x2": rx})

        groups.sort(key=lambda item: item["y"])
        best: Optional[List[int]] = None
        best_score = 10**9
        for top_idx, top in enumerate(groups):
            for bottom in groups[top_idx + 1 :]:
                box_h = int(bottom["y"] - top["y"])
                if box_h < 38 or box_h > 210:
                    continue
                left = max(0, min(top["x1"], bottom["x1"]) - 8)
                right = min(roi.shape[1], max(top["x2"], bottom["x2"]) + 8)
                box_w = int(right - left)
                if box_w < 110 or box_w > 620:
                    continue
                top_y = max(0, int(top["y"]) - 20)
                bottom_y = min(roi.shape[0], int(bottom["y"]) + 8)
                ax = int(sx1 + left)
                ay = int(sy1 + top_y)
                bx = int(sx1 + right)
                by = int(sy1 + bottom_y)
                if by > int(step_y) - 5:
                    by = int(step_y) - 5
                if bx <= ax or by <= ay:
                    continue
                crop = img[ay:by, ax:bx]
                if crop is None or crop.size == 0:
                    continue
                if _yellow_ratio_bgr(crop) > 0.18:
                    continue
                score = abs(int(step_y) - by) + abs((ax + bx) // 2 - (search_box[0] + search_box[2]) // 2) // 5
                if score < best_score:
                    best_score = score
                    best = [ax, ay, bx - ax, by - ay]
        return best
    except Exception:
        return None


def _estimate_visible_part_count_from_crop(crop_img: Any) -> int:
    """Estimate visible part count from contrast against callout background.

    This is intentionally conservative. It is used only to create slots when OCR
    found no qty labels at all, avoiding the previous bad behaviour of inventing
    repeated qtys from crop width.
    """
    try:
        if crop_img is None or getattr(crop_img, "size", 0) == 0:
            return 0
        h, w = crop_img.shape[:2]
        if h <= 0 or w <= 0:
            return 0
        # Estimate pale callout background from border pixels, then find pixels far from it.
        border = np.concatenate([
            crop_img[: max(1, h // 12), :, :].reshape(-1, 3),
            crop_img[max(0, h - max(1, h // 12)) :, :, :].reshape(-1, 3),
            crop_img[:, : max(1, w // 12), :].reshape(-1, 3),
            crop_img[:, max(0, w - max(1, w // 12)) :, :].reshape(-1, 3),
        ], axis=0)
        bg = np.median(border.astype(np.float32), axis=0)
        diff = np.linalg.norm(crop_img.astype(np.float32) - bg.reshape(1, 1, 3), axis=2)
        mask = (diff > 30).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float(max(1, h * w))
        count = 0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 55 or area > roi_area * 0.22:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 7 or bh < 7:
                continue
            if bw > w * 0.80 and bh < 18:
                continue
            if bh > h * 0.80 and bw < 18:
                continue
            count += 1
        return max(0, min(count, 12))
    except Exception:
        return 0


def _collect_qty_token_texts_from_crop(crop_img: Any) -> List[str]:
    height, width = crop_img.shape[:2] if crop_img is not None and getattr(crop_img, "size", 0) != 0 else (0, 0)
    seen: set[tuple[str, int, int]] = set()
    items: List[tuple[int, int, str]] = []
    for ox, oy, region_img in [
        (0, 0, crop_img),
        (max(0, width // 2), 0, crop_img[:, max(0, width // 2): width] if width > 0 else None),
        (0, max(0, height // 2), crop_img[max(0, height // 2): height, :] if height > 0 else None),
        (max(0, width // 2), max(0, height // 3), crop_img[max(0, height // 3): height, max(0, width // 2): width] if width > 0 and height > 0 else None),
        (max(0, int(width * 0.65)), 0, crop_img[:, max(0, int(width * 0.65)): width] if width > 0 else None),
    ]:
        for token in (_extract_qty_tokens_from_image(region_img) or []):
            text = re.sub(r"\s+", "", str(token.get("text", "") or "").lower())
            if not re.match(r"^(?:\d{1,2}x|x\d{1,2})$", text):
                continue
            key = (text, int((int(token.get("cx", 0) or 0) + ox) // 6), int((int(token.get("cy", 0) or 0) + oy) // 6))
            if key in seen:
                continue
            seen.add(key)
            abs_x = int(token.get("x", 0) or 0) + ox
            abs_y = int(token.get("y", 0) or 0) + oy
            items.append((abs_y, abs_x, text))
    items.sort(key=lambda item: (int(item[0] // 12), item[1]))
    return [text for _, _, text in items]


def _auto_qty_payload_for_crop(crop_img: Any, step_number: int) -> Dict[str, List[Any]]:
    """Extract qty text for auto crops and filter step-number contamination.

    If OCR completely fails, create a conservative default slot per visible part.
    """
    texts = _collect_qty_token_texts_from_crop(crop_img)
    nums: List[int] = [int(re.search(r"(\d{1,2})", text).group(1)) for text in texts if re.search(r"(\d{1,2})", text)]
    if not texts:
        payload = _extract_detected_qty_details_from_crop(crop_img)
        texts = _coerce_str_list(payload.get("detected_qty_text", []))
        nums = [int(re.search(r"(\d{1,2})", text).group(1)) for text in texts if re.search(r"(\d{1,2})", text)]
    clean_texts: List[str] = []
    clean_nums: List[int] = []
    for text, num in zip(texts, nums):
        if step_number and int(num) == int(step_number):
            continue
        clean_texts.append(str(text))
        clean_nums.append(int(num))

    if clean_nums:
        return {"detected_qty_text": clean_texts, "detected_qty_numbers": clean_nums, "qty_raw_tokens": texts, "qty_final_tokens": clean_texts, "qty_source_reason": "local_ocr_tokens"}

    count = _estimate_visible_part_count_from_crop(crop_img)
    if count <= 0:
        return {"detected_qty_text": [], "detected_qty_numbers": [], "qty_raw_tokens": texts, "qty_final_tokens": [], "qty_source_reason": "no_qty_detected"}
    return {
        "detected_qty_text": ["1x"] * count,
        "detected_qty_numbers": [1] * count,
        "qty_raw_tokens": texts,
        "qty_final_tokens": ["1x"] * count,
        "qty_source_reason": "visible_part_count_fallback",
    }


def _require_crop_ai_client() -> Any:
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        from openai import AzureOpenAI

        return AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        )
    return _require_openai_vision_client_debug()


def _crop_ai_model_name() -> str:
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return os.environ["AZURE_OPENAI_DEPLOYMENT"]
    return os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")


def _azure_ai_rank_available() -> bool:
    return bool(
        os.getenv("AZURE_OPENAI_ENDPOINT")
        and os.getenv("AZURE_OPENAI_API_KEY")
        and os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )


def _ai_rank_is_ambiguous(candidates: List[Dict[str, Any]]) -> bool:
    rows = list(candidates or [])
    if len(rows) < 2:
        return False
    first = _coerce_float(rows[0].get("score"))
    second = _coerce_float(rows[1].get("score"))
    if first is None or second is None:
        return False
    return abs(first - second) <= 5.0


def _ai_rank_is_low_confidence(candidates: List[Dict[str, Any]]) -> bool:
    rows = list(candidates or [])
    if not rows:
        return False
    top_score = _coerce_float(rows[0].get("score"))
    return top_score is not None and top_score >= 25.0


def _ai_rank_skip_reason(crop: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Optional[str]:
    status = str(crop.get("status") or "").strip().lower()
    review_status = str(crop.get("review_status") or "").strip().lower()
    slot_filled = int(crop.get("slot_filled", 0) or 0)
    slot_total = int(crop.get("slot_total", 0) or 0)
    reviewed_good = status == "good" and review_status == "reviewed"
    if reviewed_good and slot_total > 0 and slot_filled >= slot_total:
        return "reviewed_good_complete"
    if status == "needs_adjust":
        return None
    if slot_total > 0 and slot_filled < slot_total:
        return None
    if _ai_rank_is_ambiguous(candidates):
        return None
    if _ai_rank_is_low_confidence(candidates):
        return None
    return "ai_not_needed"


def _input_image_url_for_debug_ref(image_ref: Any) -> Optional[str]:
    text = str(image_ref or "").strip()
    if not text:
        return None
    if text.startswith(("http://", "https://", "data:image/")):
        return text
    path = Path(text)
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    if img is None or getattr(img, "size", 0) == 0:
        return None
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _response_output_text_debug(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
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
                    return str(content.get("text") or "")
    return ""


def _dedupe_candidate_rows(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for candidate in list(candidates or []):
        key = (
            str(candidate.get("part_num") or "").strip(),
            int(candidate.get("color_id", 0) or 0),
            str(candidate.get("element_id") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _resolve_crop_image_for_ai_rank(crop: Dict[str, Any]) -> Dict[str, Any]:
    data_uri = str(crop.get("data_uri") or "").strip()
    if data_uri.startswith("data:image/"):
        return {
            "image_url": data_uri,
            "source": "crop_data_uri",
            "exists": True,
        }
    crop_image_path = str(crop.get("crop_image_path") or "").strip()
    crop_box = _coerce_box_list(crop.get("crop_box"))
    if not crop_image_path:
        return {
            "image_url": None,
            "source": "missing",
            "exists": False,
        }
    path = Path(crop_image_path)
    if not path.exists():
        return {
            "image_url": None,
            "source": "crop_image_path_missing",
            "exists": False,
        }
    img = cv2.imread(str(path))
    if img is None or getattr(img, "size", 0) == 0:
        return {
            "image_url": None,
            "source": "crop_image_path_unreadable",
            "exists": False,
        }
    cropped = img
    source = "crop_image_path_full"
    if crop_box is not None:
        x, y, w, h = crop_box
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))
        x2 = min(img.shape[1], x + w)
        y2 = min(img.shape[0], y + h)
        if x < x2 and y < y2:
            cropped = img[y:y2, x:x2]
            source = "crop_image_path_xywh"
    encoded = _encode_debug_image_data_uri(cropped)
    return {
        "image_url": str(encoded or "") or None,
        "source": source,
        "exists": bool(encoded),
    }


def _azure_ai_rank_top_candidates_for_crop(
    crop: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    deduped_candidates = service_dedupe_candidate_rows(list(candidates or []))
    crop_image_info = _resolve_crop_image_for_ai_rank(crop)
    candidate_img_urls = [str(candidate.get("img_url") or "").strip() for candidate in deduped_candidates if str(candidate.get("img_url") or "").strip()]
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    skip_reason = _ai_rank_skip_reason(crop, deduped_candidates)
    result: Dict[str, Any] = {
        "enabled": False,
        "best_part_num": None,
        "best_color_id": None,
        "best_element_id": None,
        "confidence": None,
        "reason": "",
        "matched_index": None,
        "raw_response_text": "",
        "parsed_response_json": None,
        "crop_image_included": False,
        "candidate_images_included_count": 0,
        "candidate_rows_preview": [],
        "prompt_text": "",
        "crop_image_source": str(crop_image_info.get("source") or "missing"),
        "crop_image_exists": bool(crop_image_info.get("exists")),
        "candidate_img_urls_count": len(candidate_img_urls),
        "first_candidate_img_url": candidate_img_urls[0] if candidate_img_urls else "",
        "provider_selected": "azure" if azure_endpoint else "none",
        "azure_endpoint_present": bool(azure_endpoint),
        "azure_api_key_present": bool(azure_api_key),
        "azure_deployment_present": bool(azure_deployment),
        "azure_api_version": str(azure_api_version),
        "client_created": False,
        "exception_type": "",
        "exception_message": "",
        "ai_skipped_reason": str(skip_reason or ""),
    }
    if skip_reason:
        result["reason"] = f"AI skipped: {skip_reason}"
        return result
    crop_image_url = _input_image_url_for_debug_ref(crop_image_info.get("image_url"))
    if not crop_image_url:
        stats["failures"] = int(stats.get("failures", 0) or 0) + 1
        result["reason"] = "Crop image missing."
        return result
    result["crop_image_included"] = True
    if not _azure_ai_rank_available():
        stats["disabled"] = int(stats.get("disabled", 0) or 0) + 1
        result["reason"] = "Azure AI rank unavailable."
        return result

    candidate_rows: List[Dict[str, Any]] = []
    prompt_text = (
        "Choose the best LEGO part candidate for this callout crop. "
        "Use only the provided crop image and up to 5 candidate element images. "
        "Return JSON only with best_part_num, best_color_id, best_element_id, confidence, reason. "
        "Do not invent candidates outside the provided list."
    )
    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": prompt_text,
        },
        {"type": "input_image", "image_url": crop_image_url, "detail": "high"},
    ]
    result["prompt_text"] = prompt_text
    for idx, candidate in enumerate(deduped_candidates[:5], start=1):
        image_url = _input_image_url_for_debug_ref(candidate.get("img_url"))
        if not image_url:
            continue
        row = {
            "index": idx,
            "part_num": str(candidate.get("part_num") or "").strip(),
            "color_id": int(candidate.get("color_id", 0) or 0),
            "element_id": str(candidate.get("element_id") or "").strip() or None,
            "candidate_source": str(candidate.get("candidate_source") or ""),
            "score": float(candidate.get("score", 0.0) or 0.0),
            "img_url": str(candidate.get("img_url") or ""),
        }
        candidate_rows.append(row)
        content.append({"type": "input_text", "text": f"Candidate {idx}: {json.dumps(row)}"})
        content.append({"type": "input_image", "image_url": image_url, "detail": "high"})
    result["candidate_rows_preview"] = list(candidate_rows)
    result["candidate_images_included_count"] = len(candidate_rows)
    result["candidate_img_urls_count"] = len(candidate_rows)
    result["first_candidate_img_url"] = str(candidate_rows[0].get("img_url") or "") if candidate_rows else ""
    if not candidate_rows:
        stats["failures"] = int(stats.get("failures", 0) or 0) + 1
        result["reason"] = "No candidate images available."
        return result
    if not _azure_ai_rank_available():
        stats["disabled"] = int(stats.get("disabled", 0) or 0) + 1
        missing_bits: List[str] = []
        if not azure_endpoint:
            missing_bits.append("AZURE_OPENAI_ENDPOINT missing")
        if not azure_api_key:
            missing_bits.append("AZURE_OPENAI_API_KEY missing")
        if not azure_deployment:
            missing_bits.append("AZURE_OPENAI_DEPLOYMENT missing")
        result["reason"] = "; ".join(missing_bits) or "Azure AI rank unavailable."
        return result

    try:
        client = service_require_crop_ai_client()
        result["client_created"] = True
        stats["calls"] = int(stats.get("calls", 0) or 0) + 1
        stats["images_sent"] = int(stats.get("images_sent", 0) or 0) + 1 + len(candidate_rows)
        response = client.responses.create(
            model=service_crop_ai_model_name(),
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "lego_candidate_rank",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "best_part_num": {"type": "string"},
                            "best_color_id": {"type": "integer"},
                            "best_element_id": {"type": ["string", "null"]},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["best_part_num", "best_color_id", "best_element_id", "confidence", "reason"],
                    },
                }
            },
        )
        usage = getattr(response, "usage", None)
        stats["input_tokens"] = int(stats.get("input_tokens", 0) or 0) + int(getattr(usage, "input_tokens", 0) or 0)
        stats["output_tokens"] = int(stats.get("output_tokens", 0) or 0) + int(getattr(usage, "output_tokens", 0) or 0)
        raw_text = _response_output_text_debug(response)
        result["raw_response_text"] = raw_text
        payload = {}
        try:
            payload = _response_text_to_json_debug(response) or {}
        except Exception:
            payload = {}
        result["parsed_response_json"] = payload or None
        if not payload and raw_text:
            lowered_raw = raw_text.lower()
            for row in candidate_rows:
                part_num = str(row.get("part_num") or "").strip()
                if part_num and part_num.lower() in lowered_raw:
                    payload = {
                        "best_part_num": part_num,
                        "best_color_id": int(row.get("color_id", 0) or 0),
                        "best_element_id": row.get("element_id"),
                        "confidence": 0.5,
                        "reason": "Parsed from raw AI text.",
                    }
                    result["parsed_response_json"] = dict(payload)
                    break
        result.update(
            {
                "enabled": True,
                "best_part_num": str(payload.get("best_part_num") or "").strip() or None,
                "best_color_id": _coerce_int(payload.get("best_color_id")),
                "best_element_id": str(payload.get("best_element_id") or "").strip() or None,
                "confidence": _coerce_float(payload.get("confidence")),
                "reason": str(payload.get("reason") or "").strip() or (raw_text[:300] if raw_text else ""),
            }
        )
        if result["best_part_num"] and result["confidence"] is None:
            result["confidence"] = 0.5
        for idx, candidate in enumerate(deduped_candidates):
            if str(candidate.get("part_num") or "").strip() != str(result.get("best_part_num") or "").strip():
                continue
            if int(candidate.get("color_id", 0) or 0) != int(result.get("best_color_id", 0) or 0):
                continue
            best_element_id = str(result.get("best_element_id") or "").strip()
            candidate_element_id = str(candidate.get("element_id") or "").strip()
            if best_element_id and candidate_element_id and best_element_id != candidate_element_id:
                continue
            result["matched_index"] = idx
            break
        if not result.get("best_part_num") and raw_text and not result.get("reason"):
            result["reason"] = raw_text[:300]
        return result
    except Exception as exc:
        stats["failures"] = int(stats.get("failures", 0) or 0) + 1
        result["exception_type"] = exc.__class__.__name__
        result["exception_message"] = str(exc)
        result["reason"] = f"Azure rank failed: {str(exc)}"
        return result


def _ai_qty_payload_for_crop(
    crop_img: Any,
    set_num: str,
    page: int,
    step_number: int,
) -> Optional[Dict[str, List[Any]]]:
    try:
        trim_issue = ""
        trimmed_crop = None
        if crop_img is not None and getattr(crop_img, "size", 0) != 0:
            height, width = crop_img.shape[:2]
            if height > 0 and width > 0:
                hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                yellow_mask = (
                    (hsv[:, :, 0] >= 15)
                    & (hsv[:, :, 0] <= 45)
                    & (hsv[:, :, 1] > 25)
                    & (hsv[:, :, 2] > 120)
                )
                col_ratio = yellow_mask.mean(axis=0)
                band_cols = np.where(col_ratio > 0.20)[0]
                if len(band_cols) > 0:
                    band_start = int(band_cols[0])
                    if band_start >= int(width * 0.40):
                        trim_x = max(1, band_start - 4)
                        if trim_x < width:
                            trimmed_crop = crop_img[:, :trim_x]
                            trim_issue = "includes yellow substep panel"
    except Exception:
        trimmed_crop = None
        trim_issue = ""

    try:
        client = service_require_crop_ai_client()
    except Exception as exc:
        return {
            "detected_qty_text": [],
            "detected_qty_numbers": [],
            "qty_source": "needs_adjust",
            "ai_part_count": None,
            "ai_issues": [f"OpenAI unavailable: {str(exc)}"],
        }

    def _issue_list(result: Any) -> List[str]:
        return [str(value) for value in list((result or {}).get("issues", []) or []) if str(value or "").strip()]

    def _needs_adjust_result(result: Any) -> bool:
        issues_text = " ".join(_issue_list(result)).lower()
        return (
            bool(result is None)
            or result.get("box_ok") is not True
            or "yellow" in issues_text
            or "substep" in issues_text
            or "too loose" in issues_text
        )

    ai_result: Optional[Dict[str, Any]] = None
    try:
        ok, buf = cv2.imencode(".png", crop_img)
        if not ok:
            raise RuntimeError("Could not encode crop image")
        response = client.responses.create(
                model=service_crop_ai_model_name(),
            input=[{"role": "user", "content": [{"type": "input_text", "text": ("Check this LEGO callout crop. Return JSON only with box_ok, part_count, qty_labels, crop_box, issues. If the crop is too loose, crop_box must be the correct blue callout box inside this image, normalized 0..1, excluding yellow substep panels and the big step number. Set number: %s. Page: %d. Step: %d.") % (set_num, int(page), int(step_number))}, {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii"), "detail": "high"}]}],
            text={"format": {"type": "json_schema", "name": "lego_callout_crop_debug", "strict": True, "schema": {"type": "object", "additionalProperties": False, "properties": {"box_ok": {"type": "boolean"}, "part_count": {"type": "integer"}, "qty_labels": {"type": "array", "items": {"type": "string"}}, "crop_box": {"type": ["object", "null"], "additionalProperties": False, "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "w": {"type": "number"}, "h": {"type": "number"}}, "required": ["x", "y", "w", "h"]}, "issues": {"type": "array", "items": {"type": "string"}}}, "required": ["box_ok", "part_count", "qty_labels", "crop_box", "issues"]}}},
        )
        ai_result = _response_text_to_json_debug(response)
    except Exception as exc:
        ai_result = {
            "box_ok": False,
            "part_count": None,
            "qty_labels": [],
            "crop_box": None,
            "issues": [f"OpenAI failed: {str(exc)}"],
        }

    if trimmed_crop is not None and _needs_adjust_result(ai_result):
        try:
            ok, buf = cv2.imencode(".png", trimmed_crop)
            if not ok:
                raise RuntimeError("Could not encode crop image")
            response = client.responses.create(
                model=service_crop_ai_model_name(),
                input=[{"role": "user", "content": [{"type": "input_text", "text": ("Check this LEGO callout crop. Return JSON only with box_ok, part_count, qty_labels, crop_box, issues. If the crop is too loose, crop_box must be the correct blue callout box inside this image, normalized 0..1, excluding yellow substep panels and the big step number. Set number: %s. Page: %d. Step: %d.") % (set_num, int(page), int(step_number))}, {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii"), "detail": "high"}]}],
                text={"format": {"type": "json_schema", "name": "lego_callout_crop_debug", "strict": True, "schema": {"type": "object", "additionalProperties": False, "properties": {"box_ok": {"type": "boolean"}, "part_count": {"type": "integer"}, "qty_labels": {"type": "array", "items": {"type": "string"}}, "crop_box": {"type": ["object", "null"], "additionalProperties": False, "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "w": {"type": "number"}, "h": {"type": "number"}}, "required": ["x", "y", "w", "h"]}, "issues": {"type": "array", "items": {"type": "string"}}}, "required": ["box_ok", "part_count", "qty_labels", "crop_box", "issues"]}}},
            )
            ai_result = _response_text_to_json_debug(response)
        except Exception as exc:
            ai_result = {
                "box_ok": False,
                "part_count": None,
                "qty_labels": [],
                "crop_box": None,
                "issues": [trim_issue or "includes yellow substep panel", f"OpenAI failed: {str(exc)}"],
            }

    if _needs_adjust_result(ai_result):
        issues = _issue_list(ai_result)
        if trim_issue and not any(trim_issue.lower() in issue.lower() for issue in issues):
            issues.append(trim_issue)
        return {
            "detected_qty_text": [],
            "detected_qty_numbers": [],
            "qty_source": "needs_adjust",
            "ai_part_count": ai_result.get("part_count"),
            "ai_issues": issues or ["needs_adjust"],
            "ai_crop_box": ai_result.get("crop_box"),
            "ai_suggested_fix": bool(ai_result.get("crop_box")),
        }

    local_qty_labels = _collect_qty_token_texts_from_crop(crop_img)
    qty_labels = [str(value) for value in list(ai_result.get("qty_labels", []) or []) if str(value or "").strip()]
    ai_part_count = int(ai_result.get("part_count", 0) or 0)
    qty_source_reason = "ai_qty_labels"
    final_labels = list(local_qty_labels) if local_qty_labels else list(qty_labels)
    non_one_local = [label for label in local_qty_labels if label != "1x"]
    if ai_part_count > 0 and len(final_labels) < ai_part_count:
        if local_qty_labels and any(label == "1x" for label in local_qty_labels) and len(non_one_local) <= 1:
            final_labels = ["1x"] * ai_part_count
            qty_source_reason = "expanded_repeated_1x_from_ai_part_count"
        else:
            final_labels = list(final_labels) + (["1x"] * (ai_part_count - len(final_labels)))
            qty_source_reason = "padded_missing_slots_to_ai_part_count"
    elif local_qty_labels:
        qty_source_reason = "local_ocr_tokens_preferred"
    qty_numbers: List[int] = []
    for label in final_labels:
        match = re.match(r"^\s*(\d+)\s*x\s*$", str(label), flags=re.IGNORECASE)
        if match:
            qty_numbers.append(int(match.group(1)))

    return {
        "detected_qty_text": final_labels,
        "detected_qty_numbers": qty_numbers,
        "qty_source": "openai",
        "ai_part_count": ai_part_count,
        "ai_issues": _issue_list(ai_result),
        "ai_crop_box": ai_result.get("crop_box"),
        "ai_suggested_fix": False,
        "qty_raw_tokens": local_qty_labels or qty_labels,
        "qty_final_tokens": final_labels,
        "qty_source_reason": qty_source_reason,
    }


def _build_qty_sequence(qty_values: Any, qty_text_values: Any) -> List[Dict[str, Any]]:
    qty_list = _coerce_int_list(qty_values)
    qty_text_list = _coerce_str_list(qty_text_values)
    sequence: List[Dict[str, Any]] = []

    if qty_text_list:
        for index, qty_text in enumerate(qty_text_list):
            parsed_qty = _extract_qty_from_text(qty_text)
            fallback_qty = qty_list[index] if index < len(qty_list) else None
            sequence.append(
                {
                    "qty": parsed_qty if parsed_qty is not None else fallback_qty,
                    "qty_text": qty_text,
                }
            )
        return sequence

    for qty in qty_list:
        sequence.append({"qty": qty, "qty_text": f"{qty}x"})
    return sequence


def _qty_slot_signature(qty_value: Any, qty_text_value: Any) -> str:
    qty_text = _coerce_str(qty_text_value)
    if qty_text:
        return f"text:{qty_text.lower()}"
    qty = _coerce_int(qty_value)
    return f"qty:{qty}" if qty is not None else ""


def _crop_qty_slot_state(crop_record: Dict[str, Any], qty_values: Any, qty_text_values: Any) -> Dict[str, Any]:
    sequence = _build_qty_sequence(qty_values, qty_text_values)
    parts = list(crop_record.get("parts", []) or [])
    if not sequence:
        return {
            "sequence": [],
            "total_slots": 0,
            "filled_slots": len(parts),
            "slots_full": False,
            "no_qty_detected": True,
            "next_slot": {"qty": 1, "qty_text": "1x"},
            "next_qty_index": 0,
            "next_qty_label": "1x",
        }

    assigned_counts: Dict[str, int] = {}
    for part in parts:
        signature = _qty_slot_signature(part.get("qty"), part.get("qty_text"))
        if not signature:
            continue
        assigned_counts[signature] = assigned_counts.get(signature, 0) + 1

    consumed_counts: Dict[str, int] = {}
    filled_slots = 0
    next_slot: Optional[Dict[str, Any]] = None
    next_qty_index = len(sequence)
    for slot_index, slot in enumerate(sequence):
        signature = _qty_slot_signature(slot.get("qty"), slot.get("qty_text"))
        if signature and assigned_counts.get(signature, 0) > consumed_counts.get(signature, 0):
            consumed_counts[signature] = consumed_counts.get(signature, 0) + 1
            filled_slots += 1
            continue
        next_slot = dict(slot)
        next_qty_index = slot_index
        break

    return {
        "sequence": sequence,
        "total_slots": len(sequence),
        "filled_slots": filled_slots,
        "slots_full": next_slot is None,
        "no_qty_detected": False,
        "next_slot": next_slot,
        "next_qty_index": next_qty_index,
        "next_qty_label": (
            str(next_slot.get("qty_text") or next_slot.get("qty") or "none")
            if next_slot is not None
            else "filled"
        ),
    }


def _refresh_crop_next_qty_index(crop_record: Dict[str, Any]) -> Dict[str, Any]:
    slot_state = _crop_qty_slot_state(
        crop_record,
        crop_record.get("qty", []),
        crop_record.get("qty_text", []),
    )
    crop_record["next_qty_index"] = int(slot_state.get("next_qty_index", 0) or 0)
    return slot_state


def _pick_qty_assignment(
    crop_record: Dict[str, Any],
    qty_values: Any,
    qty_text_values: Any,
    allow_extra_part: bool = False,
) -> Dict[str, Any]:
    slot_state = _crop_qty_slot_state(crop_record, qty_values, qty_text_values)
    if slot_state["no_qty_detected"]:
        return {
            "qty": 1,
            "qty_text": "1x",
            "next_qty_index": int(crop_record.get("next_qty_index", 0) or 0),
            "slots_full": False,
            "no_qty_detected": True,
        }

    next_slot = slot_state.get("next_slot")
    if next_slot is not None:
        return {
            "qty": next_slot.get("qty"),
            "qty_text": next_slot.get("qty_text"),
            "next_qty_index": int(slot_state.get("next_qty_index", 0) or 0) + 1,
            "slots_full": False,
            "no_qty_detected": False,
        }

    if allow_extra_part:
        return {
            "qty": 1,
            "qty_text": "1x",
            "next_qty_index": int(slot_state.get("next_qty_index", 0) or 0),
            "slots_full": False,
            "no_qty_detected": False,
            "extra_part": True,
        }

    return {
        "qty": None,
        "qty_text": None,
        "next_qty_index": int(slot_state.get("next_qty_index", 0) or 0),
        "slots_full": True,
        "no_qty_detected": False,
    }


def _manual_crop_id(page: int, step: int, serial: int) -> str:
    return f"manual_p{int(page)}_s{max(int(step), 0)}_c{max(int(serial), 1)}"


def _is_manual_crop_id(crop_id: Any) -> bool:
    return str(crop_id or "").strip().startswith("manual_")


def _parse_qty_text_input(value: Any) -> Dict[str, List[Any]]:
    raw_text = str(value or "").strip()
    if not raw_text:
        return {"qty_text": [], "qty": []}

    tokens = [
        token.strip()
        for token in re.split(r"[\s,;]+", raw_text.replace("\n", " ").replace("\r", " "))
        if token.strip()
    ]
    qty_text: List[str] = []
    qty_numbers: List[int] = []
    for token in tokens:
        qty_value = _extract_qty_from_text(token)
        if qty_value is None:
            continue
        normalized_text = token if token.lower().endswith("x") else f"{qty_value}x"
        qty_text.append(normalized_text)
        qty_numbers.append(qty_value)
    return {"qty_text": qty_text, "qty": qty_numbers}


def _normalize_part_entry(data: Dict[str, Any]) -> Dict[str, Any]:
    raw_qty = data.get("qty")
    if isinstance(raw_qty, (list, tuple)):
        qty = _coerce_int(raw_qty[0]) if len(raw_qty) == 1 else None
    else:
        qty = _coerce_int(raw_qty)

    raw_qty_text = data.get("qty_text")
    if isinstance(raw_qty_text, (list, tuple)):
        qty_text = _coerce_str(raw_qty_text[0]) if len(raw_qty_text) == 1 else None
    else:
        qty_text = _coerce_str(raw_qty_text)

    if qty_text is None and qty is not None:
        qty_text = f"{qty}x"

    return {
        "part_num": str(data.get("part_num") or "").strip(),
        "color_id": int(data.get("color_id", 0) or 0),
        "color_name": (
            None
            if data.get("color_name") in (None, "", "n/a")
            else str(data.get("color_name")).strip()
        ),
        "element_id": (
            None
            if data.get("element_id") in (None, "", "n/a")
            else str(data.get("element_id")).strip()
        ),
        "qty": qty,
        "qty_text": qty_text,
        "part_bbox": _coerce_box_list(data.get("part_bbox")),
        "confidence": _coerce_float(data.get("confidence")),
    }


def _same_part_entry(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return (
        str(left.get("part_num") or "").strip() == str(right.get("part_num") or "").strip()
        and int(left.get("color_id", 0) or 0) == int(right.get("color_id", 0) or 0)
        and (left.get("element_id") or None) == (right.get("element_id") or None)
    )


def _upsert_crop_entry(
    payload: Dict[str, Any],
    crop_id: str,
    page: Any,
    step: Any,
    qty: Any,
    qty_text: Any = None,
    crop_box: Any = None,
    crop_box_format: Any = None,
    crop_image_path: Any = None,
    annotator: Any = None,
    confidence: Any = None,
    review_status: Any = None,
    adjustments: Any = None,
    notes: Any = None,
    next_qty_index: Any = None,
) -> Dict[str, Any]:
    crops = payload.setdefault("crops", {})
    existing = dict(crops.get(crop_id) or {})
    status = str(existing.get("status") or "needs_adjust").strip().lower()
    if status not in VALID_CROP_STATUSES:
        status = "needs_adjust"
    resolved_review_status = str(review_status or existing.get("review_status") or "unreviewed").strip().lower()
    if resolved_review_status not in VALID_REVIEW_STATUSES:
        resolved_review_status = "unreviewed"
    resolved_crop_box = _coerce_box_list(
        crop_box if crop_box is not None else existing.get("crop_box")
    )
    resolved_qty = _coerce_int_list(qty if qty is not None else existing.get("qty", []))
    resolved_qty_text = _coerce_str_list(
        qty_text if qty_text is not None else existing.get("qty_text", [])
    )
    resolved_next_qty_index = _coerce_int(
        next_qty_index
        if next_qty_index not in (None, "")
        else existing.get("next_qty_index", len(list(existing.get("parts", []) or [])))
    )

    crop_record = {
        "page": int(page or existing.get("page", 0) or 0),
        "step": int(step or existing.get("step", 0) or 0),
        "qty": resolved_qty,
        "qty_text": resolved_qty_text,
        "status": status,
        "crop_box": resolved_crop_box,
        "crop_box_format": str(crop_box_format or existing.get("crop_box_format") or "xywh"),
        "crop_image_path": (
            None
            if crop_image_path in (None, "")
            else str(crop_image_path)
        ) if crop_image_path is not None else existing.get("crop_image_path"),
        "annotator": (
            None
            if annotator in (None, "")
            else str(annotator).strip()
        ) if annotator is not None else existing.get("annotator"),
        "annotated_at": str(existing.get("annotated_at") or ""),
        "confidence": _coerce_float(confidence if confidence is not None else existing.get("confidence")),
        "review_status": resolved_review_status,
        "parts": list(existing.get("parts", []) or []),
        "adjustments": list(adjustments if isinstance(adjustments, list) else existing.get("adjustments", []) or []),
        "notes": str(notes if notes is not None else existing.get("notes") or ""),
        "next_qty_index": max(0, resolved_next_qty_index if resolved_next_qty_index is not None else 0),
    }
    crops[crop_id] = crop_record
    return crop_record


def _load_existing_labels(path: Path) -> Dict[str, Any]:
    stem = path.stem
    set_num = stem.rsplit("_bag", 1)[0] if "_bag" in stem else stem
    bag_text = stem.rsplit("_bag", 1)[1] if "_bag" in stem else "1"
    try:
        bag = int(bag_text or 1)
    except ValueError:
        bag = 1

    existing = _empty_label_store(set_num, bag)
    if not path.exists():
        return existing

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return existing

    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return existing

    if not isinstance(loaded, dict):
        return existing

    existing["schema_version"] = "1.1"
    existing["set_num"] = str(loaded.get("set_num") or existing["set_num"]).strip() or existing["set_num"]
    try:
        existing["bag"] = max(1, int(loaded.get("bag", existing["bag"]) or existing["bag"]))
    except (TypeError, ValueError):
        pass
    existing["created_at"] = str(loaded.get("created_at") or existing["created_at"]).strip() or existing["created_at"]
    if isinstance(loaded.get("source"), dict):
        existing["source"] = dict(loaded.get("source") or {})

    if isinstance(loaded.get("crops"), dict):
        for crop_id, crop_data in loaded.get("crops", {}).items():
            crop_key = str(crop_id or "").strip()
            if not crop_key:
                continue
            crop_dict = crop_data if isinstance(crop_data, dict) else {}
            legacy_qty = _coerce_int_list(crop_dict.get("qty", []))
            legacy_qty_text = _coerce_str_list(crop_dict.get("qty_text", []))
            crop_record = _upsert_crop_entry(
                existing,
                crop_key,
                crop_dict.get("page", 0),
                crop_dict.get("step", 0),
                legacy_qty,
                qty_text=legacy_qty_text,
                crop_box=crop_dict.get("crop_box") or crop_dict.get("coords_xywh") or crop_dict.get("coords"),
                crop_box_format=crop_dict.get("crop_box_format") or "xywh",
                crop_image_path=crop_dict.get("crop_image_path"),
                annotator=crop_dict.get("annotator"),
                confidence=crop_dict.get("confidence"),
                review_status=crop_dict.get("review_status"),
                adjustments=crop_dict.get("adjustments"),
                notes=crop_dict.get("notes"),
                next_qty_index=crop_dict.get("next_qty_index"),
            )
            status = str(crop_dict.get("status") or crop_record["status"]).strip().lower()
            crop_record["status"] = status if status in VALID_CROP_STATUSES else "needs_adjust"
            crop_record["annotated_at"] = (
                str(crop_dict.get("annotated_at") or "").strip()
                or crop_record.get("annotated_at")
                or existing["created_at"]
            )
            crop_record["qty"] = legacy_qty
            crop_record["qty_text"] = legacy_qty_text
            parts: List[Dict[str, Any]] = []
            legacy_sequence = _build_qty_sequence(legacy_qty, legacy_qty_text)
            for index, part_data in enumerate(list(crop_dict.get("parts", []) or [])):
                part_entry = _normalize_part_entry(part_data)
                if not part_entry["part_num"]:
                    continue
                if part_entry["qty"] is None and legacy_sequence:
                    assigned = legacy_sequence[min(index, len(legacy_sequence) - 1)]
                    part_entry["qty"] = assigned.get("qty")
                    part_entry["qty_text"] = assigned.get("qty_text")
                if not any(_same_part_entry(existing_part, part_entry) for existing_part in parts):
                    parts.append(part_entry)
            crop_record["parts"] = parts
            _refresh_crop_next_qty_index(crop_record)
        return existing

    if isinstance(loaded.get("labels"), list):
        for row in loaded.get("labels", []):
            crop_id = str((row or {}).get("crop_id") or "").strip()
            if not crop_id:
                continue
            row_qty = _coerce_int_list((row or {}).get("qty", []))
            row_qty_text = _coerce_str_list((row or {}).get("qty_text", []))
            crop_record = _upsert_crop_entry(
                existing,
                crop_id,
                (row or {}).get("page", 0),
                (row or {}).get("step", 0),
                row_qty,
                qty_text=row_qty_text,
                crop_box=(row or {}).get("crop_box"),
                crop_box_format=(row or {}).get("crop_box_format") or "xywh",
                crop_image_path=(row or {}).get("crop_image_path"),
                annotator=(row or {}).get("annotator"),
                confidence=(row or {}).get("confidence"),
                review_status=(row or {}).get("review_status"),
                adjustments=(row or {}).get("adjustments"),
                notes=(row or {}).get("notes"),
                next_qty_index=(row or {}).get("next_qty_index"),
            )
            crop_record["qty"] = row_qty
            crop_record["qty_text"] = row_qty_text
            part_entry = _normalize_part_entry(row or {})
            row_sequence = _build_qty_sequence(row_qty, row_qty_text)
            if part_entry["qty"] is None and row_sequence:
                assigned = row_sequence[0]
                part_entry["qty"] = assigned.get("qty")
                part_entry["qty_text"] = assigned.get("qty_text")
            if part_entry["part_num"] and not any(
                _same_part_entry(existing_part, part_entry) for existing_part in crop_record["parts"]
            ):
                crop_record["parts"].append(part_entry)
            _refresh_crop_next_qty_index(crop_record)
            crop_record["annotated_at"] = crop_record.get("annotated_at") or str((row or {}).get("annotated_at") or "") or existing["created_at"]
        return existing

    return existing


def _load_verified_training_examples(set_num: str, bag: int) -> List[Dict[str, Any]]:
    payload = _load_existing_labels(_label_store_path(set_num, bag))
    out: List[Dict[str, Any]] = []
    for crop_id, crop_data in dict(payload.get("crops") or {}).items():
        crop_record = crop_data if isinstance(crop_data, dict) else {}
        status = str(crop_record.get("status") or "").strip().lower()
        review_status = str(crop_record.get("review_status") or "").strip().lower()
        if status == "good" and review_status in ("", "unreviewed"):
            review_status = "reviewed"
        if status != "good":
            continue
        if review_status != "reviewed":
            continue
        for part_data in list(crop_record.get("parts", []) or []):
            part_entry = _normalize_part_entry(part_data if isinstance(part_data, dict) else {})
            part_num = str(part_entry.get("part_num") or "").strip()
            if not part_num:
                continue
            color_id = int(part_entry.get("color_id", 0) or 0)
            qty = int(_coerce_int(part_entry.get("qty")) or 1)
            out.append(
                {
                    "crop_id": str(crop_id or ""),
                    "page": int(crop_record.get("page", 0) or 0),
                    "step": int(crop_record.get("step", 0) or 0),
                    "part_num": part_num,
                    "color_id": color_id,
                    "qty": max(1, qty),
                }
            )
    return out


def _debug_verified_training_examples_report(set_num: str, bag: int) -> Dict[str, Any]:
    label_path = _label_store_path(set_num, bag)
    export_path = _training_export_path(set_num, bag)
    payload = _load_existing_labels(label_path)
    exported_examples = _load_saved_training_examples(set_num, bag)
    report: Dict[str, Any] = {
        "label_store_path": str(label_path),
        "label_store_exists": bool(label_path.exists()),
        "export_path": str(export_path),
        "export_exists": bool(export_path.exists()),
        "label_crop_count": 0,
        "label_part_rows_total": 0,
        "export_examples_count": len(exported_examples),
        "kept_examples": 0,
        "skipped_examples": 0,
        "skip_reasons": {
            "missing_status": 0,
            "status_not_good": 0,
            "missing_review_status": 0,
            "review_status_not_reviewed": 0,
            "missing_parts": 0,
            "missing_part_num": 0,
            "missing_color_id": 0,
            "missing_qty": 0,
        },
    }
    crops = dict(payload.get("crops") or {})
    report["label_crop_count"] = len(crops)
    for _, crop_data in crops.items():
        crop_record = crop_data if isinstance(crop_data, dict) else {}
        raw_status = crop_record.get("status")
        status = str(raw_status or "").strip().lower()
        raw_review_status = crop_record.get("review_status")
        review_status = str(raw_review_status or "").strip().lower()
        if status == "good" and review_status in ("", "unreviewed"):
            review_status = "reviewed"
        parts = list(crop_record.get("parts", []) or [])
        if not parts:
            report["skip_reasons"]["missing_parts"] += 1
            continue
        for part_data in parts:
            report["label_part_rows_total"] += 1
            part_entry = _normalize_part_entry(part_data if isinstance(part_data, dict) else {})
            if raw_status in (None, ""):
                report["skipped_examples"] += 1
                report["skip_reasons"]["missing_status"] += 1
                continue
            if status != "good":
                report["skipped_examples"] += 1
                report["skip_reasons"]["status_not_good"] += 1
                continue
            if raw_review_status in (None, "") and review_status != "reviewed":
                report["skipped_examples"] += 1
                report["skip_reasons"]["missing_review_status"] += 1
                continue
            if review_status != "reviewed":
                report["skipped_examples"] += 1
                report["skip_reasons"]["review_status_not_reviewed"] += 1
                continue
            if not str(part_entry.get("part_num") or "").strip():
                report["skipped_examples"] += 1
                report["skip_reasons"]["missing_part_num"] += 1
                continue
            if part_entry.get("color_id") in (None, ""):
                report["skipped_examples"] += 1
                report["skip_reasons"]["missing_color_id"] += 1
                continue
            if part_entry.get("qty") in (None, ""):
                report["skipped_examples"] += 1
                report["skip_reasons"]["missing_qty"] += 1
                continue
            report["kept_examples"] += 1
    return report


def _refresh_set_bag_parts_from_verified_examples(set_num: str, bag_number: int) -> List[Dict[str, Any]]:
    review = _get_bag_review_status(set_num, bag_number)
    if str(review.get("status") or "").strip().lower() != "complete":
        return []

    grouped: Dict[Any, Dict[str, Any]] = {}
    for item in _load_verified_training_examples(set_num, bag_number):
        key = (str(item.get("part_num") or "").strip(), int(item.get("color_id", 0) or 0))
        if not key[0]:
            continue
        row = grouped.setdefault(
            key,
            {
                "set_num": str(set_num or "").strip(),
                "bag_number": int(bag_number or 1),
                "part_num": key[0],
                "color_id": key[1],
                "qty_total": 0,
                "verified_examples": 0,
            },
        )
        row["qty_total"] += max(1, int(item.get("qty", 1) or 1))
        row["verified_examples"] += 1

    rows = sorted(grouped.values(), key=lambda item: (str(item["part_num"]), int(item["color_id"])))
    conn = _connect_bag_inspector_db()
    try:
        _ensure_bag_inventory_tables(conn)
        conn.execute(
            "DELETE FROM set_bag_parts WHERE set_num = ? AND bag_number = ?",
            (str(set_num or "").strip(), int(bag_number or 1)),
        )
        updated_at = _iso_now()
        for row in rows:
            conn.execute(
                """
                INSERT INTO set_bag_parts
                (set_num, bag_number, part_num, color_id, qty_total, verified_examples, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["set_num"],
                    row["bag_number"],
                    row["part_num"],
                    row["color_id"],
                    row["qty_total"],
                    row["verified_examples"],
                    updated_at,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return rows


def _load_set_bag_parts_rows(set_num: str, bag_number: int) -> List[Dict[str, Any]]:
    conn = _connect_bag_inspector_db()
    try:
        _ensure_bag_inventory_tables(conn)
        rows = conn.execute(
            """
            SELECT set_num, bag_number, part_num, color_id, qty_total, verified_examples, updated_at
            FROM set_bag_parts
            WHERE set_num = ? AND bag_number = ?
            ORDER BY part_num ASC, color_id ASC
            """,
            (str(set_num or "").strip(), int(bag_number or 1)),
        ).fetchall()
        return [
            {
                "set_num": str(row["set_num"] or ""),
                "bag_number": int(row["bag_number"] or 1),
                "part_num": str(row["part_num"] or ""),
                "color_id": int(row["color_id"] or 0),
                "qty_total": int(row["qty_total"] or 0),
                "verified_examples": int(row["verified_examples"] or 0),
                "updated_at": str(row["updated_at"] or ""),
            }
            for row in rows
        ]
    finally:
        conn.close()


def _write_remaining_parts_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def get_remaining_parts_for_set(
    set_num: str,
    up_to_bag: Optional[int] = None,
    include_current_reviewed: bool = True,
) -> Dict[str, Any]:
    set_key = str(set_num or "").strip()
    bag_limit = _coerce_int(up_to_bag)
    parts_payload = load_instruction_set_parts(set_key)
    rows_by_key: Dict[str, Dict[str, Any]] = {}

    for part in list(parts_payload.get("parts", []) or []):
        part_num = str(part.get("part_num") or "").strip()
        color_id = int(part.get("color_id", 0) or 0)
        key = f"{part_num}::{color_id}"
        rows_by_key[key] = {
            "part_num": part_num,
            "color_id": color_id,
            "element_id": str(part.get("element_id") or "").strip() or None,
            "img_url": str(part.get("img_url") or "").strip(),
            "starting_qty": int(part.get("qty", 0) or 0),
            "used_qty": 0,
            "used_completed_bags": 0,
            "used_current_reviewed": 0,
            "remaining_qty": int(part.get("qty", 0) or 0),
            "source_types": [],
            "source_refs": [],
        }

    if bag_limit is not None:
        conn = _connect_bag_inspector_db()
        try:
            _ensure_bag_inventory_tables(conn)
            completed_rows = conn.execute(
                """
                SELECT sp.bag_number, sp.part_num, sp.color_id, sp.qty_total
                FROM set_bag_parts sp
                JOIN bag_review_status br
                  ON br.set_num = sp.set_num
                 AND br.bag_number = sp.bag_number
                WHERE sp.set_num = ?
                  AND sp.bag_number < ?
                  AND LOWER(COALESCE(br.status, '')) = 'complete'
                ORDER BY sp.bag_number ASC, sp.part_num ASC, sp.color_id ASC
                """,
                (set_key, int(bag_limit)),
            ).fetchall()
        finally:
            conn.close()

        for row in completed_rows:
            part_num = str(row["part_num"] or "").strip()
            color_id = int(row["color_id"] or 0)
            key = f"{part_num}::{color_id}"
            item = rows_by_key.setdefault(
                key,
                {
                    "part_num": part_num,
                    "color_id": color_id,
                    "element_id": None,
                    "img_url": _lookup_element_image_url(part_num, color_id) or "",
                    "starting_qty": 0,
                    "used_qty": 0,
                    "used_completed_bags": 0,
                    "used_current_reviewed": 0,
                    "remaining_qty": 0,
                    "source_types": [],
                    "source_refs": [],
                },
            )
            qty_total = int(row["qty_total"] or 0)
            item["used_qty"] += qty_total
            item["used_completed_bags"] += qty_total
            item["remaining_qty"] = int(item.get("starting_qty", 0) or 0) - int(item.get("used_qty", 0) or 0)
            if "completed_bags" not in item["source_types"]:
                item["source_types"].append("completed_bags")
            item["source_refs"].append(f"bag {int(row['bag_number'] or 0)}")

        if include_current_reviewed:
            for example in _load_verified_training_examples(set_key, int(bag_limit)):
                part_num = str(example.get("part_num") or "").strip()
                color_id = int(example.get("color_id", 0) or 0)
                key = f"{part_num}::{color_id}"
                item = rows_by_key.setdefault(
                    key,
                    {
                        "part_num": part_num,
                        "color_id": color_id,
                        "element_id": None,
                        "img_url": _lookup_element_image_url(part_num, color_id) or "",
                        "starting_qty": 0,
                        "used_qty": 0,
                        "used_completed_bags": 0,
                        "used_current_reviewed": 0,
                        "remaining_qty": 0,
                        "source_types": [],
                        "source_refs": [],
                    },
                )
                qty_value = int(example.get("qty", 0) or 0)
                item["used_qty"] += qty_value
                item["used_current_reviewed"] += qty_value
                item["remaining_qty"] = int(item.get("starting_qty", 0) or 0) - int(item.get("used_qty", 0) or 0)
                if "current_reviewed" not in item["source_types"]:
                    item["source_types"].append("current_reviewed")
                item["source_refs"].append(
                    f"{str(example.get('crop_id') or '')}/p{int(example.get('page', 0) or 0)}"
                )

    rows = sorted(rows_by_key.values(), key=lambda item: (str(item.get("part_num") or ""), int(item.get("color_id", 0) or 0)))
    remaining_parts_map = {
        f"{str(row.get('part_num') or '').strip()}::{int(row.get('color_id', 0) or 0)}": row
        for row in rows
    }
    payload = {
        "set_num": set_key,
        "bag": bag_limit,
        "include_current_reviewed": bool(include_current_reviewed),
        "rows": rows,
        "remaining_parts_map": remaining_parts_map,
        "starting_qty_total": sum(int(row.get("starting_qty", 0) or 0) for row in rows),
        "used_qty_total": sum(int(row.get("used_qty", 0) or 0) for row in rows),
        "remaining_qty_total": sum(int(row.get("remaining_qty", 0) or 0) for row in rows),
        "generated_at": _iso_now(),
    }
    if bag_limit is not None:
        _write_remaining_parts_cache(_remaining_parts_cache_path(set_key, int(bag_limit)), payload)
    return payload


def _build_bag_qty_audit_rows(set_num: str, bag_number: int) -> Dict[str, Any]:
    reviewed_examples = _load_verified_training_examples(set_num, bag_number)
    set_bag_parts = _load_set_bag_parts_rows(set_num, bag_number)
    debug_report = _debug_verified_training_examples_report(set_num, bag_number)

    reviewed_by_key: Dict[Any, Dict[str, Any]] = {}
    for item in reviewed_examples:
        key = (str(item.get("part_num") or "").strip(), int(item.get("color_id", 0) or 0))
        row = reviewed_by_key.setdefault(
            key,
            {
                "part_num": key[0],
                "color_id": key[1],
                "reviewed_qty_total": 0,
                "sources": [],
            },
        )
        row["reviewed_qty_total"] += int(item.get("qty", 0) or 0)
        row["sources"].append(
            {
                "crop_id": str(item.get("crop_id") or ""),
                "page": int(item.get("page", 0) or 0),
            }
        )

    stored_by_key = {
        (str(row.get("part_num") or "").strip(), int(row.get("color_id", 0) or 0)): row
        for row in set_bag_parts
    }
    all_keys = sorted(set(reviewed_by_key.keys()) | set(stored_by_key.keys()), key=lambda item: (item[0], item[1]))
    rows: List[Dict[str, Any]] = []
    for key in all_keys:
        reviewed_row = reviewed_by_key.get(key) or {}
        stored_row = stored_by_key.get(key) or {}
        reviewed_qty_total = int(reviewed_row.get("reviewed_qty_total", 0) or 0)
        stored_qty_total = int(stored_row.get("qty_total", 0) or 0)
        sources = list(reviewed_row.get("sources", []) or [])
        rows.append(
            {
                "part_num": key[0],
                "color_id": key[1],
                "qty_total": stored_qty_total,
                "reviewed_qty_total": reviewed_qty_total,
                "sources": sources,
                "status": "OK" if stored_qty_total == reviewed_qty_total else "MISMATCH",
            }
        )

    return {
        "reviewed_examples_count": len(reviewed_examples),
        "set_bag_parts_rows": set_bag_parts,
        "audit_rows": rows,
        "reviewed_qty_total": sum(int(item.get("qty", 0) or 0) for item in reviewed_examples),
        "set_bag_parts_qty_total": sum(int(row.get("qty_total", 0) or 0) for row in set_bag_parts),
        "debug_report": debug_report,
    }


def _write_labels(path: Path, payload: Dict[str, Any]) -> None:
    try:
        bag = int(payload.get("bag", 1) or 1)
    except (TypeError, ValueError):
        bag = 1

    normalized = _empty_label_store(
        str(payload.get("set_num") or ""),
        bag,
    )
    normalized["created_at"] = str(payload.get("created_at") or normalized["created_at"]).strip() or normalized["created_at"]
    if isinstance(payload.get("source"), dict):
        normalized["source"] = dict(payload.get("source") or {})

    for crop_id, crop_data in dict(payload.get("crops") or {}).items():
        crop_key = str(crop_id or "").strip()
        if not crop_key:
            continue
        crop_dict = crop_data if isinstance(crop_data, dict) else {}
        crop_record = _upsert_crop_entry(
            normalized,
            crop_key,
            crop_dict.get("page", 0),
            crop_dict.get("step", 0),
            crop_dict.get("qty", []),
            qty_text=crop_dict.get("qty_text", []),
            crop_box=crop_dict.get("crop_box"),
            crop_box_format=crop_dict.get("crop_box_format"),
            crop_image_path=crop_dict.get("crop_image_path"),
            annotator=crop_dict.get("annotator"),
            confidence=crop_dict.get("confidence"),
            review_status=crop_dict.get("review_status"),
            adjustments=crop_dict.get("adjustments"),
            notes=crop_dict.get("notes"),
            next_qty_index=crop_dict.get("next_qty_index"),
        )
        crop_record["status"] = str(crop_dict.get("status") or crop_record["status"]).strip().lower()
        if crop_record["status"] not in VALID_CROP_STATUSES:
            crop_record["status"] = "needs_adjust"
        crop_record["annotated_at"] = (
            str(crop_dict.get("annotated_at") or "").strip()
            or normalized["created_at"]
        )
        clean_parts: List[Dict[str, Any]] = []
        for part_data in list(crop_dict.get("parts", []) or []):
            part_entry = _normalize_part_entry(part_data)
            if not part_entry["part_num"]:
                continue
            if not any(_same_part_entry(existing_part, part_entry) for existing_part in clean_parts):
                clean_parts.append(part_entry)
        crop_record["parts"] = clean_parts
        _refresh_crop_next_qty_index(crop_record)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _build_instruction_callout_crops(set_num: str, bag: int, ai_enabled: bool = False) -> List[Dict[str, Any]]:
    rendered_pages, start_page, end_page = _resolve_bag_page_range(str(set_num), int(bag))
    crops: List[Dict[str, Any]] = []

    for page in rendered_pages:
        if int(page) < int(start_page) or int(page) > int(end_page):
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
            include_minifig=False,
        )

        callout_candidates = [
            item
            for item in crop_candidates
            if str(item.get("candidate_origin", "")) == "callout_box_candidate"
            and bool(item.get("match_enabled"))
        ]

        # Robust debug fallback: when the existing material crop pipeline finds
        # no strict callout boxes, detect the blue callout rectangle directly
        # above each detected step number.  This keeps the step number out of
        # the crop and gives the OCR/slot code a much cleaner image.
        if not callout_candidates:
            for sb_idx, step_box in enumerate(step_boxes or [], start=1):
                try:
                    sx = int(step_box.get("x", 0) or 0)
                    sy = int(step_box.get("y", 0) or 0)
                    sw = int(step_box.get("w", 0) or 0)
                    sh = int(step_box.get("h", 0) or 0)
                except Exception:
                    continue
                if sw <= 0 or sh <= 0:
                    continue
                step_number = int(step_box.get("step_number", 0) or 0)
                pad_left = max(18, int(sw * 0.8))
                pad_above = max(75, int(sh * 5.0))
                pad_right = max(220, int(sw * 8.0))
                search_x1 = max(0, sx - pad_left)
                search_y1 = max(0, sy - pad_above)
                search_x2 = min(page_width, sx + sw + pad_right)
                search_y2 = min(page_height, sy)
                if search_y2 <= search_y1 or search_x2 <= search_x1:
                    continue

                rect = _detect_callout_rect_by_color_fill(
                    img,
                    [search_x1, search_y1, search_x2, search_y2],
                    step_y=sy,
                    page_width=page_width,
                    page_height=page_height,
                )

                if rect is not None:
                    rx, ry, rw, rh = [int(v) for v in rect]
                    crop_img = img[ry : ry + rh, rx : rx + rw]
                    if crop_img is None or crop_img.size == 0:
                        continue
                    try:
                        data_uri = _encode_debug_image_data_uri(crop_img, max_width=420)
                    except Exception:
                        continue
                    qty_payload = _auto_qty_payload_for_crop(crop_img, step_number)
                    callout_candidates.append(
                        {
                            "candidate_origin": "callout_box_candidate",
                            "match_enabled": True,
                            "data_uri": data_uri,
                            "coords_xywh": [rx, ry, rw, rh],
                            "coords_label": "edge detected callout",
                            "confidence": 0.45,
                            "step_number": step_number,
                            "detected_qty_text": qty_payload.get("detected_qty_text", []),
                            "detected_qty_numbers": qty_payload.get("detected_qty_numbers", []),
                        }
                    )
                    continue

                # Last resort: show a conservative region above the step number
                # so the debug workflow never goes completely empty.
                bounds = _safe_crop_bounds(search_x1, search_y1, search_x2, search_y2, page_width, page_height)
                if bounds is None:
                    continue
                fx1, fy1, fx2, fy2 = bounds
                crop_img = img[fy1:fy2, fx1:fx2]
                if crop_img is None or crop_img.size == 0:
                    continue
                if _yellow_ratio_bgr(crop_img) > 0.18:
                    continue
                try:
                    data_uri = _encode_debug_image_data_uri(crop_img, max_width=420)
                except Exception:
                    continue
                qty_payload = _auto_qty_payload_for_crop(crop_img, step_number)
                callout_candidates.append(
                    {
                        "candidate_origin": "step_box_fallback",
                        "match_enabled": True,
                        "data_uri": data_uri,
                        "coords_xywh": [fx1, fy1, fx2 - fx1, fy2 - fy1],
                        "coords_label": "step box fallback",
                        "confidence": 0.25,
                        "step_number": step_number,
                        "detected_qty_text": qty_payload.get("detected_qty_text", []),
                        "detected_qty_numbers": qty_payload.get("detected_qty_numbers", []),
                    }
                )

        for idx, candidate in enumerate(callout_candidates, start=1):
            step_number = int(candidate.get("step_number", 0) or 0)
            qty_payload: Dict[str, Any] = {
                "detected_qty_text": list(candidate.get("detected_qty_text", []) or []),
                "detected_qty_numbers": list(candidate.get("detected_qty_numbers", []) or []),
                "qty_source": "local",
                "ai_part_count": None,
                "ai_issues": [],
                "ai_crop_box": None,
                "ai_suggested_fix": False,
            }
            if ai_enabled:
                crop_box = _coerce_box_list(candidate.get("coords_xywh"))
                if crop_box is not None:
                    x = int(crop_box[0] or 0)
                    y = int(crop_box[1] or 0)
                    w = int(crop_box[2] or 0)
                    h = int(crop_box[3] or 0)
                    if w > 0 and h > 0:
                        ai_qty_payload = _ai_qty_payload_for_crop(
                            img[y : y + h, x : x + w],
                            str(set_num),
                            int(page),
                            int(step_number),
                        )
                        if ai_qty_payload is not None:
                            qty_payload = ai_qty_payload
            detected_qty_text = [
                str(value)
                for value in list(qty_payload.get("detected_qty_text", []) or [])
                if str(value or "").strip()
            ]
            detected_qty_numbers = [
                int(value)
                for value in list(qty_payload.get("detected_qty_numbers", []) or [])
                if str(value).strip()
            ]
            # Final safety: do not allow OCR to turn the step number into a qty.
            if step_number and str(qty_payload.get("qty_source") or "local") != "openai":
                clean_pairs = [
                    (text, number)
                    for text, number in zip(detected_qty_text, detected_qty_numbers)
                    if int(number) != int(step_number)
                ]
                detected_qty_text = [text for text, _ in clean_pairs]
                detected_qty_numbers = [number for _, number in clean_pairs]
            crop_id = f"p{int(page)}_s{max(step_number, 0)}_c{idx}"
            crops.append(
                {
                    "crop_id": crop_id,
                    "page": int(page),
                    "step": step_number,
                    "qty_text": detected_qty_text,
                    "qty_numbers": detected_qty_numbers,
                    "qty_label": ", ".join(detected_qty_text) if detected_qty_text else "none",
                    "qty_source": str(qty_payload.get("qty_source") or "local"),
                    "ai_part_count": qty_payload.get("ai_part_count"),
                    "ai_issues": list(qty_payload.get("ai_issues", []) or []),
                    "ai_crop_box": qty_payload.get("ai_crop_box"),
                    "ai_suggested_fix": bool(qty_payload.get("ai_suggested_fix")),
                    "data_uri": str(candidate.get("data_uri") or ""),
                    "coords_label": str(candidate.get("coords_label") or candidate.get("candidate_origin") or "fallback crop"),
                    "crop_box": list(candidate.get("coords_xywh", []) or []),
                    "crop_box_format": "xywh",
                    "crop_image_path": str(image_path),
                    "confidence": _coerce_float(candidate.get("confidence")),
                }
            )

    return crops

def _build_manual_crop_pages(set_num: str, bag: int) -> List[Dict[str, Any]]:
    rendered_pages, start_page, end_page = _resolve_bag_page_range(str(set_num), int(bag))
    pages: List[Dict[str, Any]] = []

    for page in rendered_pages:
        if int(page) < int(start_page) or int(page) > int(end_page):
            continue

        image_path = debug_service.resolve_page_image_path(str(set_num), int(page))
        if image_path is None:
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        page_height, page_width = img.shape[:2]
        try:
            data_uri = _encode_debug_image_data_uri(img, max_width=1200)
        except Exception:
            continue

        pages.append(
            {
                "page": int(page),
                "image_path": str(image_path),
                "data_uri": data_uri,
                "width": int(page_width),
                "height": int(page_height),
            }
        )

    return pages


def _build_crop_image_html(crop: Dict[str, Any]) -> str:
    data_uri = str(crop.get("data_uri") or "").strip()
    crop_id = str(crop.get("crop_id") or "").strip()
    if not data_uri:
        return '<div class="crop-missing">Crop unavailable</div>'
    return (
        f'<img src="{escape(data_uri)}" data-src="{escape(data_uri)}" '
        f'data-crop-id="{escape(crop_id)}" alt="{escape(crop_id)}" loading="lazy" '
        'onclick="openCropZoomFromEl(event, this)" />'
    )


def _debug_top_part_candidates_for_crop(
    crop: Dict[str, Any],
    metallic_mode: bool = False,
    set_num: str = "",
    remaining_parts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    data_uri = str(crop.get("data_uri") or "").strip()
    if not data_uri or "," not in data_uri:
        return []
    try:
        encoded = data_uri.split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return []
        color_ids = sorted({int(part.get("color_id", 0) or 0) for part in list(crop.get("parts", []) or []) if int(part.get("color_id", 0) or 0) > 0})
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if not cv2.imwrite(tmp_path, img):
                return []
            candidates = list(
                get_part_candidates_for_crop(
                    tmp_path,
                    max_candidates=12,
                    color_ids=color_ids or None,
                    metallic_mode=bool(metallic_mode),
                    set_num=str(set_num or "").strip() or None,
                    remaining_parts=remaining_parts,
                )
                or []
            )
            return service_dedupe_candidate_rows(candidates)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception:
        return []


def _build_export_training_payload(set_num: str, bag: int) -> Dict[str, Any]:
    path = _label_store_path(set_num, bag)
    existing = _load_existing_labels(path)
    examples: List[Dict[str, Any]] = []

    for crop_id, crop_data in sorted(
        dict(existing.get("crops") or {}).items(),
        key=lambda item: (
            int((item[1] or {}).get("page", 0) or 0),
            int((item[1] or {}).get("step", 0) or 0),
            str(item[0] or ""),
        ),
    ):
        crop_record = crop_data if isinstance(crop_data, dict) else {}
        crop_status = str(crop_record.get("status") or "needs_adjust")
        persisted_review_status = str(crop_record.get("review_status") or "").strip()
        normalized_review_status = persisted_review_status or "unreviewed"
        review_status_source = "persisted" if persisted_review_status else "defaulted"
        if str(crop_status).strip().lower() == "good" and normalized_review_status.lower() == "unreviewed":
            normalized_review_status = "reviewed"
            review_status_source = "normalized_from_good"
        for part_data in list(crop_record.get("parts", []) or []):
            part_entry = _normalize_part_entry(part_data if isinstance(part_data, dict) else {})
            if not part_entry["part_num"]:
                continue
            element_img_url = _lookup_element_image_url(
                str(part_entry.get("part_num") or ""),
                int(part_entry.get("color_id", 0) or 0),
            )
            examples.append(
                {
                    "crop_id": str(crop_id or ""),
                    "page": int(crop_record.get("page", 0) or 0),
                    "step": int(crop_record.get("step", 0) or 0),
                    "crop_image_path": str(crop_record.get("crop_image_path") or ""),
                    "status": crop_status,
                    "review_status": normalized_review_status,
                    "review_status_source": review_status_source,
                    "part_num": str(part_entry.get("part_num") or ""),
                    "color_id": int(part_entry.get("color_id", 0) or 0),
                    "color_name": str(part_entry.get("color_name") or "n/a"),
                    "qty": _coerce_int(part_entry.get("qty")) or 1,
                    "qty_text": str(part_entry.get("qty_text") or ""),
                    "element_img_url": element_img_url or "",
                    "thumbnail_source": "element_images",
                    "thumbnail_status": "ok" if element_img_url else "missing_element_image",
                    "metallic_mode": bool(
                        crop_record.get("metallic_mode")
                        or crop_record.get("manual_metallic_mode")
                    ),
                }
            )

    for example in examples:
        if not example["qty_text"]:
            example["qty_text"] = f"{int(example['qty'])}x"

    return {
        "set_num": str(existing.get("set_num") or str(set_num or "").strip() or "70618"),
        "bag": max(1, int(existing.get("bag", bag) or bag or 1)),
        "examples": examples,
    }


def _write_export_training_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _load_saved_training_examples(set_num: str, bag: int) -> List[Dict[str, Any]]:
    path = _training_export_path(set_num, bag)
    if not path.exists():
        return []

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(loaded, dict):
        return []

    examples = loaded.get("examples")
    if not isinstance(examples, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in examples:
        if not isinstance(item, dict):
            continue
        part_num = str(item.get("part_num") or "").strip()
        color_id = _coerce_int(item.get("color_id"))
        if not part_num or color_id is None:
            continue
        normalized.append(
            {
                "part_num": part_num,
                "color_id": int(color_id),
                "crop_id": str(item.get("crop_id") or "").strip(),
                "page": _coerce_int(item.get("page")),
                "step": _coerce_int(item.get("step")),
                "qty": _coerce_int(item.get("qty")),
                "qty_text": str(item.get("qty_text") or "").strip(),
                "metallic_mode": bool(item.get("metallic_mode")),
            }
        )
    return normalized


def _load_local_part_image_manifest(set_num: str) -> Dict[str, Dict[str, Any]]:
    manifest_path = Path("/Users/olly/aim2build-instruction/debug/part_image_cache") / str(set_num or "").strip() / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in loaded:
        if not isinstance(row, dict):
            continue
        part_num = str(row.get("part_num") or "").strip()
        color_id = _coerce_int(row.get("color_id"))
        if not part_num or color_id is None:
            continue
        out[f"{part_num}::{int(color_id)}"] = {
            "part_num": part_num,
            "color_id": int(color_id),
            "element_id": str(row.get("element_id") or "").strip() or None,
            "qty": _coerce_int(row.get("qty")),
            "img_url": str(row.get("img_url") or "").strip(),
            "local_path": str(row.get("local_path") or "").strip(),
        }
    return out


def _load_training_examples_for_set(set_num: str) -> List[Dict[str, Any]]:
    root = Path("/Users/olly/aim2build-instruction/debug/training_data")
    if not root.exists():
        return []
    out: List[Dict[str, Any]] = []
    prefix = f"{str(set_num or '').strip()}_bag"
    for path in sorted(root.glob(f"{prefix}*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8") or "{}")
        except Exception:
            continue
        for item in list((payload or {}).get("examples", []) or []):
            if not isinstance(item, dict):
                continue
            part_num = str(item.get("part_num") or "").strip()
            color_id = _coerce_int(item.get("color_id"))
            if not part_num or color_id is None:
                continue
            out.append(
                {
                    "part_num": part_num,
                    "color_id": int(color_id),
                    "color_name": str(item.get("color_name") or "").strip(),
                    "crop_id": str(item.get("crop_id") or "").strip(),
                    "page": _coerce_int(item.get("page")),
                    "bag": _coerce_int((payload or {}).get("bag")),
                }
            )
    return out


def _build_missing_element_images_report(set_num: str) -> Dict[str, Any]:
    set_key = str(set_num or "").strip()
    manifest_map = _load_local_part_image_manifest(set_key)
    training_examples = _load_training_examples_for_set(set_key)
    parts_payload = load_instruction_set_parts(set_key)
    remaining_cache_root = Path("/Users/olly/aim2build-instruction/debug/remaining_parts")

    entries: Dict[str, Dict[str, Any]] = {}

    def _add_source(part_num: str, color_id: int, color_name: str, source_name: str, source_ref: str = "") -> None:
        key = f"{part_num}::{int(color_id)}"
        entry = entries.setdefault(
            key,
            {
                "part_num": part_num,
                "color_id": int(color_id),
                "color_name": color_name or _lookup_color_name_for_part_image(int(color_id)),
                "sources": [],
            },
        )
        if source_name not in entry["sources"]:
            entry["sources"].append(source_name)
        if source_ref:
            entry.setdefault("source_refs", []).append(source_ref)

    for part in list(parts_payload.get("parts", []) or []):
        part_num = str(part.get("part_num") or "").strip()
        color_id = _coerce_int(part.get("color_id"))
        if not part_num or color_id is None:
            continue
        _add_source(part_num, int(color_id), str(part.get("color_name") or ""), "set_parts")

    for item in training_examples:
        _add_source(
            str(item.get("part_num") or "").strip(),
            int(item.get("color_id", 0) or 0),
            str(item.get("color_name") or ""),
            "training_examples",
            f"{str(item.get('crop_id') or '')}/p{int(item.get('page', 0) or 0)}",
        )

    if remaining_cache_root.exists():
        for path in sorted(remaining_cache_root.glob(f"{set_key}_bag*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8") or "{}")
            except Exception:
                continue
            for row in list((payload or {}).get("rows", []) or []):
                if not isinstance(row, dict):
                    continue
                part_num = str(row.get("part_num") or "").strip()
                color_id = _coerce_int(row.get("color_id"))
                if not part_num or color_id is None:
                    continue
                _add_source(part_num, int(color_id), "", "remaining_parts")

    rows: List[Dict[str, Any]] = []
    for key, entry in sorted(entries.items(), key=lambda item: (item[1]["part_num"], item[1]["color_id"])):
        part_num = str(entry.get("part_num") or "").strip()
        color_id = int(entry.get("color_id", 0) or 0)
        existing_img_url = _lookup_element_image_url(part_num, color_id)
        if existing_img_url:
            continue
        manifest_entry = dict(manifest_map.get(key) or {})
        expected_r2_url = _canonical_element_r2_url(part_num, color_id)
        local_cache_path = str(manifest_entry.get("local_path") or "").strip()
        checked_sources = ["catalog_db", "r2_convention", "local_cache"]
        if local_cache_path:
            action_needed = "upload_local_cache_to_r2"
            missing_reason = "missing element_images.img_url; local cache exists"
        elif str(manifest_entry.get("img_url") or "").strip():
            action_needed = "verify_cached_remote_then_confirm_r2"
            missing_reason = "missing element_images.img_url; cache manifest has remote img_url"
        else:
            action_needed = "manual_or_queued_fetch_needed"
            missing_reason = "missing element_images.img_url; no local cache found"
        rows.append(
            {
                "part_num": part_num,
                "color_id": color_id,
                "color_name": str(entry.get("color_name") or _lookup_color_name_for_part_image(color_id) or ""),
                "missing_reason": missing_reason,
                "candidate_source_checked": ",".join(checked_sources),
                "action_needed": action_needed,
                "expected_r2_url": expected_r2_url,
                "local_cache_path": local_cache_path,
                "manifest_img_url": str(manifest_entry.get("img_url") or "").strip(),
                "element_id": manifest_entry.get("element_id"),
                "sources": list(entry.get("sources", []) or []),
                "source_refs": list(entry.get("source_refs", []) or []),
            }
        )

    payload = {
        "set_num": set_key,
        "generated_at": _iso_now(),
        "count": len(rows),
        "rows": rows,
    }
    _write_export_training_payload(_missing_images_report_path(set_key), payload)
    return payload


@router.post("/debug/save-label")
async def save_label(req: Request):
    data = await req.json()

    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    try:
        bag = int(data.get("bag", 1) or 1)
    except Exception:
        bag = 1

    crop_id = str(data.get("crop_id") or "").strip()
    part_entry = _normalize_part_entry(data)
    allow_extra_part = bool(data.get("allow_extra_part"))

    if not crop_id:
        raise HTTPException(status_code=400, detail="crop_id is required")
    if not part_entry["part_num"]:
        raise HTTPException(status_code=400, detail="part_num is required")

    path = _label_store_path(set_num, bag)
    existing = _load_existing_labels(path)
    crop_record = _upsert_crop_entry(
        existing,
        crop_id,
        data.get("page", 0),
        data.get("step", 0),
        data.get("crop_qty", data.get("qty", [])),
        qty_text=data.get("crop_qty_text", data.get("qty_text", [])),
        crop_box=data.get("crop_box"),
        crop_box_format=data.get("crop_box_format"),
        crop_image_path=data.get("crop_image_path"),
        annotator=data.get("annotator"),
        confidence=data.get("crop_confidence", data.get("confidence")),
        review_status=data.get("review_status"),
        adjustments=data.get("adjustments"),
        notes=data.get("notes"),
    )
    already_present = any(
        _same_part_entry(existing_part, part_entry) for existing_part in crop_record["parts"]
    )
    if not already_present:
        assigned_qty = _pick_qty_assignment(
            crop_record,
            data.get("crop_qty", data.get("qty", [])),
            data.get("crop_qty_text", data.get("qty_text", [])),
            allow_extra_part=allow_extra_part,
        )
        if assigned_qty.get("slots_full"):
            raise HTTPException(status_code=400, detail="All qty slots filled")
        part_entry["qty"] = assigned_qty.get("qty")
        part_entry["qty_text"] = assigned_qty.get("qty_text")
        crop_record["parts"].append(part_entry)
    _refresh_crop_next_qty_index(crop_record)
    crop_record["annotated_at"] = _iso_now()
    _write_labels(path, existing)
    return {"ok": True, "path": str(path), "crop": existing["crops"].get(crop_id)}


@router.post("/debug/remove-label")
async def remove_label(req: Request):
    data = await req.json()

    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    try:
        bag = int(data.get("bag", 1) or 1)
    except Exception:
        bag = 1

    crop_id = str(data.get("crop_id") or "").strip()
    part_entry = _normalize_part_entry(data)

    if not crop_id:
        raise HTTPException(status_code=400, detail="crop_id is required")
    if not part_entry["part_num"]:
        raise HTTPException(status_code=400, detail="part_num is required")

    path = _label_store_path(set_num, bag)
    existing = _load_existing_labels(path)
    crop_record = existing.get("crops", {}).get(crop_id)
    if not crop_record:
        raise HTTPException(status_code=404, detail="crop_id not found")

    current_parts = list(crop_record.get("parts", []) or [])
    removed_index = None
    remaining_parts: List[Dict[str, Any]] = []
    for index, existing_part in enumerate(current_parts):
        if removed_index is None and _same_part_entry(existing_part, part_entry):
            removed_index = index
            continue
        remaining_parts.append(existing_part)
    crop_record["parts"] = remaining_parts
    removed = 1 if removed_index is not None else 0
    _refresh_crop_next_qty_index(crop_record)
    crop_record["annotated_at"] = _iso_now()
    _write_labels(path, existing)
    return {"ok": True, "path": str(path), "removed": removed, "crop": crop_record}


@router.post("/debug/set-crop-status")
async def set_crop_status(req: Request):
    data = await req.json()

    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    try:
        bag = int(data.get("bag", 1) or 1)
    except Exception:
        bag = 1

    crop_id = str(data.get("crop_id") or "").strip()
    status = str(data.get("status") or "").strip().lower()

    if not crop_id:
        raise HTTPException(status_code=400, detail="crop_id is required")
    if status not in VALID_CROP_STATUSES:
        raise HTTPException(status_code=400, detail="status must be one of good, bad, needs_adjust, hidden")
    review_status = data.get("review_status")
    if status == "good":
        review_status = "reviewed"

    path = _label_store_path(set_num, bag)
    existing = _load_existing_labels(path)
    crop_record = _upsert_crop_entry(
        existing,
        crop_id,
        data.get("page", 0),
        data.get("step", 0),
        data.get("crop_qty", data.get("qty", [])),
        qty_text=data.get("crop_qty_text", data.get("qty_text", [])),
        crop_box=data.get("crop_box"),
        crop_box_format=data.get("crop_box_format"),
        crop_image_path=data.get("crop_image_path"),
        annotator=data.get("annotator"),
        confidence=data.get("crop_confidence", data.get("confidence")),
        review_status=review_status,
        adjustments=data.get("adjustments"),
        notes=data.get("notes"),
    )
    crop_record["status"] = status
    _refresh_crop_next_qty_index(crop_record)
    crop_record["annotated_at"] = _iso_now()
    _write_labels(path, existing)
    return {"ok": True, "path": str(path), "crop": crop_record}


@router.post("/debug/delete-crop")
async def delete_crop(req: Request):
    data = await req.json()

    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    try:
        bag = int(data.get("bag", 1) or 1)
    except Exception:
        bag = 1

    crop_id = str(data.get("crop_id") or "").strip()
    if not crop_id:
        raise HTTPException(status_code=400, detail="crop_id is required")

    path = _label_store_path(set_num, bag)
    existing = _load_existing_labels(path)
    existing_crops = existing.setdefault("crops", {})

    if _is_manual_crop_id(crop_id):
        removed_crop = existing_crops.pop(crop_id, None)
        _write_labels(path, existing)
        return {
            "ok": True,
            "path": str(path),
            "crop_id": crop_id,
            "deleted": bool(removed_crop),
            "hidden": False,
        }

    crop_record = _upsert_crop_entry(
        existing,
        crop_id,
        data.get("page", 0),
        data.get("step", 0),
        data.get("crop_qty", data.get("qty", [])),
        qty_text=data.get("crop_qty_text", data.get("qty_text", [])),
        crop_box=data.get("crop_box"),
        crop_box_format=data.get("crop_box_format"),
        crop_image_path=data.get("crop_image_path"),
        annotator=data.get("annotator"),
        confidence=data.get("crop_confidence", data.get("confidence")),
        review_status=data.get("review_status"),
        adjustments=data.get("adjustments"),
        notes=data.get("notes"),
    )
    crop_record["status"] = "hidden"
    crop_record["annotated_at"] = _iso_now()
    _write_labels(path, existing)
    return {
        "ok": True,
        "path": str(path),
        "crop_id": crop_id,
        "deleted": False,
        "hidden": True,
        "crop": crop_record,
    }


@router.post("/debug/save-manual-crop")
async def save_manual_crop(req: Request):
    data = await req.json()

    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    try:
        bag = int(data.get("bag", 1) or 1)
    except Exception:
        bag = 1

    page = _coerce_int(data.get("page"))
    step = _coerce_int(data.get("step"))
    crop_box = _coerce_box_list(data.get("crop_box"))
    crop_image_path = str(data.get("crop_image_path") or "").strip()

    if page is None or page <= 0:
        raise HTTPException(status_code=400, detail="page is required")
    if step is None or step < 0:
        raise HTTPException(status_code=400, detail="step is required")
    if crop_box is None:
        raise HTTPException(status_code=400, detail="crop_box must be xywh")
    if not crop_image_path:
        raise HTTPException(status_code=400, detail="crop_image_path is required")

    img = cv2.imread(crop_image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not load crop_image_path")

    crop_img = img
    if crop_box is not None:
        x, y, w, h = crop_box
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(0, int(w))
        h = max(0, int(h))
        crop_img = img[y : y + h, x : x + w]
    if crop_img is None or crop_img.size == 0:
        raise HTTPException(status_code=400, detail="Manual crop box produced an empty crop")

    qty_payload = _extract_detected_qty_details_from_crop(crop_img)
    detected_qty_text = _coerce_str_list(qty_payload.get("detected_qty_text", []))
    detected_qty_numbers = _coerce_int_list(qty_payload.get("detected_qty_numbers", []))

    path = _label_store_path(set_num, bag)
    existing = _load_existing_labels(path)

    matched_crop_id: Optional[str] = None
    manual_serial = 1
    for existing_crop_id, existing_crop in dict(existing.get("crops") or {}).items():
        crop_key = str(existing_crop_id or "").strip()
        crop_dict = existing_crop if isinstance(existing_crop, dict) else {}
        if crop_key.startswith(f"manual_p{int(page)}_"):
            manual_serial += 1
        if (
            crop_key.startswith("manual_")
            and int(crop_dict.get("page", 0) or 0) == int(page)
            and int(crop_dict.get("step", 0) or 0) == int(step)
            and _coerce_box_list(crop_dict.get("crop_box")) == crop_box
            and str(crop_dict.get("crop_image_path") or "").strip() == crop_image_path
        ):
            matched_crop_id = crop_key
            break

    crop_id = matched_crop_id or _manual_crop_id(page, step, manual_serial)
    crop_record = _upsert_crop_entry(
        existing,
        crop_id,
        page,
        step,
        detected_qty_numbers,
        qty_text=detected_qty_text,
        crop_box=crop_box,
        crop_box_format="xywh",
        crop_image_path=crop_image_path,
        annotator=data.get("annotator"),
        confidence=data.get("crop_confidence"),
        review_status=data.get("review_status"),
        adjustments=data.get("adjustments"),
        notes=data.get("notes"),
    )
    crop_record["status"] = "good"
    crop_record["qty"] = detected_qty_numbers
    crop_record["qty_text"] = detected_qty_text
    _refresh_crop_next_qty_index(crop_record)
    crop_record["annotated_at"] = _iso_now()
    _write_labels(path, existing)
    return {"ok": True, "path": str(path), "crop": existing["crops"].get(crop_id)}


@router.post("/debug/update-crop-qty")
async def update_crop_qty(req: Request):
    data = await req.json()

    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    try:
        bag = int(data.get("bag", 1) or 1)
    except Exception:
        bag = 1

    crop_id = str(data.get("crop_id") or "").strip()
    if not crop_id:
        raise HTTPException(status_code=400, detail="crop_id is required")

    parsed_qty = _parse_qty_text_input(data.get("qty_input"))
    path = _label_store_path(set_num, bag)
    existing = _load_existing_labels(path)
    crop_record = _upsert_crop_entry(
        existing,
        crop_id,
        data.get("page", 0),
        data.get("step", 0),
        parsed_qty["qty"],
        qty_text=parsed_qty["qty_text"],
        crop_box=data.get("crop_box"),
        crop_box_format=data.get("crop_box_format"),
        crop_image_path=data.get("crop_image_path"),
        annotator=data.get("annotator"),
        confidence=data.get("crop_confidence", data.get("confidence")),
        review_status=data.get("review_status"),
        adjustments=data.get("adjustments"),
        notes=data.get("notes"),
        next_qty_index=0,
    )
    crop_record["qty"] = parsed_qty["qty"]
    crop_record["qty_text"] = parsed_qty["qty_text"]
    _refresh_crop_next_qty_index(crop_record)
    crop_record["annotated_at"] = _iso_now()
    _write_labels(path, existing)
    return {"ok": True, "path": str(path), "crop": existing["crops"].get(crop_id)}


@router.post("/debug/update-assigned-part-qty")
async def update_assigned_part_qty(req: Request):
    data = await req.json()

    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    try:
        bag = int(data.get("bag", 1) or 1)
    except Exception:
        bag = 1

    crop_id = str(data.get("crop_id") or "").strip()
    part_num = str(data.get("part_num") or "").strip()
    color_id = _coerce_int(data.get("color_id"))
    element_id = _coerce_str(data.get("element_id"))
    if not crop_id:
        raise HTTPException(status_code=400, detail="crop_id is required")
    if not part_num:
        raise HTTPException(status_code=400, detail="part_num is required")
    if color_id is None:
        raise HTTPException(status_code=400, detail="color_id is required")

    parsed_qty = _parse_qty_text_input(data.get("qty_input"))
    qty_values = list(parsed_qty.get("qty", []) or [])
    qty_text_values = list(parsed_qty.get("qty_text", []) or [])
    if not qty_values or not qty_text_values:
        raise HTTPException(status_code=400, detail="qty_input must contain at least one qty label")
    qty_value = int(qty_values[0])
    qty_text_value = str(qty_text_values[0])

    path = _label_store_path(set_num, bag)
    existing = _load_existing_labels(path)
    crop_record = existing.get("crops", {}).get(crop_id)
    if not crop_record:
        raise HTTPException(status_code=404, detail="crop_id not found")

    updated = False
    for part in list(crop_record.get("parts", []) or []):
        if str(part.get("part_num") or "").strip() != part_num:
            continue
        if int(part.get("color_id", 0) or 0) != int(color_id):
            continue
        if (str(part.get("element_id") or "").strip() or None) != (element_id or None):
            continue
        part["qty"] = qty_value
        part["qty_text"] = qty_text_value
        updated = True
        break

    if not updated:
        raise HTTPException(status_code=404, detail="assigned part not found")

    _refresh_crop_next_qty_index(crop_record)
    crop_record["annotated_at"] = _iso_now()
    _write_labels(path, existing)
    return {"ok": True, "path": str(path), "crop": crop_record}


@router.post("/debug/ai-rank-crop")
async def ai_rank_crop(req: Request):
    data = await req.json()
    crop = data.get("crop") if isinstance(data.get("crop"), dict) else {}
    candidates = data.get("top_part_candidates") if isinstance(data.get("top_part_candidates"), list) else []
    deduped_candidates = service_dedupe_candidate_rows(list(candidates or []))
    stats: Dict[str, Any] = {
        "enabled": True,
        "calls": 0,
        "images_sent": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "failures": 0,
        "disabled": 0,
    }
    ai_rank_result = rank_crop_candidates(dict(crop or {}), deduped_candidates, stats)
    ai_rank_best_candidate = (
        f"{str(ai_rank_result.get('best_part_num') or '')} / c{int(ai_rank_result.get('best_color_id') or 0)}"
        if ai_rank_result.get("best_part_num") and ai_rank_result.get("best_color_id") is not None
        else ""
    )
    ai_rank_candidate = None
    reordered_candidates = list(deduped_candidates)
    matched_index = _coerce_int(ai_rank_result.get("matched_index"))
    if matched_index is not None and 0 <= matched_index < len(reordered_candidates):
        best_candidate = dict(reordered_candidates.pop(int(matched_index)) or {})
        ai_rank_candidate = best_candidate
        reordered_candidates = [best_candidate] + reordered_candidates
    ai_rank_debug = {
        "ai_enabled": True,
        "ai_rank_enabled": True,
        "candidate_count": len(deduped_candidates),
        "sent_candidate_count": min(5, len(deduped_candidates)),
        "returned_ai_result": dict(ai_rank_result or {}),
        "failure_reason": "" if ai_rank_best_candidate else str(ai_rank_result.get("reason") or "No AI pick returned."),
        "candidate_rows_preview": list(ai_rank_result.get("candidate_rows_preview", []) or []),
        "crop_image_included": bool(ai_rank_result.get("crop_image_included")),
        "candidate_images_included_count": int(ai_rank_result.get("candidate_images_included_count", 0) or 0),
        "raw_response_text": str(ai_rank_result.get("raw_response_text") or ""),
        "parsed_response_json": ai_rank_result.get("parsed_response_json"),
        "prompt_text": str(ai_rank_result.get("prompt_text") or ""),
        "crop_image_source": str(ai_rank_result.get("crop_image_source") or ""),
        "crop_image_exists": bool(ai_rank_result.get("crop_image_exists")),
        "candidate_img_urls_count": int(ai_rank_result.get("candidate_img_urls_count", 0) or 0),
        "first_candidate_img_url": str(ai_rank_result.get("first_candidate_img_url") or ""),
        "provider_selected": str(ai_rank_result.get("provider_selected") or ""),
        "azure_endpoint_present": bool(ai_rank_result.get("azure_endpoint_present")),
        "azure_api_key_present": bool(ai_rank_result.get("azure_api_key_present")),
        "azure_deployment_present": bool(ai_rank_result.get("azure_deployment_present")),
        "azure_api_version": str(ai_rank_result.get("azure_api_version") or ""),
        "client_created": bool(ai_rank_result.get("client_created")),
        "exception_type": str(ai_rank_result.get("exception_type") or ""),
        "exception_message": str(ai_rank_result.get("exception_message") or ""),
        "ai_skipped_reason": str(ai_rank_result.get("ai_skipped_reason") or ""),
    }
    return {
        "ok": True,
        "top_part_candidates": reordered_candidates,
        "ai_rank_best_candidate": ai_rank_best_candidate,
        "ai_rank_confidence": _coerce_float(ai_rank_result.get("confidence")),
        "ai_rank_reason": str(ai_rank_result.get("reason") or ""),
        "ai_rank_candidate": ai_rank_candidate,
        "ai_rank_debug": ai_rank_debug,
        "stats": stats,
    }


@router.get("/debug/export-training-data")
def export_training_data(
    set_num: str = Query(...),
    bag: Optional[int] = Query(None, ge=1),
):
    bag_number = int(bag or 1)
    payload = _build_export_training_payload(set_num, bag_number)
    export_path = _training_export_path(
        str(payload.get("set_num") or set_num),
        int(payload.get("bag", bag_number) or bag_number),
    )
    _write_export_training_payload(export_path, payload)
    filename = _coerce_label_filename(str(payload.get("set_num") or set_num), int(payload.get("bag", bag_number)))
    export_name = filename.replace(".json", "_export.json")
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'inline; filename="{export_name}"'},
    )


@router.get("/debug/bag-review-status", response_class=HTMLResponse)
def bag_review_status_page(
    set_num: str = Query(...),
    bag: Optional[int] = Query(None, ge=1),
):
    bag_number = int(bag or 1)
    review = _get_bag_review_status(set_num, bag_number)
    reviewed_examples = _load_verified_training_examples(set_num, bag_number)
    debug_report = _debug_verified_training_examples_report(set_num, bag_number)
    part_rows = _load_set_bag_parts_rows(set_num, bag_number)
    rows_html = "".join(
        f"<tr><td>{escape(str(row.get('part_num') or ''))}</td><td>{int(row.get('color_id', 0) or 0)}</td><td>{int(row.get('qty_total', 0) or 0)}</td><td>{int(row.get('verified_examples', 0) or 0)}</td><td>{escape(str(row.get('updated_at') or ''))}</td></tr>"
        for row in part_rows
    )
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Bag Review Status — {escape(str(set_num))} bag {bag_number}</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Bag Review Status — set {escape(str(set_num))} bag {bag_number}</h1>
  <p>status: {escape(str(review.get('status') or '')) or 'unset'}</p>
  <p>notes: {escape(str(review.get('notes') or '')) or '-'}</p>
  <p>reviewed good examples: {len(reviewed_examples)}</p>
  <h2>Debug Filter Report</h2>
  <pre>{escape(json.dumps(debug_report, indent=2))}</pre>
  <h2>set_bag_parts</h2>
  <table>
    <tr><th>part_num</th><th>color_id</th><th>qty_total</th><th>verified_examples</th><th>updated_at</th></tr>
    {rows_html or "<tr><td colspan='5'>No rows.</td></tr>"}
  </table>
</body>
</html>"""
    )


@router.post("/debug/mark-bag-complete")
def mark_bag_complete(
    set_num: str = Query(...),
    bag: Optional[int] = Query(None, ge=1),
):
    bag_number = int(bag or 1)
    review = _set_bag_review_status(set_num, bag_number, "complete")
    part_rows = _refresh_set_bag_parts_from_verified_examples(set_num, bag_number)
    reviewed_examples = _load_verified_training_examples(set_num, bag_number)
    return JSONResponse(
        {
            "ok": True,
            "set_num": str(set_num),
            "bag": bag_number,
            "status": str(review.get("status") or ""),
            "reviewed_good_examples": len(reviewed_examples),
            "set_bag_parts_count": len(part_rows),
            "set_bag_parts": part_rows,
        }
    )


@router.get("/debug/bag-qty-audit", response_class=HTMLResponse)
def bag_qty_audit_page(
    set_num: str = Query(...),
    bag: Optional[int] = Query(None, ge=1),
):
    bag_number = int(bag or 1)
    audit = _build_bag_qty_audit_rows(set_num, bag_number)
    audit_rows = list(audit.get("audit_rows", []) or [])
    set_bag_parts_rows = list(audit.get("set_bag_parts_rows", []) or [])
    debug_report = dict(audit.get("debug_report") or {})
    row_chunks: List[str] = []
    for row in audit_rows:
        source_text = ", ".join(
            f"{str(src.get('crop_id') or '')}/p{int(src.get('page', 0) or 0)}"
            for src in list(row.get("sources", []) or [])
        )
        status_text = str(row.get("status") or "")
        status_class = "ok" if status_text == "OK" else "bad"
        part_num = str(row.get("part_num") or "")
        color_id = int(row.get("color_id", 0) or 0)
        element_img_url = _lookup_element_image_url(part_num, color_id) or ""
        if element_img_url:
            thumb_html = f"<img src='{escape(element_img_url)}' style='max-width:72px;max-height:72px;border:1px solid #bbb' />"
        else:
            thumb_html = "<div class='missing-thumb'>missing</div>"
        row_chunks.append(
            f"<tr><td>{thumb_html}</td><td>{escape(part_num)}</td><td>{color_id}</td><td>{int(row.get('qty_total', 0) or 0)}</td><td>{int(row.get('reviewed_qty_total', 0) or 0)}</td><td>{escape(source_text)}</td><td class='{status_class}'>{escape(status_text)}</td></tr>"
        )
    rows_html = "".join(row_chunks)
    stored_rows_html = "".join(
        f"<tr><td>{escape(str(row.get('part_num') or ''))}</td><td>{int(row.get('color_id', 0) or 0)}</td><td>{int(row.get('qty_total', 0) or 0)}</td><td>{int(row.get('verified_examples', 0) or 0)}</td><td>{escape(str(row.get('updated_at') or ''))}</td></tr>"
        for row in set_bag_parts_rows
    )
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Bag Qty Audit — {escape(str(set_num))} bag {bag_number}</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1100px; margin-bottom: 18px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .ok {{ color: #166534; font-weight: 600; }}
    .bad {{ color: #b91c1c; font-weight: 600; }}
    .missing-thumb {{ width: 72px; height: 72px; border: 1px dashed #bbb; display: flex; align-items: center; justify-content: center; color: #777; background: #fafafa; }}
    pre {{ background: #f8fafc; border: 1px solid #ddd; padding: 10px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Bag Qty Audit — set {escape(str(set_num))} bag {bag_number}</h1>
  <p>reviewed good examples count: {int(audit.get('reviewed_examples_count', 0) or 0)}</p>
  <p>total qty from reviewed examples: {int(audit.get('reviewed_qty_total', 0) or 0)}</p>
  <p>total qty from set_bag_parts: {int(audit.get('set_bag_parts_qty_total', 0) or 0)}</p>
  <h2>Audit Rows</h2>
  <table>
    <tr><th>thumbnail</th><th>part_num</th><th>color_id</th><th>set_bag_parts qty_total</th><th>reviewed example qty sum</th><th>source crop_ids/pages</th><th>status</th></tr>
    {rows_html or "<tr><td colspan='7'>No rows.</td></tr>"}
  </table>
  <h2>set_bag_parts Rows</h2>
  <table>
    <tr><th>part_num</th><th>color_id</th><th>qty_total</th><th>verified_examples</th><th>updated_at</th></tr>
    {stored_rows_html or "<tr><td colspan='5'>No rows.</td></tr>"}
  </table>
  <h2>Skipped Examples And Reasons</h2>
  <pre>{escape(json.dumps(debug_report, indent=2))}</pre>
</body>
</html>"""
    )


@router.get("/debug/set-remaining-parts", response_class=HTMLResponse)
def set_remaining_parts_page(
    set_num: str = Query(...),
    bag: Optional[int] = Query(None, ge=1),
):
    bag_number = int(bag or 1)
    ledger = get_remaining_parts_for_set(str(set_num), up_to_bag=bag_number, include_current_reviewed=True)
    rows = list(ledger.get("rows", []) or [])
    row_chunks: List[str] = []
    for row in rows:
        img_url = str(row.get("img_url") or "").strip()
        if img_url:
            thumb_html = f"<img src='{escape(img_url)}' style='max-width:72px;max-height:72px;border:1px solid #bbb' />"
        else:
            thumb_html = "<div class='missing-thumb'>missing</div>"
        source_text = "/".join(list(row.get("source_types", []) or [])) or "-"
        row_chunks.append(
            f"<tr><td>{thumb_html}</td><td>{escape(str(row.get('part_num') or ''))}</td><td>{int(row.get('color_id', 0) or 0)}</td><td>{int(row.get('starting_qty', 0) or 0)}</td><td>{int(row.get('used_qty', 0) or 0)}</td><td>{int(row.get('remaining_qty', 0) or 0)}</td><td>{escape(source_text)}</td></tr>"
        )
    rows_html = "".join(row_chunks)
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Remaining Parts — {escape(str(set_num))} bag {bag_number}</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1100px; margin-bottom: 18px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .missing-thumb {{ width: 72px; height: 72px; border: 1px dashed #bbb; display: flex; align-items: center; justify-content: center; color: #777; background: #fafafa; }}
    pre {{ background: #f8fafc; border: 1px solid #ddd; padding: 10px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Remaining Parts — set {escape(str(set_num))} bag {bag_number}</h1>
  <p>starting qty total: {int(ledger.get('starting_qty_total', 0) or 0)}</p>
  <p>used qty total: {int(ledger.get('used_qty_total', 0) or 0)}</p>
  <p>remaining qty total: {int(ledger.get('remaining_qty_total', 0) or 0)}</p>
  <table>
    <tr><th>thumbnail</th><th>part_num</th><th>color_id</th><th>starting qty</th><th>used qty</th><th>remaining qty</th><th>source</th></tr>
    {rows_html or "<tr><td colspan='7'>No rows.</td></tr>"}
  </table>
  <pre>{escape(json.dumps(ledger, indent=2))}</pre>
</body>
</html>"""
    )


@router.get("/debug/missing-element-images", response_class=HTMLResponse)
def missing_element_images_page(
    set_num: str = Query(...),
):
    payload = _build_missing_element_images_report(str(set_num))
    rows = list(payload.get("rows", []) or [])
    row_chunks: List[str] = []
    for row in rows:
        row_chunks.append(
            f"<tr><td>{escape(str(row.get('part_num') or ''))}</td><td>{int(row.get('color_id', 0) or 0)}</td><td>{escape(str(row.get('color_name') or ''))}</td><td>{escape(str(row.get('missing_reason') or ''))}</td><td>{escape(str(row.get('candidate_source_checked') or ''))}</td><td>{escape(str(row.get('action_needed') or ''))}</td></tr>"
        )
    rows_html = "".join(row_chunks)
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Missing Element Images — {escape(str(set_num))}</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1200px; margin-bottom: 18px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    pre {{ background: #f8fafc; border: 1px solid #ddd; padding: 10px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Missing Element Images — set {escape(str(set_num))}</h1>
  <p>report path: {escape(str(_missing_images_report_path(str(set_num))))}</p>
  <p>missing rows: {int(payload.get('count', 0) or 0)}</p>
  <table>
    <tr><th>part_num</th><th>color_id</th><th>color_name</th><th>missing reason</th><th>candidate source checked</th><th>action needed</th></tr>
    {rows_html or "<tr><td colspan='6'>No missing rows.</td></tr>"}
  </table>
  <pre>{escape(json.dumps(payload, indent=2))}</pre>
</body>
</html>"""
    )


@router.get("/debug/training-review", response_class=HTMLResponse)
def training_review(
    set_num: str = Query(...),
    bag: Optional[int] = Query(None, ge=1),
):
    bag_number = int(bag or 1)
    export_path = _training_export_path(str(set_num), bag_number)
    if not export_path.exists():
        return HTMLResponse(f"<html><body><h1>Training Review</h1><p>No export file found for set {escape(str(set_num))} bag {bag_number}.</p></body></html>")

    try:
        payload = json.loads(export_path.read_text(encoding="utf-8") or "{}")
    except Exception:
        payload = {}
    examples = list(payload.get("examples", []) or [])
    labels_payload = _load_existing_labels(_label_store_path(str(set_num), bag_number))
    crops_by_id = dict(labels_payload.get("crops") or {})
    examples.sort(key=lambda item: (int((item or {}).get("page", 0) or 0), int((item or {}).get("step", 0) or 0), str((item or {}).get("crop_id") or "")))

    cards: List[str] = []
    for item in examples:
        example = item if isinstance(item, dict) else {}
        crop_id = str(example.get("crop_id") or "").strip()
        crop_record = dict(crops_by_id.get(crop_id) or {})
        crop_box = _coerce_box_list(crop_record.get("crop_box"))
        crop_image_path = str(example.get("crop_image_path") or crop_record.get("crop_image_path") or "").strip()
        crop_data_uri = ""
        if crop_image_path:
            img = cv2.imread(crop_image_path)
            if img is not None:
                if crop_box is not None:
                    crop_data_uri = str(_encode_contact_sheet_crop(img, crop_box, max_edge=360) or "")
                else:
                    crop_data_uri = _encode_debug_image_data_uri(img, max_width=360)
        element_img_url = str(example.get("element_img_url") or "").strip()
        left_html = (
            f"<img src='{escape(crop_data_uri)}' style='max-width:360px;border:1px solid #bbb'>" if crop_data_uri
            else "<div class='missing'>missing instruction crop</div>"
        )
        right_html = (
            f"<img src='{escape(element_img_url)}' style='max-width:220px;border:1px solid #bbb'>" if element_img_url
            else "<div class='missing'>missing element image</div>"
        )
        cards.append(
            f"""
            <div class='card'>
              <div class='images'>
                <div class='pane'><div class='label'>Instruction crop</div>{left_html}</div>
                <div class='pane'><div class='label'>Element image</div>{right_html}</div>
              </div>
              <div class='meta'>
                part_num: {escape(str(example.get('part_num') or ''))}<br/>
                color_id: {escape(str(example.get('color_id') or ''))}<br/>
                qty_text: {escape(str(example.get('qty_text') or ''))}<br/>
                crop_id: {escape(crop_id)}<br/>
                thumbnail_status: {escape(str(example.get('thumbnail_status') or ''))}
              </div>
            </div>
            """
        )

    return HTMLResponse(
        f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Training Review — {escape(str(set_num))} bag {bag_number}</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; background: #f6f7f9; color: #222; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 0 0 14px 0; }}
    .images {{ display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }}
    .pane {{ min-width: 220px; }}
    .label {{ font-weight: 600; margin: 0 0 6px 0; }}
    .meta {{ margin-top: 10px; font-size: 14px; line-height: 1.5; }}
    .missing {{ width: 220px; min-height: 80px; border: 1px dashed #bbb; display: flex; align-items: center; justify-content: center; color: #777; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>Training Review — set {escape(str(set_num))} bag {bag_number}</h1>
  <p>Examples: {len(examples)}</p>
  {''.join(cards) if cards else "<p>No examples found.</p>"}
</body>
</html>"""
    )


@router.get("/debug/instruction-buildability", response_class=HTMLResponse)
def instruction_buildability(
    set_num: str = Query(...),
    bag: Optional[int] = Query(None, ge=1),
    ai: Optional[int] = Query(0),
    ai_rank: Optional[int] = Query(0),
):
    bag_number = int(bag or 1)
    ai_rank_enabled = int(ai_rank or 0) == 1
    ai_rank_stats: Dict[str, Any] = {
        "enabled": ai_rank_enabled,
        "calls": 0,
        "images_sent": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "failures": 0,
        "disabled": 0,
    }
    parts_payload = load_instruction_set_parts(set_num)
    parts = list(parts_payload.get("parts", []) or [])
    color_ids = sorted(
        {
            int(part["color_id"])
            for part in parts
            if part.get("color_id") is not None and _coerce_int(part.get("color_id")) is not None
        }
    )
    lego_colors = sorted(
        _load_catalog_colors_for_ids(color_ids),
        key=lambda item: (
            1 if int(item.get("color_id", 0)) == 9999 else 0,
            str(item.get("color_name") or "").lower(),
            int(item.get("color_id", 0)),
        ),
    )
    print(f"[debug] legoColors count: {len(lego_colors)}")
    lego_colors_warning = "No colors loaded from set part library" if not lego_colors else ""
    training_examples = _load_saved_training_examples(str(set_num), bag_number)
    labels_path = _label_store_path(str(set_num), bag_number)
    labels_payload = _load_existing_labels(labels_path)
    remaining_parts_payload = get_remaining_parts_for_set(str(set_num), up_to_bag=bag_number, include_current_reviewed=True)
    remaining_parts_map = dict(remaining_parts_payload.get("remaining_parts_map") or {})
    crops = _build_instruction_callout_crops(str(set_num), bag_number, ai_enabled=int(ai or 0) == 1)
    manual_pages = _build_manual_crop_pages(str(set_num), bag_number)
    parts_by_key = {
        f"{str(part.get('part_num') or '').strip()}::{int(part.get('color_id', 0) or 0)}": part
        for part in parts
    }
    crop_ids_present = {str(crop.get("crop_id") or "").strip() for crop in crops}

    for crop in crops:
        saved_crop = dict(labels_payload.get("crops", {}).get(crop["crop_id"]) or {})
        crop["detected_qty_text"] = _coerce_str_list(crop.get("qty_text", []))
        crop_status = str(saved_crop.get("status") or "needs_adjust").strip().lower()
        crop["status"] = crop_status if crop_status in VALID_CROP_STATUSES else "needs_adjust"
        crop["is_hidden"] = crop["status"] == "hidden"
        crop["is_manual"] = _is_manual_crop_id(crop.get("crop_id"))
        crop["crop_box"] = _coerce_box_list(saved_crop.get("crop_box")) or _coerce_box_list(crop.get("crop_box")) or []
        crop["crop_box_format"] = str(saved_crop.get("crop_box_format") or crop.get("crop_box_format") or "xywh")
        crop["crop_image_path"] = str(saved_crop.get("crop_image_path") or crop.get("crop_image_path") or "")
        crop["confidence"] = _coerce_float(saved_crop.get("confidence", crop.get("confidence")))
        saved_crop_qty = _coerce_int_list(saved_crop.get("qty", []))
        saved_crop_qty_text = _coerce_str_list(saved_crop.get("qty_text", []))
        crop["manual_qty_text"] = saved_crop_qty_text
        if saved_crop_qty_text:
            crop["qty_text"] = saved_crop_qty_text
            crop["qty_numbers"] = saved_crop_qty
            crop["qty_label"] = ", ".join(saved_crop_qty_text) if saved_crop_qty_text else "none"
        else:
            crop["manual_qty_text"] = []
        crop["review_status"] = str(saved_crop.get("review_status") or "unreviewed")
        crop["annotator"] = str(saved_crop.get("annotator") or "")
        crop["annotated_at"] = str(saved_crop.get("annotated_at") or "")
        crop["adjustments"] = list(saved_crop.get("adjustments", []) or [])
        crop["notes"] = str(saved_crop.get("notes") or "")
        crop["next_qty_index"] = max(
            0,
            _coerce_int(saved_crop.get("next_qty_index")) or len(list(saved_crop.get("parts", []) or [])),
        )
        crop["parts"] = []
        for saved_part in list(saved_crop.get("parts", []) or []):
            normalized_part = _normalize_part_entry(saved_part)
            if not normalized_part["part_num"]:
                continue
            part_meta = parts_by_key.get(
                f"{normalized_part['part_num']}::{int(normalized_part['color_id'] or 0)}",
                {},
            )
            crop["parts"].append(
                {
                    "part_num": normalized_part["part_num"],
                    "color_id": normalized_part["color_id"],
                    "element_id": normalized_part["element_id"],
                    "color_name": str(normalized_part.get("color_name") or part_meta.get("color_name") or "n/a"),
                    "img_url": str(part_meta.get("img_url") or ""),
                    "qty": normalized_part["qty"],
                    "qty_text": normalized_part["qty_text"],
                    "selected_qty_label": normalized_part["qty_text"] or (str(normalized_part["qty"]) if normalized_part["qty"] is not None else "none"),
                    "part_bbox": normalized_part["part_bbox"],
                    "confidence": normalized_part["confidence"],
                }
            )
        crop["top_part_candidates"] = _debug_top_part_candidates_for_crop(
            crop,
            metallic_mode=bool(saved_crop.get("metallic_mode") or saved_crop.get("manual_metallic_mode")),
            set_num=str(set_num),
            remaining_parts=remaining_parts_map,
        )
        crop["ai_rank_debug"] = {
            "ai_enabled": int(ai or 0) == 1,
            "ai_rank_enabled": ai_rank_enabled,
            "candidate_count": len(list(crop.get("top_part_candidates", []) or [])),
            "sent_candidate_count": min(5, len(list(crop.get("top_part_candidates", []) or []))),
            "returned_ai_result": None,
            "failure_reason": "",
            "candidate_rows_preview": [],
            "crop_image_included": bool(crop.get("data_uri")),
            "candidate_images_included_count": 0,
            "raw_response_text": "",
            "parsed_response_json": None,
            "prompt_text": "",
            "crop_image_source": "missing",
            "crop_image_exists": bool(crop.get("data_uri")),
            "candidate_img_urls_count": len([item for item in list(crop.get("top_part_candidates", []) or []) if str(item.get("img_url") or "").strip()]),
            "first_candidate_img_url": str((next((item.get("img_url") for item in list(crop.get("top_part_candidates", []) or []) if str(item.get("img_url") or "").strip()), "")) or ""),
            "provider_selected": "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") else "none",
            "azure_endpoint_present": bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
            "azure_api_key_present": bool(os.getenv("AZURE_OPENAI_API_KEY")),
            "azure_deployment_present": bool(os.getenv("AZURE_OPENAI_DEPLOYMENT")),
            "azure_api_version": str(os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")),
            "client_created": False,
            "exception_type": "",
            "exception_message": "",
            "ai_skipped_reason": "",
        }
        slot_state = _crop_qty_slot_state(
            {"parts": crop["parts"]},
            crop.get("qty_numbers", []),
            crop.get("qty_text", []),
        )
        crop["slot_total"] = int(slot_state.get("total_slots", 0) or 0)
        crop["slot_filled"] = int(slot_state.get("filled_slots", 0) or 0)
        crop["next_qty_label"] = str(slot_state.get("next_qty_label") or "1x")
        crop["slots_full"] = bool(slot_state.get("slots_full"))
        crop["no_qty_detected"] = bool(slot_state.get("no_qty_detected"))
        crop["next_qty_index"] = int(slot_state.get("next_qty_index", crop.get("next_qty_index", 0)) or 0)
        crop["ai_rank_best_candidate"] = ""
        crop["ai_rank_confidence"] = None
        crop["ai_rank_reason"] = ""
        crop["ai_rank_candidate"] = None
        if ai_rank_enabled:
            ai_rank_result = rank_crop_candidates(crop, list(crop.get("top_part_candidates", []) or []), ai_rank_stats)
            crop["ai_rank_debug"]["returned_ai_result"] = dict(ai_rank_result or {})
            crop["ai_rank_debug"]["candidate_rows_preview"] = list(ai_rank_result.get("candidate_rows_preview", []) or [])
            crop["ai_rank_debug"]["crop_image_included"] = bool(ai_rank_result.get("crop_image_included"))
            crop["ai_rank_debug"]["candidate_images_included_count"] = int(ai_rank_result.get("candidate_images_included_count", 0) or 0)
            crop["ai_rank_debug"]["raw_response_text"] = str(ai_rank_result.get("raw_response_text") or "")
            crop["ai_rank_debug"]["parsed_response_json"] = ai_rank_result.get("parsed_response_json")
            crop["ai_rank_debug"]["prompt_text"] = str(ai_rank_result.get("prompt_text") or "")
            crop["ai_rank_debug"]["crop_image_source"] = str(ai_rank_result.get("crop_image_source") or "")
            crop["ai_rank_debug"]["crop_image_exists"] = bool(ai_rank_result.get("crop_image_exists"))
            crop["ai_rank_debug"]["candidate_img_urls_count"] = int(ai_rank_result.get("candidate_img_urls_count", 0) or 0)
            crop["ai_rank_debug"]["first_candidate_img_url"] = str(ai_rank_result.get("first_candidate_img_url") or "")
            crop["ai_rank_debug"]["provider_selected"] = str(ai_rank_result.get("provider_selected") or "")
            crop["ai_rank_debug"]["azure_endpoint_present"] = bool(ai_rank_result.get("azure_endpoint_present"))
            crop["ai_rank_debug"]["azure_api_key_present"] = bool(ai_rank_result.get("azure_api_key_present"))
            crop["ai_rank_debug"]["azure_deployment_present"] = bool(ai_rank_result.get("azure_deployment_present"))
            crop["ai_rank_debug"]["azure_api_version"] = str(ai_rank_result.get("azure_api_version") or "")
            crop["ai_rank_debug"]["client_created"] = bool(ai_rank_result.get("client_created"))
            crop["ai_rank_debug"]["exception_type"] = str(ai_rank_result.get("exception_type") or "")
            crop["ai_rank_debug"]["exception_message"] = str(ai_rank_result.get("exception_message") or "")
            crop["ai_rank_debug"]["ai_skipped_reason"] = str(ai_rank_result.get("ai_skipped_reason") or "")
            crop["ai_rank_best_candidate"] = (
                f"{str(ai_rank_result.get('best_part_num') or '')} / c{int(ai_rank_result.get('best_color_id') or 0)}"
                if ai_rank_result.get("best_part_num") and ai_rank_result.get("best_color_id") is not None
                else ""
            )
            crop["ai_rank_confidence"] = _coerce_float(ai_rank_result.get("confidence"))
            crop["ai_rank_reason"] = str(ai_rank_result.get("reason") or "")
            crop["ai_rank_debug"]["failure_reason"] = (
                ""
                if crop["ai_rank_best_candidate"]
                else str(ai_rank_result.get("reason") or "No AI pick returned.")
            )
            matched_index = _coerce_int(ai_rank_result.get("matched_index"))
            if matched_index is not None and 0 <= matched_index < len(list(crop.get("top_part_candidates", []) or [])):
                ranked_candidates = list(crop.get("top_part_candidates", []) or [])
                best_candidate = ranked_candidates.pop(int(matched_index))
                crop["ai_rank_candidate"] = dict(best_candidate or {})
                crop["top_part_candidates"] = [best_candidate] + ranked_candidates
        elif not crop["ai_rank_debug"]["candidate_count"]:
            crop["ai_rank_debug"]["failure_reason"] = "No local candidates available."
        else:
            crop["ai_rank_debug"]["failure_reason"] = "AI rank disabled."

    for saved_crop_id, saved_crop_data in dict(labels_payload.get("crops") or {}).items():
        crop_id = str(saved_crop_id or "").strip()
        if not crop_id or crop_id in crop_ids_present or not crop_id.startswith("manual_"):
            continue

        crop_dict = saved_crop_data if isinstance(saved_crop_data, dict) else {}
        crop_box = _coerce_box_list(crop_dict.get("crop_box"))
        crop_image_path = str(crop_dict.get("crop_image_path") or "").strip()
        data_uri = ""
        coords_label = "manual crop"
        if crop_box is not None:
            coords_label = (
                f"x={crop_box[0]}, y={crop_box[1]}, w={crop_box[2]}, h={crop_box[3]}"
            )

        if crop_box is not None and crop_image_path:
            img = cv2.imread(crop_image_path)
            if img is not None:
                data_uri = str(_encode_contact_sheet_crop(img, crop_box, max_edge=420) or "")

        manual_crop = {
            "crop_id": crop_id,
            "page": int(crop_dict.get("page", 0) or 0),
            "step": int(crop_dict.get("step", 0) or 0),
            "detected_qty_text": [],
            "manual_qty_text": _coerce_str_list(crop_dict.get("qty_text", [])),
            "qty_text": _coerce_str_list(crop_dict.get("qty_text", [])),
            "qty_numbers": _coerce_int_list(crop_dict.get("qty", [])),
            "qty_label": ", ".join(_coerce_str_list(crop_dict.get("qty_text", []))) or "none",
            "qty_source": "local",
            "ai_part_count": None,
            "ai_issues": [],
            "data_uri": data_uri,
            "coords_label": coords_label,
            "crop_box": crop_box or [],
            "crop_box_format": str(crop_dict.get("crop_box_format") or "xywh"),
            "crop_image_path": crop_image_path,
            "confidence": _coerce_float(crop_dict.get("confidence")),
            "status": (
                str(crop_dict.get("status") or "good").strip().lower()
                if str(crop_dict.get("status") or "good").strip().lower() in VALID_CROP_STATUSES
                else "good"
            ),
            "is_hidden": str(crop_dict.get("status") or "").strip().lower() == "hidden",
            "is_manual": True,
            "review_status": str(crop_dict.get("review_status") or "unreviewed"),
            "annotator": str(crop_dict.get("annotator") or ""),
            "annotated_at": str(crop_dict.get("annotated_at") or ""),
            "adjustments": list(crop_dict.get("adjustments", []) or []),
            "notes": str(crop_dict.get("notes") or ""),
            "next_qty_index": max(
                0,
                _coerce_int(crop_dict.get("next_qty_index")) or len(list(crop_dict.get("parts", []) or [])),
            ),
            "parts": [],
        }

        for saved_part in list(crop_dict.get("parts", []) or []):
            normalized_part = _normalize_part_entry(saved_part)
            if not normalized_part["part_num"]:
                continue
            part_meta = parts_by_key.get(
                f"{normalized_part['part_num']}::{int(normalized_part['color_id'] or 0)}",
                {},
            )
            manual_crop["parts"].append(
                {
                    "part_num": normalized_part["part_num"],
                    "color_id": normalized_part["color_id"],
                    "element_id": normalized_part["element_id"],
                    "color_name": str(normalized_part.get("color_name") or part_meta.get("color_name") or "n/a"),
                    "img_url": str(part_meta.get("img_url") or ""),
                    "qty": normalized_part["qty"],
                    "qty_text": normalized_part["qty_text"],
                    "selected_qty_label": normalized_part["qty_text"] or (str(normalized_part["qty"]) if normalized_part["qty"] is not None else "none"),
                    "part_bbox": normalized_part["part_bbox"],
                    "confidence": normalized_part["confidence"],
                }
            )
        manual_crop["top_part_candidates"] = _debug_top_part_candidates_for_crop(
            manual_crop,
            metallic_mode=bool(crop_dict.get("metallic_mode") or crop_dict.get("manual_metallic_mode")),
            set_num=str(set_num),
            remaining_parts=remaining_parts_map,
        )
        manual_crop["ai_rank_debug"] = {
            "ai_enabled": int(ai or 0) == 1,
            "ai_rank_enabled": ai_rank_enabled,
            "candidate_count": len(list(manual_crop.get("top_part_candidates", []) or [])),
            "sent_candidate_count": min(5, len(list(manual_crop.get("top_part_candidates", []) or []))),
            "returned_ai_result": None,
            "failure_reason": "",
            "candidate_rows_preview": [],
            "crop_image_included": bool(manual_crop.get("data_uri")),
            "candidate_images_included_count": 0,
            "raw_response_text": "",
            "parsed_response_json": None,
            "prompt_text": "",
            "crop_image_source": "missing",
            "crop_image_exists": bool(manual_crop.get("data_uri")),
            "candidate_img_urls_count": len([item for item in list(manual_crop.get("top_part_candidates", []) or []) if str(item.get("img_url") or "").strip()]),
            "first_candidate_img_url": str((next((item.get("img_url") for item in list(manual_crop.get("top_part_candidates", []) or []) if str(item.get("img_url") or "").strip()), "")) or ""),
            "provider_selected": "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") else "none",
            "azure_endpoint_present": bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
            "azure_api_key_present": bool(os.getenv("AZURE_OPENAI_API_KEY")),
            "azure_deployment_present": bool(os.getenv("AZURE_OPENAI_DEPLOYMENT")),
            "azure_api_version": str(os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")),
            "client_created": False,
            "exception_type": "",
            "exception_message": "",
            "ai_skipped_reason": "",
        }
        slot_state = _crop_qty_slot_state(
            {"parts": manual_crop["parts"]},
            manual_crop.get("qty_numbers", []),
            manual_crop.get("qty_text", []),
        )
        manual_crop["slot_total"] = int(slot_state.get("total_slots", 0) or 0)
        manual_crop["slot_filled"] = int(slot_state.get("filled_slots", 0) or 0)
        manual_crop["next_qty_label"] = str(slot_state.get("next_qty_label") or "1x")
        manual_crop["slots_full"] = bool(slot_state.get("slots_full"))
        manual_crop["no_qty_detected"] = bool(slot_state.get("no_qty_detected"))
        manual_crop["next_qty_index"] = int(slot_state.get("next_qty_index", manual_crop.get("next_qty_index", 0)) or 0)
        manual_crop["ai_rank_best_candidate"] = ""
        manual_crop["ai_rank_confidence"] = None
        manual_crop["ai_rank_reason"] = ""
        manual_crop["ai_rank_candidate"] = None
        if ai_rank_enabled:
            ai_rank_result = rank_crop_candidates(manual_crop, list(manual_crop.get("top_part_candidates", []) or []), ai_rank_stats)
            manual_crop["ai_rank_debug"]["returned_ai_result"] = dict(ai_rank_result or {})
            manual_crop["ai_rank_debug"]["candidate_rows_preview"] = list(ai_rank_result.get("candidate_rows_preview", []) or [])
            manual_crop["ai_rank_debug"]["crop_image_included"] = bool(ai_rank_result.get("crop_image_included"))
            manual_crop["ai_rank_debug"]["candidate_images_included_count"] = int(ai_rank_result.get("candidate_images_included_count", 0) or 0)
            manual_crop["ai_rank_debug"]["raw_response_text"] = str(ai_rank_result.get("raw_response_text") or "")
            manual_crop["ai_rank_debug"]["parsed_response_json"] = ai_rank_result.get("parsed_response_json")
            manual_crop["ai_rank_debug"]["prompt_text"] = str(ai_rank_result.get("prompt_text") or "")
            manual_crop["ai_rank_debug"]["crop_image_source"] = str(ai_rank_result.get("crop_image_source") or "")
            manual_crop["ai_rank_debug"]["crop_image_exists"] = bool(ai_rank_result.get("crop_image_exists"))
            manual_crop["ai_rank_debug"]["candidate_img_urls_count"] = int(ai_rank_result.get("candidate_img_urls_count", 0) or 0)
            manual_crop["ai_rank_debug"]["first_candidate_img_url"] = str(ai_rank_result.get("first_candidate_img_url") or "")
            manual_crop["ai_rank_debug"]["provider_selected"] = str(ai_rank_result.get("provider_selected") or "")
            manual_crop["ai_rank_debug"]["azure_endpoint_present"] = bool(ai_rank_result.get("azure_endpoint_present"))
            manual_crop["ai_rank_debug"]["azure_api_key_present"] = bool(ai_rank_result.get("azure_api_key_present"))
            manual_crop["ai_rank_debug"]["azure_deployment_present"] = bool(ai_rank_result.get("azure_deployment_present"))
            manual_crop["ai_rank_debug"]["azure_api_version"] = str(ai_rank_result.get("azure_api_version") or "")
            manual_crop["ai_rank_debug"]["client_created"] = bool(ai_rank_result.get("client_created"))
            manual_crop["ai_rank_debug"]["exception_type"] = str(ai_rank_result.get("exception_type") or "")
            manual_crop["ai_rank_debug"]["exception_message"] = str(ai_rank_result.get("exception_message") or "")
            manual_crop["ai_rank_debug"]["ai_skipped_reason"] = str(ai_rank_result.get("ai_skipped_reason") or "")
            manual_crop["ai_rank_best_candidate"] = (
                f"{str(ai_rank_result.get('best_part_num') or '')} / c{int(ai_rank_result.get('best_color_id') or 0)}"
                if ai_rank_result.get("best_part_num") and ai_rank_result.get("best_color_id") is not None
                else ""
            )
            manual_crop["ai_rank_confidence"] = _coerce_float(ai_rank_result.get("confidence"))
            manual_crop["ai_rank_reason"] = str(ai_rank_result.get("reason") or "")
            manual_crop["ai_rank_debug"]["failure_reason"] = (
                ""
                if manual_crop["ai_rank_best_candidate"]
                else str(ai_rank_result.get("reason") or "No AI pick returned.")
            )
            matched_index = _coerce_int(ai_rank_result.get("matched_index"))
            if matched_index is not None and 0 <= matched_index < len(list(manual_crop.get("top_part_candidates", []) or [])):
                ranked_candidates = list(manual_crop.get("top_part_candidates", []) or [])
                best_candidate = ranked_candidates.pop(int(matched_index))
                manual_crop["ai_rank_candidate"] = dict(best_candidate or {})
                manual_crop["top_part_candidates"] = [best_candidate] + ranked_candidates
        elif not manual_crop["ai_rank_debug"]["candidate_count"]:
            manual_crop["ai_rank_debug"]["failure_reason"] = "No local candidates available."
        else:
            manual_crop["ai_rank_debug"]["failure_reason"] = "AI rank disabled."
        crops.append(manual_crop)

    crops.sort(
        key=lambda item: (
            int(item.get("page", 0) or 0),
            int(item.get("step", 0) or 0),
            1 if str(item.get("crop_id", "")).startswith("manual_") else 0,
            str(item.get("crop_id", "")),
        )
    )

    crop_cards_html = (
        "\n".join(
            f"""
            <div
              id="{escape(crop['crop_id'])}"
              class="crop-card crop-status-{escape(str(crop.get('status') or 'needs_adjust'))}"
              data-hidden="{str(bool(crop.get('is_hidden'))).lower()}"
              data-manual="{str(bool(crop.get('is_manual'))).lower()}"
            >
              <button
                class="crop-select"
                type="button"
                onclick="selectCrop('{escape(crop['crop_id'])}')"
              >
              <div class="crop-meta">
                <strong>{escape(crop['crop_id'])}</strong><br/>
                page {int(crop['page'])} | step {int(crop['step']) if int(crop['step']) > 0 else "?"}<br/>
                detected_qty_text: {escape(", ".join(list(crop.get('detected_qty_text', []) or [])) or "none")}<br/>
                manual_qty_text: {escape(", ".join(list(crop.get('manual_qty_text', []) or [])) or "none")}<br/>
                qty_text: {escape(", ".join(list(crop.get('qty_text', []) or [])) or "none")}<br/>
                qty: <span id="qty-label-{escape(crop['crop_id'])}">{escape(str(crop['qty_label']))}</span><br/>
                qty source: {escape(str(crop.get('qty_source') or 'local'))}<br/>
                ai part_count: {escape('—' if crop.get('ai_part_count') is None else str(crop.get('ai_part_count')))}<br/>
                ai issues: {escape(', '.join(list(crop.get('ai_issues', []) or [])) or '—')}<br/>
                top candidates: {escape(', '.join(f"{str(item.get('part_num') or '')} ({float(item.get('score', 0.0) or 0.0):.1f})" for item in list(crop.get('top_part_candidates', []) or [])[:3]) or 'none')}<br/>
                ai rank best: {escape(str(crop.get('ai_rank_best_candidate') or '—'))}<br/>
                ai rank confidence/reason: {escape('—' if crop.get('ai_rank_confidence') is None else f"{float(crop.get('ai_rank_confidence') or 0.0):.2f}")} / {escape(str(crop.get('ai_rank_reason') or '—'))}<br/>
                slots: <span id="slots-label-{escape(crop['crop_id'])}">{int(crop.get('slot_filled', 0) or 0)} / {int(crop.get('slot_total', 0) or 0)} filled</span><br/>
                next qty: <span id="next-qty-label-{escape(crop['crop_id'])}">{escape(str(crop.get('next_qty_label') or '1x'))}</span><br/>
                status: <span id="status-label-{escape(crop['crop_id'])}" class="crop-status-label">{escape(str(crop.get('status') or 'needs_adjust'))}</span><br/>
                metallic mode: <span id="metallic-label-{escape(crop['crop_id'])}" class="crop-status-label">OFF</span><br/>
                <span id="qty-warning-{escape(crop['crop_id'])}" class="crop-warning">{escape('No qty detected' if bool(crop.get('no_qty_detected')) else ('All qty slots filled' if bool(crop.get('slots_full')) else ''))}</span><br/>
                <span class="coords">{escape(str(crop['coords_label']))}</span>
              </div>
              <div class="crop-image">
                {_build_crop_image_html(crop)}
              </div>
              </button>
              <div class="crop-actions">
                <button type="button" class="status-btn" data-status="good" onclick="setCropStatus(event, '{escape(crop['crop_id'])}', 'good')">Good</button>
                <button type="button" class="status-btn" data-status="bad" onclick="setCropStatus(event, '{escape(crop['crop_id'])}', 'bad')">Bad</button>
                <button type="button" class="status-btn" data-status="needs_adjust" onclick="setCropStatus(event, '{escape(crop['crop_id'])}', 'needs_adjust')">Needs Adjust</button>
                <button type="button" class="remove-btn delete-crop-btn" onclick="deleteCrop(event, '{escape(crop['crop_id'])}')">{'Delete Crop' if bool(crop.get('is_manual')) else 'Hide Crop'}</button>
              </div>
              {(
                f'''<div class="crop-qty-editor">
                <div><strong>AI-ranked candidate</strong></div>
                <div>AI confidence: {escape('—' if crop.get('ai_rank_confidence') is None else f"{float(crop.get('ai_rank_confidence') or 0.0):.2f}")}</div>
                <div>AI reason: {escape(str(crop.get('ai_rank_reason') or '—'))}</div>
                <div class="crop-qty-row">
                  <button type="button" class="remove-btn" onclick="useAiPick(event, '{escape(crop['crop_id'])}')">Use AI Pick</button>
                </div>
              </div>'''
                if ai_rank_enabled and crop.get("ai_rank_candidate")
                else ""
              )}
              <div class="crop-qty-editor">
                <label for="qty-input-{escape(crop['crop_id'])}">Qty text</label>
                <div class="crop-qty-row">
                  <input id="qty-input-{escape(crop['crop_id'])}" class="crop-qty-input" type="text" value="{escape(','.join(list(crop.get('qty_text', []) or [])))}" placeholder="1x,2x,1x" />
                  <button type="button" class="remove-btn" onclick="updateCropQty(event, '{escape(crop['crop_id'])}')">Save Qty</button>
                </div>
                <div class="crop-qty-row">
                  <label class="hidden-toggle" for="metallic-toggle-{escape(crop['crop_id'])}">
                    <input id="metallic-toggle-{escape(crop['crop_id'])}" type="checkbox" onchange="toggleMetallicMode(event, '{escape(crop['crop_id'])}', this.checked)" />
                    Metallic mode
                  </label>
                </div>
              </div>
              <div class="assigned-parts" id="assigned-{escape(crop['crop_id'])}"></div>
            </div>
            """
            for crop in crops
        )
        if crops
        else "<div class='empty'>No real callout crops were found for this bag with the current debug pipeline.</div>"
    )

    parts_tiles_html = "\n".join(
        f"""
        <button
          class="part-tile"
          data-part-num="{escape(str(part['part_num']))}"
          data-part-color-id="{int(part['color_id'])}"
          data-color-id="{int(part['color_id'])}"
          data-required-qty="{int(part.get('qty', 0) or 0)}"
          data-part-color-name="{escape(str(part.get('color_name') or 'n/a'))}"
          type="button"
          onclick="selectTile('{escape(str(part['part_num']))}', {int(part['color_id'])}, '{escape(str(part.get('element_id') or ''))}', '{escape(str(part.get('color_name') or ''))}')"
        >
          <div class="part-thumb">
            {f'<img src="{escape(str(part.get("img_url") or ""))}" alt="{escape(str(part["part_num"]))}" loading="lazy" />' if part.get("img_url") else '<div class="crop-missing">No image</div>'}
          </div>
          <div class="part-meta">
            <strong>{escape(str(part['part_num']))}</strong><br/>
            color: {escape(str(part.get('color_name') or 'n/a'))}<br/>
            Required / Assigned / Remaining:
            <span class="required-qty">{int(part.get('qty', 0) or 0)}</span> /
            <span class="assigned-qty">0</span> /
            <span class="remaining-qty">{int(part.get('qty', 0) or 0)}</span><br/>
            <span class="over-assigned-note"></span>
            element: {escape(str(part.get('element_id') or 'n/a'))}
            <div class="part-tile-actions">
              <span
                class="remove-btn where-used-btn"
                role="button"
                tabindex="0"
                onclick="event.stopPropagation(); openWhereUsed('{escape(str(part['part_num']))}', {int(part['color_id'])});"
              >Where used</span>
            </div>
          </div>
        </button>
        """
        for part in parts
    )
    manual_pages_html = "\n".join(
        f"""
        <div
          class="manual-page-card"
          data-page="{int(page_item['page'])}"
          data-image-path="{escape(str(page_item['image_path']))}"
        >
          <div class="manual-page-header">
            <strong>Page {int(page_item['page'])}</strong>
          </div>
          <div
            class="manual-page-canvas"
            id="manual-page-canvas-{int(page_item['page'])}"
            data-page="{int(page_item['page'])}"
            data-image-width="{int(page_item['width'])}"
            data-image-height="{int(page_item['height'])}"
          >
            <img
              src="{escape(str(page_item['data_uri']))}"
              alt="Full page {int(page_item['page'])}"
              loading="lazy"
              draggable="false"
            />
            <div class="manual-selection-box" id="manual-selection-box-{int(page_item['page'])}"></div>
          </div>
          <div class="manual-page-controls">
            <label class="manual-step-label" for="manual-step-{int(page_item['page'])}">Step</label>
            <input id="manual-step-{int(page_item['page'])}" class="manual-step-input" type="number" min="0" step="1" placeholder="Enter step" />
            <button type="button" class="manual-save-btn" onclick="saveManualCrop({int(page_item['page'])})">Save Manual Crop</button>
          </div>
          <div class="manual-selection-readout" id="manual-selection-readout-{int(page_item['page'])}">
            Drag on the page image to select a crop.
          </div>
        </div>
        """
        for page_item in manual_pages
    )

    crops_json = json.dumps(crops)
    parts_json = json.dumps(parts)
    lego_colors_json = json.dumps(lego_colors)
    training_examples_json = json.dumps(training_examples)
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Instruction Buildability Debug</title>
      <style>
        body {{
          margin: 0;
          padding: 24px;
          background: #f4f7fb;
          color: #17212b;
          font-family: Arial, sans-serif;
        }}
        .shell {{
          max-width: 1600px;
          margin: 0 auto;
        }}
        .hero, .panel {{
          background: #fff;
          border: 1px solid #d6dee8;
          border-radius: 16px;
          box-shadow: 0 10px 28px rgba(20, 42, 58, 0.08);
          padding: 18px 20px;
          margin-bottom: 18px;
        }}
        .hero h1, .panel h2 {{
          margin: 0 0 10px;
        }}
        .hero p {{
          margin: 4px 0;
        }}
        .status-line {{
          margin-top: 10px;
          padding: 10px 12px;
          background: #eef5ff;
          border-radius: 12px;
          font-weight: 700;
        }}
        .manual-pages-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
          gap: 16px;
        }}
        .manual-page-card {{
          border: 1px solid #d6dee8;
          border-radius: 16px;
          background: #fbfdff;
          padding: 14px;
        }}
        .manual-page-header {{
          margin-bottom: 10px;
          font-size: 14px;
        }}
        .manual-page-canvas {{
          position: relative;
          width: 100%;
          border: 1px solid #d6dee8;
          border-radius: 14px;
          overflow: hidden;
          background: #eef3f8;
          cursor: crosshair;
          user-select: none;
          touch-action: none;
        }}
        .manual-page-canvas img {{
          width: 100%;
          height: auto;
          display: block;
          user-select: none;
          pointer-events: none;
        }}
        .manual-selection-box {{
          position: absolute;
          border: 2px solid #cf1f1f;
          background: rgba(207, 31, 31, 0.12);
          display: none;
          pointer-events: none;
        }}
        .manual-selection-box.visible {{
          display: block;
        }}
        .manual-page-controls {{
          display: flex;
          align-items: center;
          gap: 10px;
          margin-top: 12px;
        }}
        .manual-step-label {{
          font-size: 13px;
          font-weight: 700;
        }}
        .manual-step-input {{
          width: 110px;
          padding: 8px 10px;
          border: 1px solid #c4d2e1;
          border-radius: 10px;
          font-size: 14px;
        }}
        .manual-save-btn {{
          border: 1px solid #17212b;
          background: #17212b;
          color: #fff;
          border-radius: 999px;
          padding: 8px 14px;
          font-size: 13px;
          cursor: pointer;
        }}
        .manual-selection-readout {{
          margin-top: 10px;
          color: #536576;
          font-size: 13px;
          line-height: 1.4;
        }}
        .crop-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
          gap: 14px;
        }}
        .crop-panel-head {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
          margin-bottom: 12px;
        }}
        .crop-panel-head h2 {{
          margin: 0;
        }}
        .hidden-toggle {{
          display: inline-flex;
          align-items: center;
          gap: 8px;
          color: #536576;
          font-size: 13px;
        }}
        .crop-toolbar {{
          display: inline-flex;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
        }}
        .toolbar-link {{
          display: inline-flex;
          align-items: center;
          justify-content: center;
          text-decoration: none;
        }}
        .parts-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 14px;
        }}
        .suggested-panel {{
          margin-bottom: 16px;
          padding: 14px;
          border: 1px solid #d6dee8;
          border-radius: 14px;
          background: #fbfdff;
        }}
        .suggested-panel.flash-highlight {{
          border-color: #17212b;
          box-shadow: 0 0 0 3px rgba(23, 33, 43, 0.12);
          transition: border-color 0.18s ease, box-shadow 0.18s ease;
        }}
        .colour-picker-panel {{
          margin-bottom: 16px;
          padding: 14px;
          border: 1px solid #d6dee8;
          border-radius: 14px;
          background: #fbfdff;
        }}
        .colour-picker-panel h3 {{
          margin: 0 0 8px;
          font-size: 16px;
        }}
        .colour-picker-help {{
          margin: 0 0 10px;
          color: #627283;
          font-size: 13px;
        }}
        .picker-layout {{
          display: grid;
          grid-template-columns: minmax(260px, 420px) 1fr;
          gap: 14px;
          align-items: start;
        }}
        .picker-canvas-wrap {{
          border: 1px solid #d6dee8;
          border-radius: 12px;
          overflow: hidden;
          background: #f4f7fb;
          min-height: 180px;
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        .picker-canvas {{
          display: block;
          max-width: 100%;
          height: auto;
          cursor: crosshair;
          pointer-events: auto;
        }}
        .picker-empty {{
          padding: 16px;
          color: #627283;
          font-size: 13px;
        }}
        .picked-rgb-row {{
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 10px;
          flex-wrap: wrap;
        }}
        .colour-swatch {{
          width: 22px;
          height: 22px;
          border-radius: 999px;
          border: 1px solid rgba(23, 33, 43, 0.18);
          flex: 0 0 22px;
        }}
        .colour-match-list {{
          display: flex;
          flex-direction: column;
          gap: 8px;
        }}
        .colour-match {{
          display: flex;
          align-items: center;
          gap: 10px;
          width: 100%;
          text-align: left;
          border: 1px solid #d6dee8;
          border-radius: 12px;
          background: #fff;
          padding: 10px 12px;
          cursor: pointer;
        }}
        .colour-match.active {{
          border-color: #17212b;
          box-shadow: inset 0 0 0 1px #17212b;
          background: #f5f8fc;
        }}
        .colour-match-meta {{
          flex: 1;
          font-size: 13px;
          line-height: 1.35;
        }}
        .colour-match-distance {{
          color: #627283;
          font-size: 12px;
          white-space: nowrap;
        }}
        .colour-picker-actions {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          margin: 10px 0 8px;
          flex-wrap: wrap;
        }}
        .colour-picker-actions strong {{
          font-size: 13px;
          color: #17212b;
        }}
        .picker-diagnostics {{
          margin-top: 12px;
          padding: 12px;
          border: 1px dashed #d6dee8;
          border-radius: 12px;
          background: #fff;
          font-size: 12px;
          line-height: 1.45;
          color: #425364;
        }}
        .picker-diagnostics strong {{
          color: #17212b;
        }}
        .suggested-panel h3 {{
          margin: 0 0 8px;
          font-size: 16px;
        }}
        .suggested-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 12px;
        }}
        .suggested-empty {{
          padding: 12px;
          border: 1px dashed #d6dee8;
          border-radius: 12px;
          color: #627283;
          font-size: 13px;
          background: #fff;
        }}
        .suggested-part {{
          border: 1px solid #d6dee8;
          border-radius: 14px;
          padding: 12px;
          background: #fff;
        }}
        .suggested-part .part-thumb {{
          margin-top: 0;
          min-height: 120px;
        }}
        .suggested-part-actions {{
          margin-top: 10px;
          display: flex;
          justify-content: flex-end;
        }}
        .parts-toolbar {{
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 12px;
          flex-wrap: wrap;
        }}
        .slot-status-note {{
          color: #536576;
          font-size: 13px;
          font-weight: 700;
        }}
        .parts-filter-toggle {{
          display: inline-flex;
          gap: 8px;
          padding: 4px;
          border-radius: 999px;
          background: #eef3f8;
          border: 1px solid #d6dee8;
        }}
        .parts-filter-btn {{
          border: 0;
          background: transparent;
          border-radius: 999px;
          padding: 7px 12px;
          font-size: 13px;
          cursor: pointer;
          color: #425364;
        }}
        .parts-filter-btn.active {{
          background: #17212b;
          color: #fff;
        }}
        .parts-filter-status {{
          color: #536576;
          font-size: 13px;
        }}
        .bag-summary {{
          color: #425364;
          font-size: 13px;
          font-weight: 700;
        }}
        .crop-card, .part-tile {{
          width: 100%;
          background: #fff;
          border: 2px solid #b8c8da;
          border-radius: 14px;
          padding: 12px;
          text-align: left;
          transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease, background 0.15s ease;
        }}
        .crop-card:hover, .part-tile:hover {{
          transform: translateY(-1px);
          box-shadow: 0 8px 20px rgba(32, 52, 70, 0.10);
        }}
        .crop-card.selected {{
          border-color: #cf1f1f;
          background: #fff1f1;
          box-shadow: 0 10px 24px rgba(207, 31, 31, 0.18);
        }}
        .crop-card.crop-status-good {{
          border-color: #2f8f5b;
        }}
        .crop-card.crop-status-bad {{
          border-color: #b44141;
        }}
        .crop-card.crop-status-needs_adjust {{
          border-color: #b88a2d;
        }}
        .crop-card.crop-status-hidden {{
          border-color: #7b8794;
          background: #f4f7fb;
        }}
        .crop-card.is-hidden-crop {{
          display: none;
        }}
        .crop-select {{
          width: 100%;
          padding: 0;
          border: 0;
          background: transparent;
          text-align: left;
          cursor: pointer;
        }}
        .crop-meta, .part-meta {{
          font-size: 14px;
          line-height: 1.4;
        }}
        .crop-status-label {{
          display: inline-block;
          padding: 2px 8px;
          border-radius: 999px;
          background: #eef5ff;
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.02em;
        }}
        .coords {{
          color: #627283;
          font-size: 12px;
        }}
        .crop-warning {{
          color: #9a5a14;
          font-size: 12px;
          font-weight: 700;
        }}
        .part-thumb {{
          margin-top: 10px;
          min-height: 140px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #f4f7fb;
          border: 1px solid #d6dee8;
          border-radius: 12px;
          overflow: hidden;
        }}
        .part-thumb img {{
          max-width: 100%;
          max-height: 220px;
          display: block;
        }}
        .part-tile.filtered-out {{
          display: none;
        }}
        .part-tile:disabled, .remove-btn:disabled {{
          opacity: 0.55;
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }}
        .part-tile.over-assigned {{
          opacity: 0.56;
          background: #f5f7fa;
          border-color: #c8d1db;
        }}
        .over-assigned-note {{
          display: inline-block;
          min-height: 16px;
          color: #b44141;
          font-size: 12px;
          font-weight: 700;
        }}
        .part-tile-actions {{
          margin-top: 8px;
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }}
        .where-used-btn {{
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }}
        .crop-image {{
          width: 100%;
          min-height: 180px;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-top: 10px;
          background: #f4f7fb;
          border: 1px solid #d6dee8;
          border-radius: 12px;
          overflow: hidden;
        }}
        .crop-image img {{
          max-width: 100%;
          max-height: 200px;
          height: auto;
          object-fit: contain;
          display: block;
          cursor: zoom-in;
        }}
        .crop-missing, .empty {{
          padding: 18px;
          color: #6c7c8d;
        }}
        .crop-actions {{
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 12px;
        }}
        .crop-qty-editor {{
          margin-top: 12px;
        }}
        .crop-qty-editor label {{
          display: block;
          font-size: 12px;
          font-weight: 700;
          color: #536576;
          margin-bottom: 6px;
        }}
        .crop-qty-row {{
          display: flex;
          gap: 8px;
          align-items: center;
        }}
        .crop-qty-input {{
          flex: 1;
          min-width: 0;
          padding: 8px 10px;
          border: 1px solid #c4d2e1;
          border-radius: 10px;
          font-size: 13px;
          background: #fff;
        }}
        .status-btn, .remove-btn {{
          border: 1px solid #b8c8da;
          background: #f7f9fc;
          border-radius: 999px;
          padding: 6px 10px;
          font-size: 12px;
          cursor: pointer;
        }}
        .delete-crop-btn {{
          margin-left: auto;
        }}
        .status-btn.active {{
          background: #17212b;
          color: #fff;
          border-color: #17212b;
        }}
        .assigned-parts {{
          margin-top: 12px;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }}
        .assigned-empty {{
          padding: 10px 12px;
          background: #f4f7fb;
          border: 1px dashed #d6dee8;
          border-radius: 12px;
          color: #6c7c8d;
          font-size: 13px;
        }}
        .assigned-part {{
          display: flex;
          gap: 10px;
          align-items: center;
          padding: 10px;
          background: #f8fbff;
          border: 1px solid #d6dee8;
          border-radius: 12px;
        }}
        .assigned-part-thumb {{
          width: 52px;
          height: 52px;
          flex: 0 0 52px;
          border-radius: 10px;
          overflow: hidden;
          background: #fff;
          border: 1px solid #d6dee8;
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        .assigned-part-thumb img {{
          max-width: 100%;
          max-height: 100%;
          display: block;
        }}
        .assigned-part-meta {{
          flex: 1;
          font-size: 13px;
          line-height: 1.35;
        }}
        .assigned-part-actions {{
          display: flex;
          flex-direction: column;
          gap: 8px;
          align-items: flex-end;
        }}
        .assigned-qty-row {{
          display: flex;
          gap: 6px;
          align-items: center;
          flex-wrap: wrap;
          justify-content: flex-end;
        }}
        .assigned-qty-input {{
          width: 64px;
          padding: 6px 8px;
          border: 1px solid #c8d2dc;
          border-radius: 8px;
          background: #fff;
          font-size: 13px;
        }}
        .zoom-modal {{
          position: fixed;
          inset: 0;
          display: none;
          align-items: center;
          justify-content: center;
          padding: 24px;
          background: rgba(12, 19, 26, 0.78);
          z-index: 1000;
        }}
        .zoom-modal.open {{
          display: flex;
        }}
        .zoom-modal-panel {{
          position: relative;
          max-width: min(92vw, 1100px);
          max-height: 90vh;
          padding: 18px;
          background: #fff;
          border-radius: 18px;
          box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
        }}
        .zoom-modal-close {{
          position: absolute;
          top: 10px;
          right: 10px;
          border: 0;
          width: 36px;
          height: 36px;
          border-radius: 999px;
          background: rgba(23, 33, 43, 0.08);
          font-size: 20px;
          line-height: 1;
          cursor: pointer;
        }}
        .zoom-modal-image {{
          max-width: 100%;
          max-height: calc(90vh - 72px);
          width: auto;
          height: auto;
          object-fit: contain;
          display: block;
        }}
        .zoom-modal-caption {{
          margin-top: 10px;
          color: #4f6070;
          font-size: 14px;
          text-align: center;
        }}
        .usage-modal-grid {{
          display: flex;
          flex-direction: column;
          gap: 10px;
          min-width: min(92vw, 780px);
          max-width: min(92vw, 780px);
          max-height: calc(90vh - 96px);
          overflow: auto;
        }}
        .where-used-toolbar {{
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          align-items: flex-end;
          justify-content: space-between;
          margin-bottom: 10px;
        }}
        .where-used-heading {{
          flex: 1 1 300px;
          min-width: 0;
        }}
        .where-used-title {{
          margin: 0;
          color: #213446;
          font-size: 15px;
          font-weight: 700;
          text-align: left;
        }}
        .where-used-summary {{
          margin: 6px 0 0;
          color: #4f6070;
          font-size: 13px;
          text-align: left;
        }}
        .where-used-tools {{
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          align-items: center;
          justify-content: flex-end;
        }}
        .where-used-filter {{
          min-width: 220px;
          padding: 8px 10px;
          border-radius: 10px;
          border: 1px solid #c9d5e3;
          background: #fff;
          font-size: 13px;
        }}
        .where-used-empty {{
          color: #627283;
          font-size: 13px;
        }}
        .usage-row {{
          display: grid;
          grid-template-columns: 88px 1fr auto;
          gap: 12px;
          align-items: center;
          padding: 12px;
          border: 1px solid #d6dee8;
          border-radius: 14px;
          background: #f8fbff;
        }}
        .usage-row.current-crop {{
          border-color: #2d6cdf;
          background: #edf4ff;
          box-shadow: 0 0 0 2px rgba(45, 108, 223, 0.12);
        }}
        .usage-thumb {{
          width: 88px;
          height: 88px;
          border-radius: 12px;
          overflow: hidden;
          border: 1px solid #d6dee8;
          background: #fff;
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        .usage-thumb img {{
          max-width: 100%;
          max-height: 100%;
          display: block;
        }}
        .usage-meta {{
          font-size: 13px;
          line-height: 1.4;
        }}
        .usage-current-badge {{
          display: inline-block;
          margin-left: 8px;
          padding: 2px 8px;
          border-radius: 999px;
          background: #2d6cdf;
          color: #fff;
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.02em;
          vertical-align: middle;
        }}
        .usage-actions {{
          display: flex;
          flex-direction: column;
          gap: 8px;
          align-items: stretch;
        }}
        .save-note {{
          color: #627283;
          font-size: 13px;
        }}
        .warning-note {{
          margin-top: 10px;
          padding: 10px 12px;
          border-radius: 12px;
          background: #fff4d8;
          color: #8a5a00;
          font-size: 13px;
          font-weight: 700;
        }}
        .selected-crop-workspace {{
          position: relative;
          margin-bottom: 16px;
          width: 100%;
          max-height: none;
          overflow: visible;
          padding: 14px;
          border: 1px solid #d6dee8;
          border-radius: 16px;
          background: linear-gradient(180deg, #ffffff 0%, #f6f9fd 100%);
          box-shadow: 0 10px 24px rgba(23, 33, 43, 0.06);
        }}
        .selected-crop-workspace.empty {{
          color: #627283;
        }}
        .selected-crop-layout {{
          display: grid;
          grid-template-columns: minmax(300px, 0.95fr) minmax(320px, 1fr) minmax(360px, 1.05fr);
          gap: 14px;
          align-items: start;
        }}
        .selected-crop-column {{
          display: flex;
          flex-direction: column;
          gap: 10px;
        }}
        .selected-crop-preview {{
          width: 100%;
          min-height: 180px;
          border: 1px solid #d6dee8;
          border-radius: 14px;
          background: #fff;
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
        }}
        .selected-crop-preview img {{
          max-width: 100%;
          max-height: 260px;
          display: block;
          cursor: pointer;
        }}
        .selected-crop-meta {{
          font-size: 13px;
          line-height: 1.4;
        }}
        .selected-crop-meta-card,
        .selected-crop-assignments,
        .selected-crop-ai,
        .selected-crop-picker {{
          padding: 10px 12px;
          border: 1px solid #d6dee8;
          border-radius: 12px;
          background: #fff;
        }}
        .selected-crop-actions {{
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-top: 8px;
        }}
        .selected-crop-status {{
          margin-top: 8px;
          font-size: 12px;
          color: #627283;
        }}
        .selected-crop-ai {{
          margin-top: 0;
        }}
        .selected-crop-assignments {{
          margin-top: 0;
        }}
        .library-assignments .assigned-part {{
          background: #fff;
          padding: 7px 8px;
          gap: 8px;
        }}
        .library-assignments .assigned-part-thumb {{
          width: 40px;
          height: 40px;
          flex: 0 0 40px;
        }}
        .library-assignments .assigned-part-meta,
        .library-assignments .assigned-part-actions,
        .library-assignments .assigned-qty-input {{
          font-size: 12px;
        }}
        .selected-crop-candidate-list {{
          display: flex;
          flex-direction: column;
          gap: 6px;
          margin-top: 8px;
          max-height: 240px;
          overflow: auto;
        }}
        .selected-crop-candidate-row {{
          display: grid;
          grid-template-columns: 34px 1fr;
          gap: 8px;
          align-items: center;
          padding: 6px 8px;
          border-radius: 10px;
          background: #f8fbff;
          border: 1px solid #e1e8f0;
        }}
        .selected-crop-candidate-thumb {{
          width: 34px;
          height: 34px;
          border-radius: 8px;
          overflow: hidden;
          background: #fff;
          border: 1px solid #d6dee8;
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        .selected-crop-candidate-thumb img {{
          max-width: 100%;
          max-height: 100%;
          display: block;
        }}
        .selected-crop-debug {{
          margin-top: 8px;
          padding-top: 8px;
          border-top: 1px dashed #d6dee8;
          color: #627283;
          font-size: 11px;
          line-height: 1.35;
          max-height: 180px;
          overflow: auto;
        }}
        .selected-crop-picker {{
          margin-top: 0;
        }}
        .selected-crop-picker h3 {{
          margin: 0 0 8px;
          font-size: 16px;
        }}
        .selected-crop-picker .picker-layout {{
          display: block;
        }}
        .selected-crop-picker .picker-canvas-wrap {{
          display: none;
        }}
        .selected-crop-panel-title {{
          margin: 0 0 6px;
          font-size: 15px;
        }}
        @media (max-width: 820px) {{
          .selected-crop-layout {{
            grid-template-columns: 1fr;
          }}
          .selected-crop-preview {{
            width: 100%;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="shell">
        <section class="hero">
          <h1>Instruction Buildability Debug</h1>
          <p><strong>Set:</strong> {escape(str(parts_payload.get("set_num") or set_num))}</p>
          <p><strong>Bag:</strong> {bag_number}</p>
          <p><strong>Real callout crops:</strong> {len(crops)}</p>
          <p><strong>Total set parts:</strong> {len(parts)}</p>
          <p><strong>Loaded LEGO colours:</strong> <span id="loaded-lego-colours-count">{len(lego_colors)}</span></p>
          <p><strong>AI rank:</strong> {"ON" if ai_rank_enabled else "OFF"} | calls: {int(ai_rank_stats.get("calls", 0) or 0)} | images: {int(ai_rank_stats.get("images_sent", 0) or 0)} | input tokens: {int(ai_rank_stats.get("input_tokens", 0) or 0)} | output tokens: {int(ai_rank_stats.get("output_tokens", 0) or 0)} | failures: {int(ai_rank_stats.get("failures", 0) or 0)}</p>
          <p><strong>Save path:</strong> {escape(str(labels_path))}</p>
          {f'<div class="warning-note">{escape(lego_colors_warning)}</div>' if lego_colors_warning else ''}
          <div id="selected-crop-text" class="status-line">Selected crop: none | Metallic mode: OFF</div>
          <div id="save-status" class="save-note"></div>
        </section>

        <section class="panel">
          <h2>Manual Crop Mode</h2>
          <p class="save-note">Drag a rectangle on any page, enter the step number, then save the manual crop as training ground truth.</p>
          <div class="manual-pages-grid">
            {manual_pages_html}
          </div>
        </section>

        <section class="panel">
          <div class="crop-panel-head">
            <h2>Detected Callout Crops</h2>
            <div class="crop-toolbar">
              <a
                class="remove-btn toolbar-link"
                href="/debug/export-training-data?set_num={escape(str(set_num))}&bag={int(bag_number)}"
                target="_blank"
                rel="noopener noreferrer"
              >Export Training Data</a>
              <button type="button" class="remove-btn" onclick="goToNextCrop()">Next crop</button>
              <label class="hidden-toggle" for="show-hidden-crops">
                <input id="show-hidden-crops" type="checkbox" onchange="setShowHiddenCrops(this.checked)" />
                Show hidden crops
              </label>
            </div>
          </div>
          <div class="crop-grid">
            {crop_cards_html}
          </div>
        </section>

        <section class="panel">
          <h2>Set Part Library</h2>
          <div id="selected-crop-workspace" class="selected-crop-workspace empty">
            Select a crop to start matching parts from the Set Part Library.
          </div>
          <div id="suggested-parts-panel" class="suggested-panel" tabindex="-1">
            <h3>Suggested parts</h3>
            <div id="suggested-parts-grid" class="suggested-grid">
              <div class="suggested-empty">Select a crop to see first-pass suggestions.</div>
            </div>
          </div>
          <div class="parts-toolbar">
            <div class="parts-filter-toggle">
              <button type="button" id="parts-filter-filtered" class="parts-filter-btn active" onclick="setPartFilterMode(false)">Filtered</button>
              <button type="button" id="parts-filter-show-all" class="parts-filter-btn" onclick="setPartFilterMode(true)">Show all</button>
            </div>
            <label class="hidden-toggle" for="allow-extra-part">
              <input id="allow-extra-part" type="checkbox" onchange="updateAddAvailability()" />
              Allow extra part
            </label>
            <label class="hidden-toggle" for="allow-over-assign">
              <input id="allow-over-assign" type="checkbox" onchange="updateAddAvailability()" />
              Allow over-assign
            </label>
            <div id="parts-filter-status" class="parts-filter-status">Filtered mode is ready. Select a crop to match LEGO colours.</div>
            <div id="slot-status-note" class="slot-status-note"></div>
            <div id="bag-assignment-summary" class="bag-summary">Bag assigned parts: 0</div>
            <div id="remaining-candidates-summary" class="bag-summary">Remaining candidate parts: 0</div>
          </div>
          <div class="parts-grid">
            {parts_tiles_html}
          </div>
        </section>
      </div>

      <div id="crop-zoom-modal" class="zoom-modal" onclick="closeCropZoom()">
        <div class="zoom-modal-panel" onclick="event.stopPropagation()">
          <button type="button" class="zoom-modal-close" onclick="closeCropZoom()" aria-label="Close crop zoom">&times;</button>
          <img id="crop-zoom-image" class="zoom-modal-image" alt="Zoomed crop preview" />
          <div id="crop-zoom-caption" class="zoom-modal-caption"></div>
        </div>
      </div>

      <div id="where-used-modal" class="zoom-modal" onclick="closeWhereUsed()">
        <div class="zoom-modal-panel" onclick="event.stopPropagation()">
          <button type="button" class="zoom-modal-close" onclick="closeWhereUsed()" aria-label="Close where used">&times;</button>
          <div class="where-used-toolbar">
            <div class="where-used-heading">
              <div id="where-used-title" class="where-used-title"></div>
              <div id="where-used-summary" class="where-used-summary"></div>
            </div>
            <div class="where-used-tools">
              <input id="where-used-filter" class="where-used-filter" type="search" placeholder="Filter by page or step" oninput="handleWhereUsedFilterInput(this.value)" />
              <button id="where-used-remove-all-btn" type="button" class="remove-btn" onclick="removeAllWhereUsed()">Remove all from this part</button>
            </div>
          </div>
          <div id="where-used-grid" class="usage-modal-grid">
            <div class="suggested-empty">No usages yet.</div>
          </div>
        </div>
      </div>

      <script>
        const cropRecords = {crops_json};
        const partRecords = {parts_json};
        const cropMap = new Map(cropRecords.map(item => [item.crop_id, item]));
        const partMap = new Map(
          partRecords.map(item => [partKey(item.part_num, item.color_id), item])
        );
        const manualSelections = new Map();
        const partTiles = Array.from(document.querySelectorAll(".part-tile"));
        window.legoColors = {lego_colors_json};
        window.trainingExamples = {training_examples_json};
        const colors = window.legoColors || [];
        const colorNameById = new Map();
        let activeCropId = null;
        window.selectedCropId = null;
        const aiEnabled = {str(int(ai or 0) == 1).lower()};
        const aiRankEnabled = {str(ai_rank_enabled).lower()};
        let showAllParts = false;
        let showHiddenCrops = false;
        let colourPickerImage = null;
        let showAllManualColours = false;

        function partKey(partNum, colorId) {{
          return String(partNum || "") + "::" + Number(colorId || 0);
        }}

        const trainingExamplesByKey = new Map();
        (window.trainingExamples || []).forEach((example) => {{
          const key = partKey(example && example.part_num, example && example.color_id);
          if (!trainingExamplesByKey.has(key)) {{
            trainingExamplesByKey.set(key, []);
          }}
          trainingExamplesByKey.get(key).push(example);
        }});

        function parseRgbHex(hex) {{
          if (!hex) {{
            return null;
          }}
          const normalized = String(hex || "")
            .trim()
            .replace(/^#/, "")
            .replace(/^0x/i, "");
          if (!/^[0-9A-Fa-f]{{6}}$/.test(normalized)) {{
            return null;
          }}
          return {{
            r: parseInt(normalized.slice(0, 2), 16),
            g: parseInt(normalized.slice(2, 4), 16),
            b: parseInt(normalized.slice(4, 6), 16)
          }};
        }}

        function colorDistance(left, right) {{
          const leftR = Number(left && left.r);
          const leftG = Number(left && left.g);
          const leftB = Number(left && left.b);
          const rightR = Number(right && right.r);
          const rightG = Number(right && right.g);
          const rightB = Number(right && right.b);
          if (
            !Number.isFinite(leftR) || !Number.isFinite(leftG) || !Number.isFinite(leftB) ||
            !Number.isFinite(rightR) || !Number.isFinite(rightG) || !Number.isFinite(rightB)
          ) {{
            return Number.POSITIVE_INFINITY;
          }}
          const dr = leftR - rightR;
          const dg = leftG - rightG;
          const db = leftB - rightB;
          return Math.sqrt((dr * dr) + (dg * dg) + (db * db));
        }}

        function candidateSuggestionDistance(crop, colorId) {{
          if (!crop || !crop.picked_rgb) {{
            return Number.POSITIVE_INFINITY;
          }}
          const candidate = normalizedLegoColors.find((item) => Number(item && item.color_id) === Number(colorId || 0));
          if (!candidate) {{
            return Number.POSITIVE_INFINITY;
          }}
          return colorDistance(crop.picked_rgb, candidate);
        }}

        function trainingBoostMultiplier(crop, partNum, colorId, distanceScore) {{
          const key = partKey(partNum, colorId);
          const examples = trainingExamplesByKey.get(key) || [];
          if (!examples.length) {{
            return 1;
          }}
          if (!Number.isFinite(distanceScore) || distanceScore > 45) {{
            return 1;
          }}
          const cropMetallicMode = metallicModeEnabled(crop);
          const hasMatchingExample = examples.some((example) => {{
            if (example && typeof example.metallic_mode === "boolean") {{
              return example.metallic_mode === cropMetallicMode;
            }}
            return true;
          }});
          if (!hasMatchingExample) {{
            return 1;
          }}
          console.log("training boost applied", {{
            part_num: partNum,
            color_id: colorId,
            distance: distanceScore
          }});
          return 0.6;
        }}

        function closestLegoColorId(rgb) {{
          const candidates = window.legoColors || [];
          let best = null;
          for (const candidate of candidates) {{
            const parsedRgb = parseRgbHex(candidate && candidate.rgb);
            if (!parsedRgb) {{
              continue;
            }}
            const distance = colorDistance(rgb, parsedRgb);
            if (!best || distance < best.distance) {{
              best = {{
                color_id: candidate.color_id,
                color_name: candidate.color_name,
                distance
              }};
            }}
          }}
          return best;
        }}

        function escapeHtml(value) {{
          return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
        }}

        function selectedQtyLabel(qtyValue, qtyTextValue) {{
          if (Array.isArray(qtyTextValue) && qtyTextValue.length) {{
            return qtyTextValue.join(", ");
          }}
          if (typeof qtyTextValue === "string" && qtyTextValue.trim()) {{
            return qtyTextValue;
          }}
          if (Array.isArray(qtyValue) && qtyValue.length) {{
            return qtyValue.join(", ");
          }}
          if (qtyValue !== null && qtyValue !== undefined && qtyValue !== "") {{
            return String(qtyValue);
          }}
          return "none";
        }}

        function currentCropTargetQty(crop) {{
          const slotState = computeCropSlotState(crop);
          return Number.isFinite(Number(slotState.nextQtyValue)) ? Number(slotState.nextQtyValue) : 1;
        }}

        function isTypingIntoField(target) {{
          if (!target) {{
            return false;
          }}
          const tagName = String(target.tagName || "").toUpperCase();
          return tagName === "INPUT" || tagName === "TEXTAREA" || Boolean(target.isContentEditable);
        }}

        function colorSwatchStyle(rgb) {{
          if (!rgb) {{
            return "transparent";
          }}
          return "rgb(" + rgb.r + ", " + rgb.g + ", " + rgb.b + ")";
        }}

        function formatRgb(rgb) {{
          if (!rgb) {{
            return "n/a";
          }}
          return Number(rgb.r) + ", " + Number(rgb.g) + ", " + Number(rgb.b);
        }}

        function availableSetLegoColors(includeAllManual = false) {{
          return colors
            .filter((candidate) => includeAllManual || Number(candidate.color_id) !== 9999)
            .slice();
        }}

        function availableNormalizedLegoColors(includeAllManual = false) {{
          return normalizedLegoColors
            .filter((candidate) => includeAllManual || Number(candidate.color_id) !== 9999)
            .slice();
        }}

        function commonManualColorNames() {{
          return [
            "black",
            "dark bluish gray",
            "light bluish gray",
            "reddish brown",
            "tan",
            "dark brown",
            "red",
            "white",
            "pearl gold",
            "trans-clear",
          ];
        }}

        function manualColorFallbackMatches(includeAllManual = false) {{
          const preferredOrder = new Map(commonManualColorNames().map((name, index) => [name, index]));
          const allCandidates = availableSetLegoColors(includeAllManual);
          const candidates = includeAllManual
            ? allCandidates
            : allCandidates.filter((candidate) => preferredOrder.has(String(candidate.color_name || "").trim().toLowerCase()));
          const resolvedCandidates = candidates.length ? candidates : allCandidates;
          return resolvedCandidates
            .map((candidate) => {{
              const parsedRgb = parseRgbHex(candidate && candidate.rgb);
              if (!parsedRgb) {{
                return null;
              }}
              return {{
                color_id: candidate.color_id,
                color_name: candidate.color_name,
                rgb: parsedRgb,
                distance_score: null,
              }};
            }})
            .filter(Boolean)
            .sort((left, right) => {{
              const leftName = String(left.color_name || "").trim().toLowerCase();
              const rightName = String(right.color_name || "").trim().toLowerCase();
              const leftRank = preferredOrder.has(leftName) ? Number(preferredOrder.get(leftName)) : 999;
              const rightRank = preferredOrder.has(rightName) ? Number(preferredOrder.get(rightName)) : 999;
              return leftRank - rightRank || leftName.localeCompare(rightName) || Number(left.color_id) - Number(right.color_id);
            }});
        }}

        function ensurePickerDiagnostics(crop) {{
          if (!crop) {{
            return {{
              selectedCropId: "none",
              canvasImageLoaded: "no",
              canvasSize: "0 x 0",
              lastClick: "n/a",
              sampledRgb: "n/a",
              closestCount: 0,
              metallicMode: "OFF",
              errorMessage: "",
            }};
          }}
          if (!crop.pickerDiagnostics) {{
            crop.pickerDiagnostics = {{
              selectedCropId: String(crop.crop_id || "none"),
              canvasImageLoaded: "no",
              canvasSize: "0 x 0",
              lastClick: "n/a",
              sampledRgb: "n/a",
              closestCount: 0,
              metallicMode: "OFF",
              errorMessage: "",
            }};
          }}
          crop.pickerDiagnostics.selectedCropId = String(crop.crop_id || "none");
          return crop.pickerDiagnostics;
        }}

        function updatePickerDiagnostics(crop, patch = {{}}) {{
          const diagnostics = ensurePickerDiagnostics(crop);
          Object.assign(diagnostics, patch || {{}});
          return diagnostics;
        }}

        function renderPickerDiagnostics(crop) {{
          const diagnosticsEl = document.getElementById("picker-diagnostics");
          if (!diagnosticsEl) {{
            return;
          }}
          const diagnostics = ensurePickerDiagnostics(crop);
          diagnosticsEl.innerHTML = `
            <div><strong>selected crop_id:</strong> ${{escapeHtml(diagnostics.selectedCropId)}}</div>
            <div><strong>canvas image loaded:</strong> ${{escapeHtml(String(diagnostics.canvasImageLoaded))}}</div>
            <div><strong>canvas size:</strong> ${{escapeHtml(String(diagnostics.canvasSize))}}</div>
            <div><strong>last click x/y:</strong> ${{escapeHtml(String(diagnostics.lastClick))}}</div>
            <div><strong>sampled RGB:</strong> ${{escapeHtml(String(diagnostics.sampledRgb))}}</div>
            <div><strong>closest color result count:</strong> ${{escapeHtml(String(diagnostics.closestCount))}}</div>
            <div><strong>Metallic mode:</strong> ${{escapeHtml(String(diagnostics.metallicMode || "OFF"))}}</div>
            <div><strong>JS error message:</strong> ${{escapeHtml(String(diagnostics.errorMessage || ""))}}</div>
          `;
        }}

        function metallicModeEnabled(crop) {{
          return !!(crop && crop.manual_metallic_mode);
        }}

        function metallicModeText(crop) {{
          return metallicModeEnabled(crop) ? "ON" : "OFF";
        }}

        function updateSelectedCropStatus(crop) {{
          const status = document.getElementById("selected-crop-text");
          if (!status) {{
            return;
          }}
          if (!crop) {{
            status.textContent = "Selected crop: none | Metallic mode: OFF";
            return;
          }}
          status.textContent = "Selected crop: " + crop.crop_id + " | Metallic mode: " + metallicModeText(crop);
        }}

        function setSelectedCropInlineStatus(message, isError = false) {{
          const statusEl = document.getElementById("selected-crop-inline-status");
          if (!statusEl) {{
            return;
          }}
          statusEl.textContent = String(message || "");
          statusEl.style.color = isError ? "#b91c1c" : "#627283";
        }}

        function renderSelectedCropWorkspace(cropId) {{
          const workspace = document.getElementById("selected-crop-workspace");
          const crop = cropId ? cropMap.get(cropId) : null;
          if (!workspace) {{
            return;
          }}
          if (!crop) {{
            window.selectedCropId = null;
            workspace.classList.add("empty");
            workspace.innerHTML = "Select a crop to start matching parts from the Set Part Library.";
            return;
          }}
          window.selectedCropId = String(crop.crop_id || "");
          workspace.classList.remove("empty");
          const slotState = computeCropSlotState(crop);
          const aiDebug = crop.ai_rank_debug || {{}};
          const aiResultPayload = aiDebug.returned_ai_result || null;
          const aiFailureReason = String(aiDebug.failure_reason || "");
          const localCandidates = Array.isArray(crop.top_part_candidates) ? crop.top_part_candidates.slice(0, 5) : [];
          const aiResultHtml = crop.ai_rank_candidate
            ? (
              '<div class="selected-crop-candidate-row">'
              + '<div class="selected-crop-candidate-thumb">'
              + (crop.ai_rank_candidate.img_url ? ('<img src="' + escapeHtml(String(crop.ai_rank_candidate.img_url || "")) + '" alt="' + escapeHtml(String(crop.ai_rank_candidate.part_num || "")) + '" loading="lazy" />') : '<div class="crop-missing">?</div>')
              + '</div>'
              + '<div><strong>' + escapeHtml(String(crop.ai_rank_candidate.part_num || "—")) + '</strong>'
              + '<br/>color: ' + escapeHtml(String(crop.ai_rank_candidate.color_name || crop.ai_rank_candidate.color_id || "—"))
              + '<br/>confidence: ' + escapeHtml(crop.ai_rank_confidence === null || crop.ai_rank_confidence === undefined ? "—" : Number(crop.ai_rank_confidence).toFixed(2))
              + '<br/>reason: ' + escapeHtml(String(crop.ai_rank_reason || "—")) + '</div>'
              + '</div>'
            )
            : ('<div>No AI pick available for this crop.</div>' + (aiFailureReason ? ('<div>Why: ' + escapeHtml(aiFailureReason) + '</div>') : ''));
          const localCandidatesHtml = localCandidates.length
            ? localCandidates.map((candidate) => `
              <div class="selected-crop-candidate-row">
                <div class="selected-crop-candidate-thumb">
                  ${{candidate.img_url ? `<img src="${{escapeHtml(String(candidate.img_url || ""))}}" alt="${{escapeHtml(String(candidate.part_num || ""))}}" loading="lazy" />` : '<div class="crop-missing">?</div>'}}
                </div>
                <div>
                  <strong>${{escapeHtml(String(candidate.part_num || "—"))}}</strong><br/>
                  color: ${{escapeHtml(String(candidate.color_name || candidate.color_id || "—"))}}<br/>
                  score: ${{escapeHtml(Number(candidate.score || 0).toFixed(2))}}
                </div>
              </div>
            `).join("")
            : '<div class="assigned-empty">No local candidates available.</div>';
          const aiDebugHtml = `
            <div class="selected-crop-debug">
              <div><strong>selected crop id:</strong> ${{escapeHtml(String(window.selectedCropId || crop.crop_id || "none"))}}</div>
              <div><strong>ai enabled:</strong> ${{escapeHtml(String(aiEnabled))}}</div>
              <div><strong>ai_rank enabled:</strong> ${{escapeHtml(String(aiRankEnabled))}}</div>
              <div><strong>candidate count sent to AI:</strong> ${{escapeHtml(String(Number(aiDebug.sent_candidate_count || localCandidates.length || 0)))}}</div>
              <div><strong>returned ai result:</strong> ${{escapeHtml(aiResultPayload ? JSON.stringify(aiResultPayload) : "none")}}</div>
              <div><strong>failure reason:</strong> ${{escapeHtml(aiFailureReason || "—")}}</div>
            </div>
          `;
          const aiPanel = aiRankEnabled ? `
            <div class="selected-crop-ai">
              <div><strong>AI match</strong></div>
              <div id="selected-crop-ai-result">${{aiResultHtml}}</div>
              <div class="selected-crop-actions">
                <button type="button" class="remove-btn" onclick="findAiMatchForSelectedCrop()">Find AI Match</button>
                ${{crop.ai_rank_candidate ? '<button type="button" class="remove-btn" onclick="useAiPickFromLibrary(event)">Use AI Pick</button>' : ''}}
              </div>
              <div class="selected-crop-candidate-list">${{localCandidatesHtml}}</div>
              ${{aiDebugHtml}}
            </div>
          ` : "";
          const previewHtml = crop.data_uri
            ? '<img src="' + escapeHtml(crop.data_uri) + '" data-src="' + escapeHtml(crop.data_uri) + '" data-crop-id="' + escapeHtml(crop.crop_id) + '" alt="' + escapeHtml(crop.crop_id) + '" onclick="openCropZoomFromEl(event, this)" />'
            : '<div class="crop-missing">No image</div>';
          const allSlotsHtml = slotState.slotsFull && !slotState.noQtyDetected ? '<strong>All slots filled</strong><br/>' : '';
          const aiResponseSummary = crop.ai_rank_candidate
            ? ("<div><strong>Best match</strong></div>"
              + '<div class="selected-crop-candidate-row">'
              + '<div class="selected-crop-candidate-thumb">'
              + (crop.ai_rank_candidate.img_url ? ('<img src="' + escapeHtml(String(crop.ai_rank_candidate.img_url || "")) + '" alt="' + escapeHtml(String(crop.ai_rank_candidate.part_num || "")) + '" loading="lazy" />') : '<div class="crop-missing">?</div>')
              + '</div>'
              + '<div><strong>' + escapeHtml(String(crop.ai_rank_candidate.part_num || "—")) + '</strong>'
              + '<br/>color: ' + escapeHtml(String(crop.ai_rank_candidate.color_name || crop.ai_rank_candidate.color_id || "—"))
              + '<br/>confidence: ' + escapeHtml(crop.ai_rank_confidence === null || crop.ai_rank_confidence === undefined ? "—" : Number(crop.ai_rank_confidence).toFixed(2))
              + '<br/>reason: ' + escapeHtml(String(crop.ai_rank_reason || "—")) + '</div>'
              + '</div>')
            : ('<div>No AI pick selected.</div>' + (aiFailureReason ? ('<div>Why: ' + escapeHtml(aiFailureReason) + '</div>') : ''));
          const aiPayloadPreview = Array.isArray(aiDebug.candidate_rows_preview) && aiDebug.candidate_rows_preview.length
            ? aiDebug.candidate_rows_preview.map((row) => `
              <div>
                ${{escapeHtml(String(row.part_num || "—"))}}
                / c${{escapeHtml(String(row.color_id || 0))}}
                / e${{escapeHtml(String(row.element_id || "—"))}}
                / img: ${{escapeHtml(String(row.img_url || ""))}}
              </div>
            `).join("")
            : "<div>none</div>";
          workspace.innerHTML = `
            <div class="selected-crop-layout">
              <div class="selected-crop-column">
                <div class="selected-crop-panel-title"><strong>Selected crop</strong></div>
                <div class="selected-crop-preview">${{previewHtml}}</div>
                <div class="selected-crop-meta-card selected-crop-meta">
                  <strong>${{escapeHtml(crop.crop_id)}}</strong><br/>
                  qty_text: ${{escapeHtml(Array.isArray(crop.qty_text) ? crop.qty_text.join(", ") : "none")}}<br/>
                  slots filled / total: ${{slotState.filledSlots}} / ${{slotState.totalSlots}}<br/>
                  next qty: ${{escapeHtml(String(slotState.nextQtyLabel || "1x"))}}<br/>
                  ${{allSlotsHtml}}
                  <div class="selected-crop-actions">
                    <button type="button" class="remove-btn" onclick="findAiMatchForSelectedCrop()">Find AI Match</button>
                    ${{crop.ai_rank_candidate ? '<button type="button" class="remove-btn" onclick="useAiPickFromLibrary(event)">Use AI Pick</button>' : ''}}
                  </div>
                  <div id="selected-crop-ai-result" class="selected-crop-ai">${{aiResponseSummary}}</div>
                  <div class="selected-crop-debug">
                    <div><strong>selected crop id:</strong> ${{escapeHtml(String(window.selectedCropId || crop.crop_id || "none"))}}</div>
                    <div><strong>ai enabled:</strong> ${{escapeHtml(String(aiEnabled))}}</div>
                    <div><strong>ai_rank enabled:</strong> ${{escapeHtml(String(aiRankEnabled))}}</div>
                    <div><strong>provider selected:</strong> ${{escapeHtml(String(aiDebug.provider_selected || "none"))}}</div>
                    <div><strong>AZURE_OPENAI_ENDPOINT present:</strong> ${{escapeHtml(String(Boolean(aiDebug.azure_endpoint_present)))}}</div>
                    <div><strong>AZURE_OPENAI_API_KEY present:</strong> ${{escapeHtml(String(Boolean(aiDebug.azure_api_key_present)))}}</div>
                    <div><strong>AZURE_OPENAI_DEPLOYMENT present:</strong> ${{escapeHtml(String(Boolean(aiDebug.azure_deployment_present)))}}</div>
                    <div><strong>AZURE_OPENAI_API_VERSION:</strong> ${{escapeHtml(String(aiDebug.azure_api_version || "none"))}}</div>
                    <div><strong>client created:</strong> ${{escapeHtml(String(Boolean(aiDebug.client_created)))}}</div>
                    <div><strong>candidate count:</strong> ${{escapeHtml(String(Number(aiDebug.sent_candidate_count || localCandidates.length || 0)))}}</div>
                    <div><strong>ai_skipped_reason:</strong> ${{escapeHtml(String(aiDebug.ai_skipped_reason || "none"))}}</div>
                    <div><strong>crop_image_source:</strong> ${{escapeHtml(String(aiDebug.crop_image_source || "missing"))}}</div>
                    <div><strong>crop_image_exists:</strong> ${{escapeHtml(String(Boolean(aiDebug.crop_image_exists)))}}</div>
                    <div><strong>crop image included:</strong> ${{escapeHtml(String(Boolean(aiDebug.crop_image_included)))}}</div>
                    <div><strong>candidate_img_urls count:</strong> ${{escapeHtml(String(Number(aiDebug.candidate_img_urls_count || 0)))}}</div>
                    <div><strong>candidate image count:</strong> ${{escapeHtml(String(Number(aiDebug.candidate_images_included_count || 0)))}}</div>
                    <div><strong>first candidate img_url:</strong> ${{escapeHtml(String(aiDebug.first_candidate_img_url || "none"))}}</div>
                    <div><strong>exception type:</strong> ${{escapeHtml(String(aiDebug.exception_type || "none"))}}</div>
                    <div><strong>exception message:</strong> ${{escapeHtml(String(aiDebug.exception_message || "none"))}}</div>
                    <div><strong>raw response:</strong> ${{escapeHtml(String(aiDebug.raw_response_text || "none"))}}</div>
                    <div><strong>failure reason:</strong> ${{escapeHtml(aiFailureReason || "—")}}</div>
                  </div>
                  <div class="selected-crop-actions">
                    <button type="button" class="remove-btn" onclick="jumpToSelectedCropCard()">Back to crop</button>
                    <button type="button" class="remove-btn" onclick="goToNextCrop()">Next crop</button>
                  </div>
                  <div id="selected-crop-inline-status" class="selected-crop-status"></div>
                </div>
              </div>

              <div class="selected-crop-column">
                <div class="selected-crop-panel-title"><strong>Current assignments</strong></div>
                <div class="selected-crop-assignments library-assignments">${{assignedPartsMarkup(crop.crop_id)}}</div>
              </div>

              <div class="selected-crop-column">
                <div class="selected-crop-picker">
                  <div class="selected-crop-panel-title"><strong>Match tools</strong></div>
                  <h3>Colour Picker</h3>
                  <p class="colour-picker-help">Use manual colour filters and top candidates for the selected crop.</p>
                  <div class="picker-layout">
                    <div class="picker-canvas-wrap">
                      <canvas id="colour-picker-canvas" class="picker-canvas"></canvas>
                      <div id="colour-picker-empty" class="picker-empty">Select a crop with an image to start sampling colours.</div>
                    </div>
                    <div>
                      <div id="picked-rgb-row" class="picked-rgb-row">
                        <span class="colour-swatch" id="picked-rgb-swatch" style="background: transparent;"></span>
                        <span id="picked-rgb-text" class="save-note">No colour sampled yet.</span>
                        <button type="button" class="remove-btn" onclick="clearManualColorFilter()">Clear colour filter</button>
                      </div>
                      <div class="colour-picker-actions">
                        <strong>Manual colour filter</strong>
                        <button id="manual-colours-toggle" type="button" class="remove-btn" onclick="toggleShowAllManualColours()">Show all colours</button>
                      </div>
                      <div id="colour-match-list" class="colour-match-list">
                        <div class="suggested-empty">Pick a colour from the crop to see the closest LEGO colours.</div>
                      </div>
                      <div class="colour-picker-actions">
                        <strong>Top local candidates</strong>
                      </div>
                      <div class="selected-crop-candidate-list">${{localCandidatesHtml}}</div>
                      <div id="picker-diagnostics" class="picker-diagnostics"></div>
                    </div>
                  </div>
                </div>
                <div class="selected-crop-debug">
                  <div><strong>prompt text:</strong> ${{escapeHtml(String(aiDebug.prompt_text || "none"))}}</div>
                  <div><strong>candidate payload preview:</strong>${{aiPayloadPreview}}</div>
                  <div><strong>parsed response JSON:</strong> ${{escapeHtml(aiDebug.parsed_response_json ? JSON.stringify(aiDebug.parsed_response_json) : "none")}}</div>
                </div>
              </div>
            </div>
          `;
          bindColourPickerCanvas();
          renderColourPicker(crop.crop_id);
        }}

        function updateManualColoursToggleLabel() {{
          const toggle = document.getElementById("manual-colours-toggle");
          if (!toggle) {{
            return;
          }}
          toggle.textContent = showAllManualColours ? "Show common colours" : "Show all colours";
        }}

        function activeFilterColorIds(crop, fallbackColorIds) {{
          if (crop && Number.isFinite(Number(crop.manual_color_filter_id))) {{
            return [Number(crop.manual_color_filter_id)];
          }}
          const colorIds = Array.isArray(fallbackColorIds) ? fallbackColorIds.slice() : [];
          if (metallicModeEnabled(crop)) {{
            const seen = new Set(colorIds.map((value) => Number(value)));
            for (const candidate of window.legoColors || []) {{
              const colorId = Number(candidate && candidate.color_id);
              if (!Number.isFinite(colorId) || seen.has(colorId)) {{
                continue;
              }}
              if (metallicColourRank(candidate && candidate.color_name) === 0) {{
                colorIds.push(colorId);
                seen.add(colorId);
              }}
            }}
          }}
          return colorIds;
        }}

        function normalizeColorName(value) {{
          return String(value || "").trim().toLowerCase();
        }}

        function metallicColourRank(colorName) {{
          const normalized = normalizeColorName(colorName);
          if (
            normalized.includes("pearl gold")
            || normalized.includes("pearl dark gray")
            || normalized.includes("pearl light gray")
          ) {{
            return 0;
          }}
          if (
            normalized === "yellow"
            || normalized === "orange"
            || normalized === "tan"
          ) {{
            return 2;
          }}
          return 1;
        }}

        function qtySlotSignature(qtyValue, qtyTextValue) {{
          if (typeof qtyTextValue === "string" && qtyTextValue.trim()) {{
            return "text:" + qtyTextValue.trim().toLowerCase();
          }}
          const qtyNumber = Number(qtyValue);
          return Number.isFinite(qtyNumber) ? "qty:" + qtyNumber : "";
        }}

        function buildClientQtySequence(crop) {{
          const qtyText = Array.isArray(crop && crop.qty_text) ? crop.qty_text.filter(Boolean) : [];
          const qtyNumbers = Array.isArray(crop && crop.qty_numbers) ? crop.qty_numbers : [];
          if (qtyText.length) {{
            return qtyText.map((qtyTextValue, index) => {{
              const parsedQty = Number(String(qtyTextValue).replace(/[^0-9]/g, ""));
              const fallbackQty = Number(qtyNumbers[index]);
              return {{
                qty: Number.isFinite(parsedQty) && parsedQty > 0 ? parsedQty : (Number.isFinite(fallbackQty) ? fallbackQty : null),
                qty_text: String(qtyTextValue).trim()
              }};
            }});
          }}
          return qtyNumbers
            .map((qtyValue) => Number(qtyValue))
            .filter((qtyValue) => Number.isFinite(qtyValue))
            .map((qtyValue) => ({{
              qty: qtyValue,
              qty_text: qtyValue + "x"
            }}));
        }}

        function computeCropSlotState(crop) {{
          if (!crop) {{
            return {{
              totalSlots: 0,
              filledSlots: 0,
              slotsFull: false,
              noQtyDetected: true,
              nextQtyLabel: "1x",
              nextQtyValue: 1,
            }};
          }}

          const sequence = buildClientQtySequence(crop);
          const parts = Array.isArray(crop.parts) ? crop.parts : [];
          if (!sequence.length) {{
            return {{
              totalSlots: 0,
              filledSlots: parts.length,
              slotsFull: false,
              noQtyDetected: true,
              nextQtyLabel: "1x",
              nextQtyValue: 1,
            }};
          }}

          const assignedCounts = new Map();
          parts.forEach((part) => {{
            const signature = qtySlotSignature(part.qty, part.qty_text);
            if (!signature) {{
              return;
            }}
            assignedCounts.set(signature, Number(assignedCounts.get(signature) || 0) + 1);
          }});

          const consumedCounts = new Map();
          let filledSlots = 0;
          let nextSlot = null;
          for (const slot of sequence) {{
            const signature = qtySlotSignature(slot.qty, slot.qty_text);
            const assignedCount = Number(assignedCounts.get(signature) || 0);
            const consumedCount = Number(consumedCounts.get(signature) || 0);
            if (signature && assignedCount > consumedCount) {{
              consumedCounts.set(signature, consumedCount + 1);
              filledSlots += 1;
              continue;
            }}
            nextSlot = slot;
            break;
          }}

          return {{
            totalSlots: sequence.length,
            filledSlots,
            slotsFull: nextSlot === null,
            noQtyDetected: false,
            nextQtyLabel: nextSlot ? String(nextSlot.qty_text || nextSlot.qty || "none") : "filled",
            nextQtyValue: nextSlot && Number.isFinite(Number(nextSlot.qty)) ? Number(nextSlot.qty) : null,
          }};
        }}

        function statusClassName(status) {{
          return "crop-status-" + String(status || "needs_adjust").replaceAll(/[^a-z_]/g, "");
        }}

        function hydratePart(part) {{
          const meta = partMap.get(partKey(part.part_num, part.color_id)) || {{}};
          return {{
            part_num: part.part_num,
            color_id: Number(part.color_id || 0),
            element_id: part.element_id || null,
            color_name: part.color_name || meta.color_name || "n/a",
            img_url: part.img_url || meta.img_url || "",
            qty: part.qty ?? null,
            qty_text: part.qty_text ?? null,
            part_bbox: Array.isArray(part.part_bbox) ? part.part_bbox : null,
            confidence: part.confidence ?? null,
            selected_qty_label: part.selected_qty_label || selectedQtyLabel(part.qty, part.qty_text)
          }};
        }}

        function syncCropFromResponse(localCrop, responseCrop) {{
          if (!localCrop || !responseCrop) {{
            return;
          }}
          localCrop.status = responseCrop.status || localCrop.status || "needs_adjust";
          localCrop.is_hidden = localCrop.status === "hidden";
          localCrop.is_manual = String(localCrop.crop_id || "").startsWith("manual_");
          localCrop.next_qty_index = Number(responseCrop.next_qty_index || 0);
          localCrop.qty_numbers = Array.isArray(responseCrop.qty) ? responseCrop.qty : (localCrop.qty_numbers || []);
          localCrop.qty_text = Array.isArray(responseCrop.qty_text) ? responseCrop.qty_text : (localCrop.qty_text || []);
          localCrop.qty_label = localCrop.qty_text.length ? localCrop.qty_text.join(", ") : (localCrop.qty_numbers.length ? localCrop.qty_numbers.join(", ") : "none");
          localCrop.parts = Array.isArray(responseCrop.parts)
            ? responseCrop.parts.map((part) => hydratePart(part))
            : [];
        }}

        function updateCropCardVisuals(cropId) {{
          const crop = cropMap.get(cropId);
          const el = document.getElementById(cropId);
          if (!crop || !el) {{
            return;
          }}
          const slotState = computeCropSlotState(crop);
          el.classList.remove("crop-status-good", "crop-status-bad", "crop-status-needs_adjust", "crop-status-hidden");
          el.classList.add(statusClassName(crop.status));
          el.dataset.hidden = String(Boolean(crop.is_hidden));
          el.dataset.manual = String(Boolean(crop.is_manual));
          const label = document.getElementById("status-label-" + cropId);
          if (label) {{
            label.textContent = crop.status || "needs_adjust";
          }}
          const qtyLabel = document.getElementById("qty-label-" + cropId);
          if (qtyLabel) {{
            qtyLabel.textContent = crop.qty_label || "none";
          }}
          const slotsLabel = document.getElementById("slots-label-" + cropId);
          if (slotsLabel) {{
            slotsLabel.textContent = slotState.totalSlots > 0
              ? (slotState.filledSlots + " / " + slotState.totalSlots + " filled")
              : (slotState.filledSlots + " / 0 filled");
          }}
          const nextQtyLabel = document.getElementById("next-qty-label-" + cropId);
          if (nextQtyLabel) {{
            nextQtyLabel.textContent = slotState.nextQtyLabel;
          }}
          const qtyWarning = document.getElementById("qty-warning-" + cropId);
          if (qtyWarning) {{
            qtyWarning.textContent = slotState.noQtyDetected
              ? "No qty detected"
              : (slotState.slotsFull ? "All qty slots filled" : "");
          }}
          const metallicLabel = document.getElementById("metallic-label-" + cropId);
          if (metallicLabel) {{
            metallicLabel.textContent = metallicModeText(crop);
          }}
          const metallicToggle = document.getElementById("metallic-toggle-" + cropId);
          if (metallicToggle) {{
            metallicToggle.checked = metallicModeEnabled(crop);
          }}
          const qtyInput = document.getElementById("qty-input-" + cropId);
          if (qtyInput && document.activeElement !== qtyInput) {{
            qtyInput.value = Array.isArray(crop.qty_text) ? crop.qty_text.join(",") : "";
          }}
          el.querySelectorAll(".status-btn").forEach((button) => {{
            button.classList.toggle("active", button.dataset.status === crop.status);
          }});
          const deleteButton = el.querySelector(".delete-crop-btn");
          if (deleteButton) {{
            deleteButton.textContent = crop.is_manual ? "Delete Crop" : "Hide Crop";
          }}
          if (activeCropId === cropId) {{
            updateSelectedCropStatus(crop);
            renderSelectedCropWorkspace(cropId);
            updateAddAvailability();
          }}
          applyHiddenCropVisibility();
        }}

        function applyHiddenCropVisibility() {{
          cropRecords.forEach((crop) => {{
            const el = document.getElementById(crop.crop_id);
            if (!el) {{
              return;
            }}
            const hideCard = Boolean(crop.is_hidden) && !showHiddenCrops;
            el.classList.toggle("is-hidden-crop", hideCard);
            if (hideCard && activeCropId === crop.crop_id) {{
              el.classList.remove("selected");
            }}
          }});
          if (activeCropId) {{
            const activeCrop = cropMap.get(activeCropId);
            if (activeCrop && activeCrop.is_hidden && !showHiddenCrops) {{
              activeCropId = null;
              updateSelectedCropStatus(null);
              renderSelectedCropWorkspace(null);
              document.getElementById("save-status").textContent = "";
              updateAddAvailability();
              renderSuggestedParts(null);
              renderColourPicker(null);
            }}
          }}
        }}

        function isCropVisibleInCurrentView(crop) {{
          if (!crop) {{
            return false;
          }}
          return !crop.is_hidden || showHiddenCrops;
        }}

        function nextVisibleCropId(fromCropId) {{
          const visibleCropIds = cropRecords
            .filter((crop) => cropMap.has(crop.crop_id) && isCropVisibleInCurrentView(crop))
            .map((crop) => crop.crop_id);
          if (!visibleCropIds.length) {{
            return null;
          }}
          if (!fromCropId) {{
            return visibleCropIds[0];
          }}
          const startIndex = visibleCropIds.indexOf(fromCropId);
          if (startIndex === -1) {{
            return visibleCropIds[0];
          }}
          return visibleCropIds[(startIndex + 1) % visibleCropIds.length];
        }}

        function goToNextCrop() {{
          const nextCropId = nextVisibleCropId(activeCropId);
          if (nextCropId) {{
            selectCrop(nextCropId);
          }}
        }}

        function setShowHiddenCrops(showHidden) {{
          showHiddenCrops = Boolean(showHidden);
          applyHiddenCropVisibility();
        }}

        function updatePartFilterButtons() {{
          const filteredBtn = document.getElementById("parts-filter-filtered");
          const showAllBtn = document.getElementById("parts-filter-show-all");
          if (filteredBtn) {{
            filteredBtn.classList.toggle("active", !showAllParts);
          }}
          if (showAllBtn) {{
            showAllBtn.classList.toggle("active", showAllParts);
          }}
        }}

        function allowExtraPartEnabled() {{
          const checkbox = document.getElementById("allow-extra-part");
          return Boolean(checkbox && checkbox.checked);
        }}

        function allowOverAssignEnabled() {{
          const checkbox = document.getElementById("allow-over-assign");
          return Boolean(checkbox && checkbox.checked);
        }}

        function assignedQtyValue(part) {{
          const qty = Number(part && part.qty);
          return Number.isFinite(qty) && qty > 0 ? qty : 1;
        }}

        function computeAssignedPartTotals() {{
          const totals = new Map();
          cropRecords.forEach((crop) => {{
            const parts = Array.isArray(crop && crop.parts) ? crop.parts : [];
            parts.forEach((part) => {{
              const key = partKey(part && part.part_num, part && part.color_id);
              totals.set(key, Number(totals.get(key) || 0) + assignedQtyValue(part));
            }});
          }});
          return totals;
        }}

        function partInventoryState(partNum, colorId) {{
          const key = partKey(partNum, colorId);
          const meta = partMap.get(key) || {{}};
          const requiredQty = Number(meta.qty || 0);
          const assignedQty = Number(computeAssignedPartTotals().get(key) || 0);
          const remainingQty = requiredQty - assignedQty;
          return {{
            key,
            requiredQty,
            assignedQty,
            remainingQty,
            fullyAssigned: remainingQty <= 0,
          }};
        }}

        let whereUsedState = null;

        function computeWhereUsedRows(partNum, colorId) {{
          const rows = [];
          cropRecords.forEach((crop) => {{
            const cropParts = Array.isArray(crop && crop.parts) ? crop.parts : [];
            cropParts.forEach((part) => {{
              if (String(part && part.part_num || "") !== String(partNum || "")) {{
                return;
              }}
              if (Number(part && part.color_id || 0) !== Number(colorId || 0)) {{
                return;
              }}
              rows.push({{
                crop_id: String(crop.crop_id || ""),
                page: Number(crop.page || 0),
                step: Number(crop.step || 0),
                selected_qty: selectedQtyLabel(part.qty, part.qty_text),
                qty_value: assignedQtyValue(part),
                part_num: String(part.part_num || ""),
                color_id: Number(part.color_id || 0),
                element_id: part.element_id || null,
                thumb: String(crop.data_uri || ""),
              }});
            }});
          }});
          return rows.sort((left, right) => (
            left.page - right.page
            || left.step - right.step
            || left.crop_id.localeCompare(right.crop_id)
          ));
        }}

        function closeWhereUsed() {{
          const modal = document.getElementById("where-used-modal");
          const grid = document.getElementById("where-used-grid");
          const title = document.getElementById("where-used-title");
          const summary = document.getElementById("where-used-summary");
          const filterInput = document.getElementById("where-used-filter");
          const removeAllButton = document.getElementById("where-used-remove-all-btn");
          if (modal) {{
            modal.classList.remove("open");
          }}
          if (grid) {{
            grid.innerHTML = '<div class="suggested-empty">No usages yet.</div>';
          }}
          if (title) {{
            title.textContent = "";
          }}
          if (summary) {{
            summary.textContent = "";
          }}
          if (filterInput) {{
            filterInput.value = "";
          }}
          if (removeAllButton) {{
            removeAllButton.disabled = true;
          }}
          whereUsedState = null;
        }}

        function handleWhereUsedFilterInput(value) {{
          if (!whereUsedState) {{
            return;
          }}
          whereUsedState.filterText = String(value || "");
          renderWhereUsed(whereUsedState.partNum, whereUsedState.colorId);
        }}

        function filterWhereUsedRows(rows, filterText) {{
          const query = String(filterText || "").trim().toLowerCase();
          if (!query) {{
            return rows;
          }}
          return rows.filter((row) => {{
            const haystack = [
              row.crop_id,
              "page " + String(row.page || ""),
              "step " + String(row.step || ""),
              String(row.page || ""),
              String(row.step || ""),
              String(row.page || "") + "/" + String(row.step || ""),
            ].join(" ").toLowerCase();
            return haystack.includes(query);
          }});
        }}

        function renderWhereUsed(partNum, colorId) {{
          const grid = document.getElementById("where-used-grid");
          const title = document.getElementById("where-used-title");
          const summary = document.getElementById("where-used-summary");
          const filterInput = document.getElementById("where-used-filter");
          const removeAllButton = document.getElementById("where-used-remove-all-btn");
          if (!grid || !title || !summary || !filterInput || !removeAllButton) {{
            return;
          }}
          const rows = computeWhereUsedRows(partNum, colorId);
          const filterText = whereUsedState ? String(whereUsedState.filterText || "") : "";
          const filteredRows = filterWhereUsedRows(rows, filterText);
          const inventory = partInventoryState(partNum, colorId);
          title.textContent = String(partNum || "") + " / color " + String(colorId || 0) + " | Required " + inventory.requiredQty + " | Assigned " + inventory.assignedQty + " | Remaining " + inventory.remainingQty;
          const totalQty = rows.reduce((sum, row) => sum + Number(row.qty_value || 0), 0);
          summary.textContent = "Used in " + rows.length + " crops (" + totalQty + " qty)" + (filterText ? " | Showing " + filteredRows.length + " match" + (filteredRows.length === 1 ? "" : "es") : "");
          if (document.activeElement !== filterInput) {{
            filterInput.value = filterText;
          }}
          removeAllButton.disabled = !rows.length;
          if (!rows.length) {{
            grid.innerHTML = '<div class="suggested-empty">No usages yet.</div>';
            return;
          }}
          if (!filteredRows.length) {{
            grid.innerHTML = '<div class="where-used-empty">No matching usages for this filter.</div>';
            return;
          }}
          grid.innerHTML = filteredRows.map((row) => {{
            const thumb = row.thumb
              ? '<img src="' + escapeHtml(row.thumb) + '" alt="' + escapeHtml(row.crop_id) + '" loading="lazy" />'
              : '<div class="crop-missing">No image</div>';
            const isCurrentCrop = row.crop_id === activeCropId;
            const currentBadge = isCurrentCrop ? '<span class="usage-current-badge">Current crop</span>' : '';
            return `
              <div class="usage-row${{isCurrentCrop ? " current-crop" : ""}}">
                <div class="usage-thumb">${{thumb}}</div>
                <div class="usage-meta">
                  <strong>${{escapeHtml(row.crop_id)}}</strong>${{currentBadge}}<br/>
                  page ${{escapeHtml(String(row.page))}} | step ${{escapeHtml(String(row.step || "?"))}}<br/>
                  selected qty: ${{escapeHtml(String(row.selected_qty))}}
                </div>
                <div class="usage-actions">
                  <button type="button" class="remove-btn" onclick='jumpToCropFromUsage(${{JSON.stringify(row.crop_id)}})'>Jump to crop</button>
                  <button type="button" class="remove-btn" onclick='removeFromUsage(${{JSON.stringify(row.crop_id)}}, ${{JSON.stringify(row.part_num)}}, ${{Number(row.color_id || 0)}}, ${{JSON.stringify(row.element_id || "")}})'>Remove from this crop</button>
                </div>
              </div>
            `;
          }}).join("");
        }}

        function openWhereUsed(partNum, colorId) {{
          whereUsedState = {{
            partNum: String(partNum || ""),
            colorId: Number(colorId || 0),
            filterText: "",
          }};
          renderWhereUsed(whereUsedState.partNum, whereUsedState.colorId);
          const modal = document.getElementById("where-used-modal");
          if (modal) {{
            modal.classList.add("open");
          }}
        }}

        function jumpToCropFromUsage(cropId) {{
          selectCrop(String(cropId || ""));
          const el = document.getElementById(String(cropId || ""));
          if (el) {{
            el.scrollIntoView({{ behavior: "smooth", block: "center" }});
          }}
          if (whereUsedState) {{
            renderWhereUsed(whereUsedState.partNum, whereUsedState.colorId);
          }}
        }}

        async function removeFromUsage(cropId, partNum, colorId, elementId) {{
          await removeAssignedPart(String(cropId || ""), String(partNum || ""), Number(colorId || 0), elementId || null);
          if (whereUsedState) {{
            renderWhereUsed(whereUsedState.partNum, whereUsedState.colorId);
          }}
        }}

        async function removeAllWhereUsed() {{
          if (!whereUsedState) {{
            return;
          }}
          const rows = computeWhereUsedRows(whereUsedState.partNum, whereUsedState.colorId);
          if (!rows.length) {{
            return;
          }}
          const totalQty = rows.reduce((sum, row) => sum + Number(row.qty_value || 0), 0);
          const confirmed = window.confirm(
            "Remove " + rows.length + " crop assignment" + (rows.length === 1 ? "" : "s") +
            " (" + totalQty + " qty) for part " + whereUsedState.partNum +
            " / color " + whereUsedState.colorId + "?"
          );
          if (!confirmed) {{
            return;
          }}
          for (const row of rows) {{
            await removeAssignedPart(row.crop_id, row.part_num, row.color_id, row.element_id || null);
          }}
          if (whereUsedState) {{
            renderWhereUsed(whereUsedState.partNum, whereUsedState.colorId);
          }}
        }}

        function updateBagAssignmentSummary() {{
          const assignedTotals = computeAssignedPartTotals();
          let bagAssignedParts = 0;
          let remainingCandidateParts = 0;
          partRecords.forEach((part) => {{
            const key = partKey(part.part_num, part.color_id);
            const requiredQty = Number(part.qty || 0);
            const assignedQty = Number(assignedTotals.get(key) || 0);
            const remainingQty = requiredQty - assignedQty;
            bagAssignedParts += assignedQty;
            if (remainingQty > 0) {{
              remainingCandidateParts += 1;
            }}
          }});
          const assignedEl = document.getElementById("bag-assignment-summary");
          if (assignedEl) {{
            assignedEl.textContent = "Bag assigned parts: " + bagAssignedParts;
          }}
          const remainingEl = document.getElementById("remaining-candidates-summary");
          if (remainingEl) {{
            remainingEl.textContent = "Remaining candidate parts: " + remainingCandidateParts;
          }}
        }}

        function updatePartTileAssignmentState() {{
          const assignedTotals = computeAssignedPartTotals();
          const overAssignEnabled = allowOverAssignEnabled();
          const activeCrop = activeCropId ? cropMap.get(activeCropId) : null;
          const slotState = computeCropSlotState(activeCrop);
          const slotLocked = Boolean(activeCrop) && slotState.slotsFull && !slotState.noQtyDetected && !allowExtraPartEnabled();
          partTiles.forEach((tile) => {{
            const partNum = String(tile.dataset.partNum || "");
            const colorId = Number(tile.dataset.partColorId || 0);
            const key = partKey(partNum, colorId);
            const requiredQty = Number(tile.dataset.requiredQty || 0);
            const assignedQty = Number(assignedTotals.get(key) || 0);
            const remainingQty = requiredQty - assignedQty;
            const requiredEl = tile.querySelector(".required-qty");
            const assignedEl = tile.querySelector(".assigned-qty");
            const remainingEl = tile.querySelector(".remaining-qty");
            const overAssignedEl = tile.querySelector(".over-assigned-note");
            if (requiredEl) {{
              requiredEl.textContent = String(requiredQty);
            }}
            if (assignedEl) {{
              assignedEl.textContent = String(assignedQty);
            }}
            if (remainingEl) {{
              remainingEl.textContent = String(remainingQty);
            }}
            if (overAssignedEl) {{
              overAssignedEl.textContent = remainingQty < 0 ? ("Over-assigned by " + Math.abs(remainingQty)) : "";
            }}
            tile.classList.toggle("over-assigned", remainingQty <= 0);
            tile.dataset.remainingQty = String(remainingQty);
            tile.dataset.assignedQty = String(assignedQty);
            tile.dataset.fullyAssigned = String(remainingQty <= 0);
            tile.disabled = slotLocked || (remainingQty <= 0 && !overAssignEnabled);
          }});
          updateBagAssignmentSummary();
          if (whereUsedState) {{
            renderWhereUsed(whereUsedState.partNum, whereUsedState.colorId);
          }}
        }}

        function updateAddAvailability() {{
          const slotStatus = document.getElementById("slot-status-note");
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          const slotState = computeCropSlotState(crop);
          const addLocked = Boolean(crop) && slotState.slotsFull && !slotState.noQtyDetected && !allowExtraPartEnabled();
          const overAssignEnabled = allowOverAssignEnabled();
          const assignedTotals = computeAssignedPartTotals();

          partTiles.forEach((tile) => {{
            const partNum = String(tile.dataset.partNum || "");
            const colorId = Number(tile.dataset.partColorId || 0);
            const key = partKey(partNum, colorId);
            const requiredQty = Number(tile.dataset.requiredQty || 0);
            const assignedQty = Number(assignedTotals.get(key) || 0);
            const remainingQty = requiredQty - assignedQty;
            const fullyAssigned = remainingQty <= 0;
            tile.disabled = addLocked || (fullyAssigned && !overAssignEnabled);
            tile.classList.toggle("slot-locked", addLocked);
            tile.classList.toggle("over-assigned", fullyAssigned);
          }});

          if (!slotStatus) {{
            updateBagAssignmentSummary();
            return;
          }}
          if (!crop) {{
            slotStatus.textContent = "";
            updateBagAssignmentSummary();
            return;
          }}
          if (slotState.noQtyDetected) {{
            slotStatus.textContent = "No qty detected. Manual add defaults to 1x.";
            updateBagAssignmentSummary();
            return;
          }}
          if (addLocked) {{
            slotStatus.textContent = "All qty slots filled. Enable Allow extra part to override.";
            updateBagAssignmentSummary();
            return;
          }}
          slotStatus.textContent = "Slots: " + slotState.filledSlots + " / " + slotState.totalSlots + " filled. Next qty: " + slotState.nextQtyLabel;
          updateBagAssignmentSummary();
        }}

        async function applyPartFilter() {{
          const status = document.getElementById("parts-filter-status");
          if (!partTiles.length) {{
            return;
          }}

          if (showAllParts || !activeCropId) {{
            partTiles.forEach((tile) => tile.classList.remove("filtered-out"));
            if (status) {{
              status.textContent = activeCropId
                ? "Showing all part colors for " + activeCropId + "."
                : "Showing all parts. Select a crop to filter by dominant LEGO colors.";
            }}
            updatePartFilterButtons();
            return;
          }}

          const crop = cropMap.get(activeCropId);
          if (!crop) {{
            return;
          }}

          const dominantColorIds = await sampleDominantColorIdsForCrop(crop);
          const filterColorIds = activeFilterColorIds(crop, dominantColorIds);
          if (!filterColorIds.length) {{
            partTiles.forEach((tile) => tile.classList.remove("filtered-out"));
            if (status) {{
              status.textContent = "No dominant LEGO colors detected for " + crop.crop_id + ". Showing all parts.";
            }}
            updatePartFilterButtons();
            return;
          }}

          const allowed = new Set(filterColorIds.map((value) => Number(value)));
          const assignedTotals = computeAssignedPartTotals();
          let visibleCount = 0;
          partTiles.forEach((tile) => {{
            const partNum = String(tile.dataset.partNum || "");
            const colorId = Number(tile.dataset.partColorId || 0);
            const requiredQty = Number(tile.dataset.requiredQty || 0);
            const assignedQty = Number(assignedTotals.get(partKey(partNum, colorId)) || 0);
            const remainingQty = requiredQty - assignedQty;
            const keep = allowed.has(colorId);
            tile.classList.toggle("filtered-out", !keep);
            tile.classList.toggle("over-assigned", remainingQty <= 0);
            if (keep) {{
              visibleCount += 1;
            }}
          }});

          if (status) {{
            const colorLabels = filterColorIds.map((colorId) => colorNameById.get(Number(colorId)) || ("color " + colorId));
            status.textContent = (crop.manual_color_filter_id
              ? "Manual colour filter for "
              : "Filtered ") + crop.crop_id + " to " + visibleCount + " parts matching: " + colorLabels.join(", ");
          }}
          updatePartFilterButtons();
        }}

        async function renderSuggestedParts(cropId) {{
          const panel = document.getElementById("suggested-parts-grid");
          if (!panel) {{
            return;
          }}
          if (!cropId) {{
            panel.innerHTML = '<div class="suggested-empty">Select a crop to see first-pass suggestions.</div>';
            return;
          }}

          const crop = cropMap.get(cropId);
          if (!crop) {{
            panel.innerHTML = '<div class="suggested-empty">Select a crop to see first-pass suggestions.</div>';
            return;
          }}
          const slotState = computeCropSlotState(crop);
          const addLocked = slotState.slotsFull && !slotState.noQtyDetected && !allowExtraPartEnabled();

          const dominantColorIds = await sampleDominantColorIdsForCrop(crop);
          const filterColorIds = activeFilterColorIds(crop, dominantColorIds);
          if (!filterColorIds.length) {{
            panel.innerHTML = '<div class="suggested-empty">No suggestions - use Show all.</div>';
            return;
          }}

          const colorOrder = new Map(filterColorIds.map((value, index) => [Number(value), index]));
          const targetQty = slotState.nextQtyValue;
          const metallicMode = metallicModeEnabled(crop);
          const overAssignEnabled = allowOverAssignEnabled();
          const assignedTotals = computeAssignedPartTotals();
          const assignedKeys = new Set(
            (Array.isArray(crop.parts) ? crop.parts : []).map((part) => partKey(part.part_num, part.color_id))
          );

          const suggestions = partRecords
            .filter((part) => colorOrder.has(Number(part.color_id || 0)))
            .map((part) => {{
              const colorId = Number(part.color_id || 0);
              const setQty = Number(part.qty || 0);
              const key = partKey(part.part_num, colorId);
              const assignedQty = Number(assignedTotals.get(key) || 0);
              const remainingQty = setQty - assignedQty;
              const distanceScore = candidateSuggestionDistance(crop, colorId);
              let suggestionScore = Number.isFinite(distanceScore)
                ? distanceScore
                : (Number(colorOrder.get(colorId) || 0) + 1) * 1000;
              if (metallicMode && metallicColourRank(part.color_name) === 0) {{
                suggestionScore *= 0.7;
              }}
              if (remainingQty > 0) {{
                suggestionScore *= 0.9;
              }}
              suggestionScore *= trainingBoostMultiplier(crop, part.part_num, colorId, distanceScore);
              return {{
                part_num: String(part.part_num || ""),
                color_id: colorId,
                color_name: String(part.color_name || ("color " + colorId)),
                element_id: String(part.element_id || ""),
                img_url: String(part.img_url || ""),
                set_qty: setQty,
                assigned_qty: assignedQty,
                remaining_qty: remainingQty,
                remaining_rank: remainingQty > 0 ? 0 : 1,
                metallic_rank: metallicMode ? metallicColourRank(part.color_name) : 1,
                color_rank: Number(colorOrder.get(colorId) || 0),
                qty_rank: targetQty !== null && setQty >= targetQty ? 0 : 1,
                image_rank: part.img_url ? 0 : 1,
                used_rank: assignedKeys.has(key) ? 1 : 0,
                distance_score: distanceScore,
                suggestion_score: suggestionScore,
              }};
            }})
            .sort((left, right) => {{
              return (
                left.remaining_rank - right.remaining_rank ||
                left.suggestion_score - right.suggestion_score ||
                left.metallic_rank - right.metallic_rank ||
                left.color_rank - right.color_rank ||
                left.qty_rank - right.qty_rank ||
                left.image_rank - right.image_rank ||
                left.used_rank - right.used_rank ||
                right.remaining_qty - left.remaining_qty ||
                right.set_qty - left.set_qty ||
                left.part_num.localeCompare(right.part_num) ||
                left.color_id - right.color_id
              );
            }})
            .slice(0, 12);

          if (!suggestions.length) {{
            panel.innerHTML = '<div class="suggested-empty">No suggestions - use Show all.</div>';
            return;
          }}

          panel.innerHTML = suggestions.map((part) => {{
            const thumb = part.img_url
              ? '<img src="' + escapeHtml(part.img_url) + '" alt="' + escapeHtml(part.part_num) + '" loading="lazy" />'
              : '<div class="crop-missing">No image</div>';
            return `
              <div class="suggested-part">
                <div class="part-thumb">${{thumb}}</div>
                <div class="part-meta">
                  <strong>${{escapeHtml(part.part_num)}}</strong><br/>
                  color: ${{escapeHtml(String(part.color_id))}} / ${{escapeHtml(part.color_name)}}<br/>
                  required / assigned / remaining: ${{escapeHtml(String(part.set_qty || 0))}} / ${{escapeHtml(String(part.assigned_qty || 0))}} / ${{escapeHtml(String(part.remaining_qty || 0))}}<br/>
                  element: ${{escapeHtml(part.element_id || "n/a")}}<br/>
                  set qty: ${{escapeHtml(String(part.set_qty || 0))}}
                </div>
                <div class="suggested-part-actions">
                  <button
                    type="button"
                    class="remove-btn suggested-add-btn"
                    ${{(addLocked || (part.remaining_qty <= 0 && !overAssignEnabled)) ? "disabled" : ""}}
                    onclick='addSuggestedPart(event, ${{JSON.stringify(part.part_num)}}, ${{Number(part.color_id || 0)}}, ${{JSON.stringify(part.element_id || "")}}, ${{JSON.stringify(part.color_name || "")}})'
                  >
                    ${{addLocked ? "Full" : ((part.remaining_qty <= 0 && !overAssignEnabled) ? "Assigned" : "Add")}}
                  </button>
                </div>
              </div>
            `;
          }}).join("");
        }}

        function jumpToSuggestedParts() {{
          const panel = document.getElementById("suggested-parts-panel");
          if (!panel) {{
            return;
          }}
          panel.scrollIntoView({{ behavior: "smooth", block: "start" }});
          panel.focus({{ preventScroll: true }});
          panel.classList.add("flash-highlight");
          window.setTimeout(() => {{
            panel.classList.remove("flash-highlight");
          }}, 900);
          window.setTimeout(() => {{
            const firstSuggestedButton = panel.querySelector(".suggested-part-actions .remove-btn");
            if (firstSuggestedButton) {{
              firstSuggestedButton.focus({{ preventScroll: true }});
            }}
          }}, 220);
        }}

        async function ensureColourPickerImage(crop) {{
          updatePickerDiagnostics(crop, {{ errorMessage: "" }});
          if (!crop || !crop.data_uri) {{
            colourPickerImage = null;
            return null;
          }}
          if (colourPickerImage && colourPickerImage.src === crop.data_uri) {{
            return colourPickerImage;
          }}
          const image = new Image();
          image.decoding = "async";
          const loaded = await new Promise((resolve, reject) => {{
            image.onload = resolve;
            image.onerror = reject;
            image.src = crop.data_uri;
          }}).then(() => true).catch(() => false);
          if (!loaded) {{
            colourPickerImage = null;
            updatePickerDiagnostics(crop, {{ canvasImageLoaded: "no", errorMessage: "Crop preview image failed to load" }});
            return null;
          }}
          colourPickerImage = image;
          return image;
        }}

        async function renderColourPicker(cropId) {{
          const canvas = document.getElementById("colour-picker-canvas");
          const empty = document.getElementById("colour-picker-empty");
          const list = document.getElementById("colour-match-list");
          const pickedText = document.getElementById("picked-rgb-text");
          const pickedSwatch = document.getElementById("picked-rgb-swatch");
          if (!canvas || !empty || !list || !pickedText || !pickedSwatch) {{
            return;
          }}
          updateManualColoursToggleLabel();

          const crop = cropId ? cropMap.get(cropId) : null;
          ensurePickerDiagnostics(crop);
          if (!crop || !crop.data_uri) {{
            canvas.style.display = "none";
            empty.style.display = "none";
            list.innerHTML = '<div class="suggested-empty">Pick a colour from the crop to see the closest LEGO colours.</div>';
            pickedText.textContent = "No colour sampled yet.";
            pickedSwatch.style.background = "transparent";
            renderPickerDiagnostics(crop);
            return;
          }}

          const image = await ensureColourPickerImage(crop);
          if (!image) {{
            canvas.style.display = "none";
            empty.style.display = "none";
            list.innerHTML = '<div class="suggested-empty">Pick a colour from the crop to see the closest LEGO colours.</div>';
            renderPickerDiagnostics(crop);
            return;
          }}

          const maxWidth = 420;
          const scale = Math.min(1, maxWidth / Math.max(1, image.naturalWidth));
          canvas.width = image.naturalWidth;
          canvas.height = image.naturalHeight;
          canvas.style.width = Math.max(1, Math.round(image.naturalWidth * scale)) + "px";
          canvas.style.height = Math.max(1, Math.round(image.naturalHeight * scale)) + "px";
          const ctx = canvas.getContext("2d", {{ willReadFrequently: true }});
          if (!ctx) {{
            updatePickerDiagnostics(crop, {{ canvasImageLoaded: "no", errorMessage: "Canvas context unavailable" }});
            renderPickerDiagnostics(crop);
            return;
          }}
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
          canvas.style.display = "none";
          empty.style.display = "none";
          updatePickerDiagnostics(crop, {{
            canvasImageLoaded: "yes",
            canvasSize: canvas.width + " x " + canvas.height,
            errorMessage: "",
          }});

          if (crop.picked_rgb) {{
            pickedText.textContent = "Picked RGB: " + crop.picked_rgb.r + ", " + crop.picked_rgb.g + ", " + crop.picked_rgb.b;
            pickedSwatch.style.background = colorSwatchStyle(crop.picked_rgb);
          }} else {{
            pickedText.textContent = crop.manual_color_filter_id
              ? "Manual LEGO colour filter: " + (colorNameById.get(Number(crop.manual_color_filter_id)) || ("color " + crop.manual_color_filter_id))
              : "No colour sampled yet.";
            pickedSwatch.style.background = "transparent";
          }}

          renderColourMatches(crop);
          renderPickerDiagnostics(crop);
        }}

        function renderColourMatches(crop) {{
          const list = document.getElementById("colour-match-list");
          if (!list) {{
            return;
          }}
          updateManualColoursToggleLabel();
          const sampledMatches = Array.isArray(crop && crop.closest_color_matches) ? crop.closest_color_matches : [];
          const fallbackMatches = manualColorFallbackMatches(showAllManualColours);
          const matches = sampledMatches.length ? sampledMatches : fallbackMatches;
          updatePickerDiagnostics(crop, {{ closestCount: matches.length }});
          if (!colors.length) {{
            list.innerHTML = '<div class="suggested-empty">No colours available in this set.</div>';
            return;
          }}
          list.innerHTML = matches.map((match) => `
            <button
              type="button"
              class="colour-match${{Number(crop.manual_color_filter_id) === Number(match.color_id) ? " active" : ""}}"
              onclick="setManualColorFilter(${{Number(match.color_id)}})"
            >
              <span class="colour-swatch" style="background: ${{escapeHtml(colorSwatchStyle(match.rgb))}};"></span>
              <span class="colour-match-meta">
                <strong>${{escapeHtml(String(match.color_id))}} / ${{escapeHtml(match.color_name || ("color " + match.color_id))}}</strong><br/>
                picked rgb: ${{escapeHtml(formatRgb(crop.picked_rgb))}}<br/>
                candidate rgb: ${{escapeHtml(formatRgb(match.rgb))}}
              </span>
              <span class="colour-match-distance">${{match.distance_score === null ? "manual" : ("distance " + escapeHtml(match.distance_score.toFixed(3)))}}</span>
            </button>
          `).join("");
        }}

        function isIgnoredSamplePixel(r, g, b) {{
          if (r > 238 && g > 238 && b > 238) {{
            return true;
          }}
          if (r > 210 && g > 220 && b > 228 && b >= g && g >= r) {{
            return true;
          }}
          return false;
        }}

        function sampleRgbArea(ctx, centerX, centerY, radius) {{
          const startX = Math.max(0, centerX - radius);
          const startY = Math.max(0, centerY - radius);
          const width = Math.max(1, Math.min(ctx.canvas.width - startX, (radius * 2) + 1));
          const height = Math.max(1, Math.min(ctx.canvas.height - startY, (radius * 2) + 1));
          const data = ctx.getImageData(startX, startY, width, height).data;
          let totalR = 0;
          let totalG = 0;
          let totalB = 0;
          let count = 0;
          for (let index = 0; index < data.length; index += 4) {{
            const alpha = data[index + 3];
            if (alpha < 10) {{
              continue;
            }}
            if (isIgnoredSamplePixel(data[index], data[index + 1], data[index + 2])) {{
              continue;
            }}
            totalR += data[index];
            totalG += data[index + 1];
            totalB += data[index + 2];
            count += 1;
          }}
          if (!count) {{
            return null;
          }}
          return {{
            r: Math.round(totalR / count),
            g: Math.round(totalG / count),
            b: Math.round(totalB / count)
          }};
        }}

        function closestLegoColorMatches(rgb, limit = 6, options = {{}}) {{
          const metallicMode = !!(options && options.metallicMode);
          const candidates = (window.legoColors || [])
            .filter((candidate) => showAllManualColours || Number(candidate && candidate.color_id) !== 9999);
          console.log("nearest candidates", candidates.length);
          const results = candidates
            .map((candidate) => {{
              const parsedRgb = parseRgbHex(candidate && candidate.rgb);
              console.log("parse test", candidate && candidate.rgb, parsedRgb);
              if (!parsedRgb) {{
                return null;
              }}
              return {{
                color_id: candidate.color_id,
                color_name: candidate.color_name,
                rgb: parsedRgb,
                metallic_rank: metallicMode ? metallicColourRank(candidate && candidate.color_name) : 1,
                distance_score: colorDistance(rgb, parsedRgb)
              }};
            }})
            .filter(Boolean)
            .sort((left, right) => (
              left.metallic_rank - right.metallic_rank
              || left.distance_score - right.distance_score
              || left.color_id - right.color_id
            ))
            .slice(0, limit);
          console.log("nearest results", results.length);
          if (typeof console !== "undefined" && console.debug) {{
            results.forEach((match) => {{
              console.debug("[colour-picker]", {{
                picked_rgb: rgb,
                candidate_rgb: match.rgb,
                color_id: match.color_id,
                color_name: match.color_name,
                distance: Number(match.distance_score.toFixed(3))
              }});
            }});
          }}
          return results;
        }}

        function setManualColorFilter(colorId) {{
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          if (!crop) {{
            return;
          }}
          crop.manual_color_filter_id = Number(colorId);
          updatePickerDiagnostics(crop, {{ errorMessage: "" }});
          const pickedText = document.getElementById("picked-rgb-text");
          const colorName = colorNameById.get(Number(colorId)) || ("color " + colorId);
          if (pickedText) {{
            pickedText.textContent = "Manual LEGO colour filter: " + colorName + " (" + colorId + ")";
          }}
          renderColourMatches(crop);
          applyPartFilter();
          renderSuggestedParts(crop.crop_id);
          renderColourPicker(crop.crop_id);
        }}

        function toggleShowAllManualColours() {{
          showAllManualColours = !showAllManualColours;
          updateManualColoursToggleLabel();
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          if (!crop) {{
            return;
          }}
          if (crop.picked_rgb) {{
            crop.closest_color_matches = closestLegoColorMatches(crop.picked_rgb, 6, {{
              metallicMode: metallicModeEnabled(crop),
            }});
          }}
          renderColourPicker(crop.crop_id);
        }}

        function clearManualColorFilter() {{
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          if (!crop) {{
            return;
          }}
          delete crop.manual_color_filter_id;
          delete crop.closest_color_matches;
          delete crop.picked_rgb;
          updatePickerDiagnostics(crop, {{
            sampledRgb: "n/a",
            closestCount: 0,
            metallicMode: metallicModeText(crop),
            errorMessage: "",
          }});
          applyPartFilter();
          renderSuggestedParts(crop.crop_id);
          renderColourPicker(crop.crop_id);
        }}

        function handleColourPickerCanvasClick(event) {{
          const colourPickerCanvas = event && event.currentTarget ? event.currentTarget : document.getElementById("colour-picker-canvas");
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          if (!crop || !colourPickerCanvas) {{
            return;
          }}
          const ctx = colourPickerCanvas.getContext("2d", {{ willReadFrequently: true }});
          if (!ctx) {{
            updatePickerDiagnostics(crop, {{ errorMessage: "Canvas context unavailable on click" }});
            renderPickerDiagnostics(crop);
            return;
          }}
          const rect = colourPickerCanvas.getBoundingClientRect();
          if (!rect.width || !rect.height || !colourPickerCanvas.width || !colourPickerCanvas.height) {{
            updatePickerDiagnostics(crop, {{ errorMessage: "Canvas is not ready for sampling" }});
            renderPickerDiagnostics(crop);
            return;
          }}
          const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
          const y = Math.max(0, Math.min(rect.height, event.clientY - rect.top));
          const canvasX = Math.min(
            colourPickerCanvas.width - 1,
            Math.max(0, Math.round((x / rect.width) * colourPickerCanvas.width))
          );
          const canvasY = Math.min(
            colourPickerCanvas.height - 1,
            Math.max(0, Math.round((y / rect.height) * colourPickerCanvas.height))
          );
          updatePickerDiagnostics(crop, {{
            lastClick: canvasX + ", " + canvasY,
            canvasImageLoaded: "yes",
            canvasSize: colourPickerCanvas.width + " x " + colourPickerCanvas.height,
            errorMessage: "",
          }});
          const rgb = sampleRgbArea(ctx, canvasX, canvasY, 2);
          if (!rgb) {{
            delete crop.picked_rgb;
            crop.closest_color_matches = [];
            updatePickerDiagnostics(crop, {{
              sampledRgb: "No valid colour sampled",
              closestCount: 0,
              errorMessage: "No valid colour sampled",
            }});
            const pickedText = document.getElementById("picked-rgb-text");
            const pickedSwatch = document.getElementById("picked-rgb-swatch");
            if (pickedText) {{
              pickedText.textContent = "No valid colour sampled";
            }}
            if (pickedSwatch) {{
              pickedSwatch.style.background = "transparent";
            }}
            renderColourMatches(crop);
            renderPickerDiagnostics(crop);
            return;
          }}
          crop.picked_rgb = rgb;
          crop.closest_color_matches = closestLegoColorMatches(rgb, 6, {{
            metallicMode: metallicModeEnabled(crop),
          }});
          updatePickerDiagnostics(crop, {{
            sampledRgb: formatRgb(rgb),
            closestCount: crop.closest_color_matches.length,
            metallicMode: metallicModeText(crop),
            errorMessage: "",
          }});
          if (crop.closest_color_matches.length) {{
            crop.manual_color_filter_id = Number(crop.closest_color_matches[0].color_id);
          }}
          renderColourPicker(crop.crop_id);
          applyPartFilter();
          renderSuggestedParts(crop.crop_id);
        }}

        function bindColourPickerCanvas() {{
          const colourPickerCanvas = document.getElementById("colour-picker-canvas");
          if (!colourPickerCanvas || colourPickerCanvas.dataset.boundClick === "1") {{
            return;
          }}
          colourPickerCanvas.addEventListener("click", handleColourPickerCanvasClick);
          colourPickerCanvas.dataset.boundClick = "1";
        }}

        function setPartFilterMode(showAll) {{
          showAllParts = Boolean(showAll);
          applyPartFilter();
        }}

        function openCropZoomFromEl(event, el) {{
          if (!el || !el.dataset) {{
            return;
          }}
          openCropZoom(event, el.dataset.src, el.dataset.cropId);
        }}

        function openCropZoom(event, imageSrc, cropId) {{
          event.stopPropagation();
          const modal = document.getElementById("crop-zoom-modal");
          const image = document.getElementById("crop-zoom-image");
          const caption = document.getElementById("crop-zoom-caption");
          if (!modal || !image || !caption || !imageSrc) {{
            return;
          }}
          image.src = imageSrc;
          caption.textContent = cropId ? "Crop: " + cropId : "";
          modal.classList.add("open");
        }}

        function closeCropZoom() {{
          const modal = document.getElementById("crop-zoom-modal");
          const image = document.getElementById("crop-zoom-image");
          const caption = document.getElementById("crop-zoom-caption");
          if (!modal || !image || !caption) {{
            return;
          }}
          modal.classList.remove("open");
          image.removeAttribute("src");
          caption.textContent = "";
        }}

        async function sampleDominantColorIdsForCrop(crop) {{
          if (!crop || !crop.data_uri) {{
            return [];
          }}
          if (Array.isArray(crop.dominant_color_ids)) {{
            return crop.dominant_color_ids;
          }}

          const image = new Image();
          image.decoding = "async";
          const loaded = await new Promise((resolve, reject) => {{
            image.onload = resolve;
            image.onerror = reject;
            image.src = crop.data_uri;
          }}).then(() => true).catch(() => false);

          if (!loaded || !image.naturalWidth || !image.naturalHeight) {{
            crop.dominant_color_ids = [];
            return crop.dominant_color_ids;
          }}

          const canvas = document.createElement("canvas");
          const width = Math.max(1, Math.min(120, image.naturalWidth));
          const height = Math.max(1, Math.round(image.naturalHeight * (width / image.naturalWidth)));
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext("2d", {{ willReadFrequently: true }});
          if (!ctx) {{
            crop.dominant_color_ids = [];
            return crop.dominant_color_ids;
          }}

          ctx.drawImage(image, 0, 0, width, height);
          const data = ctx.getImageData(0, 0, width, height).data;
          const buckets = new Map();
          for (let index = 0; index < data.length; index += 16) {{
            const r = data[index];
            const g = data[index + 1];
            const b = data[index + 2];
            const a = data[index + 3];
            if (a < 10) {{
              continue;
            }}
            if (isIgnoredSamplePixel(r, g, b)) {{
              continue;
            }}
            const key = [
              Math.floor(r / 32),
              Math.floor(g / 32),
              Math.floor(b / 32)
            ].join(":");
            const current = buckets.get(key) || {{ count: 0, r: 0, g: 0, b: 0 }};
            current.count += 1;
            current.r += r;
            current.g += g;
            current.b += b;
            buckets.set(key, current);
          }}

          const dominantIds = [];
          const seenIds = new Set();
          const sortedBuckets = Array.from(buckets.values()).sort((left, right) => right.count - left.count);
          for (const bucket of sortedBuckets.slice(0, 6)) {{
            if (!bucket.count) {{
              continue;
            }}
            const avg = {{
              r: Math.round(bucket.r / bucket.count),
              g: Math.round(bucket.g / bucket.count),
              b: Math.round(bucket.b / bucket.count)
            }};
            const closest = closestLegoColorId(avg);
            if (!closest || seenIds.has(closest.color_id)) {{
              continue;
            }}
            seenIds.add(closest.color_id);
            dominantIds.push(closest.color_id);
            if (dominantIds.length >= 3) {{
              break;
            }}
          }}

          crop.dominant_color_ids = dominantIds;
          updatePickerDiagnostics(crop, {{
            metallicMode: metallicModeText(crop),
          }});
          return dominantIds;
        }}

        function updateManualSelectionBox(page, rect) {{
          const box = document.getElementById("manual-selection-box-" + page);
          const readout = document.getElementById("manual-selection-readout-" + page);
          if (!box || !readout) {{
            return;
          }}
          if (!rect) {{
            box.classList.remove("visible");
            box.style.left = "0px";
            box.style.top = "0px";
            box.style.width = "0px";
            box.style.height = "0px";
            readout.textContent = "Drag on the page image to select a crop.";
            return;
          }}
          box.classList.add("visible");
          box.style.left = rect.displayX + "px";
          box.style.top = rect.displayY + "px";
          box.style.width = rect.displayW + "px";
          box.style.height = rect.displayH + "px";
          readout.textContent = "Selected crop: x=" + rect.cropBox[0] + ", y=" + rect.cropBox[1] + ", w=" + rect.cropBox[2] + ", h=" + rect.cropBox[3];
        }}

        function setupManualCropCanvas(canvas) {{
          const page = Number(canvas.dataset.page || 0);
          const naturalWidth = Number(canvas.dataset.imageWidth || 0);
          const naturalHeight = Number(canvas.dataset.imageHeight || 0);
          const state = {{
            dragging: false,
            startX: 0,
            startY: 0
          }};

          function pointerPosition(event) {{
            const rect = canvas.getBoundingClientRect();
            const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
            const y = Math.min(Math.max(event.clientY - rect.top, 0), rect.height);
            return {{ x, y, width: rect.width, height: rect.height }};
          }}

          function buildSelection(event) {{
            const pos = pointerPosition(event);
            const left = Math.min(state.startX, pos.x);
            const top = Math.min(state.startY, pos.y);
            const width = Math.abs(pos.x - state.startX);
            const height = Math.abs(pos.y - state.startY);
            const scaleX = naturalWidth > 0 && pos.width > 0 ? naturalWidth / pos.width : 1;
            const scaleY = naturalHeight > 0 && pos.height > 0 ? naturalHeight / pos.height : 1;
            return {{
              displayX: Math.round(left),
              displayY: Math.round(top),
              displayW: Math.round(width),
              displayH: Math.round(height),
              cropBox: [
                Math.round(left * scaleX),
                Math.round(top * scaleY),
                Math.round(width * scaleX),
                Math.round(height * scaleY)
              ]
            }};
          }}

          canvas.addEventListener("pointerdown", (event) => {{
            state.dragging = true;
            canvas.setPointerCapture(event.pointerId);
            const pos = pointerPosition(event);
            state.startX = pos.x;
            state.startY = pos.y;
            const initial = {{
              displayX: Math.round(pos.x),
              displayY: Math.round(pos.y),
              displayW: 0,
              displayH: 0,
              cropBox: [
                Math.round(pos.x * (naturalWidth > 0 && pos.width > 0 ? naturalWidth / pos.width : 1)),
                Math.round(pos.y * (naturalHeight > 0 && pos.height > 0 ? naturalHeight / pos.height : 1)),
                0,
                0
              ]
            }};
            manualSelections.set(page, initial);
            updateManualSelectionBox(page, initial);
          }});

          canvas.addEventListener("pointermove", (event) => {{
            if (!state.dragging) {{
              return;
            }}
            const selection = buildSelection(event);
            manualSelections.set(page, selection);
            updateManualSelectionBox(page, selection);
          }});

          function finishSelection(event) {{
            if (!state.dragging) {{
              return;
            }}
            state.dragging = false;
            const selection = buildSelection(event);
            if (selection.cropBox[2] < 4 || selection.cropBox[3] < 4) {{
              manualSelections.delete(page);
              updateManualSelectionBox(page, null);
              return;
            }}
            manualSelections.set(page, selection);
            updateManualSelectionBox(page, selection);
          }}

          canvas.addEventListener("pointerup", finishSelection);
          canvas.addEventListener("pointercancel", finishSelection);
        }}

        async function saveManualCrop(page) {{
          const selection = manualSelections.get(page);
          if (!selection) {{
            alert("Draw a crop rectangle first");
            return;
          }}

          const stepInput = document.getElementById("manual-step-" + page);
          const pageCard = document.querySelector('.manual-page-card[data-page="' + page + '"]');
          const cropImagePath = pageCard ? pageCard.dataset.imagePath || "" : "";
          const stepValue = stepInput ? Number(stepInput.value || "") : NaN;
          if (!Number.isFinite(stepValue) || stepValue < 0) {{
            alert("Enter a step number first");
            return;
          }}
          if (!cropImagePath) {{
            alert("Missing page image path");
            return;
          }}

          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            page: page,
            step: Math.round(stepValue),
            crop_box: selection.cropBox,
            crop_image_path: cropImagePath
          }};

          const res = await fetch("/debug/save-manual-crop", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload)
          }});

          const saveStatus = document.getElementById("save-status");
          if (!res.ok) {{
            saveStatus.textContent = "Manual crop save failed";
            alert("Manual crop save failed");
            return;
          }}

          saveStatus.textContent = "Saved manual crop for page " + page + ". Refreshing view...";
          window.location.reload();
        }}

        function assignedPartsMarkup(cropId) {{
          const crop = cropMap.get(cropId);
          if (!crop) {{
            return '<div class="assigned-empty">No assigned parts yet.</div>';
          }}

          const parts = Array.isArray(crop.parts) ? crop.parts : [];
          if (!parts.length) {{
            return '<div class="assigned-empty">No assigned parts yet.</div>';
          }}

          return parts.map((part) => {{
            const partForView = hydratePart(part);
            const thumb = partForView.img_url
              ? '<img src="' + escapeHtml(partForView.img_url) + '" alt="' + escapeHtml(partForView.part_num) + '" loading="lazy" />'
              : '<div class="crop-missing">No image</div>';
            const qtyInputId = "assigned-qty-input-" + cropId + "-" + partKey(part.part_num, part.color_id) + "-" + String(part.element_id || "none");
            return `
              <div class="assigned-part">
                <div class="assigned-part-thumb">${{thumb}}</div>
                <div class="assigned-part-meta">
                  <strong>${{escapeHtml(partForView.part_num)}}</strong><br/>
                  color: ${{escapeHtml(partForView.color_name)}}<br/>
                  selected qty: ${{escapeHtml(partForView.selected_qty_label)}}<br/>
                  element: ${{escapeHtml(partForView.element_id || "n/a")}}
                </div>
                <div class="assigned-part-actions">
                  <div class="assigned-qty-row">
                    <label for="${{escapeHtml(qtyInputId)}}">selected qty:</label>
                    <input
                      id="${{escapeHtml(qtyInputId)}}"
                      class="assigned-qty-input"
                      type="text"
                      value="${{escapeHtml(partForView.selected_qty_label)}}"
                      placeholder="1x"
                    />
                    <button
                      type="button"
                      class="remove-btn"
                      onclick='updateAssignedPartQty(event, "${{escapeHtml(cropId)}}", ${{JSON.stringify(part.part_num)}}, ${{Number(part.color_id || 0)}}, ${{JSON.stringify(part.element_id || "")}}, ${{JSON.stringify(qtyInputId)}})'
                    >
                      Save
                    </button>
                  </div>
                  <button
                    type="button"
                    class="remove-btn"
                    onclick='removeAssignedPart("${{escapeHtml(cropId)}}", ${{JSON.stringify(part.part_num)}}, ${{Number(part.color_id || 0)}}, ${{JSON.stringify(part.element_id || "")}})'
                  >
                    Remove
                  </button>
                </div>
              </div>
            `;
          }}).join("");
        }}

        function renderAssignedParts(cropId) {{
          const crop = cropMap.get(cropId);
          const container = document.getElementById("assigned-" + cropId);
          if (!crop || !container) {{
            return;
          }}
          container.innerHTML = assignedPartsMarkup(cropId);
          updateCropCardVisuals(cropId);
          updatePartTileAssignmentState();
        }}

        function selectCrop(cropId) {{
          const crop = cropMap.get(cropId);
          if (!crop || (crop.is_hidden && !showHiddenCrops)) {{
            return;
          }}
          console.log("[instruction-buildability] selected crop id", cropId);
          activeCropId = cropId;
          window.selectedCropId = String(cropId || "");
          document.querySelectorAll('.crop-card').forEach((el) => {{
            el.classList.remove('selected');
          }});
          const el = document.getElementById(cropId);
          if (el) {{
            el.classList.add('selected');
          }}
          updateSelectedCropStatus(crop);
          renderSelectedCropWorkspace(cropId);
          document.getElementById('save-status').textContent = "";
          updateAddAvailability();
          applyPartFilter();
          renderSuggestedParts(cropId);
          renderColourPicker(cropId);
          if (whereUsedState) {{
            renderWhereUsed(whereUsedState.partNum, whereUsedState.colorId);
          }}
        }}

        function toggleMetallicMode(event, cropId, enabled) {{
          event.stopPropagation();
          const crop = cropMap.get(cropId);
          if (!crop) {{
            return;
          }}
          crop.manual_metallic_mode = Boolean(enabled);
          updatePickerDiagnostics(crop, {{
            metallicMode: metallicModeText(crop),
            errorMessage: "",
          }});
          if (crop.picked_rgb) {{
            crop.closest_color_matches = closestLegoColorMatches(crop.picked_rgb, 6, {{
              metallicMode: metallicModeEnabled(crop),
            }});
          }}
          updateCropCardVisuals(cropId);
          if (activeCropId === cropId) {{
            applyPartFilter();
            renderSuggestedParts(cropId);
            renderColourPicker(cropId);
          }}
        }}

        async function setCropStatus(event, cropId, statusValue) {{
          event.stopPropagation();
          const crop = cropMap.get(cropId);
          if (!crop) {{
            return;
          }}

          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            crop_id: crop.crop_id,
            page: crop.page,
            step: crop.step,
            crop_qty: crop.qty_numbers,
            crop_qty_text: crop.qty_text,
            crop_box: crop.crop_box,
            crop_box_format: crop.crop_box_format,
            crop_image_path: crop.crop_image_path,
            crop_confidence: crop.confidence,
            status: statusValue
          }};

          const res = await fetch("/debug/set-crop-status", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload)
          }});

          const saveStatus = document.getElementById('save-status');
          if (!res.ok) {{
            saveStatus.textContent = "Status update failed";
            alert("Status update failed");
            return;
          }}

          const result = await res.json();
          syncCropFromResponse(crop, result.crop);
          updateCropCardVisuals(cropId);
          if (activeCropId === cropId) {{
            renderSuggestedParts(cropId);
          }}
          updatePartTileAssignmentState();
          saveStatus.textContent = "Updated " + crop.crop_id + " status to " + statusValue;
        }}

        async function hideCrop(event, cropId) {{
          event.stopPropagation();
          const crop = cropMap.get(cropId);
          if (!crop) {{
            return;
          }}

          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            crop_id: crop.crop_id,
            page: crop.page,
            step: crop.step,
            crop_qty: crop.qty_numbers,
            crop_qty_text: crop.qty_text,
            crop_box: crop.crop_box,
            crop_box_format: crop.crop_box_format,
            crop_image_path: crop.crop_image_path,
            crop_confidence: crop.confidence,
            status: "hidden"
          }};

          const res = await fetch("/debug/set-crop-status", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload)
          }});

          const saveStatus = document.getElementById("save-status");
          if (!res.ok) {{
            saveStatus.textContent = "Hide failed";
            alert("Hide failed");
            return;
          }}

          const result = await res.json();
          syncCropFromResponse(crop, result.crop);
          updateCropCardVisuals(cropId);
          saveStatus.textContent = "Hidden " + crop.crop_id;
          if (activeCropId === cropId && crop.is_hidden && !showHiddenCrops) {{
            activeCropId = null;
            updateSelectedCropStatus(null);
            renderSelectedCropWorkspace(null);
            updateAddAvailability();
            const nextCropId = nextVisibleCropId(cropId);
            if (nextCropId && nextCropId !== cropId) {{
              selectCrop(nextCropId);
            }}
          }}
          updatePartTileAssignmentState();
          applyPartFilter();
        }}

        async function deleteCrop(event, cropId) {{
          event.stopPropagation();
          const crop = cropMap.get(cropId);
          if (!crop) {{
            return;
          }}

          const confirmMessage = crop.is_manual
            ? "Delete this manual crop from the training JSON?"
            : "Hide this detected crop from the training UI?";
          if (!window.confirm(confirmMessage)) {{
            return;
          }}

          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            crop_id: crop.crop_id,
            page: crop.page,
            step: crop.step,
            crop_qty: crop.qty_numbers,
            crop_qty_text: crop.qty_text,
            crop_box: crop.crop_box,
            crop_box_format: crop.crop_box_format,
            crop_image_path: crop.crop_image_path,
            crop_confidence: crop.confidence
          }};

          const res = await fetch("/debug/delete-crop", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload)
          }});

          const saveStatus = document.getElementById("save-status");
          if (!res.ok) {{
            saveStatus.textContent = "Crop delete failed";
            alert("Crop delete failed");
            return;
          }}

          const result = await res.json();
          if (result.deleted) {{
            const nextCropId = nextVisibleCropId(cropId);
            cropMap.delete(cropId);
            const cropIndex = cropRecords.findIndex((item) => item.crop_id === cropId);
            if (cropIndex >= 0) {{
              cropRecords.splice(cropIndex, 1);
            }}
            const card = document.getElementById(cropId);
            if (card) {{
              card.remove();
            }}
            if (activeCropId === cropId) {{
              activeCropId = null;
              updateSelectedCropStatus(null);
              renderSelectedCropWorkspace(null);
              updateAddAvailability();
            }}
            saveStatus.textContent = "Deleted manual crop " + cropId;
            if (nextCropId && nextCropId !== cropId) {{
              selectCrop(nextCropId);
            }}
          }} else if (result.crop) {{
            syncCropFromResponse(crop, result.crop);
            updateCropCardVisuals(cropId);
            if (activeCropId === cropId && crop.is_hidden && !showHiddenCrops) {{
              activeCropId = null;
              updateSelectedCropStatus(null);
              renderSelectedCropWorkspace(null);
              updateAddAvailability();
              const nextCropId = nextVisibleCropId(cropId);
              if (nextCropId && nextCropId !== cropId) {{
                selectCrop(nextCropId);
              }}
            }}
            saveStatus.textContent = "Hidden detected crop " + cropId;
          }}
          updatePartTileAssignmentState();
          applyPartFilter();
        }}

        async function updateCropQty(event, cropId) {{
          event.stopPropagation();
          const crop = cropMap.get(cropId);
          const qtyInput = document.getElementById("qty-input-" + cropId);
          if (!crop || !qtyInput) {{
            return;
          }}

          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            crop_id: crop.crop_id,
            page: crop.page,
            step: crop.step,
            crop_box: crop.crop_box,
            crop_box_format: crop.crop_box_format,
            crop_image_path: crop.crop_image_path,
            crop_confidence: crop.confidence,
            qty_input: qtyInput.value
          }};

          const res = await fetch("/debug/update-crop-qty", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload)
          }});

          const saveStatus = document.getElementById("save-status");
          if (!res.ok) {{
            saveStatus.textContent = "Qty update failed";
            alert("Qty update failed");
            return;
          }}

          const result = await res.json();
          syncCropFromResponse(crop, result.crop);
          updateCropCardVisuals(cropId);
          renderAssignedParts(cropId);
          if (activeCropId === cropId) {{
            renderSuggestedParts(cropId);
          }}
          updatePartTileAssignmentState();
          saveStatus.textContent = "Updated qty sequence for " + cropId;
        }}

        async function updateAssignedPartQty(event, cropId, partNum, colorId, elementId, inputId) {{
          event.stopPropagation();
          const crop = cropMap.get(cropId);
          const qtyInput = document.getElementById(String(inputId || ""));
          if (!crop || !qtyInput) {{
            return;
          }}

          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            crop_id: crop.crop_id,
            part_num: partNum,
            color_id: colorId,
            element_id: elementId || null,
            qty_input: qtyInput.value
          }};

          const res = await fetch("/debug/update-assigned-part-qty", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload)
          }});

          const saveStatus = document.getElementById("save-status");
          if (!res.ok) {{
            saveStatus.textContent = "Assigned qty update failed";
            alert("Assigned qty update failed");
            return;
          }}

          const result = await res.json();
          syncCropFromResponse(crop, result.crop);
          renderAssignedParts(cropId);
          if (activeCropId === cropId) {{
            renderSuggestedParts(cropId);
          }}
          if (whereUsedState) {{
            renderWhereUsed(whereUsedState.partNum, whereUsedState.colorId);
          }}
          updatePartTileAssignmentState();
          saveStatus.textContent = "Updated selected qty for " + partNum + " / color " + colorId + " in " + crop.crop_id;
        }}

        async function removeAssignedPart(cropId, partNum, colorId, elementId) {{
          const crop = cropMap.get(cropId);
          if (!crop) {{
            return;
          }}

          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            crop_id: crop.crop_id,
            part_num: partNum,
            color_id: colorId,
            element_id: elementId || null
          }};

          const res = await fetch("/debug/remove-label", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload)
          }});

          const saveStatus = document.getElementById('save-status');
          if (!res.ok) {{
            saveStatus.textContent = "Remove failed";
            alert("Remove failed");
            return;
          }}

          const result = await res.json();
          syncCropFromResponse(crop, result.crop);
          renderAssignedParts(cropId);
          if (activeCropId === cropId) {{
            renderSuggestedParts(cropId);
          }}
          updatePartTileAssignmentState();
          saveStatus.textContent = "Removed " + partNum + " / color " + colorId + " from " + crop.crop_id;
        }}

        async function addSuggestedPart(event, partNum, colorId, elementId, colorName) {{
          event.stopPropagation();
          await selectTile(partNum, colorId, elementId, colorName);
        }}

        function jumpToSelectedCropCard() {{
          if (!activeCropId) {{
            setSelectedCropInlineStatus("No crop selected.", true);
            return;
          }}
          const el = document.getElementById(activeCropId);
          if (!el) {{
            setSelectedCropInlineStatus("Selected crop card not found.", true);
            return;
          }}
          el.scrollIntoView({{ behavior: "smooth", block: "start" }});
          el.classList.add("flash-highlight");
          window.setTimeout(() => {{
            el.classList.remove("flash-highlight");
          }}, 900);
        }}

        async function findAiMatchForSelectedCrop() {{
          const selectedCropId = activeCropId || window.selectedCropId || null;
          console.log("[instruction-buildability] find ai clicked", selectedCropId);
          const crop = selectedCropId ? cropMap.get(selectedCropId) : null;
          const saveStatus = document.getElementById("save-status");
          const findButton = document.querySelector("#selected-crop-workspace button[onclick='findAiMatchForSelectedCrop()']");
          if (!crop) {{
            setSelectedCropInlineStatus("Select a crop first.", true);
            if (saveStatus) {{
              saveStatus.textContent = "Select a crop first";
            }}
            return;
          }}
          if (!{str(ai_rank_enabled).lower()}) {{
            setSelectedCropInlineStatus("Enable ai_rank=1 to use AI matching.", true);
            if (saveStatus) {{
              saveStatus.textContent = "Enable ai_rank=1 to use AI matching";
            }}
            return;
          }}
          if (!Array.isArray(crop.top_part_candidates) || !crop.top_part_candidates.length) {{
            setSelectedCropInlineStatus("No local candidates available for AI ranking.", true);
            if (saveStatus) {{
              saveStatus.textContent = "No local candidates available for AI ranking";
            }}
            return;
          }}
          if (findButton) {{
            findButton.disabled = true;
            findButton.textContent = "Loading...";
          }}
          try {{
            setSelectedCropInlineStatus("Loading AI match...");
            const res = await fetch("/debug/ai-rank-crop", {{
              method: "POST",
              headers: {{"Content-Type": "application/json"}},
              body: JSON.stringify({{
                crop: {{
                  crop_id: crop.crop_id,
                  page: crop.page,
                  step: crop.step,
                  data_uri: crop.data_uri,
                  crop_image_path: crop.crop_image_path,
                  crop_box: crop.crop_box,
                  crop_box_format: crop.crop_box_format,
                  status: crop.status,
                  review_status: crop.review_status,
                  slot_filled: crop.slot_filled,
                  slot_total: crop.slot_total,
                }},
                top_part_candidates: Array.isArray(crop.top_part_candidates) ? crop.top_part_candidates : [],
              }})
            }});
            if (!res.ok) {{
              let detail = "AI match failed";
              try {{
                const errorPayload = await res.json();
                detail = errorPayload.detail || detail;
              }} catch (_error) {{
                detail = "AI match failed";
              }}
              throw new Error(detail);
            }}
            const result = await res.json();
            crop.top_part_candidates = Array.isArray(result.top_part_candidates) ? result.top_part_candidates : (crop.top_part_candidates || []);
            crop.ai_rank_best_candidate = String(result.ai_rank_best_candidate || "");
            crop.ai_rank_confidence = result.ai_rank_confidence;
            crop.ai_rank_reason = String(result.ai_rank_reason || "");
            crop.ai_rank_candidate = result.ai_rank_candidate || null;
            crop.ai_rank_debug = result.ai_rank_debug || crop.ai_rank_debug || {{}};
            renderSelectedCropWorkspace(crop.crop_id);
            console.log("[instruction-buildability] ai response", result);
            if (saveStatus) {{
              saveStatus.textContent = crop.ai_rank_candidate
                ? "AI match ready for " + crop.crop_id
                : ("No AI pick available for " + crop.crop_id + (crop.ai_rank_reason ? (": " + crop.ai_rank_reason) : ""));
            }}
            setSelectedCropInlineStatus(
              crop.ai_rank_candidate
                ? "AI match ready."
                : ("No AI pick available for this crop." + (crop.ai_rank_reason ? (" " + crop.ai_rank_reason) : "")),
              !crop.ai_rank_candidate
            );
          }} catch (error) {{
            console.error("[instruction-buildability] ai match error", error);
            setSelectedCropInlineStatus(String((error && error.message) || "AI match failed."), true);
            if (saveStatus) {{
              saveStatus.textContent = String((error && error.message) || "AI match failed");
            }}
          }} finally {{
            const refreshedFindButton = document.querySelector("#selected-crop-workspace button[onclick='findAiMatchForSelectedCrop()']");
            if (refreshedFindButton) {{
              refreshedFindButton.disabled = false;
              refreshedFindButton.textContent = "Find AI Match";
            }}
          }}
        }}

        async function useAiPickFromLibrary(event) {{
          const selectedCropId = activeCropId || window.selectedCropId || null;
          console.log("[instruction-buildability] use ai pick clicked", selectedCropId);
          if (!selectedCropId) {{
            setSelectedCropInlineStatus("Select a crop first.", true);
            return;
          }}
          await useAiPick(event, selectedCropId);
        }}

        async function useAiPick(event, cropId) {{
          if (event && event.stopPropagation) {{
            event.stopPropagation();
          }}
          console.log("[instruction-buildability] use ai pick clicked", cropId);
          const crop = cropMap.get(cropId);
          if (!crop) {{
            setSelectedCropInlineStatus("Selected crop not found.", true);
            return;
          }}
          const aiPick = crop.ai_rank_candidate;
          console.log("[instruction-buildability] ai response", aiPick || null, crop.ai_rank_confidence, crop.ai_rank_reason);
          if (!aiPick) {{
            setSelectedCropInlineStatus("No AI pick available for this crop.", true);
            alert("No AI pick available for this crop");
            return;
          }}
          selectCrop(cropId);
          setSelectedCropInlineStatus("Using AI pick...");
          await selectTile(
            String(aiPick.part_num || ""),
            Number(aiPick.color_id || 0),
            aiPick.element_id || null,
            String(aiPick.color_name || "")
          );
        }}

        async function selectTile(partNum, colorId, elementId, colorName) {{
          if (!activeCropId) {{
            setSelectedCropInlineStatus("Select a crop first.", true);
            alert("Select a crop first");
            return;
          }}
          const crop = cropMap.get(activeCropId);
          if (!crop) {{
            setSelectedCropInlineStatus("Selected crop metadata is missing.", true);
            alert("Selected crop metadata is missing");
            return;
          }}
          const inventoryState = partInventoryState(partNum, colorId);
          if (inventoryState.fullyAssigned && !allowOverAssignEnabled()) {{
            alert("This part is already fully assigned for the current bag. Enable Allow over-assign to add more.");
            return;
          }}

          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            crop_id: crop.crop_id,
            page: crop.page,
            step: crop.step,
            crop_qty: crop.qty_numbers,
            crop_qty_text: crop.qty_text,
            crop_box: crop.crop_box,
            crop_box_format: crop.crop_box_format,
            crop_image_path: crop.crop_image_path,
            crop_confidence: crop.confidence,
            part_num: partNum,
            color_id: colorId,
            color_name: colorName || null,
            element_id: elementId || null,
            allow_extra_part: allowExtraPartEnabled()
          }};

          const res = await fetch("/debug/save-label", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload)
          }});

          const status = document.getElementById('save-status');
          if (res.ok) {{
            const result = await res.json();
            syncCropFromResponse(crop, result.crop);
            renderAssignedParts(crop.crop_id);
            if (activeCropId === crop.crop_id) {{
              renderSuggestedParts(crop.crop_id);
              renderSelectedCropWorkspace(crop.crop_id);
            }}
            updatePartTileAssignmentState();
            status.textContent = "Saved " + crop.crop_id + " -> " + partNum + " / color " + colorId;
            setSelectedCropInlineStatus("Saved " + partNum + " / color " + colorId + " to " + crop.crop_id + ".");
          }} else {{
            let detail = "Save failed";
            try {{
              const errorPayload = await res.json();
              detail = errorPayload.detail || detail;
            }} catch (_error) {{
              detail = "Save failed";
            }}
            status.textContent = detail;
            setSelectedCropInlineStatus(detail, true);
            alert(detail);
          }}
        }}

        cropRecords.forEach((crop) => {{
          crop.parts = Array.isArray(crop.parts) ? crop.parts.map((part) => {{
            const hydrated = hydratePart(part);
            hydrated.selected_qty_label = selectedQtyLabel(hydrated.qty, hydrated.qty_text);
            return hydrated;
          }}) : [];
          crop.status = crop.status || "needs_adjust";
          crop.is_hidden = crop.status === "hidden";
          crop.is_manual = String(crop.crop_id || "").startsWith("manual_");
          crop.manual_metallic_mode = Boolean(crop.manual_metallic_mode);
          crop.next_qty_index = Number(crop.next_qty_index || crop.parts.length || 0);
          updatePickerDiagnostics(crop, {{
            metallicMode: metallicModeText(crop),
          }});
          renderAssignedParts(crop.crop_id);
          updateCropCardVisuals(crop.crop_id);
        }});
        renderSelectedCropWorkspace(null);

        const normalizedLegoColors = (window.legoColors || [])
          .map((candidate) => {{
            const colorId = Number(candidate && candidate.color_id);
            const colorName = String((candidate && candidate.color_name) || ("color " + colorId));
            const rgbHex = String((candidate && candidate.rgb) || "").trim().replace(/^#/, "").replace(/^0x/i, "").toUpperCase();
            if (!Number.isFinite(colorId) || rgbHex.length !== 6 || !/^[0-9A-F]{6}$/.test(rgbHex)) {{
              return null;
            }}
            return {{
              color_id: colorId,
              color_name: colorName,
              rgb: rgbHex,
              r: parseInt(rgbHex.slice(0, 2), 16),
              g: parseInt(rgbHex.slice(2, 4), 16),
              b: parseInt(rgbHex.slice(4, 6), 16),
            }};
          }})
          .filter(Boolean);
        const loadedColorsCount = document.getElementById("loaded-lego-colours-count");
        if (loadedColorsCount) {{
          loadedColorsCount.textContent = String(colors.length);
        }}
        console.log("legoColors loaded", window.legoColors);
        console.log("training examples loaded", window.trainingExamples);

        colors.forEach((candidate) => {{
          if (!colorNameById.has(candidate.color_id)) {{
            colorNameById.set(candidate.color_id, candidate.color_name);
          }}
        }});

        updatePartFilterButtons();
        updateManualColoursToggleLabel();
        updatePartTileAssignmentState();
        applyHiddenCropVisibility();
        updateAddAvailability();
        applyPartFilter();
        renderSuggestedParts(activeCropId);
        renderColourPicker(activeCropId);

        document.querySelectorAll(".manual-page-canvas").forEach((canvas) => {{
          setupManualCropCanvas(canvas);
        }});

        window.addEventListener("error", (event) => {{
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          updatePickerDiagnostics(crop, {{
            errorMessage: event && event.message ? String(event.message) : "Unknown JS error"
          }});
          renderPickerDiagnostics(crop);
        }});

        window.addEventListener("unhandledrejection", (event) => {{
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          const reason = event && event.reason ? String(event.reason) : "Unhandled promise rejection";
          updatePickerDiagnostics(crop, {{ errorMessage: reason }});
          renderPickerDiagnostics(crop);
        }});

        bindColourPickerCanvas();

        document.addEventListener("keydown", (event) => {{
          if (event.key === "Escape") {{
            closeCropZoom();
            return;
          }}
          if (event.metaKey || event.ctrlKey || event.altKey) {{
            return;
          }}
          const shortcut = String(event.key || "").toLowerCase();
          if (shortcut === "s") {{
            if (isTypingIntoField(event.target)) {{
              return;
            }}
            if (!activeCropId) {{
              return;
            }}
            event.preventDefault();
            jumpToSuggestedParts();
            return;
          }}
          if (!activeCropId) {{
            return;
          }}
          if (shortcut === "g") {{
            event.preventDefault();
            setCropStatus(event, activeCropId, "good");
            return;
          }}
          if (shortcut === "b") {{
            event.preventDefault();
            setCropStatus(event, activeCropId, "bad");
            return;
          }}
          if (shortcut === "h") {{
            event.preventDefault();
            hideCrop(event, activeCropId);
            return;
          }}
          if (shortcut === "d") {{
            event.preventDefault();
            deleteCrop(event, activeCropId);
            return;
          }}
          if (shortcut === "n") {{
            event.preventDefault();
            goToNextCrop();
            return;
          }}
        }});
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
