def _filter_invalid_step_anchor_boxes(step_boxes):
  """
  Filter out impossible OCR step anchors (e.g., 4-digit numbers like 1583) while preserving real steps (e.g., 79).
  Accepts a list of dicts with a 'step_number' key.
  """
  filtered = []
  for box in step_boxes or []:
    step_number = box.get("step_number")
    try:
      step_number = int(step_number)
    except Exception:
      filtered.append(box)
      continue
    # Reject step numbers that are 4 digits or more (e.g., 1583)
    if 0 < step_number < 1000:
      filtered.append(box)
  return filtered

import base64
from html import escape
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from clean.routers.debug import (
    _build_material_crop_candidates,
    _contact_sheet_step_boxes_from_detected,
    _encode_contact_sheet_crop,
    _encode_debug_image_data_uri,
    _extract_detected_qty_details_from_crop as _debug_extract_detected_qty_details_from_crop,
    _extract_qty_tokens_from_image,
    _require_openai_vision_client_debug,
    _resolve_bag_page_range,
    _response_text_to_json_debug,
)
from clean.services import debug_service, step_detector_service
from clean.services.azure_openai_service import rank_crop_candidates
from clean.services.ai_snap_crop_service import (
    create_shape_mask_for_slot_crop,
    create_shape_masks_for_callout_slots,
    refine_slot_cutout_with_sam,
)
from clean.services.part_candidate_service import get_part_candidates_for_crop
from clean.services.part_crop_normalize_service import normalize_part_crop, normalize_slot_crop_from_qty
from clean.services.instruction_buildability_source import load_instruction_set_parts

router = APIRouter()

_PAGE_CALLOUT_DETECTION_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}


def _page_callout_cache_key(set_num: str, page: int) -> Tuple[str, int]:
    return (str(set_num or "").strip(), int(page or 0))


def _page_callout_cache_entry(set_num: str, page: int, *, rebuild: bool = False) -> Dict[str, Any]:
    key = _page_callout_cache_key(set_num, page)
    if rebuild:
        _PAGE_CALLOUT_DETECTION_CACHE.pop(key, None)
    return _PAGE_CALLOUT_DETECTION_CACHE.setdefault(key, {})


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


def _manual_color_calibration_path(set_num: str) -> Path:
    normalized = str(set_num or "").strip() or "70618"
    safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", normalized)
    return Path("/Users/olly/aim2build-instruction/debug/training_labels") / f"{safe_name}_manual_color_calibration.json"


def _clip_memory_path(set_num: str, bag: int) -> Path:
    return Path("/Users/olly/aim2build-instruction/debug/training_labels") / (
        f"{Path(_coerce_label_filename(set_num, bag)).stem}_clip_memory.json"
    )


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


def _empty_manual_color_calibration(set_num: str) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "set_num": str(set_num or "").strip() or "70618",
        "updated_at": _iso_now(),
        "samples": [],
    }


def _empty_clip_memory(set_num: str, bag: int) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "set_num": str(set_num or "").strip() or "70618",
        "bag": max(1, int(bag or 1)),
        "updated_at": _iso_now(),
        "items": [],
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
        roi_h, roi_w = roi.shape[:2]

        # Primary stage: find the callout as the main foreground component relative
        # to the local page background. If this is weak, we fall back to the
        # existing Hough-line logic below unchanged.
        try:
            border_h = max(1, roi_h // 12)
            border_w = max(1, roi_w // 12)
            border = np.concatenate(
                [
                    roi[:border_h, :, :].reshape(-1, 3),
                    roi[max(0, roi_h - border_h) :, :, :].reshape(-1, 3),
                    roi[:, :border_w, :].reshape(-1, 3),
                    roi[:, max(0, roi_w - border_w) :, :].reshape(-1, 3),
                ],
                axis=0,
            )
            bg = np.median(border.astype(np.float32), axis=0)
            diff = np.linalg.norm(
                roi.astype(np.float32) - bg.reshape(1, 1, 3),
                axis=2,
            )
            fg_mask = (diff > 30.0).astype(np.uint8) * 255
            fg_mask = cv2.morphologyEx(
                fg_mask,
                cv2.MORPH_OPEN,
                np.ones((2, 2), np.uint8),
                iterations=1,
            )
            fg_mask = cv2.morphologyEx(
                fg_mask,
                cv2.MORPH_CLOSE,
                np.ones((5, 5), np.uint8),
                iterations=2,
            )
            contours, _ = cv2.findContours(
                fg_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            components: List[Dict[str, Any]] = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w <= 0 or h <= 0:
                    continue
                page_bottom = int(sy1 + y + h)
                if page_bottom > int(step_y) - 5:
                    continue
                if w < 90 or h < 35:
                    continue
                aspect = w / float(max(h, 1))
                if aspect < 1.1 or aspect > 5.5:
                    continue
                area = float(w * h)
                area_ratio = area / float(max(1, roi_w * roi_h))
                if area_ratio < 0.03 or area_ratio > 0.75:
                    continue
                crop = roi[y : y + h, x : x + w]
                if crop is None or crop.size == 0:
                    continue
                if _yellow_ratio_bgr(crop) > 0.18:
                    continue
                components.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "area": area,
                        "bottom": page_bottom,
                        "center_x": int(sx1 + x + (w // 2)),
                    }
                )

            valid_components: List[Dict[str, Any]] = []
            gap_threshold = max(12, int(roi_w * 0.025))
            for component in components:
                cx = int(component["x"])
                cy = int(component["y"])
                cw = int(component["w"])
                ch = int(component["h"])
                # Reject a far-right isolated image when there is a much larger,
                # separate component to its left with a clear whitespace gap.
                separate_right_image = False
                for other in components:
                    if other is component:
                        continue
                    ox = int(other["x"])
                    oy = int(other["y"])
                    ow = int(other["w"])
                    oh = int(other["h"])
                    y_overlap = max(0, min(cy + ch, oy + oh) - max(cy, oy))
                    if y_overlap < int(min(ch, oh) * 0.45):
                        continue
                    if ox + ow + gap_threshold <= cx:
                        if float(other["area"]) >= float(component["area"]) * 1.4 and cx >= int(roi_w * 0.55):
                            separate_right_image = True
                            break
                if not separate_right_image:
                    valid_components.append(component)

            if valid_components:
                target_center_x = int(search_box[0] + ((search_box[2] - search_box[0]) * 0.42))
                best_component: Optional[Dict[str, Any]] = None
                best_score = 10**9
                for component in valid_components:
                    ax = int(sx1 + component["x"])
                    ay = int(sy1 + component["y"])
                    bx = int(ax + component["w"])
                    by = min(int(sy1 + component["y"] + component["h"]), int(step_y) - 5)
                    if bx <= ax or by <= ay:
                        continue
                    score = (
                        abs(int(step_y) - by)
                        + abs(int(component["center_x"]) - target_center_x) // 5
                        - min(int(component["w"]), 260) // 6
                    )
                    if score < best_score:
                        best_score = score
                        best_component = {
                            "x": ax,
                            "y": ay,
                            "w": bx - ax,
                            "h": by - ay,
                        }
                if best_component is not None:
                    return [
                        int(best_component["x"]),
                        int(best_component["y"]),
                        int(best_component["w"]),
                        int(best_component["h"]),
                    ]
        except Exception:
            pass

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


def _quantized_bgr_keys(img: Any, bin_size: int = 24) -> Any:
    arr = np.asarray(img, dtype=np.uint16)
    quantized = np.minimum(arr // max(1, int(bin_size)), 255)
    return (
        (quantized[:, :, 0].astype(np.uint32) << 16)
        | (quantized[:, :, 1].astype(np.uint32) << 8)
        | quantized[:, :, 2].astype(np.uint32)
    )


def _quantized_color_counts(img: Any, bin_size: int = 24) -> Dict[int, int]:
    try:
        if img is None or getattr(img, "size", 0) == 0:
            return {}
        keys = _quantized_bgr_keys(img, bin_size=bin_size).reshape(-1)
        values, counts = np.unique(keys, return_counts=True)
        return {int(value): int(count) for value, count in zip(values.tolist(), counts.tolist())}
    except Exception:
        return {}


def _dominant_color_key_and_pct(counts: Dict[int, int], total: int) -> Tuple[Optional[int], float]:
    if not counts or total <= 0:
        return None, 0.0
    key, count = max(counts.items(), key=lambda item: int(item[1]))
    return int(key), float(count) / float(max(1, total))


def _page_background_colour_stats(
    img: Any,
    *,
    set_num: Optional[str] = None,
    page: Optional[int] = None,
    rebuild: bool = False,
) -> Dict[str, Any]:
    cache_entry: Optional[Dict[str, Any]] = None
    if set_num is not None and page is not None:
        cache_entry = _page_callout_cache_entry(str(set_num), int(page), rebuild=rebuild)
        cached = cache_entry.get("page_background_colour_stats")
        if isinstance(cached, dict):
            return cached
    page_height, page_width = img.shape[:2]
    page_total = int(page_width) * int(page_height)
    page_counts = _quantized_color_counts(img)
    main_key, main_pct = _dominant_color_key_and_pct(page_counts, page_total)
    stats = {
        "page_counts": page_counts,
        "page_total": page_total,
        "main_page_key": main_key,
        "main_page_pct": main_pct,
    }
    if cache_entry is not None:
        cache_entry["page_background_colour_stats"] = stats
    return stats


def _panel_colour_contrast_stats(
    crop_img: Any,
    page_counts: Dict[int, int],
    page_total: int,
    main_page_key: Optional[int],
    main_page_pct: float,
    *,
    bin_size: int = 24,
) -> Dict[str, float]:
    try:
        if crop_img is None or getattr(crop_img, "size", 0) == 0:
            return {"ok": 0.0, "local_pct": 0.0, "page_pct": 1.0}
        counts = _quantized_color_counts(crop_img, bin_size=bin_size)
        crop_total = int(crop_img.shape[0]) * int(crop_img.shape[1])
        local_key, local_pct = _dominant_color_key_and_pct(counts, crop_total)
        if local_key is None:
            return {"ok": 0.0, "local_pct": 0.0, "page_pct": 1.0}
        page_pct = float(page_counts.get(int(local_key), 0)) / float(max(1, page_total))
        ok = (
            int(local_key) != int(main_page_key or -1)
            and float(local_pct) >= 0.24
            and float(page_pct) < max(float(main_page_pct) * 0.82, 0.035)
        )
        return {"ok": 1.0 if ok else 0.0, "local_pct": float(local_pct), "page_pct": float(page_pct)}
    except Exception:
        return {"ok": 0.0, "local_pct": 0.0, "page_pct": 1.0}


def _page_panel_colour_mask(
    img: Any,
    page_counts: Dict[int, int],
    page_total: int,
    main_page_key: Optional[int],
    main_page_pct: float,
    *,
    bin_size: int = 24,
) -> Any:
    keys = _quantized_bgr_keys(img, bin_size=bin_size)
    allowed_keys = {
        int(key)
        for key, count in page_counts.items()
        if int(key) != int(main_page_key or -1)
        and (float(count) / float(max(1, page_total))) < max(float(main_page_pct) * 0.82, 0.035)
        and count >= max(120, int(page_total * 0.00025))
    }
    if not allowed_keys:
        return np.zeros(keys.shape, dtype=np.uint8)
    mask = np.isin(keys, np.array(sorted(allowed_keys), dtype=np.uint32)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    return mask


def _pale_blue_callout_ratio_bgr(crop_img: Any) -> float:
    try:
        if crop_img is None or getattr(crop_img, "size", 0) == 0:
            return 0.0
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        bgr = crop_img
        mask = (
            (
                (hsv[:, :, 0] >= 80)
                & (hsv[:, :, 0] <= 132)
                & (hsv[:, :, 1] <= 150)
                & (hsv[:, :, 2] >= 135)
            )
            | (
                (bgr[:, :, 0] >= 150)
                & (bgr[:, :, 1] >= 150)
                & (bgr[:, :, 2] >= 130)
                & ((bgr[:, :, 0].astype(np.int16) - bgr[:, :, 2].astype(np.int16)) >= 8)
            )
        )
        return float(mask.mean())
    except Exception:
        return 0.0


def _box_contains_box(outer: List[int], inner: List[int], pad: int = 0) -> bool:
    ox, oy, ow, oh = [int(value) for value in outer]
    ix, iy, iw, ih = [int(value) for value in inner]
    return (
        ox - int(pad) <= ix
        and oy - int(pad) <= iy
        and ox + ow + int(pad) >= ix + iw
        and oy + oh + int(pad) >= iy + ih
    )


def _repair_callout_box_candidate_crop(
    img: Any,
    candidate: Dict[str, Any],
    *,
    page_width: int,
    page_height: int,
) -> Optional[Dict[str, Any]]:
    """Conservatively expand a narrow fallback candidate to the full callout.

    This is only for material-pipeline `callout_box_candidate` rows. It never
    creates a replacement unless the larger box still contains the original
    candidate and known qty token boxes.
    """
    try:
        if str(candidate.get("candidate_origin") or "") != "callout_box_candidate":
            return None
        if str(candidate.get("source") or "").strip().startswith("edge_detect"):
            return None
        crop_box = _coerce_box_list(candidate.get("coords_xywh"))
        if crop_box is None:
            return None
        x, y, w, h = [int(value) for value in crop_box]
        if w <= 0 or h <= 0:
            return None
        aspect = w / float(max(1, h))
        if w >= 340 and h >= 120 and aspect <= 3.9:
            return None

        search_pad_left = max(80, int(round(w * 0.45)))
        search_pad_right = max(180, int(round(w * 0.95)))
        search_pad_y = max(70, int(round(h * 0.70)))
        search_bounds = _safe_crop_bounds(
            x - search_pad_left,
            y - search_pad_y,
            x + w + search_pad_right,
            y + h + search_pad_y,
            page_width,
            page_height,
        )
        if search_bounds is None:
            return None
        sx1, sy1, sx2, sy2 = search_bounds
        roi = img[sy1:sy2, sx1:sx2]
        if roi is None or roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bgr = roi
        pale_mask = (
            (
                (hsv[:, :, 0] >= 80)
                & (hsv[:, :, 0] <= 132)
                & (hsv[:, :, 1] <= 155)
                & (hsv[:, :, 2] >= 130)
            )
            | (
                (bgr[:, :, 0] >= 145)
                & (bgr[:, :, 1] >= 145)
                & (bgr[:, :, 2] >= 120)
                & ((bgr[:, :, 0].astype(np.int16) - bgr[:, :, 2].astype(np.int16)) >= 6)
            )
        ).astype(np.uint8) * 255
        pale_mask = cv2.morphologyEx(pale_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
        pale_mask = cv2.morphologyEx(pale_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(pale_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        token_boxes = [
            dict(item)
            for item in list(candidate.get("qty_token_boxes") or [])
            if isinstance(item, dict)
        ]
        token_page_boxes = []
        for token in token_boxes:
            tx = int(token.get("x", 0) or 0)
            ty = int(token.get("y", 0) or 0)
            tw = max(1, int(token.get("w", 0) or 0))
            th = max(1, int(token.get("h", 0) or 0))
            token_page_boxes.append([x + tx, y + ty, tw, th])

        best_box: Optional[List[int]] = None
        best_score = -1.0
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            if cw <= 0 or ch <= 0:
                continue
            # Include the dark rounded border around the pale-blue interior.
            pad = 4
            repaired = [
                max(0, sx1 + cx - pad),
                max(0, sy1 + cy - pad),
                min(page_width, sx1 + cx + cw + pad) - max(0, sx1 + cx - pad),
                min(page_height, sy1 + cy + ch + pad) - max(0, sy1 + cy - pad),
            ]
            rx, ry, rw, rh = [int(value) for value in repaired]
            if rw <= w or rh < h:
                continue
            if rw < 160 or rh < 55:
                continue
            if not _box_contains_box(repaired, crop_box, pad=2):
                continue
            if any(not _box_contains_box(repaired, token_box, pad=2) for token_box in token_page_boxes):
                continue
            repaired_crop = img[ry : ry + rh, rx : rx + rw]
            if repaired_crop is None or repaired_crop.size == 0:
                continue
            if _yellow_ratio_bgr(repaired_crop) > 0.18:
                continue
            pale_ratio = _pale_blue_callout_ratio_bgr(repaired_crop)
            if pale_ratio < 0.38:
                continue
            gray = cv2.cvtColor(repaired_crop, cv2.COLOR_BGR2GRAY)
            dark = gray < 100
            band = max(2, min(7, min(rw, rh) // 14))
            border_dark = (
                int(dark[:band, :].sum())
                + int(dark[rh - band :, :].sum())
                + int(dark[:, :band].sum())
                + int(dark[:, rw - band :].sum())
            )
            if border_dark < max(24, int((rw + rh) * 0.12)):
                continue
            score = float(rw * rh) + (pale_ratio * 10000.0) + float(border_dark)
            if score > best_score:
                best_score = score
                best_box = repaired

        if best_box is None:
            return None
        repaired_candidate = dict(candidate)
        repaired_candidate["coords_xywh"] = [int(value) for value in best_box]
        repaired_candidate["coords_label"] = "callout_box_candidate repaired"
        repaired_candidate["source"] = "callout_box_candidate_repaired"
        repaired_candidate["edge_rect"] = [int(value) for value in best_box]
        return repaired_candidate
    except Exception:
        return None


def _detect_page_step_number_boxes(
    img: Any,
    step_boxes: List[Dict[str, Any]],
    *,
    page_width: int,
    page_height: int,
    set_num: Optional[str] = None,
    page: Optional[int] = None,
    rebuild: bool = False,
) -> List[Dict[str, Any]]:
    cache_entry: Optional[Dict[str, Any]] = None
    if set_num is not None and page is not None:
        cache_entry = _page_callout_cache_entry(str(set_num), int(page), rebuild=rebuild)
        cached_boxes = cache_entry.get("detected_step_number_boxes")
        if not rebuild and isinstance(cached_boxes, list):
            return [dict(item) for item in cached_boxes if isinstance(item, dict)]

    detected: List[Dict[str, Any]] = []
    for step_box in step_boxes or []:
        try:
            value = int(step_box.get("step_number", 0) or 0)
            x = int(step_box.get("x", 0) or 0)
            y = int(step_box.get("y", 0) or 0)
            w = int(step_box.get("w", 0) or 0)
            h = int(step_box.get("h", 0) or 0)
        except Exception:
            continue
        if value <= 0 or w <= 0 or h <= 0:
            continue
        detected.append({"step_number": value, "x": x, "y": y, "w": w, "h": h, "source": step_box.get("source") or "step_detector"})

    if not detected:
        try:
            import pytesseract

            data = pytesseract.image_to_data(
                img,
                config="--psm 11 -c tessedit_char_whitelist=0123456789",
                output_type=pytesseract.Output.DICT,
            )
            ocr_tokens: List[Dict[str, Any]] = []
            for idx in range(len(data.get("text", []) or [])):
                text = re.sub(r"\D+", "", str((data.get("text", [""])[idx] or "")).strip())
                if not text:
                    continue
                value = int(text)
                if value <= 0 or value >= 1000:
                    continue
                try:
                    conf = float((data.get("conf", ["-1"])[idx] or -1))
                except Exception:
                    conf = -1.0
                if conf < 25:
                    continue
                x = int(data.get("left", [0])[idx] or 0)
                y = int(data.get("top", [0])[idx] or 0)
                w = int(data.get("width", [0])[idx] or 0)
                h = int(data.get("height", [0])[idx] or 0)
                ocr_tokens.append({"text": text, "x": x, "y": y, "w": w, "h": h, "conf": conf})
                if w < 8 or h < 16 or w > int(page_width * 0.16) or h > int(page_height * 0.16):
                    continue
                detected.append({"step_number": value, "x": x, "y": y, "w": w, "h": h, "source": "page_ocr"})
            if cache_entry is not None:
                cache_entry["ocr_tokens"] = ocr_tokens
        except Exception:
            pass

    deduped: List[Dict[str, Any]] = []
    for item in sorted(detected, key=lambda row: (int(row.get("step_number", 0) or 0), int(row.get("y", 0) or 0), int(row.get("x", 0) or 0))):
        ix = int(item.get("x", 0) or 0)
        iy = int(item.get("y", 0) or 0)
        iw = int(item.get("w", 0) or 0)
        ih = int(item.get("h", 0) or 0)
        value = int(item.get("step_number", 0) or 0)
        duplicate = False
        for existing in deduped:
            if int(existing.get("step_number", 0) or 0) != value:
                continue
            ex = int(existing.get("x", 0) or 0)
            ey = int(existing.get("y", 0) or 0)
            if abs((ix + iw // 2) - (ex + int(existing.get("w", 0) or 0) // 2)) <= 28 and abs((iy + ih // 2) - (ey + int(existing.get("h", 0) or 0) // 2)) <= 28:
                duplicate = True
                break
        if not duplicate:
            deduped.append(item)
    if cache_entry is not None:
        cache_entry["detected_step_number_boxes"] = [dict(item) for item in deduped]
    return deduped


def _detect_step_number_below_panel(
    img: Any,
    panel_box: List[int],
    *,
    page_width: int,
    page_height: int,
) -> List[Dict[str, Any]]:
    try:
        import pytesseract

        px, py, pw, ph = [int(value) for value in panel_box]
        bounds = _safe_crop_bounds(
            px - max(130, int(pw * 0.50)),
            py + ph - 8,
            px + max(170, int(pw * 0.45)),
            py + ph + max(170, int(ph * 1.20)),
            page_width,
            page_height,
        )
        if bounds is None:
            return []
        sx1, sy1, sx2, sy2 = bounds
        roi = img[sy1:sy2, sx1:sx2]
        if roi is None or roi.size == 0:
            return []
        original_roi = roi
        roi = cv2.resize(original_roi, None, fx=2.8, fy=2.8, interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(
            roi,
            config="--psm 11 -c tessedit_char_whitelist=0123456789",
            output_type=pytesseract.Output.DICT,
        )
        steps: List[Dict[str, Any]] = []
        ocr_tokens: List[Dict[str, Any]] = []
        scale = 2.8
        for idx in range(len(data.get("text", []) or [])):
            text = re.sub(r"\D+", "", str((data.get("text", [""])[idx] or "")).strip())
            if not text:
                continue
            value = int(text)
            if value <= 0 or value >= 1000:
                continue
            try:
                conf = float((data.get("conf", ["-1"])[idx] or -1))
            except Exception:
                conf = -1.0
            if conf < 15:
                continue
            x = sx1 + int(float(data.get("left", [0])[idx] or 0) / scale)
            y = sy1 + int(float(data.get("top", [0])[idx] or 0) / scale)
            w = max(1, int(float(data.get("width", [0])[idx] or 0) / scale))
            h = max(1, int(float(data.get("height", [0])[idx] or 0) / scale))
            ocr_tokens.append({"text": text, "x": x, "y": y, "w": w, "h": h, "conf": conf, "source": "panel_below_ocr"})
            if w < 7 or h < 14:
                continue
            steps.append({"step_number": value, "x": x, "y": y, "w": w, "h": h, "source": "panel_below_ocr"})
        if steps:
            return [{"_ocr_tokens": ocr_tokens}, *steps]

        gray = cv2.cvtColor(original_roi, cv2.COLOR_BGR2GRAY)
        dark = (gray < 70).astype(np.uint8) * 255
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 8 or h < 28 or w > int((sx2 - sx1) * 0.30) or h > int((sy2 - sy1) * 0.60):
                continue
            pad = 8
            bounds = _safe_crop_bounds(x - pad, y - pad, x + w + pad, y + h + pad, sx2 - sx1, sy2 - sy1)
            if bounds is None:
                continue
            x1, y1, x2, y2 = bounds
            crop = original_roi[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue
            crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(
                crop,
                config="--psm 10 -c tessedit_char_whitelist=0123456789",
            )
            text = re.sub(r"\D+", "", str(text or ""))
            if not text:
                continue
            value = int(text)
            if value <= 0 or value >= 1000:
                continue
            token = {"text": text, "x": sx1 + x, "y": sy1 + y, "w": w, "h": h, "conf": None, "source": "panel_below_component_ocr"}
            ocr_tokens.append(token)
            steps.append({"step_number": value, "x": sx1 + x, "y": sy1 + y, "w": w, "h": h, "source": "panel_below_component_ocr"})
        return ([{"_ocr_tokens": ocr_tokens}] if ocr_tokens else []) + steps
    except Exception:
        return []


def _callout_panel_has_boundary(crop_img: Any) -> bool:
    try:
        if crop_img is None or getattr(crop_img, "size", 0) == 0:
            return False
        h, w = crop_img.shape[:2]
        if h <= 0 or w <= 0:
            return False
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        dark = gray < 95
        band = max(2, min(8, min(w, h) // 12))
        border_dark = (
            int(dark[:band, :].sum())
            + int(dark[h - band :, :].sum())
            + int(dark[:, :band].sum())
            + int(dark[:, w - band :].sum())
        )
        if border_dark >= max(28, int((w + h) * 0.13)):
            return True
        edges = cv2.Canny(gray, 60, 150)
        edge_band = (
            int(edges[:band, :].sum() // 255)
            + int(edges[h - band :, :].sum() // 255)
            + int(edges[:, :band].sum() // 255)
            + int(edges[:, w - band :].sum() // 255)
        )
        return edge_band >= max(34, int((w + h) * 0.20))
    except Exception:
        return False


def _dark_line_group_centers(values: Any, threshold: int) -> List[int]:
    centers: List[int] = []
    start: Optional[int] = None
    for idx, value in enumerate(list(values)):
        if int(value) >= int(threshold):
            if start is None:
                start = int(idx)
        elif start is not None:
            centers.append((start + int(idx) - 1) // 2)
            start = None
    if start is not None:
        centers.append((start + len(values) - 1) // 2)
    return centers


def _expand_panel_box_to_dark_boundary(
    img: Any,
    box: List[int],
    *,
    page_width: int,
    page_height: int,
) -> List[int]:
    try:
        x, y, w, h = [int(value) for value in box]
        bounds = _safe_crop_bounds(
            x - max(90, int(w * 0.75)),
            y - max(35, int(h * 0.35)),
            x + w + max(45, int(w * 0.18)),
            y + h + max(70, int(h * 0.70)),
            page_width,
            page_height,
        )
        if bounds is None:
            return box
        sx1, sy1, sx2, sy2 = bounds
        roi = img[sy1:sy2, sx1:sx2]
        if roi is None or roi.size == 0:
            return box
        roi_h, roi_w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark = (gray < 100).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        col_counts = dark.sum(axis=0)
        row_counts = dark.sum(axis=1)
        vertical_threshold = max(10, int(roi_h * 0.12))
        horizontal_threshold = max(18, int(roi_w * 0.16))
        col_centers = _dark_line_group_centers(col_counts, vertical_threshold)
        row_centers = _dark_line_group_centers(row_counts, horizontal_threshold)

        local_left = x - sx1
        local_right = x + w - sx1
        local_top = y - sy1
        local_bottom = y + h - sy1

        left_options = [value for value in col_centers if value <= local_left + 8]
        right_options = [value for value in col_centers if value >= local_right - 8]
        top_options = [value for value in row_centers if value <= local_top + 8]
        bottom_options = [value for value in row_centers if value >= local_bottom - 8]
        if not left_options or not right_options or not top_options or not bottom_options:
            return box

        left = max(left_options)
        right = min(right_options)
        top = max(top_options)
        bottom = min(bottom_options)
        pad = 3
        expanded = [
            max(0, sx1 + left - pad),
            max(0, sy1 + top - pad),
            min(page_width, sx1 + right + pad) - max(0, sx1 + left - pad),
            min(page_height, sy1 + bottom + pad) - max(0, sy1 + top - pad),
        ]
        ex, ey, ew, eh = [int(value) for value in expanded]
        if ew <= w or eh < max(35, int(h * 0.65)):
            return box
        if ew > int(page_width * 0.60) or eh > int(page_height * 0.35):
            return box
        if not _box_contains_box(expanded, box, pad=2):
            return box
        return expanded
    except Exception:
        return box


def _refine_page_level_panel_with_step_geometry(
    img: Any,
    panel_box: List[int],
    *,
    step_y: int,
    page_width: int,
    page_height: int,
) -> Optional[List[int]]:
    try:
        px, py, pw, ph = [int(value) for value in panel_box]
        bounds = _safe_crop_bounds(
            px - max(45, int(pw * 0.20)),
            py - max(28, int(ph * 0.25)),
            px + pw + max(45, int(pw * 0.20)),
            min(page_height, int(step_y) - 1),
            page_width,
            page_height,
        )
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
            minLineLength=max(70, min(180, int(pw * 0.35))),
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
                if box_h < 38 or box_h > 240:
                    continue
                left = max(0, min(int(top["x1"]), int(bottom["x1"])) - 3)
                right = min(roi.shape[1], max(int(top["x2"]), int(bottom["x2"])) + 3)
                box_w = int(right - left)
                if box_w < 110 or box_w > 700:
                    continue
                top_y = max(0, int(top["y"]) - 3)
                bottom_y = min(roi.shape[0], int(bottom["y"]) + 3)
                ax = int(sx1 + left)
                ay = int(sy1 + top_y)
                bx = int(sx1 + right)
                by = int(sy1 + bottom_y)
                if by > int(step_y) - 5:
                    by = int(step_y) - 5
                if bx <= ax or by <= ay:
                    continue
                candidate = [ax, ay, bx - ax, by - ay]
                if not _box_contains_box(candidate, panel_box, pad=14):
                    continue
                crop = img[ay:by, ax:bx]
                if crop is None or crop.size == 0:
                    continue
                if _yellow_ratio_bgr(crop) > 0.18:
                    continue
                score = abs(int(step_y) - by) + abs((ax + bx) // 2 - (px + pw // 2)) // 5
                if score < best_score:
                    best_score = score
                    best = candidate
        return best
    except Exception:
        return None


def _detect_page_level_callout_panels(
    img: Any,
    *,
    page_width: int,
    page_height: int,
    set_num: Optional[str] = None,
    page: Optional[int] = None,
    rebuild: bool = False,
) -> List[Dict[str, Any]]:
    try:
        stats = _page_background_colour_stats(img, set_num=set_num, page=page, rebuild=rebuild)
        page_total = int(stats.get("page_total") or (int(page_width) * int(page_height)))
        page_counts = dict(stats.get("page_counts") or {})
        main_key = stats.get("main_page_key")
        main_pct = float(stats.get("main_page_pct") or 0.0)
        mask = _page_panel_colour_mask(img, page_counts, page_total, main_key, main_pct)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        panels: List[Dict[str, Any]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 95 or h < 45:
                continue
            if w > int(page_width * 0.55) or h > int(page_height * 0.28):
                continue
            if w * h > int(page_total * 0.16):
                continue
            aspect = w / float(max(1, h))
            if aspect < 1.15 or aspect > 7.0:
                continue
            pad = 5
            bounds = _safe_crop_bounds(x - pad, y - pad, x + w + pad, y + h + pad, page_width, page_height)
            if bounds is None:
                continue
            x1, y1, x2, y2 = bounds
            expanded_box = _expand_panel_box_to_dark_boundary(
                img,
                [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                page_width=page_width,
                page_height=page_height,
            )
            x1, y1, ew, eh = [int(value) for value in expanded_box]
            x2 = min(page_width, x1 + ew)
            y2 = min(page_height, y1 + eh)
            crop = img[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue
            if _yellow_ratio_bgr(crop) > 0.18:
                continue
            colour_stats = _panel_colour_contrast_stats(crop, page_counts, page_total, main_key, main_pct)
            if float(colour_stats.get("ok", 0.0)) <= 0.0:
                continue
            if not _callout_panel_has_boundary(crop):
                continue
            qty_payload = _auto_qty_payload_for_crop(crop, 0)
            qty_tokens = list(qty_payload.get("qty_token_boxes") or [])
            if not qty_tokens:
                continue
            panels.append(
                {
                    "coords_xywh": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "detected_qty_text": qty_payload.get("detected_qty_text", []),
                    "detected_qty_numbers": qty_payload.get("detected_qty_numbers", []),
                    "qty_token_boxes": qty_tokens,
                    "panel_colour_local_pct": colour_stats.get("local_pct"),
                    "panel_colour_page_pct": colour_stats.get("page_pct"),
                }
            )

        deduped: List[Dict[str, Any]] = []
        for panel in sorted(panels, key=lambda item: (int(item["coords_xywh"][1]), int(item["coords_xywh"][0]), -(int(item["coords_xywh"][2]) * int(item["coords_xywh"][3])))):
            box = panel["coords_xywh"]
            duplicate = False
            for existing in deduped:
                existing_box = existing["coords_xywh"]
                if _box_contains_box(existing_box, box, pad=10) or _box_contains_box(box, existing_box, pad=10):
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(panel)
        return deduped
    except Exception:
        return []


def _page_level_callout_candidates_for_fallback(
    img: Any,
    *,
    page_width: int,
    page_height: int,
    step_boxes: List[Dict[str, Any]],
    set_num: Optional[str] = None,
    page: Optional[int] = None,
    rebuild: bool = False,
) -> List[Dict[str, Any]]:
    panels = _detect_page_level_callout_panels(
        img,
        page_width=page_width,
        page_height=page_height,
        set_num=set_num,
        page=page,
        rebuild=rebuild,
    )
    if not panels:
        return []
    steps = _detect_page_step_number_boxes(
        img,
        step_boxes,
        page_width=page_width,
        page_height=page_height,
        set_num=set_num,
        page=page,
        rebuild=False,
    )
    if not steps:
        cache_entry = _page_callout_cache_entry(str(set_num), int(page), rebuild=False) if set_num is not None and page is not None else None
        cached_panel_steps = cache_entry.get("panel_step_number_boxes") if cache_entry is not None else None
        used_cached_panel_steps = False
        if not rebuild and isinstance(cached_panel_steps, list):
            steps = [dict(item) for item in cached_panel_steps if isinstance(item, dict)]
            used_cached_panel_steps = True
        panel_ocr_tokens: List[Dict[str, Any]] = []
        if not used_cached_panel_steps:
            for panel in panels:
                panel_box = _coerce_box_list(panel.get("coords_xywh"))
                if panel_box is None:
                    continue
                detected_rows = _detect_step_number_below_panel(
                    img,
                    panel_box,
                    page_width=page_width,
                    page_height=page_height,
                )
                for row in detected_rows:
                    if not isinstance(row, dict):
                        continue
                    if isinstance(row.get("_ocr_tokens"), list):
                        panel_ocr_tokens.extend([dict(item) for item in row.get("_ocr_tokens", []) if isinstance(item, dict)])
                        continue
                    steps.append(row)
            if cache_entry is not None:
                cache_entry["panel_step_number_boxes"] = [dict(item) for item in steps]
                existing_tokens = list(cache_entry.get("ocr_tokens") or [])
                cache_entry["ocr_tokens"] = existing_tokens + panel_ocr_tokens
    if not steps:
        return []

    assigned: Dict[int, Dict[str, Any]] = {}
    for panel in panels:
        px, py, pw, ph = [int(value) for value in panel.get("coords_xywh", [])]
        panel_bottom = py + ph
        best_step: Optional[Dict[str, Any]] = None
        best_score: Optional[float] = None
        for step in steps:
            sx = int(step.get("x", 0) or 0)
            sy = int(step.get("y", 0) or 0)
            sw = int(step.get("w", 0) or 0)
            sh = int(step.get("h", 0) or 0)
            step_left = sx
            step_mid_y = sy + sh // 2
            if step_mid_y < panel_bottom - 4:
                continue
            if step_left > px + max(85, int(pw * 0.20)):
                continue
            horizontal_gap = abs(step_left - px)
            if horizontal_gap > max(180, int(pw * 0.55)):
                continue
            vertical_gap = max(0, step_mid_y - panel_bottom)
            score = float(vertical_gap) + float(horizontal_gap) * 0.45
            if best_score is None or score < best_score:
                best_score = score
                best_step = step
        if best_step is None or best_score is None:
            continue
        step_number = int(best_step.get("step_number", 0) or 0)
        if step_number <= 0:
            continue
        refined_box = _refine_page_level_panel_with_step_geometry(
            img,
            [px, py, pw, ph],
            step_y=int(best_step.get("y", 0) or 0),
            page_width=page_width,
            page_height=page_height,
        )
        if refined_box is not None:
            px, py, pw, ph = [int(value) for value in refined_box]
        current = assigned.get(step_number)
        if current is not None and float(current.get("_assignment_score", 999999.0)) <= best_score:
            continue
        crop_img = img[py : py + ph, px : px + pw]
        if crop_img is None or crop_img.size == 0:
            continue
        try:
            data_uri = _encode_debug_image_data_uri(crop_img, max_width=420)
        except Exception:
            continue
        qty_payload = _qty_payload_for_page_level_callout_crop(crop_img, step_number, "page_level_callout_assignment")
        candidate = {
            "candidate_origin": "callout_box_candidate",
            "source": "page_level_callout_assignment",
            "match_enabled": True,
            "data_uri": data_uri,
            "coords_xywh": [px, py, pw, ph],
            "coords_label": "page-level assigned callout",
            "edge_rect": [px, py, pw, ph],
            "confidence": 0.44,
            "step_number": step_number,
            "qty_source": qty_payload.get("qty_source") or "page_level_callout_assignment",
            "detected_qty_text": qty_payload.get("detected_qty_text", []),
            "detected_qty_numbers": qty_payload.get("detected_qty_numbers", []),
            "qty_token_boxes": qty_payload.get("qty_token_boxes"),
            "qty_ocr_source_regions": qty_payload.get("qty_ocr_source_regions", []),
            "qty_ocr_ordered_qty_list": qty_payload.get("qty_ocr_ordered_qty_list", []),
            "_assignment_score": float(best_score),
        }
        assigned[step_number] = candidate

    cleaned: List[Dict[str, Any]] = []
    for candidate in sorted(assigned.values(), key=lambda item: (int(item.get("step_number", 0) or 0), int(item.get("coords_xywh", [0, 0])[1]), int(item.get("coords_xywh", [0, 0])[0]))):
        item = dict(candidate)
        item.pop("_assignment_score", None)
        cleaned.append(item)
    return cleaned


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


def _normalize_qty_token_text(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").lower())
    if re.match(r"^\d+x$", text) or re.match(r"^x\d+$", text):
        return text
    return ""


def _token_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1 = int(a.get("x", 0) or 0)
    ay1 = int(a.get("y", 0) or 0)
    ax2 = ax1 + max(0, int(a.get("w", 0) or 0))
    ay2 = ay1 + max(0, int(a.get("h", 0) or 0))
    bx1 = int(b.get("x", 0) or 0)
    by1 = int(b.get("y", 0) or 0)
    bx2 = bx1 + max(0, int(b.get("w", 0) or 0))
    by2 = by1 + max(0, int(b.get("h", 0) or 0))
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    a_area = max(1, max(0, ax2 - ax1) * max(0, ay2 - ay1))
    b_area = max(1, max(0, bx2 - bx1) * max(0, by2 - by1))
    return float(inter_area) / float(max(1, min(a_area, b_area)))


def _tokens_are_same_qty_label(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if _normalize_qty_token_text(a.get("text")) != _normalize_qty_token_text(b.get("text")):
        return False
    if _token_overlap_ratio(a, b) >= 0.55:
        return True
    return (
        abs(int(a.get("cx", 0) or 0) - int(b.get("cx", 0) or 0)) <= 14
        and abs(int(a.get("cy", 0) or 0) - int(b.get("cy", 0) or 0)) <= 10
    )


def _dedupe_qty_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for token in sorted(
        list(tokens or []),
        key=lambda item: (
            int(item.get("cy", 0) or 0),
            int(item.get("cx", 0) or 0),
            int(item.get("w", 0) or 0) * int(item.get("h", 0) or 0),
        ),
    ):
        normalized = _normalize_qty_token_text(token.get("text"))
        if not normalized:
            continue
        normalized_token = {
            "text": normalized,
            "x": int(token.get("x", 0) or 0),
            "y": int(token.get("y", 0) or 0),
            "w": int(token.get("w", 0) or 0),
            "h": int(token.get("h", 0) or 0),
            "cx": int(token.get("cx", 0) or 0),
            "cy": int(token.get("cy", 0) or 0),
        }
        if any(_tokens_are_same_qty_label(existing, normalized_token) for existing in deduped):
            continue
        deduped.append(normalized_token)
    return deduped


def _dedupe_qty_tokens_high_overlap_only(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for token in sorted(
        list(tokens or []),
        key=lambda item: (
            int(item.get("cy", 0) or 0),
            int(item.get("x", 0) or 0),
            int(item.get("w", 0) or 0) * int(item.get("h", 0) or 0),
        ),
    ):
        normalized = _normalize_qty_token_text(token.get("text"))
        if not normalized:
            continue
        normalized_token = {
            "text": normalized,
            "x": int(token.get("x", 0) or 0),
            "y": int(token.get("y", 0) or 0),
            "w": int(token.get("w", 0) or 0),
            "h": int(token.get("h", 0) or 0),
            "cx": int(token.get("cx", 0) or 0),
            "cy": int(token.get("cy", 0) or 0),
        }
        if "source_region" in token:
            normalized_token["source_region"] = str(token.get("source_region") or "")
        if "confidence" in token:
            normalized_token["confidence"] = token.get("confidence")
        if any(
            _normalize_qty_token_text(existing.get("text")) == normalized
            and _token_overlap_ratio(existing, normalized_token) >= 0.75
            for existing in deduped
        ):
            continue
        deduped.append(normalized_token)
    return deduped


def _final_crop_qty_token_is_valid(token: Dict[str, Any], crop_width: int, crop_height: int) -> bool:
    try:
        x = int(token.get("x", 0) or 0)
        y = int(token.get("y", 0) or 0)
        w = int(token.get("w", 0) or 0)
        h = int(token.get("h", 0) or 0)
    except Exception:
        return False
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0 or x + w > int(crop_width) or y + h > int(crop_height):
        return False
    if y < max(4, int(crop_height * 0.14)):
        return False
    if w > max(48, int(crop_width * 0.16)) or h > max(30, int(crop_height * 0.20)):
        return False
    return True


def _order_qty_tokens_by_rows(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[List[Dict[str, Any]]] = []
    for token in sorted(list(tokens or []), key=lambda item: (int(item.get("cy", 0) or 0), int(item.get("x", 0) or 0))):
        cy = int(token.get("cy", 0) or 0)
        placed = False
        for row in rows:
            row_cy = int(round(sum(int(item.get("cy", 0) or 0) for item in row) / max(1, len(row))))
            row_h = max(int(item.get("h", 0) or 0) for item in row)
            token_h = int(token.get("h", 0) or 0)
            if abs(cy - row_cy) <= max(18, int(max(row_h, token_h) * 1.8)):
                row.append(token)
                placed = True
                break
        if not placed:
            rows.append([token])

    ordered: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda items: sum(int(item.get("cy", 0) or 0) for item in items) / max(1, len(items))):
        ordered.extend(sorted(row, key=lambda item: int(item.get("x", 0) or 0)))
    return ordered


def _qty_payload_for_page_level_callout_crop(crop_img: Any, step_number: int, source_label: str = "page_level_callout_assignment") -> Dict[str, Any]:
    payload = _auto_qty_payload_for_crop(crop_img, step_number)
    debug_regions: List[Dict[str, Any]] = []
    if crop_img is None or getattr(crop_img, "size", 0) == 0:
        payload["qty_ocr_source_regions"] = debug_regions
        payload["qty_ocr_ordered_qty_list"] = list(payload.get("detected_qty_text", []) or [])
        return payload

    height, width = crop_img.shape[:2]
    region_specs = [
        ("final_crop_full", 0, 0, width, height),
        ("final_crop_up_right", 0, 0, width, min(height, max(1, int(height * 0.72)))),
        ("final_crop_lower_expanded_up_right", 0, max(0, int(height * 0.38)), width, height),
        ("final_crop_right_expanded", max(0, int(width * 0.22)), 0, width, height),
    ]

    tokens: List[Dict[str, Any]] = []
    for name, x1, y1, x2, y2 in region_specs:
        if x2 <= x1 or y2 <= y1:
            continue
        region_img = crop_img[y1:y2, x1:x2]
        if region_img is None or region_img.size == 0:
            continue
        debug_regions.append({"name": name, "x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)})
        for token in _extract_qty_tokens_from_image(region_img) or []:
            normalized = _normalize_qty_token_text(token.get("text"))
            if not normalized:
                continue
            tx = int(token.get("x", 0) or 0) + int(x1)
            ty = int(token.get("y", 0) or 0) + int(y1)
            tw = int(token.get("w", 0) or 0)
            th = int(token.get("h", 0) or 0)
            tokens.append(
                {
                    "text": normalized,
                    "x": tx,
                    "y": ty,
                    "w": tw,
                    "h": th,
                    "cx": tx + (tw // 2),
                    "cy": ty + (th // 2),
                    "source_region": name,
                    "confidence": None,
                }
            )
        try:
            import pytesseract

            gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
            for scale, psm_values in ((4.0, (6, 11, 12)), (5.0, (6, 11, 12))):
                enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                thresholded = cv2.adaptiveThreshold(
                    enlarged,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    6,
                )
                for psm in psm_values:
                    data = pytesseract.image_to_data(
                        thresholded,
                        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789xX",
                        output_type=pytesseract.Output.DICT,
                    )
                    for idx in range(len(data.get("text", []) or [])):
                        normalized = _normalize_qty_token_text(data.get("text", [""])[idx])
                        if not normalized:
                            continue
                        try:
                            confidence = float(data.get("conf", ["-1"])[idx] or -1)
                        except Exception:
                            confidence = -1.0
                        if confidence < 25:
                            continue
                        tw = max(1, int(float(data.get("width", [0])[idx] or 0) / scale))
                        th = max(1, int(float(data.get("height", [0])[idx] or 0) / scale))
                        if tw < 8 or tw > 42 or th < 7 or th > 30:
                            continue
                        tx = int(x1 + (float(data.get("left", [0])[idx] or 0) / scale))
                        ty = int(y1 + (float(data.get("top", [0])[idx] or 0) / scale))
                        if ty < 0 or tx < 0:
                            continue
                        tokens.append(
                            {
                                "text": normalized,
                                "x": tx,
                                "y": ty,
                                "w": tw,
                                "h": th,
                                "cx": tx + (tw // 2),
                                "cy": ty + (th // 2),
                                "source_region": f"{name}:adaptive_s{int(scale)}_psm{psm}",
                                "confidence": confidence,
                            }
                        )
        except Exception:
            pass

    filtered_tokens = [
        token
        for token in _dedupe_qty_tokens_high_overlap_only(tokens)
        if _final_crop_qty_token_is_valid(token, int(width), int(height))
    ]
    ordered_tokens = _order_qty_tokens_by_rows(
        filtered_tokens,
    )
    if ordered_tokens:
        detected_qty_text: List[str] = []
        detected_qty_numbers: List[int] = []
        for token in ordered_tokens:
            text = str(token.get("text") or "")
            number_match = re.search(r"(\d{1,2})", text)
            qty_val = int(number_match.group(1)) if number_match else None
            detected_qty_text.append(text)
            if qty_val is not None:
                detected_qty_numbers.append(qty_val)
        payload["detected_qty_text"] = detected_qty_text
        payload["detected_qty_numbers"] = detected_qty_numbers
        payload["qty_token_boxes"] = [
            token
            for token in ordered_tokens
        ]

    payload["qty_source"] = str(source_label or "page_level_callout_assignment")
    payload["qty_ocr_source_regions"] = debug_regions
    payload["qty_ocr_ordered_qty_list"] = list(payload.get("detected_qty_text", []) or [])
    print(
        f"[qty-ocr] source={source_label}",
        "step=",
        int(step_number or 0),
        "regions=",
        debug_regions,
        "tokens=",
        payload.get("qty_token_boxes") or [],
        "ordered=",
        payload.get("qty_ocr_ordered_qty_list") or [],
    )
    return payload


def _extract_detected_qty_details_from_crop(crop_img) -> Dict[str, Any]:
    payload = _debug_extract_detected_qty_details_from_crop(crop_img)
    full_crop_tokens = _dedupe_qty_tokens(_extract_qty_tokens_from_image(crop_img) or [])
    if full_crop_tokens:
        detected_qty_text: List[str] = []
        detected_qty_numbers: List[int] = []
        for token in sorted(full_crop_tokens, key=lambda item: (int(item.get("cy", 0) or 0), int(item.get("x", 0) or 0))):
            text = str(token.get("text") or "")
            number_match = re.search(r"(\d{1,2})", text)
            qty_val = int(number_match.group(1)) if number_match else None
            detected_qty_text.append(text)
            if qty_val is not None:
                detected_qty_numbers.append(qty_val)
        payload["detected_qty_text"] = detected_qty_text
        payload["detected_qty_numbers"] = detected_qty_numbers
        payload["qty_token_boxes"] = full_crop_tokens
        return payload
    if _coerce_str_list(payload.get("detected_qty_text", [])):
        return payload
    if crop_img is None or getattr(crop_img, "size", 0) == 0:
        return payload

    height, width = crop_img.shape[:2]
    region_specs = [
        ("lower_half", crop_img[max(0, height // 2) : height, :], 0, max(0, height // 2)),
        (
            "lower_right",
            crop_img[max(0, height // 3) : height, max(0, width // 2) : width],
            max(0, width // 2),
            max(0, height // 3),
        ),
        ("right_half", crop_img[:, max(0, width // 2) : width], max(0, width // 2), 0),
        ("right_band", crop_img[:, max(0, int(width * 0.65)) : width], max(0, int(width * 0.65)), 0),
    ]

    best_tokens: List[Dict[str, Any]] = []
    best_priority = len(region_specs)
    for priority, (_, region_img, offset_x, offset_y) in enumerate(region_specs):
        if region_img is None or region_img.size == 0:
            continue
        region_tokens: List[Dict[str, Any]] = []
        for token in _extract_qty_tokens_from_image(region_img) or []:
            normalized = _normalize_qty_token_text(token.get("text"))
            if not normalized:
                continue
            x = int(token.get("x", 0) or 0) + int(offset_x)
            y = int(token.get("y", 0) or 0) + int(offset_y)
            w = int(token.get("w", 0) or 0)
            h = int(token.get("h", 0) or 0)
            region_tokens.append(
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
        deduped_tokens = _dedupe_qty_tokens(region_tokens)
        if not deduped_tokens:
            continue
        if len(deduped_tokens) > len(best_tokens) or (
            len(deduped_tokens) == len(best_tokens) and priority < best_priority
        ):
            best_tokens = deduped_tokens
            best_priority = priority

    if not best_tokens:
        return payload

    detected_qty_text: List[str] = []
    detected_qty_numbers: List[int] = []
    for token in sorted(best_tokens, key=lambda item: (int(item.get("cy", 0) or 0), int(item.get("x", 0) or 0))):
        text = str(token.get("text") or "")
        number_match = re.search(r"(\d{1,2})", text)
        qty_val = int(number_match.group(1)) if number_match else None
        detected_qty_text.append(text)
        if qty_val is not None:
            detected_qty_numbers.append(qty_val)

    payload["detected_qty_text"] = detected_qty_text
    payload["detected_qty_numbers"] = detected_qty_numbers
    payload["qty_token_boxes"] = best_tokens
    return payload


def _auto_qty_payload_for_crop(crop_img: Any, step_number: int) -> Dict[str, List[Any]]:
    """Extract qty text for auto crops and filter step-number contamination.

    If OCR completely fails, create a conservative default slot per visible part.
    """
    payload = _extract_detected_qty_details_from_crop(crop_img)
    texts = _coerce_str_list(payload.get("detected_qty_text", []))
    nums = _coerce_int_list(payload.get("detected_qty_numbers", []))
    clean_texts: List[str] = []
    clean_nums: List[int] = []
    for text, num in zip(texts, nums):
        if step_number and int(num) == int(step_number):
            continue
        clean_texts.append(str(text))
        clean_nums.append(int(num))

    if clean_nums:
        return {
            "detected_qty_text": clean_texts,
            "detected_qty_numbers": clean_nums,
            "qty_token_boxes": payload.get("qty_token_boxes") or payload.get("token_boxes") or payload.get("qty_raw_tokens"),
        }

    count = _estimate_visible_part_count_from_crop(crop_img)
    if count <= 0:
        return {"detected_qty_text": [], "detected_qty_numbers": []}
    return {
        "detected_qty_text": ["1x"] * count,
        "detected_qty_numbers": [1] * count,
        "qty_token_boxes": payload.get("qty_token_boxes") or payload.get("token_boxes") or payload.get("qty_raw_tokens"),
    }


def _debug_json_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return "—"
    try:
        return json.dumps(value, ensure_ascii=True)
    except Exception:
        return str(value)


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
        client = _require_openai_vision_client_debug()
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
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1"),
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
                model=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1"),
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

    qty_labels = [str(value) for value in list(ai_result.get("qty_labels", []) or []) if str(value or "").strip()]
    qty_numbers: List[int] = []
    for label in qty_labels:
        match = re.match(r"^\s*(\d+)\s*x\s*$", str(label), flags=re.IGNORECASE)
        if match:
            qty_numbers.append(int(match.group(1)))

    return {
        "detected_qty_text": qty_labels,
        "detected_qty_numbers": qty_numbers,
        "qty_source": "openai",
        "ai_part_count": int(ai_result.get("part_count", 0) or 0),
        "ai_issues": _issue_list(ai_result),
        "ai_crop_box": ai_result.get("crop_box"),
        "ai_suggested_fix": False,
    }


def _build_qty_sequence(qty_values: Any, qty_text_values: Any) -> List[Dict[str, Any]]:
    qty_list = _coerce_int_list(qty_values)
    qty_text_list = _coerce_str_list(qty_text_values)
    sequence: List[Dict[str, Any]] = []

    if qty_text_list:
        explicit_slots: List[Dict[str, Any]] = []
        for qty_text in qty_text_list:
            for match in re.finditer(r"(\d+)\s*x", str(qty_text), flags=re.IGNORECASE):
                qty = int(match.group(1))
                explicit_slots.append({"qty": qty, "qty_text": f"{qty}x"})
        if explicit_slots:
            return explicit_slots

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
    filled_by_slot_index = {
        int(slot_index)
        for slot_index in (
            _coerce_int(part.get("selected_slot_index"))
            for part in parts
        )
        if slot_index is not None and int(slot_index) >= 0
    }
    for part in parts:
        if _coerce_int(part.get("selected_slot_index")) is not None:
            continue
        signature = _qty_slot_signature(part.get("qty"), part.get("qty_text"))
        if not signature:
            continue
        assigned_counts[signature] = assigned_counts.get(signature, 0) + 1

    consumed_counts: Dict[str, int] = {}
    filled_slots = 0
    next_slot: Optional[Dict[str, Any]] = None
    next_qty_index = len(sequence)
    for slot_index, slot in enumerate(sequence):
        if int(slot_index) in filled_by_slot_index:
            filled_slots += 1
            continue
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
        "selected_slot_index": _coerce_int(data.get("selected_slot_index")),
        "part_bbox": _coerce_box_list(data.get("part_bbox")),
        "confidence": _coerce_float(data.get("confidence")),
        "ai_snap_input_path": _coerce_str(data.get("ai_snap_input_path")),
    }


def _base_part_num_for_display_fallback(part_num: Any) -> Optional[str]:
    text = str(part_num or "").strip()
    if re.fullmatch(r".+[A-Za-z]", text) and not re.search(r"[A-Za-z]{2,}$", text):
        return text[:-1]
    return None


def _resolve_display_part_record(
    raw_parts_by_key: Dict[str, Dict[str, Any]],
    part_num: Any,
    color_id: Any,
) -> Dict[str, Any]:
    normalized_part_num = str(part_num or "").strip()
    normalized_color_id = int(color_id or 0)
    exact = dict(
        raw_parts_by_key.get(f"{normalized_part_num}::{normalized_color_id}", {}) or {}
    )
    resolved = dict(exact)
    fallback_part_num = _base_part_num_for_display_fallback(normalized_part_num)
    if not fallback_part_num:
        return resolved
    fallback = dict(
        raw_parts_by_key.get(f"{fallback_part_num}::{normalized_color_id}", {}) or {}
    )
    if not fallback:
        return resolved
    if not str(resolved.get("img_url") or "").strip() and str(fallback.get("img_url") or "").strip():
        resolved["img_url"] = str(fallback.get("img_url") or "").strip()
    if not str(resolved.get("element_id") or "").strip() and str(fallback.get("element_id") or "").strip():
        resolved["element_id"] = str(fallback.get("element_id") or "").strip()
    if (
        (
            str(resolved.get("img_url") or "").strip()
            or str(resolved.get("element_id") or "").strip()
        )
        and fallback_part_num != normalized_part_num
    ):
        resolved["display_part_num"] = fallback_part_num
    return resolved


def _prepare_instruction_parts_for_display(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw_parts = [dict(part or {}) for part in list(parts or [])]
    raw_parts_by_key = {
        f"{str(part.get('part_num') or '').strip()}::{int(part.get('color_id', 0) or 0)}": part
        for part in raw_parts
    }
    prepared: List[Dict[str, Any]] = []
    for part in raw_parts:
        part_copy = dict(part)
        resolved = _resolve_display_part_record(
            raw_parts_by_key,
            part_copy.get("part_num"),
            part_copy.get("color_id"),
        )
        if str(resolved.get("img_url") or "").strip():
            part_copy["img_url"] = str(resolved.get("img_url") or "").strip()
        if str(resolved.get("element_id") or "").strip():
            part_copy["element_id"] = str(resolved.get("element_id") or "").strip()
        if str(resolved.get("display_part_num") or "").strip():
            part_copy["display_part_num"] = str(resolved.get("display_part_num") or "").strip()
        prepared.append(part_copy)
    return prepared


def _candidate_part_key(part_num: Any, color_id: Any) -> str:
    return f"{str(part_num or '').strip()}::{int(color_id or 0)}"


def _debug_bag_specific_part_rows(set_num: str, bag: int) -> List[Dict[str, Any]]:
    db_path = Path("/Users/olly/aim2build-instruction/bag_inspector.db")
    if not db_path.exists():
        return []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT part_num, color_id, qty_total
            FROM set_bag_parts
            WHERE set_num = ? AND bag_number = ?
            ORDER BY part_num, color_id
            """,
            (str(set_num or "").strip(), int(bag or 1)),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    for row in rows:
        part_num = str(row["part_num"] or "").strip()
        color_id = _coerce_int(row["color_id"])
        if not part_num or color_id is None:
            continue
        out.append(
            {
                "part_num": part_num,
                "color_id": int(color_id),
                "qty": int(row["qty_total"] or 0),
            }
        )
    return out


def _slot_mask_candidate_pool(set_num: str, bag: int) -> Tuple[List[Dict[str, Any]], str]:
    parts_payload = load_instruction_set_parts(str(set_num))
    set_parts = _prepare_instruction_parts_for_display(list(parts_payload.get("parts", []) or []))
    set_by_key = {
        _candidate_part_key(part.get("part_num"), part.get("color_id")): dict(part or {})
        for part in set_parts
    }
    bag_rows = _debug_bag_specific_part_rows(str(set_num), int(bag or 1))
    source = "bag_specific" if bag_rows else "set_fallback"
    seed_rows = bag_rows if bag_rows else set_parts
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in seed_rows:
        part_num = str(row.get("part_num") or "").strip()
        color_id = _coerce_int(row.get("color_id"))
        if not part_num or color_id is None:
            continue
        key = _candidate_part_key(part_num, color_id)
        if key in seen:
            continue
        seen.add(key)
        meta = dict(set_by_key.get(key, {}) or {})
        candidate = dict(meta)
        candidate.update(
            {
                "part_num": part_num,
                "color_id": int(color_id),
                "qty": int(row.get("qty", meta.get("qty", 0)) or 0),
            }
        )
        if not str(candidate.get("img_url") or "").strip():
            candidate["img_url"] = str(meta.get("img_url") or "").strip()
        if not str(candidate.get("color_name") or "").strip():
            candidate["color_name"] = str(meta.get("color_name") or "n/a")
        if not str(candidate.get("element_id") or "").strip():
            candidate["element_id"] = str(meta.get("element_id") or "")
        candidates.append(candidate)
    return candidates, source


def _slot_mask_resolve_local_image_path(img_url: Any) -> Optional[Path]:
    text = str(img_url or "").strip()
    if not text:
        return None
    if text.startswith(("http://", "https://")):
        return None
    if text.startswith("file://"):
        text = text[7:]
    path = Path(text).expanduser()
    if path.exists():
        return path
    repo_path = Path("/Users/olly/aim2build-instruction") / text
    if repo_path.exists():
        return repo_path
    return None


def _slot_mask_read_rgba(path_value: Any) -> Optional[np.ndarray]:
    path = Path(str(path_value or "").strip()).expanduser()
    if not path.exists() or not path.is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or getattr(img, "size", 0) == 0:
        return None
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.full(img.shape, 255, dtype=np.uint8)
        return np.dstack([bgr, alpha])
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        return np.dstack([img, alpha])
    return img[:, :, :4]


def _slot_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _slot_mask_normalized(mask: np.ndarray, size: int = 64) -> np.ndarray:
    bbox = _slot_mask_bbox(mask)
    canvas = np.zeros((size, size), dtype=np.uint8)
    if bbox is None:
        return canvas
    x, y, w, h = bbox
    crop = (mask[y : y + h, x : x + w] > 0).astype(np.uint8) * 255
    scale = min(size / max(1, w), size / max(1, h))
    nw = max(1, min(size, int(round(w * scale))))
    nh = max(1, min(size, int(round(h * scale))))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def _slot_mask_profile_from_rgba(rgba: np.ndarray, mask_override: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    bgr = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    if mask_override is not None:
        mask = (mask_override > 20).astype(np.uint8) * 255
    else:
        mask = (alpha > 20).astype(np.uint8) * 255
        if int(np.count_nonzero(mask)) >= int(mask.size * 0.96):
            border = np.concatenate(
                [
                    bgr[:2, :, :].reshape(-1, 3),
                    bgr[-2:, :, :].reshape(-1, 3),
                    bgr[:, :2, :].reshape(-1, 3),
                    bgr[:, -2:, :].reshape(-1, 3),
                ],
                axis=0,
            )
            bg = np.median(border, axis=0)
            delta = np.linalg.norm(bgr.astype(np.float32) - bg.reshape(1, 1, 3), axis=2)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            mask = ((delta > 22) & (val < 248) & ~((sat < 18) & (val > 235))).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    bbox = _slot_mask_bbox(mask)
    if bbox is None:
        return None
    pixels = bgr[mask > 0]
    if pixels.size == 0:
        return None
    median_bgr = np.median(pixels.reshape(-1, 3), axis=0).astype(np.float32)
    x, y, w, h = bbox
    aspect = float(w) / float(max(1, h))
    return {
        "bgr": median_bgr,
        "aspect": aspect,
        "mask": _slot_mask_normalized(mask),
        "area_ratio": float(np.count_nonzero(mask)) / float(max(1, mask.size)),
    }


def _slot_mask_query_profile(part_cutout_path: Any, shape_mask_path: Any) -> Optional[Dict[str, Any]]:
    cutout = _slot_mask_read_rgba(part_cutout_path)
    if cutout is None:
        return None
    mask_override: Optional[np.ndarray] = None
    mask_path = Path(str(shape_mask_path or "").strip()).expanduser()
    if mask_path.exists() and mask_path.is_file():
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is not None and getattr(mask_img, "size", 0) != 0:
            if mask_img.shape[:2] != cutout.shape[:2]:
                mask_img = cv2.resize(mask_img, (cutout.shape[1], cutout.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_override = mask_img
    return _slot_mask_profile_from_rgba(cutout, mask_override=mask_override)


def _slot_mask_candidate_profile(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    image_path = _slot_mask_resolve_local_image_path(part.get("img_url"))
    if image_path is None:
        return None
    rgba = _slot_mask_read_rgba(image_path)
    if rgba is None:
        return None
    return _slot_mask_profile_from_rgba(rgba)


def _slot_mask_hex_to_bgr(value: Any) -> Optional[np.ndarray]:
    rgb_hex = _normalize_rgb_hex(value)
    if not rgb_hex:
        return None
    return np.array(
        [
            int(rgb_hex[4:6], 16),
            int(rgb_hex[2:4], 16),
            int(rgb_hex[0:2], 16),
        ],
        dtype=np.float32,
    )


def _slot_mask_score_candidate(
    query: Dict[str, Any],
    candidate: Dict[str, Any],
    color_bgr_by_id: Dict[int, np.ndarray],
) -> Dict[str, float]:
    candidate_profile = _slot_mask_candidate_profile(candidate)
    if candidate_profile is not None:
        candidate_bgr = candidate_profile["bgr"]
        aspect_score = 1.0 - min(
            1.0,
            abs(float(query["aspect"]) - float(candidate_profile["aspect"]))
            / max(float(query["aspect"]), float(candidate_profile["aspect"]), 0.01),
        )
        query_mask = (query["mask"] > 20).astype(np.uint8)
        candidate_mask = (candidate_profile["mask"] > 20).astype(np.uint8)
        intersection = int(np.count_nonzero(query_mask & candidate_mask))
        union = int(np.count_nonzero(query_mask | candidate_mask))
        silhouette_score = float(intersection) / float(union) if union else 0.0
    else:
        candidate_bgr = color_bgr_by_id.get(int(candidate.get("color_id", 0) or 0))
        aspect_score = 0.0
        silhouette_score = 0.0
    if candidate_bgr is None:
        colour_score = 0.0
    else:
        distance = float(np.linalg.norm(np.asarray(query["bgr"], dtype=np.float32) - candidate_bgr))
        colour_score = max(0.0, 1.0 - distance / 441.7)
    confidence = (0.50 * colour_score) + (0.25 * aspect_score) + (0.25 * silhouette_score)
    return {
        "colour": round(float(colour_score), 4),
        "aspect": round(float(aspect_score), 4),
        "silhouette": round(float(silhouette_score), 4),
        "confidence": round(float(confidence), 4),
        "candidate_image_available": 1.0 if candidate_profile is not None else 0.0,
    }


# ---------------------------------------------------------------------------
# Confirmed-label visual memory — local, deterministic, no FAISS/CLIP needed.
# ---------------------------------------------------------------------------

_CONFIRMED_MEMORY_THRESHOLD = 0.72


def _confirmed_label_memory(set_num: str, bag: int) -> List[Dict[str, Any]]:
    """Build a list of visual memory entries from confirmed slot labels in this bag.

    Each entry carries:
        profile  – same dict as _slot_mask_profile_from_rgba (bgr, aspect, mask, area_ratio)
        part_num, color_id, element_id, color_name
        source_crop_id, source_slot_index
    Only entries whose ai_snap_input_path file still exists on disk are included.
    """
    labels_path = _label_store_path(str(set_num), int(bag))
    labels_payload = _load_existing_labels(labels_path)
    memory: List[Dict[str, Any]] = []
    for crop_id, saved_crop in dict(labels_payload.get("crops") or {}).items():
        if not isinstance(saved_crop, dict):
            continue
        for part in list(saved_crop.get("parts") or []):
            if not isinstance(part, dict):
                continue
            part_num = str(part.get("part_num") or "").strip()
            color_id = _coerce_int(part.get("color_id"))
            if not part_num or color_id is None:
                continue
            ai_snap_path = str(part.get("ai_snap_input_path") or "").strip()
            if not ai_snap_path:
                continue
            profile = _slot_mask_query_profile(ai_snap_path, "")
            if profile is None:
                continue
            memory.append(
                {
                    "profile": profile,
                    "part_num": part_num,
                    "color_id": int(color_id),
                    "element_id": str(part.get("element_id") or ""),
                    "color_name": str(part.get("color_name") or ""),
                    "source_crop_id": str(crop_id),
                    "source_slot_index": _coerce_int(part.get("selected_slot_index")),
                }
            )
    return memory


def _compare_slot_profiles(q: Dict[str, Any], m: Dict[str, Any]) -> float:
    """Combined similarity between two slot profiles (both from _slot_mask_profile_from_rgba).

    Weights: 50 % colour, 25 % aspect, 25 % mask-IoU — identical to _slot_mask_score_candidate
    so thresholds are directly comparable.
    Returns a float in [0.0, 1.0].
    """
    q_bgr = np.asarray(q["bgr"], dtype=np.float32)
    m_bgr = np.asarray(m["bgr"], dtype=np.float32)
    colour_score = max(0.0, 1.0 - float(np.linalg.norm(q_bgr - m_bgr)) / 441.7)

    q_asp = float(q.get("aspect") or 1.0)
    m_asp = float(m.get("aspect") or 1.0)
    aspect_score = 1.0 - min(1.0, abs(q_asp - m_asp) / max(q_asp, m_asp, 0.01))

    q_mask = (np.asarray(q["mask"]) > 20).astype(np.uint8)
    m_mask = (np.asarray(m["mask"]) > 20).astype(np.uint8)
    intersection = int(np.count_nonzero(q_mask & m_mask))
    union = int(np.count_nonzero(q_mask | m_mask))
    silhouette_score = float(intersection) / float(union) if union else 0.0

    return round(
        (0.50 * colour_score) + (0.25 * aspect_score) + (0.25 * silhouette_score), 4
    )


def _apply_confirmed_memory_predictions(
    slots: List[Dict[str, Any]],
    set_num: str,
    bag: int,
) -> None:
    """Annotate masked slots with predictions from confirmed visual memory.

    Mutates slot dicts in-place; adds three optional keys when a match is found:
        predicted_part        – {part_num, color_id, element_id, color_name}
        prediction_source     – "predicted_from_confirmed"
        prediction_similarity – float score that triggered the match

    Only masked slots with a part_cutout_path are considered.
    Does not overwrite any previously set predicted_part.
    """
    masked = [
        s for s in (slots or [])
        if str(s.get("status") or "") == "masked" and s.get("part_cutout_path")
    ]
    if not masked:
        return

    memory = _confirmed_label_memory(set_num, bag)
    if not memory:
        return

    for slot in masked:
        if slot.get("predicted_part"):
            continue  # already predicted (e.g. from a previous pass)
        query = _slot_mask_query_profile(
            str(slot["part_cutout_path"]),
            str(slot.get("shape_mask_path") or ""),
        )
        if query is None:
            continue
        best_score = 0.0
        best_entry: Optional[Dict[str, Any]] = None
        for entry in memory:
            score = _compare_slot_profiles(query, entry["profile"])
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry is not None and best_score >= _CONFIRMED_MEMORY_THRESHOLD:
            slot["predicted_part"] = {
                "part_num": best_entry["part_num"],
                "color_id": best_entry["color_id"],
                "element_id": best_entry["element_id"],
                "color_name": best_entry["color_name"],
            }
            slot["prediction_source"] = "predicted_from_confirmed"
            slot["prediction_similarity"] = round(best_score, 4)
            print(
                "[confirmed-memory] "
                f"crop_id={slot.get('slot_index')} slot_index={slot.get('slot_index')} "
                f"predicted={best_entry['part_num']}:{best_entry['color_id']} "
                f"similarity={best_score:.4f} "
                f"source_crop={best_entry['source_crop_id']}"
            )


def _ai_snap_crop_from_saved_record(crop_id: str, saved_crop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    crop_box = _coerce_box_list(saved_crop.get("crop_box"))
    crop_image_path = str(saved_crop.get("crop_image_path") or "").strip()
    data_uri = ""
    if crop_box is not None and crop_image_path:
        img = cv2.imread(crop_image_path)
        if img is not None:
            data_uri = str(_encode_contact_sheet_crop(img, crop_box, max_edge=420) or "")
    qty_token_boxes = [
        dict(item)
        for item in list(saved_crop.get("qty_token_boxes", []) or [])
        if isinstance(item, dict)
    ]
    return {
        "crop_id": str(crop_id or "").strip(),
        "page": int(saved_crop.get("page", 0) or 0),
        "step": int(saved_crop.get("step", 0) or 0),
        "qty_text": _coerce_str_list(saved_crop.get("qty_text", [])),
        "qty_numbers": _coerce_int_list(saved_crop.get("qty", [])),
        "qty_label": ", ".join(_coerce_str_list(saved_crop.get("qty_text", []))) or "none",
        "crop_box": crop_box or [],
        "crop_box_format": str(saved_crop.get("crop_box_format") or "xywh"),
        "crop_image_path": crop_image_path,
        "confidence": _coerce_float(saved_crop.get("confidence")),
        "data_uri": data_uri,
        "qty_token_boxes": qty_token_boxes,
    }


def _load_crop_for_ai_snap(set_num: str, bag: int, crop_id: str) -> Optional[Dict[str, Any]]:
    labels_path = _label_store_path(str(set_num), int(bag))
    labels_payload = _load_existing_labels(labels_path)
    saved_crop = dict(labels_payload.get("crops", {}).get(crop_id) or {})
    built_crops = _build_instruction_callout_crops(str(set_num), int(bag), ai_enabled=False)
    built_by_id = {
        str(item.get("crop_id") or "").strip(): dict(item or {})
        for item in built_crops
        if str(item.get("crop_id") or "").strip()
    }
    crop = dict(built_by_id.get(crop_id) or {})
    if not crop and saved_crop:
        crop = _ai_snap_crop_from_saved_record(crop_id, saved_crop) or {}
    if not crop:
        return None

    saved_qty_text = _coerce_str_list(saved_crop.get("qty_text", []))
    saved_qty_numbers = _coerce_int_list(saved_crop.get("qty", []))
    crop["page"] = int(saved_crop.get("page", crop.get("page", 0)) or crop.get("page", 0) or 0)
    crop["step"] = int(saved_crop.get("step", crop.get("step", 0)) or crop.get("step", 0) or 0)
    crop["crop_box"] = _coerce_box_list(saved_crop.get("crop_box")) or _coerce_box_list(crop.get("crop_box")) or []
    crop["crop_box_format"] = str(saved_crop.get("crop_box_format") or crop.get("crop_box_format") or "xywh")
    crop["crop_image_path"] = str(saved_crop.get("crop_image_path") or crop.get("crop_image_path") or "")
    crop["confidence"] = _coerce_float(saved_crop.get("confidence", crop.get("confidence")))
    built_qty_token_boxes = [
        dict(item)
        for item in list(crop.get("qty_token_boxes", []) or [])
        if isinstance(item, dict)
    ]
    saved_qty_token_boxes = [
        dict(item)
        for item in list(saved_crop.get("qty_token_boxes", []) or [])
        if isinstance(item, dict)
    ]
    crop["qty_token_boxes"] = saved_qty_token_boxes or built_qty_token_boxes
    if saved_qty_text:
        crop["qty_text"] = saved_qty_text
        crop["qty_numbers"] = saved_qty_numbers
    else:
        crop["qty_text"] = _coerce_str_list(crop.get("qty_text", []))
        crop["qty_numbers"] = _coerce_int_list(crop.get("qty_numbers", crop.get("qty", [])))
    crop["qty_label"] = ", ".join(list(crop.get("qty_text", []) or [])) or "none"
    if not str(crop.get("data_uri") or "").strip():
        crop_box = _coerce_box_list(crop.get("crop_box"))
        crop_image_path = str(crop.get("crop_image_path") or "").strip()
        if crop_box is not None and crop_image_path:
            img = cv2.imread(crop_image_path)
            if img is not None:
                crop["data_uri"] = str(_encode_contact_sheet_crop(img, crop_box, max_edge=420) or "")
    return crop


def _assigned_part_totals_from_labels(labels_payload: Dict[str, Any]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for crop_data in dict(labels_payload.get("crops") or {}).values():
        crop_dict = crop_data if isinstance(crop_data, dict) else {}
        for raw_part in list(crop_dict.get("parts", []) or []):
            normalized_part = _normalize_part_entry(raw_part if isinstance(raw_part, dict) else {})
            if not normalized_part["part_num"]:
                continue
            qty_value = _coerce_int(normalized_part.get("qty"))
            assigned_qty = int(qty_value) if qty_value is not None and int(qty_value) > 0 else 1
            key = _candidate_part_key(normalized_part["part_num"], normalized_part["color_id"])
            totals[key] = totals.get(key, 0) + assigned_qty
    return totals


def _latest_manual_color_sample_for_crop(set_num: str, crop_id: str, page: int, step: int) -> Optional[Dict[str, Any]]:
    samples = list(_load_manual_color_calibration(set_num).get("samples", []) or [])
    exact_match: Optional[Dict[str, Any]] = None
    page_step_match: Optional[Dict[str, Any]] = None
    for sample in reversed(samples):
        if not isinstance(sample, dict):
            continue
        sample_color_id = _coerce_int(sample.get("color_id"))
        if sample_color_id is None:
            continue
        if str(sample.get("crop_id") or "").strip() == str(crop_id or "").strip():
            exact_match = sample
            break
        if (
            _coerce_int(sample.get("page")) == int(page or 0)
            and _coerce_int(sample.get("step")) == int(step or 0)
        ):
            page_step_match = sample
    return exact_match or page_step_match


def _remaining_part_rows_for_ai_snap(
    parts: List[Dict[str, Any]],
    assigned_totals: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    remaining_rows: Dict[str, Dict[str, Any]] = {}
    for part in list(parts or []):
        part_num = str(part.get("part_num") or "").strip()
        color_id = int(part.get("color_id", 0) or 0)
        if not part_num:
            continue
        key = _candidate_part_key(part_num, color_id)
        required_qty = int(part.get("qty", 0) or 0)
        assigned_qty = int(assigned_totals.get(key, 0) or 0)
        remaining_qty = required_qty - assigned_qty
        if remaining_qty <= 0:
            continue
        remaining_rows[key] = {
            "part_num": part_num,
            "color_id": color_id,
            "remaining_qty": remaining_qty,
            "required_qty": required_qty,
            "assigned_qty": assigned_qty,
        }
    return remaining_rows


def _write_ai_snap_temp_crop_image(crop: Dict[str, Any]) -> Optional[Path]:
    crop_box = _coerce_box_list(crop.get("crop_box"))
    crop_image_path = str(crop.get("crop_image_path") or "").strip()
    if not crop_box or not crop_image_path:
        return None
    image_path = Path(crop_image_path)
    if not image_path.exists():
        return None
    img = cv2.imread(str(image_path))
    if img is None or getattr(img, "size", 0) == 0:
        return None
    x, y, w, h = [int(value) for value in crop_box]
    crop_img = img[max(0, y) : max(0, y) + max(0, h), max(0, x) : max(0, x) + max(0, w)]
    if crop_img is None or getattr(crop_img, "size", 0) == 0:
        return None
    handle = tempfile.NamedTemporaryFile(prefix="ai_snap_crop_", suffix=".png", delete=False)
    handle.close()
    out_path = Path(handle.name)
    ok = cv2.imwrite(str(out_path), crop_img)
    if not ok:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    return out_path


def _mock_rank_slot_candidates(
    parts: List[Dict[str, Any]],
    assigned_totals: Dict[str, int],
    slot_qty: Optional[int],
    manual_color_filter_id: Optional[int],
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for part in list(parts or []):
        part_num = str(part.get("part_num") or "").strip()
        color_id = int(part.get("color_id", 0) or 0)
        if not part_num:
            continue
        key = _candidate_part_key(part_num, color_id)
        required_qty = int(part.get("qty", 0) or 0)
        assigned_qty = int(assigned_totals.get(key, 0) or 0)
        remaining_qty = required_qty - assigned_qty
        if remaining_qty <= 0:
            continue
        score = 0.0
        reasons: List[str] = ["remaining_candidate"]
        if manual_color_filter_id is not None:
            if color_id == int(manual_color_filter_id):
                score += 500.0
                reasons.append("manual_color_match")
            else:
                score -= 60.0
        if slot_qty is not None and slot_qty > 0:
            if remaining_qty >= slot_qty:
                score += 140.0
                reasons.append("remaining_qty_fits_slot")
            else:
                score -= float((slot_qty - remaining_qty) * 18)
                reasons.append("remaining_qty_below_slot")
            if required_qty >= slot_qty:
                score += 40.0
                reasons.append("set_qty_fits_slot")
        if str(part.get("img_url") or "").strip():
            score += 25.0
            reasons.append("image_available")
        if str(part.get("element_id") or "").strip():
            score += 8.0
        score += min(remaining_qty, 12) * 2.5
        score += min(required_qty, 20) * 0.75
        ranked.append(
            {
                "part_num": part_num,
                "display_part_num": str(part.get("display_part_num") or "").strip() or part_num,
                "color_id": color_id,
                "color_name": str(part.get("color_name") or f"color {color_id}"),
                "element_id": str(part.get("element_id") or "").strip() or None,
                "img_url": str(part.get("img_url") or "").strip(),
                "required_qty": required_qty,
                "assigned_qty": assigned_qty,
                "remaining_qty": remaining_qty,
                "mock_score": round(score, 3),
                "reason": ", ".join(reasons),
            }
        )
    ranked.sort(
        key=lambda item: (
            -float(item.get("mock_score", 0.0) or 0.0),
            -int(item.get("remaining_qty", 0) or 0),
            str(item.get("part_num") or ""),
            int(item.get("color_id", 0) or 0),
        )
    )
    top_score = float(ranked[0].get("mock_score", 0.0) or 0.0) if ranked else 0.0
    for index, candidate in enumerate(ranked):
        relative_boost = 0.0
        if top_score > 0:
            relative_boost = min(0.08, max(0.0, float(candidate.get("mock_score", 0.0) or 0.0) / top_score * 0.08))
        confidence = max(0.28, min(0.98, 0.9 - (index * 0.12) + relative_boost))
        candidate["confidence"] = round(confidence, 2)
        candidate["rank"] = index + 1
    return ranked


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


def _load_manual_color_calibration(set_num: str) -> Dict[str, Any]:
    path = _manual_color_calibration_path(set_num)
    existing = _empty_manual_color_calibration(set_num)
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

    existing["schema_version"] = "1.0"
    existing["set_num"] = str(loaded.get("set_num") or existing["set_num"]).strip() or existing["set_num"]
    existing["updated_at"] = str(loaded.get("updated_at") or existing["updated_at"]).strip() or existing["updated_at"]

    samples: List[Dict[str, Any]] = []
    for item in list(loaded.get("samples", []) or []):
        if not isinstance(item, dict):
            continue
        sample_rgb = item.get("sample_rgb") if isinstance(item.get("sample_rgb"), dict) else {}
        sample_xy = item.get("sample_xy") if isinstance(item.get("sample_xy"), dict) else {}
        color_id = _coerce_int(item.get("color_id"))
        if color_id is None:
            continue
        rgb_r = _coerce_int(sample_rgb.get("r"))
        rgb_g = _coerce_int(sample_rgb.get("g"))
        rgb_b = _coerce_int(sample_rgb.get("b"))
        sample_x = _coerce_int(sample_xy.get("x"))
        sample_y = _coerce_int(sample_xy.get("y"))
        if rgb_r is None or rgb_g is None or rgb_b is None:
            continue
        normalized_sample = {
            "sample_id": str(item.get("sample_id") or "").strip(),
            "page": _coerce_int(item.get("page")),
            "step": _coerce_int(item.get("step")),
            "crop_id": str(item.get("crop_id") or "").strip(),
            "crop_image_path": str(item.get("crop_image_path") or "").strip(),
            "sample_xy": {
                "x": sample_x if sample_x is not None else 0,
                "y": sample_y if sample_y is not None else 0,
            },
            "sample_radius": max(0, int(_coerce_int(item.get("sample_radius")) or 0)),
            "sample_rgb": {
                "r": int(rgb_r),
                "g": int(rgb_g),
                "b": int(rgb_b),
            },
            "color_id": int(color_id),
            "color_name": str(item.get("color_name") or "").strip(),
            "source": str(item.get("source") or "manual_picker").strip() or "manual_picker",
            "saved_at": str(item.get("saved_at") or "").strip(),
        }
        if not normalized_sample["sample_id"]:
            page_part = normalized_sample["page"] if normalized_sample["page"] is not None else "na"
            step_part = normalized_sample["step"] if normalized_sample["step"] is not None else "na"
            normalized_sample["sample_id"] = (
                f"{normalized_sample['crop_id'] or 'manual'}_p{page_part}_s{step_part}_{len(samples) + 1}"
            )
        samples.append(normalized_sample)
    existing["samples"] = samples
    return existing


def _load_clip_memory(set_num: str, bag: int) -> Dict[str, Any]:
    path = _clip_memory_path(set_num, bag)
    existing = _empty_clip_memory(set_num, bag)
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

    existing["schema_version"] = "1.0"
    existing["set_num"] = str(loaded.get("set_num") or existing["set_num"]).strip() or existing["set_num"]
    existing["bag"] = max(1, int(_coerce_int(loaded.get("bag")) or existing["bag"]))
    existing["updated_at"] = str(loaded.get("updated_at") or existing["updated_at"]).strip() or existing["updated_at"]

    items: List[Dict[str, Any]] = []
    for item in list(loaded.get("items", []) or []):
        if not isinstance(item, dict):
            continue
        part_num = str(item.get("part_num") or "").strip()
        color_id = _coerce_int(item.get("color_id"))
        slot_idx = _coerce_int(item.get("slot_index"))
        embedding_raw = item.get("embedding")
        if not part_num or color_id is None or slot_idx is None or not isinstance(embedding_raw, list):
            continue
        embedding: List[float] = []
        for value in embedding_raw:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                embedding = []
                break
            if not np.isfinite(numeric):
                embedding = []
                break
            embedding.append(numeric)
        if not embedding:
            continue
        items.append(
            {
                "crop_id": str(item.get("crop_id") or "").strip(),
                "slot_index": int(slot_idx),
                "part_num": part_num,
                "color_id": int(color_id),
                "embedding": embedding,
                "updated_at": str(item.get("updated_at") or "").strip(),
            }
        )
    existing["items"] = items
    return existing


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


def _write_manual_color_calibration(set_num: str, payload: Dict[str, Any]) -> None:
    path = _manual_color_calibration_path(set_num)
    normalized = _empty_manual_color_calibration(set_num)
    if isinstance(payload, dict):
        normalized["set_num"] = str(payload.get("set_num") or normalized["set_num"]).strip() or normalized["set_num"]
        normalized["updated_at"] = str(payload.get("updated_at") or normalized["updated_at"]).strip() or normalized["updated_at"]
        normalized["samples"] = list(payload.get("samples", []) or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _write_clip_memory(set_num: str, bag: int, payload: Dict[str, Any]) -> None:
    path = _clip_memory_path(set_num, bag)
    normalized = _empty_clip_memory(set_num, bag)
    if isinstance(payload, dict):
        normalized["set_num"] = str(payload.get("set_num") or normalized["set_num"]).strip() or normalized["set_num"]
        normalized["bag"] = max(1, int(_coerce_int(payload.get("bag")) or normalized["bag"]))
        normalized["updated_at"] = str(payload.get("updated_at") or normalized["updated_at"]).strip() or normalized["updated_at"]
        normalized["items"] = list(payload.get("items", []) or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _build_instruction_callout_crops(
    set_num: str,
    bag: int,
    ai_enabled: bool = False,
    step_filter: Optional[int] = None,
    page_filter: Optional[int] = None,
    rebuild: bool = False,
) -> List[Dict[str, Any]]:
    rendered_pages, start_page, end_page = _resolve_bag_page_range(str(set_num), int(bag))
    if page_filter is not None:
        rendered_pages = [int(page_filter)]
        start_page = min(int(start_page), int(page_filter))
        end_page = max(int(end_page), int(page_filter))
    crops: List[Dict[str, Any]] = []

    def _recover_right_half_missing_steps(
        img: Any,
        page_width: int,
        page_height: int,
        step_boxes: List[Dict[str, Any]],
        missing_steps: List[int],
    ) -> List[Dict[str, Any]]:
        import pytesseract

        missing = {
            int(value)
            for value in (missing_steps or [])
            if int(value) > 0 and int(value) not in {int(item.get("step_number", 0) or 0) for item in (step_boxes or [])}
        }
        recovered: List[Dict[str, Any]] = []
        for y0 in range(0, max(220, int(page_height * 0.60)), 120):
            for x0 in range(int(page_width * 0.45), max(int(page_width * 0.45) + 1, page_width - 180), 160):
                if not missing:
                    return recovered
                roi = img[y0 : min(page_height, y0 + 220), x0 : min(page_width, x0 + 260)]
                if roi is None or roi.size == 0:
                    continue
                try:
                    data = pytesseract.image_to_data(roi, config="--psm 11 -c tessedit_char_whitelist=0123456789", output_type=pytesseract.Output.DICT)
                except Exception:
                    continue
                for idx in range(len(data.get("text", []) or [])):
                    text = str((data.get("text", [""])[idx] or "")).strip()
                    value = int(text) if text.isdigit() else 0
                    if value not in missing:
                        continue
                    try:
                        conf = float((data.get("conf", ["-1"])[idx] or -1))
                    except Exception:
                        conf = -1.0
                    if conf <= 30:
                        continue
                    recovered.append({"x": x0 + int(data.get("left", [0])[idx] or 0), "y": y0 + int(data.get("top", [0])[idx] or 0), "w": int(data.get("width", [0])[idx] or 0), "h": int(data.get("height", [0])[idx] or 0), "step_number": value, "source": "gap_recovery_ocr", "label": str(value)})
                    missing.discard(value)
        return recovered

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
        step_boxes = _filter_invalid_step_anchor_boxes(_contact_sheet_step_boxes_from_detected(detected))
        current_steps = sorted({int(item.get("step_number", 0) or 0) for item in (step_boxes or []) if int(item.get("step_number", 0) or 0) > 0})
        next_page = next((int(candidate) for candidate in rendered_pages if int(candidate) > int(page) and int(candidate) >= int(start_page) and int(candidate) <= int(end_page)), None)
        if current_steps and next_page is not None:
            try:
              next_detected = step_detector_service.detect_steps(str(set_num), int(next_page))
              next_step_boxes = _filter_invalid_step_anchor_boxes(_contact_sheet_step_boxes_from_detected(next_detected))
            except Exception:
              next_step_boxes = []
            next_steps = sorted({int(item.get("step_number", 0) or 0) for item in (next_step_boxes or []) if int(item.get("step_number", 0) or 0) > 0})
            if next_steps and int(min(next_steps)) - int(max(current_steps)) >= 3:
                recovered = _recover_right_half_missing_steps(img, page_width, page_height, step_boxes, list(range(int(max(current_steps)) + 1, int(min(next_steps)))))
                if recovered:
                  step_boxes = sorted(_filter_invalid_step_anchor_boxes(list(step_boxes or []) + recovered), key=lambda item: (int(item.get("y", 0) or 0), int(item.get("x", 0) or 0), int(item.get("step_number", 0) or 0)))
        edge_callout_candidates: List[Dict[str, Any]] = []
        for step_box in step_boxes or []:
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
            search_x1 = max(0, sx - 35)
            search_y1 = max(0, sy - 290)
            search_x2 = min(page_width, sx + sw + 690)
            search_y2 = max(0, min(page_height, sy - 5))
            if search_y2 <= search_y1 or search_x2 <= search_x1:
                continue

            rect = _detect_callout_rect_by_edges(
                img,
                [search_x1, search_y1, search_x2, search_y2],
                step_y=sy,
                page_width=page_width,
                page_height=page_height,
            )

            if rect is None:
                continue

            rx, ry, rw, rh = [int(v) for v in rect]
            crop_img = img[ry : ry + rh, rx : rx + rw]
            if crop_img is None or crop_img.size == 0:
                continue
            try:
                data_uri = _encode_debug_image_data_uri(crop_img, max_width=420)
            except Exception:
                continue
            qty_payload = _auto_qty_payload_for_crop(crop_img, step_number)
            edge_callout_candidates.append(
                {
                    "candidate_origin": "callout_box_candidate",
                    "source": "edge_detect",
                    "match_enabled": True,
                    "data_uri": data_uri,
                    "coords_xywh": [rx, ry, rw, rh],
                    "coords_label": "edge detected callout",
                    "edge_rect": [rx, ry, rw, rh],
                    "confidence": 0.45,
                    "step_number": step_number,
                    "detected_qty_text": qty_payload.get("detected_qty_text", []),
                    "detected_qty_numbers": qty_payload.get("detected_qty_numbers", []),
                    "qty_token_boxes": qty_payload.get("qty_token_boxes"),
                }
            )

        if edge_callout_candidates:
            callout_candidates = edge_callout_candidates
        else:
            callout_candidates = _page_level_callout_candidates_for_fallback(
                img,
                page_width=page_width,
                page_height=page_height,
                step_boxes=step_boxes,
                set_num=str(set_num),
                page=int(page),
                rebuild=bool(rebuild),
            )
            if not callout_candidates:
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

        for idx, candidate in enumerate(callout_candidates, start=1):
            step_number = int(candidate.get("step_number", 0) or 0)
            if step_filter is not None and int(step_number) != int(step_filter):
                continue
            qty_payload: Dict[str, Any] = {
                "detected_qty_text": list(candidate.get("detected_qty_text", []) or []),
                "detected_qty_numbers": list(candidate.get("detected_qty_numbers", []) or []),
                "qty_source": str(candidate.get("qty_source") or "local"),
                "qty_ocr_source_regions": list(candidate.get("qty_ocr_source_regions", []) or []),
                "qty_ocr_ordered_qty_list": list(candidate.get("qty_ocr_ordered_qty_list", []) or []),
                "ai_part_count": None,
                "ai_issues": [],
                "ai_crop_box": None,
                "ai_suggested_fix": False,
            }
            candidate_source = str(candidate.get("source") or candidate.get("candidate_origin") or "")
            if candidate_source in {"edge_detect", "page_level_callout_assignment"}:
                crop_box = _coerce_box_list(candidate.get("coords_xywh"))
                if crop_box is not None:
                    x = int(crop_box[0] or 0)
                    y = int(crop_box[1] or 0)
                    w = int(crop_box[2] or 0)
                    h = int(crop_box[3] or 0)
                    if w > 0 and h > 0:
                        qty_payload = _qty_payload_for_page_level_callout_crop(
                            img[y : y + h, x : x + w],
                            int(step_number),
                            candidate_source,
                        )
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
            if step_number and str(qty_payload.get("qty_source") or "local") not in {"openai", "page_level_callout_assignment", "edge_detect"}:
                clean_pairs = [
                    (text, number)
                    for text, number in zip(detected_qty_text, detected_qty_numbers)
                    if int(number) != int(step_number)
                ]
                detected_qty_text = [text for text, _ in clean_pairs]
                detected_qty_numbers = [number for _, number in clean_pairs]
            qty_missing = not detected_qty_text and not detected_qty_numbers
            if qty_missing:
              candidate_box = _coerce_box_list(candidate.get("coords_xywh"))
              visible_part_estimate = 0
              if candidate_box is not None:
                cx, cy, cw, ch = [int(value) for value in candidate_box]
                if cw > 0 and ch > 0:
                  candidate_crop = img[cy : cy + ch, cx : cx + cw]
                  visible_part_estimate = _estimate_visible_part_count_from_crop(candidate_crop)
              if int(visible_part_estimate) <= 0:
                continue
            crop_id = f"p{int(page)}_s{max(step_number, 0)}_c{idx}"
            crops.append(
                {
                    "crop_id": crop_id,
                    "page": int(page),
                    "step": step_number,
                    "qty_text": detected_qty_text,
                    "qty_numbers": detected_qty_numbers,
                "candidate_detected_qty_text": list(candidate.get("detected_qty_text", []) or []),
                "candidate_detected_qty_numbers": list(candidate.get("detected_qty_numbers", []) or []),
                    "qty_label": ", ".join(detected_qty_text) if detected_qty_text else "none",
                    "qty_source": str(qty_payload.get("qty_source") or "local"),
                    "ai_part_count": qty_payload.get("ai_part_count"),
                    "ai_issues": list(qty_payload.get("ai_issues", []) or []),
                    "ai_crop_box": qty_payload.get("ai_crop_box"),
                    "ai_suggested_fix": bool(qty_payload.get("ai_suggested_fix")),
                    "source": str(candidate.get("source") or candidate.get("candidate_origin") or candidate.get("coords_label") or ""),
                    "data_uri": str(candidate.get("data_uri") or ""),
                    "coords_label": str(candidate.get("coords_label") or candidate.get("candidate_origin") or "fallback crop"),
                    "crop_box": list(candidate.get("coords_xywh", []) or []),
                    "crop_box_format": "xywh",
                    "crop_image_path": str(image_path),
                    "confidence": _coerce_float(candidate.get("confidence")),
                    "qty_token_boxes": qty_payload.get("qty_token_boxes") or candidate.get("qty_token_boxes"),
                    "qty_ocr_source_regions": qty_payload.get("qty_ocr_source_regions") or candidate.get("qty_ocr_source_regions"),
                    "qty_ocr_ordered_qty_list": qty_payload.get("qty_ocr_ordered_qty_list") or candidate.get("qty_ocr_ordered_qty_list"),
                    "edge_rect": candidate.get("edge_rect"),
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


def _debug_write_crop_temp_png(crop: Dict[str, Any]) -> Optional[str]:
    data_uri = str(crop.get("data_uri") or "").strip()
    if data_uri.startswith("data:image/") and "," in data_uri:
        try:
            encoded = data_uri.split(",", 1)[1]
            raw = base64.b64decode(encoded)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="manual_match_crop_") as handle:
                handle.write(raw)
                return str(handle.name)
        except Exception:
            pass

    crop_box = _coerce_box_list(crop.get("crop_box"))
    crop_image_path = str(crop.get("crop_image_path") or "").strip()
    if crop_box is None or not crop_image_path:
        return None
    img = cv2.imread(crop_image_path)
    if img is None:
        return None
    encoded = _encode_contact_sheet_crop(img, crop_box, max_edge=420)
    if not encoded or "," not in encoded:
        return None
    try:
        raw = base64.b64decode(encoded.split(",", 1)[1])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="manual_match_crop_") as handle:
            handle.write(raw)
            return str(handle.name)
    except Exception:
        return None


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
        for part_data in list(crop_record.get("parts", []) or []):
            part_entry = _normalize_part_entry(part_data if isinstance(part_data, dict) else {})
            if not part_entry["part_num"]:
                continue
            examples.append(
                {
                    "crop_id": str(crop_id or ""),
                    "page": int(crop_record.get("page", 0) or 0),
                    "step": int(crop_record.get("step", 0) or 0),
                    "crop_image_path": str(crop_record.get("crop_image_path") or ""),
                    "part_num": str(part_entry.get("part_num") or ""),
                    "color_id": int(part_entry.get("color_id", 0) or 0),
                    "color_name": str(part_entry.get("color_name") or "n/a"),
                    "qty": _coerce_int(part_entry.get("qty")) or 1,
                    "qty_text": str(part_entry.get("qty_text") or ""),
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
    selected_slot_index = _coerce_int(data.get("selected_slot_index"))
    sequence = _build_qty_sequence(
        data.get("crop_qty", data.get("qty", [])),
        data.get("crop_qty_text", data.get("qty_text", [])),
    )
    has_explicit_slot = selected_slot_index is not None and 0 <= int(selected_slot_index) < len(sequence)
    if has_explicit_slot:
        slot = dict(sequence[int(selected_slot_index)] or {})
        part_entry["qty"] = slot.get("qty")
        part_entry["qty_text"] = slot.get("qty_text")
        part_entry["selected_slot_index"] = int(selected_slot_index)
        replaced = False
        for index, existing_part in enumerate(list(crop_record["parts"] or [])):
            if _coerce_int((existing_part or {}).get("selected_slot_index")) == int(selected_slot_index):
                crop_record["parts"][index] = part_entry
                replaced = True
                break
        if (
            not replaced
            and int(selected_slot_index) < len(crop_record["parts"] or [])
            and _coerce_int((crop_record["parts"][int(selected_slot_index)] or {}).get("selected_slot_index")) is None
        ):
            crop_record["parts"][int(selected_slot_index)] = part_entry
            replaced = True
        if not replaced:
            crop_record["parts"].append(part_entry)
    else:
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


@router.post("/debug/save-manual-color-calibration")
async def save_manual_color_calibration(req: Request):
    data = await req.json()

    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    color_id = _coerce_int(data.get("color_id"))
    sample_rgb = data.get("sample_rgb") if isinstance(data.get("sample_rgb"), dict) else {}
    sample_xy = data.get("sample_xy") if isinstance(data.get("sample_xy"), dict) else {}
    rgb_r = _coerce_int(sample_rgb.get("r"))
    rgb_g = _coerce_int(sample_rgb.get("g"))
    rgb_b = _coerce_int(sample_rgb.get("b"))
    sample_x = _coerce_int(sample_xy.get("x"))
    sample_y = _coerce_int(sample_xy.get("y"))

    if color_id is None:
        raise HTTPException(status_code=400, detail="color_id is required")
    if rgb_r is None or rgb_g is None or rgb_b is None:
        raise HTTPException(status_code=400, detail="sample_rgb with r,g,b is required")
    if sample_x is None or sample_y is None:
        raise HTTPException(status_code=400, detail="sample_xy with x,y is required")

    page = _coerce_int(data.get("page"))
    step = _coerce_int(data.get("step"))
    saved_at = _iso_now()
    existing = _load_manual_color_calibration(set_num)
    samples = list(existing.get("samples", []) or [])
    sample_id = (
        str(data.get("crop_id") or "").strip()
        or f"manual_p{page if page is not None else 'na'}_s{step if step is not None else 'na'}"
    )
    sample_id = f"{sample_id}_{saved_at}"
    sample_entry = {
        "sample_id": sample_id,
        "page": page,
        "step": step,
        "crop_id": str(data.get("crop_id") or "").strip(),
        "crop_image_path": str(data.get("crop_image_path") or "").strip(),
        "sample_xy": {
            "x": int(sample_x),
            "y": int(sample_y),
        },
        "sample_radius": max(0, int(_coerce_int(data.get("sample_radius")) or 0)),
        "sample_rgb": {
            "r": int(rgb_r),
            "g": int(rgb_g),
            "b": int(rgb_b),
        },
        "color_id": int(color_id),
        "color_name": str(data.get("color_name") or "").strip(),
        "source": "manual_picker",
        "saved_at": saved_at,
    }
    samples.append(sample_entry)
    existing["schema_version"] = "1.0"
    existing["set_num"] = set_num
    existing["updated_at"] = saved_at
    existing["samples"] = samples
    _write_manual_color_calibration(set_num, existing)
    return {
        "ok": True,
        "path": str(_manual_color_calibration_path(set_num)),
        "sample": sample_entry,
        "sample_count": len(samples),
        "updated_at": saved_at,
    }


@router.post("/debug/ai-rank-slot")
async def ai_rank_slot(req: Request):
    data = await req.json()

    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    bag = _coerce_int(data.get("bag"))
    crop_id = str(data.get("crop_id") or "").strip()
    slot_index = _coerce_int(data.get("slot_index"))
    request_manual_color_filter_id = _coerce_int(data.get("manual_color_filter_id"))
    request_picked_rgb = data.get("picked_rgb") if isinstance(data.get("picked_rgb"), dict) else {}

    if bag is None or bag < 1:
        bag = 1
    if not crop_id:
        raise HTTPException(status_code=400, detail="crop_id is required")
    if slot_index is None or slot_index < 0:
        raise HTTPException(status_code=400, detail="slot_index is required")

    labels_path = _label_store_path(set_num, int(bag))
    labels_payload = _load_existing_labels(labels_path)
    crop = _load_crop_for_ai_snap(set_num, int(bag), crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="crop not found")

    slot_state = _crop_qty_slot_state(
        {"parts": list((dict(labels_payload.get("crops", {}).get(crop_id) or {}).get("parts", []) or []))},
        crop.get("qty_numbers", []),
        crop.get("qty_text", []),
    )
    sequence = list(slot_state.get("sequence", []) or [])
    if slot_index >= len(sequence):
        raise HTTPException(status_code=400, detail="slot_index out of range")

    slot = dict(sequence[slot_index] or {})
    slot_qty = _coerce_int(slot.get("qty"))
    slot_qty_text = str(slot.get("qty_text") or (f"{slot_qty}x" if slot_qty is not None else "none"))
    qty_token_boxes = [
        dict(item)
        for item in list(crop.get("qty_token_boxes", []) or [])
        if isinstance(item, dict)
    ]
    selected_qty_box: Optional[Dict[str, Any]] = None
    selected_qty_box_index = int(slot_index) - 1
    if 0 <= selected_qty_box_index < len(qty_token_boxes):
        selected_qty_box = dict(qty_token_boxes[selected_qty_box_index])
    elif qty_token_boxes:
        selected_qty_box = dict(qty_token_boxes[0])
    manual_color_sample = _latest_manual_color_sample_for_crop(
        set_num,
        crop_id,
        int(crop.get("page", 0) or 0),
        int(crop.get("step", 0) or 0),
    )
    manual_color_filter_id = request_manual_color_filter_id
    manual_color_source = "request"
    if manual_color_filter_id is None and manual_color_sample is not None:
        manual_color_filter_id = _coerce_int(manual_color_sample.get("color_id"))
        manual_color_source = "saved_calibration"
    if manual_color_filter_id is None:
        manual_color_source = "none"

    parts_payload = load_instruction_set_parts(set_num)
    parts = _prepare_instruction_parts_for_display(list(parts_payload.get("parts", []) or []))
    assigned_totals = _assigned_part_totals_from_labels(labels_payload)
    mock_ranked_pool = _mock_rank_slot_candidates(
        parts,
        assigned_totals,
        slot_qty=slot_qty,
        manual_color_filter_id=manual_color_filter_id,
    )
    mock_ranked_candidates = mock_ranked_pool[:5]
    remaining_part_rows = _remaining_part_rows_for_ai_snap(parts, assigned_totals)
    ranked_candidates = list(mock_ranked_candidates)
    model_name = "mock-local-ai-snap-v1"
    ai_enabled = False
    fallback_used = True
    ai_failure_reason = ""
    candidate_count = len(mock_ranked_candidates)
    ai_snap_input_path = ""
    shape_mask_path = ""
    part_cutout_path = ""
    mask_slot_index: Optional[int] = None
    cutout_slot_index: Optional[int] = None
    normalized_path = ""
    component_path = ""
    selected_box: Optional[Dict[str, Any]] = None
    normalization_fallback_reason = ""

    temp_crop_path: Optional[Path] = None
    try:
        temp_crop_path = _write_ai_snap_temp_crop_image(crop)
        if temp_crop_path is None:
            ai_failure_reason = "crop_image_unavailable"
            normalization_fallback_reason = "temp_crop_unavailable"
        else:
            rank_input_path = str(temp_crop_path)
            if selected_qty_box is not None:
                normalized_result = normalize_slot_crop_from_qty(str(temp_crop_path), selected_qty_box)
                if bool(normalized_result.get("ok")) and str(normalized_result.get("normalized_path") or "").strip():
                    rank_input_path = str(normalized_result.get("normalized_path") or "").strip()
                    normalized_path = rank_input_path
                    component_path = str(normalized_result.get("component_path") or "").strip()
                    selected_box = normalized_result.get("selected_box")
                else:
                    normalization_fallback_reason = str(
                        ((normalized_result.get("debug") or {}).get("error"))
                        or "slot_normalization_failed"
                    ).strip() or "slot_normalization_failed"
            else:
                normalization_fallback_reason = "selected_qty_box_unavailable"
            ai_snap_input_path = rank_input_path
            if ai_snap_input_path:
                shape_input_path = ai_snap_input_path
                shape_qty_box: Optional[Dict[str, Any]] = None
                if 0 <= int(slot_index) < len(qty_token_boxes):
                    shape_qty_box = dict(qty_token_boxes[int(slot_index)])
                if temp_crop_path is not None and shape_qty_box is not None:
                    shape_normalized_result = normalize_slot_crop_from_qty(str(temp_crop_path), shape_qty_box)
                    if bool(shape_normalized_result.get("ok")) and str(shape_normalized_result.get("normalized_path") or "").strip():
                        shape_input_path = str(shape_normalized_result.get("normalized_path") or "").strip()
                shape_result = create_shape_mask_for_slot_crop(
                    shape_input_path,
                    set_num=set_num,
                    bag=int(bag),
                    crop_id=crop_id,
                    slot_index=int(slot_index),
                )
                if bool(shape_result.get("ok")):
                    shape_mask_path = str(shape_result.get("shape_mask_path") or "")
                    part_cutout_path = str(shape_result.get("part_cutout_path") or "")
                    mask_slot_index = _coerce_int(shape_result.get("mask_slot_index"))
                    cutout_slot_index = _coerce_int(shape_result.get("cutout_slot_index"))
            if not normalized_path and not normalization_fallback_reason:
                normalization_fallback_reason = "normalized_path_unavailable"
            print(
                "[ai-snap] crop_id=%s slot_index=%s selected_qty_box=%s normalized_path=%s component_path=%s ai_snap_input_path=%s fallback_reason=%s"
                % (
                    str(crop_id),
                    str(slot_index),
                    json.dumps(selected_qty_box, ensure_ascii=True, sort_keys=True) if selected_qty_box is not None else "null",
                    str(normalized_path or ""),
                    str(component_path or ""),
                    str(ai_snap_input_path or ""),
                    str(normalization_fallback_reason or ""),
                )
            )
            local_color_ids = [int(manual_color_filter_id)] if manual_color_filter_id is not None else None
            local_candidates = get_part_candidates_for_crop(
                rank_input_path,
                max_candidates=8,
                color_ids=local_color_ids,
                metallic_mode=bool(crop.get("manual_metallic_mode")),
                set_num=set_num,
                remaining_parts=remaining_part_rows,
                hide_depleted=True,
            )
            parts_by_key = {
                _candidate_part_key(part.get("part_num"), part.get("color_id")): dict(part or {})
                for part in parts
            }
            filtered_candidates: List[Dict[str, Any]] = []
            seen_candidate_keys = set()
            for candidate in list(local_candidates or []):
                key = _candidate_part_key(candidate.get("part_num"), candidate.get("color_id"))
                if key in seen_candidate_keys or key not in remaining_part_rows:
                    continue
                meta = parts_by_key.get(key, {})
                filtered_candidates.append(
                    {
                        "part_num": str(candidate.get("part_num") or "").strip(),
                        "display_part_num": str(meta.get("display_part_num") or candidate.get("part_num") or "").strip(),
                        "color_id": int(candidate.get("color_id", 0) or 0),
                        "color_name": str(meta.get("color_name") or candidate.get("color_name") or f"color {int(candidate.get('color_id', 0) or 0)}"),
                        "element_id": str(meta.get("element_id") or candidate.get("element_id") or "").strip() or None,
                        "img_url": str(meta.get("img_url") or candidate.get("img_url") or "").strip(),
                        "remaining_qty": int((remaining_part_rows.get(key) or {}).get("remaining_qty", 0) or 0),
                        "required_qty": int((remaining_part_rows.get(key) or {}).get("required_qty", 0) or 0),
                        "assigned_qty": int((remaining_part_rows.get(key) or {}).get("assigned_qty", 0) or 0),
                        "candidate_source": str(candidate.get("candidate_source") or "local_prerank"),
                        "score": float(candidate.get("score", 0.0) or 0.0),
                    }
                )
                seen_candidate_keys.add(key)
            if manual_color_filter_id is not None and filtered_candidates:
                exact_color_candidates = [
                    dict(candidate)
                    for candidate in filtered_candidates
                    if int(candidate.get("color_id", 0) or 0) == int(manual_color_filter_id)
                ]
                if exact_color_candidates:
                    filtered_candidates = exact_color_candidates
            candidate_count = len(filtered_candidates)
            if not filtered_candidates:
                ai_failure_reason = "no_preranked_candidates"
            else:
                ai_crop_payload = {
                    "crop_id": crop_id,
                    "page": int(crop.get("page", 0) or 0),
                    "step": int(crop.get("step", 0) or 0),
                    "crop_image_path": rank_input_path,
                    "slot_index": int(slot_index),
                    "slot_qty_text": slot_qty_text,
                    "manual_color_filter_id": manual_color_filter_id,
                    "manual_color_name": (
                        str((manual_color_sample or {}).get("color_name") or "").strip()
                        if manual_color_filter_id is not None
                        else ""
                    ),
                }
                ai_result = rank_crop_candidates(ai_crop_payload, filtered_candidates[:8])
                ai_enabled = bool(ai_result.get("enabled"))
                if ai_enabled and list(ai_result.get("ranked_candidates") or []):
                    ranked_candidates = list(ai_result.get("ranked_candidates") or [])[:5]
                    model_name = str(ai_result.get("model") or model_name)
                    fallback_used = False
                    ai_failure_reason = ""
                    candidate_count = len(filtered_candidates[:8])
                else:
                    ai_failure_reason = str(ai_result.get("reason") or "azure_ai_rank_unavailable").strip() or "azure_ai_rank_unavailable"
                    model_name = str(ai_result.get("model") or model_name)
    finally:
        if temp_crop_path is not None:
            try:
                temp_crop_path.unlink(missing_ok=True)
            except Exception:
                pass

    return {
        "ok": True,
        "crop_id": crop_id,
        "slot_index": int(slot_index),
        "ranked_candidates": ranked_candidates,
        "model": model_name,
        "debug": {
            "set_num": set_num,
            "bag": int(bag),
            "crop_page": int(crop.get("page", 0) or 0),
            "crop_step": int(crop.get("step", 0) or 0),
            "slot_qty": slot_qty,
            "slot_qty_text": slot_qty_text,
            "selected_slot_index": int(slot_index),
            "sequence_length": len(sequence),
            "ai_snap_input_path": ai_snap_input_path,
            "shape_mask_path": shape_mask_path,
            "part_cutout_path": part_cutout_path,
            "mask_slot_index": mask_slot_index,
            "cutout_slot_index": cutout_slot_index,
            "normalized_path": normalized_path,
            "component_path": component_path,
            "selected_qty_box": selected_qty_box,
            "selected_box": selected_box,
            "qty_token_box_count": len(qty_token_boxes),
            "crop_image_path": str(crop.get("crop_image_path") or ""),
            "crop_image_available": bool(str(crop.get("data_uri") or "").strip() or str(crop.get("crop_image_path") or "").strip()),
            "manual_color_filter_id": manual_color_filter_id,
            "manual_color_filter_source": manual_color_source,
            "manual_color_sample_id": str((manual_color_sample or {}).get("sample_id") or ""),
            "picked_rgb_present": bool(
                _coerce_int(request_picked_rgb.get("r")) is not None
                and _coerce_int(request_picked_rgb.get("g")) is not None
                and _coerce_int(request_picked_rgb.get("b")) is not None
            ),
            "remaining_candidate_count": len(mock_ranked_pool),
            "candidate_count": candidate_count,
            "ai_enabled": ai_enabled,
            "fallback_used": fallback_used,
            "ai_failure_reason": ai_failure_reason,
            "model": model_name,
            "labels_path": str(labels_path),
        },
    }


@router.get("/debug/ai-snap-artifact")
def ai_snap_artifact(path: str = Query(...)):
    requested = Path(str(path or "").strip()).expanduser()
    allowed_root = (Path("/Users/olly/aim2build-instruction") / "debug" / "ai_training").resolve()
    try:
        resolved = requested.resolve()
    except Exception:
        raise HTTPException(status_code=404, detail="artifact not found")
    if allowed_root not in resolved.parents or not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(str(resolved))


@router.post("/debug/auto-mask-slots")
async def auto_mask_slots(req: Request):
    data = await req.json()
    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    bag = _coerce_int(data.get("bag"))
    crop_id = str(data.get("crop_id") or "").strip()
    sam_refine_flag = int(data.get("sam_refine") or 0) == 1
    fast_map_flag = int(data.get("fast_map") or 0) == 1
    if bag is None or bag < 1:
        bag = 1
    if not crop_id:
        raise HTTPException(status_code=400, detail="crop_id is required")

    crop = _load_crop_for_ai_snap(set_num, int(bag), crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="crop not found")
    qty_token_boxes = [
        dict(item)
        for item in list(crop.get("qty_token_boxes", []) or [])
        if isinstance(item, dict)
    ]
    temp_crop_path: Optional[Path] = None
    try:
        temp_crop_path = _write_ai_snap_temp_crop_image(crop)
        if temp_crop_path is None:
            raise HTTPException(status_code=400, detail="crop image unavailable")
        result = create_shape_masks_for_callout_slots(
            str(temp_crop_path),
            qty_token_boxes,
            set_num=set_num,
            bag=int(bag),
            crop_id=crop_id,
            desktop_overlays=not fast_map_flag,
        )
    finally:
        if temp_crop_path is not None:
            try:
                temp_crop_path.unlink(missing_ok=True)
            except Exception:
                pass

    # Optional SAM refinement: runs only when sam_refine=1 is in the request.
    # The core extraction result is never modified; SAM adds two optional fields.
    if sam_refine_flag:
        for slot in list(result.get("slots") or []):
            if str(slot.get("status") or "") != "masked":
                continue
            cutout = str(slot.get("part_cutout_path") or "")
            if not cutout:
                slot["sam_refine_status"] = "skipped_no_cutout"
                slot["sam_refined_path"] = ""
                continue
            refine_result = refine_slot_cutout_with_sam(
                cutout,
                set_num=set_num,
                bag=int(bag),
                crop_id=crop_id,
                slot_index=int(slot.get("slot_index", 0)),
            )
            slot["sam_refined_path"] = refine_result.get("sam_refined_path", "")
            slot["sam_refine_status"] = refine_result.get("sam_refine_status", "")

    # Confirmed-label memory: auto-predict parts for new slots from prior confirmed labels.
    # Runs after extraction and SAM (if any); never overwrites confirmed parts — that check
    # happens on the frontend using crop.parts.
    _apply_confirmed_memory_predictions(list(result.get("slots") or []), set_num, int(bag))

    return {
        "ok": bool(result.get("ok")),
        "crop_id": crop_id,
        "slots": list(result.get("slots") or []),
        "slot_count": int(result.get("slot_count") or 0),
        "full_crop_mask_path": str(result.get("full_crop_mask_path") or ""),
        "full_crop_mask_overlay_path": str(result.get("full_crop_mask_overlay_path") or ""),
        "full_crop_mask_error": str(result.get("full_crop_mask_error") or ""),
        "generated_at": str(result.get("generated_at") or ""),
        "error": str(result.get("error") or ""),
    }


@router.post("/debug/slot-mask-candidates")
async def slot_mask_candidates(req: Request):
    data = await req.json()
    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    bag = _coerce_int(data.get("bag"))
    crop_id = str(data.get("crop_id") or "").strip()
    slot_index = _coerce_int(data.get("slot_index"))
    part_cutout_path = str(data.get("part_cutout_path") or "").strip()
    shape_mask_path = str(data.get("shape_mask_path") or "").strip()
    _raw_clip_k = _coerce_int(data.get("clip_k"))
    clip_k = max(1, min(200, _raw_clip_k)) if _raw_clip_k is not None and _raw_clip_k > 0 else 5
    if bag is None or bag < 1:
        bag = 1
    if slot_index is None or slot_index < 0:
        raise HTTPException(status_code=400, detail="slot_index is required")
    if not part_cutout_path:
        raise HTTPException(status_code=400, detail="part_cutout_path is required")

    query_profile = _slot_mask_query_profile(part_cutout_path, shape_mask_path)
    if query_profile is None:
        raise HTTPException(status_code=400, detail="slot mask/cutout unavailable")

    candidate_rows, pool_source = _slot_mask_candidate_pool(set_num, int(bag))
    color_ids = [int(part.get("color_id", 0) or 0) for part in candidate_rows]
    color_bgr_by_id = {
        int(item["color_id"]): _slot_mask_hex_to_bgr(item.get("rgb"))
        for item in _load_catalog_colors_for_ids(color_ids)
    }
    color_bgr_by_id = {
        color_id: bgr
        for color_id, bgr in color_bgr_by_id.items()
        if bgr is not None
    }

    ranked: List[Dict[str, Any]] = []
    for candidate in candidate_rows:
        part_num = str(candidate.get("part_num") or "").strip()
        color_id = _coerce_int(candidate.get("color_id"))
        if not part_num or color_id is None:
            continue
        scores = _slot_mask_score_candidate(query_profile, candidate, color_bgr_by_id)
        ranked.append(
            {
                "part_num": part_num,
                "color_id": int(color_id),
                "color_name": str(candidate.get("color_name") or "n/a"),
                "element_id": str(candidate.get("element_id") or ""),
                "image_url": str(candidate.get("img_url") or "").strip(),
                "image_path": str(_slot_mask_resolve_local_image_path(candidate.get("img_url")) or ""),
                "confidence": scores["confidence"],
                "score_breakdown": {
                    "colour": scores["colour"],
                    "aspect": scores["aspect"],
                    "silhouette": scores["silhouette"],
                    "candidate_image_available": bool(scores["candidate_image_available"]),
                },
            }
        )
    ranked.sort(
        key=lambda item: (
            float(item.get("confidence", 0.0) or 0.0),
            float((item.get("score_breakdown") or {}).get("colour", 0.0) or 0.0),
            str(item.get("part_num") or ""),
        ),
        reverse=True,
    )
    top = ranked[:clip_k]
    print(
        "[slot-mask-candidates] "
        f"set={set_num} bag={int(bag)} crop_id={crop_id} slot_index={int(slot_index)} "
        f"pool={pool_source} candidates={len(candidate_rows)} clip_k={clip_k} returned={len(top)}"
    )
    return {
        "ok": True,
        "set_num": set_num,
        "bag": int(bag),
        "crop_id": crop_id,
        "slot_index": int(slot_index),
        "candidate_pool_source": pool_source,
        "candidate_count": len(candidate_rows),
        "ranked_candidates": top,
    }


@router.post("/debug/manual-match-clip-suggest")
async def manual_match_clip_suggest(req: Request):
    data = await req.json()

    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    bag = _coerce_int(data.get("bag"))
    crop_id = str(data.get("crop_id") or "").strip()
    slot_index = _coerce_int(data.get("slot_index"))

    if bag is None or bag < 1:
        bag = 1
    if not crop_id:
        raise HTTPException(status_code=400, detail="crop_id is required")
    if slot_index is None or slot_index < 0:
        raise HTTPException(status_code=400, detail="slot_index is required")

    return {
        "ok": True,
        "set_num": set_num,
        "bag": int(bag),
        "crop_id": crop_id,
        "slot_index": int(slot_index),
        "suggestions": [],
        "debug": {
            "mode": "placeholder",
        },
    }


@router.post("/debug/buildability-clip-suggest")
async def buildability_clip_suggest(req: Request):
    data = await req.json()
    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    bag = _coerce_int(data.get("bag")) or 1
    crop_id = str(data.get("crop_id") or "").strip()
    slot_index = _coerce_int(data.get("slot_index"))
    if not crop_id:
        raise HTTPException(status_code=400, detail="crop_id is required")
    if slot_index is None or slot_index < 0:
        raise HTTPException(status_code=400, detail="slot_index is required")

    labels_payload = _load_existing_labels(_label_store_path(set_num, int(bag)))
    clip_memory_payload = _load_clip_memory(set_num, int(bag))
    crop = _load_crop_for_ai_snap(set_num, int(bag), crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="crop not found")

    saved_good_pairs: set[tuple[str, int]] = set()
    for saved_crop in list((labels_payload.get("crops") or {}).values()):
        if not isinstance(saved_crop, dict):
            continue
        if str(saved_crop.get("status") or "").strip().lower() != "good":
            continue
        for saved_part in list(saved_crop.get("parts", []) or []):
            if not isinstance(saved_part, dict):
                continue
            saved_part_num = str(saved_part.get("part_num") or "").strip()
            saved_color_id = _coerce_int(saved_part.get("color_id"))
            if not saved_part_num or saved_color_id is None:
                continue
            saved_good_pairs.add((saved_part_num, int(saved_color_id)))

    qty_token_boxes = [
        dict(item)
        for item in list(crop.get("qty_token_boxes", []) or [])
        if isinstance(item, dict)
    ]
    selected_qty_box = dict(qty_token_boxes[int(slot_index)]) if 0 <= int(slot_index) < len(qty_token_boxes) else None

    from tools.a2b_clip_match_probe import (
        _embed_images,
        _ensure_catalog_image_for_pair,
        _load_clip_model,
        _load_rgb_image,
        _sanitize_embeddings,
    )

    temp_crop_path: Optional[Path] = None
    try:
        temp_crop_path = _write_ai_snap_temp_crop_image(crop)
        if temp_crop_path is None:
            raise HTTPException(status_code=400, detail="crop image unavailable")

        rank_input_path = str(temp_crop_path)
        if selected_qty_box is not None:
            normalized_result = normalize_slot_crop_from_qty(str(temp_crop_path), selected_qty_box)
            if bool(normalized_result.get("ok")) and str(normalized_result.get("normalized_path") or "").strip():
                rank_input_path = str(normalized_result.get("normalized_path") or "").strip()

        query_image = _load_rgb_image(Path(rank_input_path))
        if query_image is None:
            raise HTTPException(status_code=400, detail="normalized crop unavailable")

        parts_payload = load_instruction_set_parts(set_num)
        parts = _prepare_instruction_parts_for_display(list(parts_payload.get("parts", []) or []))
        assigned_totals = _assigned_part_totals_from_labels(labels_payload)
        candidate_rows: List[Dict[str, Any]] = []
        for part in list(parts or []):
            part_num = str(part.get("part_num") or "").strip()
            color_id = int(part.get("color_id", 0) or 0)
            if not part_num:
                continue
            key = _candidate_part_key(part_num, color_id)
            required_qty = int(part.get("qty", 0) or 0)
            assigned_qty = int(assigned_totals.get(key, 0) or 0)
            remaining_qty = required_qty - assigned_qty
            if remaining_qty <= 0:
                continue
            image_path = _ensure_catalog_image_for_pair(part_num, color_id)
            if image_path is None or not image_path.exists():
                continue
            clip_image = _load_rgb_image(image_path)
            if clip_image is None:
                continue
            candidate_rows.append(
                {
                    "part_num": part_num,
                    "color_id": color_id,
                    "required_qty": required_qty,
                    "assigned_qty": assigned_qty,
                    "remaining_qty": remaining_qty,
                    "img_url": str(part.get("img_url") or "").strip(),
                    "image_path": str(image_path),
                    "clip_image": clip_image,
                }
            )

        if not candidate_rows:
            return {"ok": True, "mode": "clip-placeholder", "ranked_candidates": []}

        model, processor, device = _load_clip_model()
        _, query_vectors = _sanitize_embeddings([query_image], _embed_images([query_image], model, processor, device))
        candidate_rows, candidate_vectors = _sanitize_embeddings(
            candidate_rows,
            _embed_images([item["clip_image"] for item in candidate_rows], model, processor, device),
        )
        if query_vectors.size == 0 or not candidate_rows or candidate_vectors.size == 0:
            return {"ok": True, "mode": "clip-placeholder", "ranked_candidates": []}

        clip_memory_items = list(clip_memory_payload.get("items", []) or [])
        clip_memory_index = {
            (
                str(item.get("crop_id") or "").strip(),
                int(_coerce_int(item.get("slot_index")) or 0),
                str(item.get("part_num") or "").strip(),
                int(_coerce_int(item.get("color_id")) or 0),
            ): item
            for item in clip_memory_items
            if isinstance(item, dict)
        }
        clip_memory_dirty = False
        for saved_crop_id, saved_crop in dict(labels_payload.get("crops") or {}).items():
            if not isinstance(saved_crop, dict):
                continue
            if str(saved_crop.get("status") or "").strip().lower() != "good":
                continue
            saved_parts = [dict(part) for part in list(saved_crop.get("parts", []) or []) if isinstance(part, dict)]
            if not saved_parts:
                continue
            resolved_saved_crop = _load_crop_for_ai_snap(set_num, int(bag), str(saved_crop_id))
            if not resolved_saved_crop:
                continue
            saved_qty_boxes = [
                dict(item)
                for item in list(resolved_saved_crop.get("qty_token_boxes", []) or [])
                if isinstance(item, dict)
            ]
            for saved_slot_index, saved_part in enumerate(saved_parts):
                saved_part_num = str(saved_part.get("part_num") or "").strip()
                saved_color_id = _coerce_int(saved_part.get("color_id"))
                if not saved_part_num or saved_color_id is None:
                    continue
                memory_key = (str(saved_crop_id).strip(), int(saved_slot_index), saved_part_num, int(saved_color_id))
                if memory_key in clip_memory_index:
                    continue
                if not (0 <= int(saved_slot_index) < len(saved_qty_boxes)):
                    continue
                saved_temp_crop_path = _write_ai_snap_temp_crop_image(resolved_saved_crop)
                if saved_temp_crop_path is None:
                    continue
                try:
                    with tempfile.TemporaryDirectory(prefix="clip_memory_") as memory_dir:
                        normalized_saved = normalize_slot_crop_from_qty(
                            str(saved_temp_crop_path),
                            saved_qty_boxes[int(saved_slot_index)],
                            output_dir=memory_dir,
                        )
                        normalized_saved_path = str(normalized_saved.get("normalized_path") or "").strip()
                        if not bool(normalized_saved.get("ok")) or not normalized_saved_path:
                            continue
                        memory_image = _load_rgb_image(Path(normalized_saved_path))
                        if memory_image is None:
                            continue
                        _, memory_vector_rows = _sanitize_embeddings(
                            [memory_key],
                            _embed_images([memory_image], model, processor, device),
                        )
                        if memory_vector_rows.size == 0:
                            continue
                        memory_vector = np.asarray(memory_vector_rows, dtype=np.float32).reshape(-1)
                        memory_vector = np.nan_to_num(
                            memory_vector,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        ).astype(np.float32, copy=False)
                        if memory_vector.size == 0:
                            continue
                        memory_norm = float(np.sqrt(np.sum(memory_vector * memory_vector, dtype=np.float32)))
                        if not np.isfinite(memory_norm) or memory_norm <= 1e-8:
                            continue
                        memory_item = {
                            "crop_id": str(saved_crop_id).strip(),
                            "slot_index": int(saved_slot_index),
                            "part_num": saved_part_num,
                            "color_id": int(saved_color_id),
                            "embedding": memory_vector.tolist(),
                            "updated_at": _iso_now(),
                        }
                        clip_memory_items.append(memory_item)
                        clip_memory_index[memory_key] = memory_item
                        clip_memory_dirty = True
                finally:
                    try:
                        saved_temp_crop_path.unlink(missing_ok=True)
                    except Exception:
                        pass
        if clip_memory_dirty:
            clip_memory_payload["items"] = clip_memory_items
            clip_memory_payload["updated_at"] = _iso_now()
            _write_clip_memory(set_num, int(bag), clip_memory_payload)

        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        candidate_vectors = np.asarray(candidate_vectors, dtype=np.float32)
        candidate_count_before = int(candidate_vectors.shape[0])
        finite_mask = np.isfinite(candidate_vectors).all(axis=1)
        invalid_candidate_vectors_removed = int(candidate_count_before - np.count_nonzero(finite_mask))
        candidate_rows = [
            candidate
            for candidate, keep in zip(candidate_rows, finite_mask.tolist())
            if bool(keep)
        ]
        candidate_vectors = candidate_vectors[finite_mask]
        candidate_count_before = int(candidate_vectors.shape[0])

        query_vectors = np.nan_to_num(
            query_vectors, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
        candidate_vectors = np.nan_to_num(
            candidate_vectors, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
        finite_candidate_count = int(np.count_nonzero(np.isfinite(candidate_vectors).all(axis=1)))

        query_norms = np.sqrt(np.sum(query_vectors * query_vectors, axis=1, dtype=np.float32)).astype(
            np.float32, copy=False
        )
        candidate_norms = np.sqrt(
            np.sum(candidate_vectors * candidate_vectors, axis=1, dtype=np.float32)
        ).astype(np.float32, copy=False)
        if query_norms.size == 0 or float(query_norms[0]) <= 1e-8:
            print(
                "[buildability-clip-suggest] query_vector_norm=",
                float(query_norms[0]) if query_norms.size else None,
                "candidate_count_before=",
                candidate_count_before,
                "candidate_count_after=",
                0,
                "finite_candidate_count=",
                finite_candidate_count,
                "invalid_candidate_vectors_removed=",
                invalid_candidate_vectors_removed,
            )
            return {
                "ok": False,
                "mode": "clip-local-v1",
                "ranked_candidates": [],
                "error": "query_clip_vector_norm_too_small",
            }

        valid_candidate_mask = candidate_norms > 1e-8
        if not np.any(valid_candidate_mask):
            return {"ok": True, "mode": "clip-placeholder", "ranked_candidates": []}

        candidate_rows = [
            candidate
            for candidate, keep in zip(candidate_rows, valid_candidate_mask.tolist())
            if bool(keep)
        ]
        candidate_vectors = candidate_vectors[valid_candidate_mask]
        candidate_norms = candidate_norms[valid_candidate_mask]
        candidate_count_after = int(candidate_vectors.shape[0])

        query_vectors = query_vectors / np.maximum(query_norms[:, None], np.float32(1e-8))
        candidate_vectors = candidate_vectors / np.maximum(
            candidate_norms[:, None], np.float32(1e-8)
        )
        query_vectors = np.nan_to_num(query_vectors, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )
        candidate_vectors = np.nan_to_num(
            candidate_vectors, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
        print(
            "[buildability-clip-suggest] query_vector_norm=",
            float(query_norms[0]),
            "candidate_count_before=",
            candidate_count_before,
            "candidate_count_after=",
            candidate_count_after,
            "finite_candidate_count=",
            finite_candidate_count,
            "invalid_candidate_vectors_removed=",
            invalid_candidate_vectors_removed,
        )
        similarity = query_vectors @ candidate_vectors.T
        memory_similarity_by_pair: Dict[tuple[str, int], float] = {}
        memory_vectors_list: List[np.ndarray] = []
        memory_rows: List[Dict[str, Any]] = []
        query_dim = int(query_vectors.shape[1]) if query_vectors.ndim == 2 else 0
        for item in clip_memory_items:
            if not isinstance(item, dict):
                continue
            embedding = np.asarray(item.get("embedding", []), dtype=np.float32).reshape(-1)
            if embedding.size != query_dim:
                continue
            memory_vectors_list.append(embedding)
            memory_rows.append(item)
        if memory_vectors_list and memory_rows:
            memory_vectors = np.asarray(memory_vectors_list, dtype=np.float32)
            memory_vectors = np.nan_to_num(memory_vectors, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32, copy=False
            )
            memory_finite_mask = np.isfinite(memory_vectors).all(axis=1)
            memory_rows = [
                item
                for item, keep in zip(memory_rows, memory_finite_mask.tolist())
                if bool(keep)
            ]
            memory_vectors = memory_vectors[memory_finite_mask]
            if memory_rows and memory_vectors.size:
                memory_norms = np.sqrt(
                    np.sum(memory_vectors * memory_vectors, axis=1, dtype=np.float32)
                ).astype(np.float32, copy=False)
                valid_memory_mask = memory_norms > 1e-8
                memory_rows = [
                    item
                    for item, keep in zip(memory_rows, valid_memory_mask.tolist())
                    if bool(keep)
                ]
                memory_vectors = memory_vectors[valid_memory_mask]
                memory_norms = memory_norms[valid_memory_mask]
                if memory_rows and memory_vectors.size:
                    memory_vectors = memory_vectors / np.maximum(
                        memory_norms[:, None], np.float32(1e-8)
                    )
                    memory_vectors = np.nan_to_num(
                        memory_vectors, nan=0.0, posinf=0.0, neginf=0.0
                    ).astype(np.float32, copy=False)
                    memory_similarity = query_vectors @ memory_vectors.T
                    for idx, memory_item in enumerate(memory_rows):
                        pair = (
                            str(memory_item.get("part_num") or "").strip(),
                            int(_coerce_int(memory_item.get("color_id")) or 0),
                        )
                        score = float(memory_similarity[0, int(idx)])
                        if pair not in memory_similarity_by_pair or score > memory_similarity_by_pair[pair]:
                            memory_similarity_by_pair[pair] = score
        scored_candidates: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(candidate_rows):
            clip_score = float(similarity[0, int(idx)])
            pair = (
                str(candidate.get("part_num") or "").strip(),
                int(candidate.get("color_id", 0) or 0),
            )
            good_label_boost = 0.08 if pair in saved_good_pairs else 0.0
            memory_similarity_score = float(memory_similarity_by_pair.get(pair, 0.0))
            memory_bank_boost = min(0.12, max(0.0, memory_similarity_score - 0.85) * 0.6)
            boosted_score = clip_score + good_label_boost + memory_bank_boost
            scored_candidates.append(
                {
                    "candidate": candidate,
                    "clip_score": clip_score,
                    "good_label_boost": good_label_boost,
                    "boosted_score": boosted_score,
                }
            )
        scored_candidates.sort(key=lambda item: item["boosted_score"], reverse=True)
        top_candidates = scored_candidates[: min(10, len(scored_candidates))]
        ranked_candidates = [
            {
                "rank": int(index + 1),
                "part_num": str(item["candidate"].get("part_num") or ""),
                "color_id": int(item["candidate"].get("color_id", 0) or 0),
                "required_qty": int(item["candidate"].get("required_qty", 0) or 0),
                "assigned_qty": int(item["candidate"].get("assigned_qty", 0) or 0),
                "remaining_qty": int(item["candidate"].get("remaining_qty", 0) or 0),
                "score": round(float(item["boosted_score"]), 4),
                "image_url": str(item["candidate"].get("img_url") or ""),
                "image_path": str(item["candidate"].get("image_path") or ""),
                "clip_score": round(float(item["clip_score"]), 4),
                "good_label_boost": round(float(item["good_label_boost"]), 4),
                "boosted_score": round(float(item["boosted_score"]), 4),
            }
            for index, item in enumerate(top_candidates)
        ]
    finally:
        if temp_crop_path is not None:
            try:
                temp_crop_path.unlink(missing_ok=True)
            except Exception:
                pass
    return {
        "ok": True,
        "mode": "clip-local-v1",
        "ranked_candidates": ranked_candidates,
    }


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


@router.get("/debug/normalize-part-crop", response_class=HTMLResponse)
def normalize_part_crop_debug(image_path: str = Query(...)):
    result = normalize_part_crop(image_path)

    def _file_to_data_uri(path_text: str) -> str:
        path = Path(str(path_text or "").strip())
        if not path.exists() or not path.is_file():
            return ""
        suffix = path.suffix.lower()
        mime_type = "image/png" if suffix == ".png" else ("image/jpeg" if suffix in {".jpg", ".jpeg"} else "application/octet-stream")
        return "data:%s;base64,%s" % (
            mime_type,
            base64.b64encode(path.read_bytes()).decode("ascii"),
        )

    original_data_uri = _file_to_data_uri(str(result.get("original_path") or ""))
    mask_data_uri = _file_to_data_uri(str(result.get("mask_path") or ""))
    normalized_data_uri = _file_to_data_uri(str(result.get("normalized_path") or ""))
    debug_json = json.dumps(
        {
            "ok": bool(result.get("ok")),
            "original_path": str(result.get("original_path") or ""),
            "normalized_path": str(result.get("normalized_path") or ""),
            "mask_path": str(result.get("mask_path") or ""),
            "box": result.get("box"),
            "debug": result.get("debug"),
        },
        indent=2,
    )

    html = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <title>Normalize Part Crop</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f4f7fb;
            color: #1d2a36;
          }}
          h1 {{
            margin: 0 0 14px;
            font-size: 24px;
          }}
          .meta {{
            margin-bottom: 16px;
            padding: 12px 14px;
            border-radius: 12px;
            background: #fff;
            border: 1px solid #d6dee8;
          }}
          .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
          }}
          .card {{
            background: #fff;
            border: 1px solid #d6dee8;
            border-radius: 14px;
            padding: 14px;
          }}
          .card h2 {{
            margin: 0 0 10px;
            font-size: 16px;
          }}
          img {{
            width: 100%;
            height: auto;
            display: block;
            background: #fff;
            border: 1px solid #d6dee8;
            border-radius: 10px;
          }}
          pre {{
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 12px;
            line-height: 1.45;
          }}
        </style>
      </head>
      <body>
        <h1>Normalize Part Crop</h1>
        <div class="meta">
          <div><strong>ok:</strong> {escape(str(bool(result.get("ok"))).lower())}</div>
          <div><strong>original:</strong> {escape(str(result.get("original_path") or ""))}</div>
          <div><strong>normalized:</strong> {escape(str(result.get("normalized_path") or ""))}</div>
          <div><strong>mask:</strong> {escape(str(result.get("mask_path") or ""))}</div>
        </div>
        <div class="grid">
          <div class="card">
            <h2>Original</h2>
            {f'<img src="{original_data_uri}" alt="Original crop" />' if original_data_uri else '<div>Original image unavailable</div>'}
          </div>
          <div class="card">
            <h2>Mask</h2>
            {f'<img src="{mask_data_uri}" alt="Foreground mask" />' if mask_data_uri else '<div>Mask image unavailable</div>'}
          </div>
          <div class="card">
            <h2>Normalized</h2>
            {f'<img src="{normalized_data_uri}" alt="Normalized crop" />' if normalized_data_uri else '<div>Normalized image unavailable</div>'}
          </div>
        </div>
        <div class="card">
          <h2>Box / Debug</h2>
          <pre>{escape(debug_json)}</pre>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)


@router.get("/debug/instruction-buildability", response_class=HTMLResponse)
def instruction_buildability(
    set_num: str = Query(...),
    bag: Optional[int] = Query(None, ge=1),
    ai: Optional[int] = Query(0),
    step: Optional[int] = Query(None),
    page: Optional[int] = Query(None, ge=1),
    rebuild: Optional[int] = Query(0),
    v: Optional[str] = Query(None),
    sam_refine: Optional[int] = Query(0),
    clip_k: Optional[int] = Query(None),
    fast_map: Optional[int] = Query(0),
):
    bag_number = int(bag or 1)
    parts_payload = load_instruction_set_parts(set_num)
    parts = _prepare_instruction_parts_for_display(list(parts_payload.get("parts", []) or []))
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
    crops = _build_instruction_callout_crops(
        str(set_num),
        bag_number,
        ai_enabled=int(ai or 0) == 1,
        step_filter=step,
        page_filter=page,
        rebuild=int(rebuild or 0) == 1,
    )
    manual_pages = _build_manual_crop_pages(str(set_num), bag_number)
    parts_by_key = {
        f"{str(part.get('part_num') or '').strip()}::{int(part.get('color_id', 0) or 0)}": part
        for part in parts
    }
    crop_ids_present = {str(crop.get("crop_id") or "").strip() for crop in crops}

    for crop in crops:
        saved_crop = dict(labels_payload.get("crops", {}).get(crop["crop_id"]) or {})
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
        if saved_crop_qty_text:
            crop["qty_text"] = saved_crop_qty_text
            crop["qty_numbers"] = saved_crop_qty
            crop["qty_label"] = ", ".join(saved_crop_qty_text) if saved_crop_qty_text else "none"
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
            "qty_text": _coerce_str_list(crop_dict.get("qty_text", [])),
            "qty_numbers": _coerce_int_list(crop_dict.get("qty", [])),
            "qty_label": ", ".join(_coerce_str_list(crop_dict.get("qty_text", []))) or "none",
            "qty_source": "local",
            "ai_part_count": None,
            "ai_issues": [],
            "source": "manual",
            "data_uri": data_uri,
            "coords_label": coords_label,
            "crop_box": crop_box or [],
            "crop_box_format": str(crop_dict.get("crop_box_format") or "xywh"),
            "crop_image_path": crop_image_path,
            "confidence": _coerce_float(crop_dict.get("confidence")),
            "qty_token_boxes": crop_dict.get("qty_token_boxes"),
            "edge_rect": crop_dict.get("edge_rect"),
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
        if step is not None and int(manual_crop.get("step", 0) or 0) != int(step):
            continue

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
                qty: <span id="qty-label-{escape(crop['crop_id'])}">{escape(str(crop['qty_label']))}</span><br/>
                qty source: {escape(str(crop.get('qty_source') or 'local'))}<br/>
                ai part_count: {escape('—' if crop.get('ai_part_count') is None else str(crop.get('ai_part_count')))}<br/>
                ai issues: {escape(', '.join(list(crop.get('ai_issues', []) or [])) or '—')}<br/>
                slots: <span id="slots-label-{escape(crop['crop_id'])}">{int(crop.get('slot_filled', 0) or 0)} / {int(crop.get('slot_total', 0) or 0)} filled</span><br/>
                next qty: <span id="next-qty-label-{escape(crop['crop_id'])}">{escape(str(crop.get('next_qty_label') or '1x'))}</span><br/>
                status: <span id="status-label-{escape(crop['crop_id'])}" class="crop-status-label">{escape(str(crop.get('status') or 'needs_adjust'))}</span><br/>
                metallic mode: <span id="metallic-label-{escape(crop['crop_id'])}" class="crop-status-label">OFF</span><br/>
                <span id="qty-warning-{escape(crop['crop_id'])}" class="crop-warning">{escape('No qty detected' if bool(crop.get('no_qty_detected')) else ('All qty slots filled' if bool(crop.get('slots_full')) else ''))}</span><br/>
                <span class="coords">{escape(str(crop['coords_label']))}</span><br/>
                debug crop_box: {escape(_debug_json_text(crop.get('crop_box')))}<br/>
                debug source: {escape(str(crop.get('source') or '—'))}<br/>
                debug step_number: {int(crop.get('step', 0) or 0)}<br/>
                debug page: {int(crop.get('page', 0) or 0)}<br/>
                debug qty token boxes: {escape(_debug_json_text(crop.get('qty_token_boxes')))}<br/>
                debug edge rect: {escape(_debug_json_text(crop.get('edge_rect')))}
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
    buildability_variant_json = json.dumps(str(v or "").strip())
    _sam_refine_flag = 1 if int(sam_refine or 0) == 1 else 0
    _clip_k_val = max(1, min(200, int(clip_k))) if clip_k is not None else 5
    _fast_map_flag = 1 if int(fast_map or 0) == 1 else 0
    sam_refine_json = json.dumps(_sam_refine_flag)
    clip_k_json = json.dumps(_clip_k_val)
    fast_map_json = json.dumps(_fast_map_flag)
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
          display: flex;
          flex-direction: column;
          gap: 16px;
          align-items: stretch;
        }}
        .picker-main {{
          display: flex;
          flex-direction: column;
          gap: 14px;
          width: 100%;
        }}
        .picker-controls {{
          display: flex;
          flex-direction: column;
          gap: 12px;
          width: 100%;
        }}
        .picker-canvas-wrap {{
          width: 100%;
          border: 1px solid #c9d9e8;
          border-radius: 18px;
          overflow: hidden;
          background: linear-gradient(180deg, #d9eefc 0%, #c5e4fa 100%);
          min-height: 260px;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 16px;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
        }}
        .picker-canvas {{
          display: block;
          width: 100%;
          max-width: 760px;
          height: auto;
          border-radius: 16px;
          background: #cfe7f8;
          cursor: crosshair;
          pointer-events: auto;
        }}
        .picker-slots-panel {{
          width: 100%;
          border: 1px solid #d6dee8;
          border-radius: 16px;
          background: #fff;
          padding: 14px;
          box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
        }}
        .picker-slot-header {{
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          margin-bottom: 10px;
        }}
        .picker-slot-title {{
          font-size: 16px;
          font-weight: 800;
          color: #17212b;
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
        .suggested-part.ai-snap-hit {{
          border-color: #b7791f;
          background: #fff8e7;
          box-shadow: 0 0 0 2px rgba(183, 121, 31, 0.12);
        }}
        .suggested-part .part-thumb {{
          margin-top: 0;
          min-height: 120px;
        }}
        .ai-snap-badge {{
          display: inline-flex;
          align-items: center;
          gap: 6px;
          margin-bottom: 8px;
          padding: 4px 9px;
          border-radius: 999px;
          background: #f6e7bd;
          color: #7a4a08;
          font-size: 12px;
          font-weight: 700;
        }}
        .ai-snap-confidence {{
          color: #9b5b09;
          font-weight: 800;
        }}
        .ai-snap-debug {{
          margin-top: 8px;
          display: flex;
          gap: 10px;
          align-items: flex-start;
          flex-wrap: wrap;
        }}
        .ai-snap-debug-figure {{
          margin: 0;
          font-size: 11px;
          color: #627283;
        }}
        .ai-snap-debug-figure img {{
          display: block;
          width: 82px;
          max-height: 82px;
          object-fit: contain;
          border: 1px solid #d6dee8;
          border-radius: 8px;
          background-color: #fff;
          background-image:
            linear-gradient(45deg, #edf1f5 25%, transparent 25%),
            linear-gradient(-45deg, #edf1f5 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #edf1f5 75%),
            linear-gradient(-45deg, transparent 75%, #edf1f5 75%);
          background-size: 14px 14px;
          background-position: 0 0, 0 7px, 7px -7px, -7px 0;
        }}
        .ai-snap-debug-figure figcaption {{
          margin-top: 4px;
          max-width: 82px;
          word-break: break-word;
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
        .picker-slot-list {{
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          align-items: stretch;
        }}
        .picker-slot-toolbar {{
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 10px;
          justify-content: flex-end;
        }}
        .picker-slot-btn {{
          border: 1px solid #d6dee8;
          border-radius: 14px;
          background: #f8fbff;
          padding: 12px 16px;
          text-align: center;
          font-size: 13px;
          font-weight: 600;
          line-height: 1.25;
          color: #2f4153;
          cursor: default;
          min-width: 112px;
          min-height: 72px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 6px;
        }}
        .picker-slot-btn.selected {{
          border-color: #2f6fed;
          background: #eef4ff;
          color: #1947a6;
          box-shadow: 0 0 0 2px rgba(47, 111, 237, 0.12);
        }}
        .picker-slot-btn.assigned {{
          background: #eef6ef;
          border-color: #7db28a;
          color: #2f6c41;
        }}
        .picker-slot-mask {{
          margin-top: 4px;
          display: flex;
          justify-content: center;
        }}
        .picker-slot-mask img {{
          width: 54px;
          height: 54px;
          object-fit: contain;
          border: 1px solid #d6dee8;
          border-radius: 8px;
          background-color: #fff;
          background-image:
            linear-gradient(45deg, #edf1f5 25%, transparent 25%),
            linear-gradient(-45deg, #edf1f5 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #edf1f5 75%),
            linear-gradient(-45deg, transparent 75%, #edf1f5 75%);
          background-size: 12px 12px;
          background-position: 0 0, 0 6px, 6px -6px, -6px 0;
        }}
        .picker-slot-review {{
          margin-top: 4px;
          color: #8a5a00;
          font-size: 11px;
        }}
        .picker-slot-candidates {{
          display: grid;
          grid-template-columns: repeat(5, minmax(34px, 1fr));
          gap: 4px;
          margin-top: 6px;
          width: 100%;
        }}
        .picker-slot-candidate {{
          appearance: none;
          border: 0;
          background: transparent;
          padding: 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 2px;
          min-width: 0;
          color: #536576;
          font-size: 9px;
          font-weight: 700;
          line-height: 1.1;
        }}
        .picker-slot-candidate img {{
          width: 30px;
          height: 30px;
          object-fit: contain;
          border: 1px solid #d6dee8;
          border-radius: 6px;
          background: #fff;
        }}
        .picker-slot-candidate span {{
          max-width: 42px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }}
        .picker-slot-debug {{
          margin-top: 6px;
          color: #6c7c8d;
          font-size: 9px;
          font-weight: 600;
          line-height: 1.2;
          text-align: left;
          word-break: break-word;
          max-width: 150px;
        }}
        .picker-slot-color {{
          display: block;
          font-size: 10px;
          color: #555;
          margin-top: 2px;
          line-height: 1.3;
        }}
        .picker-slot-color-swatch {{
          display: inline-block;
          width: 10px;
          height: 10px;
          border: 1px solid #999;
          border-radius: 2px;
          vertical-align: middle;
          margin-right: 3px;
        }}
        .picker-slot-low-alpha {{
          display: block;
          background: #fff3cd;
          color: #856404;
          font-size: 10px;
          padding: 1px 4px;
          border-radius: 3px;
          margin-top: 2px;
        }}
        .picker-slot-confidence {{
          display: block;
          font-size: 10px;
          font-weight: 600;
          margin-top: 2px;
        }}
        .picker-slot-confidence-high {{ color: #155724; }}
        .picker-slot-confidence-medium {{ color: #856404; }}
        .picker-slot-confidence-low {{ color: #721c24; }}
        .picker-slot-btn.predicted {{
          border-color: #fd7e14;
        }}
        .picker-slot-predicted {{
          display: block;
          margin-top: 4px;
          padding: 3px 5px;
          background: #fff3cd;
          border: 1px solid #ffc107;
          border-radius: 3px;
          color: #856404;
          font-size: 0.7em;
          line-height: 1.3;
          word-break: break-all;
        }}
        .picker-slot-predicted-actions {{
          display: flex;
          gap: 4px;
          margin-top: 3px;
        }}
        .predicted-accept-btn,
        .predicted-reject-btn {{
          border: 1px solid transparent;
          border-radius: 3px;
          cursor: pointer;
          font-size: 0.72em;
          padding: 2px 7px;
        }}
        .predicted-accept-btn {{
          background: #d4edda;
          border-color: #c3e6cb;
          color: #155724;
        }}
        .predicted-accept-btn:hover {{ background: #c3e6cb; }}
        .predicted-reject-btn {{
          background: #f8d7da;
          border-color: #f5c6cb;
          color: #721c24;
        }}
        .predicted-reject-btn:hover {{ background: #f5c6cb; }}
        .picker-slot-name {{
          font-size: 12px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.02em;
        }}
        .picker-slot-qty {{
          font-size: 22px;
          font-weight: 900;
          line-height: 1;
        }}
        .picker-slot-empty {{
          color: #6c7c8d;
          font-size: 12px;
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
          <div class="colour-picker-panel">
            <h3>Colour Picker</h3>
            <p class="colour-picker-help">Click the selected crop image to sample RGB and override the automatic colour filter.</p>
            <div class="picker-layout">
              <div class="picker-main">
                <div class="picker-canvas-wrap">
                  <canvas id="colour-picker-canvas" class="picker-canvas"></canvas>
                  <div id="colour-picker-empty" class="picker-empty">Select a crop with an image to start sampling colours.</div>
                </div>
                <div class="picker-slots-panel">
                  <div class="picker-slot-header">
                    <div>
                      <div class="picker-slot-title">Detected slots in crop</div>
                      <div class="save-note">Select the slot you are filling. AI Snap ranks candidates for the current open slot.</div>
                    </div>
                    <div class="picker-slot-toolbar">
                      <button type="button" class="remove-btn" id="ai-snap-btn" onclick="runAiSnap()">AI Snap</button>
                      <button type="button" class="remove-btn" id="auto-mask-slots-btn" onclick="runAutoMaskSlots()">Auto Mask Slots</button>
                      <button type="button" class="remove-btn" id="next-unfilled-btn" onclick="goToNextUnfilledCrop()">Next Unfilled</button>
                      <div id="ai-snap-status" class="save-note"></div>
                    </div>
                  </div>
                  <div id="picker-slot-list" class="picker-slot-list">
                    <div class="picker-slot-empty">Select a crop to view qty slots.</div>
                  </div>
                </div>
              </div>
              <div class="picker-controls">
                <div id="picked-rgb-row" class="picked-rgb-row">
                  <span class="colour-swatch" id="picked-rgb-swatch" style="background: transparent;"></span>
                  <span id="picked-rgb-text" class="save-note">No colour sampled yet.</span>
                  <button type="button" class="remove-btn" id="save-manual-calibration-btn" onclick="saveManualColorCalibration()" disabled>Save calibration</button>
                  <button type="button" class="remove-btn" onclick="clearManualColorFilter()">Clear colour filter</button>
                </div>
                <div id="manual-calibration-status" class="save-note"></div>
                <div class="colour-picker-actions">
                  <strong>Manual colour filter</strong>
                  <button id="manual-colours-toggle" type="button" class="remove-btn" onclick="toggleShowAllManualColours()">Show all colours</button>
                </div>
                <div id="colour-match-list" class="colour-match-list">
                  <div class="suggested-empty">Pick a colour from the crop to see the closest LEGO colours.</div>
                </div>
                <div id="picker-diagnostics" class="picker-diagnostics"></div>
              </div>
            </div>
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
        const buildabilityVariant = {buildability_variant_json};
        const SHOW_SLOT_MATCHES = false;
        const SAM_REFINE = {sam_refine_json};
        const SLOT_MATCH_K = {clip_k_json};
        const FAST_MAP = {fast_map_json};
        const manualSelections = new Map();
        const partTiles = Array.from(document.querySelectorAll(".part-tile"));
        window.legoColors = {lego_colors_json};
        window.trainingExamples = {training_examples_json};
        const colors = window.legoColors || [];
        const colorNameById = new Map();
        let activeCropId = null;
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
          const filledBySlotIndex = new Set();
          parts.forEach((part) => {{
            const explicitSlotIndex = Number(part && part.selected_slot_index);
            if (Number.isInteger(explicitSlotIndex) && explicitSlotIndex >= 0) {{
              filledBySlotIndex.add(explicitSlotIndex);
              return;
            }}
            const signature = qtySlotSignature(part.qty, part.qty_text);
            if (!signature) {{
              return;
            }}
            assignedCounts.set(signature, Number(assignedCounts.get(signature) || 0) + 1);
          }});

          const consumedCounts = new Map();
          let filledSlots = 0;
          let nextSlot = null;
          for (let currentIndex = 0; currentIndex < sequence.length; currentIndex += 1) {{
            const slot = sequence[currentIndex];
            if (filledBySlotIndex.has(currentIndex)) {{
              filledSlots += 1;
              continue;
            }}
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

        function currentBuildabilitySlotIndex(crop) {{
          const slotState = computeCropSlotState(crop);
          if (!crop || slotState.noQtyDetected || slotState.totalSlots <= 0 || slotState.slotsFull) {{
            return null;
          }}
          const explicitSlotIndex = Number(crop.ai_snap_selected_slot_index);
          if (
            Number.isInteger(explicitSlotIndex)
            && explicitSlotIndex >= 0
            && explicitSlotIndex < slotState.totalSlots
            && explicitSlotIndex >= slotState.filledSlots
          ) {{
            return explicitSlotIndex;
          }}
          return Math.max(0, Math.min(slotState.filledSlots, slotState.totalSlots - 1));
        }}

        function activeAiSnapResultForCrop(crop) {{
          if (!crop || !crop.ai_snap_result) {{
            return null;
          }}
          const currentSlotIndex = currentBuildabilitySlotIndex(crop);
          if (currentSlotIndex === null) {{
            return null;
          }}
          return Number(crop.ai_snap_result.slot_index) === Number(currentSlotIndex)
            ? crop.ai_snap_result
            : null;
        }}

        function aiSnapArtifactUrl(pathValue, cacheKey = "") {{
          const path = String(pathValue || "").trim();
          if (!path) {{
            return "";
          }}
          const version = String(cacheKey || path).trim();
          return "/debug/ai-snap-artifact?path=" + encodeURIComponent(path)
            + "&v=" + encodeURIComponent(version || String(Date.now()));
        }}

        function pathBasename(pathValue) {{
          const parts = String(pathValue || "").split(/[\\\\/]/).filter(Boolean);
          return parts.length ? parts[parts.length - 1] : "";
        }}

        function formatDebugBox(box) {{
          if (Array.isArray(box)) {{
            return box.map((value) => Number(value) || 0).join(",");
          }}
          if (box && typeof box === "object") {{
            return [
              Number(box.x) || 0,
              Number(box.y) || 0,
              Number(box.w) || 0,
              Number(box.h) || 0
            ].join(",");
          }}
          return "";
        }}

        function renderSlotMaskDebug(maskSlot) {{
          if (FAST_MAP) {{
            return "";
          }}
          if (!maskSlot) {{
            return "";
          }}
          const generatedAt = String(maskSlot.generated_at || "");
          const cutoutBase = String(maskSlot.cutout_basename || pathBasename(maskSlot.part_cutout_path));
          const sourceBase = String(maskSlot.source_mask_basename || pathBasename(maskSlot.source_mask_path));
          const cropBase = String(maskSlot.source_crop_basename || pathBasename(maskSlot.source_crop_path));
          const overlayBase = String(maskSlot.slot_window_overlay_basename || pathBasename(maskSlot.slot_window_overlay_path));
          return '<span class="picker-slot-debug">'
            + 'slot_index: ' + escapeHtml(String(maskSlot.slot_index ?? "")) + '<br/>'
            + 'path: ' + escapeHtml(String(maskSlot.function_path_used || "")) + '<br/>'
            + 'source crop: ' + escapeHtml(cropBase) + '<br/>'
            + 'master mask: ' + escapeHtml(String(maskSlot.master_mask_basename || sourceBase)) + '<br/>'
            + 'qty_box: ' + escapeHtml(formatDebugBox(maskSlot.qty_box || maskSlot.qty_token_box)) + '<br/>'
            + 'slot_window: ' + escapeHtml(formatDebugBox(maskSlot.slot_window || maskSlot.slot_crop_box)) + '<br/>'
            + 'cutout: ' + escapeHtml(cutoutBase) + '<br/>'
            + 'cutout size: ' + escapeHtml(formatDebugBox(maskSlot.actual_saved_cutout_size)) + '<br/>'
            + 'alpha bbox: ' + escapeHtml(formatDebugBox(maskSlot.non_transparent_bbox)) + '<br/>'
            + 'alpha pixels: ' + escapeHtml(String(maskSlot.alpha_pixel_count ?? "")) + '<br/>'
            + 'generated_at: ' + escapeHtml(generatedAt) + '<br/>'
            + 'using_master_mask: ' + escapeHtml(String(Boolean(maskSlot.using_master_mask))) + '<br/>'
            + 'overwritten: ' + escapeHtml(String(Boolean(maskSlot.existing_file_overwritten))) + '<br/>'
            + 'reused: ' + escapeHtml(String(Boolean(maskSlot.existing_file_reused))) + '<br/>'
            + 'candidate before save: ' + escapeHtml(String(Boolean(maskSlot.candidate_matching_started_before_cutout_save))) + '<br/>'
            + 'candidate after save: ' + escapeHtml(String(maskSlot.candidate_matching_started_after_cutout_save ?? "")) + '<br/>'
            + 'slot overlay: ' + escapeHtml(overlayBase)
            + '</span>';
        }}

        function renderSlotMaskCandidates(maskSlot) {{
          if (!SHOW_SLOT_MATCHES) {{
            return "";
          }}
          const candidates = Array.isArray(maskSlot && maskSlot.ranked_candidates)
            ? maskSlot.ranked_candidates
            : [];
          if (!candidates.length) {{
            if (maskSlot && maskSlot.candidates_loading) {{
              return '<span class="picker-slot-review">matching...</span>';
            }}
            if (maskSlot && maskSlot.candidates_error) {{
              return '<span class="picker-slot-review">' + escapeHtml(String(maskSlot.candidates_error)) + '</span>';
            }}
            return "";
          }}
          return '<span class="picker-slot-candidates">' + candidates.slice(0, SLOT_MATCH_K).map((candidate) => {{
            const imageUrl = String(candidate && (candidate.image_url || candidate.img_url || "") || "");
            const confidence = Number(candidate && candidate.confidence);
            const label = String(candidate && candidate.part_num || "") + ":" + String(candidate && candidate.color_id || "");
            return '<span role="button" tabindex="0" class="picker-slot-candidate" data-slot-suggestion="true" data-slot-index="' + escapeHtml(String(maskSlot.slot_index ?? "")) + '" data-part-num="' + escapeHtml(String(candidate && candidate.part_num || "")) + '" data-color-id="' + escapeHtml(String(candidate && candidate.color_id || 0)) + '" data-element-id="' + escapeHtml(String(candidate && candidate.element_id || "")) + '" data-color-name="' + escapeHtml(String(candidate && candidate.color_name || "")) + '" title="' + escapeHtml(label + (Number.isFinite(confidence) ? " " + confidence.toFixed(2) : "")) + '">'
              + (imageUrl ? '<img src="' + escapeHtml(imageUrl) + '" alt="' + escapeHtml(label) + '" loading="lazy" />' : '')
              + '<span>' + escapeHtml(label) + '</span>'
              + '</span>';
          }}).join("") + '</span>';
        }}

        function confirmedPartForSlot(crop, slotIndex) {{
          const parts = Array.isArray(crop && crop.parts) ? crop.parts : [];
          const explicit = parts.find((part) => Number(part && part.selected_slot_index) === Number(slotIndex));
          if (explicit) {{
            return explicit;
          }}
          return Number.isInteger(Number(slotIndex)) && Number(slotIndex) >= 0 && Number(slotIndex) < parts.length
            ? parts[Number(slotIndex)]
            : null;
        }}

        function renderFullCropMaskDebug(crop) {{
          const cacheKey = String(crop && crop.full_crop_mask_generated_at || "");
          const maskUrl = aiSnapArtifactUrl(crop && crop.full_crop_mask_path, cacheKey);
          const overlayUrl = aiSnapArtifactUrl(crop && crop.full_crop_mask_overlay_path, cacheKey);
          if (!maskUrl && !overlayUrl) {{
            return "";
          }}
          return `
            <div class="ai-snap-debug">
              ${{maskUrl ? `
                <figure class="ai-snap-debug-figure">
                  <img src="${{escapeHtml(maskUrl)}}" alt="Full crop mask" loading="lazy" />
                  <figcaption>full crop mask</figcaption>
                </figure>
              ` : ""}}
              ${{overlayUrl ? `
                <figure class="ai-snap-debug-figure">
                  <img src="${{escapeHtml(overlayUrl)}}" alt="Full crop mask overlay" loading="lazy" />
                  <figcaption>full crop mask overlay</figcaption>
                </figure>
              ` : ""}}
            </div>
          `;
        }}

        function renderAiSnapArtifactDebug(result) {{
          const debug = result && result.debug ? result.debug : {{}};
          const cacheKey = String(debug.generated_at || debug.ai_snap_input_path || "");
          const cutoutUrl = aiSnapArtifactUrl(debug.part_cutout_path, cacheKey);
          const maskUrl = aiSnapArtifactUrl(debug.shape_mask_path, cacheKey);
          if (!cutoutUrl && !maskUrl) {{
            return "";
          }}
          return `
            <div class="ai-snap-debug">
              <div class="ai-snap-debug-figure">
                <figcaption>
                  selected slot: ${{escapeHtml(String(debug.selected_slot_index ?? ""))}}<br/>
                  cutout slot: ${{escapeHtml(String(debug.cutout_slot_index ?? ""))}}<br/>
                  mask slot: ${{escapeHtml(String(debug.mask_slot_index ?? ""))}}
                </figcaption>
              </div>
              ${{cutoutUrl ? `
                <figure class="ai-snap-debug-figure">
                  <img src="${{escapeHtml(cutoutUrl)}}" alt="AI Snap part cutout" loading="lazy" />
                  <figcaption>part cutout</figcaption>
                </figure>
              ` : ""}}
              ${{maskUrl ? `
                <figure class="ai-snap-debug-figure">
                  <img src="${{escapeHtml(maskUrl)}}" alt="AI Snap shape mask" loading="lazy" />
                  <figcaption>shape mask</figcaption>
                </figure>
              ` : ""}}
            </div>
          `;
        }}

        function renderAiSnapStatus(crop) {{
          const button = document.getElementById("ai-snap-btn");
          const autoMaskButton = document.getElementById("auto-mask-slots-btn");
          const status = document.getElementById("ai-snap-status");
          if (!button || !status) {{
            return;
          }}
          const slotIndex = currentBuildabilitySlotIndex(crop);
          const aiSnapResult = activeAiSnapResultForCrop(crop);
          if (!crop) {{
            button.disabled = true;
            button.style.display = SHOW_SLOT_MATCHES ? "" : "none";
            if (autoMaskButton) {{
              autoMaskButton.disabled = true;
            }}
            button.textContent = "AI Snap";
            status.textContent = "Select a crop to rank the current open slot.";
            return;
          }}
          if (crop.ai_snap_loading) {{
            button.disabled = true;
            button.style.display = SHOW_SLOT_MATCHES ? "" : "none";
            if (autoMaskButton) {{
              autoMaskButton.disabled = true;
            }}
            button.textContent = "AI Snap...";
            status.textContent = "Ranking remaining candidates for this slot...";
            return;
          }}
          button.textContent = "AI Snap";
          button.style.display = SHOW_SLOT_MATCHES ? "" : "none";
          button.disabled = slotIndex === null;
          if (autoMaskButton) {{
            autoMaskButton.disabled = Boolean(crop.auto_mask_loading) || !Array.isArray(buildClientQtySequence(crop)) || buildClientQtySequence(crop).length === 0;
            autoMaskButton.textContent = crop.auto_mask_loading ? "Masking..." : "Auto Mask Slots";
          }}
          if (slotIndex === null) {{
            status.textContent = "No open slot available for AI Snap.";
            return;
          }}
          if (crop.ai_snap_error) {{
            status.innerHTML = escapeHtml(String(crop.ai_snap_error)) + renderFullCropMaskDebug(crop);
            return;
          }}
          if (SHOW_SLOT_MATCHES && aiSnapResult && Array.isArray(aiSnapResult.ranked_candidates)) {{
            status.innerHTML = escapeHtml(
              "AI Snap ranked " + aiSnapResult.ranked_candidates.length + " candidates for slot " + (slotIndex + 1)
                + (aiSnapResult.model ? " using " + aiSnapResult.model : "") + "."
            ) + renderAiSnapArtifactDebug(aiSnapResult) + renderFullCropMaskDebug(crop);
            return;
          }}
          status.innerHTML = escapeHtml("Run AI Snap for slot " + (slotIndex + 1) + ".") + renderFullCropMaskDebug(crop);
        }}

        function renderBuildabilitySlots(cropId) {{
          const list = document.getElementById("picker-slot-list");
          if (!list) {{
            return;
          }}
          const crop = cropId ? cropMap.get(cropId) : null;
          if (!crop) {{
            list.innerHTML = '<div class="picker-slot-empty">Select a crop to view qty slots.</div>';
            renderAiSnapStatus(null);
            return;
          }}
          const sequence = buildClientQtySequence(crop);
          const slotState = computeCropSlotState(crop);
          if (!sequence.length) {{
            list.innerHTML = '<div class="picker-slot-empty">No qty slots detected for this crop.</div>';
            renderAiSnapStatus(crop);
            return;
          }}
          const selectedIndex = currentBuildabilitySlotIndex(crop);
          const autoMaskSlots = new Map();
          (Array.isArray(crop.auto_mask_slots) ? crop.auto_mask_slots : []).forEach((slot) => {{
            autoMaskSlots.set(Number(slot && slot.slot_index), slot);
          }});
          // Returns the maskSlot if a confirmed-memory prediction is available and
          // no confirmed part has been saved for this slot yet.
          function _maskSlotPrediction(idx) {{
            const ms = autoMaskSlots.get(Number(idx));
            return (!confirmedPartForSlot(crop, idx) && ms && ms.predicted_part) ? ms : null;
          }}
          list.innerHTML = sequence.map((slot, idx) => `
            <button
              type="button"
              class="picker-slot-btn${{confirmedPartForSlot(crop, idx) ? " assigned" : ""}}${{_maskSlotPrediction(idx) ? " predicted" : ""}}${{selectedIndex !== null && idx === selectedIndex ? " selected" : ""}}"
              data-picker-slot-index="${{idx}}"
              title="${{confirmedPartForSlot(crop, idx) ? "Filled slot" : (_maskSlotPrediction(idx) ? "Predicted slot — accept or reject" : (selectedIndex !== null && idx === selectedIndex ? "Current assignment slot" : "Waiting for earlier slot"))}}"
            >
              <span class="picker-slot-name">Slot ${{idx + 1}}</span>
              <span class="picker-slot-qty">${{escapeHtml(String((slot && (slot.qty_text || slot.qty)) || "none"))}}</span>
              ${{confirmedPartForSlot(crop, idx) ? '<span class="picker-slot-review">confirmed: ' + escapeHtml(String(confirmedPartForSlot(crop, idx).part_num || "")) + ':' + escapeHtml(String(confirmedPartForSlot(crop, idx).color_id || "")) + '</span>' : ''}}
              ${{(() => {{
                const maskSlot = autoMaskSlots.get(Number(idx));
                const cutoutUrl = maskSlot ? aiSnapArtifactUrl(maskSlot.part_cutout_path, maskSlot.generated_at) : "";
                const slotOverlayUrl = maskSlot ? aiSnapArtifactUrl(maskSlot.slot_window_overlay_path, maskSlot.generated_at) : "";
                // Predicted badge — shown only when no confirmed part exists for this slot.
                const predSlot = _maskSlotPrediction(idx);
                const predictedHtml = predSlot
                  ? '<span class="picker-slot-predicted">'
                    + 'predicted: ' + escapeHtml(String(predSlot.predicted_part.part_num || ""))
                    + ':' + escapeHtml(String(predSlot.predicted_part.color_id || ""))
                    + ' • sim ' + escapeHtml(String((predSlot.prediction_similarity || 0).toFixed(2)))
                    + '<span class="picker-slot-predicted-actions">'
                    + '<button type="button" class="predicted-accept-btn" onclick="acceptPredictedSlot(event,' + idx + ')">Accept</button>'
                    + '<button type="button" class="predicted-reject-btn" onclick="rejectSlotPrediction(event,' + idx + ')">Reject</button>'
                    + '</span>'
                    + '</span>'
                  : '';
                let colorHtml = "";
                if (maskSlot && maskSlot.slot_rgb_median) {{
                  const [r, g, b] = maskSlot.slot_rgb_median;
                  const colorMatch = closestLegoColorId({{r, g, b}});
                  const colorName = (colorMatch && colorMatch.color_name) || String((colorMatch && colorMatch.color_id) || "");
                  const dist = (colorMatch && Number.isFinite(colorMatch.distance)) ? colorMatch.distance.toFixed(0) : "";
                  colorHtml = '<span class="picker-slot-color">'
                    + '<span class="picker-slot-color-swatch" style="background:rgb(' + r + ',' + g + ',' + b + ')"></span>'
                    + escapeHtml(colorName) + (dist ? " (" + dist + ")" : "")
                    + '</span>';
                }}
                let lowAlphaHtml = "";
                if (maskSlot && String(maskSlot.status || "") === "needs_review_low_alpha") {{
                  lowAlphaHtml = '<span class="picker-slot-low-alpha">⚠ low alpha</span>';
                }}
                let confidenceHtml = "";
                if (maskSlot && maskSlot.slot_confidence) {{
                  const conf = String(maskSlot.slot_confidence);
                  confidenceHtml = '<span class="picker-slot-confidence picker-slot-confidence-' + escapeHtml(conf) + '">Confidence: ' + escapeHtml(conf) + '</span>';
                }}
                if (cutoutUrl) {{
                  return '<span class="picker-slot-mask"><img src="' + escapeHtml(cutoutUrl) + '" alt="Slot ' + escapeHtml(String(idx + 1)) + ' cutout" loading="lazy" /></span>'
                    + (slotOverlayUrl ? '<span class="picker-slot-mask"><img src="' + escapeHtml(slotOverlayUrl) + '" alt="Slot ' + escapeHtml(String(idx + 1)) + ' window overlay" loading="lazy" /></span>' : '')
                    + renderSlotMaskDebug(maskSlot)
                    + renderSlotMaskCandidates(maskSlot)
                    + predictedHtml
                    + colorHtml + lowAlphaHtml + confidenceHtml;
                }}
                if (maskSlot && String(maskSlot.status || "") === "needs_review") {{
                  return '<span class="picker-slot-review">needs review</span>'
                    + (slotOverlayUrl ? '<span class="picker-slot-mask"><img src="' + escapeHtml(slotOverlayUrl) + '" alt="Slot ' + escapeHtml(String(idx + 1)) + ' window overlay" loading="lazy" /></span>' : '')
                    + renderSlotMaskDebug(maskSlot)
                    + predictedHtml
                    + colorHtml + lowAlphaHtml + confidenceHtml;
                }}
                return predictedHtml + colorHtml + lowAlphaHtml + confidenceHtml;
              }})()}}
            </button>
          `).join("");
          list.querySelectorAll("[data-picker-slot-index]").forEach((button) => {{
            button.addEventListener("click", () => {{
              const nextIndex = Number(button.dataset.pickerSlotIndex);
              if (!Number.isInteger(nextIndex) || nextIndex < slotState.filledSlots || nextIndex >= slotState.totalSlots) {{
                return;
              }}
              crop.ai_snap_selected_slot_index = nextIndex;
              if (crop.ai_snap_result && Number(crop.ai_snap_result.slot_index) !== Number(nextIndex)) {{
                crop.ai_snap_result = null;
              }}
              crop.ai_snap_error = "";
              renderBuildabilitySlots(crop.crop_id);
              renderSuggestedParts(crop.crop_id);
            }});
          }});
          list.querySelectorAll("[data-slot-suggestion]").forEach((el) => {{
            el.addEventListener("click", (event) => {{
              event.stopPropagation();
              acceptSlotSuggestion(event, el);
            }});
            el.addEventListener("keydown", (event) => {{
              if (event.key !== "Enter" && event.key !== " ") {{
                return;
              }}
              event.preventDefault();
              event.stopPropagation();
              acceptSlotSuggestion(event, el);
            }});
          }});
          renderAiSnapStatus(crop);
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
            selected_slot_index: Number.isInteger(Number(part.selected_slot_index)) ? Number(part.selected_slot_index) : null,
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
            updateAddAvailability();
            renderBuildabilitySlots(cropId);
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
              document.getElementById("save-status").textContent = "";
              updateAddAvailability();
              renderSuggestedParts(null);
              renderColourPicker(null);
              renderBuildabilitySlots(null);
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

        function nextUnfilledCropId(fromCropId) {{
          const ids = cropRecords
            .filter((c) => cropMap.has(c.crop_id) && isCropVisibleInCurrentView(c))
            .map((c) => c.crop_id);
          const start = ids.indexOf(String(fromCropId));
          for (let i = start + 1; i < ids.length; i++) {{
            const c = cropMap.get(ids[i]);
            if (!c) continue;
            const st = computeCropSlotState(c);
            if (!st.noQtyDetected && st.filledSlots < st.totalSlots) return ids[i];
          }}
          return null;
        }}

        function goToNextUnfilledCrop() {{
          const nxt = nextUnfilledCropId(activeCropId);
          if (nxt) selectCrop(nxt);
          else alert("No unfilled crops remaining.");
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
            const hasManualColorFilter = Number.isFinite(Number(crop && crop.manual_color_filter_id));
            status.textContent = (hasManualColorFilter
              ? "Manual colour filter for "
              : "Filtered ") + crop.crop_id + " to " + visibleCount + " parts matching: " + colorLabels.join(", ");
          }}
          updatePartFilterButtons();
        }}

        async function renderSuggestedParts(cropId) {{
          const panel = document.getElementById("suggested-parts-grid");
          const suggestedPanel = document.getElementById("suggested-parts-panel");
          if (suggestedPanel) {{
            suggestedPanel.style.display = SHOW_SLOT_MATCHES ? "" : "none";
          }}
          if (!panel) {{
            return;
          }}
          if (!SHOW_SLOT_MATCHES) {{
            panel.innerHTML = "";
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

          const aiSnapResult = activeAiSnapResultForCrop(crop);
          const aiSnapRankByKey = new Map();
          const aiSnapSuggestions = Array.isArray(aiSnapResult && aiSnapResult.ranked_candidates)
            ? aiSnapResult.ranked_candidates.map((candidate) => {{
                const key = partKey(candidate && candidate.part_num, candidate && candidate.color_id);
                aiSnapRankByKey.set(key, {{
                  rank: Number(candidate && candidate.rank || 0),
                  confidence: Number(candidate && candidate.confidence || 0),
                  model: String(aiSnapResult.model || ""),
                }});
                const meta = partMap.get(key) || {{}};
                return {{
                  part_num: String(candidate && candidate.part_num || ""),
                  color_id: Number(candidate && candidate.color_id || 0),
                  color_name: String(candidate && (candidate.color_name || meta.color_name) || "n/a"),
                  element_id: String(candidate && (candidate.element_id || meta.element_id) || ""),
                  img_url: String(candidate && (candidate.img_url || meta.img_url) || ""),
                  set_qty: Number(candidate && candidate.required_qty || meta.qty || 0),
                  assigned_qty: Number(candidate && candidate.assigned_qty || 0),
                  remaining_qty: Number(candidate && candidate.remaining_qty || 0),
                  ai_snap_rank: Number(candidate && candidate.rank || 0),
                  ai_snap_confidence: Number(candidate && candidate.confidence || 0),
                  ai_snap_model: String(aiSnapResult.model || ""),
                  suggestion_score: Number(candidate && candidate.mock_score || 0),
                }};
              }})
            : [];

          const mergedSuggestions = [];
          const seenSuggestionKeys = new Set();
          [...aiSnapSuggestions, ...suggestions].forEach((part) => {{
            const key = partKey(part && part.part_num, part && part.color_id);
            if (!key || seenSuggestionKeys.has(key)) {{
              return;
            }}
            const aiSnapMatch = aiSnapRankByKey.get(key);
            if (aiSnapMatch) {{
              part.ai_snap_rank = Number(aiSnapMatch.rank || 0);
              part.ai_snap_confidence = Number(aiSnapMatch.confidence || 0);
              part.ai_snap_model = String(aiSnapMatch.model || "");
            }}
            seenSuggestionKeys.add(key);
            mergedSuggestions.push(part);
          }});

          if (!mergedSuggestions.length) {{
            panel.innerHTML = '<div class="suggested-empty">No suggestions - use Show all.</div>';
            return;
          }}

          panel.innerHTML = mergedSuggestions.map((part) => {{
            const thumb = part.img_url
              ? '<img src="' + escapeHtml(part.img_url) + '" alt="' + escapeHtml(part.part_num) + '" loading="lazy" />'
              : '<div class="crop-missing">No image</div>';
            const hasAiSnap = Number.isFinite(Number(part.ai_snap_rank)) && Number(part.ai_snap_rank) > 0;
            const aiSnapBadge = hasAiSnap
              ? '<div class="ai-snap-badge">AI #' + escapeHtml(String(part.ai_snap_rank))
                + ' <span class="ai-snap-confidence">' + escapeHtml(Number(part.ai_snap_confidence || 0).toFixed(2)) + '</span></div>'
              : '';
            return `
              <div class="suggested-part${{hasAiSnap ? " ai-snap-hit" : ""}}">
                <div class="part-thumb">${{thumb}}</div>
                <div class="part-meta">
                  ${{aiSnapBadge}}
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
            empty.style.display = "block";
            list.innerHTML = '<div class="suggested-empty">Pick a colour from the crop to see the closest LEGO colours.</div>';
            pickedText.textContent = "No colour sampled yet.";
            pickedSwatch.style.background = "transparent";
            renderPickerDiagnostics(crop);
            renderBuildabilitySlots(cropId);
            updateSaveCalibrationUI(crop);
            return;
          }}

          const image = await ensureColourPickerImage(crop);
          if (!image) {{
            canvas.style.display = "none";
            empty.style.display = "block";
            list.innerHTML = '<div class="suggested-empty">Pick a colour from the crop to see the closest LEGO colours.</div>';
            renderPickerDiagnostics(crop);
            renderBuildabilitySlots(cropId);
            updateSaveCalibrationUI(crop);
            return;
          }}

          const maxWidth = 760;
          const scale = Math.min(2.2, Math.max(1, maxWidth / Math.max(1, image.naturalWidth)));
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
          canvas.style.display = "block";
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
            const hasManualColorFilter = Number.isFinite(Number(crop && crop.manual_color_filter_id));
            pickedText.textContent = hasManualColorFilter
              ? "Manual LEGO colour filter: " + (colorNameById.get(Number(crop.manual_color_filter_id)) || ("color " + crop.manual_color_filter_id))
              : "No colour sampled yet.";
            pickedSwatch.style.background = "transparent";
          }}

          renderColourMatches(crop);
          renderPickerDiagnostics(crop);
          renderBuildabilitySlots(cropId);
          updateSaveCalibrationUI(crop);
        }}

        function updateSaveCalibrationUI(crop) {{
          const button = document.getElementById("save-manual-calibration-btn");
          const status = document.getElementById("manual-calibration-status");
          if (!button || !status) {{
            return;
          }}
          const colorId = crop ? Number(crop.manual_color_filter_id) : NaN;
          const canSave = !!(
            crop
            && crop.picked_rgb
            && Number.isFinite(colorId)
          );
          button.disabled = !canSave;
          button.title = canSave
            ? "Save this sampled instruction colour as a manual LEGO colour calibration"
            : "Pick a crop colour and choose a LEGO colour first.";
          status.textContent = crop && crop.manual_calibration_status
            ? String(crop.manual_calibration_status)
            : "";
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
          crop.manual_calibration_status = "";
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
          delete crop.picked_sample_xy;
          delete crop.picked_sample_radius;
          crop.manual_calibration_status = "";
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

        async function saveManualColorCalibration() {{
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          if (!crop || !crop.picked_rgb) {{
            return;
          }}
          const colorId = Number(crop.manual_color_filter_id);
          if (!Number.isFinite(colorId)) {{
            crop.manual_calibration_status = "Choose a LEGO colour match first.";
            updateSaveCalibrationUI(crop);
            return;
          }}
          const button = document.getElementById("save-manual-calibration-btn");
          if (button) {{
            button.disabled = true;
          }}
          crop.manual_calibration_status = "Saving calibration...";
          updateSaveCalibrationUI(crop);
          const payload = {{
            set_num: {json.dumps(str(set_num))},
            page: crop.page,
            step: crop.step,
            crop_id: crop.crop_id,
            crop_image_path: crop.crop_image_path,
            sample_xy: crop.picked_sample_xy || {{ x: 0, y: 0 }},
            sample_radius: Number(crop.picked_sample_radius || 2),
            sample_rgb: {{
              r: Number(crop.picked_rgb.r || 0),
              g: Number(crop.picked_rgb.g || 0),
              b: Number(crop.picked_rgb.b || 0),
            }},
            color_id: colorId,
            color_name: colorNameById.get(colorId) || ("color " + colorId),
          }};
          try {{
            const res = await fetch("/debug/save-manual-color-calibration", {{
              method: "POST",
              headers: {{"Content-Type": "application/json"}},
              body: JSON.stringify(payload)
            }});
            if (!res.ok) {{
              let detail = "Calibration save failed";
              try {{
                const errorPayload = await res.json();
                detail = errorPayload.detail || detail;
              }} catch (_error) {{
                detail = "Calibration save failed";
              }}
              crop.manual_calibration_status = detail;
              updateSaveCalibrationUI(crop);
              return;
            }}
            const result = await res.json();
            const colorName = payload.color_name || ("color " + colorId);
            crop.manual_calibration_status = "Saved calibration: " + colorName + " (" + colorId + ")"
              + (result && Number.isFinite(Number(result.sample_count)) ? " | samples: " + Number(result.sample_count) : "");
          }} catch (_error) {{
            crop.manual_calibration_status = "Calibration save failed";
          }}
          updateSaveCalibrationUI(crop);
        }}

        async function runAiSnap() {{
          if (!SHOW_SLOT_MATCHES) {{
            return;
          }}
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          if (!crop) {{
            return;
          }}
          const slotIndex = currentBuildabilitySlotIndex(crop);
          if (slotIndex === null) {{
            crop.ai_snap_error = "No open slot available for AI Snap.";
            renderAiSnapStatus(crop);
            return;
          }}
          crop.ai_snap_loading = true;
          crop.ai_snap_error = "";
          renderAiSnapStatus(crop);
          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            crop_id: crop.crop_id,
            slot_index: slotIndex,
            manual_color_filter_id: Number.isFinite(Number(crop.manual_color_filter_id))
              ? Number(crop.manual_color_filter_id)
              : null,
            picked_rgb: crop.picked_rgb || null,
          }};
          const aiSnapEndpoint = buildabilityVariant === "realaisnap2"
            ? "/debug/buildability-clip-suggest"
            : "/debug/ai-rank-slot";
          try {{
            const res = await fetch(aiSnapEndpoint, {{
              method: "POST",
              headers: {{"Content-Type": "application/json"}},
              body: JSON.stringify(payload)
            }});
            if (!res.ok) {{
              let detail = "AI Snap failed";
              try {{
                const errorPayload = await res.json();
                detail = errorPayload.detail || detail;
              }} catch (_error) {{
                detail = "AI Snap failed";
              }}
              crop.ai_snap_error = detail;
              crop.ai_snap_result = null;
              renderAiSnapStatus(crop);
              return;
            }}
            const result = await res.json();
            if (buildabilityVariant === "realaisnap2" && Array.isArray(result && result.ranked_candidates)) {{
              result.model = String(result.model || "local-clip-probe-v1");
              result.ranked_candidates = result.ranked_candidates.map((candidate, index) => {{
                const key = partKey(candidate && candidate.part_num, candidate && candidate.color_id);
                const meta = partMap.get(key) || {{}};
                const remainingQty = Number(candidate && candidate.remaining_qty);
                const assignedQty = Number(candidate && candidate.assigned_qty);
                const requiredQty = Number(candidate && candidate.required_qty);
                const rawScore = Number(candidate && candidate.score);
                return {{
                  rank: Number(candidate && candidate.rank) || (index + 1),
                  part_num: String(candidate && candidate.part_num || ""),
                  color_id: Number(candidate && candidate.color_id || 0),
                  color_name: String(candidate && (candidate.color_name || meta.color_name) || "n/a"),
                  element_id: String(candidate && (candidate.element_id || meta.element_id) || ""),
                  img_url: String(candidate && (candidate.image_url || candidate.img_url || meta.img_url) || ""),
                  required_qty: Number.isFinite(requiredQty) ? requiredQty : Number(meta.qty || 0),
                  assigned_qty: Number.isFinite(assignedQty) ? assignedQty : 0,
                  remaining_qty: Number.isFinite(remainingQty)
                    ? remainingQty
                    : Math.max(0, Number(meta.qty || 0) - 0),
                  confidence: Number.isFinite(rawScore)
                    ? Math.max(0, Math.min(1, (rawScore + 1) / 2))
                    : 0,
                  score: rawScore,
                }};
              }});
            }}
            crop.ai_snap_result = result;
            crop.ai_snap_error = "";
            renderAiSnapStatus(crop);
            renderSuggestedParts(crop.crop_id);
            jumpToSuggestedParts();
          }} catch (_error) {{
            crop.ai_snap_error = "AI Snap failed";
            crop.ai_snap_result = null;
            renderAiSnapStatus(crop);
          }} finally {{
            crop.ai_snap_loading = false;
            renderAiSnapStatus(crop);
          }}
        }}

        async function runAutoMaskSlots() {{
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          if (!crop) {{
            return;
          }}
          crop.auto_mask_loading = true;
          crop.ai_snap_error = "";
          renderAiSnapStatus(crop);
          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            crop_id: crop.crop_id,
            sam_refine: SAM_REFINE,
            fast_map: FAST_MAP,
          }};
          try {{
            const res = await fetch("/debug/auto-mask-slots", {{
              method: "POST",
              headers: {{"Content-Type": "application/json"}},
              body: JSON.stringify(payload)
            }});
            if (!res.ok) {{
              let detail = "Auto mask failed";
              try {{
                const errorPayload = await res.json();
                detail = errorPayload.detail || detail;
              }} catch (_error) {{
                detail = "Auto mask failed";
              }}
              crop.ai_snap_error = detail;
              crop.auto_mask_slots = [];
              renderBuildabilitySlots(crop.crop_id);
              return;
            }}
            const result = await res.json();
            crop.auto_mask_slots = Array.isArray(result && result.slots) ? result.slots : [];
            crop.full_crop_mask_path = String(result && result.full_crop_mask_path || "");
            crop.full_crop_mask_overlay_path = String(result && result.full_crop_mask_overlay_path || "");
            crop.full_crop_mask_error = String(result && result.full_crop_mask_error || "");
            crop.full_crop_mask_generated_at = String(result && result.generated_at || "");
            const maskedCount = crop.auto_mask_slots.filter((slot) => String(slot && slot.status || "") === "masked").length;
            crop.ai_snap_error = "";
            const status = document.getElementById("ai-snap-status");
            if (status) {{
              status.innerHTML = escapeHtml(
                "Auto masked " + maskedCount + " / " + crop.auto_mask_slots.length + " slots."
                + (SHOW_SLOT_MATCHES ? " Matching candidates..." : "")
              )
                + renderFullCropMaskDebug(crop);
            }}
            renderBuildabilitySlots(crop.crop_id);
            const maskedSlots = crop.auto_mask_slots.filter((slot) => String(slot && slot.status || "") === "masked");
            if (SHOW_SLOT_MATCHES) {{
              await Promise.all(maskedSlots.map((slot) => loadSlotMaskCandidates(crop, slot)));
            }}
            const readyCount = maskedSlots.filter((slot) => Array.isArray(slot && slot.ranked_candidates)).length;
            if (status) {{
              status.innerHTML = escapeHtml(
                SHOW_SLOT_MATCHES
                  ? ("Auto masked " + maskedCount + " / " + crop.auto_mask_slots.length + " slots; candidates ready for " + readyCount + ".")
                  : ("Auto masked " + maskedCount + " / " + crop.auto_mask_slots.length + " slots.")
              )
                + renderFullCropMaskDebug(crop);
            }}
            renderBuildabilitySlots(crop.crop_id);
          }} catch (_error) {{
            crop.ai_snap_error = "Auto mask failed";
            crop.auto_mask_slots = [];
            renderBuildabilitySlots(crop.crop_id);
          }} finally {{
            crop.auto_mask_loading = false;
            renderAiSnapStatus(crop);
          }}
        }}

        async function loadSlotMaskCandidates(crop, maskSlot) {{
          if (!SHOW_SLOT_MATCHES) {{
            return;
          }}
          if (!crop || !maskSlot || String(maskSlot.status || "") !== "masked") {{
            return;
          }}
          if (!maskSlot.part_cutout_path || maskSlot.ranked_candidates) {{
            return;
          }}
          maskSlot.candidate_matching_started_at = new Date().toISOString();
          maskSlot.candidate_matching_started_before_cutout_save = false;
          maskSlot.candidate_matching_started_after_cutout_save = true;
          console.log("slot mask candidate matching started", {{
            crop_id: crop.crop_id,
            slot_index: Number(maskSlot.slot_index),
            generated_at: String(maskSlot.generated_at || ""),
            started_at: maskSlot.candidate_matching_started_at,
            cutout_path: String(maskSlot.part_cutout_path || ""),
            function_path_used: String(maskSlot.function_path_used || "")
          }});
          maskSlot.candidates_loading = true;
          maskSlot.candidates_error = "";
          renderBuildabilitySlots(crop.crop_id);
          try {{
            const res = await fetch("/debug/slot-mask-candidates", {{
              method: "POST",
              headers: {{"Content-Type": "application/json"}},
              body: JSON.stringify({{
                set_num: {json.dumps(str(set_num))},
                bag: {bag_number},
                crop_id: crop.crop_id,
                slot_index: Number(maskSlot.slot_index),
                part_cutout_path: String(maskSlot.part_cutout_path || ""),
                shape_mask_path: String(maskSlot.shape_mask_path || ""),
                clip_k: SLOT_MATCH_K,
              }})
            }});
            if (!res.ok) {{
              maskSlot.candidates_error = "match failed";
              return;
            }}
            const result = await res.json();
            maskSlot.ranked_candidates = Array.isArray(result && result.ranked_candidates)
              ? result.ranked_candidates
              : [];
            maskSlot.candidate_pool_source = String(result && result.candidate_pool_source || "");
            maskSlot.candidate_count = Number(result && result.candidate_count || 0);
          }} catch (_error) {{
            maskSlot.candidates_error = "match failed";
          }} finally {{
            maskSlot.candidates_loading = false;
            renderBuildabilitySlots(crop.crop_id);
          }}
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

        function renderAssignedParts(cropId) {{
          const crop = cropMap.get(cropId);
          const container = document.getElementById("assigned-" + cropId);
          if (!crop || !container) {{
            return;
          }}

          const parts = Array.isArray(crop.parts) ? crop.parts : [];
          if (!parts.length) {{
            container.innerHTML = '<div class="assigned-empty">No assigned parts yet.</div>';
            updateCropCardVisuals(cropId);
            updatePartTileAssignmentState();
            return;
          }}

          container.innerHTML = parts.map((part) => {{
            const partForView = hydratePart(part);
            const thumb = partForView.img_url
              ? '<img src="' + escapeHtml(partForView.img_url) + '" alt="' + escapeHtml(partForView.part_num) + '" loading="lazy" />'
              : '<div class="crop-missing">No image</div>';
            return `
              <div class="assigned-part">
                <div class="assigned-part-thumb">${{thumb}}</div>
                <div class="assigned-part-meta">
                  <strong>${{escapeHtml(partForView.part_num)}}</strong><br/>
                  color: ${{escapeHtml(partForView.color_name)}}<br/>
                  selected qty: ${{escapeHtml(partForView.selected_qty_label)}}<br/>
                  element: ${{escapeHtml(partForView.element_id || "n/a")}}
                </div>
                <button
                  type="button"
                  class="remove-btn"
                  onclick='removeAssignedPart("${{escapeHtml(cropId)}}", ${{JSON.stringify(part.part_num)}}, ${{Number(part.color_id || 0)}}, ${{JSON.stringify(part.element_id || "")}})'
                >
                  Remove
                </button>
              </div>
            `;
          }}).join("");
          updateCropCardVisuals(cropId);
          updatePartTileAssignmentState();
        }}

        function selectCrop(cropId) {{
          const crop = cropMap.get(cropId);
          if (!crop || (crop.is_hidden && !showHiddenCrops)) {{
            return;
          }}
          activeCropId = cropId;
          document.querySelectorAll('.crop-card').forEach((el) => {{
            el.classList.remove('selected');
          }});
          const el = document.getElementById(cropId);
          if (el) {{
            el.classList.add('selected');
          }}
          updateSelectedCropStatus(crop);
          document.getElementById('save-status').textContent = "";
          const qtyInput = document.getElementById("qty-input-" + cropId);
          if (qtyInput) {{
            window.setTimeout(() => {{
              qtyInput.focus();
              qtyInput.select();
            }}, 0);
          }}
          updateAddAvailability();
          applyPartFilter();
          renderSuggestedParts(cropId);
          renderColourPicker(cropId);
          renderAiSnapStatus(crop);
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
            renderColourPicker(cropId);
          }}
          updatePartTileAssignmentState();
          saveStatus.textContent = "Updated qty sequence for " + cropId;
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
            renderColourPicker(cropId);
          }}
          updatePartTileAssignmentState();
          saveStatus.textContent = "Removed " + partNum + " / color " + colorId + " from " + crop.crop_id;
        }}

        async function addSuggestedPart(event, partNum, colorId, elementId, colorName) {{
          event.stopPropagation();
          await selectTile(partNum, colorId, elementId, colorName);
        }}

        async function acceptSlotSuggestion(event, el) {{
          if (!SHOW_SLOT_MATCHES) {{
            return;
          }}
          if (event) {{
            event.stopPropagation();
          }}
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          if (!crop || !el || !el.dataset) {{
            return;
          }}
          const slotIndex = Number(el.dataset.slotIndex);
          const sequence = buildClientQtySequence(crop);
          const slot = Number.isInteger(slotIndex) ? sequence[slotIndex] : null;
          if (!slot) {{
            return;
          }}
          const partNum = String(el.dataset.partNum || "");
          const colorId = Number(el.dataset.colorId || 0);
          const elementId = String(el.dataset.elementId || "");
          const colorName = String(el.dataset.colorName || "");
          const maskSlotForSave = (Array.isArray(crop.auto_mask_slots) ? crop.auto_mask_slots : []).find(
            (s) => Number(s && s.slot_index) === Number(slotIndex)
          );
          const aiSnapInputPath = (maskSlotForSave && maskSlotForSave.part_cutout_path) || null;
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
            ai_snap_input_path: aiSnapInputPath,
            qty: slot.qty ?? null,
            qty_text: slot.qty_text || null,
            selected_slot_index: slotIndex,
            adjustments: [{{ type: "auto_mask_slot_suggestion", slot_index: slotIndex }}],
            allow_extra_part: true
          }};
          const res = await fetch("/debug/save-label", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload)
          }});
          const status = document.getElementById("save-status");
          if (!res.ok) {{
            let detail = "Save failed";
            try {{
              const errorPayload = await res.json();
              detail = errorPayload.detail || detail;
            }} catch (_error) {{}}
            if (status) {{
              status.textContent = detail;
            }}
            return;
          }}
          const result = await res.json();
          syncCropFromResponse(crop, result.crop);
          const maskSlot = (Array.isArray(crop.auto_mask_slots) ? crop.auto_mask_slots : []).find((slotItem) => Number(slotItem && slotItem.slot_index) === Number(slotIndex));
          if (maskSlot) {{
            maskSlot.accepted_part = {{
              part_num: partNum,
              color_id: colorId,
              element_id: elementId,
              color_name: colorName,
            }};
          }}
          crop.ai_snap_result = null;
          crop.ai_snap_error = "";
          renderAssignedParts(crop.crop_id);
          renderBuildabilitySlots(crop.crop_id);
          renderSuggestedParts(crop.crop_id);
          updatePartTileAssignmentState();
          if (status) {{
            status.textContent = "Saved slot " + (slotIndex + 1) + " -> " + partNum + " / color " + colorId;
          }}
        }}

        // Accept a confirmed-memory prediction for a slot.
        // Saves it as a real label then clears predicted_part so the tile
        // re-renders as confirmed.  Independent of SHOW_SLOT_MATCHES.
        async function acceptPredictedSlot(event, slotIndex) {{
          if (event) {{
            event.stopPropagation();
          }}
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          if (!crop) {{
            return;
          }}
          const maskSlot = (Array.isArray(crop.auto_mask_slots) ? crop.auto_mask_slots : []).find(
            (s) => Number(s && s.slot_index) === Number(slotIndex)
          );
          if (!maskSlot || !maskSlot.predicted_part) {{
            return;
          }}
          const p = maskSlot.predicted_part;
          const sequence = buildClientQtySequence(crop);
          const seqSlot = Number.isInteger(Number(slotIndex)) ? sequence[Number(slotIndex)] : null;
          if (!seqSlot) {{
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
            part_num: String(p.part_num || ""),
            color_id: Number(p.color_id || 0),
            color_name: p.color_name || null,
            element_id: p.element_id || null,
            ai_snap_input_path: maskSlot.part_cutout_path || null,
            qty: seqSlot.qty ?? null,
            qty_text: seqSlot.qty_text || null,
            selected_slot_index: Number(slotIndex),
            adjustments: [{{
              type: "predicted_from_confirmed",
              slot_index: Number(slotIndex),
              similarity: maskSlot.prediction_similarity || 0,
            }}],
            allow_extra_part: true,
          }};
          const res = await fetch("/debug/save-label", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload),
          }});
          const status = document.getElementById("save-status");
          if (!res.ok) {{
            let detail = "Save failed";
            try {{
              const errPayload = await res.json();
              detail = errPayload.detail || detail;
            }} catch (_err) {{}}
            if (status) {{
              status.textContent = detail;
            }}
            return;
          }}
          const result = await res.json();
          syncCropFromResponse(crop, result.crop);
          // Clear the prediction so the slot now renders as confirmed.
          maskSlot.predicted_part = null;
          maskSlot.prediction_source = null;
          maskSlot.prediction_similarity = null;
          renderAssignedParts(crop.crop_id);
          renderBuildabilitySlots(crop.crop_id);
          updatePartTileAssignmentState();
          if (status) {{
            status.textContent = "Accepted prediction: slot " + (Number(slotIndex) + 1)
              + " → " + String(p.part_num || "") + " / color " + Number(p.color_id || 0);
          }}
        }}

        // Dismiss a confirmed-memory prediction without saving.
        function rejectSlotPrediction(event, slotIndex) {{
          if (event) {{
            event.stopPropagation();
          }}
          const crop = activeCropId ? cropMap.get(activeCropId) : null;
          if (!crop) {{
            return;
          }}
          const maskSlot = (Array.isArray(crop.auto_mask_slots) ? crop.auto_mask_slots : []).find(
            (s) => Number(s && s.slot_index) === Number(slotIndex)
          );
          if (!maskSlot) {{
            return;
          }}
          maskSlot.predicted_part = null;
          maskSlot.prediction_source = null;
          maskSlot.prediction_similarity = null;
          renderBuildabilitySlots(crop.crop_id);
        }}

        async function selectTile(partNum, colorId, elementId, colorName) {{
          if (!activeCropId) {{
            alert("Select a crop first");
            return;
          }}
          const crop = cropMap.get(activeCropId);
          if (!crop) {{
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
            crop.ai_snap_result = null;
            crop.ai_snap_error = "";
            renderAssignedParts(crop.crop_id);
            if (activeCropId === crop.crop_id) {{
              renderSuggestedParts(crop.crop_id);
              renderColourPicker(crop.crop_id);
              renderAiSnapStatus(crop);
            }}
            updatePartTileAssignmentState();
            status.textContent = "Saved " + crop.crop_id + " -> " + partNum + " / color " + colorId;
            const updatedSlotState = computeCropSlotState(crop);
            if (!updatedSlotState.noQtyDetected && updatedSlotState.slotsFull) {{
              const nextCropId = nextVisibleCropId(crop.crop_id);
              if (nextCropId) {{
                selectCrop(nextCropId);
              }}
            }}
          }} else {{
            let detail = "Save failed";
            try {{
              const errorPayload = await res.json();
              detail = errorPayload.detail || detail;
            }} catch (_error) {{
              detail = "Save failed";
            }}
            status.textContent = detail;
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
        renderAiSnapStatus(activeCropId ? cropMap.get(activeCropId) : null);

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

        const colourPickerCanvas = document.getElementById("colour-picker-canvas");
        if (colourPickerCanvas) {{
          colourPickerCanvas.addEventListener("click", (event) => {{
            const crop = activeCropId ? cropMap.get(activeCropId) : null;
            if (!crop) {{
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
              delete crop.picked_sample_xy;
              delete crop.picked_sample_radius;
              crop.manual_calibration_status = "";
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
            crop.picked_sample_xy = {{
              x: Number(canvasX),
              y: Number(canvasY),
            }};
            crop.picked_sample_radius = 2;
            crop.manual_calibration_status = "";
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
          }});
        }}

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


@router.get("/debug/normalize-slot-crop", response_class=HTMLResponse)
def normalize_slot_crop_debug(
    image_path: Optional[str] = Query(None),
    set_num: Optional[str] = Query(None),
    bag: Optional[int] = Query(None, ge=1),
    crop_id: Optional[str] = Query(None),
    x: int = Query(...),
    y: int = Query(...),
    w: int = Query(...),
    h: int = Query(...),
):
    resolved_image_path = str(image_path or "").strip()
    resolved_crop: Optional[Dict[str, Any]] = None
    temp_crop_path: Optional[Path] = None
    bag_number = int(bag or 1)

    if not resolved_image_path:
        resolved_set_num = str(set_num or "").strip()
        resolved_crop_id = str(crop_id or "").strip()
        if not resolved_set_num or not resolved_crop_id:
            raise HTTPException(
                status_code=400,
                detail="Provide image_path or set_num + crop_id",
            )
        resolved_crop = _load_crop_for_ai_snap(resolved_set_num, bag_number, resolved_crop_id)
        if not resolved_crop:
            raise HTTPException(status_code=404, detail="crop_id not found")
        temp_crop_path = _write_ai_snap_temp_crop_image(resolved_crop)
        if temp_crop_path is not None:
            resolved_image_path = str(temp_crop_path)
        else:
            fallback_crop_image_path = str(resolved_crop.get("crop_image_path") or "").strip()
            if not fallback_crop_image_path:
                raise HTTPException(status_code=400, detail="Resolved crop image unavailable")
            resolved_image_path = fallback_crop_image_path

    try:
        result = normalize_slot_crop_from_qty(
            resolved_image_path,
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            },
        )
    finally:
        if temp_crop_path is not None:
            try:
                temp_crop_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _file_to_data_uri(path_text: str) -> str:
        path = Path(str(path_text or "").strip())
        if not path.exists() or not path.is_file():
            return ""
        suffix = path.suffix.lower()
        mime_type = "image/png" if suffix == ".png" else ("image/jpeg" if suffix in {".jpg", ".jpeg"} else "application/octet-stream")
        return "data:%s;base64,%s" % (
            mime_type,
            base64.b64encode(path.read_bytes()).decode("ascii"),
        )

    original_data_uri = _file_to_data_uri(str(result.get("original_path") or ""))
    mask_data_uri = _file_to_data_uri(str(result.get("mask_path") or ""))
    component_data_uri = _file_to_data_uri(str(result.get("component_path") or ""))
    normalized_data_uri = _file_to_data_uri(str(result.get("normalized_path") or ""))
    debug_json = json.dumps(
        {
            "ok": bool(result.get("ok")),
            "original_path": str(result.get("original_path") or ""),
            "mask_path": str(result.get("mask_path") or ""),
            "component_path": str(result.get("component_path") or ""),
            "normalized_path": str(result.get("normalized_path") or ""),
            "selected_box": result.get("selected_box"),
            "qty_box": result.get("qty_box"),
            "resolved_from_crop_id": str((resolved_crop or {}).get("crop_id") or ""),
            "resolved_set_num": str(set_num or ""),
            "resolved_bag": int(bag_number),
            "resolved_crop_image_path": str((resolved_crop or {}).get("crop_image_path") or ""),
            "debug": result.get("debug"),
        },
        indent=2,
    )

    html = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <title>Normalize Slot Crop</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f4f7fb;
            color: #1d2a36;
          }}
          h1 {{
            margin: 0 0 14px;
            font-size: 24px;
          }}
          .meta {{
            margin-bottom: 16px;
            padding: 12px 14px;
            border-radius: 12px;
            background: #fff;
            border: 1px solid #d6dee8;
          }}
          .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
          }}
          .card {{
            background: #fff;
            border: 1px solid #d6dee8;
            border-radius: 14px;
            padding: 14px;
          }}
          .card h2 {{
            margin: 0 0 10px;
            font-size: 16px;
          }}
          img {{
            width: 100%;
            height: auto;
            display: block;
            background: #fff;
            border: 1px solid #d6dee8;
            border-radius: 10px;
          }}
          pre {{
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 12px;
            line-height: 1.45;
          }}
        </style>
      </head>
      <body>
        <h1>Normalize Slot Crop</h1>
        <div class="meta">
          <div><strong>Image:</strong> {escape(str(result.get("original_path") or resolved_image_path))}</div>
          {f'<div><strong>Resolved crop:</strong> {escape(str((resolved_crop or {}).get("crop_id") or ""))}</div>' if resolved_crop else ''}
          <div><strong>Qty box:</strong> {escape(json.dumps(result.get("qty_box"), separators=(",", ":")))}</div>
          <div><strong>Selected box:</strong> {escape(json.dumps(result.get("selected_box"), separators=(",", ":")))}</div>
        </div>
        <div class="grid">
          <div class="card">
            <h2>Original</h2>
            {f'<img src="{original_data_uri}" alt="Original crop" />' if original_data_uri else '<p>Original image unavailable.</p>'}
          </div>
          <div class="card">
            <h2>Mask</h2>
            {f'<img src="{mask_data_uri}" alt="Foreground mask" />' if mask_data_uri else '<p>Mask unavailable.</p>'}
          </div>
          <div class="card">
            <h2>Selected component</h2>
            {f'<img src="{component_data_uri}" alt="Selected component crop" />' if component_data_uri else '<p>Selected component unavailable.</p>'}
          </div>
          <div class="card">
            <h2>Normalized</h2>
            {f'<img src="{normalized_data_uri}" alt="Normalized slot crop" />' if normalized_data_uri else '<p>Normalized image unavailable.</p>'}
          </div>
        </div>
        <div class="card">
          <h2>Debug JSON</h2>
          <pre>{escape(debug_json)}</pre>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@router.get("/debug/manual-match-review", response_class=HTMLResponse)
def manual_match_review(
    set_num: str = Query(...),
    bag: Optional[int] = Query(None, ge=1),
):
    bag_number = int(bag or 1)
    parts_payload = load_instruction_set_parts(set_num)
    parts = _prepare_instruction_parts_for_display(list(parts_payload.get("parts", []) or []))
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
    lego_colors_json = json.dumps(lego_colors)
    labels_payload = _load_existing_labels(_label_store_path(str(set_num), bag_number))
    crops = _build_instruction_callout_crops(str(set_num), bag_number, ai_enabled=False)
    crop_tiles: List[str] = []
    review_crops: List[Dict[str, Any]] = []
    for crop in crops:
        saved_crop = dict(labels_payload.get("crops", {}).get(crop["crop_id"]) or {})
        status = str(saved_crop.get("status") or "needs_adjust").strip().lower()
        if status == "hidden":
            continue
        saved_qty_text = _coerce_str_list(saved_crop.get("qty_text", []) or saved_crop.get("crop_qty_text", []))
        if saved_qty_text:
            crop["qty_text"] = saved_qty_text
            crop["qty_numbers"] = _coerce_int_list(saved_crop.get("qty", []))
            crop["qty_label"] = ", ".join(saved_qty_text) if saved_qty_text else "none"
        crop_qty_text = (
            _coerce_str_list(crop.get("candidate_detected_qty_text", []))
            or _coerce_str_list(crop.get("detected_qty_text", []))
            or saved_qty_text
            or _coerce_str_list(crop.get("qty_text", []))
        )
        crop_qty = (
            _coerce_int_list(crop.get("candidate_detected_qty_numbers", []))
            if crop_qty_text
            else (
                _coerce_int_list(saved_crop.get("qty", []) or saved_crop.get("crop_qty", []))
                or _coerce_int_list(crop.get("qty_numbers", []))
            )
        )
        saved_parts = list(saved_crop.get("parts", []) or [])
        slot_state = _crop_qty_slot_state({"parts": saved_parts}, crop_qty, crop_qty_text)
        slot_sequence = _build_qty_sequence(crop_qty, crop_qty_text)
        review_crops.append(
            {
                "crop_id": str(crop.get("crop_id") or ""),
                "page": int(crop.get("page", 0) or 0),
                "step": int(crop.get("step", 0) or 0),
            "crop_qty": list(crop_qty),
            "crop_qty_text": list(crop_qty_text),
                "crop_box": list(crop.get("crop_box", []) or []),
                "crop_box_format": str(crop.get("crop_box_format") or "xywh"),
                "crop_image_path": str(crop.get("crop_image_path") or ""),
                "next_qty": slot_state.get("next_slot", {}),
                "slot_sequence": slot_sequence,
                "filled_slots": int(slot_state.get("filled_slots", 0) or 0),
            }
        )
        slot_buttons = "".join(
            f'<button type="button" class="slot-btn{" assigned" if idx < int(slot_state.get("filled_slots", 0) or 0) else ""}" data-crop-slot data-crop-id="{escape(str(crop.get("crop_id") or ""))}" data-slot-index="{idx}" data-slot-assigned="{str(idx < int(slot_state.get("filled_slots", 0) or 0)).lower()}">Slot {idx + 1}: {escape(str(slot.get("qty_text") or slot.get("qty") or "none"))}</button>'
            for idx, slot in enumerate(slot_sequence)
        ) or '<div class="slot-empty">No qty slots</div>'
        thumb = _build_crop_image_html(crop)
        crop_tiles.append(
            f"""
            <div class="crop-tile" data-crop-tile data-crop-id="{escape(str(crop.get('crop_id') or ''))}">
              <div class="crop-thumb">{thumb}</div>
              <div class="crop-meta">
                <strong>{escape(str(crop.get("crop_id") or ""))}</strong><br/>
                page {int(crop.get("page", 0) or 0)} | step {int(crop.get("step", 0) or 0) if int(crop.get("step", 0) or 0) > 0 else "?"}<br/>
                qty: {escape(str(crop.get("qty_label") or "none"))}
              </div>
              <div class="slot-list" id="slot-list-{escape(str(crop.get('crop_id') or ''))}">{slot_buttons}</div>
            </div>
            """
        )
    assigned_qty_by_key: Dict[str, int] = {}
    for crop_data in dict(labels_payload.get("crops") or {}).values():
        for part_data in list((crop_data or {}).get("parts", []) or []):
            part_entry = _normalize_part_entry(part_data if isinstance(part_data, dict) else {})
            if not part_entry["part_num"]:
                continue
            key = f"{part_entry['part_num']}::{int(part_entry['color_id'] or 0)}"
            assigned_qty_by_key[key] = assigned_qty_by_key.get(key, 0) + int(part_entry.get("qty") or 1)
    candidate_tiles: List[str] = []
    review_parts: Dict[str, Dict[str, Any]] = {}
    for idx, part in enumerate(sorted(parts, key=lambda item: (str(item.get("part_num") or ""), int(item.get("color_id", 0) or 0))), start=1):
        key = f"{str(part.get('part_num') or '').strip()}::{int(part.get('color_id', 0) or 0)}"
        required_qty = int(part.get("qty", 0) or 0)
        assigned_qty = int(assigned_qty_by_key.get(key, 0) or 0)
        remaining_qty = required_qty - assigned_qty
        review_parts[key] = {
            "part_num": str(part.get("part_num") or "").strip(),
            "color_id": int(part.get("color_id", 0) or 0),
            "color_name": str(part.get("color_name") or "n/a"),
            "element_id": str(part.get("element_id") or ""),
            "remaining_qty": remaining_qty,
        }
        candidate_tiles.append(
            f"""
            <button type="button" class="part-tile-review" data-part-tile data-part-key="{escape(key)}" data-part-tile-index="{idx}" data-part-color-id="{int(part.get('color_id', 0) or 0)}" data-part-color-name="{escape(str(part.get('color_name') or 'n/a'))}">
              <div class="part-thumb-review">{f'<img src="{escape(str(part.get("img_url") or ""))}" alt="{escape(str(part.get("part_num") or ""))}" />' if str(part.get("img_url") or "").strip() else 'No image'}</div>
              <div class="crop-meta">
                <strong>{escape(str(part.get("part_num") or "unknown"))}</strong><br/>
                color: {int(part.get("color_id", 0) or 0)} / {escape(str(part.get("color_name") or "n/a"))}<br/>
                element: {escape(str(part.get("element_id") or "n/a"))}<br/>
                remaining qty: <span id="remaining-qty-{idx}">{remaining_qty}</span>
              </div>
            </button>
            """
        )
    review_crops_json = json.dumps(review_crops)
    review_parts_json = json.dumps(review_parts)
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>manual match review</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; background: #f4f7fb; color: #1f2d3d; }}
        .card {{ background: #fff; border: 1px solid #d6dee8; border-radius: 14px; padding: 18px; }}
        .layout {{ display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.9fr); gap: 16px; align-items: start; }}
        .manual-review-left, .manual-review-right {{ min-height: 0; }}
        .manual-review-left {{ display: block; }}
        .manual-review-right {{ display: block; }}
        .manual-review-panel {{ display: block; }}
        .crop-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 12px; margin-top: 16px; }}
        .crop-grid-wrap {{ padding-top: 12px; }}
        .crop-tile {{ border: 1px solid #d6dee8; border-radius: 12px; background: #fff; padding: 10px; text-align: left; cursor: pointer; }}
        .crop-tile.selected, .part-tile-review.selected {{ border-color: #cf1f1f; background: #fff1f1; }}
        .crop-thumb {{ min-height: 110px; display: flex; align-items: center; justify-content: center; background: #f4f7fb; border: 1px solid #d6dee8; border-radius: 10px; overflow: hidden; }}
        .crop-thumb img {{ max-width: 100%; max-height: 110px; display: block; }}
        .part-grid-review {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; margin-top: 12px; align-content: start; }}
        .part-tile-review {{ border: 1px solid #d6dee8; border-radius: 12px; background: #fff; padding: 10px; text-align: left; cursor: pointer; }}
        .part-thumb-review {{ min-height: 96px; display: flex; align-items: center; justify-content: center; background: #f4f7fb; border: 1px solid #d6dee8; border-radius: 10px; overflow: hidden; }}
        .part-thumb-review img {{ max-width: 100%; max-height: 96px; display: block; }}
        .crop-meta {{ margin-top: 8px; font-size: 12px; line-height: 1.35; }}
        .slot-list {{ margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }}
        .slot-btn {{ border: 1px solid #d6dee8; border-radius: 8px; background: #f8fbff; padding: 6px 8px; text-align: left; cursor: pointer; font-size: 12px; }}
        .slot-btn.selected {{ border-color: #cf1f1f; background: #fff1f1; }}
        .slot-btn.assigned {{ background: #eef6ef; border-color: #7db28a; color: #2f6c41; cursor: default; }}
        .slot-empty {{ color: #6c7c8d; font-size: 12px; }}
        .hidden {{ display: none !important; }}
        .candidate-filter-bar {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 6px; font-size: 13px; color: #627283; flex-wrap: wrap; }}
        .candidate-filter-controls {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; overflow: visible; align-content: flex-start; padding: 2px 0 4px; }}
        .candidate-filter-clear {{ border: 1px solid #d6dee8; border-radius: 12px; background: #fff; color: #627283; padding: 10px 14px; font-size: 13px; font-weight: 600; cursor: pointer; }}
        .candidate-filter-controls .colour-match {{ display: inline-flex; align-items: center; gap: 9px; text-align: left; border: 1px solid #cbd6e2; border-radius: 999px; background: #fff; padding: 10px 16px; min-height: 42px; cursor: pointer; font-size: 13px; font-weight: 600; line-height: 1.2; white-space: normal; max-width: 100%; box-shadow: 0 1px 2px rgba(31, 45, 61, 0.06); }}
        .candidate-filter-controls .colour-match .colour-swatch {{ width: 16px; height: 16px; flex: 0 0 16px; }}
        .candidate-filter-controls .colour-match.active {{ border-color: #cf1f1f; background: #fff1f1; box-shadow: 0 0 0 2px rgba(207, 31, 31, 0.08); color: #7d1d1d; }}
        .status-bar {{ margin-top: 14px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; font-size: 13px; }}
        .manual-crop-preview {{ position: sticky; top: 12px; z-index: 2; flex: 0 0 auto; margin-top: 16px; border: 1px solid #cf1f1f; border-radius: 14px; background: #fff7f7; box-shadow: 0 14px 28px rgba(31, 45, 61, 0.12); overflow: hidden; max-height: min(56vh, 540px); }}
        .manual-crop-preview.hidden {{ display: none; }}
        .manual-crop-preview-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 14px; border-bottom: 1px solid #f0d1d1; }}
        .manual-crop-preview-title-row {{ display: flex; align-items: center; gap: 10px; min-width: 0; }}
        .manual-crop-preview-title {{ font-size: 13px; font-weight: 700; min-width: 0; }}
        .manual-crop-picker-btn {{ border: 1px solid #d6dee8; border-radius: 999px; background: #fff; color: #4d6175; padding: 7px 10px; font-size: 12px; cursor: pointer; white-space: nowrap; }}
        .manual-crop-picker-btn.active {{ border-color: #cf1f1f; background: #fff1f1; color: #7d1d1d; }}
        .manual-crop-preview-close {{ border: 0; border-radius: 999px; width: 30px; height: 30px; background: #ffffff; color: #7d1d1d; font-size: 18px; line-height: 1; cursor: pointer; box-shadow: 0 2px 8px rgba(31, 45, 61, 0.12); }}
        .manual-crop-preview-body {{ display: flex; flex-direction: column; max-height: calc(min(56vh, 540px) - 56px); overflow-y: auto; }}
        .manual-crop-preview-frame {{ height: clamp(220px, 26vh, 280px); min-height: 220px; max-height: 280px; display: flex; align-items: center; justify-content: center; padding: 12px 14px; background: #f4f7fb; overflow: hidden; }}
        .manual-crop-preview-frame img {{ width: 100%; height: 100%; max-width: 100%; max-height: 100%; display: block; object-fit: contain; }}
        .manual-crop-preview-frame.picker-active, .manual-crop-preview-frame.picker-active img {{ cursor: crosshair; }}
        .manual-crop-preview-meta {{ padding: 10px 14px 12px; font-size: 12px; line-height: 1.45; color: #4d6175; }}
        .manual-crop-preview-slot-list {{ display: flex; flex-direction: column; gap: 6px; padding: 0 14px 12px; }}
        .assign-btn {{ border: 0; border-radius: 10px; background: #cf1f1f; color: #fff; padding: 10px 14px; font-weight: 700; cursor: pointer; }}
        .assign-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        @media (max-width: 980px) {{
          .layout {{ grid-template-columns: 1fr; }}
        }}
      </style>
    </head>
    <body>
      <div class="layout">
        <div class="card manual-review-panel manual-review-left">
          <h1>Manual Match Review</h1>
          <p>set_num: {escape(str(set_num))}</p>
          <p>bag: {bag_number}</p>
          <div class="status-bar">
            <span id="manual-match-status">Selected crop: none | Selected part: none</span>
            <button type="button" class="assign-btn" id="manual-assign-btn" disabled>Assign Selected</button>
          </div>
          <div class="manual-crop-preview hidden" id="manual-crop-preview">
            <div class="manual-crop-preview-head">
              <div class="manual-crop-preview-title-row">
                <div class="manual-crop-preview-title" id="manual-crop-preview-title">Selected crop preview</div>
                <button type="button" class="manual-crop-picker-btn" id="manual-crop-picker-btn">Pick colour from crop</button>
              </div>
              <button type="button" class="manual-crop-preview-close" id="manual-crop-preview-close" aria-label="Close crop preview">×</button>
            </div>
            <div class="manual-crop-preview-body">
              <div class="manual-crop-preview-frame" id="manual-crop-preview-frame"></div>
              <div class="manual-crop-preview-meta" id="manual-crop-preview-meta"></div>
              <div class="manual-crop-preview-slot-list" id="manual-crop-preview-slots"></div>
            </div>
          </div>
          <div class="crop-grid-wrap" id="manual-crop-grid-wrap">
            <div class="crop-grid">
              {"".join(crop_tiles) if crop_tiles else "<div>No crops found.</div>"}
            </div>
          </div>
        </div>
        <div class="card manual-review-panel manual-review-right">
          <h2>Candidate Parts</h2>
          <p>Full remaining part library for set {escape(str(set_num))} / bag {bag_number}</p>
          <div class="candidate-filter-bar">
            <span id="candidate-filter-status">Colour filter: none</span>
            <button type="button" class="candidate-filter-clear hidden" id="candidate-filter-clear">Clear filter</button>
          </div>
          <div class="candidate-filter-controls" id="candidate-filter-controls"></div>
          <div class="part-grid-review">
            {"".join(candidate_tiles) if candidate_tiles else "<div>No candidates available.</div>"}
          </div>
        </div>
      </div>
      <script>
        const reviewCrops = {review_crops_json};
        const reviewParts = {review_parts_json};
        window.legoColors = {lego_colors_json};
        const cropReviewMap = new Map(reviewCrops.map((item) => [String(item.crop_id || ""), item]));
        const candidateColorRgbById = new Map(
          (window.legoColors || [])
            .map((candidate) => {{
              const colorId = Number(candidate && candidate.color_id);
              const rgbHex = String((candidate && candidate.rgb) || "").trim().replace(/^#/, "").replace(/^0x/i, "").toUpperCase();
              if (!Number.isFinite(colorId) || rgbHex.length !== 6 || !/^[0-9A-F]{{6}}$/.test(rgbHex)) {{
                return null;
              }}
              return [colorId, rgbHex];
            }})
            .filter(Boolean)
        );
        const availableFilterColors = Array.from(
          new Map(
            Object.values(reviewParts || {{}})
              .map((part) => {{
                const colorId = Number(part && part.color_id);
                if (!Number.isFinite(colorId)) {{
                  return null;
                }}
                return [
                  colorId,
                  {{
                    color_id: colorId,
                    color_name: String((part && part.color_name) || ("Color " + colorId)),
                    rgb: String(candidateColorRgbById.get(colorId) || ""),
                  }},
                ];
              }})
              .filter(Boolean)
          ).values()
        ).sort((a, b) => Number(a.color_id || 0) - Number(b.color_id || 0));
        let selectedCropId = "";
        let selectedPartKey = "";
        let selectedSlotIndex = null;
        let colourOverride = null;
        let activeColourFilter = null;
        let metallicOnly = false;
        let cropPickerActive = false;
        let previewDismissed = false;
        let previewSelectionKey = "";
        function isMetallicStyleColorName(colorName) {{
          const normalized = String(colorName || "").toLowerCase();
          return [
            "pearl",
            "metallic",
            "chrome",
            "silver",
            "gold",
            "trans",
            "titanium",
            "copper",
          ].some((term) => normalized.includes(term));
        }}
        function rgbHexToTuple(rgbHex) {{
          if (!rgbHex || rgbHex.length !== 6 || !/^[0-9A-F]{{6}}$/.test(rgbHex)) {{
            return null;
          }}
          return [
            parseInt(rgbHex.slice(0, 2), 16),
            parseInt(rgbHex.slice(2, 4), 16),
            parseInt(rgbHex.slice(4, 6), 16),
          ];
        }}
        function squaredRgbDistance(a, b) {{
          return (
            Math.pow(Number(a[0] || 0) - Number(b[0] || 0), 2) +
            Math.pow(Number(a[1] || 0) - Number(b[1] || 0), 2) +
            Math.pow(Number(a[2] || 0) - Number(b[2] || 0), 2)
          );
        }}
        function activeFilterLabel() {{
          if (activeColourFilter === null) {{
            return "none";
          }}
          const match = availableFilterColors.find((candidate) => candidate.color_id === Number(activeColourFilter));
          return match ? (match.color_name + " (" + match.color_id + ")") : String(activeColourFilter);
        }}
        function setActiveColourFilter(colorId) {{
          activeColourFilter = Number.isFinite(Number(colorId)) ? Number(colorId) : null;
          applyActiveColourFilter();
          renderCandidateFilterControls();
        }}
        function clearActiveColourFilter() {{
          activeColourFilter = null;
          applyActiveColourFilter();
          renderCandidateFilterControls();
        }}
        function toggleMetallicOnly() {{
          metallicOnly = !metallicOnly;
          applyActiveColourFilter();
          renderCandidateFilterControls();
        }}
        function applyActiveColourFilter() {{
          document.querySelectorAll("[data-part-tile]").forEach((node) => {{
            const colorId = Number(node.dataset.partColorId || NaN);
            const colorName = String(node.dataset.partColorName || "");
            const matchesColour = activeColourFilter === null || colorId === Number(activeColourFilter);
            const matchesMetallic = !metallicOnly || isMetallicStyleColorName(colorName);
            const isVisible = matchesColour && matchesMetallic;
            node.classList.toggle("hidden", !isVisible);
          }});
          const filterStatus = document.getElementById("candidate-filter-status");
          const clearButton = document.getElementById("candidate-filter-clear");
          if (filterStatus) {{
            filterStatus.textContent = "Colour filter: " + activeFilterLabel() + (metallicOnly ? " | Metallic only" : "");
          }}
          if (clearButton) {{
            clearButton.classList.toggle("hidden", activeColourFilter === null);
          }}
        }}
        function renderCandidateFilterControls() {{
          const list = document.getElementById("candidate-filter-controls");
          if (!list) {{
            return;
          }}
          if (!availableFilterColors.length) {{
            list.innerHTML = '<div class="slot-empty">No part colours available.</div>';
            return;
          }}
          list.innerHTML = `
            <button
              type="button"
              class="colour-match${{activeColourFilter === null ? " active" : ""}}"
              data-candidate-filter-clear="true"
            >
              All colours
            </button>
            <button
              type="button"
              class="colour-match${{metallicOnly ? " active" : ""}}"
              data-candidate-filter-metallic="true"
            >
              Metallic only
            </button>
          ` + availableFilterColors.map((candidate) => `
            <button
              type="button"
              class="colour-match${{activeColourFilter === candidate.color_id ? " active" : ""}}"
              data-candidate-filter-colour="${{candidate.color_id}}"
            >
              <span>${{candidate.color_name}} (${{candidate.color_id}})</span>
            </button>
          `).join("");
          list.querySelector('[data-candidate-filter-clear="true"]')?.addEventListener("click", () => {{
            clearActiveColourFilter();
          }});
          list.querySelector('[data-candidate-filter-metallic="true"]')?.addEventListener("click", () => {{
            toggleMetallicOnly();
          }});
          list.querySelectorAll("[data-candidate-filter-colour]").forEach((button) => {{
            button.addEventListener("click", () => {{
              setActiveColourFilter(Number(button.dataset.candidateFilterColour || 0));
            }});
          }});
        }}
        function currentPreviewSelectionKey() {{
          return String(selectedCropId || "") + "::" + String(selectedSlotIndex === null ? "none" : selectedSlotIndex);
        }}
        function syncCropPickerUI() {{
          const button = document.getElementById("manual-crop-picker-btn");
          const frame = document.getElementById("manual-crop-preview-frame");
          if (button) {{
            button.classList.toggle("active", cropPickerActive);
            button.textContent = cropPickerActive ? "Click crop to sample" : "Pick colour from crop";
          }}
          if (frame) {{
            frame.classList.toggle("picker-active", cropPickerActive);
          }}
        }}
        function setCropPickerActive(isActive) {{
          cropPickerActive = !!isActive;
          syncCropPickerUI();
        }}
        function sampleNearestCandidateColourFromPreview(event) {{
          if (!cropPickerActive) {{
            return;
          }}
          const frame = document.getElementById("manual-crop-preview-frame");
          const image = frame ? frame.querySelector("img") : null;
          if (!frame || !image || !image.naturalWidth || !image.naturalHeight) {{
            return;
          }}
          const rect = image.getBoundingClientRect();
          if (!rect.width || !rect.height) {{
            return;
          }}
          const relX = event.clientX - rect.left;
          const relY = event.clientY - rect.top;
          if (relX < 0 || relY < 0 || relX > rect.width || relY > rect.height) {{
            return;
          }}
          const sampleX = Math.max(0, Math.min(image.naturalWidth - 1, Math.round((relX / rect.width) * image.naturalWidth)));
          const sampleY = Math.max(0, Math.min(image.naturalHeight - 1, Math.round((relY / rect.height) * image.naturalHeight)));
          const canvas = document.createElement("canvas");
          canvas.width = image.naturalWidth;
          canvas.height = image.naturalHeight;
          const ctx = canvas.getContext("2d", {{ willReadFrequently: true }});
          if (!ctx) {{
            return;
          }}
          ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight);
          const radius = 2;
          let r = 0;
          let g = 0;
          let b = 0;
          let count = 0;
          for (let dy = -radius; dy <= radius; dy += 1) {{
            for (let dx = -radius; dx <= radius; dx += 1) {{
              const px = Math.max(0, Math.min(image.naturalWidth - 1, sampleX + dx));
              const py = Math.max(0, Math.min(image.naturalHeight - 1, sampleY + dy));
              const pixel = ctx.getImageData(px, py, 1, 1).data;
              r += Number(pixel[0] || 0);
              g += Number(pixel[1] || 0);
              b += Number(pixel[2] || 0);
              count += 1;
            }}
          }}
          if (!count) {{
            return;
          }}
          const sampled = [Math.round(r / count), Math.round(g / count), Math.round(b / count)];
          let bestMatch = null;
          let bestDistance = Number.POSITIVE_INFINITY;
          availableFilterColors.forEach((candidate) => {{
            const rgb = rgbHexToTuple(String(candidate.rgb || ""));
            if (!rgb) {{
              return;
            }}
            const distance = squaredRgbDistance(sampled, rgb);
            if (distance < bestDistance) {{
              bestDistance = distance;
              bestMatch = candidate;
            }}
          }});
          if (bestMatch) {{
            setActiveColourFilter(Number(bestMatch.color_id));
          }}
        }}
        function updateManualMatchStatus() {{
          const status = document.getElementById("manual-match-status");
          const button = document.getElementById("manual-assign-btn");
          if (status) {{
            const slotText = selectedSlotIndex === null ? "none" : String(Number(selectedSlotIndex) + 1);
            status.textContent = "Selected crop: " + (selectedCropId || "none") + " | Selected slot: " + slotText + " | Selected part: " + (selectedPartKey || "none");
          }}
          if (button) {{
            button.disabled = !(selectedCropId && selectedPartKey && selectedSlotIndex !== null);
          }}
        }}
        function refreshSlotUI() {{
          reviewCrops.forEach((crop) => {{
            const filled = Number(crop.filled_slots || 0);
            document.querySelectorAll('[data-crop-slot][data-crop-id="' + crop.crop_id + '"]').forEach((node) => {{
              const idx = Number(node.dataset.slotIndex || -1);
              const assigned = idx > -1 && idx < filled;
              node.dataset.slotAssigned = assigned ? "true" : "false";
              node.classList.toggle("assigned", assigned);
              node.classList.toggle("selected", crop.crop_id === selectedCropId && idx === selectedSlotIndex);
            }});
            const cropTile = document.querySelector('[data-crop-tile][data-crop-id="' + crop.crop_id + '"]');
            if (cropTile) {{
              cropTile.classList.toggle("selected", crop.crop_id === selectedCropId);
            }}
          }});
          syncCropPreview();
        }}
        function syncCropPreview() {{
          const preview = document.getElementById("manual-crop-preview");
          const frame = document.getElementById("manual-crop-preview-frame");
          const meta = document.getElementById("manual-crop-preview-meta");
          const slots = document.getElementById("manual-crop-preview-slots");
          const title = document.getElementById("manual-crop-preview-title");
          if (!preview || !frame || !meta || !slots || !title) {{
            return;
          }}
          const crop = selectedCropId ? cropReviewMap.get(selectedCropId) : null;
          const selectionKey = currentPreviewSelectionKey();
          if (selectionKey !== previewSelectionKey) {{
            previewSelectionKey = selectionKey;
            previewDismissed = false;
          }}
          if (!crop || previewDismissed) {{
            preview.classList.add("hidden");
            setCropPickerActive(false);
            return;
          }}
          const cropTile = document.querySelector('[data-crop-tile][data-crop-id="' + crop.crop_id + '"]');
          const sourceImage = cropTile ? cropTile.querySelector(".crop-thumb img") : null;
          frame.innerHTML = "";
          if (sourceImage) {{
            const previewImage = sourceImage.cloneNode(true);
            previewImage.removeAttribute("onclick");
            previewImage.removeAttribute("loading");
            previewImage.removeAttribute("data-src");
            frame.appendChild(previewImage);
          }} else {{
            frame.textContent = "Crop preview unavailable";
          }}
          const slotText = selectedSlotIndex === null ? "none" : "Slot " + String(Number(selectedSlotIndex) + 1);
          title.textContent = "Preview: " + String(crop.crop_id || "selected crop");
          meta.innerHTML = `
            <strong>${{String(crop.crop_id || "selected crop")}}</strong><br/>
            page ${{String(crop.page || "?")}} | step ${{String(crop.step || "?")}} | selected slot: ${{slotText}}
          `;
          const sequence = Array.isArray(crop.slot_sequence) ? crop.slot_sequence : [];
          const filled = Number(crop.filled_slots || 0);
          slots.innerHTML = sequence.length
            ? sequence.map((slot, idx) => `
                <button
                  type="button"
                  class="slot-btn${{idx < filled ? " assigned" : ""}}${{idx === Number(selectedSlotIndex) ? " selected" : ""}}"
                  data-preview-slot-index="${{idx}}"
                  data-preview-slot-assigned="${{idx < filled ? "true" : "false"}}"
                >
                  Slot ${{idx + 1}}: ${{slot && (slot.qty_text || slot.qty || "none")}}
                </button>
              `).join("")
            : '<div class="slot-empty">No qty slots</div>';
          slots.querySelectorAll("[data-preview-slot-index]").forEach((button) => {{
            button.addEventListener("click", () => {{
              if (String(button.dataset.previewSlotAssigned || "") === "true") {{
                return;
              }}
              selectedCropId = String(crop.crop_id || "");
              selectedSlotIndex = Number(button.dataset.previewSlotIndex || 0);
              refreshSlotUI();
              updateManualMatchStatus();
            }});
          }});
          syncCropPickerUI();
          preview.classList.remove("hidden");
        }}
        function selectNextOpenSlot(fromCropId) {{
          const startIndex = Math.max(0, reviewCrops.findIndex((item) => String(item.crop_id || "") === String(fromCropId || "")));
          for (let offset = 0; offset < reviewCrops.length; offset += 1) {{
            const crop = reviewCrops[startIndex + offset];
            if (!crop) {{
              continue;
            }}
            const filled = Number(crop.filled_slots || 0);
            const sequence = Array.isArray(crop.slot_sequence) ? crop.slot_sequence : [];
            if (filled < sequence.length) {{
              selectedCropId = String(crop.crop_id || "");
              selectedSlotIndex = filled;
              refreshSlotUI();
              updateManualMatchStatus();
              return;
            }}
          }}
          selectedCropId = "";
          selectedSlotIndex = null;
          refreshSlotUI();
          updateManualMatchStatus();
        }}
        document.querySelectorAll("[data-crop-slot]").forEach((el) => {{
          el.addEventListener("click", () => {{
            if (String(el.dataset.slotAssigned || "") === "true") {{
              return;
            }}
            selectedCropId = String(el.dataset.cropId || "");
            selectedSlotIndex = Number(el.dataset.slotIndex || 0);
            refreshSlotUI();
            updateManualMatchStatus();
          }});
        }});
        document.querySelectorAll("[data-crop-tile]").forEach((el) => {{
          el.addEventListener("click", (event) => {{
            if (event.target.closest("[data-crop-slot]")) {{
              return;
            }}
            selectedCropId = String(el.dataset.cropId || "");
            selectedSlotIndex = null;
            refreshSlotUI();
            updateManualMatchStatus();
          }});
        }});
        document.getElementById("manual-crop-preview-close")?.addEventListener("click", () => {{
          previewDismissed = true;
          setCropPickerActive(false);
          document.getElementById("manual-crop-preview")?.classList.add("hidden");
        }});
        document.getElementById("manual-crop-picker-btn")?.addEventListener("click", () => {{
          setCropPickerActive(!cropPickerActive);
        }});
        document.getElementById("manual-crop-preview-frame")?.addEventListener("click", (event) => {{
          sampleNearestCandidateColourFromPreview(event);
        }});
        document.getElementById("candidate-filter-clear")?.addEventListener("click", () => {{
          clearActiveColourFilter();
        }});
        document.querySelectorAll("[data-part-tile]").forEach((el) => {{
          el.addEventListener("click", () => {{
            selectedPartKey = String(el.dataset.partKey || "");
            document.querySelectorAll("[data-part-tile]").forEach((node) => node.classList.toggle("selected", node === el));
            updateManualMatchStatus();
          }});
        }});
        document.getElementById("manual-assign-btn")?.addEventListener("click", async () => {{
          const crop = cropReviewMap.get(selectedCropId);
          const part = reviewParts[selectedPartKey];
          const sequence = crop && Array.isArray(crop.slot_sequence) ? crop.slot_sequence : [];
          const slot = selectedSlotIndex !== null ? sequence[Number(selectedSlotIndex)] : null;
          if (!crop || !part || selectedSlotIndex === null || !slot) {{
            return;
          }}
          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            crop_id: crop.crop_id,
            page: crop.page,
            step: crop.step,
            crop_qty: crop.crop_qty || [],
            crop_qty_text: crop.crop_qty_text || [],
            crop_box: crop.crop_box || [],
            crop_box_format: crop.crop_box_format || "xywh",
            crop_image_path: crop.crop_image_path || "",
            qty: slot.qty != null ? slot.qty : null,
            qty_text: slot.qty_text ? slot.qty_text : null,
            part_num: part.part_num,
            color_id: colourOverride !== null ? colourOverride.color_id : part.color_id,
            color_name: colourOverride !== null ? colourOverride.color_name : (part.color_name || null),
            element_id: part.element_id || null,
            selected_slot_index: selectedSlotIndex,
            adjustments: [{{ type: "manual_match_slot", slot_index: selectedSlotIndex }}],
          }};
          const res = await fetch("/debug/save-label", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload),
          }});
          if (!res.ok) {{
            let detail = "Assign failed";
            try {{
              const errorPayload = await res.json();
              detail = errorPayload.detail || detail;
            }} catch (_error) {{}}
            document.getElementById("manual-match-status").textContent = detail;
            return;
          }}
          crop.filled_slots = Math.max(Number(crop.filled_slots || 0), Number(selectedSlotIndex) + 1);
          document.getElementById("manual-match-status").textContent = "Assigned: " + selectedCropId + " slot " + String(Number(selectedSlotIndex) + 1) + " -> " + selectedPartKey;
          if (Number.isFinite(Number(part.remaining_qty))) {{
            part.remaining_qty = Number(part.remaining_qty) - Number(slot.qty || 1);
          }}
          const selectedTile = document.querySelector('[data-part-tile].selected');
          if (selectedTile) {{
            const tileIndex = String(selectedTile.dataset.partTileIndex || "");
            const remainingEl = tileIndex ? document.getElementById("remaining-qty-" + tileIndex) : null;
            if (remainingEl) {{
              remainingEl.textContent = String(part.remaining_qty);
            }}
            if (part.remaining_qty <= 0) {{
              selectedTile.style.opacity = "0.45";
            }}
          }}
          selectNextOpenSlot(selectedCropId);
          console.log("Assigned", payload);
        }});
        renderCandidateFilterControls();
        applyActiveColourFilter();
        refreshSlotUI();
        updateManualMatchStatus();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
