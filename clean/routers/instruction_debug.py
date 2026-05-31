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
import hashlib
import json
import os
import random
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote as _url_quote

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
from clean.services.training_store_service import list_registered_bundles, register_analysis_bundle, update_bundle_review
from clean.services.training_cloud_sync_service import (
    prepare_bundle_for_azure,
    prepare_bundle_for_r2,
    upload_bundle_to_r2,
    upload_confirmed_candidate_assets,
)
from clean.services.training_bundle_index_service import (
    confirm_candidate_part,
    get_bundle as get_training_bundle_index_row,
    get_review_stats,
    list_candidate_training_examples,
    list_confirmed_part_totals_for_set,
    list_confirmed_part_usage,
    list_review_queue,
    reset_bag_index_rows,
    unconfirm_candidate_part,
    update_review as update_training_bundle_review,
    update_split_candidates,
)
from clean.services.training_ai_review_service import (
    analyse_reviewed_bundle,
    generate_split_candidates,
    mark_split_candidate,
    scrub_candidate_qty,
    set_split_candidate_review_state,
)
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

_AUTO_MASK_CACHE_DIR = Path(__file__).resolve().parents[2] / "debug" / "ai_training" / "auto_mask_cache"
_STEP_SEG_DIR        = Path(__file__).resolve().parents[2] / "debug" / "ai_training" / "step_segmented_cutouts"
_ELEMENT_PAGE_EXTRACT_ROOT = Path("/Users/olly/aim2build-instruction/debug/element_page_extract")
_SAM2_TEST_ROOT            = Path("/Users/olly/aim2build-instruction/debug/sam2_test")
_TRAINING_PACK_ROOT        = Path("/Users/olly/aim2build-instruction/debug/element_training_packs")
_CLIP_EMBEDDINGS_ROOT      = Path("/Users/olly/aim2build-instruction/debug/clip_training_embeddings")
_CATALOG_CLIP_ROOT         = Path("/Users/olly/aim2build-instruction/debug/catalog_clip_embeddings")
_CATALOG_DB_PATH           = Path("/Users/olly/aim2build-instruction/debug/server_catalog/lego_catalog.db")
_CATALOG_IMG_CACHE_ROOT    = Path("/Users/olly/aim2build-instruction/debug/part_image_cache")
_CATALOG_MATCH_FEEDBACK_DIR = Path("/Users/olly/aim2build-instruction/debug/catalog_match_feedback")

# Lazy-loaded SAM2 model state (populated on first POST /debug/sam2-test call)
_sam2_processor: Any = None
_sam2_model: Any = None
_sam2_load_error: Optional[str] = None

# Lazy-loaded CLIP model state
_CLIP_MODEL_NAME  = "ViT-B-32"
_CLIP_PRETRAINED  = "laion2b_s34b_b79k"
_clip_model: Any       = None
_clip_preprocess: Any  = None
_clip_load_error: Optional[str] = None

# Lazy-loaded catalog embedding cache (populated on first match request)
_cat_emb_matrix: Any = None        # np.ndarray (N, 512) float32
_cat_emb_items:  Any = None        # List[Dict]  — aligned with matrix rows


def _clip_load() -> Tuple[bool, str]:
    """Lazy-load open_clip ViT-B-32 / laion2b_s34b_b79k (once per process)."""
    global _clip_model, _clip_preprocess, _clip_load_error
    if _clip_model is not None:
        return True, ""
    if _clip_load_error is not None:
        return False, _clip_load_error
    try:
        import open_clip  # type: ignore
        import torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL_NAME, pretrained=_CLIP_PRETRAINED
        )
        model.eval()
        # Use MPS if available (Apple Silicon), else CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
        _clip_model      = model
        _clip_preprocess = preprocess
        return True, ""
    except ImportError:
        _clip_load_error = (
            "open_clip not installed. "
            "Install with: pip install open-clip-torch"
        )
        return False, _clip_load_error
    except Exception as exc:
        _clip_load_error = str(exc)
        return False, _clip_load_error


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


# ---------------------------------------------------------------------------
# Persistent crop-detection disk cache
# Caches the raw output of _build_instruction_callout_crops() (incl. data_uri)
# so that OCR / step-detection is only re-run when rebuild=1 or cache is absent.
# ---------------------------------------------------------------------------

def _crop_cache_path(set_num: str, bag: int) -> Path:
    safe_set = "".join(ch for ch in str(set_num or "").strip() if ch.isalnum() or ch in "-_") or "unknown"
    safe_bag = max(1, int(bag or 1))
    return Path("/Users/olly/aim2build-instruction/debug/crop_cache") / f"{safe_set}_bag{safe_bag}.json"


def _load_crop_detection_cache(set_num: str, bag: int) -> Optional[List[Dict[str, Any]]]:
    """Return cached crop list or None if cache is absent / invalid."""
    path = _crop_cache_path(set_num, bag)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    # Basic validity: every entry must be a dict with a non-empty crop_id.
    if not all(isinstance(item, dict) and str(item.get("crop_id") or "").strip() for item in data):
        return None
    return data


def _write_crop_detection_cache(set_num: str, bag: int, crops: List[Dict[str, Any]]) -> None:
    """Atomically write crop list to disk cache."""
    path = _crop_cache_path(set_num, bag)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(crops, default=str), encoding="utf-8")
        os.replace(str(tmp), str(path))
        print(f"[crop-cache] wrote set={set_num} bag={bag} crops={len(crops)}")
    except Exception as exc:
        print(f"[crop-cache] write error set={set_num} bag={bag}: {exc}")


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
        for key in (
            "raw_ocr_object",
            "raw_box",
            "normalized_box",
            "converted_bbox",
            "raw_polygon",
            "normalized_polygon",
            "coordinate_source",
            "coordinate_space",
            "crop_image_size",
        ):
            if key in token:
                normalized_token[key] = token.get(key)
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
        if slot_index is not None and 0 <= int(slot_index) < len(sequence)
    }
    for part in parts:
        if _coerce_int(part.get("selected_slot_index")) is not None:
            continue
        signature = _qty_slot_signature(part.get("qty"), part.get("qty_text"))
        if not signature:
            continue
        assigned_counts[signature] = assigned_counts.get(signature, 0) + 1

    consumed_counts: Dict[str, int] = {}
    filled_slot_indices = set(filled_by_slot_index)
    for slot_index, slot in enumerate(sequence):
        if int(slot_index) in filled_by_slot_index:
            continue
        signature = _qty_slot_signature(slot.get("qty"), slot.get("qty_text"))
        if signature and assigned_counts.get(signature, 0) > consumed_counts.get(signature, 0):
            consumed_counts[signature] = consumed_counts.get(signature, 0) + 1
            filled_slot_indices.add(int(slot_index))
            continue

    next_slot: Optional[Dict[str, Any]] = None
    next_qty_index = len(sequence)
    for slot_index, slot in enumerate(sequence):
        if int(slot_index) not in filled_slot_indices:
            next_slot = dict(slot)
            next_qty_index = slot_index
            break

    return {
        "sequence": sequence,
        "total_slots": len(sequence),
        "filled_slots": len(filled_slot_indices),
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
                "bag_required_qty": int(row["qty_total"] or 0),
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
                "set_required_qty": int(meta.get("set_required_qty", 0) or 0),
                "bag_required_qty": int(row.get("bag_required_qty", meta.get("set_required_qty", 0) if source == "set_fallback" else 0) or 0),
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


# Module-level cache for candidate part image profiles.
# Key: "<absolute_path>:<mtime_float>" — mtime ensures stale entries are rebuilt
# if a file is replaced on disk, while static part images (the common case) are
# returned immediately on every subsequent call within the same process lifetime.
_CANDIDATE_PROFILE_CACHE: Dict[str, Any] = {}


def _slot_mask_candidate_profile(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    image_path = _slot_mask_resolve_local_image_path(part.get("img_url"))
    if image_path is None:
        return None

    # Build a cache key that incorporates mtime so replaced files are detected.
    try:
        cache_key = f"{image_path}:{image_path.stat().st_mtime}"
    except OSError:
        cache_key = None

    if cache_key is not None:
        cached = _CANDIDATE_PROFILE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    rgba = _slot_mask_read_rgba(image_path)
    if rgba is None:
        return None

    profile = _slot_mask_profile_from_rgba(rgba)
    # Only cache successful profiles; failures (None) are not stored so a
    # transient read error doesn't permanently poison the cache.
    if profile is not None and cache_key is not None:
        _CANDIDATE_PROFILE_CACHE[cache_key] = profile

    return profile


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
# ---------------------------------------------------------------------------
# Auto-mask disk cache — avoids re-running extraction for unchanged crops.
# Cache key = SHA1(crop_image_bytes + \x00 + canonical_json(qty_token_boxes)).
# A cached entry is invalidated if any part_cutout_path file is missing.
# ---------------------------------------------------------------------------


def _auto_mask_cache_key(crop_image_bytes: bytes, qty_token_boxes: list) -> str:
    """Return a hex SHA1 that uniquely identifies this (image, boxes) pair.

    Token text is included so that an OCR correction (e.g. "1x" → "2x") that
    flips the _force_window_fallback decision invalidates the cached result even
    when the bounding-box geometry is unchanged.
    """
    _GEOM_KEYS = ("x", "y", "w", "h", "cx", "cy")
    canonical = json.dumps(
        sorted(
            [
                {
                    **{k: b[k] for k in _GEOM_KEYS if k in b},
                    "text": str(b.get("text") or "").strip(),
                }
                for b in qty_token_boxes
                if isinstance(b, dict)
            ],
            key=lambda item: (item.get("y", 0), item.get("x", 0)),
        )
    ).encode()
    return hashlib.sha1(crop_image_bytes + b"\x00" + canonical).hexdigest()


def _read_auto_mask_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Load a cached auto-mask result; return None on miss or if any cutout file is gone."""
    path = _AUTO_MASK_CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        payload: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        for slot in list(payload.get("slots") or []):
            cp = str(slot.get("part_cutout_path") or "").strip()
            if cp and not Path(cp).exists():
                return None  # stale — cutout deleted; regenerate
        return payload
    except Exception:
        return None


def _write_auto_mask_cache(cache_key: str, payload: Dict[str, Any]) -> None:
    """Persist an auto-mask result to disk; silently swallows write errors."""
    try:
        _AUTO_MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _AUTO_MASK_CACHE_DIR / f"{cache_key}.json"
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Confirmed-label visual memory — local, deterministic, no FAISS/CLIP needed.
# ---------------------------------------------------------------------------

_CONFIRMED_MEMORY_THRESHOLD = 0.72
# Lower bar for surfacing memory entries as suggestions (vs. the auto-predict threshold).
_CONFIRMED_MEMORY_SUGGEST_THRESHOLD = 0.50
# Max LEGO-colour L2 distance (BGR space, range 0-441) between a memory candidate's confirmed
# colour and the slot's best colour evidence before the candidate is colour-filtered out.
# ~110 allows Black ↔ Dark Bluish Gray but rejects Black ↔ Reddish Brown (~145).
_MEMORY_COLOUR_COMPAT_DISTANCE = 110

# In-process cache for confirmed-memory profiles.
# Key: "set_num:bag:<labels_mtime>" — rebuilt whenever the labels file changes.
_CONFIRMED_MEMORY_PROFILE_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _confirmed_label_memory_cached(set_num: str, bag: int) -> List[Dict[str, Any]]:
    """Return confirmed-memory entries, reusing a cached result when labels are unchanged."""
    labels_path = _label_store_path(str(set_num), int(bag))
    try:
        mtime = str(labels_path.stat().st_mtime) if labels_path.exists() else "0"
    except OSError:
        mtime = "0"
    cache_key = f"{set_num}:{bag}:{mtime}"
    cached = _CONFIRMED_MEMORY_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = _confirmed_label_memory(set_num, bag)
    _CONFIRMED_MEMORY_PROFILE_CACHE[cache_key] = result
    # Evict stale entries for this set+bag so the dict doesn't grow unboundedly.
    prefix = f"{set_num}:{bag}:"
    for stale in [k for k in list(_CONFIRMED_MEMORY_PROFILE_CACHE) if k.startswith(prefix) and k != cache_key]:
        _CONFIRMED_MEMORY_PROFILE_CACHE.pop(stale, None)
    return result


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
    total_parts = 0
    with_snap_path = 0
    profiles_built = 0
    for crop_id, saved_crop in dict(labels_payload.get("crops") or {}).items():
        if not isinstance(saved_crop, dict):
            continue
        for part in list(saved_crop.get("parts") or []):
            if not isinstance(part, dict):
                continue
            total_parts += 1
            part_num = str(part.get("part_num") or "").strip()
            color_id = _coerce_int(part.get("color_id"))
            if not part_num or color_id is None:
                continue
            ai_snap_path = str(part.get("ai_snap_input_path") or "").strip()
            if not ai_snap_path:
                continue
            with_snap_path += 1
            profile = _slot_mask_query_profile(ai_snap_path, "")
            if profile is None:
                continue
            profiles_built += 1
            memory.append(
                {
                    "profile": profile,
                    "part_num": part_num,
                    "color_id": int(color_id),
                    "element_id": str(part.get("element_id") or ""),
                    "color_name": str(part.get("color_name") or ""),
                    "source_crop_id": str(crop_id),
                    "source_slot_index": _coerce_int(part.get("selected_slot_index")),
                    "ai_snap_input_path": ai_snap_path,
                }
            )
    print(
        "[confirmed-memory] "
        f"set={set_num} bag={int(bag)} total_parts={total_parts} "
        f"with_snap_path={with_snap_path} profiles_built={profiles_built}"
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
            slot["prediction_reference_path"] = best_entry.get("ai_snap_input_path", "")
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
    built_crops = _load_crop_detection_cache(str(set_num), int(bag))
    if built_crops is None:
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


def _segment_step_callout_slot(
    crop: Dict[str, Any],
    set_num: str,
    bag: int,
    crop_id: str,
    slot_index: int,
    slot_window: List[int],
) -> Dict[str, Any]:
    """Segment a single slot within a step-callout crop using the same pipeline as catalog-match-test.

    Steps:
      1. Read full page PNG at crop["crop_image_path"].
      2. Slice at crop["crop_box"] to get the raw callout region.
      3. Blank qty/OCR token boxes (_blank_text_regions_in_crop).
      4. Sub-slice at slot_window to isolate this slot.
      5. Run _hybrid_segment_crop() — SAM2 → quality guards → colour fallback.
      6. Build masked (part on white) and overlay (green contour) images.
      7. Persist to _STEP_SEG_DIR; write meta.json.
      8. Return { ok, masked_path, overlay_path, mask_path, segmentation_method, coverage_pct, cache_hit }.

    slot_window coords are relative to the callout crop (not the full page).
    """
    _error = lambda msg: {"ok": False, "masked_path": "", "overlay_path": "", "mask_path": "", "reason": msg, "cache_hit": False}

    crop_image_path = str(crop.get("crop_image_path") or "").strip()
    crop_box = _coerce_box_list(crop.get("crop_box"))
    if not crop_image_path or crop_box is None:
        return _error("crop_image_path or crop_box missing")

    safe_set  = re.sub(r"[^0-9A-Za-z._-]+", "_", str(set_num or "").strip() or "set")
    safe_crop = re.sub(r"[^0-9A-Za-z._-]+", "_", str(crop_id or "").strip() or "crop")
    stem = f"{safe_set}_bag{int(bag)}_{safe_crop}_slot{int(slot_index)}"

    _STEP_SEG_DIR.mkdir(parents=True, exist_ok=True)
    masked_path  = _STEP_SEG_DIR / f"{stem}_masked.png"
    overlay_path = _STEP_SEG_DIR / f"{stem}_overlay.png"
    mask_path    = _STEP_SEG_DIR / f"{stem}_mask.png"
    meta_path    = _STEP_SEG_DIR / f"{stem}_meta.json"

    # Cache hit: all files present
    if masked_path.exists() and mask_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except Exception:
            meta = {}
        return {
            "ok": True,
            "masked_path": str(masked_path),
            "overlay_path": str(overlay_path),
            "mask_path": str(mask_path),
            "segmentation_method": meta.get("segmentation_method", ""),
            "coverage_pct": meta.get("coverage_pct", 0.0),
            "cache_hit": True,
        }

    # Load full page image
    page_img = cv2.imread(crop_image_path)
    if page_img is None or getattr(page_img, "size", 0) == 0:
        return _error("page image unreadable")

    # Slice callout region from page
    cx, cy, cw, ch = [int(v) for v in crop_box]
    raw_callout = page_img[max(0, cy):max(0, cy) + max(0, ch), max(0, cx):max(0, cx) + max(0, cw)]
    if raw_callout is None or getattr(raw_callout, "size", 0) == 0:
        return _error("callout slice empty")

    # Blank qty/OCR token boxes (coordinates are in original-page space)
    qty_token_boxes = [dict(t) for t in list(crop.get("qty_token_boxes") or []) if isinstance(t, dict)]
    bg_color = _detect_crop_bg_color(raw_callout)
    clean_callout, _, _ = _blank_text_regions_in_crop(raw_callout, cx, cy, qty_token_boxes, bg_color)

    # Sub-slice at slot_window (relative to callout crop)
    if not slot_window or len(slot_window) < 4:
        return _error("slot_window missing or invalid")
    sw_x, sw_y, sw_w, sw_h = [int(v) for v in slot_window[:4]]
    callout_h, callout_w = clean_callout.shape[:2]
    sw_x = max(0, sw_x)
    sw_y = max(0, sw_y)
    sw_w = max(1, min(sw_w, callout_w - sw_x))
    sw_h = max(1, min(sw_h, callout_h - sw_y))
    slot_slice = clean_callout[sw_y:sw_y + sw_h, sw_x:sw_x + sw_w]
    if slot_slice is None or getattr(slot_slice, "size", 0) == 0:
        return _error("slot slice empty after clipping")

    # Ensure SAM2 is loaded (lazy — no-op if already loaded or unavailable)
    _sam2_load()

    # Segment
    mask, method, info = _hybrid_segment_crop(slot_slice)

    # Build outputs
    masked, overlay = _make_masked_and_overlay(slot_slice, mask)

    cv2.imwrite(str(mask_path),    mask)
    cv2.imwrite(str(masked_path),  masked)
    cv2.imwrite(str(overlay_path), overlay)

    coverage_pct = float(info.get("coverage_pct", 0.0))
    meta_out = {
        "stem": stem,
        "set_num": set_num,
        "bag": int(bag),
        "crop_id": crop_id,
        "slot_index": int(slot_index),
        "segmentation_method": method,
        "coverage_pct": coverage_pct,
        "iou": info.get("iou"),
        "reject_reason": info.get("reject_reason"),
        "generated_at": _iso_now(),
    }
    try:
        meta_path.write_text(json.dumps(meta_out, indent=2, ensure_ascii=True), encoding="utf-8")
    except Exception:
        pass

    return {
        "ok": method != "failed",
        "masked_path": str(masked_path),
        "overlay_path": str(overlay_path),
        "mask_path": str(mask_path),
        "segmentation_method": method,
        "coverage_pct": coverage_pct,
        "cache_hit": False,
    }


def _analysis_bundle_slug(set_num: str, bag: int, crop_id: str) -> str:
    safe_set = re.sub(r"[^0-9A-Za-z._-]+", "_", str(set_num or "").strip() or "set")
    safe_crop = re.sub(r"[^0-9A-Za-z._-]+", "_", str(crop_id or "").strip() or "crop")
    return f"{safe_set}_bag{int(bag or 0)}_{safe_crop}"


def _copy_analysis_bundle_file(path_value: Any, bundle_dir: Path, name: str) -> str:
    source = Path(str(path_value or "").strip())
    if not source.exists() or not source.is_file():
        return ""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    dest = bundle_dir / name
    shutil.copy2(str(source), str(dest))
    return str(dest)


def _normalize_qty_token_boxes_for_bundle(
    qty_token_boxes: List[Dict[str, Any]],
    *,
    crop_box: Any,
    original_crop_path: Any,
) -> List[Dict[str, Any]]:
    crop_dims = [0, 0]
    original_path = Path(str(original_crop_path or "").strip())
    if original_path.exists() and original_path.is_file():
        img = cv2.imread(str(original_path))
        if img is not None and getattr(img, "size", 0) != 0:
            crop_dims = [int(img.shape[1]), int(img.shape[0])]
    crop_box_list = _coerce_box_list(crop_box) or [0, 0, crop_dims[0], crop_dims[1]]
    crop_x, crop_y, crop_w, crop_h = [int(value) for value in crop_box_list]
    if crop_dims[0] <= 0 or crop_dims[1] <= 0:
        crop_dims = [max(0, crop_w), max(0, crop_h)]
    crop_width, crop_height = crop_dims

    def clip_xywh(x: float, y: float, w: float, h: float) -> List[int]:
        x1 = max(0, min(crop_width, int(round(x))))
        y1 = max(0, min(crop_height, int(round(y))))
        x2 = max(0, min(crop_width, int(round(x + max(0, w)))))
        y2 = max(0, min(crop_height, int(round(y + max(0, h)))))
        if x2 <= x1 or y2 <= y1:
            return []
        return [x1, y1, x2 - x1, y2 - y1]

    normalized: List[Dict[str, Any]] = []
    for token in list(qty_token_boxes or []):
        if not isinstance(token, dict):
            continue
        raw = dict(token)
        try:
            x = float(raw.get("x", raw.get("left", 0)) or 0)
            y = float(raw.get("y", raw.get("top", 0)) or 0)
            w = float(raw.get("w", raw.get("width", 0)) or 0)
            h = float(raw.get("h", raw.get("height", 0)) or 0)
        except Exception:
            continue
        source = "crop_local_xywh"
        box = clip_xywh(x, y, w, h)
        if not box and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
            source = "crop_local_normalized_xywh"
            box = clip_xywh(x * crop_width, y * crop_height, w * crop_width, h * crop_height)
        if not box and x >= crop_x and y >= crop_y:
            source = "page_xywh_minus_crop_origin"
            box = clip_xywh(x - crop_x, y - crop_y, w, h)
        if not box and w > x and h > y:
            source = "crop_local_xyxy"
            box = clip_xywh(x, y, w - x, h - y)
        if not box and w > x and h > y and x >= crop_x and y >= crop_y:
            source = "page_xyxy_minus_crop_origin"
            box = clip_xywh(x - crop_x, y - crop_y, w - x, h - y)
        if not box:
            box = clip_xywh(x, y, max(1, w), max(1, h))
            source = "best_effort_crop_local_xywh"
        if not box:
            continue
        item = dict(raw)
        item["raw_box"] = [raw.get("x"), raw.get("y"), raw.get("w"), raw.get("h")]
        item["raw_ocr_object"] = raw
        item["raw_polygon"] = [
            [int(round(x)), int(round(y))],
            [int(round(x + w)), int(round(y))],
            [int(round(x + w)), int(round(y + h))],
            [int(round(x)), int(round(y + h))],
        ]
        item["normalized_box"] = box
        item["converted_bbox"] = box
        item["normalized_polygon"] = [
            [box[0], box[1]],
            [box[0] + box[2], box[1]],
            [box[0] + box[2], box[1] + box[3]],
            [box[0], box[1] + box[3]],
        ]
        item["coordinate_source"] = source
        item["coordinate_space"] = "original_crop_local_xywh"
        item["crop_box"] = [crop_x, crop_y, crop_w, crop_h]
        item["crop_image_size"] = [crop_width, crop_height]
        item["x"], item["y"], item["w"], item["h"] = box
        item["cx"] = int(box[0] + box[2] / 2)
        item["cy"] = int(box[1] + box[3] / 2)
        normalized.append(item)
    return normalized


def _direct_qty_ocr_boxes_from_crop_image(image_path: Any) -> List[Dict[str, Any]]:
    path = Path(str(image_path or "").strip())
    if not path.exists() or not path.is_file():
        return []
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or getattr(img, "size", 0) == 0:
        return []
    try:
        import pytesseract
    except Exception:
        return []

    height, width = img.shape[:2]
    variants: List[Tuple[str, Any, float]] = [("original_psm6", img, 1.0)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for scale in (2.0, 3.0):
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        thresholded = cv2.adaptiveThreshold(
            enlarged,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            6,
        )
        variants.append((f"gray_s{int(scale)}_psm11", enlarged, scale))
        variants.append((f"adaptive_s{int(scale)}_psm11", thresholded, scale))
        variants.append((f"adaptive_s{int(scale)}_psm12", thresholded, scale))

    tokens: List[Dict[str, Any]] = []
    for name, variant, scale in variants:
        psm = 6 if "psm6" in name else (12 if "psm12" in name else 11)
        try:
            data = pytesseract.image_to_data(
                variant,
                config=f"--psm {psm} -c tessedit_char_whitelist=0123456789xX",
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue
        for idx in range(len(data.get("text", []) or [])):
            raw_text = str((data.get("text", [""])[idx] or "")).strip()
            normalized = _normalize_qty_token_text(raw_text)
            if not normalized or not re.match(r"^\d{1,2}x$", normalized):
                continue
            try:
                confidence = float(data.get("conf", ["-1"])[idx] or -1)
            except Exception:
                confidence = -1.0
            if confidence < 20:
                continue
            x = int(round(float(data.get("left", [0])[idx] or 0) / scale))
            y = int(round(float(data.get("top", [0])[idx] or 0) / scale))
            w = int(round(float(data.get("width", [0])[idx] or 0) / scale))
            h = int(round(float(data.get("height", [0])[idx] or 0) / scale))
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
                continue
            raw_ocr_object = {
                "text": raw_text,
                "conf": confidence,
                "left": data.get("left", [0])[idx],
                "top": data.get("top", [0])[idx],
                "width": data.get("width", [0])[idx],
                "height": data.get("height", [0])[idx],
                "ocr_variant": name,
                "scale": scale,
                "psm": psm,
            }
            tokens.append(
                {
                    "text": normalized,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "cx": x + (w // 2),
                    "cy": y + (h // 2),
                    "confidence": confidence,
                    "raw_ocr_object": raw_ocr_object,
                    "raw_box": [x, y, w, h],
                    "normalized_box": [x, y, w, h],
                    "converted_bbox": [x, y, w, h],
                    "raw_polygon": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    "normalized_polygon": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    "coordinate_source": f"direct_original_crop_image_to_data:{name}",
                    "coordinate_space": "original_crop_local_xywh",
                    "crop_image_size": [width, height],
                }
            )
    return _dedupe_qty_tokens_high_overlap_only(tokens)


def _master_islands_from_mask(mask_path: str) -> List[Dict[str, Any]]:
    mask_file = Path(str(mask_path or "").strip())
    if not mask_file.exists():
        return []
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if mask is None or getattr(mask, "size", 0) == 0:
        return []
    labels_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        8,
    )
    islands: List[Dict[str, Any]] = []
    for label in range(1, labels_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        fill = float(area) / float(max(1, w * h))
        islands.append(
            {
                "label": int(label),
                "bbox": [x, y, w, h],
                "area": area,
                "fill": round(fill, 4),
                "centroid": [round(float(centroids[label][0]), 2), round(float(centroids[label][1]), 2)],
            }
        )
    return islands


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


def _append_part_entry_preserving_slot(parts: List[Dict[str, Any]], part_entry: Dict[str, Any]) -> None:
    selected_slot_index = _coerce_int(part_entry.get("selected_slot_index"))
    if selected_slot_index is not None and selected_slot_index >= 0:
        for index, existing_part in enumerate(list(parts or [])):
            if _coerce_int((existing_part or {}).get("selected_slot_index")) == int(selected_slot_index):
                parts[index] = part_entry
                return
        parts.append(part_entry)
        return

    if not any(_same_part_entry(existing_part, part_entry) for existing_part in parts):
        parts.append(part_entry)


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
                _append_part_entry_preserving_slot(parts, part_entry)
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
            if part_entry["part_num"]:
                _append_part_entry_preserving_slot(crop_record["parts"], part_entry)
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
        # data_uri intentionally omitted — page images are lazy-loaded via
        # /debug/ai-snap-artifact when the Manual Crop tab is first opened.

        pages.append(
            {
                "page": int(page),
                "image_path": str(image_path),
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
    request_step_masked_path  = str(data.get("step_masked_path") or "").strip()
    request_part_cutout_path  = str(data.get("part_cutout_path") or "").strip()

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
    qty_token_boxes.sort(key=lambda t: (int(t.get("cy", 0) or 0), int(t.get("x", 0) or 0)))
    selected_qty_box: Optional[Dict[str, Any]] = None
    selected_qty_box_index = int(slot_index)
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
            if request_step_masked_path and Path(request_step_masked_path).is_file():
                rank_input_path = request_step_masked_path
                normalization_fallback_reason = ""
                print(
                    "[ai-rank-slot] using step_masked_path as query image crop_id=%s slot_index=%s query_image_path=%s"
                    % (str(crop_id), str(slot_index), request_step_masked_path)
                )
            elif request_part_cutout_path and Path(request_part_cutout_path).is_file():
                rank_input_path = request_part_cutout_path
                normalization_fallback_reason = ""
                print(
                    "[ai-rank-slot] using part_cutout_path as query image crop_id=%s slot_index=%s query_image_path=%s"
                    % (str(crop_id), str(slot_index), request_part_cutout_path)
                )
            elif selected_qty_box is not None:
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


@router.get("/debug/export-crop-analysis-bundle")
def export_crop_analysis_bundle(
    set_num: str = Query("70618"),
    bag: int = Query(2),
    crop_id: str = Query(...),
):
    set_num = str(set_num or "70618").strip() or "70618"
    bag_number = int(bag or 1)
    crop_id = str(crop_id or "").strip()
    if not crop_id:
        raise HTTPException(status_code=400, detail="crop_id is required")

    crop = _load_crop_for_ai_snap(set_num, bag_number, crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="crop not found")
    qty_token_boxes = [
        dict(item)
        for item in list(crop.get("qty_token_boxes", []) or [])
        if isinstance(item, dict)
    ]

    bundle_dir = (
        Path("/Users/olly/aim2build-instruction")
        / "debug"
        / "ai_training"
        / "analysis_bundles"
        / _analysis_bundle_slug(set_num, bag_number, crop_id)
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    temp_crop_path: Optional[Path] = None
    try:
        temp_crop_path = _write_ai_snap_temp_crop_image(crop)
        if temp_crop_path is None:
            raise HTTPException(status_code=400, detail="crop image unavailable")

        original_crop_path = _copy_analysis_bundle_file(temp_crop_path, bundle_dir, "original_crop.png")
        result = create_shape_masks_for_callout_slots(
            str(temp_crop_path),
            qty_token_boxes,
            set_num=set_num,
            bag=bag_number,
            crop_id=crop_id,
            desktop_overlays=False,
        )
    finally:
        if temp_crop_path is not None:
            try:
                temp_crop_path.unlink(missing_ok=True)
            except Exception:
                pass
    normalized_qty_token_boxes = _normalize_qty_token_boxes_for_bundle(
        qty_token_boxes,
        crop_box=crop.get("crop_box"),
        original_crop_path=original_crop_path,
    )
    direct_qty_ocr_boxes = _direct_qty_ocr_boxes_from_crop_image(original_crop_path)

    full_crop_mask_path = str(result.get("full_crop_mask_path") or "")
    full_crop_overlay_path = str(result.get("full_crop_mask_overlay_path") or "")
    raw_master_mask_path = ""
    master_island_overlay_path = ""
    if full_crop_mask_path:
        full_mask_file = Path(full_crop_mask_path)
        debug_stem = full_mask_file.stem.removesuffix("_full_mask")
        raw_master_mask_path = str(full_mask_file.parent / f"{debug_stem}_raw_master_mask.png")
        master_island_overlay_path = str(
            Path("/Users/olly/aim2build-instruction")
            / "debug"
            / "ai_training"
            / "full_crop_mask_overlays"
            / f"{debug_stem}_master_island_overlay.png"
        )

    copied_files: Dict[str, Any] = {
        "original_crop": original_crop_path,
        "full_mask_overlay": _copy_analysis_bundle_file(
            full_crop_overlay_path,
            bundle_dir,
            "full_mask_overlay.png",
        ),
        "raw_master_mask": _copy_analysis_bundle_file(
            raw_master_mask_path,
            bundle_dir,
            "raw_master_mask.png",
        ),
        "master_island_overlay": _copy_analysis_bundle_file(
            master_island_overlay_path,
            bundle_dir,
            "master_island_overlay.png",
        ),
        "slot_cutouts": [],
    }

    slot_assignments: List[Dict[str, Any]] = []
    for slot in list(result.get("slots") or []):
        slot_index = int(slot.get("slot_index", len(slot_assignments)) or 0)
        cutout_path = str(slot.get("part_cutout_path") or "")
        copied_cutout = _copy_analysis_bundle_file(
            cutout_path,
            bundle_dir,
            f"slot_{slot_index}_cutout.png",
        )
        copied_files["slot_cutouts"].append(copied_cutout)
        slot_assignments.append(
            {
                "slot_index": slot_index,
                "status": str(slot.get("status") or ""),
                "component_box": slot.get("component_box"),
                "component_area": slot.get("component_area"),
                "function_path_used": str(slot.get("function_path_used") or ""),
                "qty_token_box": slot.get("qty_token_box"),
                "shape_mask_path": str(slot.get("shape_mask_path") or ""),
                "part_cutout_path": cutout_path,
                "bundle_cutout_path": copied_cutout,
                "alpha_pixel_count": int(slot.get("alpha_pixel_count", 0) or 0),
                "reason": str(slot.get("reason") or ""),
            }
        )

    metadata = {
        "set_num": set_num,
        "bag": bag_number,
        "crop_id": crop_id,
        "generated_at": _iso_now(),
        "bundle_dir": str(bundle_dir),
        "source_crop": {
            "page": int(crop.get("page", 0) or 0),
            "step": int(crop.get("step", 0) or 0),
            "crop_box": crop.get("crop_box"),
            "crop_image_path": str(crop.get("crop_image_path") or ""),
        },
        "qty_token_boxes": normalized_qty_token_boxes,
        "qty_token_boxes_raw": qty_token_boxes,
        "direct_qty_ocr_boxes": direct_qty_ocr_boxes,
        "master_islands": _master_islands_from_mask(raw_master_mask_path or full_crop_mask_path),
        "slot_assignments": slot_assignments,
        "cutout_paths": [item for item in copied_files["slot_cutouts"] if item],
        "copied_files": copied_files,
        "source_artifacts": {
            "full_crop_mask_path": full_crop_mask_path,
            "full_crop_mask_overlay_path": full_crop_overlay_path,
            "raw_master_mask_path": raw_master_mask_path,
            "master_island_overlay_path": master_island_overlay_path,
        },
        "error": str(result.get("error") or ""),
    }
    metadata_path = bundle_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    return JSONResponse(
        {
            "ok": True,
            "bundle_dir": str(bundle_dir),
            "metadata_path": str(metadata_path),
            "copied_files": copied_files,
            "slot_count": int(result.get("slot_count") or len(slot_assignments)),
            "error": str(result.get("error") or ""),
        }
    )


@router.post("/debug/training-store/register-bundle")
def training_store_register_bundle(bundle_id: str = Query(...)):
    try:
        result = register_analysis_bundle(bundle_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result)


@router.post("/debug/training-store/export-batch")
def training_store_export_batch(
    set_num: str = Query("70618"),
    bag_num: int = Query(1),
    force: int = Query(0),
):
    set_text = str(set_num or "70618").strip() or "70618"
    bag_number = int(bag_num or 1)
    force_export = int(force or 0) == 1
    try:
        rendered_pages, start_page, end_page = _resolve_bag_page_range(set_text, bag_number)
        crops = _build_instruction_callout_crops(set_text, bag_number, ai_enabled=False)
        _write_crop_detection_cache(set_text, bag_number, crops)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"batch crop discovery failed: {type(exc).__name__}")

    analysis_root = (
        Path("/Users/olly/aim2build-instruction")
        / "debug"
        / "ai_training"
        / "analysis_bundles"
    )
    exported_bundle_ids: List[str] = []
    skipped_bundle_ids: List[str] = []
    registered_bundle_ids: List[str] = []
    local_bundle_paths: Dict[str, str] = {}
    registration_results: Dict[str, Any] = {}
    failed: List[Dict[str, Any]] = []

    for crop in list(crops or []):
        crop_id = str((crop or {}).get("crop_id") or "").strip()
        if not crop_id:
            continue
        bundle_id = _analysis_bundle_slug(set_text, bag_number, crop_id)
        metadata_path = analysis_root / bundle_id / "metadata.json"
        local_bundle_paths[bundle_id] = str(metadata_path.parent)
        if metadata_path.exists() and not force_export:
            skipped_bundle_ids.append(bundle_id)
            try:
                register_result = register_analysis_bundle(bundle_id)
                registration_results[bundle_id] = register_result
                postgres_result = register_result.get("postgres") if isinstance(register_result.get("postgres"), dict) else {}
                if bool(register_result.get("ok")) and bool(postgres_result.get("ok")):
                    registered_bundle_ids.append(bundle_id)
                else:
                    failed.append({
                        "bundle_id": bundle_id,
                        "crop_id": crop_id,
                        "error": str(postgres_result.get("error") or "postgres_register_failed"),
                    })
            except Exception as exc:
                failed.append({"bundle_id": bundle_id, "crop_id": crop_id, "error": type(exc).__name__})
            continue
        try:
            response = export_crop_analysis_bundle(set_num=set_text, bag=bag_number, crop_id=crop_id)
            payload = json.loads(response.body.decode("utf-8")) if isinstance(response, JSONResponse) else {}
            if not bool(payload.get("ok")):
                failed.append({"bundle_id": bundle_id, "crop_id": crop_id, "error": str(payload.get("error") or "export_failed")})
                continue
            register_result = register_analysis_bundle(bundle_id)
            registration_results[bundle_id] = register_result
            exported_bundle_ids.append(bundle_id)
            postgres_result = register_result.get("postgres") if isinstance(register_result.get("postgres"), dict) else {}
            if bool(register_result.get("ok")) and bool(postgres_result.get("ok")):
                registered_bundle_ids.append(bundle_id)
            else:
                failed.append({
                    "bundle_id": bundle_id,
                    "crop_id": crop_id,
                    "error": str(postgres_result.get("error") or "postgres_register_failed"),
                })
        except HTTPException as exc:
            failed.append({"bundle_id": bundle_id, "crop_id": crop_id, "error": str(exc.detail)})
        except Exception as exc:
            failed.append({"bundle_id": bundle_id, "crop_id": crop_id, "error": type(exc).__name__})

    return JSONResponse(
        {
            "ok": not failed,
            "set_num": set_text,
            "bag_num": bag_number,
            "page_range": {
                "rendered_pages": [int(page) for page in list(rendered_pages or [])],
                "start_page": int(start_page),
                "end_page": int(end_page),
            },
            "force": force_export,
            "crop_count": len(list(crops or [])),
            "exported_count": len(exported_bundle_ids),
            "registered_count": len(registered_bundle_ids),
            "skipped_existing_count": len(skipped_bundle_ids),
            "failed_count": len(failed),
            "exported_bundle_ids": exported_bundle_ids,
            "registered_bundle_ids": registered_bundle_ids,
            "skipped_bundle_ids": skipped_bundle_ids,
            "local_bundle_paths": local_bundle_paths,
            "registration_results": registration_results,
            "failed": failed,
        }
    )


def _training_store_analysis_root() -> Path:
    return (
        Path("/Users/olly/aim2build-instruction")
        / "debug"
        / "ai_training"
        / "analysis_bundles"
    )


def _auto_batch_bundle_candidates(set_num: str, bag_num: int) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, str]]:
    analysis_root = _training_store_analysis_root()
    set_text = str(set_num or "").strip()
    bag_number = int(bag_num or 1)
    expected_bundle_ids: List[str] = []
    discovery_errors: List[Dict[str, Any]] = []
    try:
        crops = _build_instruction_callout_crops(set_text, bag_number, ai_enabled=False)
        _write_crop_detection_cache(set_text, bag_number, crops)
        for crop in list(crops or []):
            crop_id = str((crop or {}).get("crop_id") or "").strip()
            if crop_id:
                expected_bundle_ids.append(_analysis_bundle_slug(set_text, bag_number, crop_id))
    except Exception as exc:
        discovery_errors.append({"error": type(exc).__name__, "detail": str(exc)})

    bundle_prefix = f"{set_text}_bag{bag_number}_"
    local_bundle_ids = [
        path.name
        for path in sorted(analysis_root.glob(f"{bundle_prefix}*"))
        if path.is_dir()
    ] if analysis_root.exists() else []

    postgres_bundle_ids: List[str] = []
    try:
        queue = list_review_queue(set_num=set_text, bag_num=bag_number, limit=500)
        postgres_bundle_ids = [
            str(row.get("bundle_id") or "")
            for row in list(queue.get("rows") or [])
            if isinstance(row, dict) and str(row.get("bundle_id") or "").startswith(bundle_prefix)
        ]
    except Exception as exc:
        discovery_errors.append({"error": type(exc).__name__, "detail": f"postgres list failed: {exc}"})

    ordered: List[str] = []
    for bundle_id in expected_bundle_ids + local_bundle_ids + postgres_bundle_ids:
        if bundle_id and bundle_id not in ordered:
            ordered.append(bundle_id)
    local_paths = {bundle_id: str(analysis_root / bundle_id) for bundle_id in ordered}
    return ordered, discovery_errors, local_paths


def _auto_batch_manifest_for_bundle(bundle_id: str, artifact_debug: Dict[str, Any]) -> Dict[str, Any]:
    metadata_path = Path(str(artifact_debug.get("metadata_path") or "").strip())
    if metadata_path.exists() and metadata_path.is_file():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}
    return {}


def _auto_batch_split_candidate_counts(split_candidate_paths: Any) -> Dict[str, Any]:
    paths = split_candidate_paths if isinstance(split_candidate_paths, dict) else {}
    candidates = [
        dict(item)
        for item in list(paths.get("candidates") or [])
        if isinstance(item, dict)
    ]
    review_state_counts = {
        "needs_mask_expand": 0,
        "needs_ocr_review": 0,
        "needs_manual_crop": 0,
        "rejected": 0,
    }
    accepted_count = 0
    for candidate in candidates:
        status = str(candidate.get("status") or "").strip()
        review_state = str(candidate.get("review_state") or "").strip()
        if status == "accepted":
            accepted_count += 1
        if status == "rejected":
            review_state_counts["rejected"] += 1
        if review_state in review_state_counts:
            review_state_counts[review_state] += 1
    return {
        "split_candidate_count": len(candidates),
        "split_candidates_exist": bool(candidates),
        "accepted_count": accepted_count,
        "review_state_counts": review_state_counts,
        "needs_review_count": sum(int(review_state_counts.get(key, 0) or 0) for key in review_state_counts),
    }


_TRAINING_REVIEW_COMPLETE_STATES = {"needs_mask_expand", "needs_ocr_review", "needs_manual_crop"}


def _training_review_candidate_confirmed(confirmed: Dict[str, Any]) -> bool:
    return any(
        str(confirmed.get(field) or "").strip()
        for field in ("part_num", "element_id", "confirmed_by")
    )


def _training_review_bundle_completion(row: Dict[str, Any]) -> Dict[str, Any]:
    bundle_id = str(row.get("bundle_id") or "").strip()
    paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
    candidates = [
        dict(item)
        for item in list(paths.get("candidates") or [])
        if isinstance(item, dict)
    ]
    try:
        confirmed_rows = [
            dict(item)
            for item in list(list_candidate_training_examples(bundle_id).get("rows") or [])
            if isinstance(item, dict)
        ]
    except Exception:
        confirmed_rows = []
    confirmed_by_index = {
        int(_coerce_int(item.get("candidate_index"))): item
        for item in confirmed_rows
        if _coerce_int(item.get("candidate_index")) is not None
    }
    incomplete_indexes: List[int] = []
    complete_count = 0
    for pos, candidate in enumerate(candidates):
        candidate_index = _coerce_int(candidate.get("index"))
        if candidate_index is None:
            candidate_index = pos
        status = str(candidate.get("status") or "").strip()
        review_state = str(candidate.get("review_state") or "").strip()
        confirmed = _training_review_candidate_confirmed(confirmed_by_index.get(int(candidate_index), {}))
        candidate_complete = bool(
            confirmed
            or status == "rejected"
            or review_state in _TRAINING_REVIEW_COMPLETE_STATES
        )
        if candidate_complete:
            complete_count += 1
        else:
            incomplete_indexes.append(int(candidate_index))
    is_complete = bool(candidates) and not incomplete_indexes
    return {
        "bundle_id": bundle_id,
        "complete": is_complete,
        "candidate_count": len(candidates),
        "complete_candidate_count": complete_count,
        "incomplete_candidate_indexes": incomplete_indexes,
        "confirmed_count": len(confirmed_rows),
    }


def _training_review_url(bundle_id: str, set_num: str, bag_num: Any, limit: int = 50) -> str:
    return (
        f"/debug/training-store/review-ui?bundle_id={_url_quote(str(bundle_id or ''))}"
        f"&review_status=pending&set_num={_url_quote(str(set_num or ''))}"
        f"&bag_num={_url_quote(str(bag_num or ''))}&limit={int(limit or 50)}"
    )


def _next_pending_bundle_payload(
    *,
    set_num: str,
    bag_num: Any,
    current_bundle_id: str = "",
    limit: int = 50,
) -> Dict[str, Any]:
    set_text = str(set_num or "").strip()
    bag_number = _coerce_int(bag_num)
    if not set_text:
        raise ValueError("set_num is required")
    if bag_number is None:
        raise ValueError("bag_num is required")

    queue = list_review_queue(set_num=set_text, bag_num=bag_number, limit=500)
    rows = [dict(row) for row in list(queue.get("rows") or []) if isinstance(row, dict)]
    incomplete_rows: List[Dict[str, Any]] = []
    completion_by_bundle: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        completion = _training_review_bundle_completion(row)
        completion_by_bundle[str(row.get("bundle_id") or "")] = completion
        if not bool(completion.get("complete")):
            incomplete_rows.append(row)

    incomplete_bundle_ids = [str(row.get("bundle_id") or "") for row in incomplete_rows]
    current_id = str(current_bundle_id or "").strip()
    next_row: Optional[Dict[str, Any]] = None
    if current_id and current_id in incomplete_bundle_ids:
        current_index = incomplete_bundle_ids.index(current_id)
        if current_index + 1 < len(incomplete_rows):
            next_row = incomplete_rows[current_index + 1]
    elif incomplete_rows:
        next_row = incomplete_rows[0]

    current_bag_complete = not incomplete_rows
    next_bag_num: Optional[int] = None
    next_bag_bundle_id = ""
    next_bag_url = ""
    if current_bag_complete:
        try:
            set_queue = list_review_queue(set_num=set_text, limit=500)
            set_rows = [dict(row) for row in list(set_queue.get("rows") or []) if isinstance(row, dict)]
        except Exception:
            set_rows = []
        bag_values = sorted({
            int(value)
            for value in (_coerce_int(row.get("bag_num")) for row in set_rows)
            if value is not None and int(value) > int(bag_number)
        })
        for candidate_bag in bag_values:
            bag_rows = [row for row in set_rows if _coerce_int(row.get("bag_num")) == candidate_bag]
            for row in bag_rows:
                if not bool(_training_review_bundle_completion(row).get("complete")):
                    next_bag_num = int(candidate_bag)
                    next_bag_bundle_id = str(row.get("bundle_id") or "")
                    next_bag_url = _training_review_url(next_bag_bundle_id, set_text, next_bag_num, limit)
                    break
            if next_bag_bundle_id:
                break

    next_bundle_id = str(next_row.get("bundle_id") or "") if next_row else ""
    next_url = _training_review_url(next_bundle_id, set_text, bag_number, limit) if next_bundle_id else ""
    return {
        "ok": True,
        "set_num": set_text,
        "bag_num": int(bag_number),
        "current_bundle_id": current_id,
        "next_bundle_id": next_bundle_id,
        "next_url": next_url,
        "current_bag_complete": current_bag_complete,
        "next_bag_num": next_bag_num,
        "next_bag_bundle_id": next_bag_bundle_id,
        "next_bag_url": next_bag_url,
        "incomplete_bundle_count": len(incomplete_rows),
        "bundle_completion": completion_by_bundle.get(current_id, {}),
    }


@router.get("/debug/training-store/next-pending-bundle")
def training_store_next_pending_bundle(
    set_num: str = Query(...),
    bag_num: int = Query(...),
    current_bundle_id: str = Query(""),
):
    try:
        return JSONResponse(
            _next_pending_bundle_payload(
                set_num=set_num,
                bag_num=bag_num,
                current_bundle_id=current_bundle_id,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/debug/training-store/bag-completion-audit")
def training_store_bag_completion_audit(
    set_num: str = Query(...),
    bag_num: int = Query(...),
):
    """
    Bag-level completion audit: verifies OCR qty totals match confirmed mappings
    and surfaces any unmapped or inconsistent candidates.
    """
    set_text = str(set_num or "").strip()
    bag_number = int(bag_num or 1)
    if not set_text:
        raise HTTPException(status_code=400, detail="set_num is required")

    # ── 1. Collect all bundles for this bag ──────────────────────────────
    queue_result = list_review_queue(set_num=set_text, bag_num=bag_number, limit=500)
    bundle_rows = [dict(r) for r in list(queue_result.get("rows") or []) if isinstance(r, dict)]

    # ── 2. For each bundle collect candidate + confirmed-example data ────
    bundle_summaries: List[Dict[str, Any]] = []
    total_ocr_qty = 0
    total_confirmed_qty = 0
    review_state_counts: Dict[str, int] = {}
    unmapped_accepted: List[Dict[str, Any]] = []   # accepted, qty-clean, no confirmed example
    qty_no_confirmed: List[Dict[str, Any]] = []    # qty detected but no mapping
    qty_mismatch: List[Dict[str, Any]] = []        # confirmed qty ≠ detected qty
    skipped_problem: List[Dict[str, Any]] = []     # rejected / non-empty review_state

    for b_row in bundle_rows:
        bundle_id = str(b_row.get("bundle_id") or "")
        paths = b_row.get("split_candidate_paths") if isinstance(b_row.get("split_candidate_paths"), dict) else {}
        candidates = [dict(c) for c in list(paths.get("candidates") or []) if isinstance(c, dict)]

        # Confirmed examples for this bundle keyed by candidate_index
        try:
            ex_rows = [dict(r) for r in list(list_candidate_training_examples(bundle_id).get("rows") or []) if isinstance(r, dict)]
        except Exception:
            ex_rows = []
        confirmed_by_idx: Dict[int, Dict[str, Any]] = {}
        for ex in ex_rows:
            idx = _coerce_int(ex.get("candidate_index"))
            if idx is not None:
                confirmed_by_idx[int(idx)] = ex
            total_confirmed_qty += int(ex.get("qty") or 1)

        for c in candidates:
            cidx = _coerce_int(c.get("index"))
            status = str(c.get("status") or "pending")
            review_state = str(c.get("review_state") or "")
            qty_detected = bool(c.get("qty_detected"))
            qty_values = list(c.get("qty_values") or [])
            ocr_qty = int(qty_values[0]) if qty_values else 0

            # review_state counts
            rs_key = review_state if review_state else "clean"
            review_state_counts[rs_key] = review_state_counts.get(rs_key, 0) + 1

            # skipped/problem
            if status == "rejected" or review_state:
                skipped_problem.append({
                    "bundle_id": bundle_id,
                    "candidate_index": cidx,
                    "status": status,
                    "review_state": review_state,
                    "qty_detected": qty_detected,
                    "qty_values": qty_values,
                })

            if status != "accepted":
                continue

            # qty totals (accepted candidates only)
            if qty_detected and ocr_qty > 0:
                total_ocr_qty += ocr_qty

            confirmed = confirmed_by_idx.get(int(cidx)) if cidx is not None else None
            is_confirmed = _training_review_candidate_confirmed(confirmed or {}) if confirmed else False

            if not is_confirmed and _candidate_qty_is_clean(c) and not review_state:
                unmapped_accepted.append({
                    "bundle_id": bundle_id,
                    "candidate_index": cidx,
                    "qty_values": qty_values,
                    "qty_scrub_status": c.get("qty_scrub_status"),
                })

            if qty_detected and not is_confirmed:
                qty_no_confirmed.append({
                    "bundle_id": bundle_id,
                    "candidate_index": cidx,
                    "qty_values": qty_values,
                })

            if is_confirmed and qty_detected and confirmed:
                confirmed_qty_val = int(confirmed.get("qty") or 1)
                if ocr_qty and confirmed_qty_val != ocr_qty:
                    qty_mismatch.append({
                        "bundle_id": bundle_id,
                        "candidate_index": cidx,
                        "ocr_qty": ocr_qty,
                        "confirmed_qty": confirmed_qty_val,
                        "part_num": str(confirmed.get("part_num") or ""),
                        "color_id": _coerce_int(confirmed.get("color_id")),
                    })

        bundle_summaries.append({
            "bundle_id": bundle_id,
            "candidate_count": len(candidates),
            "confirmed_count": len(ex_rows),
            "review_status": str(b_row.get("review_status") or ""),
        })

    # ── 3. Per-part summary from set totals ──────────────────────────────
    per_part_summary: List[Dict[str, Any]] = []
    try:
        set_parts_payload = load_instruction_set_parts(set_text)
        set_parts = _prepare_instruction_parts_for_display(list(set_parts_payload.get("parts", []) or []))
        totals_rows = [dict(r) for r in list(list_confirmed_part_totals_for_set(set_text).get("rows") or []) if isinstance(r, dict)]
        confirmed_totals: Dict[str, int] = {}
        for t in totals_rows:
            k = _candidate_part_key(t.get("part_num"), t.get("color_id"))
            confirmed_totals[k] = int(t.get("confirmed_qty") or 0)
        for part in set_parts:
            part_num = str(part.get("part_num") or "").strip()
            color_id = _coerce_int(part.get("color_id"))
            if not part_num or color_id is None:
                continue
            k = _candidate_part_key(part_num, color_id)
            set_required = int(part.get("set_required_qty") or 0)
            confirmed = int(confirmed_totals.get(k) or 0)
            remaining = set_required - confirmed
            if set_required > 0 or confirmed > 0:
                per_part_summary.append({
                    "part_num": part_num,
                    "color_id": color_id,
                    "color_name": str(part.get("color_name") or ""),
                    "set_required_qty": set_required,
                    "confirmed_qty": confirmed,
                    "remaining_qty": remaining,
                    "complete": remaining <= 0,
                })
        per_part_summary.sort(key=lambda r: (r["remaining_qty"] < 0, abs(r["remaining_qty"]) == 0, r["part_num"]))
    except Exception as exc:
        per_part_summary = [{"error": str(exc)}]

    # ── 4. High-level verdict ─────────────────────────────────────────────
    incomplete_parts = [p for p in per_part_summary if isinstance(p.get("remaining_qty"), int) and p["remaining_qty"] > 0]
    over_confirmed_parts = [p for p in per_part_summary if isinstance(p.get("remaining_qty"), int) and p["remaining_qty"] < 0]
    qty_match = total_ocr_qty == total_confirmed_qty

    audit = {
        "ok": True,
        "set_num": set_text,
        "bag_num": bag_number,
        "bundle_count": len(bundle_rows),
        "verdict": {
            "qty_totals_match": qty_match,
            "has_unmapped_accepted": len(unmapped_accepted) > 0,
            "has_incomplete_parts": len(incomplete_parts) > 0,
            "has_over_confirmed": len(over_confirmed_parts) > 0,
            "has_qty_mismatch": len(qty_mismatch) > 0,
        },
        "totals": {
            "total_ocr_qty": total_ocr_qty,
            "total_confirmed_qty": total_confirmed_qty,
        },
        "unmapped_accepted": unmapped_accepted,
        "qty_no_confirmed": qty_no_confirmed,
        "qty_mismatch": qty_mismatch,
        "review_state_counts": review_state_counts,
        "skipped_problem": skipped_problem,
        "incomplete_parts": incomplete_parts,
        "over_confirmed_parts": over_confirmed_parts,
        "per_part_summary": per_part_summary,
        "bundle_summaries": bundle_summaries,
    }

    # ── 5. Render HTML ─────────────────────────────────────────────────────
    def _verdict_row(label: str, ok: bool, ok_text: str = "OK", fail_text: str = "⚠ Issue") -> str:
        cls = "audit-ok" if ok else "audit-warn"
        return f'<tr><td>{escape(label)}</td><td class="{cls}">{ok_text if ok else fail_text}</td></tr>'

    def _part_rows(parts: List[Dict[str, Any]]) -> str:
        out = []
        for p in parts:
            if "error" in p:
                out.append(f'<tr><td colspan="6" class="audit-warn">{escape(str(p["error"]))}</td></tr>')
                continue
            remaining = int(p.get("remaining_qty") or 0)
            cls = "audit-warn" if remaining > 0 else ("audit-over" if remaining < 0 else "audit-ok")
            out.append(
                f'<tr>'
                f'<td>{escape(str(p.get("part_num") or ""))}</td>'
                f'<td>{escape(str(p.get("color_id") or ""))}</td>'
                f'<td>{escape(str(p.get("color_name") or ""))}</td>'
                f'<td>{escape(str(p.get("set_required_qty") or 0))}</td>'
                f'<td>{escape(str(p.get("confirmed_qty") or 0))}</td>'
                f'<td class="{cls}">{escape(str(remaining))}</td>'
                f'</tr>'
            )
        return "".join(out) or '<tr><td colspan="6">—</td></tr>'

    def _candidate_list(items: List[Dict[str, Any]], cols: List[str]) -> str:
        if not items:
            return '<p class="audit-ok">None.</p>'
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{escape(str(item.get(c, '') or ''))}</td>" for c in cols) + "</tr>"
            for item in items
        )
        header = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr>"
        return f'<table class="audit-table"><thead>{header}</thead><tbody>{rows_html}</tbody></table>'

    qty_match_class = "audit-ok" if qty_match else "audit-warn"
    next_bag_num_for_url, next_bag_url_for_link = _quick_review_next_bag_url(set_text, bag_number)
    next_bag_link = (
        f'<a href="{escape(next_bag_url_for_link)}">Open Next Bag {escape(str(next_bag_num_for_url))}</a>'
        if next_bag_url_for_link else ""
    )
    training_review_link = (
        f'<a href="/debug/training-store/training-review?set_num={_url_quote(set_text)}&bag_num={bag_number}">Back to Training Review</a>'
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Bag Completion Audit — {escape(set_text)} bag {bag_number}</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f5f7f8; color:#172026; padding:24px; }}
    h1 {{ font-size:20px; margin:0 0 4px; }}
    .subtitle {{ color:#66727d; font-size:14px; margin-bottom:20px; }}
    .nav {{ display:flex; gap:12px; margin-bottom:24px; font-size:13px; }}
    .nav a {{ color:#0969da; text-decoration:none; }} .nav a:hover {{ text-decoration:underline; }}
    section {{ background:#fff; border:1px solid #d0d7de; border-radius:8px; padding:16px 20px; margin-bottom:16px; }}
    h2 {{ font-size:15px; margin:0 0 12px; }}
    table.verdict-table {{ border-collapse:collapse; width:100%; }}
    table.verdict-table td {{ padding:6px 10px; border-bottom:1px solid #eee; font-size:13px; }}
    table.verdict-table td:first-child {{ color:#66727d; width:60%; }}
    .audit-ok {{ color:#1a7f37; font-weight:600; }}
    .audit-warn {{ color:#9a6700; font-weight:600; }}
    .audit-over {{ color:#cf222e; font-weight:600; }}
    table.audit-table {{ border-collapse:collapse; width:100%; font-size:12px; margin-top:6px; }}
    table.audit-table th {{ background:#f6f8fa; padding:5px 8px; text-align:left; border-bottom:2px solid #d0d7de; }}
    table.audit-table td {{ padding:4px 8px; border-bottom:1px solid #eee; }}
    .totals-row {{ display:flex; gap:24px; font-size:13px; margin-bottom:8px; }}
    .totals-row span {{ background:#f6f8fa; border:1px solid #d0d7de; border-radius:4px; padding:4px 10px; }}
    details.raw-json summary {{ cursor:pointer; color:#66727d; font-size:12px; }}
    pre {{ font-size:11px; overflow-x:auto; background:#f6f8fa; border:1px solid #d0d7de; border-radius:4px; padding:10px; }}
  </style>
</head>
<body>
  <h1>Bag Completion Audit</h1>
  <p class="subtitle">{escape(set_text)} · Bag {bag_number} · {len(bundle_rows)} bundle{"s" if len(bundle_rows) != 1 else ""}</p>
  <nav class="nav">{training_review_link}{(" · " + next_bag_link) if next_bag_link else ""}</nav>

  <section>
    <h2>Verdict</h2>
    <table class="verdict-table">
      {_verdict_row("OCR qty total matches confirmed qty total", qty_match,
          f"Yes ({total_ocr_qty} = {total_confirmed_qty})",
          f"No — OCR {total_ocr_qty} vs confirmed {total_confirmed_qty}")}
      {_verdict_row("All accepted candidates mapped", not unmapped_accepted,
          "Yes", f"{len(unmapped_accepted)} unmapped accepted candidate(s)")}
      {_verdict_row("All set parts confirmed", not incomplete_parts,
          "Yes", f"{len(incomplete_parts)} part(s) still needed")}
      {_verdict_row("No over-confirmed parts", not over_confirmed_parts,
          "Yes", f"{len(over_confirmed_parts)} over-confirmed part(s)")}
      {_verdict_row("No qty mismatches", not qty_mismatch,
          "Yes", f"{len(qty_mismatch)} mismatch(es)")}
    </table>
  </section>

  <section>
    <h2>Totals</h2>
    <div class="totals-row">
      <span>OCR qty total (accepted): <strong>{total_ocr_qty}</strong></span>
      <span>Confirmed qty total: <strong>{total_confirmed_qty}</strong></span>
      <span class="{qty_match_class}">{"Match ✓" if qty_match else "Mismatch ✗"}</span>
    </div>
  </section>

  <section>
    <h2>Per-Part Summary</h2>
    <table class="audit-table">
      <thead><tr><th>part_num</th><th>color_id</th><th>color_name</th><th>set_required_qty</th><th>confirmed_qty</th><th>remaining_qty</th></tr></thead>
      <tbody>{_part_rows(per_part_summary)}</tbody>
    </table>
  </section>

  <section>
    <h2>Unmapped Accepted Candidates ({len(unmapped_accepted)})</h2>
    {_candidate_list(unmapped_accepted, ["bundle_id", "candidate_index", "qty_values", "qty_scrub_status"])}
  </section>

  <section>
    <h2>Candidates with Qty but No Confirmed Part ({len(qty_no_confirmed)})</h2>
    {_candidate_list(qty_no_confirmed, ["bundle_id", "candidate_index", "qty_values"])}
  </section>

  <section>
    <h2>Qty Mismatches ({len(qty_mismatch)})</h2>
    {_candidate_list(qty_mismatch, ["bundle_id", "candidate_index", "part_num", "color_id", "ocr_qty", "confirmed_qty"])}
  </section>

  <section>
    <h2>Skipped / Problem Candidates ({len(skipped_problem)})</h2>
    {_candidate_list(skipped_problem, ["bundle_id", "candidate_index", "status", "review_state", "qty_detected"])}
  </section>

  <section>
    <h2>Review State Counts</h2>
    <table class="audit-table">
      <thead><tr><th>review_state</th><th>count</th></tr></thead>
      <tbody>{"".join(f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>" for k, v in sorted(review_state_counts.items()))}</tbody>
    </table>
  </section>

  <section>
    <h2>Bundles ({len(bundle_summaries)})</h2>
    {_candidate_list(bundle_summaries, ["bundle_id", "candidate_count", "confirmed_count", "review_status"])}
  </section>

  <section>
    <details class="raw-json">
      <summary>Raw JSON</summary>
      <pre>{escape(json.dumps(audit, indent=2, default=str))}</pre>
    </details>
  </section>
</body>
</html>"""

    return HTMLResponse(html)


# ── Training Control Panel ──────────────────────────────────────────────────

@router.get("/debug/training-store/control-panel", response_class=HTMLResponse)
def training_store_control_panel(
    set_num: str = Query(...),
    bag_num: int = Query(...),
):
    set_text = str(set_num or "").strip()
    if not set_text:
        raise HTTPException(status_code=400, detail="set_num is required")
    bag_number = int(bag_num or 1)

    # ── 1. Bag-level review status counts (one DB query) ─────────────────
    queue_result = list_review_queue(set_num=set_text, bag_num=bag_number, limit=500)
    queue_rows = [dict(r) for r in list(queue_result.get("rows") or []) if isinstance(r, dict)]
    total_bundles = len(queue_rows)
    approved_count = sum(1 for r in queue_rows if str(r.get("review_status") or "") == "approved")
    bad_mask_count = sum(1 for r in queue_rows if str(r.get("review_status") or "") == "bad_mask")
    pending_count = sum(
        1 for r in queue_rows
        if str(r.get("review_status") or "") not in {"approved", "bad_mask", "rejected", "needs_split_fix"}
    )

    # ── 2. Quick-review cache (file read, best-effort) ────────────────────
    cache = _read_quick_review_cache(set_text, bag_number)
    cache_exists = bool(cache)
    cache_stale = bool(cache.get("stale"))
    cache_built_at = str(cache.get("built_at") or "")
    cache_stale_reason = str(cache.get("stale_reason") or "")
    cache_summary = dict(cache.get("summary") or {})
    problem_count = int(cache_summary.get("problem_bundle_count", 0) or 0)
    needs_review_count = int(cache_summary.get("needs_review_count", 0) or 0)
    missing_candidates_count = int(cache_summary.get("missing_candidates_count", 0) or 0)

    # ── 3. Next bag ───────────────────────────────────────────────────────
    next_bag_num, next_bag_url = _quick_review_next_bag_url(set_text, bag_number)

    # ── 4. URL helpers ────────────────────────────────────────────────────
    qs = f"set_num={_url_quote(set_text)}&bag_num={bag_number}"
    quick_review_url = f"/debug/training-store/quick-review?{qs}"
    review_queue_url = f"/debug/training-store/review-ui?{qs}&review_status=pending&limit=50"
    audit_url = f"/debug/training-store/bag-completion-audit?{qs}"
    export_url = f"/debug/export-training-data?set_num={_url_quote(set_text)}&bag={bag_number}"
    control_panel_url = f"/debug/training-store/control-panel?{qs}"
    next_bag_control_panel_url = (
        f"/debug/training-store/control-panel?set_num={_url_quote(set_text)}&bag_num={next_bag_num}"
        if next_bag_num else ""
    )

    # ── 5. Status grid ────────────────────────────────────────────────────
    def _stat(label: str, value: Any, cls: str = "") -> str:
        cls_attr = f' class="{escape(cls)}"' if cls else ""
        return (
            f'<div{cls_attr}>'
            f'<span class="stat-label">{escape(label)}</span>'
            f'<strong class="stat-value">{escape(str(value))}</strong>'
            f'</div>'
        )

    status_grid = "".join([
        _stat("bundles", total_bundles),
        _stat("problems", problem_count if cache_exists else "—", "warn" if problem_count > 0 else ""),
        _stat("approved", approved_count, "ok" if approved_count > 0 else ""),
        _stat("bad mask", bad_mask_count, "warn" if bad_mask_count > 0 else ""),
        _stat("pending", pending_count, "warn" if pending_count > 0 else ""),
        _stat("needs review", needs_review_count if cache_exists else "—", "warn" if needs_review_count > 0 else ""),
        _stat("missing candidates", missing_candidates_count if cache_exists else "—", "warn" if missing_candidates_count > 0 else ""),
    ])

    if cache_exists:
        if cache_stale:
            cache_status_text = f"stale · {escape(cache_stale_reason)} · last built: {escape(cache_built_at)}"
            cache_status_cls = "cache-status stale"
        else:
            cache_status_text = f"cache ready · built: {escape(cache_built_at)}"
            cache_status_cls = "cache-status ready"
    else:
        cache_status_text = "no cache — build it first"
        cache_status_cls = "cache-status missing"

    next_bag_btn = (
        f'<a class="btn btn-success" href="{escape(next_bag_control_panel_url)}" target="_blank">'
        f'Open Next Bag {escape(str(next_bag_num))}</a>'
        if next_bag_num else
        '<span class="btn btn-disabled" title="No later bag found">Open Next Bag</span>'
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Control Panel — {escape(set_text)} bag {escape(str(bag_number))}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; font-size:14px; color:#172026; background:#f5f7f8; }}
    header {{ display:flex; justify-content:space-between; align-items:center; gap:16px; padding:14px 18px; background:#18212b; color:white; }}
    header strong {{ font-size:18px; }}
    header a {{ color:#a8c6e8; text-decoration:none; font-size:13px; }}
    main {{ padding:20px; max-width:860px; margin:0 auto; }}
    h2 {{ margin:24px 0 10px; font-size:15px; text-transform:uppercase; letter-spacing:.05em; color:#66727d; }}
    /* Status grid */
    .status-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:10px; margin-bottom:8px; }}
    .status-grid div {{ background:white; border:1px solid #d8dee3; border-radius:8px; padding:12px 10px; text-align:center; }}
    .status-grid div.warn {{ border-color:#e8a020; background:#fffbf3; }}
    .status-grid div.ok {{ border-color:#3a9a60; background:#f2fbf5; }}
    .stat-label {{ display:block; font-size:11px; color:#66727d; margin-bottom:4px; text-transform:uppercase; letter-spacing:.04em; }}
    .stat-value {{ font-size:26px; font-weight:800; }}
    .cache-status {{ font-size:12px; padding:6px 10px; border-radius:6px; margin-bottom:18px; }}
    .cache-status.ready {{ background:#f2fbf5; color:#2f6c41; border:1px solid #7db28a; }}
    .cache-status.stale {{ background:#fffbf3; color:#8a5a00; border:1px solid #e8a020; font-weight:700; }}
    .cache-status.missing {{ background:#fdf3f3; color:#8a2020; border:1px solid #e09090; }}
    /* Action groups */
    .actions {{ display:flex; flex-direction:column; gap:12px; }}
    .action-row {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
    .action-row .label {{ min-width:200px; color:#66727d; font-size:13px; }}
    /* Buttons */
    .btn {{ display:inline-flex; align-items:center; justify-content:center; min-height:36px; padding:8px 14px; border-radius:6px; border:1px solid #1f6fc9; background:#1f6fc9; color:white; font:inherit; font-size:13px; font-weight:700; text-decoration:none; cursor:pointer; white-space:nowrap; }}
    .btn:hover {{ filter:brightness(1.1); }}
    .btn-secondary {{ background:#f5f7f8; color:#172026; border-color:#c8d0d7; }}
    .btn-success {{ background:#2f6c41; border-color:#2f6c41; }}
    .btn-warn {{ background:#755000; border-color:#755000; }}
    .btn-disabled {{ background:#e4e9ed; border-color:#c8d0d7; color:#9aa6af; cursor:default; pointer-events:none; }}
    .btn[data-loading]::after {{ content:" ···"; }}
    /* Log */
    #action-log {{ margin-top:20px; }}
    .log-entry {{ padding:10px 12px; border-radius:6px; font-size:12px; margin-bottom:8px; white-space:pre-wrap; word-break:break-all; }}
    .log-entry.ok {{ background:#f2fbf5; border:1px solid #7db28a; color:#1a4a28; }}
    .log-entry.err {{ background:#fdf3f3; border:1px solid #e09090; color:#5a1010; }}
    .log-entry.info {{ background:#f0f4ff; border:1px solid #8aaae8; color:#1a2a6c; }}
    @media (max-width:640px) {{
      .status-grid {{ grid-template-columns:repeat(3,1fr); }}
      .action-row {{ flex-direction:column; align-items:flex-start; }}
      .action-row .label {{ min-width:0; }}
    }}
  </style>
</head>
<body>
  <header>
    <strong>Training Control Panel</strong>
    <span>{escape(set_text)} · bag {escape(str(bag_number))}</span>
    <a href="{escape(control_panel_url)}">↻ Refresh status</a>
  </header>
  <main>
    <h2>Status</h2>
    <div class="status-grid">{status_grid}</div>
    <div class="{escape(cache_status_cls)}">{cache_status_text}</div>

    <h2>Actions</h2>
    <div class="actions">

      <div class="action-row">
        <button class="btn" type="button"
          data-post="/debug/training-store/build-quick-review-cache?{escape(qs)}"
          data-summary-key="summary"
        >Build / Refresh Quick Review Cache</button>
        <a class="btn btn-secondary" href="{escape(quick_review_url)}" target="_blank">Open Quick Review</a>
      </div>

      <div class="action-row">
        <a class="btn btn-secondary" href="{escape(review_queue_url)}" target="_blank">Open Review Queue</a>
      </div>

      <div class="action-row">
        <a class="btn btn-secondary" href="{escape(audit_url)}" target="_blank">Open Bag Completion Audit</a>
      </div>

      <div class="action-row">
        <button class="btn btn-warn" type="button"
          data-post="/debug/training-store/auto-batch-report?{escape(qs)}&amp;generate_missing=1"
          data-summary-key="summary"
          data-confirm="Generate missing split candidates for all eligible bundles in this bag?"
        >Generate Missing Split Candidates</button>
      </div>

      <div class="action-row">
        {next_bag_btn}
      </div>

      <div class="action-row">
        <a class="btn btn-secondary" href="{escape(export_url)}" target="_blank">Export Confirmed Training Data</a>
      </div>

    </div>

    <div id="action-log"></div>
  </main>
  <script>
    const log = document.getElementById('action-log');

    function addLog(text, cls) {{
      const el = document.createElement('div');
      el.className = 'log-entry ' + cls;
      el.textContent = text;
      log.prepend(el);
    }}

    document.querySelectorAll('[data-post]').forEach(function(btn) {{
      btn.addEventListener('click', async function() {{
        const url = btn.dataset.post;
        const confirmMsg = btn.dataset.confirm;
        const summaryKey = btn.dataset.summaryKey;
        if (confirmMsg && !confirm(confirmMsg)) return;
        btn.setAttribute('data-loading', '1');
        btn.disabled = true;
        addLog('POST ' + url + ' …', 'info');
        try {{
          const resp = await fetch(url, {{ method: 'POST' }});
          const text = await resp.text();
          let parsed = null;
          try {{ parsed = JSON.parse(text); }} catch (_) {{}}
          if (!resp.ok) {{
            addLog('Error ' + resp.status + ': ' + text.slice(0, 400), 'err');
          }} else {{
            let summary = '';
            if (parsed && summaryKey && parsed[summaryKey]) {{
              summary = JSON.stringify(parsed[summaryKey], null, 2);
            }} else if (parsed) {{
              const short = {{}};
              ['ok','set_num','bag_num','built_at','stale','cache_path','generate_missing'].forEach(function(k) {{
                if (parsed[k] !== undefined) short[k] = parsed[k];
              }});
              summary = JSON.stringify(short, null, 2);
            }} else {{
              summary = text.slice(0, 400);
            }}
            addLog('OK ✓\n' + summary, 'ok');
          }}
        }} catch (err) {{
          addLog('Network error: ' + err.message, 'err');
        }} finally {{
          btn.removeAttribute('data-loading');
          btn.disabled = false;
        }}
      }});
    }});
  </script>
</body>
</html>"""

    return HTMLResponse(html)


def _auto_batch_report_payload(
    *,
    set_num: str,
    bag_num: int,
    generate_missing: int = 0,
) -> Dict[str, Any]:
    set_text = str(set_num or "").strip()
    if not set_text:
        raise HTTPException(status_code=400, detail="set_num is required")
    bag_number = int(bag_num or 1)
    should_generate_missing = int(generate_missing or 0) == 1
    bundle_ids, discovery_errors, local_bundle_paths = _auto_batch_bundle_candidates(set_text, bag_number)

    rows: List[Dict[str, Any]] = []
    summary = {
        "total_bundles": 0,
        "ready_count": 0,
        "missing_crop_count": 0,
        "missing_mask_count": 0,
        "missing_ocr_count": 0,
        "missing_candidates_count": 0,
        "accepted_count": 0,
        "confirmed_count": 0,
        "needs_review_count": 0,
        "failed_count": 0,
    }

    for bundle_id in bundle_ids:
        reasons: List[str] = []
        postgres_row_exists = False
        postgres_row: Dict[str, Any] = {}
        try:
            postgres_row = dict(get_training_bundle_index_row(bundle_id).get("row") or {})
            postgres_row_exists = bool(postgres_row)
        except FileNotFoundError:
            reasons.append("postgres_row_missing")
        except Exception as exc:
            reasons.append(f"postgres_row_error:{type(exc).__name__}")

        try:
            artifact_debug = _training_bundle_artifact_debug(bundle_id)
        except Exception as exc:
            artifact_debug = {
                "bundle_id": bundle_id,
                "local_folder": local_bundle_paths.get(bundle_id, ""),
                "local_folder_exists": False,
                "metadata_path": "",
                "metadata_exists": False,
            }
            reasons.append(f"artifact_debug_error:{type(exc).__name__}")
        metadata = _auto_batch_manifest_for_bundle(bundle_id, artifact_debug)

        local_folder_exists = bool(artifact_debug.get("local_folder_exists"))
        original_crop = artifact_debug.get("original_crop") if isinstance(artifact_debug.get("original_crop"), dict) else {}
        raw_master_mask = artifact_debug.get("raw_master_mask") if isinstance(artifact_debug.get("raw_master_mask"), dict) else {}
        original_crop_exists = bool(original_crop.get("exists"))
        raw_master_mask_exists = bool(raw_master_mask.get("exists"))
        direct_qty_ocr_boxes_count = len([
            item for item in list(metadata.get("direct_qty_ocr_boxes") or [])
            if isinstance(item, dict)
        ])

        if not local_folder_exists:
            reasons.append("local_folder_missing")
        if not original_crop_exists:
            reasons.append("original_crop_missing")
        if not raw_master_mask_exists:
            reasons.append("raw_master_mask_missing")
        if direct_qty_ocr_boxes_count <= 0:
            reasons.append("direct_qty_ocr_missing")

        split_candidate_paths = postgres_row.get("split_candidate_paths") if isinstance(postgres_row.get("split_candidate_paths"), dict) else {}
        split_counts = _auto_batch_split_candidate_counts(split_candidate_paths)
        generated_missing_candidates = False
        generate_error = ""
        if not bool(split_counts.get("split_candidates_exist")):
            if should_generate_missing and postgres_row_exists and original_crop_exists and raw_master_mask_exists:
                try:
                    generate_result = generate_split_candidates(bundle_id)
                    generated_missing_candidates = bool(generate_result.get("ok"))
                    updated_row = generate_result.get("row") if isinstance(generate_result.get("row"), dict) else {}
                    if updated_row:
                        postgres_row = dict(updated_row)
                    split_candidate_paths = postgres_row.get("split_candidate_paths") if isinstance(postgres_row.get("split_candidate_paths"), dict) else {}
                    split_counts = _auto_batch_split_candidate_counts(split_candidate_paths)
                except Exception as exc:
                    generate_error = f"{type(exc).__name__}: {exc}"
                    reasons.append("split_candidate_generation_failed")
            if not bool(split_counts.get("split_candidates_exist")):
                reasons.append("split_candidates_missing")

        confirmed_count = 0
        confirmed_by_index: Dict[int, Dict[str, Any]] = {}
        if postgres_row_exists:
            try:
                confirmed_rows = [
                    dict(item)
                    for item in list(list_candidate_training_examples(bundle_id).get("rows") or [])
                    if isinstance(item, dict)
                ]
                confirmed_count = len(confirmed_rows)
                confirmed_by_index = {
                    int(_coerce_int(item.get("candidate_index"))): item
                    for item in confirmed_rows
                    if _coerce_int(item.get("candidate_index")) is not None
                }
            except Exception as exc:
                reasons.append(f"confirmed_examples_error:{type(exc).__name__}")

        accepted_count = int(split_counts.get("accepted_count", 0) or 0)
        candidates = [
            dict(item)
            for item in list((split_candidate_paths if isinstance(split_candidate_paths, dict) else {}).get("candidates") or [])
            if isinstance(item, dict)
        ]
        unconfirmed_accepted_count = 0
        for pos, candidate in enumerate(candidates):
            candidate_index = _coerce_int(candidate.get("index"))
            if candidate_index is None:
                candidate_index = pos
            if str(candidate.get("status") or "").strip() == "accepted" and not _training_review_candidate_confirmed(confirmed_by_index.get(int(candidate_index), {})):
                unconfirmed_accepted_count += 1
        if unconfirmed_accepted_count > 0:
            reasons.append("unconfirmed_accepted_candidates")
        needs_review_count = int(split_counts.get("needs_review_count", 0) or 0)
        ready = bool(
            postgres_row_exists
            and local_folder_exists
            and original_crop_exists
            and raw_master_mask_exists
            and direct_qty_ocr_boxes_count > 0
            and bool(split_counts.get("split_candidates_exist"))
            and needs_review_count == 0
            and unconfirmed_accepted_count == 0
        )
        failed = any(
            reason.endswith("_missing")
            or reason.startswith("postgres_row_error")
            or reason.startswith("artifact_debug_error")
            or reason == "split_candidate_generation_failed"
            for reason in reasons
        )

        summary["missing_crop_count"] += 0 if original_crop_exists else 1
        summary["missing_mask_count"] += 0 if raw_master_mask_exists else 1
        summary["missing_ocr_count"] += 0 if direct_qty_ocr_boxes_count > 0 else 1
        summary["missing_candidates_count"] += 0 if bool(split_counts.get("split_candidates_exist")) else 1
        summary["accepted_count"] += accepted_count
        summary["confirmed_count"] += confirmed_count
        summary["needs_review_count"] += needs_review_count
        summary["ready_count"] += 1 if ready else 0
        summary["failed_count"] += 1 if failed else 0

        rows.append({
            "bundle_id": bundle_id,
            "postgres_row_exists": postgres_row_exists,
            "local_folder_exists": local_folder_exists,
            "local_folder": str(artifact_debug.get("local_folder") or local_bundle_paths.get(bundle_id, "")),
            "metadata_exists": bool(artifact_debug.get("metadata_exists")),
            "metadata_path": str(artifact_debug.get("metadata_path") or ""),
            "original_crop_exists": original_crop_exists,
            "original_crop_path": str(original_crop.get("path") or ""),
            "raw_master_mask_exists": raw_master_mask_exists,
            "raw_master_mask_path": str(raw_master_mask.get("path") or ""),
            "direct_qty_ocr_boxes_count": direct_qty_ocr_boxes_count,
            "split_candidates_exist": bool(split_counts.get("split_candidates_exist")),
            "split_candidate_count": int(split_counts.get("split_candidate_count", 0) or 0),
            "accepted_count": accepted_count,
            "confirmed_count": confirmed_count,
            "unconfirmed_accepted_count": unconfirmed_accepted_count,
            "review_state_counts": split_counts.get("review_state_counts") or {},
            "needs_review_count": needs_review_count,
            "generated_missing_candidates": generated_missing_candidates,
            "generate_error": generate_error,
            "ready": ready,
            "failed": failed,
            "reasons": reasons,
        })

    summary["total_bundles"] = len(rows)
    return {
        "ok": not discovery_errors,
        "debug_only": True,
        "set_num": set_text,
        "bag_num": bag_number,
        "generate_missing": should_generate_missing,
        "summary": summary,
        "rows": rows,
        "discovery_errors": discovery_errors,
    }


@router.post("/debug/training-store/auto-batch-report")
def training_store_auto_batch_report(
    set_num: str = Query(...),
    bag_num: int = Query(...),
    generate_missing: int = Query(0),
):
    return JSONResponse(
        _auto_batch_report_payload(
            set_num=set_num,
            bag_num=bag_num,
            generate_missing=generate_missing,
        )
    )


def _quick_review_skip_info(bundle_id: str) -> Dict[str, Any]:
    try:
        artifact_debug = _training_bundle_artifact_debug(bundle_id)
        metadata = _auto_batch_manifest_for_bundle(bundle_id, artifact_debug)
    except Exception:
        return {}
    info = metadata.get("quick_review_skip") if isinstance(metadata.get("quick_review_skip"), dict) else {}
    return dict(info)


def _quick_review_problem_reasons(row: Dict[str, Any]) -> List[str]:
    if _quick_review_skip_info(str(row.get("bundle_id") or "")):
        return []
    reasons: List[str] = []
    review_state_counts = row.get("review_state_counts") if isinstance(row.get("review_state_counts"), dict) else {}
    for key in ("needs_mask_expand", "needs_ocr_review", "needs_manual_crop", "rejected"):
        count = int(review_state_counts.get(key, 0) or 0)
        if count > 0:
            reasons.append(f"{key}: {count}")
    if bool(row.get("failed")):
        reasons.append("failed report row")
    if int(row.get("direct_qty_ocr_boxes_count", 0) or 0) <= 0:
        reasons.append("missing OCR")
    if not bool(row.get("raw_master_mask_exists")):
        reasons.append("missing mask")
    if not bool(row.get("split_candidates_exist")):
        reasons.append("missing split candidates")
    unconfirmed = int(row.get("unconfirmed_accepted_count", 0) or 0)
    if unconfirmed > 0:
        reasons.append(f"unconfirmed accepted candidates: {unconfirmed}")
    return reasons


def _quick_review_next_bag_url(set_num: str, bag_num: int) -> Tuple[Optional[int], str]:
    set_text = str(set_num or "").strip()
    current_bag = int(bag_num or 1)
    bag_values: set[int] = set()
    try:
        queue = list_review_queue(set_num=set_text, limit=500)
        for row in list(queue.get("rows") or []):
            if isinstance(row, dict):
                parsed = _coerce_int(row.get("bag_num"))
                if parsed is not None and int(parsed) > current_bag:
                    bag_values.add(int(parsed))
    except Exception:
        pass
    analysis_root = _training_store_analysis_root()
    if analysis_root.exists():
        for path in analysis_root.glob(f"{set_text}_bag*_p*_s*_c*"):
            match = re.match(rf"^{re.escape(set_text)}_bag(?P<bag>\d+)_", path.name)
            if match:
                parsed = _coerce_int(match.group("bag"))
                if parsed is not None and int(parsed) > current_bag:
                    bag_values.add(int(parsed))
    if not bag_values:
        return None, ""
    next_bag = min(bag_values)
    return next_bag, f"/debug/training-store/quick-review?set_num={_url_quote(set_text)}&bag_num={int(next_bag)}"


def _quick_review_reports_root() -> Path:
    return Path("/Users/olly/aim2build-instruction") / "debug" / "ai_training" / "reports"


def _quick_review_cache_path(set_num: str, bag_num: Any) -> Path:
    safe_set_num = re.sub(r"[^A-Za-z0-9.-]+", "", str(set_num or "").strip())
    parsed_bag = _coerce_int(bag_num) or 1
    return _quick_review_reports_root() / f"{safe_set_num}_bag{int(parsed_bag)}_quick_review.json"


def _read_quick_review_cache(set_num: str, bag_num: Any) -> Dict[str, Any]:
    cache_path = _quick_review_cache_path(set_num, bag_num)
    if not cache_path.exists() or not cache_path.is_file():
        return {}
    try:
        loaded = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_quick_review_cache(payload: Dict[str, Any]) -> Path:
    cache_path = _quick_review_cache_path(str(payload.get("set_num") or ""), payload.get("bag_num"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return cache_path


def _quick_review_bundle_ids(bundle_id: str) -> Tuple[str, Optional[int]]:
    try:
        row = dict(get_training_bundle_index_row(bundle_id).get("row") or {})
        set_num = str(row.get("set_num") or "").strip()
        bag_num = _coerce_int(row.get("bag_num"))
        if set_num and bag_num is not None:
            return set_num, int(bag_num)
    except Exception:
        pass
    parsed = re.match(r"^(?P<set>[^_]+)_bag(?P<bag>\d+)_", str(bundle_id or ""))
    if not parsed:
        return "", None
    return str(parsed.group("set") or ""), _coerce_int(parsed.group("bag"))


def _mark_quick_review_cache_stale_for_bundle(bundle_id: str, reason: str) -> None:
    set_num, bag_num = _quick_review_bundle_ids(bundle_id)
    if not set_num or bag_num is None:
        return
    cache = _read_quick_review_cache(set_num, bag_num)
    if not cache:
        return
    cache["stale"] = True
    cache["stale_reason"] = str(reason or "bundle_changed")
    cache["stale_at"] = _iso_now()
    try:
        _write_quick_review_cache(cache)
    except Exception:
        pass


def _remove_quick_review_problem_from_cache(bundle_id: str) -> None:
    set_num, bag_num = _quick_review_bundle_ids(bundle_id)
    if not set_num or bag_num is None:
        return
    cache = _read_quick_review_cache(set_num, bag_num)
    if not cache:
        return
    problems = [
        dict(row)
        for row in list(cache.get("problem_rows") or [])
        if isinstance(row, dict) and str(row.get("bundle_id") or "") != str(bundle_id or "")
    ]
    cache["problem_rows"] = problems
    summary = cache.get("summary") if isinstance(cache.get("summary"), dict) else {}
    summary["problem_bundle_count"] = len(problems)
    cache["summary"] = summary
    cache["stale"] = True
    cache["stale_reason"] = "problem_skipped"
    cache["stale_at"] = _iso_now()
    try:
        _write_quick_review_cache(cache)
    except Exception:
        pass


def _build_quick_review_cache_payload(set_num: str, bag_num: int) -> Dict[str, Any]:
    set_text = str(set_num or "").strip()
    bag_number = int(bag_num or 1)
    report = _auto_batch_report_payload(set_num=set_text, bag_num=bag_number, generate_missing=0)
    rows = [dict(row) for row in list(report.get("rows") or []) if isinstance(row, dict)]
    problem_rows: List[Dict[str, Any]] = []
    for row in rows:
        problems = _quick_review_problem_reasons(row)
        if not problems:
            continue
        bundle_id = str(row.get("bundle_id") or "")
        row["quick_review_problems"] = problems
        row["review_url"] = _training_review_url(bundle_id, set_text, bag_number, 50)
        row["generated_candidate_status"] = {
            "generated_missing_candidates": bool(row.get("generated_missing_candidates")),
            "generate_error": str(row.get("generate_error") or ""),
            "split_candidates_exist": bool(row.get("split_candidates_exist")),
            "split_candidate_count": int(row.get("split_candidate_count", 0) or 0),
        }
        problem_rows.append(row)
    summary = dict(report.get("summary") or {})
    summary["problem_bundle_count"] = len(problem_rows)
    summary["skipped_problem_count"] = len(rows) - len(problem_rows) - int(summary.get("ready_count", 0) or 0)
    next_bag_num, next_bag_url = _quick_review_next_bag_url(set_text, bag_number)
    return {
        "ok": True,
        "debug_only": True,
        "cache_version": 1,
        "set_num": set_text,
        "bag_num": bag_number,
        "built_at": _iso_now(),
        "stale": False,
        "stale_reason": "",
        "stale_at": "",
        "summary": summary,
        "problem_rows": problem_rows,
        "next_bag_num": next_bag_num,
        "next_bag_url": next_bag_url,
        "report_summary": report.get("summary") or {},
        "discovery_errors": report.get("discovery_errors") or [],
    }


@router.post("/debug/training-store/build-quick-review-cache")
def training_store_build_quick_review_cache(
    set_num: str = Query(...),
    bag_num: int = Query(...),
):
    payload = _build_quick_review_cache_payload(set_num, int(bag_num or 1))
    cache_path = _write_quick_review_cache(payload)
    payload["cache_path"] = str(cache_path)
    return JSONResponse(payload)


@router.post("/debug/training-store/quick-review-skip")
def training_store_quick_review_skip(
    bundle_id: str = Query(...),
    reviewed_by: str = Query("andy"),
):
    safe_bundle_id = str(bundle_id or "").strip()
    if not safe_bundle_id:
        raise HTTPException(status_code=400, detail="bundle_id is required")
    try:
        artifact_debug = _training_bundle_artifact_debug(safe_bundle_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    metadata_path = Path(str(artifact_debug.get("metadata_path") or "").strip())
    if not metadata_path.exists() or not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="metadata.json not found")
    metadata = _auto_batch_manifest_for_bundle(safe_bundle_id, artifact_debug)
    metadata["quick_review_skip"] = {
        "skipped": True,
        "skipped_by": str(reviewed_by or "andy"),
        "skipped_at": _iso_now(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    _remove_quick_review_problem_from_cache(safe_bundle_id)
    return JSONResponse({"ok": True, "bundle_id": safe_bundle_id, "quick_review_skip": metadata["quick_review_skip"]})


@router.get("/debug/training-store/quick-review")
def training_store_quick_review(
    set_num: str = Query(...),
    bag_num: int = Query(...),
):
    quick_review_started = time.perf_counter()
    set_text = str(set_num or "").strip()
    if not set_text:
        raise HTTPException(status_code=400, detail="set_num is required")
    bag_number = int(bag_num or 1)
    cache_path = _quick_review_cache_path(set_text, bag_number)
    cache = _read_quick_review_cache(set_text, bag_number)
    quick_review_cache_load_ms = int((time.perf_counter() - quick_review_started) * 1000)
    cache_exists = bool(cache)
    problem_rows = [
        dict(row)
        for row in list(cache.get("problem_rows") or [])
        if isinstance(row, dict)
    ]
    summary = dict(cache.get("summary") or {})
    next_bag_num = cache.get("next_bag_num")
    next_bag_url = str(cache.get("next_bag_url") or "")
    cache_timestamp = str(cache.get("built_at") or "")
    cache_stale = bool(cache.get("stale"))
    cache_status_html = (
        f'<div class="cache-status {"stale" if cache_stale else ""}">'
        f'<span>Cache: {escape(cache_timestamp or "not built")}</span>'
        f'<span>{escape("stale - " + str(cache.get("stale_reason") or "needs refresh") if cache_stale else "ready" if cache_exists else "missing")}</span>'
        f'</div>'
    )

    problem_cards = []
    for row in problem_rows:
        bundle_id = str(row.get("bundle_id") or "")
        review_url = str(row.get("review_url") or _training_review_url(bundle_id, set_text, bag_number, 50))
        problem_text = ", ".join(str(item) for item in list(row.get("quick_review_problems") or []))
        generated_status = row.get("generated_candidate_status") if isinstance(row.get("generated_candidate_status"), dict) else {}
        generate_button = (
            f'<button type="button" data-generate-missing="true" data-bundle-id="{escape(bundle_id)}">Generate missing candidates</button>'
            if not bool(row.get("split_candidates_exist")) and bool(row.get("original_crop_exists")) and bool(row.get("raw_master_mask_exists")) and bool(row.get("postgres_row_exists"))
            else ""
        )
        problem_cards.append(
            '<article class="problem-card">'
            f'<div><h3>{escape(bundle_id)}</h3><p>{escape(problem_text)}</p></div>'
            '<dl>'
            f'<div><dt>OCR boxes</dt><dd>{escape(str(row.get("direct_qty_ocr_boxes_count") or 0))}</dd></div>'
            f'<div><dt>candidates</dt><dd>{escape(str(row.get("split_candidate_count") or 0))}</dd></div>'
            f'<div><dt>accepted</dt><dd>{escape(str(row.get("accepted_count") or 0))}</dd></div>'
            f'<div><dt>confirmed</dt><dd>{escape(str(row.get("confirmed_count") or 0))}</dd></div>'
            '</dl>'
            f'<div class="generated-status">generated: {escape("yes" if generated_status.get("generated_missing_candidates") else "no")} · error: {escape(str(generated_status.get("generate_error") or "none"))}</div>'
            '<div class="problem-actions">'
            f'<a href="{escape(review_url)}">Open Review</a>'
            f'{generate_button}'
            f'<button type="button" data-skip-problem="true" data-bundle-id="{escape(bundle_id)}">Mark as OK / Skip problem</button>'
            '</div>'
            '</article>'
        )
    problem_html = (
        "".join(problem_cards)
        if cache_exists
        else '<div class="empty-state">No cache yet. Build the Quick Review cache to load problem bundles.</div>'
    ) or '<div class="empty-state">No unresolved problem bundles in this bag.</div>'
    next_bag_html = (
        f'<a class="next-bag" href="{escape(next_bag_url)}">Open Next Bag {escape(str(next_bag_num))}</a>'
        if cache_exists and not problem_rows and next_bag_url
        else ""
    )
    summary_items = [
        ("total bundles", summary.get("total_bundles", 0)),
        ("ready", summary.get("ready_count", 0)),
        ("problems", summary.get("problem_bundle_count", 0)),
        ("missing OCR", summary.get("missing_ocr_count", 0)),
        ("missing mask", summary.get("missing_mask_count", 0)),
        ("missing candidates", summary.get("missing_candidates_count", 0)),
        ("needs review", summary.get("needs_review_count", 0)),
        ("failed", summary.get("failed_count", 0)),
    ]
    summary_html = "".join(
        f'<div><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>'
        for label, value in summary_items
    )
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Quick Review {escape(set_text)} bag {escape(str(bag_number))}</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#172026; background:#f5f7f8; }}
    header {{ display:flex; justify-content:space-between; align-items:center; gap:16px; padding:14px 18px; background:#18212b; color:white; }}
    main {{ padding:18px; max-width:1180px; margin:0 auto; }}
    .toolbar {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin:0 0 14px; }}
    .cache-status {{ display:flex; gap:10px; flex-wrap:wrap; color:#66727d; font-size:13px; }}
    .cache-status.stale {{ color:#8a5a00; font-weight:700; }}
    .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:0 0 18px; }}
    .summary div {{ background:white; border:1px solid #d8dee3; border-radius:6px; padding:10px; }}
    .summary span {{ display:block; color:#66727d; font-size:12px; }}
    .summary strong {{ font-size:24px; }}
    .problem-list {{ display:grid; gap:12px; }}
    .problem-card {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:14px; align-items:start; background:white; border:1px solid #d8dee3; border-radius:8px; padding:14px; }}
    .problem-card h3 {{ margin:0 0 6px; font-size:18px; overflow-wrap:anywhere; }}
    .problem-card p {{ margin:0; color:#8a5a00; font-weight:700; }}
    .generated-status {{ grid-column:1 / -1; color:#66727d; font-size:12px; }}
    dl {{ display:grid; grid-template-columns:repeat(4,86px); gap:8px; margin:0; }}
    dl div {{ border:1px solid #e4e9ed; border-radius:6px; padding:7px; background:#f8fafb; }}
    dt {{ color:#66727d; font-size:11px; }}
    dd {{ margin:0; font-weight:800; }}
    .problem-actions {{ grid-column:1 / -1; display:flex; gap:8px; flex-wrap:wrap; }}
    a, button {{ display:inline-flex; align-items:center; justify-content:center; min-height:36px; box-sizing:border-box; padding:8px 12px; border:1px solid #1f6fc9; border-radius:6px; background:#1f6fc9; color:white; font:inherit; font-weight:700; text-decoration:none; cursor:pointer; }}
    button[data-skip-problem] {{ background:#f8fafb; color:#32465a; border-color:#c8d0d7; }}
    button[data-generate-missing] {{ background:#755000; border-color:#755000; }}
    .next-bag {{ background:#2f6c41; border-color:#2f6c41; margin-bottom:14px; }}
    .empty-state {{ padding:22px; background:#eef6ef; border:1px solid #7db28a; border-radius:8px; color:#2f6c41; font-weight:800; }}
    @media (max-width:760px) {{
      .problem-card {{ grid-template-columns:1fr; }}
      dl {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    }}
  </style>
</head>
<body>
  <header><strong>Quick Review</strong><span>{escape(set_text)} bag {escape(str(bag_number))}</span></header>
  <main>
    <div class="toolbar">
      {cache_status_html}
      <button type="button" data-refresh-cache="true">{escape("Refresh / Rebuild Cache" if cache_exists else "Build Cache")}</button>
    </div>
    <section class="summary">{summary_html}</section>
    {next_bag_html}
    <section class="problem-list">{problem_html}</section>
  </main>
  <script>
    const buildCacheUrl = '/debug/training-store/build-quick-review-cache?set_num={_url_quote(set_text)}&bag_num={int(bag_number)}';
    async function rebuildQuickReviewCache(button) {{
      if (button) {{
        button.disabled = true;
        button.textContent = 'Rebuilding...';
      }}
      const res = await fetch(buildCacheUrl, {{method: 'POST'}});
      if (!res.ok) {{
        alert(await res.text());
        if (button) {{
          button.disabled = false;
          button.textContent = '{escape("Refresh / Rebuild Cache" if cache_exists else "Build Cache")}';
        }}
        return false;
      }}
      return true;
    }}
    document.querySelectorAll('[data-refresh-cache]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        const ok = await rebuildQuickReviewCache(button);
        if (ok) {{
          location.reload();
        }}
      }});
    }});
    document.querySelectorAll('[data-generate-missing]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        button.disabled = true;
        button.textContent = 'Generating...';
        const bundleId = button.dataset.bundleId || '';
        const res = await fetch('/debug/training-store/generate-split-candidates?bundle_id=' + encodeURIComponent(bundleId), {{method: 'POST'}});
        if (!res.ok) {{
          alert(await res.text());
          button.disabled = false;
          button.textContent = 'Generate missing candidates';
          return;
        }}
        await rebuildQuickReviewCache(null);
        location.reload();
      }});
    }});
    document.querySelectorAll('[data-skip-problem]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        const bundleId = button.dataset.bundleId || '';
        const res = await fetch('/debug/training-store/quick-review-skip?bundle_id=' + encodeURIComponent(bundleId), {{method: 'POST'}});
        if (!res.ok) {{
          alert(await res.text());
          return;
        }}
        location.reload();
      }});
    }});
  </script>
</body>
</html>
"""
    quick_review_render_ms = int((time.perf_counter() - quick_review_started) * 1000)
    print(
        "[quick-review-cache] "
        f"set_num={set_text} bag_num={bag_number} cache_path={cache_path} "
        f"cache_exists={cache_exists} stale={cache_stale} "
        f"quick_review_cache_load_ms={quick_review_cache_load_ms} "
        f"quick_review_render_ms={quick_review_render_ms} "
        f"quick_review_problem_count={len(problem_rows)}"
    )
    return HTMLResponse(html)


def _parse_slot_indexes(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_text = str(value or "").strip()
        if not raw_text:
            return []
        raw_values = raw_text.split(",")
    slots: List[int] = []
    for raw in raw_values:
        try:
            slot_index = int(raw)
        except Exception:
            continue
        if slot_index >= 0 and slot_index not in slots:
            slots.append(slot_index)
    slots.sort()
    return slots


async def _training_store_review_payload(
    req: Request,
    bundle_id: str,
    slots: str,
    notes: str,
    reviewer: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    content_type = str(req.headers.get("content-type") or "")
    if "application/json" in content_type:
        try:
            body = await req.json()
            if isinstance(body, dict):
                payload = dict(body)
        except Exception:
            payload = {}
    resolved_bundle_id = str(payload.get("bundle_id") or bundle_id or "").strip()
    resolved_notes = str(payload.get("notes") if payload.get("notes") is not None else notes or "")
    resolved_reviewer = str(payload.get("reviewer") if payload.get("reviewer") is not None else reviewer or "")
    raw_slots = payload.get("slot_indexes")
    if raw_slots is None:
        raw_slots = payload.get("slots")
    if raw_slots is None:
        raw_slots = slots
    return {
        "bundle_id": resolved_bundle_id,
        "slot_indexes": _parse_slot_indexes(raw_slots),
        "notes": resolved_notes,
        "reviewer": resolved_reviewer,
    }


@router.post("/debug/training-store/approve-bundle")
async def training_store_approve_bundle(
    req: Request,
    bundle_id: str = Query(""),
    slots: str = Query(""),
    notes: str = Query(""),
    reviewer: str = Query(""),
):
    payload = await _training_store_review_payload(req, bundle_id, slots, notes, reviewer)
    try:
        result = update_bundle_review(
            payload["bundle_id"],
            "approved",
            slot_indexes=payload["slot_indexes"],
            notes=payload["notes"],
            reviewer=payload["reviewer"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _mark_quick_review_cache_stale_for_bundle(bundle_id, "split_candidates_generated")
    return JSONResponse(result)


@router.post("/debug/training-store/reject-bundle")
async def training_store_reject_bundle(
    req: Request,
    bundle_id: str = Query(""),
    slots: str = Query(""),
    notes: str = Query(""),
    reviewer: str = Query(""),
):
    payload = await _training_store_review_payload(req, bundle_id, slots, notes, reviewer)
    try:
        result = update_bundle_review(
            payload["bundle_id"],
            "rejected",
            slot_indexes=payload["slot_indexes"],
            notes=payload["notes"],
            reviewer=payload["reviewer"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _mark_quick_review_cache_stale_for_bundle(bundle_id, "split_candidate_accepted")
    return JSONResponse(result)


@router.get("/debug/training-store/prepare-r2-upload")
def training_store_prepare_r2_upload(bundle_id: str = Query(...)):
    try:
        result = prepare_bundle_for_r2(bundle_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _mark_quick_review_cache_stale_for_bundle(bundle_id, "split_candidate_rejected")
    return JSONResponse(result)


def _dry_run_enabled(value: str) -> bool:
    text = str(value or "1").strip().lower()
    return text not in {"0", "false", "no", "off"}


def _resolve_debug_training_path(path_value: Any) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        return ""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path("/Users/olly/aim2build-instruction") / path
    return str(path.resolve(strict=False))


def _planned_ai_crop_fix_path(path_value: Any, suffix: str) -> str:
    resolved = _resolve_debug_training_path(path_value)
    if not resolved:
        return ""
    path = Path(resolved)
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix or '.png'}"))


def _request_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text not in {"0", "false", "no", "off"}


def _ai_crop_fix_bundle_dir(bundle_id: str) -> Path:
    safe_bundle_id = str(bundle_id or "").strip()
    if not safe_bundle_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", safe_bundle_id):
        raise ValueError("invalid bundle_id")
    analysis_root = (
        Path("/Users/olly/aim2build-instruction")
        / "debug"
        / "ai_training"
        / "analysis_bundles"
    ).resolve()
    bundle_dir = (analysis_root / safe_bundle_id).resolve()
    if analysis_root not in bundle_dir.parents:
        raise ValueError("bundle path escapes analysis bundle directory")
    return bundle_dir


def _ai_crop_fix_candidate_from_index(bundle_id: str, candidate_id: str) -> Dict[str, Any]:
    try:
        row = dict(get_training_bundle_index_row(bundle_id).get("row") or {})
    except Exception:
        row = {}
    paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
    candidates = [
        dict(item)
        for item in list(paths.get("candidates") or [])
        if isinstance(item, dict)
    ]
    if not candidates:
        return {}
    candidate_text = str(candidate_id or "").strip()
    candidate_index = _coerce_int(candidate_text)
    for candidate in candidates:
        values = {
            str(candidate.get("candidate_id") or "").strip(),
            str(candidate.get("id") or "").strip(),
            str(candidate.get("name") or "").strip(),
        }
        if candidate_text and candidate_text in values:
            return candidate
        if candidate_index is not None and _coerce_int(candidate.get("index")) == candidate_index:
            return candidate
    if candidate_index is not None and 0 <= candidate_index < len(candidates):
        return candidates[candidate_index]
    return {}


def _first_existing_path(candidates: List[Path]) -> str:
    for path in candidates:
        if path.exists() and path.is_file():
            return str(path)
    return str(candidates[0]) if candidates else ""


def _ai_crop_fix_safe_component(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return safe or fallback


def _resolve_ai_crop_fix_preview_paths(bundle_id: str, candidate_id: str) -> Dict[str, str]:
    bundle_dir = _ai_crop_fix_bundle_dir(bundle_id)
    candidate = _ai_crop_fix_candidate_from_index(bundle_id, candidate_id)
    candidate_text = str(candidate_id or "").strip()
    candidate_index = _coerce_int(candidate_text)
    stem_candidates: List[str] = []
    if candidate_text:
        stem_candidates.append(Path(candidate_text).stem)
    if candidate_index is not None:
        stem_candidates.extend(
            [
                f"baseline_slot_candidate_{int(candidate_index)}",
                f"split_candidate_{int(candidate_index)}",
                f"candidate_{int(candidate_index)}",
            ]
        )
    stem_candidates = [stem for index, stem in enumerate(stem_candidates) if stem and stem not in stem_candidates[:index]]

    original_candidate_path = _first_existing_path([bundle_dir / "original_crop.png"])
    original_alpha_path = str(
        candidate.get("qty_scrubbed_mask_path")
        or candidate.get("mask_path")
        or candidate.get("alpha_path")
        or ""
    ).strip()

    if not Path(original_candidate_path).exists():
        original_candidate_path = _first_existing_path(
            [bundle_dir / f"{stem}.png" for stem in stem_candidates]
        )
    if not original_alpha_path:
        original_alpha_path = _first_existing_path(
            [
                candidate_path
                for stem in stem_candidates
                for candidate_path in (
                    bundle_dir / f"{stem}_mask.png",
                    bundle_dir / f"{stem}_raw_mask.png",
                    bundle_dir / f"{stem}_qty_scrubbed_mask.png",
                )
            ]
        )

    repaired_candidate_path = _planned_ai_crop_fix_path(
        original_candidate_path,
        "_ai_crop_fix_repaired_candidate",
    )
    repaired_alpha_path = _planned_ai_crop_fix_path(
        original_alpha_path,
        "_ai_crop_fix_repaired_alpha",
    )
    return {
        "original_candidate_path": _resolve_debug_training_path(original_candidate_path),
        "original_alpha_path": _resolve_debug_training_path(original_alpha_path),
        "repaired_candidate_path": repaired_candidate_path,
        "repaired_alpha_path": repaired_alpha_path,
    }


def _write_ai_crop_fix_request_package(
    *,
    bundle_id: str,
    candidate_id: str,
    current_candidate_path: str,
    current_alpha_path: str,
    reference_candidate_path: str,
    slot_overlay_path: str,
    qty: Any,
    metadata: Dict[str, Any],
    resolved_absolute_paths: Dict[str, str],
) -> Dict[str, Any]:
    safe_bundle_id = _ai_crop_fix_safe_component(bundle_id, "bundle")
    safe_candidate_id = _ai_crop_fix_safe_component(candidate_id, "candidate")
    package_dir = (
        Path("/Users/olly/aim2build-instruction")
        / "debug"
        / "ai_training"
        / "ai_crop_fix_requests"
        / f"{safe_bundle_id}_{safe_candidate_id}"
    )
    package_dir.mkdir(parents=True, exist_ok=True)

    copied_files: List[str] = []
    copy_plan = [
        ("candidate.png", resolved_absolute_paths.get("current_candidate_path", "")),
        ("alpha.png", resolved_absolute_paths.get("current_alpha_path", "")),
    ]
    reference_source = resolved_absolute_paths.get("reference_candidate_path", "")
    if reference_source and Path(reference_source).exists() and Path(reference_source).is_file():
        copy_plan.append(("reference_candidate.png", reference_source))
    overlay_source = resolved_absolute_paths.get("slot_overlay_path", "")
    if overlay_source and Path(overlay_source).exists() and Path(overlay_source).is_file():
        copy_plan.append(("overlay.png", overlay_source))
    for filename, source in copy_plan:
        if not source:
            continue
        target = package_dir / filename
        shutil.copy2(source, target)
        copied_files.append(str(target))

    request_payload = {
        "bundle_id": bundle_id,
        "candidate_id": candidate_id,
        "qty": qty,
        "metadata": metadata,
        "received_paths": {
            "current_candidate_path": current_candidate_path,
            "current_alpha_path": current_alpha_path,
            "reference_candidate_path": reference_candidate_path,
            "slot_overlay_path": slot_overlay_path,
        },
        "resolved_absolute_paths": resolved_absolute_paths,
        "package_path": str(package_dir),
        "created_at": _iso_now(),
        "mode": "ai_scaffold",
        "openai_called": False,
    }
    request_path = package_dir / "request.json"
    request_path.write_text(json.dumps(request_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    copied_files.append(str(request_path))
    return {
        "package_path": str(package_dir),
        "files": copied_files,
    }


def _alpha_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if mask is None or getattr(mask, "size", 0) == 0:
        return None
    ys, xs = np.where(mask > 10)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _write_ai_crop_fix_matched_placeholder(
    *,
    original_crop_path: str,
    alpha_path: str,
    reference_candidate_path: str,
    repaired_candidate_path: str,
    repaired_alpha_path: str,
) -> Dict[str, Any]:
    original = cv2.imread(original_crop_path, cv2.IMREAD_COLOR)
    alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(reference_candidate_path, cv2.IMREAD_UNCHANGED) if reference_candidate_path else None
    if original is None or getattr(original, "size", 0) == 0:
        raise ValueError("original crop could not be read")
    if alpha is None or getattr(alpha, "size", 0) == 0:
        raise ValueError("alpha mask could not be read")
    if reference is None or getattr(reference, "size", 0) == 0:
        raise ValueError("reference candidate could not be read")

    if len(reference.shape) == 3 and reference.shape[2] == 4:
        reference_bgr = reference[:, :, :3]
        reference_alpha = reference[:, :, 3]
    elif len(reference.shape) == 3:
        reference_bgr = reference[:, :, :3]
        reference_alpha = alpha
    else:
        reference_bgr = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
        reference_alpha = alpha

    ref_h, ref_w = reference_bgr.shape[:2]
    if original.shape[0] < ref_h or original.shape[1] < ref_w:
        raise ValueError("reference candidate is larger than original crop")
    mask = (reference_alpha > 10).astype(np.uint8) * 255
    if mask.shape[:2] != (ref_h, ref_w):
        mask = cv2.resize(mask, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)

    result = cv2.matchTemplate(original, reference_bgr, cv2.TM_CCORR_NORMED, mask=mask)
    result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    match_x, match_y = int(max_loc[0]), int(max_loc[1])
    matched_bgr = original[match_y : match_y + ref_h, match_x : match_x + ref_w]
    if matched_bgr.shape[:2] != (ref_h, ref_w):
        raise ValueError("matched crop could not be extracted")

    out_alpha = alpha
    if out_alpha.shape[:2] != (ref_h, ref_w):
        out_alpha = cv2.resize(out_alpha, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
    repaired_rgba = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2BGRA)
    repaired_rgba[:, :, 3] = out_alpha

    Path(repaired_candidate_path).parent.mkdir(parents=True, exist_ok=True)
    Path(repaired_alpha_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(repaired_candidate_path, repaired_rgba)
    cv2.imwrite(repaired_alpha_path, out_alpha)
    bbox = _alpha_bbox(out_alpha)
    return {
        "match_score": float(max_val),
        "match_box_xywh": [match_x, match_y, int(ref_w), int(ref_h)],
        "alpha_bbox_xywh": list(bbox) if bbox is not None else [],
        "reference_candidate_path": reference_candidate_path,
    }


def _ai_crop_fix_preview_tile(title: str, path_value: str) -> str:
    path_text = str(path_value or "").strip()
    exists = bool(path_text and Path(path_text).exists() and Path(path_text).is_file())
    image_html = (
        f'<img src="/debug/ai-snap-artifact?path={_url_quote(path_text)}" alt="{escape(title)}" loading="lazy">'
        if exists
        else '<div class="missing-image">missing</div>'
    )
    return (
        '<figure class="preview-tile">'
        f'<div class="preview-image">{image_html}</div>'
        f'<figcaption><strong>{escape(title)}</strong>'
        f'<span class="status {"ok" if exists else "missing"}">exists: {escape(str(exists).lower())}</span>'
        f'<code>{escape(path_text or "not resolved")}</code></figcaption>'
        '</figure>'
    )


@router.post("/debug/training-store/ai-crop-fix")
async def training_store_ai_crop_fix(req: Request):
    payload: Dict[str, Any] = {}
    try:
        body = await req.json()
        if isinstance(body, dict):
            payload = dict(body)
    except Exception:
        payload = {}
    query = dict(req.query_params)

    candidate_metadata = payload.get("candidate_metadata")
    if candidate_metadata is None:
        candidate_metadata = payload.get("candidate")
    if not isinstance(candidate_metadata, dict):
        candidate_metadata = {}

    bundle_id = str(payload.get("bundle_id") or query.get("bundle_id") or "").strip()
    candidate_id = str(
        payload.get("candidate_id")
        or query.get("candidate_id")
        or candidate_metadata.get("candidate_id")
        or candidate_metadata.get("id")
        or candidate_metadata.get("index")
        or ""
    ).strip()
    current_candidate_path = str(
        payload.get("current_candidate_path")
        or query.get("current_candidate_path")
        or candidate_metadata.get("current_candidate_path")
        or candidate_metadata.get("candidate_path")
        or ""
    ).strip()
    current_alpha_path = str(
        payload.get("current_alpha_path")
        or query.get("current_alpha_path")
        or candidate_metadata.get("current_alpha_path")
        or candidate_metadata.get("alpha_path")
        or candidate_metadata.get("mask_path")
        or ""
    ).strip()
    reference_candidate_path = str(
        payload.get("reference_candidate_path")
        or query.get("reference_candidate_path")
        or candidate_metadata.get("reference_candidate_path")
        or candidate_metadata.get("template_candidate_path")
        or candidate_metadata.get("qty_scrubbed_path")
        or candidate_metadata.get("thumbnail_path")
        or candidate_metadata.get("candidate_path")
        or ""
    ).strip()
    slot_overlay_path = str(
        payload.get("slot_overlay_path")
        or query.get("slot_overlay_path")
        or candidate_metadata.get("slot_overlay_path")
        or candidate_metadata.get("overlay_path")
        or ""
    ).strip()
    qty = payload.get("qty", query.get("qty", candidate_metadata.get("qty")))
    dry_run = _request_bool(payload.get("dry_run", query.get("dry_run")), True)
    mode = str(payload.get("mode") or query.get("mode") or "").strip().lower()

    if not bundle_id:
        raise HTTPException(status_code=400, detail="bundle_id is required")

    received_paths = {
        "current_candidate_path": current_candidate_path,
        "current_alpha_path": current_alpha_path,
        "reference_candidate_path": reference_candidate_path,
        "slot_overlay_path": slot_overlay_path,
    }
    resolved_absolute_paths = {
        key: _resolve_debug_training_path(value)
        for key, value in received_paths.items()
    }
    file_exists = {
        key: bool(value and Path(value).exists() and Path(value).is_file())
        for key, value in resolved_absolute_paths.items()
    }
    candidate_image_exists = bool(file_exists.get("current_candidate_path"))
    alpha_mask_exists = bool(file_exists.get("current_alpha_path"))
    planned_repaired_alpha_path = _planned_ai_crop_fix_path(
        current_alpha_path,
        "_ai_crop_fix_repaired_alpha",
    )
    planned_repaired_candidate_path = _planned_ai_crop_fix_path(
        current_candidate_path,
        "_ai_crop_fix_repaired_candidate",
    )

    repaired_paths: Dict[str, str] = {}
    package_result: Dict[str, Any] = {}
    match_result: Dict[str, Any] = {}
    copied = False
    if not dry_run:
        if not candidate_image_exists:
            raise HTTPException(status_code=400, detail="current_candidate_path does not exist")
        if not alpha_mask_exists:
            raise HTTPException(status_code=400, detail="current_alpha_path does not exist")
        if mode == "placeholder":
            if not planned_repaired_alpha_path or not planned_repaired_candidate_path:
                raise HTTPException(status_code=400, detail="planned repaired paths could not be resolved")
            if Path(planned_repaired_alpha_path) == Path(resolved_absolute_paths["current_alpha_path"]):
                raise HTTPException(status_code=400, detail="planned repaired alpha path matches source")
            if Path(planned_repaired_candidate_path) == Path(resolved_absolute_paths["current_candidate_path"]):
                raise HTTPException(status_code=400, detail="planned repaired candidate path matches source")
            if reference_candidate_path:
                try:
                    match_result = _write_ai_crop_fix_matched_placeholder(
                        original_crop_path=resolved_absolute_paths["current_candidate_path"],
                        alpha_path=resolved_absolute_paths["current_alpha_path"],
                        reference_candidate_path=resolved_absolute_paths["reference_candidate_path"],
                        repaired_candidate_path=planned_repaired_candidate_path,
                        repaired_alpha_path=planned_repaired_alpha_path,
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
            else:
                Path(planned_repaired_alpha_path).parent.mkdir(parents=True, exist_ok=True)
                Path(planned_repaired_candidate_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(resolved_absolute_paths["current_alpha_path"], planned_repaired_alpha_path)
                shutil.copy2(resolved_absolute_paths["current_candidate_path"], planned_repaired_candidate_path)
            repaired_paths = {
                "repaired_alpha_path": planned_repaired_alpha_path,
                "repaired_candidate_path": planned_repaired_candidate_path,
            }
            copied = True
        elif mode == "ai_scaffold":
            package_result = _write_ai_crop_fix_request_package(
                bundle_id=bundle_id,
                candidate_id=candidate_id,
                current_candidate_path=current_candidate_path,
                current_alpha_path=current_alpha_path,
                reference_candidate_path=reference_candidate_path,
                slot_overlay_path=slot_overlay_path,
                qty=qty,
                metadata=candidate_metadata,
                resolved_absolute_paths=resolved_absolute_paths,
            )
            copied = True
        else:
            raise HTTPException(status_code=400, detail="mode=placeholder or mode=ai_scaffold is required when dry_run=false")

    return JSONResponse(
        {
            "ok": True,
            "dry_run": dry_run,
            "mode": mode,
            "bundle_id": bundle_id,
            "candidate_id": candidate_id,
            "candidate_metadata": candidate_metadata,
            "qty": qty,
            "received_paths": received_paths,
            "resolved_absolute_paths": resolved_absolute_paths,
            "file_exists": file_exists,
            "candidate_image_exists": candidate_image_exists,
            "alpha_mask_exists": alpha_mask_exists,
            "planned_repaired_alpha_path": planned_repaired_alpha_path,
            "planned_repaired_candidate_path": planned_repaired_candidate_path,
            "repaired_paths": repaired_paths,
            "match_result": match_result,
            "package_path": package_result.get("package_path", ""),
            "files": package_result.get("files", []),
            "copied": copied,
            "message": (
                "AI Crop Fix dry run ready"
                if dry_run
                else "AI Crop Fix request package ready"
                if mode == "ai_scaffold"
                else "AI Crop Fix placeholder repair copied"
            ),
        }
    )


@router.get("/debug/training-store/ai-crop-fix-preview")
def training_store_ai_crop_fix_preview(
    bundle_id: str = Query(...),
    candidate_id: str = Query(...),
):
    try:
        paths = _resolve_ai_crop_fix_preview_paths(bundle_id, candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    tiles = [
        _ai_crop_fix_preview_tile("Original candidate image", paths.get("original_candidate_path", "")),
        _ai_crop_fix_preview_tile("Original alpha mask", paths.get("original_alpha_path", "")),
        _ai_crop_fix_preview_tile("Repaired candidate image", paths.get("repaired_candidate_path", "")),
        _ai_crop_fix_preview_tile("Repaired alpha mask", paths.get("repaired_alpha_path", "")),
    ]
    status_rows = "".join(
        (
            '<tr>'
            f'<th>{escape(label.replace("_", " "))}</th>'
            f'<td>{escape(str(Path(path).exists() and Path(path).is_file()).lower() if path else "false")}</td>'
            f'<td><code>{escape(path or "not resolved")}</code></td>'
            '</tr>'
        )
        for label, path in paths.items()
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AI Crop Fix Preview</title>
  <style>
    body {{ margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f8fb; color: #18212f; }}
    h1 {{ margin: 0 0 4px; font-size: 24px; }}
    .subhead {{ margin: 0 0 20px; color: #5d6978; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; align-items: start; }}
    .preview-tile {{ margin: 0; background: #fff; border: 1px solid #d8e0ea; border-radius: 8px; overflow: hidden; }}
    .preview-image {{ min-height: 180px; display: flex; align-items: center; justify-content: center; background: #eef2f7; }}
    .preview-image img {{ max-width: 100%; max-height: 360px; object-fit: contain; image-rendering: auto; }}
    .missing-image {{ color: #8a2432; font-weight: 700; }}
    figcaption {{ display: grid; gap: 8px; padding: 12px; font-size: 13px; }}
    .status {{ width: fit-content; padding: 3px 8px; border-radius: 999px; font-weight: 700; }}
    .status.ok {{ background: #e8f7ee; color: #17663a; }}
    .status.missing {{ background: #fdecef; color: #8a2432; }}
    code {{ white-space: pre-wrap; word-break: break-word; font-size: 12px; color: #39485c; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #fff; border: 1px solid #d8e0ea; }}
    th, td {{ text-align: left; vertical-align: top; border-top: 1px solid #e6ebf1; padding: 10px; font-size: 13px; }}
    th {{ width: 210px; color: #4d5b6d; }}
  </style>
</head>
<body>
  <h1>AI Crop Fix Preview</h1>
  <p class="subhead">bundle_id: {escape(str(bundle_id or ""))} · candidate_id: {escape(str(candidate_id or ""))}</p>
  <section class="grid">{"".join(tiles)}</section>
  <table>
    <thead><tr><th>file</th><th>exists</th><th>path</th></tr></thead>
    <tbody>{status_rows}</tbody>
  </table>
</body>
</html>"""
    return HTMLResponse(html)


@router.post("/debug/extract-element-page")
async def extract_element_page(req: Request) -> JSONResponse:
    """
    Extract element IDs, qtys and part thumbnails from a LEGO back-page image.

    Body JSON:
      image_path  – absolute or repo-relative path to the page image
      page        – integer page number (used for output naming / run_id)

    Returns:
      { ok, run_id, items: [{element_id, qty, thumbnail_path, bbox,
                              element_bbox, ocr_conf, page}], ... }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    query = dict(req.query_params)

    image_path_raw = str(
        body.get("image_path") or query.get("image_path") or ""
    ).strip()
    try:
        page = int(body.get("page") or query.get("page") or 0)
    except (TypeError, ValueError):
        page = 0

    if not image_path_raw:
        raise HTTPException(status_code=400, detail="image_path is required")

    # Resolve path (absolute or relative to repo root)
    p = Path(image_path_raw).expanduser()
    if not p.is_absolute():
        p = Path("/Users/olly/aim2build-instruction") / p
    resolved = p.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(
            status_code=404, detail=f"image not found: {resolved}"
        )

    img = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if img is None or getattr(img, "size", 0) == 0:
        raise HTTPException(
            status_code=400, detail="image could not be read by OpenCV"
        )

    run_id = f"page_{page:03d}"
    out_dir = _ELEMENT_PAGE_EXTRACT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_mode = bool(body.get("debug", True))

    # ── OCR ────────────────────────────────────────────────────────────────
    ocr = _element_page_ocr(img)
    qty_tokens = ocr["qty_tokens"]
    elem_tokens = ocr["elem_tokens"]
    raw_ocr_tokens = ocr["raw_ocr_tokens"]

    # ── Build items ────────────────────────────────────────────────────────
    items: List[Dict[str, Any]] = []
    for elem in elem_tokens:
        qty = _nearest_qty_for_element(elem, qty_tokens)

        # Use qty bbox as explicit search window so text is excluded before CC
        qty_above = _find_qty_above_element(elem, qty_tokens)
        if qty_above is not None:
            search_top = qty_above["y"] + qty_above["h"] + 4
            search_bottom = elem["y"] - 4
            bbox, window_source = _find_part_thumbnail_above(
                img, elem["x"], elem["y"], elem["w"], elem["h"],
                search_top=search_top,
                search_bottom=search_bottom,
            )
        else:
            bbox, window_source = _find_part_thumbnail_above(
                img, elem["x"], elem["y"], elem["w"], elem["h"],
            )

        thumb_path = ""
        suspicious = False
        suspicious_reason = ""
        text_removed = False
        trim_reason = ""
        final_bbox = bbox

        if bbox is not None:
            bx, by, bw, bh = bbox

            # ── Suspicious checks ──────────────────────────────────────────
            if bh > 180:
                suspicious = True
                suspicious_reason = f"h={bh}>180"
            if not suspicious:
                for other in elem_tokens:
                    # Skip same element_id (handles duplicate OCR detections)
                    if other.get("element_id") == elem.get("element_id"):
                        continue
                    ox, oy = other["x"], other["y"]
                    if bx <= ox <= bx + bw and by <= oy <= by + bh:
                        suspicious = True
                        suspicious_reason = f"overlaps elem {other['element_id']}"
                        break

            crop = img[by : by + bh, bx : bx + bw]
            if crop.size > 0:
                # ── Blank OCR text regions before exporting ────────────────
                # Collect qty labels + element number tokens that may overlap
                text_tokens = list(qty_tokens) + [
                    {"x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"],
                     "text": t["element_id"]}
                    for t in elem_tokens
                ]
                bg_color = _detect_crop_bg_color(crop)
                clean_crop, text_removed, removed_texts = _blank_text_regions_in_crop(
                    crop, bx, by, text_tokens, bg_color
                )
                if text_removed:
                    trim_reason = "blanked: " + ", ".join(removed_texts)

                fname = f"elem_{elem['element_id']}_p{page:03d}.png"
                cv2.imwrite(str(out_dir / fname), clean_crop)
                thumb_path = str(out_dir / fname)

        items.append({
            "element_id": elem["element_id"],
            "raw_text": elem.get("raw_text", elem["element_id"]),
            "digit_len": elem.get("digit_len", len(elem["element_id"])),
            "qty": qty,
            "thumbnail_path": thumb_path,
            "bbox": list(final_bbox) if final_bbox else [],
            "bbox_size": [final_bbox[2], final_bbox[3]] if final_bbox else [],
            "element_bbox": [elem["x"], elem["y"], elem["w"], elem["h"]],
            "ocr_conf": elem["conf"],
            "window_source": window_source,
            "text_removed": text_removed,
            "trim_reason": trim_reason,
            "final_bbox": list(final_bbox) if final_bbox else [],
            "suspicious": suspicious,
            "suspicious_reason": suspicious_reason,
            "page": page,
        })

    # ── Element-only overlay (existing bboxes) ─────────────────────────────
    dbg = img.copy()
    for it in items:
        ex, ey, ew, eh = it["element_bbox"]
        cv2.rectangle(dbg, (ex, ey), (ex + ew, ey + eh), (0, 200, 0), 2)
        cv2.putText(dbg, it["element_id"], (ex, max(0, ey - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1, cv2.LINE_AA)
        if it["bbox"]:
            bx, by, bw2, bh2 = it["bbox"]
            cv2.rectangle(dbg, (bx, by), (bx + bw2, by + bh2), (200, 100, 0), 2)
    for qt in qty_tokens:
        cv2.rectangle(dbg, (qt["x"], qt["y"]),
                      (qt["x"] + qt["w"], qt["y"] + qt["h"]), (180, 0, 220), 1)
    cv2.imwrite(str(out_dir / "debug_overlay.png"), dbg)

    # ── Full OCR overlay – every raw token labelled ────────────────────────
    ocr_dbg = img.copy()
    for tok in raw_ocr_tokens:
        tx, ty, tw, th = tok["x"], tok["y"], tok["w"], tok["h"]
        conf_val = tok["conf"]
        # colour by confidence: red=low, yellow=mid, green=high
        if conf_val < 0:
            colour = (160, 160, 160)   # grey – no conf
        elif conf_val < 50:
            colour = (0, 80, 220)      # orange-red
        elif conf_val < 80:
            colour = (0, 200, 220)     # yellow
        else:
            colour = (0, 200, 60)      # green
        cv2.rectangle(ocr_dbg, (tx, ty), (tx + tw, ty + th), colour, 1)
        label = f"{tok['text']}({int(conf_val)})"
        cv2.putText(ocr_dbg, label, (tx, max(0, ty - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "ocr_overlay.png"), ocr_dbg)

    # ── Manifest ───────────────────────────────────────────────────────────
    manifest: Dict[str, Any] = {
        "ok": True,
        "run_id": run_id,
        "image_path": str(resolved),
        "page": page,
        "image_h": ocr["image_h"],
        "image_w": ocr["image_w"],
        "element_candidate_count": len(items),
        "raw_ocr_token_count": len(raw_ocr_tokens),
        "items": items,
        "qty_tokens": qty_tokens,
        "out_dir": str(out_dir),
    }
    if debug_mode:
        manifest["raw_ocr_tokens"] = raw_ocr_tokens

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    return JSONResponse(manifest)


@router.get("/debug/element-page-file")
def element_page_file(path: str = Query(...)) -> FileResponse:
    """Serve a file from the element_page_extract output directory."""
    allowed_root = _ELEMENT_PAGE_EXTRACT_ROOT.resolve()
    requested = Path(str(path or "").strip()).expanduser()
    if not requested.is_absolute():
        requested = Path("/Users/olly/aim2build-instruction") / requested
    try:
        resolved = requested.resolve()
    except Exception:
        raise HTTPException(status_code=404, detail="file not found")
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise HTTPException(status_code=404, detail="file not found")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(resolved))


def _elem_img_url(abs_path: str) -> str:
    """Return an HTTP URL for a file inside element_page_extract."""
    return f"/debug/element-page-file?path={_url_quote(abs_path)}"


@router.get("/debug/extract-element-page-preview")
def extract_element_page_preview(
    run_id: str = Query(""), page: int = Query(-1)
) -> HTMLResponse:
    """HTML preview for a previous extract-element-page run."""
    if not run_id and page >= 0:
        run_id = f"page_{page:03d}"

    run_dir: Optional[Path] = None
    if run_id:
        candidate = _ELEMENT_PAGE_EXTRACT_ROOT / run_id
        if candidate.exists() and candidate.is_dir():
            run_dir = candidate

    if run_dir is None:
        runs: List[str] = []
        if _ELEMENT_PAGE_EXTRACT_ROOT.exists():
            runs = sorted(
                p.name for p in _ELEMENT_PAGE_EXTRACT_ROOT.iterdir()
                if p.is_dir()
            )
        links = "".join(
            f'<li><a href="/debug/extract-element-page-preview?run_id='
            f'{escape(r)}">{escape(r)}</a></li>'
            for r in runs
        )
        return HTMLResponse(
            "<h2>Element Page Extractions</h2>"
            + (f"<ul>{links}</ul>" if links else "<p>No runs yet.</p>")
            + "<p>POST to <code>/debug/extract-element-page</code> first.</p>"
        )

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"manifest not found in {run_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"manifest parse error: {exc}")

    items: List[Dict[str, Any]] = list(manifest.get("items") or [])

    def _img_tag(abs_path: str, style: str = "") -> str:
        p = Path(abs_path) if abs_path else None
        if p and p.exists() and p.is_file():
            return (
                f'<img src="{_elem_img_url(abs_path)}" '
                f'loading="lazy" style="{style}">'
            )
        return (
            '<div style="display:flex;align-items:center;justify-content:center;'
            'background:#eee;border-radius:4px;color:#999;font-size:11px;'
            'min-height:60px;">not found</div>'
        )

    overlay_path = str(run_dir / "debug_overlay.png")
    ocr_overlay_path = str(run_dir / "ocr_overlay.png")

    overlay_section = (
        f'<h3 style="margin-top:24px;">Element overlay</h3>'
        f'<div style="margin-bottom:8px;">'
        + _img_tag(overlay_path,
                   "max-width:100%;border:1px solid #ccc;border-radius:6px;")
        + f'</div>'
        f'<h3 style="margin-top:24px;">OCR token overlay (all passes)</h3>'
        f'<div style="margin-bottom:16px;">'
        + _img_tag(ocr_overlay_path,
                   "max-width:100%;border:1px solid #ccc;border-radius:6px;")
        + '</div>'
    )

    cards = ""
    for it in items:
        eid = escape(str(it.get("element_id") or ""))
        raw_txt = escape(str(it.get("raw_text") or eid))
        dlen = it.get("digit_len", "")
        qty_str = escape(f"{it['qty']}x") if it.get("qty") is not None else "?"
        tp = str(it.get("thumbnail_path") or "")
        bsize = it.get("bbox_size") or []
        size_str = f"{bsize[0]}×{bsize[1]}" if len(bsize) == 2 else "—"
        susp = bool(it.get("suspicious"))
        susp_reason = escape(str(it.get("suspicious_reason") or ""))
        border_col = "#e05040" if susp else "#d0d8e0"
        thumb_style = (
            "max-width:120px;max-height:120px;display:block;"
            "border:1px solid #ccc;border-radius:4px;background:#f8f8f8;"
        )
        text_removed = bool(it.get("text_removed"))
        trim_reason = escape(str(it.get("trim_reason") or ""))
        win_src = escape(str(it.get("window_source") or ""))
        susp_badge = (
            f'<div style="font-size:10px;color:#c0392b;font-weight:600;'
            f'margin-top:3px;">⚠ {susp_reason}</div>'
            if susp else ""
        )
        trim_badge = (
            f'<div style="font-size:10px;color:#1a6e2e;margin-top:2px;'
            f'word-break:break-all;">✂ {trim_reason}</div>'
            if text_removed else ""
        )
        win_badge = (
            f'<div style="font-size:10px;color:#5566aa;margin-top:2px;">'
            f'&#x1f5d7; {win_src}</div>'
        ) if win_src else ""
        cards += (
            f'<div style="border:2px solid {border_col};border-radius:8px;padding:12px;'
            f'min-width:140px;max-width:160px;display:inline-block;'
            f'vertical-align:top;margin:6px;background:#fff;">'
            + _img_tag(tp, thumb_style)
            + f'<div style="margin-top:8px;font-size:13px;font-weight:600;">{eid}</div>'
            f'<div style="font-size:11px;color:#778899;">raw: {raw_txt} ({dlen}d)</div>'
            f'<div style="font-size:11px;color:#556677;">size: {size_str}</div>'
            f'<div style="font-size:12px;color:#4a6070;">qty: {qty_str}</div>'
            + win_badge
            + trim_badge
            + susp_badge
            + '</div>'
        )

    rid = escape(str(manifest.get("run_id") or run_id))
    pg = escape(str(manifest.get("page") or ""))
    img_p = escape(str(manifest.get("image_path") or ""))
    elem_count = manifest.get("element_candidate_count", len(items))
    raw_count = manifest.get("raw_ocr_token_count", "?")

    html = (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Element Preview – {rid}</title>"
        f"<style>body{{font-family:system-ui,sans-serif;margin:24px;"
        f"background:#f4f6f8;color:#1a2a3a}}"
        f"h1{{font-size:20px}}h3{{font-size:15px;color:#334455}}"
        f".meta{{font-size:13px;color:#556677;margin-bottom:16px}}"
        f"</style></head><body>"
        f"<h1>Element Page Preview</h1>"
        f"<div class='meta'>"
        f"run: <strong>{rid}</strong> &nbsp;|&nbsp; page: <strong>{pg}</strong><br>"
        f"image: <code>{img_p}</code><br>"
        f"elements extracted: <strong>{elem_count}</strong> &nbsp;|&nbsp; "
        f"raw OCR tokens: <strong>{raw_count}</strong>"
        f"</div>"
        + overlay_section
        + f"<h3>Element cards</h3>"
        + (cards if cards else "<p>No elements extracted.</p>")
        + "</body></html>"
    )
    return HTMLResponse(content=html)


# ── SAM2 segmentation routes ──────────────────────────────────────────────────

@router.post("/debug/segment-element-crops")
async def segment_element_crops(req: Request) -> JSONResponse:
    """
    Run hybrid segmentation (_hybrid_segment_crop) on every elem_*.png
    found in the specified page extraction directories.

    Input JSON
    ----------
    {
      "pages": [309, 310, 311],
      "mode":  "sam2_or_fallback"   (reserved; currently always hybrid)
    }

    Output per crop
    ---------------
    debug/element_page_extract/page_XXX/segmented/
      elem_<id>_pXXX_mask.png
      elem_<id>_pXXX_masked.png
      elem_<id>_pXXX_overlay.png
      elem_<id>_pXXX_meta.json

    Returns
    -------
    {
      total, sam2_success, fallback_success, failed,
      average_coverage_pct, average_runtime_ms,
      sam2_available, results
    }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}

    raw_pages = body.get("pages") or []
    try:
        pages: List[int] = [int(p) for p in raw_pages]
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="pages must be a list of integers")

    if not pages:
        raise HTTPException(status_code=400, detail="pages list is required")

    # Ensure SAM2 is loaded (best-effort; falls back to colour if unavailable)
    sam2_available, sam2_err = _sam2_load()

    all_results: List[Dict[str, Any]] = []
    sam2_raw_save_count = 0   # save raw SAM2 outputs for first 10 crops

    for page in pages:
        page_dir = _ELEMENT_PAGE_EXTRACT_ROOT / f"page_{page:03d}"
        if not page_dir.exists():
            continue

        seg_dir = page_dir / "segmented"
        seg_dir.mkdir(parents=True, exist_ok=True)

        crop_paths = sorted(page_dir.glob("elem_*.png"))
        for crop_path in crop_paths:
            stem    = crop_path.stem    # e.g. elem_4114309_p309
            img_bgr = cv2.imread(str(crop_path))
            if img_bgr is None or img_bgr.size == 0:
                all_results.append({
                    "stem": stem, "page": page,
                    "segmentation_method": "failed",
                    "error": "unreadable image",
                    "sam2_loaded": False,
                })
                continue

            # Pass a save prefix for the first 10 crops so raw SAM2 masks are
            # written before the quality guards run — used for diagnostics.
            raw_prefix: Optional[str] = None
            if sam2_raw_save_count < 10:
                raw_prefix = str(seg_dir / stem)
                sam2_raw_save_count += 1

            import time as _time
            t0 = _time.time()
            mask, method, info = _hybrid_segment_crop(img_bgr, save_sam2_raw_prefix=raw_prefix)
            elapsed_ms = round((_time.time() - t0) * 1000)

            # masked: part on white background
            masked = img_bgr.copy()
            masked[mask == 0] = 255

            # overlay: green contour on original
            overlay = img_bgr.copy()
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (0, 220, 80), 1)

            cv2.imwrite(str(seg_dir / f"{stem}_mask.png"),    mask)
            cv2.imwrite(str(seg_dir / f"{stem}_masked.png"),  masked)
            cv2.imwrite(str(seg_dir / f"{stem}_overlay.png"), overlay)

            # Paths to raw SAM2 diagnostic files (only present for first 10 crops)
            sam2_raw_mask_path    = str(seg_dir / f"{stem}_sam2_raw_mask.png")
            sam2_raw_overlay_path = str(seg_dir / f"{stem}_sam2_raw_overlay.png")

            meta: Dict[str, Any] = {
                "stem":                  stem,
                "page":                  page,
                "crop_path":             str(crop_path),
                "segmentation_method":   method,
                "iou":                   info.get("iou"),
                "all_scores":            info.get("all_scores", []),
                "coverage_pct":          info.get("coverage_pct", 0.0),
                "largest_component_pct": info.get("largest_component_pct", 0.0),
                "mask_components":       info.get("mask_components", 0),
                "mask_bbox":             info.get("mask_bbox", []),
                "reject_reason":         info.get("reject_reason"),
                "elapsed_ms":            elapsed_ms,
                "mask_path":             str(seg_dir / f"{stem}_mask.png"),
                "masked_path":           str(seg_dir / f"{stem}_masked.png"),
                "overlay_path":          str(seg_dir / f"{stem}_overlay.png"),
                # ── SAM2 diagnostics (pre-guard) ──────────────────────────
                "sam2_loaded":                      info.get("sam2_loaded", False),
                "sam2_exception":                   info.get("sam2_exception"),
                "sam2_mask_count":                  info.get("sam2_mask_count", 0),
                "sam2_scores":                      info.get("all_scores", []),
                "sam2_raw_coverage_pct":            info.get("sam2_raw_coverage_pct"),
                "sam2_raw_largest_component_pct":   info.get("sam2_raw_largest_component_pct"),
                "sam2_reject_reason":               info.get("sam2_reject_reason"),
                "sam2_raw_mask_path":    sam2_raw_mask_path if raw_prefix else None,
                "sam2_raw_overlay_path": sam2_raw_overlay_path if raw_prefix else None,
            }
            (seg_dir / f"{stem}_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8"
            )
            all_results.append(meta)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok      = [r for r in all_results if r.get("segmentation_method") != "failed"]
    n_sam2  = sum(1 for r in ok if r.get("segmentation_method") == "sam2")
    n_fb    = sum(1 for r in ok if r.get("segmentation_method") == "fallback_colour")
    n_fail  = sum(1 for r in all_results if r.get("segmentation_method") == "failed")
    avg_cov = round(sum(r.get("coverage_pct", 0) for r in ok) / max(len(ok), 1), 2)
    avg_ms  = round(sum(r.get("elapsed_ms", 0)   for r in ok) / max(len(ok), 1))

    # SAM2-specific diagnostic counters
    n_sam2_attempted = sum(1 for r in all_results if r.get("sam2_loaded"))
    n_sam2_exception = sum(1 for r in all_results if r.get("sam2_exception"))
    n_sam2_rejected  = sum(1 for r in all_results if r.get("sam2_reject_reason"))

    return JSONResponse({
        "ok":                     True,
        "total":                  len(all_results),
        "sam2_success":           n_sam2,
        "fallback_success":       n_fb,
        "failed":                 n_fail,
        "average_coverage_pct":   avg_cov,
        "average_runtime_ms":     avg_ms,
        "sam2_available":         sam2_available,
        "sam2_error":             sam2_err if not sam2_available else None,
        "pages_processed":        pages,
        # ── SAM2 diagnostic summary ───────────────────────────────────────
        "sam2_attempted":         n_sam2_attempted,
        "sam2_exception_count":   n_sam2_exception,
        "sam2_rejected_by_guard": n_sam2_rejected,
        "sam2_accepted":          n_sam2,
        "fallback_used":          n_fb,
        "sam2_raw_saves":         sam2_raw_save_count,
        "results":                all_results,
    })


@router.get("/debug/segment-element-crops-file")
def segment_element_crops_file(path: str = Query(...)) -> FileResponse:
    """Serve a file from any element_page_extract/segmented/ subdirectory."""
    allowed_root = _ELEMENT_PAGE_EXTRACT_ROOT.resolve()
    requested    = Path(str(path or "").strip()).expanduser()
    if not requested.is_absolute():
        requested = Path("/Users/olly/aim2build-instruction") / requested
    try:
        resolved = requested.resolve()
    except Exception:
        raise HTTPException(status_code=404, detail="file not found")
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise HTTPException(status_code=404, detail="file not found")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(resolved))


@router.get("/debug/segment-element-crops-preview")
def segment_element_crops_preview(page: int = Query(-1)) -> HTMLResponse:
    """
    HTML preview of segmentation results for one page.

    Shows one row per element:  original | mask | masked | overlay
    plus method badge, coverage %, reject_reason.
    """
    if page < 0:
        # List available pages
        pages_available: List[str] = []
        if _ELEMENT_PAGE_EXTRACT_ROOT.exists():
            for d in sorted(_ELEMENT_PAGE_EXTRACT_ROOT.iterdir()):
                if d.is_dir() and (d / "segmented").exists():
                    pages_available.append(d.name)
        links = "".join(
            f'<li><a href="/debug/segment-element-crops-preview?page='
            f'{escape(p.replace("page_",""))}">{escape(p)}</a></li>'
            for p in pages_available
        )
        return HTMLResponse(
            "<h2>Segmentation Previews</h2>"
            + (f"<ul>{links}</ul>" if links
               else "<p>No segmented pages yet.</p>")
            + "<p>POST to <code>/debug/segment-element-crops</code> first.</p>"
        )

    seg_dir = _ELEMENT_PAGE_EXTRACT_ROOT / f"page_{page:03d}" / "segmented"
    if not seg_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No segmented output for page {page}. "
                   "Run POST /debug/segment-element-crops first."
        )

    meta_files = sorted(seg_dir.glob("*_meta.json"))
    if not meta_files:
        raise HTTPException(status_code=404, detail="No meta.json files found.")

    def _seg_img(abs_path: str, style: str = "") -> str:
        p = Path(abs_path) if abs_path else None
        if p and p.exists():
            url = f"/debug/segment-element-crops-file?path={_url_quote(str(p))}"
            return f'<img src="{url}" loading="lazy" style="{style}">'
        return (
            '<div style="display:flex;align-items:center;justify-content:center;'
            'background:#eee;border-radius:3px;color:#aaa;font-size:10px;'
            'min-width:72px;min-height:72px;">—</div>'
        )

    def _orig_img(page_num: int, stem: str, style: str = "") -> str:
        orig = _ELEMENT_PAGE_EXTRACT_ROOT / f"page_{page_num:03d}" / f"{stem}.png"
        if orig.exists():
            url = f"/debug/element-page-file?path={_url_quote(str(orig))}"
            return f'<img src="{url}" loading="lazy" style="{style}">'
        return '<div style="min-width:72px;min-height:72px;background:#eee;"></div>'

    img_style = (
        "width:80px;height:80px;object-fit:contain;"
        "border:1px solid #ccc;border-radius:3px;background:#fff;"
    )

    METHOD_COL = {
        "sam2":            "#2a7a2a",
        "fallback_colour": "#7a5a14",
        "failed":          "#8a1a1a",
    }

    rows_html = ""
    for mf in meta_files:
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue

        stem      = escape(str(m.get("stem", "")))
        eid       = stem.replace("elem_", "").split("_p")[0]
        method    = str(m.get("segmentation_method") or "?")
        cov       = m.get("coverage_pct", 0)
        lc        = m.get("largest_component_pct", 0)
        nc        = m.get("mask_components", 0)
        iou_val   = m.get("iou")
        iou_str   = f"{iou_val:.3f}" if iou_val is not None else "—"
        rej       = escape(str(m.get("reject_reason") or ""))
        ms        = m.get("elapsed_ms", 0)
        badge_col = METHOD_COL.get(method, "#555")

        rows_html += (
            f'<tr>'
            f'<td style="font-size:12px;padding:4px 8px;white-space:nowrap;">'
            f'  <strong>{eid}</strong>'
            f'  <div style="font-size:10px;color:#778;">{stem}</div>'
            f'</td>'
            f'<td style="padding:3px;">{_orig_img(page, m.get("stem",""), img_style)}</td>'
            f'<td style="padding:3px;">{_seg_img(str(m.get("mask_path","")), img_style)}</td>'
            f'<td style="padding:3px;">{_seg_img(str(m.get("masked_path","")), img_style)}</td>'
            f'<td style="padding:3px;">{_seg_img(str(m.get("overlay_path","")), img_style)}</td>'
            f'<td style="font-size:11px;padding:4px 8px;white-space:nowrap;">'
            f'  <span style="background:{badge_col};color:#fff;padding:1px 6px;'
            f'border-radius:3px;font-size:10px;">{escape(method)}</span><br>'
            f'  cov {cov:.1f}%&nbsp; lc {lc:.1f}%&nbsp; nc {nc}<br>'
            f'  iou {iou_str}&nbsp; {ms}ms'
            + (f'  <br><span style="color:#c80;font-size:10px;">⚠ {rej}</span>'
               if rej else "")
            + '</td>'
            f'</tr>'
        )

    total     = len(meta_files)
    n_sam2    = sum(1 for mf in meta_files
                    if json.loads(mf.read_text()).get("segmentation_method") == "sam2")
    n_fb      = total - n_sam2

    html = (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Segmentation Preview – page {page}</title>"
        f"<style>"
        f"body{{font-family:system-ui,sans-serif;margin:24px;background:#f4f6f8;color:#1a2a3a}}"
        f"h1{{font-size:20px}}"
        f"table{{border-collapse:collapse;background:#fff;border-radius:8px;"
        f"overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1)}}"
        f"th{{background:#2a3a4a;color:#fff;padding:8px 12px;font-size:12px;text-align:left}}"
        f"tr:nth-child(even){{background:#f8fafc}}"
        f"td{{vertical-align:middle}}"
        f"</style></head><body>"
        f"<h1>Segmentation Preview — page {page}</h1>"
        f"<div style='font-size:13px;margin-bottom:16px;color:#556677;'>"
        f"total: <strong>{total}</strong> &nbsp;|&nbsp; "
        f"sam2: <strong>{n_sam2}</strong> &nbsp;|&nbsp; "
        f"fallback: <strong>{n_fb}</strong>"
        f"</div>"
        f"<table>"
        f"<tr><th>element</th><th>original</th><th>mask</th>"
        f"<th>masked</th><th>overlay</th><th>metrics</th></tr>"
        + rows_html
        + "</table></body></html>"
    )
    return HTMLResponse(content=html)


@router.post("/debug/export-training-packs")
async def export_training_packs(req: Request) -> JSONResponse:
    """
    Package segmented element crops into per-element training-pack folders.

    Reads existing segmented outputs from:
        debug/element_page_extract/page_XXX/segmented/

    Writes to:
        debug/element_training_packs/<stem>/
            original.png   (extracted thumbnail)
            mask.png
            masked.png
            overlay.png
            meta.json      (element_id, qty, set_num, segmentation fields,
                            clip_quality, source_paths)
        debug/element_training_packs/manifest.json

    Input JSON
    ----------
    { "pages": [309, 310, 311] }    — defaults to [309, 310, 311]

    clip_quality rules
    ------------------
    high   : coverage_pct >= 20
    medium : coverage_pct >= 10 and < 20
    low    : coverage_pct < 10
    """
    try:
        body = await req.json()
    except Exception:
        body = {}

    raw_pages = body.get("pages") or [309, 310, 311]
    try:
        pages: List[int] = [int(p) for p in raw_pages]
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="pages must be a list of integers")

    _TRAINING_PACK_ROOT.mkdir(parents=True, exist_ok=True)

    def _clip_quality(cov: float) -> str:
        if cov >= 20:
            return "high"
        if cov >= 10:
            return "medium"
        return "low"

    def _set_num_from_image_path(image_path: str) -> str:
        """Extract set number from paths like .../debug/70618/70618_01/pages/page_309.png"""
        try:
            parts = Path(image_path).parts
            idx = parts.index("debug")
            return parts[idx + 1]
        except (ValueError, IndexError):
            return "unknown"

    manifest_entries: List[Dict[str, Any]] = []
    total_exported = 0
    missing_files  = 0
    cq_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}

    for page in pages:
        page_dir = _ELEMENT_PAGE_EXTRACT_ROOT / f"page_{page:03d}"
        seg_dir  = page_dir / "segmented"
        if not seg_dir.exists():
            continue

        # ── Load per-page manifest to resolve qty and set_num ────────────────
        qty_map:  Dict[str, int]  = {}
        set_num:  str             = "unknown"
        page_manifest_path = page_dir / "manifest.json"
        if page_manifest_path.exists():
            try:
                pm = json.loads(page_manifest_path.read_text(encoding="utf-8"))
                set_num = _set_num_from_image_path(pm.get("image_path", ""))
                for item in pm.get("items", []):
                    eid = str(item.get("element_id", ""))
                    if eid:
                        qty_map[eid] = item.get("qty") or 0
            except Exception:
                pass

        # ── Process each segmented crop ───────────────────────────────────────
        for meta_path in sorted(seg_dir.glob("*_meta.json")):
            try:
                seg_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                missing_files += 1
                continue

            stem       = seg_meta.get("stem", meta_path.stem.replace("_meta", ""))
            element_id = stem.replace("elem_", "").split("_p")[0]
            cov        = seg_meta.get("coverage_pct", 0.0)
            cq         = _clip_quality(cov)

            # ── Create item folder ────────────────────────────────────────────
            item_dir = _TRAINING_PACK_ROOT / stem
            item_dir.mkdir(parents=True, exist_ok=True)

            # ── Copy source files → canonical names ───────────────────────────
            src_original = Path(seg_meta.get("crop_path", ""))
            src_mask     = Path(seg_meta.get("mask_path", ""))
            src_masked   = Path(seg_meta.get("masked_path", ""))
            src_overlay  = Path(seg_meta.get("overlay_path", ""))

            file_map = {
                "original.png": src_original,
                "mask.png":     src_mask,
                "masked.png":   src_masked,
                "overlay.png":  src_overlay,
            }
            src_ok = True
            for dest_name, src_path in file_map.items():
                dest = item_dir / dest_name
                if src_path.exists():
                    shutil.copy2(str(src_path), str(dest))
                else:
                    missing_files += 1
                    src_ok = False

            # ── Write item meta.json ──────────────────────────────────────────
            item_meta: Dict[str, Any] = {
                "set_num":              set_num,
                "page":                 page,
                "element_id":           element_id,
                "qty":                  qty_map.get(element_id, 0),
                "segmentation_method":  seg_meta.get("segmentation_method"),
                "coverage_pct":         cov,
                "iou":                  seg_meta.get("iou"),
                "clip_quality":         cq,
                "sam2_reject_reason":   seg_meta.get("sam2_reject_reason"),
                "reject_reason":        seg_meta.get("reject_reason"),
                "largest_component_pct": seg_meta.get("largest_component_pct"),
                "mask_components":      seg_meta.get("mask_components"),
                "elapsed_ms":           seg_meta.get("elapsed_ms"),
                "source_paths": {
                    "original": str(src_original),
                    "mask":     str(src_mask),
                    "masked":   str(src_masked),
                    "overlay":  str(src_overlay),
                },
                "pack_paths": {
                    "original": str(item_dir / "original.png"),
                    "mask":     str(item_dir / "mask.png"),
                    "masked":   str(item_dir / "masked.png"),
                    "overlay":  str(item_dir / "overlay.png"),
                },
            }
            (item_dir / "meta.json").write_text(
                json.dumps(item_meta, indent=2, ensure_ascii=True), encoding="utf-8"
            )

            # ── Manifest entry ────────────────────────────────────────────────
            manifest_entries.append({
                "set_num":             set_num,
                "page":                page,
                "element_id":          element_id,
                "qty":                 qty_map.get(element_id, 0),
                "segmentation_method": seg_meta.get("segmentation_method"),
                "coverage_pct":        cov,
                "iou":                 seg_meta.get("iou"),
                "clip_quality":        cq,
                "source_paths": {
                    "original": str(src_original),
                    "mask":     str(src_mask),
                    "masked":   str(src_masked),
                    "overlay":  str(src_overlay),
                },
            })

            cq_counts[cq] += 1
            if src_ok:
                total_exported += 1

    # ── Write top-level manifest ──────────────────────────────────────────────
    manifest_path = _TRAINING_PACK_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at":  __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "pages":         pages,
                "total":         len(manifest_entries),
                "clip_quality":  cq_counts,
                "missing_files": missing_files,
                "items":         manifest_entries,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    return JSONResponse({
        "ok":            True,
        "total_exported": total_exported,
        "high":           cq_counts["high"],
        "medium":         cq_counts["medium"],
        "low":            cq_counts["low"],
        "missing_files":  missing_files,
        "pages_processed": pages,
        "manifest_path":  str(manifest_path),
        "pack_root":      str(_TRAINING_PACK_ROOT),
    })


@router.post("/debug/generate-training-clip-embeddings")
async def generate_training_clip_embeddings(req: Request) -> JSONResponse:
    """
    Generate CLIP image embeddings for element training packs.

    Reads masked.png from each item in:
        debug/element_training_packs/<stem>/masked.png

    Writes to:
        debug/clip_training_embeddings/
            embeddings.npy    float32 (N, D), L2-normalised, one row per embedded item
            items.json        metadata for each embedded item (index = row in embeddings.npy)
            manifest.json     run summary

    Model: ViT-B-32 / laion2b_s34b_b79k  (open_clip)
    Device: MPS if available, else CPU

    Input JSON
    ----------
    {
      "limit":   null,      -- max items to embed (null = all)
      "quality": "all"      -- "all" | "high" | "medium" | "low"
    }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}

    limit_raw      = body.get("limit")
    quality_filter = str(body.get("quality") or "all").lower().strip()

    if quality_filter not in ("all", "high", "medium", "low"):
        raise HTTPException(status_code=400,
            detail="quality must be one of: all, high, medium, low")

    # ── Load CLIP (lazy, first call downloads weights ~350 MB) ────────────────
    clip_ok, clip_err = _clip_load()
    if not clip_ok:
        raise HTTPException(status_code=503,
            detail=f"CLIP unavailable: {clip_err}. "
                   f"Install with: pip install open-clip-torch")

    # ── Load training pack manifest ───────────────────────────────────────────
    pack_manifest_path = _TRAINING_PACK_ROOT / "manifest.json"
    if not pack_manifest_path.exists():
        raise HTTPException(status_code=404,
            detail="Training pack manifest not found. "
                   "Run POST /debug/export-training-packs first.")

    try:
        pack_manifest = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"Failed to read manifest: {exc}")

    all_items: List[Dict[str, Any]] = pack_manifest.get("items", [])

    # ── Filter by clip_quality ────────────────────────────────────────────────
    if quality_filter != "all":
        all_items = [it for it in all_items if it.get("clip_quality") == quality_filter]

    # ── Apply limit ───────────────────────────────────────────────────────────
    if limit_raw is not None:
        try:
            all_items = all_items[: int(limit_raw)]
        except (TypeError, ValueError):
            pass

    total_input = len(all_items)

    # ── Resolve masked.png paths ──────────────────────────────────────────────
    import torch
    import numpy as np
    from PIL import Image as _Pil

    BATCH_SIZE = 32
    device_str = "mps" if torch.backends.mps.is_available() else "cpu"

    # Two lists kept in sync: one for embedding, one for output metadata
    load_tensors: List[Any]       = []   # preprocessed tensors
    embed_items:  List[Dict[str, Any]] = []  # items that loaded OK
    skipped:      List[Dict[str, Any]] = []  # items that failed

    for item in all_items:
        element_id = str(item.get("element_id", ""))
        page       = int(item.get("page", 0))
        stem       = f"elem_{element_id}_p{page:03d}"
        masked_path = _TRAINING_PACK_ROOT / stem / "masked.png"
        meta_path   = _TRAINING_PACK_ROOT / stem / "meta.json"

        if not masked_path.exists():
            skipped.append({"stem": stem, "reason": "masked.png not found"})
            continue

        try:
            img    = _Pil.open(str(masked_path)).convert("RGB")
            tensor = _clip_preprocess(img).unsqueeze(0)   # (1, C, H, W)
        except Exception as exc:
            skipped.append({"stem": stem, "reason": f"load error: {exc}"})
            continue

        load_tensors.append(tensor)
        embed_items.append({
            "stem":               stem,
            "element_id":         element_id,
            "page":               page,
            "set_num":            item.get("set_num", ""),
            "qty":                item.get("qty", 0),
            "clip_quality":       item.get("clip_quality", ""),
            "segmentation_method": item.get("segmentation_method", ""),
            "coverage_pct":       item.get("coverage_pct", 0.0),
            "image_path":         str(masked_path),
            "meta_path":          str(meta_path),
        })

    if not load_tensors:
        raise HTTPException(status_code=422,
            detail=f"No images could be loaded (total_input={total_input}, "
                   f"skipped={len(skipped)}). "
                   "Run POST /debug/export-training-packs first.")

    # ── Batch encode ──────────────────────────────────────────────────────────
    all_tensors = torch.cat(load_tensors, dim=0).to(device_str)  # (N, C, H, W)

    all_feats: List[Any] = []
    with torch.no_grad():
        for b_start in range(0, all_tensors.shape[0], BATCH_SIZE):
            batch = all_tensors[b_start : b_start + BATCH_SIZE]
            feats = _clip_model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2-normalise
            all_feats.append(feats.cpu().float())

    embeddings: np.ndarray = torch.cat(all_feats, dim=0).numpy()  # (N, D)
    embed_dim  = int(embeddings.shape[1])

    # ── Assign final index (= row in embeddings.npy) and build items.json ─────
    items_out: List[Dict[str, Any]] = []
    for idx, item_meta in enumerate(embed_items):
        items_out.append({
            "index":              idx,
            "set_num":            item_meta["set_num"],
            "page":               item_meta["page"],
            "element_id":         item_meta["element_id"],
            "qty":                item_meta["qty"],
            "clip_quality":       item_meta["clip_quality"],
            "segmentation_method": item_meta["segmentation_method"],
            "coverage_pct":       item_meta["coverage_pct"],
            "image_path":         item_meta["image_path"],
            "meta_path":          item_meta["meta_path"],
        })

    # ── Write outputs ─────────────────────────────────────────────────────────
    _CLIP_EMBEDDINGS_ROOT.mkdir(parents=True, exist_ok=True)

    embeddings_path = _CLIP_EMBEDDINGS_ROOT / "embeddings.npy"
    items_path      = _CLIP_EMBEDDINGS_ROOT / "items.json"
    manifest_out    = _CLIP_EMBEDDINGS_ROOT / "manifest.json"

    np.save(str(embeddings_path), embeddings)

    items_path.write_text(
        json.dumps(items_out, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    manifest_out.write_text(
        json.dumps({
            "generated_at":    __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "model_name":      _CLIP_MODEL_NAME,
            "pretrained":      _CLIP_PRETRAINED,
            "device":          device_str,
            "quality_filter":  quality_filter,
            "total_input":     total_input,
            "embedded_count":  len(items_out),
            "skipped_count":   len(skipped),
            "embedding_dim":   embed_dim,
            "embeddings_path": str(embeddings_path),
            "items_path":      str(items_path),
            "skipped":         skipped,
        }, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return JSONResponse({
        "ok":              True,
        "total_input":     total_input,
        "embedded_count":  len(items_out),
        "skipped_count":   len(skipped),
        "embedding_dim":   embed_dim,
        "model_name":      f"{_CLIP_MODEL_NAME} / {_CLIP_PRETRAINED}",
        "device":          device_str,
        "output_paths": {
            "embeddings":  str(embeddings_path),
            "items":       str(items_path),
            "manifest":    str(manifest_out),
        },
    })


# ── Catalog CLIP helpers ───────────────────────────────────────────────────────

def _catalog_query_parts(set_num: str) -> List[Dict[str, Any]]:
    """
    Return one row per distinct (part_num, color_id) in the given set's
    inventory, with the best available image URL from the elements table.
    """
    import sqlite3 as _sqlite3
    if not _CATALOG_DB_PATH.exists():
        raise FileNotFoundError(f"Catalog DB not found: {_CATALOG_DB_PATH}")

    conn = _sqlite3.connect(f"file:{_CATALOG_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = _sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                ip.part_num,
                ip.color_id,
                SUM(COALESCE(ip.quantity, 0)) AS qty,
                MAX(COALESCE(e.img_rebrick, e.img_ldraw, e.img_custom)) AS img_url
            FROM inventory_parts ip
            INNER JOIN inventories inv
                    ON inv.inventory_id = ip.inventory_id
                   AND inv.set_num      = ?
            LEFT JOIN elements e
                   ON e.part_num  = ip.part_num
                  AND e.color_id  = ip.color_id
            GROUP BY ip.part_num, ip.color_id
            ORDER BY ip.part_num, ip.color_id
            """,
            (f"{set_num}-1",),
        ).fetchall()
    finally:
        conn.close()

    return [dict(r) for r in rows]


def _catalog_fetch_image(img_url: str, dest: Path) -> bool:
    """Download img_url → dest (PNG), return True on success."""
    import urllib.request as _urr
    import io as _io
    try:
        req = _urr.Request(str(img_url), headers={"User-Agent": "aim2build-catalog/1.0"})
        with _urr.urlopen(req, timeout=15) as resp:
            data = resp.read()
        # Convert to PNG via PIL regardless of source format
        from PIL import Image as _Pil
        img = _Pil.open(_io.BytesIO(data)).convert("RGB")
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(dest), format="PNG")
        return True
    except Exception:
        return False


@router.post("/debug/generate-catalog-clip-embeddings")
async def generate_catalog_clip_embeddings(req: Request) -> JSONResponse:
    """
    Download (if needed) and CLIP-embed all catalog part images for a set.

    Steps
    -----
    1. Query lego_catalog.db for all distinct (part_num, color_id) in the
       set's inventory, retrieving the best available img_url.
    2. Cache each image locally at:
           debug/part_image_cache/<set_num>/<part_num>_<color_id>.png
       (skip download if file already exists)
    3. CLIP-embed all cached images (ViT-B-32 / laion2b_s34b_b79k, MPS).
    4. Write:
           debug/catalog_clip_embeddings/embeddings.npy  float32 (N, 512) L2-normalised
           debug/catalog_clip_embeddings/items.json      index-aligned with embeddings
           debug/catalog_clip_embeddings/manifest.json   run summary

    Input JSON
    ----------
    { "set_num": "70618" }   -- defaults to "70618"

    Returns
    -------
    catalog_count_scanned, embedded_count, skipped_count,
    embedding_shape, output_paths, model_name
    """
    try:
        body = await req.json()
    except Exception:
        body = {}

    set_num: str = str(body.get("set_num") or "70618").strip()

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    clip_ok, clip_err = _clip_load()
    if not clip_ok:
        raise HTTPException(status_code=503,
            detail=f"CLIP unavailable: {clip_err}. "
                   "Install with: pip install open-clip-torch")

    # ── Query catalog DB ──────────────────────────────────────────────────────
    try:
        catalog_parts = _catalog_query_parts(set_num)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"Catalog DB error: {exc}")

    catalog_count_scanned = len(catalog_parts)
    img_cache_dir = _CATALOG_IMG_CACHE_ROOT / set_num
    img_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Download / cache images ───────────────────────────────────────────────
    import torch
    import numpy as np
    from PIL import Image as _Pil

    BATCH_SIZE = 32

    load_tensors:  List[Any]            = []
    embed_items:   List[Dict[str, Any]] = []
    skipped:       List[Dict[str, Any]] = []

    for part in catalog_parts:
        part_num = str(part.get("part_num") or "")
        color_id = int(part.get("color_id") or 0)
        img_url  = part.get("img_url") or ""
        qty      = int(part.get("qty") or 0)

        # Canonical local filename: <part_num>_<color_id>.png
        local_img = img_cache_dir / f"{part_num}_{color_id}.png"

        # Download if not yet cached
        if not local_img.exists():
            if not img_url:
                skipped.append({
                    "part_num": part_num, "color_id": color_id,
                    "reason": "no_img_url",
                })
                continue
            ok = _catalog_fetch_image(img_url, local_img)
            if not ok:
                skipped.append({
                    "part_num": part_num, "color_id": color_id,
                    "img_url": img_url, "reason": "download_failed",
                })
                continue

        # Load for embedding
        try:
            img    = _Pil.open(str(local_img)).convert("RGB")
            tensor = _clip_preprocess(img).unsqueeze(0)
        except Exception as exc:
            skipped.append({
                "part_num": part_num, "color_id": color_id,
                "reason": f"load_error: {exc}",
            })
            continue

        load_tensors.append(tensor)
        embed_items.append({
            "part_num": part_num,
            "color_id": color_id,
            "qty":      qty,
            "img_path": str(local_img),
            "img_url":  img_url,
        })

    if not load_tensors:
        raise HTTPException(status_code=422,
            detail=f"No catalog images available to embed "
                   f"(scanned={catalog_count_scanned}, skipped={len(skipped)}). "
                   "Check that img_url values are reachable and "
                   "debug/server_catalog/lego_catalog.db is populated.")

    # ── Batch encode ──────────────────────────────────────────────────────────
    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    all_tensors = torch.cat(load_tensors, dim=0).to(device_str)

    all_feats: List[Any] = []
    with torch.no_grad():
        for b_start in range(0, all_tensors.shape[0], BATCH_SIZE):
            batch = all_tensors[b_start : b_start + BATCH_SIZE]
            feats = _clip_model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().float())

    embeddings: np.ndarray = torch.cat(all_feats, dim=0).numpy()
    embed_dim = int(embeddings.shape[1])

    # ── Build items.json (index-aligned with embeddings.npy) ─────────────────
    items_out: List[Dict[str, Any]] = [
        {
            "embedding_index": idx,
            "part_num":        m["part_num"],
            "color_id":        m["color_id"],
            "img_path":        m["img_path"],
        }
        for idx, m in enumerate(embed_items)
    ]

    # ── Write outputs ─────────────────────────────────────────────────────────
    _CATALOG_CLIP_ROOT.mkdir(parents=True, exist_ok=True)

    embeddings_path = _CATALOG_CLIP_ROOT / "embeddings.npy"
    items_path      = _CATALOG_CLIP_ROOT / "items.json"
    manifest_path   = _CATALOG_CLIP_ROOT / "manifest.json"

    np.save(str(embeddings_path), embeddings)

    items_path.write_text(
        json.dumps(items_out, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    manifest_path.write_text(
        json.dumps(
            {
                "model":          _CLIP_MODEL_NAME,
                "pretrained":     _CLIP_PRETRAINED,
                "device":         device_str,
                "embedding_dim":  embed_dim,
                "count":          len(items_out),
                "generated_at":   __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "source_roots":   [str(img_cache_dir)],
                "set_num":        set_num,
                "skipped_count":  len(skipped),
                "skipped":        skipped,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    return JSONResponse({
        "ok":                   True,
        "set_num":              set_num,
        "catalog_count_scanned": catalog_count_scanned,
        "embedded_count":       len(items_out),
        "skipped_count":        len(skipped),
        "embedding_shape":      list(embeddings.shape),
        "model_name":           f"{_CLIP_MODEL_NAME} / {_CLIP_PRETRAINED}",
        "device":               device_str,
        "output_paths": {
            "embeddings":  str(embeddings_path),
            "items":       str(items_path),
            "manifest":    str(manifest_path),
        },
    })


# ── Catalog-match helpers ──────────────────────────────────────────────────────

def _cat_emb_load():
    """Load catalog embeddings once per process. Returns (matrix, items)."""
    global _cat_emb_matrix, _cat_emb_items
    if _cat_emb_matrix is not None:
        return _cat_emb_matrix, _cat_emb_items
    import numpy as np
    emb_path   = _CATALOG_CLIP_ROOT / "embeddings.npy"
    items_path = _CATALOG_CLIP_ROOT / "items.json"
    if not emb_path.exists() or not items_path.exists():
        raise FileNotFoundError(
            "Catalog embeddings not found. "
            "Run POST /debug/generate-catalog-clip-embeddings first."
        )
    _cat_emb_matrix = np.load(str(emb_path)).astype("float32")
    _cat_emb_items  = json.loads(items_path.read_text(encoding="utf-8"))
    return _cat_emb_matrix, _cat_emb_items


def _clip_embed_image(img_path: str):
    """Embed a single image file with the loaded CLIP model. Returns (512,) ndarray."""
    import torch, numpy as np
    from PIL import Image as _Pil
    img    = _Pil.open(img_path).convert("RGB")
    tensor = _clip_preprocess(img).unsqueeze(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with torch.no_grad():
        feat = _clip_model.encode_image(tensor.to(device))
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]  # (512,)


# ── Shape/silhouette helpers (Phase 4 reranker) ────────────────────────────────

def _catalog_mask_from_image(img: np.ndarray) -> np.ndarray:
    """
    Extract a binary foreground mask from a catalog part image.
    Handles BGRA (alpha channel) and BGR (white-bg threshold).
    Returns uint8 mask: 255 = foreground, 0 = background.
    """
    if img is None or img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        return mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    _, mask = cv2.threshold(gray, 239, 255, cv2.THRESH_BINARY_INV)
    return mask


def _shape_features(mask: np.ndarray) -> Dict[str, Any]:
    """
    Compute shape descriptors from a binary mask.
    Returns: aspect_ratio, solidity, extent, contour (largest), hu_moments, area.
    All values default to 0 on empty/bad input.
    """
    _empty: Dict[str, Any] = {
        "aspect_ratio": 0.0, "solidity": 0.0, "extent": 0.0,
        "contour": None, "hu_moments": [0.0] * 7, "area": 0.0,
    }
    if mask is None or mask.size == 0:
        return _empty
    bw = np.where(mask > 127, np.uint8(255), np.uint8(0))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return _empty
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < 4:
        return _empty
    x, y, w, h = cv2.boundingRect(cnt)
    w, h = max(w, 1), max(h, 1)
    aspect_ratio = float(min(w, h)) / float(max(w, h))   # [0,1], rotation-invariant
    extent = min(area / float(w * h), 1.0)
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = min(area / hull_area, 1.0) if hull_area > 0 else 0.0
    M = cv2.moments(cnt)
    hu = cv2.HuMoments(M).flatten().tolist()
    return {
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "extent": extent,
        "contour": cnt,
        "hu_moments": hu,
        "area": area,
    }


def _shape_score(feat_q: Dict[str, Any], feat_c: Dict[str, Any]) -> float:
    """
    Composite shape similarity in [0, 1].
    Weights: Hu-moments 0.40, aspect_ratio 0.30, solidity 0.15, extent 0.15.
    """
    import math
    ar_sim  = max(0.0, 1.0 - abs(feat_q["aspect_ratio"] - feat_c["aspect_ratio"]))
    cnt_q, cnt_c = feat_q.get("contour"), feat_c.get("contour")
    if cnt_q is not None and cnt_c is not None and len(cnt_q) >= 3 and len(cnt_c) >= 3:
        try:
            hu_dist = float(cv2.matchShapes(cnt_q, cnt_c, cv2.CONTOURS_MATCH_I2, 0))
            hu_dist = min(hu_dist, 5.0)
        except Exception:
            hu_dist = 5.0
    else:
        hu_dist = 5.0
    hu_sim  = math.exp(-0.5 * hu_dist)
    sol_sim = max(0.0, 1.0 - abs(feat_q["solidity"] - feat_c["solidity"]))
    ext_sim = max(0.0, 1.0 - abs(feat_q["extent"]   - feat_c["extent"]))
    return 0.30 * ar_sim + 0.40 * hu_sim + 0.15 * sol_sim + 0.15 * ext_sim


def _shape_rerank(
    query_mask_path: str,
    top20: List[Dict[str, Any]],
    alpha: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Rerank CLIP top-20 by blending CLIP cosine score with shape similarity.
        final_score = alpha * clip_score + (1 - alpha) * shape_score

    Input items must have: clip_rank (int), clip_score (float), cat_item (dict w/ img_path).
    Returns list sorted by final_score desc; each item gains shape_score, final_score.
    """
    q_mask = cv2.imread(query_mask_path, cv2.IMREAD_GRAYSCALE)
    feat_q = _shape_features(q_mask)

    result = []
    for item in top20:
        clip_score  = item["clip_score"]
        img_path    = item["cat_item"].get("img_path", "")
        shape_score = 0.0
        feat_c_safe: Dict[str, Any] = {}
        if img_path and Path(img_path).exists():
            try:
                cat_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if cat_img is not None and cat_img.size > 0:
                    cat_mask    = _catalog_mask_from_image(cat_img)
                    feat_c      = _shape_features(cat_mask)
                    shape_score = _shape_score(feat_q, feat_c)
                    feat_c_safe = {k: v for k, v in feat_c.items() if k != "contour"}
            except Exception:
                pass
        final_score = alpha * clip_score + (1.0 - alpha) * shape_score
        result.append({
            **item,
            "shape_score": round(shape_score, 4),
            "final_score": round(final_score, 4),
            "shape_feat_c": feat_c_safe,
        })
    result.sort(key=lambda x: -x["final_score"])
    return result


def _match_file_url(abs_path: str) -> str:
    return f"/debug/clip-match-file?path={_url_quote(abs_path)}"


@router.get("/debug/clip-match-file")
def clip_match_file(path: str = Query(...)) -> FileResponse:
    """Serve image files from training-pack and catalog-image-cache directories."""
    allowed = [
        _TRAINING_PACK_ROOT.resolve(),
        _CATALOG_IMG_CACHE_ROOT.resolve(),
        _CATALOG_CLIP_ROOT.resolve(),
        _CLIP_EMBEDDINGS_ROOT.resolve(),
    ]
    requested = Path(str(path or "").strip())
    if not requested.is_absolute():
        requested = Path("/Users/olly/aim2build-instruction") / requested
    try:
        resolved = requested.resolve()
    except Exception:
        raise HTTPException(status_code=404, detail="not found")
    if not any(
        allowed_root in resolved.parents or resolved == allowed_root
        for allowed_root in allowed
    ):
        raise HTTPException(status_code=404, detail="not found")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(resolved))


@router.get("/debug/shape-overlay-diagnostic")
def shape_overlay_diagnostic(
    stem: str = Query(...),
    part_num: str = Query(...),
    color_id: str = Query(...),
    size: int = Query(256, ge=64, le=512),
):
    """
    In-memory PNG: query silhouette (green) vs catalog silhouette (blue), both normalised.
    No files written to disk.
    """
    from fastapi.responses import Response as _RawResp

    pack_dir  = _TRAINING_PACK_ROOT / stem
    mask_path = pack_dir / "mask.png"
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail=f"mask.png not found: {stem}")

    q_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    feat_q = _shape_features(q_mask)
    cnt_q  = feat_q.get("contour")

    cnt_c = None
    try:
        _, cat_items_local = _cat_emb_load()
        for it in cat_items_local:
            if (str(it.get("part_num", "")) == str(part_num)
                    and str(it.get("color_id", "")) == str(color_id)):
                img_p = it.get("img_path", "")
                if img_p and Path(img_p).exists():
                    cat_img = cv2.imread(img_p, cv2.IMREAD_UNCHANGED)
                    if cat_img is not None:
                        cat_mask = _catalog_mask_from_image(cat_img)
                        feat_c   = _shape_features(cat_mask)
                        cnt_c    = feat_c.get("contour")
                break
    except Exception:
        pass

    sz     = max(64, min(size, 512))
    canvas = np.full((sz, sz, 3), 245, dtype=np.uint8)

    def _draw_norm(cnv: np.ndarray, cnt, color: tuple, thickness: int = 2) -> None:
        if cnt is None or len(cnt) < 3:
            return
        pts = cnt.reshape(-1, 2).astype(np.float32)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        rng = mx - mn
        rng[rng < 1] = 1
        margin = sz * 0.08
        scale  = (sz - 2 * margin) / rng.max()
        pts    = (pts - mn) * scale + margin
        pts    = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(cnv, [pts], -1, color, thickness, cv2.LINE_AA)

    _draw_norm(canvas, cnt_q, (30, 160, 30))   # green = query
    _draw_norm(canvas, cnt_c, (30,  30, 200))   # blue  = catalog
    lbl_y = sz - 8
    cv2.putText(canvas, "Q=query",   (6, lbl_y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 160, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "C=catalog", (6, lbl_y),       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30,  30, 200), 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG encode failed")
    return _RawResp(content=buf.tobytes(), media_type="image/png")


@router.get("/debug/catalog-match-test")
def catalog_match_test(
    stem: str = Query(""),
    alpha: float = Query(0.6, ge=0.0, le=1.0),
) -> HTMLResponse:
    """
    Visual nearest-neighbour test: training-pack crop → top-20 catalog matches.

    GET /debug/catalog-match-test            → list all available training-pack stems
    GET /debug/catalog-match-test?stem=...   → run match and show results
    """
    from html import escape as _esc

    # ── No stem → show pack list ───────────────────────────────────────────────
    if not stem:
        stems = sorted(
            d.name for d in _TRAINING_PACK_ROOT.iterdir()
            if d.is_dir() and (d / "masked.png").exists()
        )
        links = "".join(
            f'<li><a href="/debug/catalog-match-test?stem={_url_quote(s)}">{_esc(s)}</a></li>'
            for s in stems
        )
        return HTMLResponse(
            f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Catalog Match Test</title>"
            f"<style>body{{font-family:system-ui,sans-serif;margin:24px}}"
            f"a{{color:#2a6bcc;text-decoration:none}}a:hover{{text-decoration:underline}}"
            f"ul{{columns:3;column-gap:32px;list-style:none;padding:0}}"
            f"li{{margin:3px 0;font-size:13px}}</style></head><body>"
            f"<h2>Catalog Match Test — {len(stems)} training packs</h2>"
            f"<p>Click a pack to run top-20 catalog matching.</p>"
            f"<ul>{links}</ul></body></html>"
        )

    # ── Validate stem ──────────────────────────────────────────────────────────
    pack_dir = _TRAINING_PACK_ROOT / stem
    if not pack_dir.exists():
        raise HTTPException(status_code=404, detail=f"Training pack not found: {stem}")

    masked_path   = pack_dir / "masked.png"
    original_path = pack_dir / "original.png"
    overlay_path  = pack_dir / "overlay.png"
    mask_path     = pack_dir / "mask.png"
    meta_path     = pack_dir / "meta.json"

    if not masked_path.exists():
        raise HTTPException(status_code=404, detail=f"masked.png not found in {stem}")

    # ── Load catalog embeddings ────────────────────────────────────────────────
    try:
        cat_matrix, cat_items = _cat_emb_load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    clip_ok, clip_err = _clip_load()
    if not clip_ok:
        raise HTTPException(status_code=503,
            detail=f"CLIP unavailable: {clip_err}")

    # ── Embed query ───────────────────────────────────────────────────────────
    try:
        import numpy as np
        query_vec = _clip_embed_image(str(masked_path))   # (512,)
    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"Failed to embed query image: {exc}")

    # ── CLIP top-20 ───────────────────────────────────────────────────────────
    scores = (cat_matrix @ query_vec).tolist()   # (N,)
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:20]
    top20 = [
        {
            "clip_rank":  rank,
            "clip_score": score,
            "cat_idx":    cat_idx,
            "cat_item":   cat_items[cat_idx],
        }
        for rank, (cat_idx, score) in enumerate(ranked, 1)
    ]

    # ── Shape rerank ──────────────────────────────────────────────────────────
    if mask_path.exists():
        reranked = _shape_rerank(str(mask_path), top20, alpha=alpha)
    else:
        reranked = [
            {**it, "shape_score": 0.0, "final_score": it["clip_score"], "shape_feat_c": {}}
            for it in top20
        ]

    # ── Load pack meta ────────────────────────────────────────────────────────
    pack_meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            pack_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    element_id = pack_meta.get("element_id", stem.replace("elem_", "").split("_p")[0])
    page       = pack_meta.get("page", "?")
    method     = pack_meta.get("segmentation_method", "?")
    cov        = pack_meta.get("coverage_pct", 0)
    iou_val    = pack_meta.get("iou")
    iou_str    = f"{iou_val:.3f}" if iou_val is not None else "—"
    qty        = pack_meta.get("qty", "?")

    # ── Query shape features (for left-panel display) ─────────────────────────
    q_shape: Dict[str, Any] = {}
    if mask_path.exists():
        try:
            q_mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            q_shape = _shape_features(q_mask_arr)
        except Exception:
            pass

    # ── Build nav (prev / next stem) ──────────────────────────────────────────
    all_stems = sorted(
        d.name for d in _TRAINING_PACK_ROOT.iterdir()
        if d.is_dir() and (d / "masked.png").exists()
    )
    try:
        idx     = all_stems.index(stem)
        prev_s  = all_stems[idx - 1] if idx > 0          else all_stems[-1]
        next_s  = all_stems[idx + 1] if idx < len(all_stems) - 1 else all_stems[0]
    except ValueError:
        prev_s = next_s = stem

    def _img(path: Path, w: int = 120) -> str:
        if path.exists():
            url = _match_file_url(str(path))
            return (f'<img src="{url}" loading="lazy" '
                    f'style="width:{w}px;height:{w}px;object-fit:contain;'
                    f'border:1px solid #ccc;border-radius:4px;background:#fff;">')
        return f'<div style="width:{w}px;height:{w}px;background:#eee;border-radius:4px;"></div>'

    def _score_bar(score: float, color: str, label: str, pct_str: str) -> str:
        bw = max(3, int(score * 100))
        return (
            f'<div style="display:flex;align-items:center;gap:4px;margin-top:2px;">'
            f'<span style="font-size:9px;color:#888;width:34px;text-align:right;">{label}</span>'
            f'<div style="flex:1;background:#eee;border-radius:2px;height:4px;">'
            f'<div style="width:{bw}%;height:4px;background:{color};border-radius:2px;"></div>'
            f'</div>'
            f'<span style="font-size:10px;font-weight:600;color:{color};width:38px;">{pct_str}</span>'
            f'</div>'
        )

    # Left panel — query crop
    ar_str  = f"{q_shape.get('aspect_ratio', 0):.2f}" if q_shape else "—"
    sol_str = f"{q_shape.get('solidity', 0):.2f}"     if q_shape else "—"
    ext_str = f"{q_shape.get('extent', 0):.2f}"       if q_shape else "—"
    mask_warn = (
        '<div style="font-size:10px;color:#b00020;margin-top:4px;">⚠ mask.png missing — shape=0</div>'
        if not mask_path.exists() else ""
    )
    alpha_links = (
        f'<a href="/debug/catalog-match-test?stem={_url_quote(stem)}&alpha=1.0" style="font-size:10px;">pure CLIP</a>'
        f' &nbsp;|&nbsp; '
        f'<a href="/debug/catalog-match-test?stem={_url_quote(stem)}&alpha=0.0" style="font-size:10px;">pure shape</a>'
        f' &nbsp;|&nbsp; '
        f'<a href="/debug/catalog-match-test?stem={_url_quote(stem)}&alpha=0.6" style="font-size:10px;">α=0.6</a>'
    )
    left_html = (
        f'<div style="min-width:200px;max-width:220px;">'
        f'<div style="font-size:13px;font-weight:700;margin-bottom:8px;color:#1a2a3a;">'
        f'  element {_esc(str(element_id))} &nbsp;<span style="font-weight:400;color:#778;">p{_esc(str(page))}</span>'
        f'</div>'
        f'<div style="font-size:11px;color:#556677;margin-bottom:4px;">'
        f'  method: {_esc(method)} &nbsp;|&nbsp; cov: {cov:.1f}%<br>'
        f'  iou: {iou_str} &nbsp;|&nbsp; qty: {_esc(str(qty))}'
        f'</div>'
        f'<div style="font-size:11px;color:#556677;margin-bottom:4px;">'
        f'  shape: ar={ar_str} &nbsp; sol={sol_str} &nbsp; ext={ext_str}'
        f'</div>'
        f'<div style="font-size:11px;color:#334;margin-bottom:6px;">'
        f'  α = <strong>{alpha:.2f}</strong> &nbsp; {alpha_links}'
        f'</div>'
        + mask_warn
        + f'<div style="font-size:11px;color:#aaa;margin:6px 0 4px;">original</div>'
        + _img(original_path, 160)
        + f'<div style="font-size:11px;color:#aaa;margin:8px 0 4px;">masked (query)</div>'
        + _img(masked_path, 160)
        + f'<div style="font-size:11px;color:#aaa;margin:8px 0 4px;">overlay</div>'
        + _img(overlay_path, 160)
        + f'<div style="margin-top:16px;font-size:11px;">'
        f'<a href="/debug/catalog-match-test?stem={_url_quote(prev_s)}&alpha={alpha}">◀ prev</a>'
        f'&nbsp;&nbsp;'
        f'<a href="/debug/catalog-match-test">list</a>'
        f'&nbsp;&nbsp;'
        f'<a href="/debug/catalog-match-test?stem={_url_quote(next_s)}&alpha={alpha}">next ▶</a>'
        f'</div>'
        f'</div>'
    )

    # Right panel — reranked match cards
    match_cards = ""
    for final_rank, item in enumerate(reranked, 1):
        clip_rank   = item["clip_rank"]
        clip_score  = item["clip_score"]
        shape_score = item["shape_score"]
        final_score = item["final_score"]
        cat_item    = item["cat_item"]
        part_num    = cat_item.get("part_num", "?")
        color_id    = cat_item.get("color_id", "?")
        img_p       = Path(cat_item.get("img_path", ""))

        rank_moved = clip_rank != final_rank
        clip_col  = "#2a7a2a" if clip_score  >= 0.85 else "#7a6a14" if clip_score  >= 0.70 else "#556677"
        shape_col = "#2a7a2a" if shape_score >= 0.70 else "#7a6a14" if shape_score >= 0.50 else "#556677"
        final_col = "#2a7a2a" if final_score >= 0.80 else "#7a6a14" if final_score >= 0.65 else "#556677"
        rank_badge = (
            f'<span style="font-size:9px;color:#b07000;font-weight:700;" title="CLIP rank was #{clip_rank}">↕CLIP#{clip_rank}</span>'
            if rank_moved else
            f'<span style="font-size:9px;color:#aaa;">CLIP#{clip_rank}</span>'
        )
        overlay_url = (
            f'/debug/shape-overlay-diagnostic?stem={_url_quote(stem)}'
            f'&part_num={_url_quote(str(part_num))}&color_id={_url_quote(str(color_id))}'
        )
        match_cards += (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'background:#fff;border:1px solid #dde3ec;border-radius:6px;'
            f'padding:8px;width:155px;box-shadow:0 1px 3px rgba(0,0,0,.06);">'
            f'<div style="display:flex;justify-content:space-between;width:100%;margin-bottom:2px;">'
            f'<span style="font-size:11px;font-weight:700;color:#1a2a3a;">#{final_rank}</span>'
            f'{rank_badge}'
            f'</div>'
            + _img(img_p, 110)
            + f'<div style="font-size:12px;font-weight:600;margin-top:6px;color:#1a2a3a;">'
            f'{_esc(str(part_num))}</div>'
            f'<div style="font-size:10px;color:#778;margin-bottom:4px;">color {_esc(str(color_id))}</div>'
            + _score_bar(clip_score,  clip_col,  "CLIP",  f"{clip_score  * 100:.1f}%")
            + _score_bar(shape_score, shape_col, "shape", f"{shape_score * 100:.1f}%")
            + _score_bar(final_score, final_col, "final", f"{final_score * 100:.1f}%")
            + f'<div style="margin-top:6px;">'
            f'<a href="{overlay_url}" target="_blank" style="font-size:9px;color:#2a6bcc;">overlay ↗</a>'
            f'</div>'
            + f'<div style="display:flex;gap:6px;margin-top:6px;">'
            f'<button onclick=\'cmtConfirm(this,{json.dumps(stem)},{json.dumps(str(part_num))},{int(color_id) if str(color_id).lstrip("-").isdigit() else 0},{json.dumps(str(element_id))},{json.dumps(pack_meta.get("set_num",""))},{json.dumps(str(masked_path))})\' '
            f'style="flex:1;font-size:10px;padding:3px 0;background:#d4edda;border:1px solid #7cbb8a;border-radius:3px;cursor:pointer;color:#1a4a24;">✓ Confirm</button>'
            f'<button onclick=\'cmtReject(this,{json.dumps(stem)},{json.dumps(str(part_num))},{int(color_id) if str(color_id).lstrip("-").isdigit() else 0})\' '
            f'style="flex:1;font-size:10px;padding:3px 0;background:#f8d7da;border:1px solid #c48a8f;border-radius:3px;cursor:pointer;color:#4a1a1e;">✗ Reject</button>'
            f'</div>'
            f'</div>'
        )

    right_html = (
        f'<div>'
        f'<div style="font-size:13px;font-weight:700;color:#1a2a3a;margin-bottom:12px;">'
        f'Top 20 matches &mdash; sorted by final score (α={alpha:.2f})</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;">'
        + match_cards
        + f'</div></div>'
    )

    html = (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Catalog Match — {_esc(stem)}</title>"
        f"<style>"
        f"body{{font-family:system-ui,sans-serif;margin:0;background:#f0f2f6;color:#1a2a3a}}"
        f"h2{{font-size:17px;margin:0 0 16px}}"
        f"a{{color:#2a6bcc;text-decoration:none}}a:hover{{text-decoration:underline}}"
        f"</style>"
        f"<script>"
        f"async function cmtConfirm(btn,stem,partNum,colorId,elementId,setNum,maskedPath){{"
        f"  btn.disabled=true;btn.textContent='…';"
        f"  const payload={{crop_id:stem,set_num:setNum||'catalog_test',bag:0,"
        f"    part_num:partNum,color_id:colorId,element_id:elementId||null,"
        f"    ai_snap_input_path:maskedPath,"
        f"    adjustments:[{{type:'catalog_match_confirm'}}]}};"
        f"  const r=await fetch('/debug/save-label',{{method:'POST',"
        f"    headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});"
        f"  if(r.ok){{btn.textContent='✓';btn.style.background='#28a745';btn.style.color='#fff';}}"
        f"  else{{btn.disabled=false;btn.textContent='err';btn.style.background='#dc3545';btn.style.color='#fff';}}"
        f"}}"
        f"async function cmtReject(btn,stem,partNum,colorId){{"
        f"  btn.disabled=true;btn.textContent='…';"
        f"  const r=await fetch('/debug/catalog-match-feedback',{{method:'POST',"
        f"    headers:{{'Content-Type':'application/json'}},"
        f"    body:JSON.stringify({{stem:stem,part_num:partNum,color_id:colorId,feedback:'reject',rejected_by:'andy'}})}});"
        f"  if(r.ok){{btn.textContent='✗';btn.style.background='#6c757d';btn.style.color='#fff';}}"
        f"  else{{btn.disabled=false;btn.textContent='err';btn.style.background='#dc3545';btn.style.color='#fff';}}"
        f"}}"
        f"</script>"
        f"</head><body>"
        f"<div style='padding:20px 24px;'>"
        f"<h2>Catalog Match Test — {_esc(stem)}</h2>"
        f"<div style='display:flex;gap:32px;align-items:flex-start;'>"
        + left_html
        + right_html
        + f"</div></div></body></html>"
    )
    return HTMLResponse(content=html)


@router.post("/debug/catalog-match-feedback")
async def catalog_match_feedback(req: Request):
    """Record a reject signal for a catalog-match-test candidate.

    Writes to debug/catalog_match_feedback/{stem}.json — separate from
    training labels; no effect on ranking or save-label.
    """
    data = await req.json()
    stem = str(data.get("stem") or "").strip()
    part_num = str(data.get("part_num") or "").strip()
    color_id = int(data.get("color_id") or 0)
    feedback = str(data.get("feedback") or "reject").strip()
    rejected_by = str(data.get("rejected_by") or "").strip()
    if not stem or not part_num:
        raise HTTPException(status_code=400, detail="stem and part_num are required")
    _CATALOG_MATCH_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    path = _CATALOG_MATCH_FEEDBACK_DIR / f"{stem}.json"
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        existing = {}
    rejects = list(existing.get("rejects") or [])
    rejects.append({
        "part_num": part_num,
        "color_id": color_id,
        "feedback": feedback,
        "rejected_by": rejected_by,
        "rejected_at": _iso_now(),
    })
    existing["stem"] = stem
    existing["rejects"] = rejects
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"ok": True, "stem": stem, "rejects": len(rejects)}


@router.post("/debug/training-store/upload-r2")
def training_store_upload_r2(
    bundle_id: str = Query(...),
    dry_run: str = Query("1"),
):
    try:
        result = upload_bundle_to_r2(bundle_id, dry_run=_dry_run_enabled(dry_run))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _mark_quick_review_cache_stale_for_bundle(bundle_id, "candidate_review_state_changed")
    return JSONResponse(result)


@router.post("/debug/training-store/upload-r2-batch")
def training_store_upload_r2_batch(
    set_num: str = Query("70618"),
    bag_num: int = Query(1),
    dry_run: str = Query("1"),
):
    set_text = str(set_num or "70618").strip() or "70618"
    bag_number = int(bag_num or 1)
    dry_run_enabled = _dry_run_enabled(dry_run)
    listed = list_registered_bundles(
        set_num=set_text,
        bag=bag_number,
        review_status="approved",
    )
    entries = [
        dict(item)
        for item in list(listed.get("entries") or [])
        if isinstance(item, dict) and str(item.get("bundle_id") or "").strip()
    ]
    results: List[Dict[str, Any]] = []
    success_count = 0
    failed_count = 0
    for entry in entries:
        bundle_id = str(entry.get("bundle_id") or "").strip()
        try:
            upload_result = upload_bundle_to_r2(bundle_id, dry_run=dry_run_enabled)
            ok = bool(upload_result.get("ok"))
            if ok:
                success_count += 1
            else:
                failed_count += 1
            results.append(
                {
                    "bundle_id": bundle_id,
                    "ok": ok,
                    "dry_run": dry_run_enabled,
                    "uploaded": bool(upload_result.get("uploaded")),
                    "would_upload": bool(upload_result.get("would_upload")),
                    "error": str(upload_result.get("error") or ""),
                    "r2_status": str(upload_result.get("r2_status") or entry.get("r2_status") or ""),
                    "azure_postgres_registration": upload_result.get("azure_postgres_registration"),
                }
            )
        except Exception as exc:
            failed_count += 1
            results.append(
                {
                    "bundle_id": bundle_id,
                    "ok": False,
                    "dry_run": dry_run_enabled,
                    "uploaded": False,
                    "would_upload": False,
                    "error": type(exc).__name__,
                }
            )

    return JSONResponse(
        {
            "ok": failed_count == 0,
            "set_num": set_text,
            "bag_num": bag_number,
            "dry_run": dry_run_enabled,
            "approved_bundle_count": len(entries),
            "success_count": success_count,
            "failed_count": failed_count,
            "bundle_ids": [str(item.get("bundle_id") or "") for item in entries],
            "results": results,
        }
    )


@router.get("/debug/training-store/prepare-azure-index")
def training_store_prepare_azure_index(bundle_id: str = Query(...)):
    try:
        result = prepare_bundle_for_azure(bundle_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result)


@router.get("/debug/training-store/bundle-index-row")
def training_store_bundle_index_row(bundle_id: str = Query(...)):
    try:
        result = get_training_bundle_index_row(bundle_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result)


def _training_bundle_artifact_debug(bundle_id: str) -> Dict[str, Any]:
    safe_bundle_id = str(bundle_id or "").strip()
    if not safe_bundle_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", safe_bundle_id):
        raise ValueError("invalid bundle_id")
    analysis_root = (
        Path("/Users/olly/aim2build-instruction")
        / "debug"
        / "ai_training"
        / "analysis_bundles"
    ).resolve()
    local_folder = (analysis_root / safe_bundle_id).resolve()
    if analysis_root not in local_folder.parents:
        raise ValueError("bundle path escapes analysis bundle directory")
    metadata_path = local_folder / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists() and metadata_path.is_file():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = loaded if isinstance(loaded, dict) else {}
        except Exception:
            metadata = {}
    copied_files = metadata.get("copied_files") if isinstance(metadata.get("copied_files"), dict) else {}

    def _path_status(path_value: Any) -> Dict[str, Any]:
        path_text = str(path_value or "").strip()
        path = Path(path_text) if path_text else Path()
        return {
            "path": path_text,
            "exists": bool(path_text and path.exists() and path.is_file()),
        }

    local_files_present = (
        sorted(item.name for item in local_folder.iterdir() if item.is_file())
        if local_folder.exists() and local_folder.is_dir()
        else []
    )
    return {
        "bundle_id": safe_bundle_id,
        "local_folder": str(local_folder),
        "local_folder_exists": local_folder.exists() and local_folder.is_dir(),
        "local_files_present": local_files_present,
        "metadata_path": str(metadata_path),
        "metadata_exists": metadata_path.exists() and metadata_path.is_file(),
        "original_crop": _path_status(copied_files.get("original_crop") or (local_folder / "original_crop.png")),
        "full_mask_overlay": _path_status(copied_files.get("full_mask_overlay") or (local_folder / "full_mask_overlay.png")),
        "raw_master_mask": _path_status(copied_files.get("raw_master_mask") or (local_folder / "raw_master_mask.png")),
        "master_island_overlay": _path_status(copied_files.get("master_island_overlay") or (local_folder / "master_island_overlay.png")),
        "slot_cutouts": [
            _path_status(path_value)
            for path_value in list(copied_files.get("slot_cutouts") or [])
        ],
    }


@router.get("/debug/training-store/bundle-debug")
def training_store_bundle_debug(bundle_id: str = Query(...)):
    try:
        artifact_debug = _training_bundle_artifact_debug(bundle_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    postgres_row: Dict[str, Any] = {}
    postgres_error = ""
    postgres_row_found = False
    try:
        postgres_row = dict(get_training_bundle_index_row(bundle_id).get("row") or {})
        postgres_row_found = bool(postgres_row)
    except FileNotFoundError as exc:
        postgres_error = str(exc)
    except Exception as exc:
        postgres_error = type(exc).__name__
    return JSONResponse(
        {
            "ok": True,
            "bundle_id": str(bundle_id or "").strip(),
            "postgres_row_found": postgres_row_found,
            "postgres_row": postgres_row,
            "postgres_error": postgres_error,
            **artifact_debug,
            "original_crop_exists": bool((artifact_debug.get("original_crop") or {}).get("exists")),
            "overlay_exists": bool((artifact_debug.get("full_mask_overlay") or {}).get("exists")),
            "raw_master_mask_exists": bool((artifact_debug.get("raw_master_mask") or {}).get("exists")),
        }
    )


@router.get("/debug/training-store/review-queue")
def training_store_review_queue(
    review_status: str = Query(""),
    set_num: str = Query(""),
    bag_num: str = Query(""),
    limit: int = Query(100),
):
    try:
        result = list_review_queue(
            review_status=review_status,
            set_num=set_num,
            bag_num=bag_num,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result)


@router.post("/debug/training-store/review")
async def training_store_review(req: Request):
    payload: Dict[str, Any] = {}
    try:
        body = await req.json()
        if isinstance(body, dict):
            payload = dict(body)
    except Exception:
        payload = {}
    try:
        result = update_training_bundle_review(
            str(payload.get("bundle_id") or ""),
            review_status=str(payload.get("review_status") or ""),
            review_notes=str(payload.get("review_notes") or ""),
            mask_quality=payload.get("mask_quality"),
            split_quality=payload.get("split_quality"),
            qty_text_present=bool(payload.get("qty_text_present")),
            multi_part_merge=bool(payload.get("multi_part_merge")),
            reviewed_by=str(payload.get("reviewed_by") or ""),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _mark_quick_review_cache_stale_for_bundle(str(payload.get("bundle_id") or ""), "review_saved")
    return JSONResponse(result)


@router.get("/debug/training-store/review-stats")
def training_store_review_stats():
    try:
        result = get_review_stats()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result)


@router.get("/debug/training-store/confirmed-part-usage")
def training_store_confirmed_part_usage(
    set_num: str = Query(...),
    part_num: str = Query(...),
    color_id: int = Query(...),
    element_id: str = Query(""),
    required_qty: str = Query(""),
):
    try:
        result = list_confirmed_part_usage(
            set_num=set_num,
            part_num=part_num,
            color_id=color_id,
            element_id=element_id,
            required_qty=required_qty,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result)


@router.post("/debug/training-store/analyse-bundle")
def training_store_analyse_bundle(bundle_id: str = Query(...)):
    try:
        result = analyse_reviewed_bundle(bundle_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result)


@router.post("/debug/training-store/analyze-bundle")
def training_store_analyze_bundle(bundle_id: str = Query(...)):
    return training_store_analyse_bundle(bundle_id)


@router.post("/debug/training-store/generate-split-candidates")
def training_store_generate_split_candidates(bundle_id: str = Query(...)):
    try:
        result = generate_split_candidates(bundle_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result)


@router.post("/debug/training-store/accept-split-candidate")
def training_store_accept_split_candidate(
    bundle_id: str = Query(...),
    candidate_index: int = Query(...),
):
    try:
        result = mark_split_candidate(bundle_id, candidate_index, "accepted")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    stale_reason = "bundle_promoted_to_approved" if bool(result.get("bundle_promoted")) else "candidate_accepted"
    _mark_quick_review_cache_stale_for_bundle(bundle_id, stale_reason)
    return JSONResponse(result)


@router.post("/debug/training-store/reject-split-candidate")
def training_store_reject_split_candidate(
    bundle_id: str = Query(...),
    candidate_index: int = Query(...),
):
    try:
        result = mark_split_candidate(bundle_id, candidate_index, "rejected")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    stale_reason = "bundle_promoted_to_approved" if bool(result.get("bundle_promoted")) else "candidate_rejected"
    _mark_quick_review_cache_stale_for_bundle(bundle_id, stale_reason)
    return JSONResponse(result)


@router.post("/debug/training-store/set-candidate-review-state")
def training_store_set_candidate_review_state(
    bundle_id: str = Query(...),
    candidate_index: int = Query(...),
    review_state: str = Query(""),
):
    try:
        result = set_split_candidate_review_state(bundle_id, candidate_index, review_state)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result)


@router.post("/debug/training-store/scrub-candidate-qty")
def training_store_scrub_candidate_qty(
    bundle_id: str = Query(...),
    candidate_index: int = Query(...),
):
    try:
        result = scrub_candidate_qty(bundle_id, candidate_index)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _mark_quick_review_cache_stale_for_bundle(bundle_id, "candidate_qty_scrubbed")
    return JSONResponse(result)


def _decode_png_data_url(data_url: Any) -> np.ndarray:
    text = str(data_url or "").strip()
    if "," in text:
        text = text.split(",", 1)[1]
    if not text:
        return np.zeros((1, 1), dtype=np.uint8)
    raw = base64.b64decode(text)
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None or getattr(image, "size", 0) == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if image.ndim == 3 and image.shape[2] >= 4:
        return image[:, :, 3]
    if image.ndim == 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    return image


def _update_candidate_in_split_paths(paths: Dict[str, Any], candidate_index: int, updated_candidate: Dict[str, Any]) -> Dict[str, Any]:
    updated_paths = dict(paths or {})

    def update_items(raw_items: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for pos, raw_item in enumerate(list(raw_items or [])):
            item = dict(raw_item) if isinstance(raw_item, dict) else {}
            item_index = _coerce_int(item.get("index"))
            if item_index is None:
                item_index = pos
            if int(item_index) == int(candidate_index):
                item.update(updated_candidate)
            out.append(item)
        return out

    updated_paths["candidates"] = update_items(updated_paths.get("candidates"))
    updated_paths["baseline_slot_candidates"] = update_items(updated_paths.get("baseline_slot_candidates"))
    updated_paths["ai_suggested_candidates"] = update_items(updated_paths.get("ai_suggested_candidates"))
    return updated_paths


def _candidate_for_index_from_row(row: Dict[str, Any], candidate_index: int) -> Dict[str, Any]:
    paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
    candidates = [
        dict(item)
        for item in list(paths.get("candidates") or [])
        if isinstance(item, dict)
    ]
    for pos, candidate in enumerate(candidates):
        item_index = _coerce_int(candidate.get("index"))
        if item_index is None:
            item_index = pos
        if int(item_index) == int(candidate_index):
            return candidate
    raise ValueError("candidate_index is out of range")


def _alpha_debug_for_png(path_value: Any) -> Dict[str, Any]:
    path = Path(str(path_value or "").strip())
    if not path.exists() or not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "has_alpha": False,
            "transparent_pixel_count": 0,
            "opaque_pixel_count": 0,
            "alpha_bbox": None,
            "rgb_under_transparent_summary": {},
            "likely_real_background_leak": False,
        }
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError(f"image is unreadable: {path}")
    has_alpha = bool(image.ndim == 3 and image.shape[2] >= 4)
    if has_alpha:
        alpha = image[:, :, 3]
        rgb = image[:, :, :3]
    else:
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        rgb = image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    transparent_mask = alpha == 0
    opaque_mask = alpha > 0
    transparent_pixel_count = int(np.count_nonzero(transparent_mask))
    opaque_pixel_count = int(np.count_nonzero(opaque_mask))
    alpha_bbox = None
    nz = cv2.findNonZero(opaque_mask.astype(np.uint8))
    if nz is not None:
        x, y, w, h = cv2.boundingRect(nz)
        alpha_bbox = [int(x), int(y), int(w), int(h)]
    rgb_under_transparent_summary: Dict[str, Any] = {
        "sample_count": 0,
        "nonzero_rgb_count": 0,
        "max_rgb": [0, 0, 0],
        "mean_rgb": [0.0, 0.0, 0.0],
        "sample_rgb_values": [],
    }
    likely_real_background_leak = False
    if transparent_pixel_count > 0:
        transparent_rgb = rgb[transparent_mask]
        nonzero_rgb = np.any(transparent_rgb > 0, axis=1)
        nonzero_count = int(np.count_nonzero(nonzero_rgb))
        sample_values = transparent_rgb[nonzero_rgb][:10].tolist() if nonzero_count else transparent_rgb[:10].tolist()
        rgb_under_transparent_summary = {
            "sample_count": int(transparent_rgb.shape[0]),
            "nonzero_rgb_count": nonzero_count,
            "max_rgb": [int(v) for v in np.max(transparent_rgb, axis=0).tolist()],
            "mean_rgb": [round(float(v), 3) for v in np.mean(transparent_rgb, axis=0).tolist()],
            "sample_rgb_values": [[int(channel) for channel in value] for value in sample_values],
        }
        likely_real_background_leak = nonzero_count > 0
    return {
        "path": str(path),
        "exists": True,
        "shape": [int(value) for value in image.shape],
        "has_alpha": has_alpha,
        "transparent_pixel_count": transparent_pixel_count,
        "opaque_pixel_count": opaque_pixel_count,
        "alpha_bbox": alpha_bbox,
        "rgb_under_transparent_summary": rgb_under_transparent_summary,
        "likely_real_background_leak": likely_real_background_leak,
    }


def _manual_mask_foreground_refinement(original: np.ndarray, amended_mask: np.ndarray) -> Dict[str, np.ndarray]:
    _, constrained_mask = cv2.threshold(amended_mask, 127, 255, cv2.THRESH_BINARY)
    cleanup_kernel = np.ones((3, 3), dtype=np.uint8)
    constrained_mask = cv2.morphologyEx(constrained_mask, cv2.MORPH_CLOSE, cleanup_kernel)
    constrained_mask = cv2.morphologyEx(constrained_mask, cv2.MORPH_OPEN, cleanup_kernel)
    _, constrained_mask = cv2.threshold(constrained_mask, 127, 255, cv2.THRESH_BINARY)
    if int(np.count_nonzero(constrained_mask)) == 0:
        raise ValueError("amended mask is empty")

    sure_kernel = np.ones((5, 5), dtype=np.uint8)
    sure_foreground = cv2.erode(constrained_mask, sure_kernel, iterations=1)
    if int(np.count_nonzero(sure_foreground)) == 0:
        sure_foreground = constrained_mask.copy()

    probable_background = cv2.dilate(constrained_mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
    grabcut_mask = np.full(constrained_mask.shape, cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[probable_background > 0] = cv2.GC_PR_BGD
    grabcut_mask[constrained_mask > 0] = cv2.GC_PR_FGD
    grabcut_mask[sure_foreground > 0] = cv2.GC_FGD

    refined_mask = constrained_mask.copy()
    try:
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(original, grabcut_mask, None, bg_model, fg_model, 4, cv2.GC_INIT_WITH_MASK)
        refined_mask = np.where(
            ((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)) & (constrained_mask > 0),
            255,
            0,
        ).astype(np.uint8)
        if int(np.count_nonzero(refined_mask)) == 0:
            refined_mask = constrained_mask.copy()
    except Exception:
        refined_mask = constrained_mask.copy()

    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, cleanup_kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, cleanup_kernel)
    _, refined_mask = cv2.threshold(refined_mask, 127, 255, cv2.THRESH_BINARY)
    if int(np.count_nonzero(refined_mask)) == 0:
        refined_mask = constrained_mask.copy()
    refined_alpha = cv2.GaussianBlur(refined_mask, (3, 3), 0)
    refined_alpha[refined_alpha < 16] = 0

    trimap = np.zeros((*constrained_mask.shape, 3), dtype=np.uint8)
    trimap[probable_background > 0] = [42, 42, 120]
    trimap[constrained_mask > 0] = [0, 190, 220]
    trimap[sure_foreground > 0] = [0, 190, 70]
    return {
        "constraint_mask": constrained_mask,
        "refined_mask": refined_mask,
        "refined_alpha": refined_alpha,
        "trimap": trimap,
    }


def _manual_mask_output_path(
    candidate: Dict[str, Any],
    key: str,
    bundle_dir: Path,
    candidate_index: int,
    suffix: str,
) -> Path:
    existing = str(candidate.get(key) or "").strip()
    return Path(existing) if existing else bundle_dir / f"candidate_{int(candidate_index)}_{suffix}"


def _remove_background_fringe_from_manual_mask(
    original: np.ndarray,
    alpha_mask: np.ndarray,
    candidate_index: int,
    bundle_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Two-pass background fringe removal for manually amended candidates.

    Pass 1 — per-pixel colour removal:
      Remove opaque pixels that are colour-close to the estimated background AND
      do NOT sit on a strong local edge/texture. Edge protection uses Sobel magnitude
      at the individual pixel (no dilation), so only pixels WITH a strong gradient
      are kept — surrounding soft-edge halos are NOT protected.

    Pass 2 — connected boundary fringe cleanup:
      Any connected cluster of background-coloured pixels that still touches the
      alpha boundary after pass 1 is removed, provided the cluster is small relative
      to the total alpha area. This catches fringe rings that survive pass 1 because
      they individually pass the colour test by a tiny margin.

    Generic — background colour is estimated from the image border pixels.
    Does NOT hardcode any colour.

    Returns (cleaned_alpha, debug_info_dict).
    """
    h, w = original.shape[:2]
    if h < 8 or w < 8 or int(np.count_nonzero(alpha_mask)) == 0:
        return alpha_mask.copy(), {
            "background_like_removed_pixels": 0,
            "pass1_removed": 0,
            "pass2_removed": 0,
            "background_model_rgb": None,
            "background_distance_threshold": None,
            "background_spread": None,
        }

    # ── 1. Estimate background colour from outer border pixels ─────────────────
    border_px = max(4, min(10, h // 8, w // 8))
    border_samples_bgr = np.concatenate(
        [
            original[:border_px, :].reshape(-1, 3),
            original[h - border_px:, :].reshape(-1, 3),
            original[:, :border_px].reshape(-1, 3),
            original[:, w - border_px:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.uint8)

    border_lab = cv2.cvtColor(
        border_samples_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    bg_lab = np.median(border_lab, axis=0)  # (L, a, b)
    sample_dists = np.linalg.norm(border_lab - bg_lab, axis=1)
    bg_spread = float(np.percentile(sample_dists, 85))

    # Aggressive threshold — lower minimum, smaller spread multiplier
    # min 18 LAB units; 1.5× spread (was 22 / 2.0 before)
    dist_threshold = max(18.0, bg_spread * 1.5)

    # ── 2. Per-pixel LAB distance from background ─────────────────────────────
    img_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = img_lab - bg_lab[np.newaxis, np.newaxis, :]
    pixel_dist = np.sqrt(np.sum(delta * delta, axis=2))  # (h, w)

    # ── 3. Strong local edge protection (Sobel, per-pixel, NO dilation) ───────
    # Only the pixel's own gradient determines protection — soft-edge halos
    # around background colour patches are NOT protected.
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sx * sx + sy * sy)
    strong_local_edge = gradient_mag > 25.0  # only protect actual part edges/studs

    # ── 4. Pass 1: per-pixel colour removal ───────────────────────────────────
    is_bg_like_p1: np.ndarray = (
        (pixel_dist < dist_threshold)
        & (~strong_local_edge)
        & (alpha_mask > 16)
    )
    pass1_removed = int(np.count_nonzero(is_bg_like_p1))
    cleaned = alpha_mask.copy()
    cleaned[is_bg_like_p1] = 0

    # ── 5. Pass 2: connected boundary fringe cleanup ──────────────────────────
    # Looser colour threshold for seeding clusters
    dist_threshold_loose = max(28.0, bg_spread * 2.5)
    total_alpha_pixels = int(np.count_nonzero(cleaned > 0))
    # Cap: never remove a component larger than 12.5% of remaining alpha
    max_component_px = max(60, total_alpha_pixels // 8)

    fringe_candidates = ((pixel_dist < dist_threshold_loose) & (cleaned > 0)).astype(np.uint8)
    pass2_removed = 0
    pass2_remove_mask = np.zeros((h, w), dtype=bool)
    if int(np.count_nonzero(fringe_candidates)) > 0:
        # Alpha boundary: opaque pixels adjacent to transparent pixels
        alpha_binary = (cleaned > 0).astype(np.uint8)
        dilated_bg = cv2.dilate(
            (alpha_binary == 0).astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        boundary_zone: np.ndarray = (alpha_binary > 0) & (dilated_bg > 0)

        n_labels, labels = cv2.connectedComponents(fringe_candidates, connectivity=8)
        # Find which labels touch the boundary zone
        boundary_labels: set = set(int(v) for v in np.unique(labels[boundary_zone]))
        boundary_labels.discard(0)

        for lbl in boundary_labels:
            comp_mask = labels == lbl
            comp_size = int(np.count_nonzero(comp_mask))
            if comp_size <= max_component_px:
                pass2_remove_mask |= comp_mask
                pass2_removed += comp_size

        cleaned[pass2_remove_mask] = 0

    total_removed = pass1_removed + pass2_removed

    # ── 6. Background model → RGB for reporting ───────────────────────────────
    bg_lab_u8 = np.clip(bg_lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    bg_bgr_u8 = cv2.cvtColor(bg_lab_u8, cv2.COLOR_LAB2BGR).reshape(3)
    bg_rgb = [int(bg_bgr_u8[2]), int(bg_bgr_u8[1]), int(bg_bgr_u8[0])]

    print(
        f"[manual-bg-fringe] candidate={candidate_index}"
        f" border_px={border_px} bg_rgb={bg_rgb}"
        f" bg_spread={bg_spread:.1f} dist_threshold={dist_threshold:.1f}"
        f" pass1={pass1_removed} pass2={pass2_removed} total={total_removed}"
    )

    # ── 7. Debug artifacts ────────────────────────────────────────────────────
    bundle_dir.mkdir(parents=True, exist_ok=True)
    cyan_bgr = np.array([200, 200, 0], dtype=np.float32)

    # a) Sample preview: border strip tinted cyan
    preview = original.copy().astype(np.float32)
    for slc in (
        np.s_[:border_px, :],
        np.s_[h - border_px:, :],
        np.s_[:, :border_px],
        np.s_[:, w - border_px:],
    ):
        preview[slc] = preview[slc] * 0.5 + cyan_bgr * 0.5
    preview_path = bundle_dir / f"candidate_{candidate_index}_edge_background_sample_preview.png"
    cv2.imwrite(str(preview_path), preview.clip(0, 255).astype(np.uint8))

    # b) Background likeness mask: bright = background-like; strong-edge pixels black
    likeness_vis = np.clip(
        255.0 * np.maximum(0.0, 1.0 - pixel_dist / max(dist_threshold, 1.0)), 0, 255
    ).astype(np.uint8)
    likeness_vis[strong_local_edge] = 0
    likeness_path = bundle_dir / f"candidate_{candidate_index}_background_likeness_mask.png"
    cv2.imwrite(str(likeness_path), likeness_vis)

    # c) Removed-pixels overlay: pass1 = red, pass2 = magenta
    minus_bg = original.copy().astype(np.float32)
    if pass1_removed > 0:
        minus_bg[is_bg_like_p1] = (
            minus_bg[is_bg_like_p1] * 0.3 + np.array([0.0, 0.0, 220.0]) * 0.7
        )
    if pass2_removed > 0:
        minus_bg[pass2_remove_mask] = (
            minus_bg[pass2_remove_mask] * 0.3 + np.array([200.0, 0.0, 200.0]) * 0.7
        )
    minus_bg_path = bundle_dir / f"candidate_{candidate_index}_refined_minus_background_mask.png"
    cv2.imwrite(str(minus_bg_path), minus_bg.clip(0, 255).astype(np.uint8))

    return cleaned, {
        "background_like_removed_pixels": total_removed,
        "pass1_removed": pass1_removed,
        "pass2_removed": pass2_removed,
        "background_model_rgb": bg_rgb,
        "background_distance_threshold": round(float(dist_threshold), 2),
        "background_spread": round(float(bg_spread), 2),
        "bg_fringe_preview_path": str(preview_path),
        "bg_likeness_mask_path": str(likeness_path),
        "bg_minus_mask_path": str(minus_bg_path),
    }


_CANDIDATE_QTY_CLEAN_STATES = {"auto_scrubbed", "not_needed", "manual_mark_clean", "not_detected", "scrubbed"}


def _candidate_qty_is_clean(candidate: Dict[str, Any]) -> bool:
    """Return True if the candidate's qty text is in a clean-enough state to accept."""
    if not bool(candidate.get("qty_detected")):
        return True
    if str(candidate.get("qty_text_state") or "") in _CANDIDATE_QTY_CLEAN_STATES:
        return True
    # Legacy: qty_scrubbed_path present
    if str(candidate.get("qty_scrubbed_path") or "").strip():
        return True
    return False


def _candidate_display_path(candidate: Dict[str, Any]) -> str:
    """Return the best current image path for display / tool input."""
    return str(
        candidate.get("current_candidate_path")
        or candidate.get("colour_cleanup_path")
        or candidate.get("manual_amended_candidate_path")
        or candidate.get("qty_scrubbed_path")
        or candidate.get("candidate_path")
        or ""
    )


def _write_manual_refinement_outputs(
    *,
    row: Dict[str, Any],
    candidate: Dict[str, Any],
    candidate_index: int,
    original: np.ndarray,
    amended_mask: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    height, width = original.shape[:2]
    refinement = _manual_mask_foreground_refinement(original, amended_mask)
    final_binary_mask = refinement["constraint_mask"]
    refined_mask = refinement["refined_mask"]
    final_alpha_mask = refinement["refined_alpha"]
    trimap = refinement["trimap"]
    nz = cv2.findNonZero((refined_mask > 0).astype(np.uint8))
    if nz is None:
        raise ValueError("amended mask is empty")
    bx, by, bw, bh = cv2.boundingRect(nz)
    pad = 6
    tx = max(0, bx - pad)
    ty = max(0, by - pad)
    tx2 = min(width, bx + bw + pad)
    ty2 = min(height, by + bh + pad)
    amended_box = [tx, ty, tx2 - tx, ty2 - ty]

    bundle_dir = _training_review_metadata_path(row).parent

    # Background fringe removal — generic, colour-estimated from crop border pixels
    _bg_fringe_debug: Dict[str, Any] = {}
    try:
        final_alpha_mask, _bg_fringe_debug = _remove_background_fringe_from_manual_mask(
            original=original,
            alpha_mask=final_alpha_mask,
            candidate_index=candidate_index,
            bundle_dir=bundle_dir,
        )
        # Keep refined_mask binary consistent with cleaned alpha
        refined_mask = refined_mask.copy()
        refined_mask[final_alpha_mask == 0] = 0
    except Exception as _exc:
        print(f"[manual-bg-fringe] candidate={candidate_index} error={_exc}")

    alpha_path = _manual_mask_output_path(candidate, "manual_mask_alpha_path", bundle_dir, candidate_index, "manual_mask_alpha.png")
    refined_mask_path = _manual_mask_output_path(candidate, "manual_refined_mask_path", bundle_dir, candidate_index, "manual_refined_mask.png")
    trimap_path = _manual_mask_output_path(candidate, "manual_trimap_path", bundle_dir, candidate_index, "manual_trimap.png")
    refined_overlay_path = _manual_mask_output_path(candidate, "manual_refined_overlay_path", bundle_dir, candidate_index, "manual_refined_overlay.png")
    amended_candidate_path = _manual_mask_output_path(candidate, "manual_amended_candidate_path", bundle_dir, candidate_index, "manual_amended.png")
    amended_mask_path = _manual_mask_output_path(candidate, "manual_amended_mask_path", bundle_dir, candidate_index, "manual_amended_mask.png")

    cv2.imwrite(str(alpha_path), final_alpha_mask)
    cv2.imwrite(str(refined_mask_path), refined_mask)
    cv2.imwrite(str(trimap_path), trimap)
    refined_overlay = original.copy()
    refined_overlay_colour = np.zeros_like(original)
    refined_overlay_colour[:, :, 0] = 40
    refined_overlay_colour[:, :, 1] = 210
    refined_overlay_colour[:, :, 2] = 40
    refined_overlay_mask = refined_mask > 0
    refined_overlay[refined_overlay_mask] = cv2.addWeighted(original[refined_overlay_mask], 0.62, refined_overlay_colour[refined_overlay_mask], 0.38, 0)
    cv2.imwrite(str(refined_overlay_path), refined_overlay)

    crop_original = original[ty:ty2, tx:tx2]
    crop_binary_mask = refined_mask[ty:ty2, tx:tx2]
    crop_alpha_mask = final_alpha_mask[ty:ty2, tx:tx2]
    bgra = cv2.cvtColor(crop_original, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = crop_alpha_mask
    bgra[crop_alpha_mask == 0, 0:3] = 0
    cv2.imwrite(str(amended_candidate_path), bgra)
    cv2.imwrite(str(amended_mask_path), crop_binary_mask)

    version = int(_coerce_int(candidate.get("manual_refinement_version")) or 0) + 1
    refined_at = _iso_now()
    updated_candidate = dict(candidate)
    # Preserve base_candidate_path (original crop, never changes)
    updated_candidate.setdefault("base_candidate_path", str(candidate.get("candidate_path") or ""))
    updated_candidate.setdefault("original_candidate_path", str(candidate.get("candidate_path") or ""))
    updated_candidate.setdefault("original_mask_path", str(candidate.get("mask_path") or ""))
    updated_candidate.setdefault("original_thumbnail_path", str(candidate.get("thumbnail_path") or ""))
    updated_candidate["box"] = amended_box
    # Do NOT overwrite candidate_path — it stays as the original crop (base)
    updated_candidate["mask_path"] = str(amended_mask_path)
    updated_candidate["thumbnail_path"] = str(amended_candidate_path)
    updated_candidate["current_candidate_path"] = str(amended_candidate_path)
    updated_candidate["current_alpha_path"] = str(alpha_path)
    updated_candidate["mask_review_state"] = "mask_amended"
    updated_candidate["manual_mask_alpha_path"] = str(alpha_path)
    updated_candidate["manual_refined_mask_path"] = str(refined_mask_path)
    updated_candidate["manual_trimap_path"] = str(trimap_path)
    updated_candidate["manual_refined_overlay_path"] = str(refined_overlay_path)
    updated_candidate["manual_amended_candidate_path"] = str(amended_candidate_path)
    updated_candidate["manual_amended_mask_path"] = str(amended_mask_path)
    updated_candidate["manual_refined_at"] = refined_at
    updated_candidate["manual_refinement_version"] = version
    updated_candidate["review_state"] = "mask_amended"
    updated_candidate["qty_scrubbed_path"] = ""
    updated_candidate["qty_scrubbed_mask_path"] = ""
    if bool(updated_candidate.get("qty_detected")):
        updated_candidate["qty_scrub_status"] = "manual_amended_needs_qty_scrub"
        updated_candidate["qty_text_state"] = "manual_amended_needs_qty_scrub"
    history = list(updated_candidate.get("cleanup_history") or [])
    history.append({"op": "manual_amend", "path": str(amended_candidate_path), "at": refined_at})
    updated_candidate["cleanup_history"] = history
    # Store background fringe metrics so candidate-alpha-debug can report them
    updated_candidate["manual_bg_removed_pixels"] = int(_bg_fringe_debug.get("background_like_removed_pixels") or 0)
    updated_candidate["manual_bg_pass1_removed"] = int(_bg_fringe_debug.get("pass1_removed") or 0)
    updated_candidate["manual_bg_pass2_removed"] = int(_bg_fringe_debug.get("pass2_removed") or 0)
    updated_candidate["manual_bg_model_rgb"] = _bg_fringe_debug.get("background_model_rgb")
    updated_candidate["manual_bg_dist_threshold"] = _bg_fringe_debug.get("background_distance_threshold")
    updated_candidate["manual_bg_fringe_preview_path"] = str(_bg_fringe_debug.get("bg_fringe_preview_path") or "")
    updated_candidate["manual_bg_likeness_mask_path"] = str(_bg_fringe_debug.get("bg_likeness_mask_path") or "")
    updated_candidate["manual_bg_minus_mask_path"] = str(_bg_fringe_debug.get("bg_minus_mask_path") or "")
    return updated_candidate, {
        "manual_mask_alpha_path": str(alpha_path),
        "manual_refined_mask_path": str(refined_mask_path),
        "manual_trimap_path": str(trimap_path),
        "manual_refined_overlay_path": str(refined_overlay_path),
        "manual_amended_candidate_path": str(amended_candidate_path),
        "manual_amended_mask_path": str(amended_mask_path),
        "manual_refined_at": refined_at,
        "manual_refinement_version": version,
        "background_like_removed_pixels": int(_bg_fringe_debug.get("background_like_removed_pixels") or 0),
        "background_model_rgb": _bg_fringe_debug.get("background_model_rgb"),
        "background_distance_threshold": _bg_fringe_debug.get("background_distance_threshold"),
        "bg_fringe_preview_path": str(_bg_fringe_debug.get("bg_fringe_preview_path") or ""),
        "bg_likeness_mask_path": str(_bg_fringe_debug.get("bg_likeness_mask_path") or ""),
        "bg_minus_mask_path": str(_bg_fringe_debug.get("bg_minus_mask_path") or ""),
    }


@router.get("/debug/training-store/candidate-alpha-debug")
def training_store_candidate_alpha_debug(
    bundle_id: str = Query(...),
    candidate_index: int = Query(...),
):
    try:
        row = dict(get_training_bundle_index_row(bundle_id).get("row") or {})
        candidate = _candidate_for_index_from_row(row, int(candidate_index))
        candidate_path = _candidate_display_path(candidate)
        amend_path = str(candidate.get("manual_mask_amend_path") or "")
        alpha_path = str(candidate.get("manual_mask_alpha_path") or "")
        refined_path = str(candidate.get("manual_refined_mask_path") or "")
        result = _alpha_debug_for_png(candidate_path)
        opaque_background_like_pixel_count = 0
        refined_foreground_pixel_count = 0
        if refined_path:
            refined_image = cv2.imread(refined_path, cv2.IMREAD_GRAYSCALE)
            candidate_image = cv2.imread(candidate_path, cv2.IMREAD_UNCHANGED)
            candidate_box = [int(_coerce_int(value) or 0) for value in list(candidate.get("box") or [])[:4]]
            if (
                refined_image is not None
                and candidate_image is not None
                and candidate_image.ndim == 3
                and candidate_image.shape[2] >= 4
                and len(candidate_box) == 4
            ):
                x, y, w, h = candidate_box
                crop_refined = refined_image[y : y + h, x : x + w]
                alpha = candidate_image[:, :, 3]
                if crop_refined.shape[:2] != alpha.shape[:2]:
                    crop_refined = cv2.resize(crop_refined, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_NEAREST)
                refined_foreground_pixel_count = int(np.count_nonzero(crop_refined > 0))
                opaque_background_like_pixel_count = int(np.count_nonzero((alpha >= 192) & (crop_refined <= 0)))
        result.update(
            {
                "ok": True,
                "bundle_id": str(bundle_id or ""),
                "candidate_index": int(candidate_index),
                "candidate_path": candidate_path,
                "background_like_removed_pixels": int(candidate.get("manual_bg_removed_pixels") or 0),
                "pass1_removed": int(candidate.get("manual_bg_pass1_removed") or 0),
                "pass2_removed": int(candidate.get("manual_bg_pass2_removed") or 0),
                "background_model_rgb": candidate.get("manual_bg_model_rgb"),
                "background_distance_threshold": candidate.get("manual_bg_dist_threshold"),
                "bg_fringe_preview_path": str(candidate.get("manual_bg_fringe_preview_path") or ""),
                "bg_likeness_mask_path": str(candidate.get("manual_bg_likeness_mask_path") or ""),
                "bg_minus_mask_path": str(candidate.get("manual_bg_minus_mask_path") or ""),
                "manual_mask_amend_path": amend_path,
                "manual_mask_alpha_path": alpha_path,
                "manual_refined_mask_path": refined_path,
                "manual_mask_amend_debug": _alpha_debug_for_png(amend_path) if amend_path else {},
                "manual_mask_alpha_debug": _alpha_debug_for_png(alpha_path) if alpha_path else {},
                "opaque_background_like_pixel_count": opaque_background_like_pixel_count,
                "refined_foreground_pixel_count": refined_foreground_pixel_count,
            }
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result)


@router.post("/debug/training-store/save-manual-mask-amendment")
async def training_store_save_manual_mask_amendment(req: Request):
    try:
        payload = await req.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    bundle_id = str(payload.get("bundle_id") or "").strip()
    candidate_index = _coerce_int(payload.get("candidate_index"))
    amended_by = str(payload.get("amended_by") or "andy").strip() or "andy"
    if not bundle_id or candidate_index is None:
        raise HTTPException(status_code=400, detail="bundle_id and candidate_index are required")
    try:
        row = dict(get_training_bundle_index_row(bundle_id).get("row") or {})
        metadata = _read_training_review_metadata(row)
        copied_files = metadata.get("copied_files") if isinstance(metadata.get("copied_files"), dict) else {}
        original_path = Path(str(copied_files.get("original_crop") or "").strip())
        if not original_path.exists() or not original_path.is_file():
            raise FileNotFoundError("original crop not found")
        original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        if original is None or getattr(original, "size", 0) == 0:
            raise ValueError("original crop is unreadable")
        height, width = original.shape[:2]
        paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
        candidates = [
            dict(item)
            for item in list(paths.get("candidates") or [])
            if isinstance(item, dict)
        ]
        candidate: Optional[Dict[str, Any]] = None
        for pos, item in enumerate(candidates):
            item_index = _coerce_int(item.get("index"))
            if item_index is None:
                item_index = pos
            if int(item_index) == int(candidate_index):
                candidate = dict(item)
                break
        if candidate is None:
            raise ValueError("candidate_index is out of range")
        box = [int(_coerce_int(value) or 0) for value in list(candidate.get("box") or [])[:4]]
        if len(box) != 4:
            raise ValueError("candidate box is missing")
        x, y, w, h = box
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        w = max(1, min(width - x, w))
        h = max(1, min(height - y, h))
        current_full_mask = np.zeros((height, width), dtype=np.uint8)
        mask_path = Path(str(candidate.get("mask_path") or "").strip())
        if mask_path.exists() and mask_path.is_file():
            current_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            current_mask = None
        if current_mask is None or getattr(current_mask, "size", 0) == 0:
            candidate_path = Path(str(candidate.get("candidate_path") or "").strip())
            candidate_image = cv2.imread(str(candidate_path), cv2.IMREAD_UNCHANGED) if candidate_path.exists() else None
            if candidate_image is not None and candidate_image.ndim == 3 and candidate_image.shape[2] >= 4:
                current_mask = candidate_image[:, :, 3]
        if current_mask is not None and getattr(current_mask, "size", 0) > 0:
            if current_mask.shape[:2] != (h, w):
                current_mask = cv2.resize(current_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            current_full_mask[y : y + h, x : x + w] = (current_mask > 0).astype(np.uint8) * 255
        add_mask = _decode_png_data_url(payload.get("add_mask_data_url"))
        erase_mask = _decode_png_data_url(payload.get("erase_mask_data_url"))
        if add_mask.shape[:2] != (height, width):
            add_mask = cv2.resize(add_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if erase_mask.shape[:2] != (height, width):
            erase_mask = cv2.resize(erase_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        _, add_binary = cv2.threshold(add_mask, 16, 255, cv2.THRESH_BINARY)
        _, erase_binary = cv2.threshold(erase_mask, 16, 255, cv2.THRESH_BINARY)
        _, base_binary = cv2.threshold(current_full_mask, 16, 255, cv2.THRESH_BINARY)
        combined_mask = base_binary.copy()
        combined_mask[add_binary > 0] = 255
        combined_mask[erase_binary > 0] = 0
        bundle_dir = _training_review_metadata_path(row).parent
        amend_path = bundle_dir / f"candidate_{int(candidate_index)}_manual_mask_amend.png"
        base_path = bundle_dir / f"candidate_{int(candidate_index)}_manual_mask_base.png"
        overlay_path = bundle_dir / f"candidate_{int(candidate_index)}_manual_mask_overlay.png"
        add_path = bundle_dir / f"candidate_{int(candidate_index)}_manual_mask_add.png"
        erase_path = bundle_dir / f"candidate_{int(candidate_index)}_manual_mask_erase.png"
        cv2.imwrite(str(amend_path), combined_mask)
        cv2.imwrite(str(base_path), base_binary)
        cv2.imwrite(str(add_path), add_binary)
        cv2.imwrite(str(erase_path), erase_binary)
        overlay = original.copy()
        overlay_colour = np.zeros_like(original)
        overlay_colour[:, :, 1] = 210
        overlay_colour[:, :, 2] = 70
        overlay_mask = combined_mask > 0
        overlay[overlay_mask] = cv2.addWeighted(original[overlay_mask], 0.62, overlay_colour[overlay_mask], 0.38, 0)
        cv2.imwrite(str(overlay_path), overlay)
        updated_candidate, refinement_paths = _write_manual_refinement_outputs(
            row=row,
            candidate=candidate,
            candidate_index=int(candidate_index),
            original=original,
            amended_mask=combined_mask,
        )
        updated_candidate["manual_mask_amend_path"] = str(amend_path)
        updated_candidate["manual_mask_base_path"] = str(base_path)
        updated_candidate["manual_mask_overlay_path"] = str(overlay_path)
        updated_candidate["manual_mask_add_path"] = str(add_path)
        updated_candidate["manual_mask_erase_path"] = str(erase_path)
        updated_candidate["amended_by"] = amended_by
        updated_candidate["amended_at"] = _iso_now()
        updated_paths = _update_candidate_in_split_paths(paths, int(candidate_index), updated_candidate)
        stored = update_split_candidates(bundle_id, split_candidate_paths=updated_paths)
        _mark_quick_review_cache_stale_for_bundle(bundle_id, "manual_mask_amended")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(
        {
            "ok": True,
            "bundle_id": bundle_id,
            "candidate_index": int(candidate_index),
            "review_state": "mask_amended",
            "manual_mask_amend_path": str(amend_path),
            "manual_mask_base_path": str(base_path),
            "manual_mask_overlay_path": str(overlay_path),
            **refinement_paths,
            "row": stored.get("row"),
        }
    )


@router.post("/debug/training-store/rerun-manual-mask-refinement")
async def training_store_rerun_manual_mask_refinement(req: Request):
    try:
        payload = await req.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    query = dict(req.query_params)
    bundle_id = str(payload.get("bundle_id") or query.get("bundle_id") or "").strip()
    candidate_index = _coerce_int(payload.get("candidate_index", query.get("candidate_index")))
    if not bundle_id or candidate_index is None:
        raise HTTPException(status_code=400, detail="bundle_id and candidate_index are required")
    try:
        row = dict(get_training_bundle_index_row(bundle_id).get("row") or {})
        metadata = _read_training_review_metadata(row)
        copied_files = metadata.get("copied_files") if isinstance(metadata.get("copied_files"), dict) else {}
        original_path = Path(str(copied_files.get("original_crop") or "").strip())
        if not original_path.exists() or not original_path.is_file():
            raise FileNotFoundError("original crop not found")
        original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        if original is None or getattr(original, "size", 0) == 0:
            raise ValueError("original crop is unreadable")
        height, width = original.shape[:2]
        paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
        candidate = _candidate_for_index_from_row(row, int(candidate_index))

        def read_full_mask(path_value: Any) -> Optional[np.ndarray]:
            path = Path(str(path_value or "").strip())
            if not path.exists() or not path.is_file():
                return None
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or getattr(mask, "size", 0) == 0:
                return None
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            _, mask = cv2.threshold(mask, 16, 255, cv2.THRESH_BINARY)
            return mask

        base_mask = read_full_mask(candidate.get("manual_mask_base_path"))
        add_mask = read_full_mask(candidate.get("manual_mask_add_path"))
        erase_mask = read_full_mask(candidate.get("manual_mask_erase_path"))
        amend_mask = read_full_mask(candidate.get("manual_mask_amend_path"))
        if base_mask is not None:
            combined_mask = base_mask.copy()
            if add_mask is not None:
                combined_mask[add_mask > 0] = 255
            if erase_mask is not None:
                combined_mask[erase_mask > 0] = 0
        elif amend_mask is not None:
            combined_mask = amend_mask.copy()
        else:
            raise ValueError("manual amendment masks are missing")

        updated_candidate, refinement_paths = _write_manual_refinement_outputs(
            row=row,
            candidate=candidate,
            candidate_index=int(candidate_index),
            original=original,
            amended_mask=combined_mask,
        )
        updated_candidate["manual_refined_by"] = str(payload.get("refined_by") or "andy").strip() or "andy"
        updated_paths = _update_candidate_in_split_paths(paths, int(candidate_index), updated_candidate)
        stored = update_split_candidates(bundle_id, split_candidate_paths=updated_paths)
        _mark_quick_review_cache_stale_for_bundle(bundle_id, "manual_refinement_rerun")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(
        {
            "ok": True,
            "bundle_id": bundle_id,
            "candidate_index": int(candidate_index),
            "review_state": "mask_amended",
            **refinement_paths,
            "row": stored.get("row"),
        }
    )


@router.post("/debug/training-store/save-picked-colour-cleanup")
async def training_store_save_picked_colour_cleanup(req: Request):
    """
    Remove alpha pixels in the current candidate that are colour-close to the
    user-picked RGB, while preserving pixels on strong edges/texture.

    Inputs (JSON body):
      bundle_id       : str
      candidate_index : int
      picked_rgb      : [r, g, b]   (0–255 each)
      tolerance       : float        (LAB ΔE distance, 5–80)
    """
    try:
        payload = await req.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    bundle_id = str(payload.get("bundle_id") or "").strip()
    candidate_index = _coerce_int(payload.get("candidate_index"))
    picked_rgb_raw = payload.get("picked_rgb")
    tolerance_raw = payload.get("tolerance")

    if not bundle_id or candidate_index is None:
        raise HTTPException(status_code=400, detail="bundle_id and candidate_index are required")
    if not isinstance(picked_rgb_raw, (list, tuple)) or len(picked_rgb_raw) < 3:
        raise HTTPException(status_code=400, detail="picked_rgb must be [r, g, b]")
    try:
        picked_bgr = [int(picked_rgb_raw[2]), int(picked_rgb_raw[1]), int(picked_rgb_raw[0])]
        picked_rgb = [int(picked_rgb_raw[0]), int(picked_rgb_raw[1]), int(picked_rgb_raw[2])]
        tolerance = float(tolerance_raw if tolerance_raw is not None else 22.0)
        tolerance = max(5.0, min(80.0, tolerance))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid parameters: {exc}")

    try:
        row = dict(get_training_bundle_index_row(bundle_id).get("row") or {})
        metadata = _read_training_review_metadata(row)
        copied_files = metadata.get("copied_files") if isinstance(metadata.get("copied_files"), dict) else {}
        original_path = Path(str(copied_files.get("original_crop") or "").strip())
        if not original_path.exists() or not original_path.is_file():
            raise FileNotFoundError("original crop not found")
        original_bgr = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        if original_bgr is None or getattr(original_bgr, "size", 0) == 0:
            raise ValueError("original crop is unreadable")

        candidate = _candidate_for_index_from_row(row, int(candidate_index))
        # Read the CURRENT candidate PNG (uses current_candidate_path if available)
        candidate_img_path = Path(_candidate_display_path(candidate))
        if not candidate_img_path.exists() or not candidate_img_path.is_file():
            raise FileNotFoundError(f"candidate image not found: {candidate_img_path}")

        candidate_bgra = cv2.imread(str(candidate_img_path), cv2.IMREAD_UNCHANGED)
        if candidate_bgra is None or getattr(candidate_bgra, "size", 0) == 0:
            raise ValueError("candidate image is unreadable")
        if candidate_bgra.ndim < 3 or candidate_bgra.shape[2] < 4:
            raise ValueError("candidate image has no alpha channel")

        c_h, c_w = candidate_bgra.shape[:2]
        alpha = candidate_bgra[:, :, 3].copy()
        rgb_pixels = candidate_bgra[:, :, :3]  # BGR

        # ── candidate box for cropping the original to the same region ──
        candidate_box = [int(_coerce_int(v) or 0) for v in list(candidate.get("box") or [])[:4]]
        if len(candidate_box) == 4:
            bx, by, bw, bh = candidate_box
            original_crop_region = original_bgr[by:by+bh, bx:bx+bw]
            if original_crop_region.shape[:2] != (c_h, c_w):
                original_crop_region = cv2.resize(original_crop_region, (c_w, c_h), interpolation=cv2.INTER_LINEAR)
        else:
            original_crop_region = cv2.resize(original_bgr, (c_w, c_h), interpolation=cv2.INTER_LINEAR)

        # ── per-pixel LAB distance from picked colour ──
        picked_bgr_u8 = np.array([[picked_bgr]], dtype=np.uint8)
        picked_lab = cv2.cvtColor(picked_bgr_u8, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)

        img_lab = cv2.cvtColor(rgb_pixels, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta = img_lab - picked_lab[np.newaxis, np.newaxis, :]
        pixel_dist = np.sqrt(np.sum(delta * delta, axis=2))  # (h, w)

        # ── strong local edge protection (Sobel on original crop region) ──
        gray_orig = cv2.cvtColor(original_crop_region, cv2.COLOR_BGR2GRAY)
        sx_g = cv2.Sobel(gray_orig, cv2.CV_32F, 1, 0, ksize=3)
        sy_g = cv2.Sobel(gray_orig, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sx_g * sx_g + sy_g * sy_g)
        strong_edge = gradient_mag > 25.0

        # ── removal mask: colour-close AND not a strong edge AND currently opaque ──
        remove_mask: np.ndarray = (
            (pixel_dist < tolerance)
            & (~strong_edge)
            & (alpha > 16)
        )
        removed_count = int(np.count_nonzero(remove_mask))

        # ── apply removal ──
        cleaned_alpha = alpha.copy()
        cleaned_alpha[remove_mask] = 0

        # ── save outputs ──
        bundle_dir = _training_review_metadata_path(row).parent
        bundle_dir.mkdir(parents=True, exist_ok=True)

        cleanup_path = bundle_dir / f"candidate_{candidate_index}_colour_cleanup.png"
        cleanup_mask_path = bundle_dir / f"candidate_{candidate_index}_colour_cleanup_mask.png"
        cleanup_alpha_path = bundle_dir / f"candidate_{candidate_index}_colour_cleanup_alpha.png"
        cleanup_overlay_path = bundle_dir / f"candidate_{candidate_index}_colour_cleanup_overlay.png"

        # Cleaned candidate BGRA
        cleaned_bgra = candidate_bgra.copy()
        cleaned_bgra[:, :, 3] = cleaned_alpha
        cleaned_bgra[cleaned_alpha == 0, 0:3] = 0
        cv2.imwrite(str(cleanup_path), cleaned_bgra)

        # Binary cleanup mask (255 = removed pixel) — debug / overlay only
        cleanup_mask_img = (remove_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(cleanup_mask_path), cleanup_mask_img)

        # Current alpha mask (255 = remaining opaque pixel) — used by match scoring
        cv2.imwrite(str(cleanup_alpha_path), cleaned_alpha)

        # Overlay: original crop with removed pixels tinted magenta
        overlay = original_crop_region.copy().astype(np.float32)
        if removed_count > 0:
            overlay[remove_mask] = (
                overlay[remove_mask] * 0.3
                + np.array([180.0, 30.0, 180.0]) * 0.7
            )
        cv2.imwrite(str(cleanup_overlay_path), overlay.clip(0, 255).astype(np.uint8))

        print(
            f"[picked-colour-cleanup] bundle={bundle_id} candidate={candidate_index}"
            f" picked_rgb={picked_rgb} tolerance={tolerance:.1f}"
            f" removed={removed_count}"
        )

        # ── update candidate record ──
        paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
        now = _iso_now()
        updated_candidate = dict(candidate)
        # current_candidate_path advances; candidate_path (base) never changes
        updated_candidate["current_candidate_path"] = str(cleanup_path)
        updated_candidate["current_alpha_path"] = str(cleanup_alpha_path)   # remaining alpha, not removed-pixels
        updated_candidate["thumbnail_path"] = str(cleanup_path)
        updated_candidate["colour_cleanup_path"] = str(cleanup_path)
        updated_candidate["colour_cleanup_alpha_path"] = str(cleanup_alpha_path)
        updated_candidate["colour_cleanup_mask_path"] = str(cleanup_mask_path)
        updated_candidate["colour_cleanup_overlay_path"] = str(cleanup_overlay_path)
        updated_candidate["colour_cleanup_picked_rgb"] = picked_rgb
        updated_candidate["colour_cleanup_tolerance"] = round(tolerance, 2)
        updated_candidate["colour_cleanup_removed_pixels"] = removed_count
        updated_candidate["colour_cleanup_at"] = now
        history = list(updated_candidate.get("cleanup_history") or [])
        history.append({"op": "colour_cleanup", "path": str(cleanup_path), "at": now})
        updated_candidate["cleanup_history"] = history

        updated_paths = _update_candidate_in_split_paths(paths, int(candidate_index), updated_candidate)
        stored = update_split_candidates(bundle_id, split_candidate_paths=updated_paths)
        _mark_quick_review_cache_stale_for_bundle(bundle_id, "colour_cleanup")

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return JSONResponse({
        "ok": True,
        "bundle_id": bundle_id,
        "candidate_index": int(candidate_index),
        "picked_rgb": picked_rgb,
        "tolerance": round(tolerance, 2),
        "removed_pixels": removed_count,
        "cleanup_path": str(cleanup_path),
        "cleanup_mask_path": str(cleanup_mask_path),
        "cleanup_overlay_path": str(cleanup_overlay_path),
    })


@router.post("/debug/training-store/mark-qty-clean")
async def training_store_mark_qty_clean(req: Request):
    """
    Mark a candidate's qty text as clean without running an image scrub.
    Used when the qty token is already invisible / outside the alpha region
    after manual amend, and re-scrubbing would be redundant.

    Inputs (JSON body or query params):
      bundle_id       : str
      candidate_index : int
    """
    try:
        body = await req.json()
        payload = dict(body) if isinstance(body, dict) else {}
    except Exception:
        payload = {}
    query = dict(req.query_params)
    bundle_id = str(payload.get("bundle_id") or query.get("bundle_id") or "").strip()
    candidate_index_raw = payload.get("candidate_index", query.get("candidate_index"))
    try:
        candidate_index = int(candidate_index_raw)
    except Exception:
        raise HTTPException(status_code=400, detail="candidate_index is required")
    if not bundle_id:
        raise HTTPException(status_code=400, detail="bundle_id is required")

    try:
        row = dict(get_training_bundle_index_row(bundle_id).get("row") or {})
        paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
        candidates = list(paths.get("candidates") or [])
        if candidate_index < 0 or candidate_index >= len(candidates):
            raise ValueError("candidate_index is out of range")
        candidate = dict(candidates[candidate_index]) if isinstance(candidates[candidate_index], dict) else {}
        now = _iso_now()
        updated_candidate = dict(candidate)
        updated_candidate["qty_text_state"] = "manual_mark_clean"
        updated_candidate["qty_scrub_status"] = "manual_mark_clean"
        history = list(updated_candidate.get("cleanup_history") or [])
        history.append({"op": "mark_qty_clean", "at": now})
        updated_candidate["cleanup_history"] = history
        updated_paths = _update_candidate_in_split_paths(paths, int(candidate_index), updated_candidate)
        stored = update_split_candidates(bundle_id, split_candidate_paths=updated_paths)
        _mark_quick_review_cache_stale_for_bundle(bundle_id, "mark_qty_clean")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return JSONResponse({
        "ok": True,
        "bundle_id": bundle_id,
        "candidate_index": int(candidate_index),
        "qty_text_state": "manual_mark_clean",
    })


@router.post("/debug/training-store/confirm-candidate-part")
async def training_store_confirm_candidate_part(req: Request):
    payload: Dict[str, Any] = {}
    try:
        body = await req.json()
        if isinstance(body, dict):
            payload = dict(body)
    except Exception:
        payload = {}
    query = dict(req.query_params)
    bundle_id = str(payload.get("bundle_id") or query.get("bundle_id") or "").strip()
    candidate_index_raw = payload.get("candidate_index", query.get("candidate_index"))
    try:
        candidate_index = int(candidate_index_raw)
    except Exception:
        raise HTTPException(status_code=400, detail="candidate_index is required")
    try:
        row = dict(get_training_bundle_index_row(bundle_id).get("row") or {})
        paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
        candidates = [
            dict(item)
            for item in list(paths.get("candidates") or [])
            if isinstance(item, dict)
        ]
        candidate = next(
            (
                item
                for item in candidates
                if _coerce_int(item.get("index")) == int(candidate_index)
            ),
            None,
        )
        if candidate is None and 0 <= candidate_index < len(candidates):
            candidate = candidates[candidate_index]
        if candidate is None:
            raise ValueError("candidate_index is out of range")
        if str(candidate.get("status") or "") != "accepted":
            raise ValueError("candidate must be accepted before part confirmation")
        if not _candidate_qty_is_clean(candidate):
            raise ValueError("candidate must be qty-clean before part confirmation; scrub qty or use Mark Qty Clean")
        qty_values = list(candidate.get("qty_values") or [])
        qty = payload.get("qty", query.get("qty"))
        if qty in {None, ""} and qty_values:
            qty = qty_values[0]
        thumbnail_path = str(
            payload.get("thumbnail_path")
            or _candidate_display_path(candidate)
            or ""
        )
        result = confirm_candidate_part(
            bundle_id=bundle_id,
            candidate_index=candidate_index,
            part_num=str(payload.get("part_num") or query.get("part_num") or ""),
            color_id=payload.get("color_id", query.get("color_id")),
            element_id=str(payload.get("element_id") or query.get("element_id") or ""),
            qty=qty,
            thumbnail_path=thumbnail_path,
            r2_path=str(payload.get("r2_path") or query.get("r2_path") or candidate.get("r2_path") or ""),
            confirmed_by=str(payload.get("confirmed_by") or query.get("confirmed_by") or "andy"),
        )
        dry_run_value = payload.get("dry_run", query.get("dry_run", "0"))
        upload_result: Dict[str, Any]
        try:
            upload_result = upload_confirmed_candidate_assets(
                bundle_id=bundle_id,
                candidate_index=candidate_index,
                candidate=candidate,
                confirmation=dict(result.get("row") or {}),
                row=row,
                dry_run=_dry_run_enabled(str(dry_run_value)),
            )
        except Exception as exc:
            upload_result = {
                "ok": False,
                "uploaded": False,
                "error": type(exc).__name__,
                "detail": str(exc),
            }
        result["candidate_r2_upload"] = upload_result
        if not bool(upload_result.get("ok")):
            result["warning"] = "candidate confirmation saved, but R2 upload failed"
        elif bool(upload_result.get("skipped")):
            result["warning"] = "candidate confirmation saved, but R2 upload was skipped"
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _mark_quick_review_cache_stale_for_bundle(bundle_id, "candidate_confirmed")
    return JSONResponse(result)


@router.post("/debug/training-store/unconfirm-candidate-part")
def training_store_unconfirm_candidate_part(
    bundle_id: str = Query(...),
    candidate_index: int = Query(...),
):
    try:
        result = unconfirm_candidate_part(bundle_id=bundle_id, candidate_index=candidate_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _mark_quick_review_cache_stale_for_bundle(bundle_id, "candidate_unconfirmed")
    return JSONResponse(result)


@router.post("/debug/training-store/reset-bag")
def training_store_reset_bag(
    set_num: str = Query(...),
    bag_num: int = Query(...),
    confirm: str = Query(""),
):
    safe_set_num = re.sub(r"[^A-Za-z0-9.-]+", "", str(set_num or "").strip())
    parsed_bag_num = _coerce_int(bag_num)
    if confirm != "RESET":
        raise HTTPException(status_code=400, detail="confirm=RESET is required")
    if not safe_set_num or parsed_bag_num is None:
        raise HTTPException(status_code=400, detail="set_num and bag_num are required")

    bundle_prefix = f"{safe_set_num}_bag{int(parsed_bag_num)}_"
    analysis_root = (
        Path("/Users/olly/aim2build-instruction")
        / "debug"
        / "ai_training"
        / "analysis_bundles"
    ).resolve()
    deleted_paths: List[str] = []
    if analysis_root.exists() and analysis_root.is_dir():
        for bundle_dir in sorted(analysis_root.glob(f"{bundle_prefix}*")):
            if not bundle_dir.is_dir():
                continue
            resolved = bundle_dir.resolve()
            if analysis_root not in resolved.parents:
                continue
            shutil.rmtree(resolved)
            deleted_paths.append(str(resolved))
    try:
        db_result = reset_bag_index_rows(set_num=safe_set_num, bag_num=int(parsed_bag_num))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(
        {
            "ok": True,
            "set_num": safe_set_num,
            "bag_num": int(parsed_bag_num),
            "bundle_prefix": bundle_prefix,
            "local_analysis_bundle_deleted_count": len(deleted_paths),
            "local_analysis_bundle_deleted_paths": deleted_paths,
            **db_result,
            "r2_touched": False,
            "catalog_db_touched": False,
        }
    )


@router.get("/debug/dev/find-obsolete-candidates")
def dev_find_obsolete_candidates(
    set_num: str = Query(""),
    bag_num: str = Query(""),
    limit: int = Query(200),
):
    """
    Scans stored candidate JSON across all (or filtered) bundles and reports:
    - candidates missing current_candidate_path
    - candidates where current_alpha_path points to the old removed-pixels mask
      (i.e. colour_cleanup_mask_path == current_alpha_path)
    - candidates with legacy fields but no base_candidate_path
    - orphaned debug overlay files on disk
    - duplicate intermediate files (e.g. amended + cleanup both present)

    Read-only. Does NOT delete or modify anything.
    """
    set_filter = str(set_num or "").strip()
    bag_filter = _coerce_int(bag_num) if str(bag_num or "").strip() else None
    limit_val = max(1, min(int(limit or 200), 500))

    queue_result = list_review_queue(set_num=set_filter, bag_num=bag_filter, limit=limit_val)
    bundle_rows = [dict(r) for r in list(queue_result.get("rows") or []) if isinstance(r, dict)]

    # ── orphan overlay patterns (files written but not displayed in main UI) ──
    _ORPHAN_SUFFIXES = [
        "_background_likeness_mask.png",
        "_refined_minus_background_mask.png",
        "_manual_trimap.png",
        "_manual_refined_overlay.png",
        "_colour_cleanup_overlay.png",
    ]

    missing_current_candidate_path: List[Dict[str, Any]] = []
    wrong_alpha_path: List[Dict[str, Any]] = []
    missing_base_candidate_path: List[Dict[str, Any]] = []
    has_legacy_only_fields: List[Dict[str, Any]] = []
    duplicate_intermediates: List[Dict[str, Any]] = []
    orphaned_overlays: List[Dict[str, Any]] = []

    analysis_root = (
        Path("/Users/olly/aim2build-instruction")
        / "debug" / "ai_training" / "analysis_bundles"
    ).resolve()

    for b_row in bundle_rows:
        bundle_id = str(b_row.get("bundle_id") or "")
        paths = b_row.get("split_candidate_paths") if isinstance(b_row.get("split_candidate_paths"), dict) else {}
        candidates = [dict(c) for c in list(paths.get("candidates") or []) if isinstance(c, dict)]

        # Per-candidate checks
        for c in candidates:
            cidx = c.get("index")
            entry = {"bundle_id": bundle_id, "candidate_index": cidx}

            # 1. Missing current_candidate_path
            if not str(c.get("current_candidate_path") or "").strip():
                missing_current_candidate_path.append({
                    **entry,
                    "candidate_path": str(c.get("candidate_path") or ""),
                    "has_manual_amended": bool(c.get("manual_amended_candidate_path")),
                    "has_qty_scrubbed": bool(c.get("qty_scrubbed_path")),
                })

            # 2. current_alpha_path points to the removed-pixels mask (wrong)
            cur_alpha = str(c.get("current_alpha_path") or "")
            cleanup_mask = str(c.get("colour_cleanup_mask_path") or "")
            if cur_alpha and cleanup_mask and cur_alpha == cleanup_mask:
                wrong_alpha_path.append({
                    **entry,
                    "current_alpha_path": cur_alpha,
                    "colour_cleanup_alpha_path": str(c.get("colour_cleanup_alpha_path") or ""),
                    "note": "current_alpha_path == colour_cleanup_mask_path (removed-pixels mask); should be colour_cleanup_alpha_path",
                })

            # 3. Missing base_candidate_path
            if not str(c.get("base_candidate_path") or "").strip():
                missing_base_candidate_path.append({
                    **entry,
                    "candidate_path": str(c.get("candidate_path") or ""),
                    "note": "pre-refactor bundle; base_candidate_path not set",
                })

            # 4. Has original_candidate_path but no base_candidate_path (pre-refactor legacy)
            if str(c.get("original_candidate_path") or "").strip() and not str(c.get("base_candidate_path") or "").strip():
                has_legacy_only_fields.append({
                    **entry,
                    "original_candidate_path": str(c.get("original_candidate_path") or ""),
                    "note": "original_candidate_path present; base_candidate_path absent",
                })

            # 5. Both manual_amended_candidate_path and colour_cleanup_path present
            #    (cleanup supersedes amend; amended file may be orphaned)
            if str(c.get("manual_amended_candidate_path") or "").strip() and str(c.get("colour_cleanup_path") or "").strip():
                duplicate_intermediates.append({
                    **entry,
                    "manual_amended_candidate_path": str(c.get("manual_amended_candidate_path") or ""),
                    "colour_cleanup_path": str(c.get("colour_cleanup_path") or ""),
                    "current_candidate_path": str(c.get("current_candidate_path") or ""),
                    "note": "both amend and cleanup outputs exist; current_candidate_path should point to cleanup",
                })

        # Per-bundle: scan for orphaned overlay files on disk
        bundle_dir = analysis_root / bundle_id
        if bundle_dir.exists() and bundle_dir.is_dir():
            for suffix in _ORPHAN_SUFFIXES:
                for f in sorted(bundle_dir.glob(f"*{suffix}")):
                    # Only flag if no candidate currently references it
                    path_str = str(f)
                    referenced = any(
                        path_str in (
                            str(c.get("manual_bg_likeness_mask_path") or ""),
                            str(c.get("manual_bg_fringe_preview_path") or ""),
                            str(c.get("manual_bg_minus_mask_path") or ""),
                            str(c.get("manual_trimap_path") or ""),
                            str(c.get("manual_refined_overlay_path") or ""),
                            str(c.get("colour_cleanup_overlay_path") or ""),
                        )
                        for c in candidates
                    )
                    orphaned_overlays.append({
                        "bundle_id": bundle_id,
                        "file": path_str,
                        "size_bytes": f.stat().st_size,
                        "referenced_in_candidate": referenced,
                    })

    summary = {
        "bundles_scanned": len(bundle_rows),
        "missing_current_candidate_path": len(missing_current_candidate_path),
        "wrong_alpha_path": len(wrong_alpha_path),
        "missing_base_candidate_path": len(missing_base_candidate_path),
        "has_legacy_only_fields": len(has_legacy_only_fields),
        "duplicate_intermediates": len(duplicate_intermediates),
        "orphaned_overlays": len(orphaned_overlays),
    }
    return JSONResponse({
        "ok": True,
        "filter": {"set_num": set_filter or None, "bag_num": bag_filter, "limit": limit_val},
        "summary": summary,
        "missing_current_candidate_path": missing_current_candidate_path,
        "wrong_alpha_path": wrong_alpha_path,
        "missing_base_candidate_path": missing_base_candidate_path,
        "has_legacy_only_fields": has_legacy_only_fields,
        "duplicate_intermediates": duplicate_intermediates,
        "orphaned_overlays": orphaned_overlays,
        "note": "Read-only scan. Nothing was deleted or modified.",
    })


def _training_review_metadata_path(row: Dict[str, Any]) -> Path:
    manifest_path = Path(str(row.get("manifest_path") or "").strip())
    if manifest_path.exists() and manifest_path.is_file():
        return manifest_path
    bundle_id = str(row.get("bundle_id") or "").strip()
    return (
        Path("/Users/olly/aim2build-instruction")
        / "debug"
        / "ai_training"
        / "analysis_bundles"
        / bundle_id
        / "metadata.json"
    )


def _read_training_review_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    path = _training_review_metadata_path(row)
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _training_review_img(path_value: Any, alt: str) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        return '<div class="missing">missing</div>'
    src = f"/debug/ai-snap-artifact?path={_url_quote(path_text)}"
    return f'<img src="{src}" alt="{escape(alt)}" loading="lazy">'


def _training_review_alpha_compare(path_value: Any, alt: str) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        return '<div class="missing">missing</div>'
    src = f"/debug/ai-snap-artifact?path={_url_quote(path_text)}"
    labels = [("checkerboard", "checker"), ("black", "black"), ("white", "white")]
    panels = "".join(
        f'<div class="alpha-preview {escape(class_name)}"><img src="{src}" alt="{escape(alt + " on " + label)}" loading="lazy"><span>{escape(label)}</span></div>'
        for label, class_name in labels
    )
    return f'<div class="alpha-preview-grid">{panels}</div>'


def _training_review_catalog_img(path_value: Any, alt: str) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        return '<div class="missing">no image</div>'
    if path_text.startswith(("http://", "https://")):
        src = path_text
    else:
        src = f"/debug/ai-snap-artifact?path={_url_quote(path_text)}"
    return f'<img src="{escape(src)}" alt="{escape(alt)}" loading="lazy">'


def _training_review_confirmed_example_totals(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        part_num = str(row.get("part_num") or "").strip()
        color_id = _coerce_int(row.get("color_id"))
        if not part_num or color_id is None:
            continue
        qty = _coerce_int(row.get("qty"))
        assigned_qty = int(qty) if qty is not None and int(qty) > 0 else 1
        key = _candidate_part_key(part_num, color_id)
        totals[key] = totals.get(key, 0) + assigned_qty
    return totals


def _training_review_candidate_part_matches(
    *,
    row: Dict[str, Any],
    candidate: Dict[str, Any],
    confirmed_rows: List[Dict[str, Any]],
    limit: int = 5,
    scope: str = "bag",
    confirmed_totals_override: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    set_num = str(row.get("set_num") or "").strip()
    bag = _coerce_int(row.get("bag_num")) or 1
    if not set_num:
        parsed = re.match(r"^(?P<set>[^_]+)_bag(?P<bag>\d+)_", str(row.get("bundle_id") or ""))
        if parsed:
            set_num = str(parsed.group("set") or "")
            bag = _coerce_int(parsed.group("bag")) or bag
    thumbnail_path = str(_candidate_display_path(candidate) or candidate.get("thumbnail_path") or "").strip()
    # Prefer the dedicated alpha files (not the "removed pixels" debug mask)
    mask_path = str(
        candidate.get("colour_cleanup_alpha_path")
        or candidate.get("current_alpha_path")
        or candidate.get("qty_scrubbed_mask_path")
        or candidate.get("mask_path")
        or ""
    ).strip()
    if not set_num or not thumbnail_path:
        return []
    query_profile = _slot_mask_query_profile(thumbnail_path, mask_path)
    if query_profile is None:
        return []

    if scope == "set":
        parts_payload = load_instruction_set_parts(str(set_num))
        candidate_rows = _prepare_instruction_parts_for_display(list(parts_payload.get("parts", []) or []))
        bag_required_by_key = {
            _candidate_part_key(row.get("part_num"), row.get("color_id")): int(row.get("bag_required_qty") or 0)
            for row in _debug_bag_specific_part_rows(str(set_num), int(bag))
        }
        for candidate_row in candidate_rows:
            key = _candidate_part_key(candidate_row.get("part_num"), candidate_row.get("color_id"))
            candidate_row["bag_required_qty"] = int(bag_required_by_key.get(key, 0) or 0)
    else:
        candidate_rows, _pool_source = _slot_mask_candidate_pool(set_num, int(bag))
    if not candidate_rows:
        return []
    confirmed_totals = dict(confirmed_totals_override or _training_review_confirmed_example_totals(confirmed_rows))
    color_ids = [int(part.get("color_id", 0) or 0) for part in candidate_rows]
    color_bgr_by_id = {
        int(item["color_id"]): _slot_mask_hex_to_bgr(item.get("rgb"))
        for item in _load_catalog_colors_for_ids(color_ids)
    }
    color_catalog_by_id = {
        int(item.get("color_id", 0) or 0): dict(item)
        for item in _load_catalog_colors_for_ids(color_ids)
    }
    color_bgr_by_id = {
        color_id: bgr
        for color_id, bgr in color_bgr_by_id.items()
        if bgr is not None
    }
    ranked: List[Dict[str, Any]] = []
    for part in candidate_rows:
        part_num = str(part.get("part_num") or "").strip()
        color_id = _coerce_int(part.get("color_id"))
        if not part_num or color_id is None:
            continue
        key = _candidate_part_key(part_num, color_id)
        set_required_qty = int(part.get("set_required_qty", 0) or 0)
        bag_required_qty = int(part.get("bag_required_qty", 0) or 0)
        confirmed_qty = int(confirmed_totals.get(key, 0) or 0)
        remaining_qty = set_required_qty - confirmed_qty
        effective_remaining_qty = max(0, remaining_qty)
        catalog_color = color_catalog_by_id.get(int(color_id), {})
        scores = _slot_mask_score_candidate(query_profile, part, color_bgr_by_id)
        ranked.append(
            {
                "part_num": part_num,
                "display_part_num": str(part.get("display_part_num") or part_num),
                "color_id": int(color_id),
                "color_name": str(part.get("color_name") or catalog_color.get("color_name") or f"color {int(color_id)}"),
                "color_rgb": str(catalog_color.get("rgb") or ""),
                "element_id": str(part.get("element_id") or ""),
                "part_name": str(part.get("part_name") or part.get("name") or part.get("part_name_en") or part.get("description") or ""),
                "image_url": str(part.get("img_url") or "").strip(),
                "image_path": str(_slot_mask_resolve_local_image_path(part.get("img_url")) or ""),
                "set_required_qty": set_required_qty,
                "bag_required_qty": bag_required_qty,
                "required_qty": set_required_qty,
                "confirmed_qty": confirmed_qty,
                "remaining_qty": remaining_qty,
                "effective_remaining_qty": effective_remaining_qty,
                "over_confirmed": confirmed_qty > set_required_qty,
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
            -float(item.get("confidence", 0.0) or 0.0),
            -float((item.get("score_breakdown") or {}).get("colour", 0.0) or 0.0),
            -int(item.get("remaining_qty", 0) or 0),
            str(item.get("part_num") or ""),
        )
    )
    if limit is None or int(limit or 0) <= 0:
        return ranked
    return ranked[: max(1, int(limit or 5))]


def _training_review_required_part_rows(
    *,
    set_num: str,
    bag_num: int,
    scope: str,
    confirmed_totals: Dict[str, int],
) -> List[Dict[str, Any]]:
    parts_payload = load_instruction_set_parts(str(set_num))
    set_parts = _prepare_instruction_parts_for_display(list(parts_payload.get("parts", []) or []))
    set_by_key = {
        _candidate_part_key(part.get("part_num"), part.get("color_id")): dict(part or {})
        for part in set_parts
    }
    bag_rows = _debug_bag_specific_part_rows(str(set_num), int(bag_num or 1))
    bag_required_by_key = {
        _candidate_part_key(row.get("part_num"), row.get("color_id")): int(row.get("bag_required_qty") or 0)
        for row in list(bag_rows or [])
    }
    if scope == "bag":
        seed_rows = bag_rows if bag_rows else set_parts
    else:
        seed_rows = set_parts
    color_ids = [
        int(row.get("color_id", 0) or 0)
        for row in list(seed_rows or [])
        if _coerce_int(row.get("color_id")) is not None
    ]
    color_catalog_by_id = {
        int(item.get("color_id", 0) or 0): dict(item)
        for item in _load_catalog_colors_for_ids(color_ids)
    }
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw_row in list(seed_rows or []):
        row = dict(raw_row or {})
        part_num = str(row.get("part_num") or "").strip()
        color_id = _coerce_int(row.get("color_id"))
        if not part_num or color_id is None:
            continue
        key = _candidate_part_key(part_num, color_id)
        if key in seen:
            continue
        seen.add(key)
        meta = dict(set_by_key.get(key, {}) or {})
        catalog_color = color_catalog_by_id.get(int(color_id), {})
        set_required_qty = int(meta.get("set_required_qty", 0) or 0)
        bag_required_qty = int(row.get("bag_required_qty", bag_required_by_key.get(key, 0)) or 0)
        confirmed_qty = int(confirmed_totals.get(key, 0) or 0)
        remaining_qty = set_required_qty - confirmed_qty
        rows.append(
            {
                "part_num": part_num,
                "display_part_num": str(meta.get("display_part_num") or part_num),
                "color_id": int(color_id),
                "color_name": str(meta.get("color_name") or catalog_color.get("color_name") or f"color {int(color_id)}"),
                "color_rgb": str(catalog_color.get("rgb") or meta.get("rgb") or ""),
                "element_id": str(meta.get("element_id") or row.get("element_id") or ""),
                "part_name": str(meta.get("part_name") or meta.get("name") or meta.get("description") or ""),
                "image_url": str(meta.get("img_url") or row.get("img_url") or "").strip(),
                "image_path": str(_slot_mask_resolve_local_image_path(meta.get("img_url") or row.get("img_url")) or ""),
                "set_required_qty": set_required_qty,
                "bag_required_qty": bag_required_qty,
                "confirmed_qty": confirmed_qty,
                "remaining_qty": remaining_qty,
                "effective_remaining_qty": max(0, remaining_qty),
                "completed": set_required_qty > 0 and remaining_qty <= 0,
                "scope": scope,
            }
        )
    rows.sort(key=lambda item: (str(item.get("part_num") or ""), int(item.get("color_id", 0) or 0)))
    return rows


def _training_review_qty_overlay(copied_files: Dict[str, Any], qty_token_boxes: List[Dict[str, Any]], bundle_id: str) -> str:
    original_path = Path(str(copied_files.get("original_crop") or "").strip())
    if not original_path.exists() or not original_path.is_file() or not qty_token_boxes:
        return ""
    image = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if image is None or getattr(image, "size", 0) == 0:
        return ""
    for index, token in enumerate(qty_token_boxes):
        x = int(_coerce_int(token.get("x")) or 0)
        y = int(_coerce_int(token.get("y")) or 0)
        w = int(_coerce_int(token.get("w")) or 0)
        h = int(_coerce_int(token.get("h")) or 0)
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, str(index + 1), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    out_path = original_path.with_name(f"{bundle_id}_qty_overlay.png")
    cv2.imwrite(str(out_path), image)
    return str(out_path)


def _training_review_qty_overlay_html(copied_files: Dict[str, Any], qty_token_boxes: List[Dict[str, Any]]) -> str:
    original_path = str(copied_files.get("original_crop") or "").strip()
    if not original_path:
        return '<div class="missing">no original crop</div>'
    src = f"/debug/ai-snap-artifact?path={_url_quote(original_path)}"
    boxes_html: List[str] = []
    for index, token in enumerate(qty_token_boxes):
        box = token.get("normalized_box") if isinstance(token.get("normalized_box"), list) else [token.get("x"), token.get("y"), token.get("w"), token.get("h")]
        try:
            x, y, w, h = [float(value) for value in list(box)[:4]]
            crop_size = token.get("crop_image_size") if isinstance(token.get("crop_image_size"), list) else []
            crop_w = float(crop_size[0]) if len(crop_size) >= 2 else 0.0
            crop_h = float(crop_size[1]) if len(crop_size) >= 2 else 0.0
        except Exception:
            continue
        if crop_w <= 0 or crop_h <= 0 or w <= 0 or h <= 0:
            continue
        label = f"{index + 1}: {token.get('text') or token.get('value') or ''}"
        raw_box = token.get("raw_box") or [token.get("x"), token.get("y"), token.get("w"), token.get("h")]
        normalized_box = token.get("normalized_box") or [token.get("x"), token.get("y"), token.get("w"), token.get("h")]
        source = str(token.get("coordinate_source") or "unknown")
        raw_polygon = token.get("raw_polygon") if isinstance(token.get("raw_polygon"), list) else []
        title = f"raw={raw_box} raw_polygon={raw_polygon} normalized={normalized_box} source={source}"
        raw_box_values = []
        try:
            raw_box_values = [float(value) for value in list(raw_box)[:4]]
        except Exception:
            raw_box_values = []
        if len(raw_box_values) == 4:
            rx, ry, rw, rh = raw_box_values
            if rw > 0 and rh > 0:
                boxes_html.append(
                    '<div class="qty-box raw" '
                    f'style="left:{(rx / crop_w) * 100:.4f}%;top:{(ry / crop_h) * 100:.4f}%;width:{(rw / crop_w) * 100:.4f}%;height:{(rh / crop_h) * 100:.4f}%;" '
                    f'title="{escape(title)}"></div>'
                )
        boxes_html.append(
            '<div class="qty-box" '
            f'style="left:{(x / crop_w) * 100:.4f}%;top:{(y / crop_h) * 100:.4f}%;width:{(w / crop_w) * 100:.4f}%;height:{(h / crop_h) * 100:.4f}%;" '
            f'title="{escape(title)}"><span>{escape(label)}</span></div>'
        )
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        boxes_html.append(
            '<div class="qty-center" '
            f'style="left:{(cx / crop_w) * 100:.4f}%;top:{(cy / crop_h) * 100:.4f}%;" '
            f'title="{escape(title)}"></div>'
        )
    return (
        '<div class="ocr-overlay-wrap">'
        '<div class="ocr-overlay-stage">'
        f'<img src="{src}" alt="original crop with OCR qty overlay" loading="lazy">'
        f'{"".join(boxes_html)}'
        '</div>'
        '</div>'
    )


@router.get("/debug/training-store/review-ui")
def training_store_review_ui(
    bundle_id: str = Query(""),
    review_status: str = Query("pending"),
    set_num: str = Query(""),
    bag_num: str = Query(""),
    limit: int = Query(50),
):
    try:
        queue = list_review_queue(
            review_status=review_status,
            set_num=set_num,
            bag_num=bag_num,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    rows = [dict(row) for row in list(queue.get("rows") or []) if isinstance(row, dict)]
    selected = next((row for row in rows if str(row.get("bundle_id") or "") == str(bundle_id or "").strip()), None)
    direct_lookup_warning = ""
    if selected is None and bundle_id:
        try:
            selected = dict(get_training_bundle_index_row(bundle_id).get("row") or {})
        except Exception as exc:
            selected = None
            try:
                artifact_debug = _training_bundle_artifact_debug(bundle_id)
            except Exception:
                artifact_debug = {}
            if bool(artifact_debug.get("local_folder_exists")):
                selected = {
                    "bundle_id": str(bundle_id or "").strip(),
                    "manifest_path": str(artifact_debug.get("metadata_path") or ""),
                }
                direct_lookup_warning = (
                    "Postgres row missing for this bundle, but local bundle artifacts exist. "
                    f"local_folder={artifact_debug.get('local_folder')}; lookup_error={type(exc).__name__}"
                )
            else:
                direct_lookup_warning = (
                    "Bundle was requested directly but no Postgres row or local bundle folder was found. "
                    f"lookup_error={type(exc).__name__}"
                )
    if selected is None and rows:
        selected = rows[0]

    selected_bundle_id = str((selected or {}).get("bundle_id") or "")
    metadata = _read_training_review_metadata(selected or {}) if selected else {}
    if selected and metadata:
        source_crop = metadata.get("source_crop") if isinstance(metadata.get("source_crop"), dict) else {}
        selected.setdefault("set_num", str(metadata.get("set_num") or ""))
        selected.setdefault("bag_num", metadata.get("bag"))
        selected.setdefault("page_num", source_crop.get("page"))
        selected.setdefault("step_num", source_crop.get("step"))
        selected.setdefault("crop_num", _coerce_int(str(metadata.get("crop_id") or "").rsplit("_c", 1)[-1]))
    copied_files = metadata.get("copied_files") if isinstance(metadata.get("copied_files"), dict) else {}
    qty_token_boxes = [
        dict(item)
        for item in list(metadata.get("qty_token_boxes") or [])
        if isinstance(item, dict)
    ]
    qty_token_boxes = _normalize_qty_token_boxes_for_bundle(
        qty_token_boxes,
        crop_box=(metadata.get("source_crop") or {}).get("crop_box") if isinstance(metadata.get("source_crop"), dict) else None,
        original_crop_path=copied_files.get("original_crop"),
    )
    direct_qty_ocr_boxes = [
        dict(item)
        for item in list(metadata.get("direct_qty_ocr_boxes") or [])
        if isinstance(item, dict)
    ]
    if not direct_qty_ocr_boxes:
        direct_qty_ocr_boxes = _direct_qty_ocr_boxes_from_crop_image(copied_files.get("original_crop"))
        if direct_qty_ocr_boxes and selected:
            metadata["direct_qty_ocr_boxes"] = direct_qty_ocr_boxes
            try:
                _training_review_metadata_path(selected).write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
            except Exception:
                pass
    slot_cutouts = list(copied_files.get("slot_cutouts") or metadata.get("cutout_paths") or [])
    split_candidate_paths = (selected or {}).get("split_candidate_paths") if isinstance((selected or {}).get("split_candidate_paths"), dict) else {}
    split_candidates = [
        dict(item)
        for item in list(split_candidate_paths.get("candidates") or [])
        if isinstance(item, dict)
    ]
    baseline_slot_candidates = [
        dict(item)
        for item in list(split_candidate_paths.get("baseline_slot_candidates") or [])
        if isinstance(item, dict)
    ]
    ai_suggested_candidates = [
        dict(item)
        for item in list(split_candidate_paths.get("ai_suggested_candidates") or [])
        if isinstance(item, dict)
    ]
    if not baseline_slot_candidates and split_candidates:
        baseline_slot_candidates = [
            item for item in split_candidates if str(item.get("group") or "") == "baseline_slot"
        ]
        ai_suggested_candidates = [
            item for item in split_candidates if str(item.get("group") or "") == "ai_suggested"
        ]
    bundle_match = re.search(r"_p(\d+)_s(\d+)_c(\d+)", selected_bundle_id)
    selected_page_num = (selected or {}).get("page_num") or (bundle_match.group(1) if bundle_match else "")
    selected_step_num = (selected or {}).get("step_num") or (bundle_match.group(2) if bundle_match else "")
    selected_crop_num = (selected or {}).get("crop_num") or (bundle_match.group(3) if bundle_match else "")
    selected_slot_count = (selected or {}).get("slot_count") or len(slot_cutouts)
    queue_links = []
    for row in rows:
        row_bundle_id = str(row.get("bundle_id") or "")
        row_match = re.search(r"_p(\d+)_s(\d+)_c(\d+)", row_bundle_id)
        row_page_num = row.get("page_num") or (row_match.group(1) if row_match else "")
        row_step_num = row.get("step_num") or (row_match.group(2) if row_match else "")
        row_slot_count = row.get("slot_count") or ""
        href = (
            f"/debug/training-store/review-ui?bundle_id={_url_quote(row_bundle_id)}"
            f"&review_status={_url_quote(review_status)}&set_num={_url_quote(set_num)}"
            f"&bag_num={_url_quote(bag_num)}&limit={int(limit or 50)}"
        )
        active = " active" if row_bundle_id == selected_bundle_id else ""
        queue_links.append(
            f'<a class="queue-item{active}" href="{href}">'
            f'<strong>{escape(row_bundle_id)}</strong>'
            f'<span>p{escape(str(row_page_num))} s{escape(str(row_step_num))} slots {escape(str(row_slot_count))}</span>'
            f'</a>'
        )

    slot_html = "".join(
        f'<figure><div class="slot-img">{_training_review_img(path, f"Slot {index + 1}")}</div><figcaption>Slot {index + 1}</figcaption></figure>'
        for index, path in enumerate(slot_cutouts)
    ) or '<div class="missing">no slot cutouts</div>'
    split_overlay_html = (
        f'<figure>{_training_review_img(split_candidate_paths.get("overlay_path"), "split candidate overlay")}<figcaption>split candidate overlay</figcaption></figure>'
        if split_candidate_paths.get("overlay_path")
        else '<div class="missing">no split candidate overlay</div>'
    )
    original_crop_ready = bool(str(copied_files.get("original_crop") or "").strip() and Path(str(copied_files.get("original_crop") or "").strip()).exists())
    raw_master_mask_ready = bool(str(copied_files.get("raw_master_mask") or "").strip() and Path(str(copied_files.get("raw_master_mask") or "").strip()).exists())
    split_action_html = (
        '<div class="split-actions">'
        f'<button type="button" data-generate-split-candidates="true" '
        f'{"disabled" if not (selected_bundle_id and original_crop_ready and raw_master_mask_ready) else ""}>'
        'Generate Split Candidates'
        '</button>'
        f'<span>{escape("ready to generate/regenerate" if selected_bundle_id and original_crop_ready and raw_master_mask_ready else "original crop or raw mask missing")}</span>'
        '</div>'
    )
    old_qty_overlay_html = _training_review_qty_overlay_html(copied_files, qty_token_boxes)
    direct_qty_overlay_html = _training_review_qty_overlay_html(copied_files, direct_qty_ocr_boxes)
    qty_rows_html = "".join(
        (
            f'<tr>'
            f'<td>{escape(str(index + 1))}</td>'
            f'<td>{escape(str(token.get("text") or token.get("value") or ""))}</td>'
            f'<td>{escape(str(token.get("raw_box") or [token.get("x"), token.get("y"), token.get("w"), token.get("h")]))}</td>'
            f'<td>{escape(str(token.get("normalized_box") or [token.get("x"), token.get("y"), token.get("w"), token.get("h")]))}</td>'
            f'<td>{escape(str(token.get("coordinate_source") or ""))}</td>'
            f'<td>{escape(str(token.get("raw_polygon") or ""))}</td>'
            f'<td>{escape(str(token.get("converted_bbox") or token.get("normalized_box") or ""))}</td>'
            f'<td>{escape(str(token.get("confidence") or ""))}</td>'
            f'</tr>'
        )
        for index, token in enumerate(qty_token_boxes)
    ) or '<tr><td colspan="8">No qty token boxes in metadata.</td></tr>'
    direct_qty_rows_html = "".join(
        (
            f'<tr>'
            f'<td>{escape(str(index + 1))}</td>'
            f'<td>{escape(str(token.get("text") or token.get("value") or ""))}</td>'
            f'<td>{escape(str(token.get("raw_box") or [token.get("x"), token.get("y"), token.get("w"), token.get("h")]))}</td>'
            f'<td>{escape(str(token.get("normalized_box") or [token.get("x"), token.get("y"), token.get("w"), token.get("h")]))}</td>'
            f'<td>{escape(str(token.get("coordinate_source") or ""))}</td>'
            f'<td>{escape(str(token.get("raw_polygon") or ""))}</td>'
            f'<td>{escape(str(token.get("converted_bbox") or token.get("normalized_box") or ""))}</td>'
            f'<td>{escape(str(token.get("confidence") or ""))}</td>'
            f'</tr>'
        )
        for index, token in enumerate(direct_qty_ocr_boxes)
    ) or '<tr><td colspan="8">No direct OCR qty boxes.</td></tr>'

    def _candidate_display_index(candidate: Dict[str, Any], fallback: int) -> int:
        parsed = _coerce_int(candidate.get("index"))
        display = _coerce_int(candidate.get("display_index"))
        return int(display if display is not None else int(parsed if parsed is not None else fallback) + 1)

    review_state_labels = {
        "needs_mask_expand": "Needs Mask Expand",
        "needs_ocr_review": "Needs OCR Review",
        "needs_manual_crop": "Needs Manual Crop",
        "mask_amended": "Mask Amended",
    }

    def _manual_refinement_meta(candidate: Dict[str, Any]) -> str:
        if not str(candidate.get("manual_mask_amend_path") or "").strip():
            return ""
        version = str(candidate.get("manual_refinement_version") or "n/a")
        refined_at = str(candidate.get("manual_refined_at") or "")
        refined_text = f"Refinement: v{version}" + (f" · {refined_at}" if refined_at else "")
        return f'<br>{escape(refined_text)}'

    def _rerun_refinement_button(candidate: Dict[str, Any], index: int) -> str:
        if not str(candidate.get("manual_mask_amend_path") or "").strip():
            return ""
        return (
            f'<button type="button" data-rerun-refinement="true" '
            f'data-candidate-index="{escape(str(candidate.get("index", index)))}">Re-run Refinement</button>'
        )

    def _pick_colour_button(candidate: Dict[str, Any], index: int) -> str:
        img_path = _candidate_display_path(candidate)
        if not img_path:
            return ""
        return (
            f'<button type="button" class="clnup-open-btn" '
            f'data-pick-colour="true" '
            f'data-candidate-index="{escape(str(candidate.get("index", index)))}" '
            f'data-candidate-img-path="{escape(img_path)}">Pick Background Colour</button>'
        )

    def _mark_qty_clean_button(candidate: Dict[str, Any], index: int) -> str:
        if not bool(candidate.get("qty_detected")):
            return ""
        if _candidate_qty_is_clean(candidate):
            return ""
        return (
            f'<button type="button" '
            f'data-candidate-index="{escape(str(candidate.get("index", index)))}" '
            f'data-candidate-action="mark_qty_clean" '
            f'title="Mark qty text as already clean — no scrub needed">Mark Qty Clean</button>'
        )

    def _render_candidate_cards(candidates: List[Dict[str, Any]]) -> str:
        return "".join(
            (
                f'<figure class="split-candidate-card {escape("has-review-state" if str(candidate.get("review_state") or "") else "")}">'
                f'<div class="slot-img split-candidate-img">{_training_review_alpha_compare(_candidate_display_path(candidate), "Candidate " + str(_candidate_display_index(candidate, index)))}</div>'
                f'<figcaption><strong>Candidate {escape(str(_candidate_display_index(candidate, index)))}</strong><span>{escape(review_state_labels.get(str(candidate.get("review_state") or ""), str(candidate.get("status") or "pending")))}</span></figcaption>'
                f'<div class="candidate-meta">OCR/qty detected: {escape("yes" if candidate.get("qty_detected") else "no")}<br>Qty value: {escape(", ".join(str(v) for v in list(candidate.get("qty_values") or [])) or "n/a")}<br>Qty scrub: {escape(str(candidate.get("qty_scrub_status") or "not run"))}{_manual_refinement_meta(candidate)}</div>'
                f'<div class="candidate-actions">'
                f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="accept">Accept Clean</button>'
                f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="reject">Reject</button>'
                f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="scrub">Scrub Qty</button>'
                f'{_mark_qty_clean_button(candidate, index)}'
                f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="ai_crop_fix" '
                f'data-current-candidate-path="{escape(str(copied_files.get("original_crop") or ""))}" '
                f'data-current-alpha-path="{escape(str(candidate.get("qty_scrubbed_mask_path") or candidate.get("mask_path") or candidate.get("alpha_path") or ""))}" '
                f'data-reference-candidate-path="{escape(str(candidate.get("qty_scrubbed_path") or candidate.get("thumbnail_path") or candidate.get("candidate_path") or ""))}" '
                f'data-slot-overlay-path="{escape(str(split_candidate_paths.get("overlay_path") or ""))}" '
                f'data-candidate-qty="{escape(str((list(candidate.get("qty_values") or []) or [""])[0]))}">AI Crop Fix</button> ' 
                f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="needs_mask_expand">Needs Mask Expand</button>'
                f'<button type="button" data-amend-mask="true" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-original-path="{escape(str(copied_files.get("original_crop") or ""))}" data-mask-path="{escape(str(candidate.get("mask_path") or ""))}" data-candidate-box="{escape(json.dumps(list(candidate.get("box") or [])))}">Amend Mask</button>'
                f'{_rerun_refinement_button(candidate, index)}'
                f'{_pick_colour_button(candidate, index)}'
                f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="needs_ocr_review">Needs OCR Review</button>'
                f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="needs_manual_crop">Needs Manual Crop</button>'
                f'</div>'
                f'</figure>'
            )
            for index, candidate in enumerate(candidates)
        )

    baseline_candidate_html = _render_candidate_cards(baseline_slot_candidates) or '<div class="missing">no baseline slot candidates generated</div>'
    ai_candidate_html = _render_candidate_cards(ai_suggested_candidates) or '<div class="missing">no AI suggested candidates generated</div>'
    legacy_split_candidate_html = "".join(
        (
            f'<figure>'
            f'<div class="slot-img">{_training_review_alpha_compare(candidate.get("candidate_path"), "Candidate " + str(_candidate_display_index(candidate, index)))}</div>'
            f'<figcaption>Candidate {escape(str(_candidate_display_index(candidate, index)))} · {escape(str(candidate.get("status") or "pending"))}</figcaption>'
            f'<div class="candidate-meta">OCR/qty detected: {escape("yes" if candidate.get("qty_detected") else "no")}<br>Qty value: {escape(", ".join(str(v) for v in list(candidate.get("qty_values") or [])) or "n/a")}{_manual_refinement_meta(candidate)}</div>'
            f'<div class="candidate-actions">'
            f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="accept">Accept Clean</button>'
            f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="reject">Reject</button>'
            f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="scrub">Needs Qty Scrub / Scrub Qty</button>'
            f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="ai_crop_fix" '
            f'data-current-candidate-path="{escape(str(copied_files.get("original_crop") or ""))}" '
            f'data-current-alpha-path="{escape(str(candidate.get("qty_scrubbed_mask_path") or candidate.get("mask_path") or candidate.get("alpha_path") or ""))}" '
            f'data-reference-candidate-path="{escape(str(candidate.get("qty_scrubbed_path") or candidate.get("thumbnail_path") or candidate.get("candidate_path") or ""))}" '
            f'data-slot-overlay-path="{escape(str(split_candidate_paths.get("overlay_path") or ""))}" '
            f'data-candidate-qty="{escape(str((list(candidate.get("qty_values") or []) or [""])[0]))}">AI Crop Fix</button>'
            f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="needs_mask_expand">Needs Mask Expand</button>'
            f'<button type="button" data-amend-mask="true" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-original-path="{escape(str(copied_files.get("original_crop") or ""))}" data-mask-path="{escape(str(candidate.get("mask_path") or ""))}" data-candidate-box="{escape(json.dumps(list(candidate.get("box") or [])))}">Amend Mask</button>'
            f'{_rerun_refinement_button(candidate, index)}'
            f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="needs_ocr_review">Needs OCR Review</button>'
            f'<button type="button" data-candidate-index="{escape(str(candidate.get("index", index)))}" data-candidate-action="needs_manual_crop">Needs Manual Crop</button>'
            f'</div>'
            f'</figure>'
        )
        for index, candidate in enumerate(split_candidates)
    ) if not baseline_slot_candidates and not ai_suggested_candidates else ""
    confirmed_example_rows: List[Dict[str, Any]] = []
    confirmed_examples: Dict[int, Dict[str, Any]] = {}
    if selected_bundle_id:
        try:
            confirmed_example_rows = [
                dict(item)
                for item in list(list_candidate_training_examples(selected_bundle_id).get("rows") or [])
                if isinstance(item, dict)
            ]
            confirmed_examples = {
                int(item.get("candidate_index")): dict(item)
                for item in confirmed_example_rows
                if isinstance(item, dict) and _coerce_int(item.get("candidate_index")) is not None
            }
        except Exception:
            confirmed_example_rows = []
            confirmed_examples = {}
    def _training_review_candidate_confirmed(confirmed: Dict[str, Any]) -> bool:
        return any(
            str(confirmed.get(field) or "").strip()
            for field in ("part_num", "element_id", "confirmed_by")
        )

    clean_accepted_candidates = [
        dict(candidate)
        for candidate in split_candidates
        if isinstance(candidate, dict)
        and (
            str(candidate.get("status") or "") == "accepted"
            or _training_review_candidate_confirmed(
                confirmed_examples.get(
                    int(_coerce_int(candidate.get("index")) if _coerce_int(candidate.get("index")) is not None else -1),
                    {},
                )
            )
        )
        and not str(candidate.get("review_state") or "").strip()
        and _candidate_qty_is_clean(candidate)
    ]
    selected_set_num = str((selected or {}).get("set_num") or (metadata.get("set_num") if isinstance(metadata, dict) else "") or set_num or "")
    selected_bag_num = _coerce_int((selected or {}).get("bag_num") or (metadata.get("bag") if isinstance(metadata, dict) else None))
    if selected_bag_num is None:
        selected_bag_num = _coerce_int((metadata.get("bag_num") if isinstance(metadata, dict) else None))
    if selected_bag_num is None:
        selected_bag_num = _coerce_int(bag_num)
    set_confirmed_totals: Dict[str, int] = {}
    if selected_set_num:
        try:
            totals_rows = list(list_confirmed_part_totals_for_set(selected_set_num).get("rows") or [])
        except Exception:
            totals_rows = []
        for total_row in totals_rows:
            if not isinstance(total_row, dict):
                continue
            part_num = str(total_row.get("part_num") or "").strip()
            color_id = _coerce_int(total_row.get("color_id"))
            if not part_num or color_id is None:
                continue
            set_confirmed_totals[_candidate_part_key(part_num, color_id)] = int(total_row.get("confirmed_qty") or 0)
    full_set_part_rows = _training_review_required_part_rows(
        set_num=selected_set_num,
        bag_num=int(selected_bag_num or 1),
        scope="set",
        confirmed_totals=set_confirmed_totals,
    ) if selected_set_num else []
    full_bag_part_rows = _training_review_required_part_rows(
        set_num=selected_set_num,
        bag_num=int(selected_bag_num or 1),
        scope="bag",
        confirmed_totals=set_confirmed_totals,
    ) if selected_set_num else []
    next_pending_payload: Dict[str, Any] = {}
    next_pending_href = ""
    next_bag_href = ""
    current_bag_complete = False
    next_bag_num = None
    if selected_set_num and selected_bag_num is not None:
        try:
            next_pending_payload = _next_pending_bundle_payload(
                set_num=selected_set_num,
                bag_num=selected_bag_num,
                current_bundle_id=selected_bundle_id,
                limit=int(limit or 50),
            )
            next_pending_href = str(next_pending_payload.get("next_url") or "")
            next_bag_href = str(next_pending_payload.get("next_bag_url") or "")
            current_bag_complete = bool(next_pending_payload.get("current_bag_complete"))
            next_bag_num = next_pending_payload.get("next_bag_num")
        except Exception:
            next_pending_payload = {}
    step_href = ""
    if selected_set_num and selected_bag_num is not None and selected_page_num and selected_step_num:
        step_href = (
            f"/debug/instruction-buildability?set_num={_url_quote(selected_set_num)}"
            f"&bag={int(selected_bag_num)}"
            f"&page={_url_quote(str(selected_page_num))}"
            f"&step={_url_quote(str(selected_step_num))}"
            f"&show_hidden=1"
        )
    step_link_html = (
        f'<a class="view-step-link" href="{escape(step_href)}" target="_blank" rel="noopener">View Step</a>'
        if step_href
        else ""
    )
    def _render_required_part_rows(rows: List[Dict[str, Any]]) -> str:
        rendered: List[str] = []
        for row in rows:
            color_id = str(row.get("color_id") or "")
            completed = bool(row.get("completed"))
            rendered.append(
                f'<tr data-required-part-row="true" data-scope="{escape(str(row.get("scope") or ""))}" '
                f'data-color-id="{escape(color_id)}" data-completed="{escape("1" if completed else "0")}">'
                f'<td><div class="required-part-img">{_training_review_catalog_img(row.get("image_path") or row.get("image_url"), str(row.get("part_num") or ""))}</div></td>'
                f'<td><strong>{escape(str(row.get("part_num") or ""))}</strong></td>'
                f'<td>{escape(color_id)}</td>'
                f'<td>{escape(str(row.get("color_name") or ""))}</td>'
                f'<td>{escape(str(row.get("element_id") or ""))}</td>'
                f'<td>{escape(str(row.get("set_required_qty") or 0))}</td>'
                f'<td>{escape(str(row.get("confirmed_qty") or 0))}</td>'
                f'<td>{escape(str(row.get("remaining_qty") or 0))}</td>'
                f'</tr>'
            )
        return "".join(rendered)

    required_parts_rows_html = _render_required_part_rows(full_bag_part_rows) + _render_required_part_rows(full_set_part_rows)
    if not required_parts_rows_html:
        required_parts_rows_html = '<tr><td colspan="8">No required parts found.</td></tr>'
    required_parts_panel_html = (
        '<section class="required-parts-panel">'
        '<div class="section-heading-row"><h3>Full Bag Parts / Full Set Parts</h3>'
        '<div class="required-part-toggles">'
        '<label><input type="radio" name="required-scope" value="set" checked> Full set</label>'
        '<label style="opacity:0.45" title="Bag estimate is unreliable — hidden from main UI"><input type="radio" name="required-scope" value="bag"> Bag estimate</label>'
        '<label><input type="checkbox" data-required-current-colour="true"> Current colour</label>'
        '<label><input type="checkbox" data-required-show-completed="true"> Show already completed</label>'
        '</div></div>'
        '<details class="inventory-debug-counts"><summary>Debug counts</summary>'
        f'full_set_part_count: <strong>{escape(str(len(full_set_part_rows)))}</strong> · '
        f'full_bag_part_count: <strong>{escape(str(len(full_bag_part_rows)))}</strong> · '
        '<span>rendered_match_count: <strong data-rendered-match-count>0</strong></span> · '
        '<span>hidden_due_to_limit_count: <strong data-hidden-due-limit-count>0</strong></span>'
        '</details>'
        '<div class="required-parts-table-wrap"><table class="required-parts-table">'
        '<thead><tr><th>image</th><th>part_num</th><th>color_id</th><th>color_name</th><th>element_id</th><th>Set qty</th><th>Confirmed qty</th><th>Remaining qty</th></tr></thead>'
        f'<tbody>{required_parts_rows_html}</tbody>'
        '</table></div>'
        '</section>'
    )
    confirm_cards: List[str] = []
    for index, candidate in enumerate(clean_accepted_candidates):
        candidate_index = _coerce_int(candidate.get("index"))
        if candidate_index is None:
            candidate_index = index
        confirmed = confirmed_examples.get(int(candidate_index), {})
        is_confirmed = _training_review_candidate_confirmed(confirmed)
        display_status = "Confirmed - locked" if is_confirmed else str(candidate.get("status") or "pending")
        display_status_class = " saved" if is_confirmed else ""
        disabled_attr = " disabled" if is_confirmed else ""
        locked_class = " locked" if is_confirmed else ""
        qty_values = list(candidate.get("qty_values") or [])
        qty_value = str((qty_values or [""])[0])
        try:
            bag_matches = _training_review_candidate_part_matches(
                row=selected or {},
                candidate=candidate,
                confirmed_rows=confirmed_example_rows,
                limit=0,
                scope="bag",
                confirmed_totals_override=set_confirmed_totals,
            )
        except Exception:
            bag_matches = []
        try:
            set_matches = _training_review_candidate_part_matches(
                row=selected or {},
                candidate=candidate,
                confirmed_rows=confirmed_example_rows,
                limit=0,
                scope="set",
                confirmed_totals_override=set_confirmed_totals,
            )
        except Exception:
            set_matches = []
        matches = bag_matches + [
            match for match in set_matches
            if _candidate_part_key(match.get("part_num"), match.get("color_id"))
            not in {_candidate_part_key(item.get("part_num"), item.get("color_id")) for item in bag_matches}
        ]
        initial_selected_color_id = _coerce_int(confirmed.get("color_id"))
        initial_selected_color_text = str(initial_selected_color_id) if initial_selected_color_id is not None else ""
        initial_filtered_count = (
            sum(1 for match in matches if _coerce_int(match.get("color_id")) == initial_selected_color_id)
            if initial_selected_color_id is not None
            else len(matches)
        )
        server_total_card_count = len(matches)
        server_color_308_count = sum(1 for match in matches if _coerce_int(match.get("color_id")) == 308)
        color_buttons: List[str] = []
        seen_match_colors: set[int] = set()
        for match in matches:
            match_color_id = _coerce_int(match.get("color_id"))
            if match_color_id is None or int(match_color_id) in seen_match_colors:
                continue
            seen_match_colors.add(int(match_color_id))
            rgb_text = str(match.get("color_rgb") or "").strip().replace("#", "")
            swatch_style = f"background: #{escape(rgb_text)};" if re.match(r"^[0-9A-Fa-f]{6}$", rgb_text) else ""
            color_buttons.append(
                f'<button type="button" class="review-colour-chip{escape(" active" if initial_selected_color_id is not None and int(match_color_id) == initial_selected_color_id else "")}" data-review-colour-chip="true"{disabled_attr} '
                f'data-color-id="{escape(str(match_color_id))}" data-color-name="{escape(str(match.get("color_name") or ""))}" '
                f'data-color-rgb="{escape(rgb_text)}">'
                f'<span class="review-colour-swatch" style="{swatch_style}"></span>'
                f'<span>{escape(str(match.get("color_name") or ("color " + str(match_color_id))))} ({escape(str(match_color_id))})</span>'
                f'</button>'
            )
        color_filter_html = (
            '<div class="review-colour-filter">'
            f'<button type="button" class="review-colour-chip{escape(" active" if not initial_selected_color_text else "")}" data-review-colour-clear="true"{disabled_attr}>All colours</button>'
            f'<button type="button" class="review-colour-chip" data-review-colour-pick="true"{disabled_attr}>Pick colour from candidate</button>'
            + "".join(color_buttons)
            + '</div>'
        )
        suggestion_cards: List[str] = []
        for match in matches:
            match_key = _candidate_part_key(match.get("part_num"), match.get("color_id"))
            match_scope = "bag" if any(_candidate_part_key(item.get("part_num"), item.get("color_id")) == match_key for item in bag_matches) else "set"
            match_color_value = match.get("color_id")
            match_color_text = str(match_color_value) if match_color_value is not None else ""
            set_required_qty = int(match.get("set_required_qty", 0) or 0)
            bag_required_qty = int(match.get("bag_required_qty", 0) or 0)
            confirmed_qty = int(match.get("confirmed_qty", 0) or 0)
            remaining_qty = int(match.get("remaining_qty", set_required_qty - confirmed_qty) or 0)
            effective_remaining_qty = max(0, int(match.get("effective_remaining_qty", remaining_qty) or 0))
            over_confirmed = bool(match.get("over_confirmed")) or confirmed_qty > set_required_qty
            over_confirmed_html = '<span class="over-confirmed-badge">over-confirmed</span>' if over_confirmed else ""
            _card_color_match = bool(initial_selected_color_text and match_color_text == initial_selected_color_text)
            print(
                f"[training-review-card] candidate={candidate_index}"
                f" part_num={match.get('part_num')} color_id={match_color_text}"
                f" set_required_qty={set_required_qty} confirmed_qty={confirmed_qty}"
                f" remaining_qty={remaining_qty} rendered=True color_match={_card_color_match}"
            )
            suggestion_cards.append(
                f'<article class="part-suggestion{escape(" colour-match-highlight" if _card_color_match else "")}" '
                f'data-part-color-id="{escape(match_color_text)}" data-match-scope="{escape(match_scope)}">'
                f'<div class="part-suggestion-img">{_training_review_catalog_img(match.get("image_path") or match.get("image_url"), str(match.get("part_num") or ""))}</div>'
                f'<div class="part-suggestion-body">'
                f'<strong>{escape(str(match.get("part_num") or ""))}</strong>'
                f'<span class="part-colour">{escape(str(match.get("color_name") or ("color " + match_color_text)))} ({escape(match_color_text)})</span>'
                f'<span class="part-name">{escape(str(match.get("part_name") or ""))}</span>'
                f'<span class="part-meta">element: {escape(str(match.get("element_id") or "n/a"))}</span>'
                f'<span class="part-meta">Set qty: {escape(str(set_required_qty))}</span>'
                f'<span class="part-meta">Confirmed qty: {escape(str(confirmed_qty))}; Remaining qty: {escape(str(effective_remaining_qty))} {over_confirmed_html}</span>'
                f'<span class="part-meta debug-meta">raw remaining_qty: {escape(str(remaining_qty))}; bag_qty: {escape(str(bag_required_qty))}; scope: {escape(match_scope)}</span>'
                f'<span class="part-score">match {escape(str(match.get("confidence") or ""))}</span>'
                f'<button type="button" class="usage-toggle" data-show-confirmations="true" data-usage-set-num="{escape(selected_set_num)}" '
                f'data-usage-part-num="{escape(str(match.get("part_num") or ""))}" data-usage-color-id="{escape(match_color_text)}" '
                f'data-usage-element-id="{escape(str(match.get("element_id") or ""))}" data-usage-required-qty="{escape(str(set_required_qty))}">Show confirmations</button>'
                f'<button type="button" data-suggest-confirm="true"{disabled_attr} data-suggestion-color-id="{escape(match_color_text)}" '
                f'data-suggestion-color-rgb="{escape(str(match.get("color_rgb") or "").strip().replace("#", ""))}" '
                f'data-confirm-candidate="{escape(str(candidate_index))}" '
                f'data-part-num="{escape(str(match.get("part_num") or ""))}" '
                f'data-color-id="{escape(match_color_text)}" '
                f'data-element-id="{escape(str(match.get("element_id") or ""))}" '
                f'data-candidate-qty="{escape(qty_value)}">Confirm this part</button>'
                f'</div>'
                f'</article>'
            )
        suggestion_html = "".join(suggestion_cards)
        no_color_matches_html = '<div class="missing hidden" data-colour-empty-message="true"></div>'
        if not suggestion_html:
            suggestion_html = '<div class="missing">No required-part matches available.</div>'
        match_debug_html = (
            f'<details class="match-debug" data-match-debug="true" '
            f'data-total-matches="{escape(str(len(matches)))}" '
            f'data-full-set-part-count="{escape(str(len(full_set_part_rows)))}" '
            f'data-full-bag-part-count="{escape(str(len(full_bag_part_rows)))}" '
            f'data-server-total-card-count="{escape(str(server_total_card_count))}" '
            f'data-server-color-308-count="{escape(str(server_color_308_count))}" '
            f'data-hidden-due-to-limit-count="0" '
            f'data-selected-color-id="{escape(initial_selected_color_text)}" '
            f'data-filtered-match-count="{escape(str(initial_filtered_count))}">'
            f'<summary>Debug counts</summary>'
            f'full_set_part_count: {escape(str(len(full_set_part_rows)))} · '
            f'full_bag_part_count: {escape(str(len(full_bag_part_rows)))} · '
            f'rendered_match_count: <span data-rendered-match-count-local>{escape(str(server_total_card_count))}</span> · '
            f'hidden_due_to_limit_count: 0 · '
            f'selected_color_id: <span data-debug-selected-color>{escape(initial_selected_color_text or "all")}</span> · '
            f'colour_match_count: <span data-debug-filtered-count>{escape(str(initial_filtered_count))}</span> · '
            f'total_cards_rendered_server_side: <span data-server-total-card-count-text>{escape(str(server_total_card_count))}</span> · '
            f'total_cards_in_DOM: <span data-dom-total-card-count>{escape(str(server_total_card_count))}</span> · '
            f'cards_with_data_part_color_id_308: <span data-dom-color-308-count>{escape(str(server_color_308_count))}</span> · '
            f'visible_cards_after_clicking_color_308: <span data-dom-visible-after-308-count>{escape(str(server_color_308_count))}</span>'
            f'</details>'
        )
        unconfirm_html = (
            f'<button type="button" class="unconfirm-btn" data-unconfirm-candidate="{escape(str(candidate_index))}">Unconfirm part</button>'
            if is_confirmed
            else ""
        )
        confirm_cards.append(
            f'<article class="confirm-card annotation-workstation{locked_class}">'
            f'<div class="confirm-card-head">'
            f'<div><strong>Candidate {escape(str(_candidate_display_index(candidate, index)))}</strong>'
            f'<span>qty {escape(", ".join(str(v) for v in qty_values) or "n/a")}</span></div>'
            f'<span class="review-state{display_status_class}">{escape(display_status)}</span>'
            f'<span class="review-state {escape("saved" if is_confirmed else "unsaved")}">{escape("confirmed" if is_confirmed else "not confirmed")}</span>'
            f'{step_link_html}'
            f'</div>'
            f'<div class="candidate-workspace">'
            f'<div class="candidate-preview-panel">'
            f'<div class="confirm-thumb" data-confirm-thumb="true">{_training_review_alpha_compare(candidate.get("qty_scrubbed_path") or candidate.get("thumbnail_path") or candidate.get("candidate_path"), "Candidate " + str(_candidate_display_index(candidate, index)))}</div>'
            f'<div class="candidate-preview-meta">'
            f'<span><strong>Qty</strong> {escape(", ".join(str(v) for v in qty_values) or "n/a")}</span>'
            f'<span><strong>State</strong> {escape(display_status)}</span>'
            f'<span><strong>Scrub</strong> {escape(str(candidate.get("qty_scrub_status") or "not run"))}</span>'
            f'</div>'
            f'<label><span class="label">part_num</span><input data-confirm-field="part_num" value="{escape(str(confirmed.get("part_num") or ""))}"{disabled_attr}></label>'
            f'<label><span class="label">color_id</span><input data-confirm-field="color_id" type="number" value="{escape(str(confirmed.get("color_id") or ""))}"{disabled_attr}></label>'
            f'<label><span class="label">element_id</span><input data-confirm-field="element_id" value="{escape(str(confirmed.get("element_id") or ""))}"{disabled_attr}></label>'
            f'<label><span class="label">confirmed_by</span><input data-confirm-field="confirmed_by" value="{escape(str(confirmed.get("confirmed_by") or "andy"))}"{disabled_attr}></label>'
            f'<button type="button" class="manual-confirm-btn" data-confirm-candidate="{escape(str(candidate_index))}" data-candidate-qty="{escape(qty_value)}"{disabled_attr}>Confirm manual fields</button>'
            f'{unconfirm_html}'
            f'</div>'
            f'<div class="candidate-match-panel">'
            f'{color_filter_html}'
            f'{match_debug_html}'
            f'<div class="part-suggestions" aria-label="Suggested LEGO parts">{suggestion_html}{no_color_matches_html}</div>'
            f'</div>'
            f'</div>'
            f'</article>'
        )
    confirm_candidate_html = "".join(confirm_cards) or '<div class="missing">No clean accepted candidates ready for part confirmation.</div>'
    title = escape(selected_bundle_id or "Training Review")
    direct_lookup_warning_html = (
        f'<div class="missing">{escape(direct_lookup_warning)}</div>'
        if direct_lookup_warning
        else ""
    )
    next_pending_button_html = (
        f'<a class="top-action-link" data-next-pending="true" href="{escape(next_pending_href)}">Next pending</a>'
        if next_pending_href
        else '<button type="button" class="top-action-link" data-next-pending="true">Next pending</button>'
    )
    next_bag_button_html = (
        f'<a class="top-action-link next-bag-link" data-next-bag="true" href="{escape(next_bag_href)}">Open Next Bag</a>'
        if current_bag_complete and next_bag_href
        else '<button type="button" class="top-action-link next-bag-link hidden" data-next-bag="true">Open Next Bag</button>'
    )
    bag_audit_href = (
        f"/debug/training-store/bag-completion-audit?set_num={_url_quote(selected_set_num)}&bag_num={int(selected_bag_num or 1)}"
        if selected_set_num and selected_bag_num is not None
        else ""
    )
    bag_audit_link_html = (
        f'<a class="top-action-link" href="{escape(bag_audit_href)}" target="_blank" rel="noopener">Open Bag Audit</a>'
        if bag_audit_href
        else ""
    )
    bag_complete_html = (
        '<div class="bag-complete-banner" data-bag-complete-banner="true">'
        f'Bag complete{escape(" - next bag " + str(next_bag_num) + " is available" if next_bag_num else "")}.'
        f'{(" " + bag_audit_link_html) if bag_audit_link_html else ""}'
        '</div>'
        if current_bag_complete
        else '<div class="bag-complete-banner hidden" data-bag-complete-banner="true">Bag complete.</div>'
    )
    bag_audit_href_js = json.dumps(bag_audit_href)
    next_pending_url_js = json.dumps(next_pending_href)
    next_pending_endpoint_js = json.dumps(
        f"/debug/training-store/next-pending-bundle?set_num={_url_quote(selected_set_num)}"
        f"&bag_num={_url_quote(str(selected_bag_num or ''))}"
        f"&current_bundle_id={_url_quote(selected_bundle_id)}"
    )
    next_bag_url_js = json.dumps(next_bag_href)
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Training Review</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#172026; background:#f5f7f8; }}
    header {{ padding:14px 18px; background:#18212b; color:white; display:flex; justify-content:space-between; gap:16px; align-items:center; }}
    main {{ display:grid; grid-template-columns:320px 1fr; min-height:calc(100vh - 54px); }}
    aside {{ border-right:1px solid #d8dee3; background:white; overflow:auto; padding:12px; }}
    .queue-item {{ display:block; padding:10px; border-radius:6px; color:#172026; text-decoration:none; border:1px solid transparent; }}
    .queue-item span {{ display:block; color:#66727d; font-size:12px; margin-top:3px; }}
    .queue-item.active {{ border-color:#2f7dd1; background:#eef6ff; }}
    section {{ padding:16px; }}
    .title-row {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; }}
    .title-row h2 {{ margin:0; min-width:0; overflow-wrap:anywhere; }}
    .top-actions {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
    .top-action-link {{ display:inline-flex; align-items:center; justify-content:center; min-height:36px; box-sizing:border-box; padding:8px 12px; border:1px solid #1f6fc9; border-radius:6px; background:#1f6fc9; color:white; font:inherit; font-weight:700; text-decoration:none; cursor:pointer; white-space:nowrap; }}
    .next-bag-link {{ background:#2f6c41; border-color:#2f6c41; }}
    .bag-complete-banner {{ margin:0 0 12px; padding:10px 12px; border:1px solid #7db28a; border-radius:6px; background:#eef6ef; color:#2f6c41; font-weight:700; }}
    .bag-complete-banner a {{ color:#2f6c41; margin-left:8px; }}
    .meta {{ display:grid; grid-template-columns:repeat(5,minmax(90px,1fr)); gap:8px; margin-bottom:14px; }}
    .meta div {{ background:white; border:1px solid #d8dee3; border-radius:6px; padding:8px; }}
    .label {{ display:block; color:#66727d; font-size:12px; }}
    .images {{ display:grid; grid-template-columns:repeat(3,minmax(180px,1fr)); gap:12px; margin-bottom:14px; }}
    figure {{ margin:0; background:white; border:1px solid #d8dee3; border-radius:6px; padding:8px; }}
    figcaption {{ color:#66727d; font-size:12px; margin-top:6px; }}
    img {{ max-width:100%; height:auto; image-rendering:auto; background:repeating-conic-gradient(#f0f2f3 0 25%, #fff 0 50%) 50% / 20px 20px; }}
    .slots {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(120px,1fr)); gap:10px; }}
    .section-heading-row {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:16px; }}
    .section-heading-row h3 {{ margin:0; }}
    .split-actions {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
    .split-actions button {{ padding:8px 12px; font-size:13px; font-weight:700; }}
    .split-actions span {{ color:#66727d; font-size:12px; }}
    .split-candidates {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:12px; margin-top:10px; align-items:start; }}
    .split-candidate-card {{ display:grid; gap:9px; padding:10px; }}
    .split-candidate-card.has-review-state {{ border-color:#d59a20; background:#fff8e8; }}
    .split-candidate-card figcaption {{ display:flex; align-items:center; justify-content:space-between; gap:8px; }}
    .split-candidate-card figcaption strong {{ color:#172026; font-size:14px; }}
    .split-candidate-card.has-review-state figcaption span {{ color:#8a5a00; font-weight:700; }}
    .split-candidate-img {{ min-height:132px; display:flex; align-items:center; justify-content:center; border:1px solid #e4e9ed; border-radius:6px; background:repeating-conic-gradient(#f0f2f3 0 25%, #fff 0 50%) 50% / 20px 20px; }}
    .split-candidate-img img {{ max-height:118px; max-width:150px; object-fit:contain; background:transparent; }}
    .alpha-preview-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; width:100%; }}
    .alpha-preview {{ min-height:96px; display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden; border:1px solid #d8dee3; border-radius:6px; }}
    .alpha-preview.checker {{ background:repeating-conic-gradient(#d9dee3 0 25%, #fff 0 50%) 50% / 18px 18px; }}
    .alpha-preview.black {{ background:#050505; }}
    .alpha-preview.white {{ background:#fff; }}
    .alpha-preview img {{ max-width:100%; max-height:108px; object-fit:contain; background:transparent; }}
    .alpha-preview span {{ position:absolute; left:4px; bottom:4px; padding:2px 4px; border-radius:4px; background:rgba(15,24,33,.72); color:white; font-size:10px; line-height:1; }}
    .candidate-actions {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; margin-top:8px; }}
    .candidate-actions button {{ padding:8px 9px; font-size:12px; min-width:0; white-space:normal; line-height:1.15; }}
    .candidate-actions button[data-candidate-action="scrub"] {{ grid-column:1 / -1; }}
    .candidate-actions button[data-candidate-action="mark_qty_clean"] {{ background:#f0fff4; border-color:#3a9a60; color:#1a5c38; }}
    .candidate-actions button[data-candidate-action="ai_crop_fix"] {{ grid-column:1 / -1; background:#eef6ff; border-color:#77a9df; color:#164f87; }}
    .candidate-actions button[data-candidate-action^="needs_"] {{ background:#fff8e8; border-color:#d59a20; color:#755000; }}
    .candidate-meta {{ color:#66727d; font-size:12px; line-height:1.35; margin-top:6px; }}
    .confirm-grid {{ display:grid; grid-template-columns:1fr; gap:18px; margin-top:10px; }}
    .confirm-card {{ background:white; border:1px solid #d8dee3; border-radius:8px; padding:14px; overflow:hidden; box-shadow:0 1px 2px rgba(31,45,61,.04); }}
    .confirm-card.locked {{ border-color:#7db28a; background:#f7fbf7; }}
    .confirm-card-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; border-bottom:1px solid #e4e9ed; padding-bottom:10px; margin-bottom:12px; }}
    .confirm-card-head strong {{ display:block; font-size:18px; line-height:1.15; }}
    .confirm-card-head span {{ color:#66727d; font-size:13px; }}
    .review-state {{ display:inline-flex; align-items:center; justify-content:center; min-height:28px; border:1px solid #cbd6e2; border-radius:999px; padding:4px 10px; color:#32465a; background:#f8fafb; font-size:12px; white-space:nowrap; }}
    .review-state.saved {{ border-color:#7db28a; background:#eef6ef; color:#2f6c41; }}
    .review-state.unsaved {{ border-color:#e1c36a; background:#fff8df; color:#725b10; }}
    .view-step-link {{ display:inline-flex; align-items:center; justify-content:center; min-height:30px; border:1px solid #1f6fc9; border-radius:6px; padding:5px 11px; color:#1f6fc9; background:#eef6ff; text-decoration:none; font-size:13px; font-weight:700; white-space:nowrap; }}
    .view-step-link:hover {{ background:#dcecff; }}
    .candidate-workspace {{ display:grid; grid-template-columns:minmax(260px,320px) minmax(0,1fr); gap:16px; align-items:start; }}
    .candidate-preview-panel {{ display:grid; gap:10px; align-content:start; }}
    .confirm-thumb {{ min-height:240px; display:flex; align-items:center; justify-content:center; border:1px solid #e4e9ed; border-radius:8px; background:repeating-conic-gradient(#f0f2f3 0 25%, #fff 0 50%) 50% / 20px 20px; }}
    .confirm-thumb.picker-active {{ outline:2px solid #cf1f1f; cursor:crosshair; }}
    .confirm-thumb.picker-active img {{ cursor:crosshair; }}
    .confirm-thumb img {{ max-height:210px; max-width:230px; object-fit:contain; background:transparent; }}
    .confirm-thumb .alpha-preview-grid {{ max-width:100%; }}
    .confirm-thumb .alpha-preview {{ min-height:190px; }}
    .confirm-thumb .alpha-preview img {{ max-height:178px; max-width:100%; }}
    .candidate-preview-meta {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }}
    .candidate-preview-meta span {{ border:1px solid #e4e9ed; border-radius:6px; padding:8px; color:#66727d; font-size:12px; background:#f8fafb; }}
    .candidate-preview-meta strong {{ display:block; color:#172026; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
    .candidate-preview-panel label {{ display:block; }}
    .candidate-preview-panel input {{ width:100%; box-sizing:border-box; }}
    .manual-confirm-btn {{ width:100%; }}
    .unconfirm-btn {{ width:100%; background:#f8fafb; color:#2f6c41; border-color:#7db28a; }}
    button:disabled, input:disabled {{ opacity:.58; cursor:not-allowed; }}
    .candidate-match-panel {{ min-width:0; }}
    .review-colour-filter {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; padding:10px; border:1px solid #e4e9ed; border-radius:8px; background:#f8fafb; }}
    .review-colour-chip {{ display:inline-flex; align-items:center; gap:7px; border:1px solid #cbd6e2; border-radius:999px; background:#fff; color:#32465a; padding:8px 12px; font-size:12px; line-height:1.2; cursor:pointer; max-width:100%; }}
    .review-colour-chip.active {{ border-color:#cf1f1f; background:#fff1f1; color:#7d1d1d; box-shadow:0 0 0 2px rgba(207,31,31,.08); }}
    .review-colour-swatch {{ width:14px; height:14px; border-radius:999px; border:1px solid rgba(0,0,0,.2); flex:0 0 14px; }}
    .part-suggestions {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:12px; max-height:640px; overflow:auto; padding:2px 6px 2px 2px; align-content:start; }}
    .part-suggestion {{ display:grid; grid-template-rows:128px auto; gap:10px; border:1px solid #d8dee3; border-radius:8px; padding:10px; background:#fff; min-width:0; }}
    .part-suggestion.colour-match-highlight {{ border-color:#1f6fc9; box-shadow:0 0 0 2px rgba(31,111,201,.12); background:#f7fbff; }}
    .part-suggestion-img {{ display:flex; align-items:center; justify-content:center; background:#f4f7fb; border:1px solid #e4e9ed; border-radius:6px; overflow:hidden; }}
    .part-suggestion-img img {{ max-height:116px; max-width:140px; object-fit:contain; background:transparent; }}
    .part-suggestion-body {{ display:grid; gap:4px; font-size:12px; color:#66727d; min-width:0; }}
    .part-suggestion-body strong {{ color:#172026; font-size:16px; line-height:1.15; }}
    .part-colour {{ color:#32465a; font-weight:600; }}
    .part-name, .part-meta, .part-score {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .part-meta.debug-meta {{ color:#8a6a28; }}
    .over-confirmed-badge {{ display:inline-flex; align-items:center; margin-left:5px; padding:2px 5px; border-radius:999px; background:#fff1d2; color:#8a5a00; font-size:11px; font-weight:700; }}
    .part-score {{ color:#1f6fc9; font-weight:700; }}
    .part-suggestion-body button {{ margin-top:6px; padding:8px 10px; font-size:13px; width:100%; }}
    .usage-toggle {{ background:#f8fafb; color:#1f6fc9; border-color:#b9cee7; }}
    .usage-modal-backdrop {{ position:fixed; inset:0; z-index:80; display:none; align-items:center; justify-content:center; padding:28px; background:rgba(15,24,33,.58); }}
    .usage-modal-backdrop.open {{ display:flex; }}
    .usage-modal {{ width:min(1120px, calc(100vw - 56px)); max-height:calc(100vh - 56px); display:grid; grid-template-rows:auto auto minmax(0,1fr); overflow:hidden; background:white; border-radius:8px; box-shadow:0 18px 48px rgba(15,24,33,.32); }}
    .usage-modal-head {{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 16px; border-bottom:1px solid #d8dee3; }}
    .usage-modal-head h3 {{ margin:0; font-size:18px; overflow-wrap:anywhere; }}
    .usage-modal-close {{ width:auto; min-width:88px; background:#f8fafb; color:#32465a; border-color:#c8d0d7; }}
    .usage-summary {{ display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:10px; padding:12px 16px; border-bottom:1px solid #e4e9ed; background:#f8fafb; }}
    .usage-summary span {{ display:grid; gap:3px; padding:9px; border:1px solid #e4e9ed; border-radius:6px; background:white; color:#66727d; font-size:12px; }}
    .usage-summary strong {{ color:#172026; font-size:18px; }}
    .usage-table-wrap {{ overflow:auto; padding:0 16px 16px; }}
    .usage-table {{ width:100%; border-collapse:collapse; table-layout:fixed; background:white; }}
    .usage-table th, .usage-table td {{ padding:10px 9px; border-bottom:1px solid #e4e9ed; text-align:left; vertical-align:middle; font-size:13px; }}
    .usage-table th {{ position:sticky; top:0; z-index:1; background:white; color:#66727d; font-size:12px; }}
    .usage-table th:nth-child(1), .usage-table td:nth-child(1) {{ width:86px; }}
    .usage-table th:nth-child(2), .usage-table td:nth-child(2) {{ width:44%; }}
    .usage-table a {{ color:#1f6fc9; font-weight:700; overflow-wrap:anywhere; }}
    .usage-thumb {{ width:58px; height:58px; display:flex; align-items:center; justify-content:center; border:1px solid #e4e9ed; border-radius:6px; background:#f8fafb; overflow:hidden; }}
    .usage-thumb img {{ max-width:56px; max-height:56px; object-fit:contain; background:transparent; }}
    .usage-empty {{ padding:18px; color:#66727d; }}
    .mask-amend-backdrop {{ position:fixed; inset:0; z-index:90; display:none; align-items:center; justify-content:center; padding:18px; background:rgba(15,24,33,.62); }}
    .mask-amend-backdrop.open {{ display:flex; }}
    .mask-amend-modal {{ width:min(1180px, calc(100vw - 36px)); max-height:calc(100vh - 36px); display:grid; grid-template-rows:auto auto minmax(0,1fr); overflow:hidden; background:white; border-radius:8px; box-shadow:0 18px 48px rgba(15,24,33,.32); }}
    .mask-amend-head, .mask-amend-tools {{ display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; padding:12px 14px; border-bottom:1px solid #d8dee3; }}
    .mask-amend-head h3 {{ margin:0; font-size:18px; }}
    .mask-amend-tools label {{ display:inline-flex; align-items:center; gap:6px; color:#32465a; font-size:13px; }}
    .mask-tool.active {{ background:#2f6c41; border-color:#2f6c41; }}
    .mask-amend-stage-wrap {{ overflow:auto; padding:14px; background:#f4f7fb; }}
    .mask-amend-stage {{ position:relative; display:inline-block; line-height:0; background:#111; }}
    .mask-amend-stage canvas {{ display:block; max-width:min(100%, 1080px); height:auto; }}
    .mask-amend-layer {{ position:absolute; inset:0; }}
    .mask-amend-close {{ background:#f8fafb; color:#32465a; border-color:#c8d0d7; }}
    /* ── colour cleanup picker modal ──────────────────────────────────────── */
    .clnup-backdrop {{ position:fixed; inset:0; z-index:95; display:none; align-items:center; justify-content:center; padding:18px; background:rgba(15,24,33,.70); }}
    .clnup-backdrop.open {{ display:flex; }}
    .clnup-modal {{ width:min(1100px, calc(100vw - 36px)); max-height:calc(100vh - 36px); display:grid; grid-template-rows:auto 1fr auto; overflow:hidden; background:#1a2230; color:#e8edf2; border-radius:8px; box-shadow:0 20px 56px rgba(0,0,0,.6); }}
    .clnup-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; padding:12px 16px; border-bottom:1px solid #2e3d50; }}
    .clnup-head h3 {{ margin:0; font-size:17px; font-weight:700; }}
    .clnup-body {{ display:grid; grid-template-columns:1fr 240px; min-height:0; overflow:hidden; }}
    .clnup-stage-wrap {{ overflow:auto; padding:14px; background:#0f161e; position:relative; cursor:crosshair; }}
    .clnup-stage {{ position:relative; display:inline-block; line-height:0; }}
    .clnup-canvas {{ display:block; image-rendering:pixelated; }}
    .clnup-overlay {{ position:absolute; inset:0; pointer-events:none; image-rendering:pixelated; }}
    .clnup-crosshair {{ position:absolute; pointer-events:none; }}
    .clnup-crosshair::before, .clnup-crosshair::after {{ content:""; position:absolute; background:rgba(255,80,80,.85); }}
    .clnup-crosshair::before {{ width:1px; height:18px; left:50%; top:50%; transform:translate(-50%,-50%); }}
    .clnup-crosshair::after {{ height:1px; width:18px; top:50%; left:50%; transform:translate(-50%,-50%); }}
    .clnup-sidebar {{ display:flex; flex-direction:column; gap:12px; padding:14px; border-left:1px solid #2e3d50; overflow-y:auto; background:#182030; }}
    .clnup-mag-wrap {{ display:flex; flex-direction:column; gap:4px; }}
    .clnup-mag-label {{ font-size:11px; color:#8899aa; text-transform:uppercase; letter-spacing:.05em; }}
    .clnup-magnifier {{ width:210px; height:210px; border:1px solid #2e3d50; border-radius:4px; background:#0f161e; image-rendering:pixelated; display:block; }}
    .clnup-pixel-info {{ background:#111b27; border:1px solid #2e3d50; border-radius:4px; padding:8px; font-size:12px; font-family:monospace; min-height:52px; }}
    .clnup-swatch-row {{ display:flex; align-items:center; gap:8px; }}
    .clnup-swatch {{ width:28px; height:28px; border-radius:4px; border:2px solid #3a4f65; flex-shrink:0; }}
    .clnup-swatch-none {{ background:repeating-conic-gradient(#2a3848 0 25%, #1a2638 0 50%) 50% / 10px 10px; }}
    .clnup-swatch-label {{ font-size:12px; color:#8899aa; }}
    .clnup-controls {{ display:flex; flex-direction:column; gap:10px; }}
    .clnup-controls label {{ font-size:13px; color:#c8d8e8; display:flex; flex-direction:column; gap:4px; }}
    .clnup-controls input[type=range] {{ width:100%; }}
    .clnup-tol-row {{ display:flex; align-items:center; gap:6px; }}
    .clnup-tol-val {{ font-family:monospace; font-size:13px; min-width:28px; }}
    .clnup-btn {{ padding:8px 12px; border-radius:6px; border:1px solid #3a4f65; background:#253245; color:#d0e0f0; cursor:pointer; font:inherit; font-size:13px; }}
    .clnup-btn:hover {{ background:#2e3d55; }}
    .clnup-btn-primary {{ background:#1a5fa8; border-color:#1a5fa8; color:white; font-weight:700; }}
    .clnup-btn-primary:hover {{ background:#1869ba; }}
    .clnup-btn-close {{ background:#1a2230; color:#8899aa; border-color:#2e3d50; }}
    .clnup-status {{ font-size:12px; color:#8899aa; min-height:18px; }}
    .required-parts-panel {{ padding:0; margin:18px 0; }}
    .required-part-toggles {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; font-size:13px; }}
    .required-part-toggles label {{ display:inline-flex; align-items:center; gap:5px; padding:6px 8px; border:1px solid #d8dee3; border-radius:6px; background:white; }}
    .inventory-debug-counts {{ margin:8px 0; font-size:12px; }}
    .inventory-debug-counts > summary {{ cursor:pointer; color:#66727d; user-select:none; display:inline; }}
    .inventory-debug-counts > summary::marker {{ font-size:10px; }}
    .match-debug {{ font-size:11px; color:#888; margin:4px 0 6px; }}
    .match-debug > summary {{ cursor:pointer; color:#a0a8b0; user-select:none; }}
    .match-debug > summary::marker {{ font-size:9px; }}
    .required-parts-table-wrap {{ max-height:420px; overflow:auto; border:1px solid #d8dee3; border-radius:6px; background:white; }}
    .required-parts-table {{ width:100%; border-collapse:collapse; table-layout:fixed; }}
    .required-parts-table th, .required-parts-table td {{ padding:8px; border-bottom:1px solid #e4e9ed; text-align:left; vertical-align:middle; font-size:13px; }}
    .required-parts-table th {{ position:sticky; top:0; z-index:1; background:#f8fafb; color:#66727d; font-size:12px; }}
    .required-parts-table th:nth-child(1), .required-parts-table td:nth-child(1) {{ width:72px; }}
    .required-parts-table th:nth-child(2), .required-parts-table td:nth-child(2) {{ width:110px; }}
    .required-parts-table th:nth-child(5), .required-parts-table td:nth-child(5) {{ width:110px; }}
    .required-parts-table tr.hidden {{ display:none; }}
    .required-parts-table tr.colour-match-highlight td {{ background:#f7fbff; }}
    .required-part-img {{ width:52px; height:52px; display:flex; align-items:center; justify-content:center; border:1px solid #e4e9ed; border-radius:6px; background:#f8fafb; overflow:hidden; }}
    .required-part-img img {{ max-width:50px; max-height:50px; object-fit:contain; background:transparent; }}
    .ocr-table {{ width:100%; border-collapse:collapse; background:white; border:1px solid #d8dee3; border-radius:6px; overflow:hidden; margin:8px 0 14px; }}
    .ocr-table th, .ocr-table td {{ text-align:left; border-bottom:1px solid #e4e9ed; padding:7px 8px; font-size:13px; }}
    .ocr-table th {{ color:#66727d; background:#f8fafb; }}
    .ocr-overlay-wrap {{ display:inline-block; max-width:100%; background:white; border:1px solid #d8dee3; border-radius:6px; padding:0; overflow:visible; }}
    .ocr-overlay-stage {{ position:relative; display:inline-block; max-width:100%; line-height:0; }}
    .ocr-overlay-stage img {{ display:block; width:100%; height:auto; }}
    .qty-box {{ position:absolute; border:2px solid #e02020; box-sizing:border-box; pointer-events:auto; z-index:2; }}
    .qty-box.raw {{ border:2px dashed #19a552; z-index:1; }}
    .qty-box span {{ position:absolute; left:0; top:-18px; background:#e02020; color:white; font-size:11px; line-height:1; padding:3px 4px; border-radius:3px; white-space:nowrap; }}
    .qty-center {{ position:absolute; width:7px; height:7px; margin-left:-3.5px; margin-top:-3.5px; border-radius:999px; background:#1267ff; box-shadow:0 0 0 1px white; z-index:3; }}
    form {{ background:white; border:1px solid #d8dee3; border-radius:6px; padding:12px; margin-top:14px; display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:10px; align-items:end; }}
    input, select, textarea, button {{ font:inherit; padding:8px; border:1px solid #c8d0d7; border-radius:6px; }}
    textarea {{ grid-column:span 2; min-height:42px; }}
    button {{ background:#1f6fc9; color:white; border-color:#1f6fc9; cursor:pointer; }}
    .save-next-btn {{ background:#155da8; border-color:#155da8; font-weight:700; }}
    .missing {{ padding:18px; color:#66727d; background:#f3f5f6; border-radius:6px; }}
    .hidden {{ display:none !important; }}
    @media (max-width: 1180px) {{
      .candidate-workspace {{ grid-template-columns:1fr; }}
      .confirm-thumb {{ min-height:220px; }}
    }}
    @media (max-width: 760px) {{
      main {{ grid-template-columns:1fr; }}
      aside {{ display:none; }}
      .confirm-card-head {{ align-items:flex-start; flex-direction:column; }}
      .confirm-thumb {{ min-height:180px; }}
      .confirm-thumb img {{ max-width:170px; max-height:160px; }}
      .candidate-preview-meta {{ grid-template-columns:1fr; }}
      .part-suggestions {{ grid-template-columns:repeat(auto-fill,minmax(145px,1fr)); max-height:520px; }}
      .part-suggestion {{ grid-template-rows:104px auto; }}
      .meta, .images {{ grid-template-columns:1fr; }}
      .usage-modal-backdrop {{ padding:12px; align-items:stretch; }}
      .usage-modal {{ width:100%; max-height:calc(100vh - 24px); }}
      .usage-summary {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .usage-table {{ min-width:760px; }}
    }}
  </style>
</head>
<body>
  <header><strong>Training Review Queue</strong><span>{escape(str(queue.get("count", 0)))} queued</span></header>
  <main>
    <aside>{''.join(queue_links) or '<div class="missing">No bundles match the current filters.</div>'}</aside>
    <section>
      <div class="title-row">
        <h2>{title}</h2>
        <div class="top-actions">{next_pending_button_html}{next_bag_button_html}</div>
      </div>
      {bag_complete_html}
      {direct_lookup_warning_html}
      <div class="meta">
        <div><span class="label">status</span>{escape(str((selected or {}).get("review_status") or ""))}</div>
        <div><span class="label">set</span>{escape(str((selected or {}).get("set_num") or ""))}</div>
        <div><span class="label">bag</span>{escape(str((selected or {}).get("bag_num") or ""))}</div>
        <div><span class="label">page/step</span>{escape(str(selected_page_num))} / {escape(str(selected_step_num))}</div>
        <div><span class="label">crop</span>{escape(str(selected_crop_num))}</div>
        <div><span class="label">slots</span>{escape(str(selected_slot_count))}</div>
      </div>
      <div class="images">
        <figure>{_training_review_img(copied_files.get("original_crop"), "original crop")}<figcaption>original crop</figcaption></figure>
        <figure>{_training_review_img(copied_files.get("full_mask_overlay"), "full mask overlay")}<figcaption>overlay</figcaption></figure>
        <figure>{_training_review_img(copied_files.get("raw_master_mask"), "raw master mask")}<figcaption>mask</figcaption></figure>
      </div>
      <h3>OCR / Qty Tokens</h3>
      <h4>Direct Qty OCR Boxes (source of truth)</h4>
      <div>{direct_qty_overlay_html}</div>
      <table class="ocr-table"><thead><tr><th>#</th><th>text/value</th><th>raw bbox</th><th>normalized bbox</th><th>source</th><th>raw polygon</th><th>converted bbox</th><th>confidence</th></tr></thead><tbody>{direct_qty_rows_html}</tbody></table>
      <h4>Old Qty Token Boxes</h4>
      <div>{old_qty_overlay_html}</div>
      <table class="ocr-table"><thead><tr><th>#</th><th>text/value</th><th>raw bbox</th><th>normalized bbox</th><th>source</th><th>raw polygon</th><th>converted bbox</th><th>confidence</th></tr></thead><tbody>{qty_rows_html}</tbody></table>
      <div class="slots">{slot_html}</div>
      <div class="section-heading-row"><h3>Split Candidates</h3>{split_action_html}</div>
      <div class="images">{split_overlay_html}</div>
      <h4>Baseline Slot Candidates</h4>
      <div class="split-candidates">{baseline_candidate_html}</div>
      <h4>AI Suggested Candidates</h4>
      <div class="split-candidates">{ai_candidate_html}</div>
      {f'<h4>Legacy Candidates</h4><div class="split-candidates">{legacy_split_candidate_html}</div>' if legacy_split_candidate_html else ''}
      {required_parts_panel_html}
      <h3>Confirm LEGO Part</h3>
      <div class="confirm-grid">{confirm_candidate_html}</div>
      <form id="review-form">
        <input type="hidden" name="bundle_id" value="{escape(selected_bundle_id)}">
        <label><span class="label">status</span><select name="review_status"><option>approved</option><option>rejected</option><option>needs_split_fix</option><option>bad_mask</option></select></label>
        <label><span class="label">mask quality</span><input name="mask_quality" type="number" min="1" max="5" value="5"></label>
        <label><span class="label">split quality</span><input name="split_quality" type="number" min="1" max="5" value="5"></label>
        <label><span class="label">reviewed by</span><input name="reviewed_by" value="andy"></label>
        <label><input name="qty_text_present" type="checkbox"> qty text present</label>
        <label><input name="multi_part_merge" type="checkbox"> multi-part merge</label>
        <textarea name="review_notes" placeholder="review notes"></textarea>
        <button type="submit">Save Review</button>
        <button type="button" class="save-next-btn" data-save-review-next="true">Save + Next Crop</button>
      </form>
    </section>
  </main>
  <div class="usage-modal-backdrop" data-usage-modal="true" aria-hidden="true">
    <div class="usage-modal" role="dialog" aria-modal="true" aria-labelledby="usage-modal-title">
      <div class="usage-modal-head">
        <h3 id="usage-modal-title" data-usage-modal-title="true">Confirmations</h3>
        <button type="button" class="usage-modal-close" data-usage-modal-close="true">Close</button>
      </div>
      <div class="usage-summary" data-usage-modal-summary="true"></div>
      <div class="usage-table-wrap" data-usage-modal-body="true"></div>
    </div>
  </div>
  <div class="mask-amend-backdrop" data-mask-amend-modal="true" aria-hidden="true">
    <div class="mask-amend-modal" role="dialog" aria-modal="true" aria-labelledby="mask-amend-title">
      <div class="mask-amend-head">
        <h3 id="mask-amend-title" data-mask-amend-title="true">Amend Mask</h3>
        <button type="button" class="mask-amend-close" data-mask-amend-close="true">Close</button>
      </div>
      <div class="mask-amend-tools">
        <div>
          <button type="button" class="mask-tool active" data-mask-tool="add">Brush ADD</button>
          <button type="button" class="mask-tool" data-mask-tool="erase">Eraser</button>
        </div>
        <label>Brush size <input type="range" min="2" max="80" value="18" data-mask-brush-size="true"></label>
        <button type="button" data-save-mask-amendment="true">Save Amendment</button>
      </div>
      <div class="mask-amend-stage-wrap">
        <div class="mask-amend-stage" data-mask-stage="true">
          <canvas data-mask-base-canvas="true"></canvas>
          <canvas class="mask-amend-layer" data-mask-current-canvas="true"></canvas>
          <canvas class="mask-amend-layer" data-mask-add-canvas="true"></canvas>
          <canvas class="mask-amend-layer" data-mask-erase-canvas="true"></canvas>
        </div>
      </div>
    </div>
  </div>
  <!-- ── Colour cleanup picker modal ──────────────────────────────────── -->
  <div class="clnup-backdrop" data-clnup-modal="true" aria-hidden="true">
    <div class="clnup-modal" role="dialog" aria-modal="true" aria-labelledby="clnup-title">
      <div class="clnup-head">
        <h3 id="clnup-title">Pick Background Colour</h3>
        <button type="button" class="clnup-btn clnup-btn-close" data-clnup-close="true">Close</button>
      </div>
      <div class="clnup-body">
        <div class="clnup-stage-wrap" data-clnup-stage-wrap="true">
          <div class="clnup-stage" data-clnup-stage="true">
            <canvas class="clnup-canvas" data-clnup-img-canvas="true"></canvas>
            <canvas class="clnup-overlay" data-clnup-overlay-canvas="true"></canvas>
            <div class="clnup-crosshair" data-clnup-crosshair="true" style="display:none;position:absolute;width:0;height:0"></div>
          </div>
        </div>
        <div class="clnup-sidebar">
          <div class="clnup-mag-wrap">
            <div class="clnup-mag-label">Zoom ×10</div>
            <canvas class="clnup-magnifier" data-clnup-magnifier="true" width="210" height="210"></canvas>
          </div>
          <div class="clnup-pixel-info" data-clnup-pixel-info="true">Hover to sample colour…</div>
          <div class="clnup-swatch-row">
            <div class="clnup-swatch clnup-swatch-none" data-clnup-swatch="true"></div>
            <div class="clnup-swatch-label" data-clnup-swatch-label="true">No colour picked</div>
          </div>
          <div class="clnup-controls">
            <label>Tolerance (LAB Δ)
              <div class="clnup-tol-row">
                <input type="range" min="5" max="80" value="22" data-clnup-tolerance="true">
                <span class="clnup-tol-val" data-clnup-tol-val="true">22</span>
              </div>
            </label>
          </div>
          <button type="button" class="clnup-btn clnup-btn-primary" data-clnup-save="true" disabled>Save Cleanup</button>
          <div class="clnup-status" data-clnup-status="true">Click image to pick a colour</div>
        </div>
      </div>
    </div>
  </div>
  <script>
    let nextPendingUrl = {next_pending_url_js};
    let nextBagUrl = {next_bag_url_js};
    const nextPendingEndpoint = {next_pending_endpoint_js};
    const bagAuditHref = {bag_audit_href_js};
    function showBagComplete(nextBagUrlValue) {{
      const banner = document.querySelector('[data-bag-complete-banner]');
      if (banner) {{
        banner.classList.remove('hidden');
        const msg = nextBagUrlValue ? 'Bag complete — next bag is available.' : 'Bag complete.';
        banner.innerHTML = escape_html(msg)
          + (bagAuditHref ? ' <a href="' + bagAuditHref + '" target="_blank" rel="noopener" style="font-weight:600;text-decoration:underline">Open Bag Audit</a>' : '')
          + (nextBagUrlValue ? ' <a href="' + nextBagUrlValue + '" style="font-weight:600;text-decoration:underline">Open Next Bag</a>' : '');
      }}
      const nextBagButton = document.querySelector('[data-next-bag]');
      if (nextBagButton && nextBagUrlValue) {{
        nextBagButton.classList.remove('hidden');
        nextBagButton.setAttribute('href', nextBagUrlValue);
      }}
      const msg = nextBagUrlValue
        ? 'Bag complete.\n\nOpen Bag Audit or continue to Next Bag?'
        : 'Bag complete.\n\nOpen Bag Audit?';
      if (bagAuditHref && confirm(msg)) {{
        window.open(bagAuditHref, '_blank', 'noopener');
      }}
    }}
    function escape_html(s) {{
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }}
    async function loadNextPending() {{
      if (!nextPendingEndpoint) {{
        return {{next_url: nextPendingUrl, next_bag_url: nextBagUrl, current_bag_complete: !nextPendingUrl}};
      }}
      const res = await fetch(nextPendingEndpoint);
      if (!res.ok) {{
        throw new Error(await res.text());
      }}
      const data = await res.json();
      nextPendingUrl = data.next_url || '';
      nextBagUrl = data.next_bag_url || '';
      return data;
    }}
    async function openNextPending() {{
      let data = {{next_url: nextPendingUrl, next_bag_url: nextBagUrl, current_bag_complete: !nextPendingUrl}};
      try {{
        data = await loadNextPending();
      }} catch (err) {{
        alert(String(err && err.message ? err.message : err));
        return;
      }}
      if (data.next_url) {{
        window.location.href = data.next_url;
        return;
      }}
      if (data.current_bag_complete) {{
        showBagComplete(data.next_bag_url || '');
        return;
      }}
      alert('No next pending bundle after this crop; current bundle is still pending.');
    }}
    function buildReviewPayload(form) {{
      const payload = Object.fromEntries(new FormData(form).entries());
      payload.qty_text_present = form.qty_text_present.checked;
      payload.multi_part_merge = form.multi_part_merge.checked;
      payload.mask_quality = Number(payload.mask_quality || 0);
      payload.split_quality = Number(payload.split_quality || 0);
      return payload;
    }}
    async function saveReview(form) {{
      const payload = buildReviewPayload(form);
      const res = await fetch('/debug/training-store/review', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload)
      }});
      if (!res.ok) {{
        alert(await res.text());
        return false;
      }}
      return true;
    }}
    function escapeHtml(value) {{
      return String(value == null ? '' : value).replace(/[&<>"']/g, (ch) => ({{
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      }}[ch]));
    }}
    function artifactUrl(path) {{
      const text = String(path || '').trim();
      return text ? '/debug/ai-snap-artifact?path=' + encodeURIComponent(text) : '';
    }}
    const maskAmendState = {{
      bundleId: "{escape(selected_bundle_id)}",
      candidateIndex: '',
      tool: 'add',
      drawing: false
    }};
    const maskAmendModal = document.querySelector('[data-mask-amend-modal]');
    const maskBaseCanvas = document.querySelector('[data-mask-base-canvas]');
    const maskCurrentCanvas = document.querySelector('[data-mask-current-canvas]');
    const maskAddCanvas = document.querySelector('[data-mask-add-canvas]');
    const maskEraseCanvas = document.querySelector('[data-mask-erase-canvas]');
    const maskStage = document.querySelector('[data-mask-stage]');
    const maskBrushSize = document.querySelector('[data-mask-brush-size]');
    function setMaskCanvasSize(width, height) {{
      [maskBaseCanvas, maskCurrentCanvas, maskAddCanvas, maskEraseCanvas].forEach((canvas) => {{
        if (!canvas) {{
          return;
        }}
        canvas.width = width;
        canvas.height = height;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
      }});
      if (maskStage) {{
        maskStage.style.width = width + 'px';
        maskStage.style.height = height + 'px';
      }}
    }}
    function loadImageElement(src) {{
      return new Promise((resolve, reject) => {{
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = src;
      }});
    }}
    async function openMaskAmend(button) {{
      const originalPath = button.dataset.originalPath || '';
      const maskPath = button.dataset.maskPath || '';
      let candidateBox = [];
      try {{
        candidateBox = JSON.parse(button.dataset.candidateBox || '[]');
      }} catch (err) {{
        candidateBox = [];
      }}
      const originalUrl = artifactUrl(originalPath);
      if (!originalUrl) {{
        alert('original crop is missing');
        return;
      }}
      const originalImage = await loadImageElement(originalUrl);
      setMaskCanvasSize(originalImage.naturalWidth || originalImage.width, originalImage.naturalHeight || originalImage.height);
      const baseCtx = maskBaseCanvas.getContext('2d');
      baseCtx.clearRect(0, 0, maskBaseCanvas.width, maskBaseCanvas.height);
      baseCtx.drawImage(originalImage, 0, 0, maskBaseCanvas.width, maskBaseCanvas.height);
      [maskCurrentCanvas, maskAddCanvas, maskEraseCanvas].forEach((canvas) => {{
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }});
      if (maskPath && candidateBox.length === 4) {{
        try {{
          const maskImage = await loadImageElement(artifactUrl(maskPath));
          const [x, y, w, h] = candidateBox.map((value) => Number(value || 0));
          const offscreen = document.createElement('canvas');
          offscreen.width = Math.max(1, Math.round(w));
          offscreen.height = Math.max(1, Math.round(h));
          const offscreenCtx = offscreen.getContext('2d');
          offscreenCtx.drawImage(maskImage, 0, 0, offscreen.width, offscreen.height);
          const pixels = offscreenCtx.getImageData(0, 0, offscreen.width, offscreen.height);
          for (let i = 0; i < pixels.data.length; i += 4) {{
            const value = Math.max(pixels.data[i], pixels.data[i + 1], pixels.data[i + 2], pixels.data[i + 3]);
            pixels.data[i] = 31;
            pixels.data[i + 1] = 111;
            pixels.data[i + 2] = 201;
            pixels.data[i + 3] = value > 8 ? 96 : 0;
          }}
          offscreenCtx.putImageData(pixels, 0, 0);
          const currentCtx = maskCurrentCanvas.getContext('2d');
          currentCtx.drawImage(offscreen, x, y, w, h);
        }} catch (err) {{
          console.warn('mask overlay failed', err);
        }}
      }}
      maskAmendState.candidateIndex = button.dataset.candidateIndex || '';
      maskAmendState.tool = 'add';
      document.querySelectorAll('[data-mask-tool]').forEach((toolButton) => {{
        toolButton.classList.toggle('active', toolButton.dataset.maskTool === 'add');
      }});
      if (maskAmendModal) {{
        maskAmendModal.classList.add('open');
        maskAmendModal.setAttribute('aria-hidden', 'false');
      }}
    }}
    function closeMaskAmend() {{
      if (maskAmendModal) {{
        maskAmendModal.classList.remove('open');
        maskAmendModal.setAttribute('aria-hidden', 'true');
      }}
    }}
    function drawMaskStroke(event) {{
      if (!maskAmendState.drawing) {{
        return;
      }}
      const targetCanvas = maskAmendState.tool === 'erase' ? maskEraseCanvas : maskAddCanvas;
      const rect = targetCanvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * targetCanvas.width;
      const y = ((event.clientY - rect.top) / rect.height) * targetCanvas.height;
      const radius = Number(maskBrushSize && maskBrushSize.value || 18) / 2;
      const ctx = targetCanvas.getContext('2d');
      ctx.fillStyle = maskAmendState.tool === 'erase' ? 'rgba(255,60,40,.70)' : 'rgba(47,108,65,.72)';
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }}
    [maskAddCanvas, maskEraseCanvas].forEach((canvas) => {{
      if (!canvas) {{
        return;
      }}
      canvas.addEventListener('pointerdown', (event) => {{
        maskAmendState.drawing = true;
        canvas.setPointerCapture(event.pointerId);
        drawMaskStroke(event);
      }});
      canvas.addEventListener('pointermove', drawMaskStroke);
      canvas.addEventListener('pointerup', () => {{
        maskAmendState.drawing = false;
      }});
      canvas.addEventListener('pointercancel', () => {{
        maskAmendState.drawing = false;
      }});
    }});
    document.querySelectorAll('[data-mask-tool]').forEach((button) => {{
      button.addEventListener('click', () => {{
        maskAmendState.tool = button.dataset.maskTool || 'add';
        document.querySelectorAll('[data-mask-tool]').forEach((toolButton) => {{
          toolButton.classList.toggle('active', toolButton === button);
        }});
        if (maskAddCanvas && maskEraseCanvas) {{
          maskAddCanvas.style.pointerEvents = maskAmendState.tool === 'add' ? 'auto' : 'none';
          maskEraseCanvas.style.pointerEvents = maskAmendState.tool === 'erase' ? 'auto' : 'none';
        }}
      }});
    }});
    if (maskAddCanvas && maskEraseCanvas) {{
      maskAddCanvas.style.pointerEvents = 'auto';
      maskEraseCanvas.style.pointerEvents = 'none';
    }}
    document.querySelectorAll('[data-amend-mask]').forEach((button) => {{
      button.addEventListener('click', () => openMaskAmend(button));
    }});
    document.querySelectorAll('[data-mask-amend-close]').forEach((button) => {{
      button.addEventListener('click', closeMaskAmend);
    }});
    document.querySelectorAll('[data-save-mask-amendment]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        button.disabled = true;
        button.textContent = 'Saving...';
        const res = await fetch('/debug/training-store/save-manual-mask-amendment', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            bundle_id: maskAmendState.bundleId,
            candidate_index: maskAmendState.candidateIndex,
            amended_by: 'andy',
            add_mask_data_url: maskAddCanvas.toDataURL('image/png'),
            erase_mask_data_url: maskEraseCanvas.toDataURL('image/png')
          }})
        }});
        if (!res.ok) {{
          alert(await res.text());
          button.disabled = false;
          button.textContent = 'Save Amendment';
          return;
        }}
        location.reload();
      }});
    }});
    document.querySelectorAll('[data-rerun-refinement]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        if (button.disabled) {{
          return;
        }}
        const originalText = button.textContent;
        button.disabled = true;
        button.textContent = 'Refining...';
        const res = await fetch('/debug/training-store/rerun-manual-mask-refinement', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            bundle_id: maskAmendState.bundleId,
            candidate_index: button.dataset.candidateIndex || '',
            refined_by: 'andy'
          }})
        }});
        if (!res.ok) {{
          alert(await res.text());
          button.disabled = false;
          button.textContent = originalText;
          return;
        }}
        location.reload();
      }});
    }});
    const usageModal = document.querySelector('[data-usage-modal]');
    const usageModalTitle = document.querySelector('[data-usage-modal-title]');
    const usageModalSummary = document.querySelector('[data-usage-modal-summary]');
    const usageModalBody = document.querySelector('[data-usage-modal-body]');
    function openUsageModal(title) {{
      if (usageModalTitle) {{
        usageModalTitle.textContent = title;
      }}
      if (usageModal) {{
        usageModal.classList.add('open');
        usageModal.setAttribute('aria-hidden', 'false');
      }}
    }}
    function closeUsageModal() {{
      if (usageModal) {{
        usageModal.classList.remove('open');
        usageModal.setAttribute('aria-hidden', 'true');
      }}
    }}
    function renderUsageSummary(data) {{
      return '<span>Set qty <strong>' + escapeHtml(data.required_qty ?? 'n/a') + '</strong></span>'
        + '<span>Confirmed qty <strong>' + escapeHtml(data.confirmed_qty ?? data.total_confirmed_qty ?? 0) + '</strong></span>'
        + '<span>Remaining qty <strong>' + escapeHtml(data.effective_remaining_qty ?? 'n/a') + '</strong></span>'
        + '<span>Over-confirmed by <strong>' + escapeHtml(data.over_confirmed_by ?? 'n/a') + '</strong></span>';
    }}
    function renderUsageTable(data) {{
      const rows = Array.isArray(data.rows) ? data.rows : [];
      if (!rows.length) {{
        return '<div class="usage-empty">No matching confirmations.</div>';
      }}
      return '<table class="usage-table">'
        + '<thead><tr><th>thumbnail</th><th>bundle_id</th><th>candidate_index</th><th>qty</th><th>confirmed_by</th><th>created_at</th></tr></thead>'
        + '<tbody>'
        + rows.map((row) => {{
        const thumbUrl = artifactUrl(row.thumbnail_path);
        const reviewUrl = '/debug/training-store/review-ui?bundle_id=' + encodeURIComponent(String(row.bundle_id || ''));
        const thumbHtml = thumbUrl
          ? '<img src="' + escapeHtml(thumbUrl) + '" alt="confirmed thumbnail" loading="lazy">'
          : '';
        return '<tr>'
          + '<td><div class="usage-thumb">' + thumbHtml + '</div></td>'
          + '<td><a href="' + escapeHtml(reviewUrl) + '" target="_blank" rel="noopener">' + escapeHtml(row.bundle_id || '') + '</a></td>'
          + '<td>' + escapeHtml(row.candidate_index ?? '') + '</td>'
          + '<td>' + escapeHtml(row.qty ?? '') + '</td>'
          + '<td>' + escapeHtml(row.confirmed_by || '') + '</td>'
          + '<td>' + escapeHtml(row.created_at || '') + '</td>'
          + '</tr>';
      }}).join('')
        + '</tbody></table>';
    }}
    document.getElementById('review-form').addEventListener('submit', async (event) => {{
      event.preventDefault();
      const form = event.currentTarget;
      const saved = await saveReview(form);
      if (!saved) {{
        return;
      }}
      location.reload();
    }});
    document.querySelectorAll('[data-save-review-next]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        const form = document.getElementById('review-form');
        const saved = await saveReview(form);
        if (!saved) {{
          return;
        }}
        await openNextPending();
      }});
    }});
    document.querySelectorAll('[data-next-pending]').forEach((button) => {{
      button.addEventListener('click', async (event) => {{
        event.preventDefault();
        if (button.disabled) {{
          return;
        }}
        await openNextPending();
      }});
    }});
    document.querySelectorAll('[data-next-bag]').forEach((button) => {{
      button.addEventListener('click', (event) => {{
        const href = button.getAttribute('href') || nextBagUrl || '';
        if (!href) {{
          event.preventDefault();
          showBagComplete('');
        }}
      }});
    }});
    document.querySelectorAll('[data-usage-modal-close]').forEach((button) => {{
      button.addEventListener('click', closeUsageModal);
    }});
    if (usageModal) {{
      usageModal.addEventListener('click', (event) => {{
        if (event.target === usageModal) {{
          closeUsageModal();
        }}
      }});
    }}
    document.addEventListener('keydown', (event) => {{
      if (event.key === 'Escape' && usageModal && usageModal.classList.contains('open')) {{
        closeUsageModal();
      }}
    }});
    document.querySelectorAll('[data-show-confirmations]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        const partNum = button.dataset.usagePartNum || '';
        const colorId = button.dataset.usageColorId || '';
        openUsageModal('Confirmations for ' + partNum + ' / ' + colorId);
        if (usageModalSummary) {{
          usageModalSummary.innerHTML = renderUsageSummary({{}});
        }}
        if (usageModalBody) {{
          usageModalBody.innerHTML = '<div class="usage-empty">Loading confirmations...</div>';
        }}
        const params = new URLSearchParams({{
          set_num: button.dataset.usageSetNum || '',
          part_num: partNum,
          color_id: colorId,
          required_qty: button.dataset.usageRequiredQty || ''
        }});
        if (button.dataset.usageElementId) {{
          params.set('element_id', button.dataset.usageElementId);
        }}
        const res = await fetch('/debug/training-store/confirmed-part-usage?' + params.toString());
        if (!res.ok) {{
          if (usageModalBody) {{
            usageModalBody.innerHTML = '<div class="usage-empty">' + escapeHtml(await res.text()) + '</div>';
          }}
          return;
        }}
        const data = await res.json();
        if (usageModalSummary) {{
          usageModalSummary.innerHTML = renderUsageSummary(data);
        }}
        if (usageModalBody) {{
          usageModalBody.innerHTML = renderUsageTable(data);
        }}
      }});
    }});
    document.querySelectorAll('[data-generate-split-candidates]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        if (button.disabled) {{
          return;
        }}
        button.disabled = true;
        const originalText = button.textContent;
        button.textContent = 'Generating...';
        const params = new URLSearchParams({{bundle_id: "{escape(selected_bundle_id)}"}});
        const res = await fetch('/debug/training-store/generate-split-candidates?' + params.toString(), {{method: 'POST'}});
        if (!res.ok) {{
          button.disabled = false;
          button.textContent = originalText;
          alert(await res.text());
          return;
        }}
        location.reload();
      }});
    }});
    document.querySelectorAll('[data-candidate-action]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        const action = button.dataset.candidateAction;
        const index = button.dataset.candidateIndex;
        if (action === 'ai_crop_fix') {{
          if (button.disabled) {{
            return;
          }}
          button.disabled = true;
          const originalText = button.textContent;
          button.textContent = 'Fixing...';
          const payload = {{
            bundle_id: "{escape(selected_bundle_id)}",
            candidate_id: String(index || ""),
            current_candidate_path: button.dataset.currentCandidatePath || "",
            current_alpha_path: button.dataset.currentAlphaPath || "",
            reference_candidate_path: button.dataset.referenceCandidatePath || "",
            slot_overlay_path: button.dataset.slotOverlayPath || "",
            qty: button.dataset.candidateQty || null,
            dry_run: false,
            mode: "placeholder"
          }};
          const res = await fetch('/debug/training-store/ai-crop-fix', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify(payload)
          }});
          if (!res.ok) {{
            button.disabled = false;
            button.textContent = originalText;
            alert(await res.text());
            return;
          }}
          const previewParams = new URLSearchParams({{
            bundle_id: "{escape(selected_bundle_id)}",
            candidate_id: String(index || "")
          }});
          const previewUrl = '/debug/training-store/ai-crop-fix-preview?' + previewParams.toString();
          window.location.assign(previewUrl);
          button.disabled = false;
          button.textContent = originalText;
          return;
        }}
        const params = new URLSearchParams({{bundle_id: "{escape(selected_bundle_id)}", candidate_index: String(index || "")}});
        let endpoint = '/debug/training-store/reject-split-candidate';
        if (action === 'accept') {{
          endpoint = '/debug/training-store/accept-split-candidate';
        }} else if (action === 'scrub') {{
          endpoint = '/debug/training-store/scrub-candidate-qty';
        }} else if (action === 'mark_qty_clean') {{
          endpoint = '/debug/training-store/mark-qty-clean';
        }} else if (['needs_mask_expand', 'needs_ocr_review', 'needs_manual_crop'].includes(action)) {{
          endpoint = '/debug/training-store/set-candidate-review-state';
          params.set('review_state', action);
        }}
        const res = await fetch(endpoint + '?' + params.toString(), {{method: 'POST'}});
        if (!res.ok) {{
          alert(await res.text());
          return;
        }}
        location.reload();
      }});
    }});
    document.querySelectorAll('[data-confirm-candidate]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        if (button.disabled) {{
          return;
        }}
        const card = button.closest('.confirm-card');
        const fieldValue = (name) => {{
          const input = card && card.querySelector('[data-confirm-field="' + name + '"]');
          return input ? input.value : '';
        }};
        const payload = {{
          bundle_id: "{escape(selected_bundle_id)}",
          candidate_index: Number(button.dataset.confirmCandidate || 0),
          part_num: button.dataset.partNum || fieldValue('part_num'),
          color_id: Number(button.dataset.colorId || fieldValue('color_id') || 0),
          element_id: button.dataset.elementId || fieldValue('element_id'),
          qty: Number(button.dataset.candidateQty || 0) || null,
          confirmed_by: fieldValue('confirmed_by') || 'andy'
        }};
        const res = await fetch('/debug/training-store/confirm-candidate-part', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(payload)
        }});
        if (!res.ok) {{
          alert(await res.text());
          return;
        }}
        location.reload();
      }});
    }});
    document.querySelectorAll('[data-unconfirm-candidate]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        const index = button.dataset.unconfirmCandidate;
        const params = new URLSearchParams({{bundle_id: "{escape(selected_bundle_id)}", candidate_index: String(index || "")}});
        const res = await fetch('/debug/training-store/unconfirm-candidate-part?' + params.toString(), {{method: 'POST'}});
        if (!res.ok) {{
          alert(await res.text());
          return;
        }}
        location.reload();
      }});
    }});
    function rgbHexToTuple(rgbHex) {{
      const text = String(rgbHex || '').trim().replace(/^#/, '').toUpperCase();
      if (!/^[0-9A-F]{{6}}$/.test(text)) {{
        return null;
      }}
      return [parseInt(text.slice(0, 2), 16), parseInt(text.slice(2, 4), 16), parseInt(text.slice(4, 6), 16)];
    }}
    function rgbDistanceSquared(left, right) {{
      return Math.pow(Number(left[0] || 0) - Number(right[0] || 0), 2)
        + Math.pow(Number(left[1] || 0) - Number(right[1] || 0), 2)
        + Math.pow(Number(left[2] || 0) - Number(right[2] || 0), 2);
    }}
    function currentRequiredScope() {{
      const checked = document.querySelector('input[name="required-scope"]:checked');
      return checked ? checked.value : 'set';
    }}
    function currentSelectedColour() {{
      const active = document.querySelector('.review-colour-chip.active[data-color-id]');
      return active ? String(active.dataset.colorId || '') : '';
    }}
    function updateRequiredPartsPanel() {{
      const scope = currentRequiredScope();
      const currentColourOnly = Boolean(document.querySelector('[data-required-current-colour]') && document.querySelector('[data-required-current-colour]').checked);
      const showCompleted = Boolean(document.querySelector('[data-required-show-completed]') && document.querySelector('[data-required-show-completed]').checked);
      const selectedColor = currentSelectedColour();
      let rendered = 0;
      document.querySelectorAll('[data-required-part-row]').forEach((row) => {{
        const scopeMatch = String(row.dataset.scope || '') === scope;
        const completedMatch = showCompleted || String(row.dataset.completed || '') !== '1';
        const colorMatch = !!selectedColor && String(row.dataset.colorId || '') === selectedColor;
        const hidden = !(scopeMatch && completedMatch);
        row.classList.toggle('hidden', hidden);
        row.classList.toggle('colour-match-highlight', currentColourOnly && colorMatch);
        row.style.order = currentColourOnly && colorMatch ? '-1' : '';
        if (!hidden) {{
          rendered += 1;
        }}
      }});
      document.querySelectorAll('[data-rendered-match-count]').forEach((node) => {{
        node.textContent = String(rendered);
      }});
      document.querySelectorAll('[data-hidden-due-limit-count]').forEach((node) => {{
        node.textContent = '0';
      }});
    }}
    function updateReviewCardScope(card) {{
      if (!card) {{
        return;
      }}
      const active = card.querySelector('.review-colour-chip.active[data-color-id]');
      const selected = active ? String(active.dataset.colorId || '') : '';
      setReviewCardColour(card, selected);
    }}
    function updateReviewCardDomDebug(card) {{
      if (!card) {{
        return;
      }}
      const debug = card.querySelector('[data-match-debug]');
      if (!debug) {{
        return;
      }}
      const nodes = Array.from(card.querySelectorAll('.part-suggestion'));
      const color308Nodes = nodes.filter((node) => String(node.dataset.partColorId || '') === '308');
      const visibleColor308Nodes = color308Nodes.filter((node) => {{
        const style = window.getComputedStyle(node);
        return !node.hidden && style.display !== 'none' && style.visibility !== 'hidden';
      }});
      const writeDebugCount = (selector, value) => {{
        const target = debug.querySelector(selector);
        if (target) {{
          target.textContent = String(value);
        }}
      }};
      writeDebugCount('[data-dom-total-card-count]', nodes.length);
      writeDebugCount('[data-dom-color-308-count]', color308Nodes.length);
      writeDebugCount('[data-dom-visible-after-308-count]', visibleColor308Nodes.length);
    }}
    function setReviewCardColour(card, colorId) {{
      if (!card) {{
        return;
      }}
      const selected = colorId === null || colorId === undefined ? '' : String(colorId);
      card.querySelectorAll('[data-review-colour-chip]').forEach((button) => {{
        const buttonColor = button.dataset.colorId === undefined ? '' : String(button.dataset.colorId);
        button.classList.toggle('active', !!selected && buttonColor === selected);
      }});
      const clearButton = card.querySelector('[data-review-colour-clear]');
      if (clearButton) {{
        clearButton.classList.toggle('active', !selected);
      }}
      let filteredCount = 0;
      let selectedColorCount = 0;
      card.querySelectorAll('.part-suggestion').forEach((node) => {{
        const matchColor = node.dataset.partColorId === undefined ? '' : String(node.dataset.partColorId);
        const colorMatch = !!selected && matchColor === selected;
        node.classList.remove('hidden');
        node.classList.toggle('colour-match-highlight', colorMatch);
        node.style.order = colorMatch ? '-1' : '';
        filteredCount += 1;
        if (colorMatch) {{
          selectedColorCount += 1;
        }}
      }});
      const debug = card.querySelector('[data-match-debug]');
      if (debug) {{
        debug.dataset.selectedColorId = selected;
        debug.dataset.filteredMatchCount = String(filteredCount);
        const selectedNode = debug.querySelector('[data-debug-selected-color]');
        const countNode = debug.querySelector('[data-debug-filtered-count]');
        const renderedNode = debug.querySelector('[data-rendered-match-count-local]');
        if (selectedNode) {{
          selectedNode.textContent = selected || 'all';
        }}
        if (countNode) {{
          countNode.textContent = selected ? String(selectedColorCount) : String(filteredCount);
        }}
        if (renderedNode) {{
          renderedNode.textContent = String(filteredCount);
        }}
      }}
      updateReviewCardDomDebug(card);
      const emptyMessage = card.querySelector('[data-colour-empty-message]');
      if (emptyMessage) {{
        emptyMessage.classList.add('hidden');
        emptyMessage.textContent = '';
      }}
      const colorInput = card.querySelector('[data-confirm-field="color_id"]');
      if (colorInput && selected) {{
        colorInput.value = selected;
      }}
      updateRequiredPartsPanel();
    }}
    document.querySelectorAll('[data-review-colour-clear]').forEach((button) => {{
      button.addEventListener('click', () => {{
        if (button.disabled) {{
          return;
        }}
        setReviewCardColour(button.closest('.confirm-card'), '');
      }});
    }});
    document.querySelectorAll('[data-review-colour-chip]').forEach((button) => {{
      button.addEventListener('click', () => {{
        if (button.disabled) {{
          return;
        }}
        if (button.dataset.reviewColourClear === 'true') {{
          return;
        }}
        setReviewCardColour(button.closest('.confirm-card'), button.dataset.colorId || '');
      }});
    }});
    document.querySelectorAll('input[name="required-scope"], [data-required-current-colour], [data-required-show-completed]').forEach((control) => {{
      control.addEventListener('change', () => {{
        document.querySelectorAll('.confirm-card').forEach((card) => updateReviewCardScope(card));
        updateRequiredPartsPanel();
      }});
    }});
    document.querySelectorAll('.confirm-card').forEach((card) => updateReviewCardScope(card));
    updateRequiredPartsPanel();
    document.querySelectorAll('[data-review-colour-pick]').forEach((button) => {{
      button.addEventListener('click', () => {{
        if (button.disabled) {{
          return;
        }}
        const card = button.closest('.confirm-card');
        const thumb = card && card.querySelector('[data-confirm-thumb]');
        if (!thumb) {{
          return;
        }}
        const isActive = thumb.classList.toggle('picker-active');
        button.classList.toggle('active', isActive);
        button.textContent = isActive ? 'Click candidate image' : 'Pick colour from candidate';
      }});
    }});
    document.querySelectorAll('[data-confirm-thumb]').forEach((thumb) => {{
      thumb.addEventListener('click', (event) => {{
        if (!thumb.classList.contains('picker-active')) {{
          return;
        }}
        const card = thumb.closest('.confirm-card');
        const image = thumb.querySelector('img');
        if (!card || !image || !image.naturalWidth || !image.naturalHeight) {{
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
        const canvas = document.createElement('canvas');
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        const ctx = canvas.getContext('2d', {{ willReadFrequently: true }});
        if (!ctx) {{
          return;
        }}
        ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight);
        let r = 0, g = 0, b = 0, count = 0;
        for (let dy = -2; dy <= 2; dy += 1) {{
          for (let dx = -2; dx <= 2; dx += 1) {{
            const px = Math.max(0, Math.min(image.naturalWidth - 1, sampleX + dx));
            const py = Math.max(0, Math.min(image.naturalHeight - 1, sampleY + dy));
            const pixel = ctx.getImageData(px, py, 1, 1).data;
            if (Number(pixel[3] || 0) < 20) {{
              continue;
            }}
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
        let bestColorId = '';
        let bestDistance = Number.POSITIVE_INFINITY;
        card.querySelectorAll('[data-review-colour-chip]').forEach((button) => {{
          const rgb = rgbHexToTuple(button.dataset.colorRgb || '');
          if (!rgb) {{
            return;
          }}
          const distance = rgbDistanceSquared(sampled, rgb);
          if (distance < bestDistance) {{
            bestDistance = distance;
            bestColorId = String(button.dataset.colorId || '');
          }}
        }});
        if (bestColorId) {{
          setReviewCardColour(card, bestColorId);
        }}
        thumb.classList.remove('picker-active');
        const pickButton = card.querySelector('[data-review-colour-pick]');
        if (pickButton) {{
          pickButton.classList.remove('active');
          pickButton.textContent = 'Pick colour from candidate';
        }}
      }});
    }});

    /* ── Colour cleanup picker ──────────────────────────────────────────── */
    (function () {{
      const backdrop  = document.querySelector('[data-clnup-modal]');
      const imgCanvas = backdrop && backdrop.querySelector('[data-clnup-img-canvas]');
      const ovCanvas  = backdrop && backdrop.querySelector('[data-clnup-overlay-canvas]');
      const magCanvas = backdrop && backdrop.querySelector('[data-clnup-magnifier]');
      const pixInfo   = backdrop && backdrop.querySelector('[data-clnup-pixel-info]');
      const swatchEl  = backdrop && backdrop.querySelector('[data-clnup-swatch]');
      const swatchLbl = backdrop && backdrop.querySelector('[data-clnup-swatch-label]');
      const tolInput  = backdrop && backdrop.querySelector('[data-clnup-tolerance]');
      const tolVal    = backdrop && backdrop.querySelector('[data-clnup-tol-val]');
      const saveBtn   = backdrop && backdrop.querySelector('[data-clnup-save]');
      const statusEl  = backdrop && backdrop.querySelector('[data-clnup-status]');
      const crosshair = backdrop && backdrop.querySelector('[data-clnup-crosshair]');

      if (!backdrop) return;

      let pickerState = {{
        bundleId: '',
        candidateIndex: '',
        imgPath: '',
        imgData: null,       // ImageData of source
        srcW: 0, srcH: 0,
        pickedRgb: null,     // [r,g,b] or null
        tolerance: 22,
      }};

      /* ── RGB ↔ LAB conversion (CIELAB, D65) ── */
      function rgbToLab(r, g, b) {{
        let rl = r/255, gl = g/255, bl = b/255;
        const lin = c => c > 0.04045 ? Math.pow((c+0.055)/1.055, 2.4) : c/12.92;
        rl = lin(rl); gl = lin(gl); bl = lin(bl);
        const x = rl*0.4124564 + gl*0.3575761 + bl*0.1804375;
        const y = rl*0.2126729 + gl*0.7151522 + bl*0.0721750;
        const z = rl*0.0193339 + gl*0.1191920 + bl*0.9503041;
        const f = t => t > 0.008856 ? Math.cbrt(t) : 7.787*t + 16/116;
        const fx = f(x/0.95047), fy = f(y/1.0), fz = f(z/1.08883);
        return [116*fy - 16, 500*(fx-fy), 200*(fy-fz)];
      }}
      function labDist(rgb1, rgb2) {{
        const [L1,a1,b1] = rgbToLab(...rgb1);
        const [L2,a2,b2] = rgbToLab(...rgb2);
        return Math.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2);
      }}

      /* ── helpers ── */
      function artifactUrl(p) {{
        return p ? '/debug/ai-snap-artifact?path=' + encodeURIComponent(p) : '';
      }}
      function setStatus(msg) {{ if (statusEl) statusEl.textContent = msg; }}

      /* ── source pixel coords from pointer event ── */
      function srcCoords(canvas, ev) {{
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((ev.clientX - rect.left) * canvas.width / rect.width);
        const y = Math.floor((ev.clientY - rect.top)  * canvas.height / rect.height);
        return [
          Math.max(0, Math.min(pickerState.srcW - 1, x)),
          Math.max(0, Math.min(pickerState.srcH - 1, y)),
        ];
      }}

      /* ── read pixel RGBA from ImageData ── */
      function getPixel(imgData, x, y) {{
        const i = (y * imgData.width + x) * 4;
        return [imgData.data[i], imgData.data[i+1], imgData.data[i+2], imgData.data[i+3]];
      }}

      /* ── draw magnifier ── */
      function drawMag(sx, sy) {{
        if (!magCanvas || !pickerState.imgData) return;
        const mw = magCanvas.width, mh = magCanvas.height;
        const zoom = 10;
        const halfW = mw / zoom / 2, halfH = mh / zoom / 2;
        const mCtx = magCanvas.getContext('2d', {{willReadFrequently: true}});
        mCtx.clearRect(0, 0, mw, mh);
        // draw zoomed region from main canvas
        mCtx.imageSmoothingEnabled = false;
        mCtx.drawImage(imgCanvas, sx - halfW, sy - halfH, mw/zoom, mh/zoom, 0, 0, mw, mh);
        // draw overlay region (cleanup mask)
        if (ovCanvas) {{
          mCtx.globalAlpha = 0.55;
          mCtx.drawImage(ovCanvas, sx - halfW, sy - halfH, mw/zoom, mh/zoom, 0, 0, mw, mh);
          mCtx.globalAlpha = 1.0;
        }}
        // crosshair at centre
        mCtx.strokeStyle = 'rgba(255,60,60,.9)';
        mCtx.lineWidth = 1;
        mCtx.beginPath();
        mCtx.moveTo(mw/2, 0); mCtx.lineTo(mw/2, mh);
        mCtx.moveTo(0, mh/2); mCtx.lineTo(mw, mh/2);
        mCtx.stroke();
        // pixel grid lines
        mCtx.strokeStyle = 'rgba(255,255,255,.10)';
        mCtx.lineWidth = 0.5;
        for (let gx = 0; gx < mw; gx += zoom) {{
          mCtx.beginPath(); mCtx.moveTo(gx, 0); mCtx.lineTo(gx, mh); mCtx.stroke();
        }}
        for (let gy = 0; gy < mh; gy += zoom) {{
          mCtx.beginPath(); mCtx.moveTo(0, gy); mCtx.lineTo(mw, gy); mCtx.stroke();
        }}
      }}

      /* ── update cleanup overlay preview ── */
      function updateOverlay() {{
        if (!ovCanvas || !pickerState.imgData || !pickerState.pickedRgb) return;
        const w = pickerState.srcW, h = pickerState.srcH;
        const ov = ovCanvas.getContext('2d');
        const picked = pickerState.pickedRgb;
        const tol = pickerState.tolerance;
        const out = ov.createImageData(w, h);
        const src = pickerState.imgData;
        for (let i = 0; i < w * h; i++) {{
          const si = i * 4;
          const a = src.data[si+3];
          if (a < 16) continue;  // skip transparent
          const [r, g, b] = [src.data[si], src.data[si+1], src.data[si+2]];
          const dist = labDist([r,g,b], picked);
          if (dist < tol) {{
            // will-be-removed: red tint
            out.data[si]   = 220;
            out.data[si+1] = 30;
            out.data[si+2] = 30;
            out.data[si+3] = 160;
          }}
        }}
        ov.putImageData(out, 0, 0);
      }}

      /* ── load candidate image into canvas ── */
      function loadCandidateImage(imgPath) {{
        return new Promise((resolve, reject) => {{
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.onload = () => {{
            const w = img.naturalWidth, h = img.naturalHeight;
            pickerState.srcW = w; pickerState.srcH = h;
            imgCanvas.width  = w; imgCanvas.height = h;
            ovCanvas.width   = w; ovCanvas.height  = h;
            const ctx = imgCanvas.getContext('2d', {{willReadFrequently: true}});
            // checkerboard background for transparency
            const checker = document.createElement('canvas');
            checker.width = 20; checker.height = 20;
            const cc = checker.getContext('2d');
            cc.fillStyle = '#555'; cc.fillRect(0,0,20,20);
            cc.fillStyle = '#888'; cc.fillRect(0,0,10,10); cc.fillRect(10,10,10,10);
            ctx.fillStyle = ctx.createPattern(checker, 'repeat');
            ctx.fillRect(0, 0, w, h);
            ctx.drawImage(img, 0, 0);
            pickerState.imgData = ctx.getImageData(0, 0, w, h);
            resolve();
          }};
          img.onerror = reject;
          img.src = artifactUrl(imgPath);
        }});
      }}

      /* ── open modal ── */
      async function openPicker(bundleId, candidateIndex, imgPath) {{
        pickerState = {{
          bundleId, candidateIndex, imgPath,
          imgData: null, srcW: 0, srcH: 0,
          pickedRgb: null, tolerance: Number(tolInput ? tolInput.value : 22),
        }};
        if (swatchEl) {{ swatchEl.className = 'clnup-swatch clnup-swatch-none'; swatchEl.style.background = ''; }}
        if (swatchLbl) swatchLbl.textContent = 'No colour picked';
        if (pixInfo)   pixInfo.textContent = 'Hover to sample colour…';
        if (saveBtn)   saveBtn.disabled = true;
        const ovCtx = ovCanvas && ovCanvas.getContext('2d');
        if (ovCtx) ovCtx.clearRect(0, 0, ovCanvas.width, ovCanvas.height);
        setStatus('Loading image…');
        backdrop.classList.add('open');
        backdrop.setAttribute('aria-hidden', 'false');
        try {{
          await loadCandidateImage(imgPath);
          setStatus('Click image to pick a background colour');
        }} catch (e) {{
          setStatus('Error loading image: ' + String(e));
        }}
      }}

      /* ── close modal ── */
      function closePicker() {{
        backdrop.classList.remove('open');
        backdrop.setAttribute('aria-hidden', 'true');
      }}

      /* ── event: open picker buttons ── */
      document.querySelectorAll('[data-pick-colour]').forEach(btn => {{
        btn.addEventListener('click', () => {{
          const bundleId = maskAmendState.bundleId || '';
          const idx = String(btn.dataset.candidateIndex || '');
          const imgPath = String(btn.dataset.candidateImgPath || '');
          openPicker(bundleId, idx, imgPath);
        }});
      }});

      /* ── event: close ── */
      backdrop.addEventListener('click', ev => {{
        if (ev.target === backdrop) closePicker();
      }});
      const closeBtn = backdrop.querySelector('[data-clnup-close]');
      if (closeBtn) closeBtn.addEventListener('click', closePicker);

      /* ── event: tolerance slider ── */
      if (tolInput) {{
        tolInput.addEventListener('input', () => {{
          pickerState.tolerance = Number(tolInput.value);
          if (tolVal) tolVal.textContent = tolInput.value;
          updateOverlay();
        }});
      }}

      /* ── event: mousemove over main canvas ── */
      const stageWrap = backdrop.querySelector('[data-clnup-stage-wrap]');
      if (imgCanvas) {{
        imgCanvas.addEventListener('mousemove', ev => {{
          if (!pickerState.imgData) return;
          const [sx, sy] = srcCoords(imgCanvas, ev);
          const [r, g, b, a] = getPixel(pickerState.imgData, sx, sy);
          const [L, aL, bL] = rgbToLab(r, g, b);
          if (pixInfo) {{
            pixInfo.innerHTML =
              `<b>x:</b>${{sx}} <b>y:</b>${{sy}}<br>` +
              `<b>RGB:</b> ${{r}} ${{g}} ${{b}} (α=${{a}})<br>` +
              `<b>LAB:</b> ${{L.toFixed(1)}} ${{aL.toFixed(1)}} ${{bL.toFixed(1)}}`;
          }}
          drawMag(sx, sy);
          // move crosshair div (CSS position, relative to stage)
          if (crosshair) {{
            const rect = imgCanvas.getBoundingClientRect();
            const cssX = (ev.clientX - rect.left);
            const cssY = (ev.clientY - rect.top);
            crosshair.style.display = 'block';
            crosshair.style.left = cssX + 'px';
            crosshair.style.top  = cssY + 'px';
          }}
        }});
        imgCanvas.addEventListener('mouseleave', () => {{
          if (crosshair) crosshair.style.display = 'none';
        }});
      }}

      /* ── event: click to pick colour ── */
      if (imgCanvas) {{
        imgCanvas.addEventListener('click', ev => {{
          if (!pickerState.imgData) return;
          const [sx, sy] = srcCoords(imgCanvas, ev);
          const [r, g, b, a] = getPixel(pickerState.imgData, sx, sy);
          if (a < 16) {{
            setStatus('Picked transparent pixel — choose an opaque pixel');
            return;
          }}
          pickerState.pickedRgb = [r, g, b];
          const hex = '#' + [r,g,b].map(c => c.toString(16).padStart(2,'0')).join('');
          if (swatchEl) {{ swatchEl.className = 'clnup-swatch'; swatchEl.style.background = hex; }}
          if (swatchLbl) swatchLbl.textContent = `RGB(${{r}}, ${{g}}, ${{b}})`;
          if (saveBtn) saveBtn.disabled = false;
          setStatus(`Picked ${{hex}} — adjust tolerance then Save`);
          updateOverlay();
        }});
      }}

      /* ── event: save cleanup ── */
      if (saveBtn) {{
        saveBtn.addEventListener('click', async () => {{
          if (!pickerState.pickedRgb || saveBtn.disabled) return;
          const originalText = saveBtn.textContent;
          saveBtn.disabled = true;
          saveBtn.textContent = 'Saving…';
          setStatus('Sending to server…');
          try {{
            const res = await fetch('/debug/training-store/save-picked-colour-cleanup', {{
              method: 'POST',
              headers: {{'Content-Type': 'application/json'}},
              body: JSON.stringify({{
                bundle_id: pickerState.bundleId,
                candidate_index: pickerState.candidateIndex,
                picked_rgb: pickerState.pickedRgb,
                tolerance: pickerState.tolerance,
              }}),
            }});
            if (!res.ok) {{
              const msg = await res.text();
              setStatus('Error: ' + msg);
              saveBtn.disabled = false;
              saveBtn.textContent = originalText;
              return;
            }}
            const data = await res.json();
            setStatus('Saved — reloading…');
            closePicker();
            location.reload();
          }} catch (e) {{
            setStatus('Network error: ' + String(e));
            saveBtn.disabled = false;
            saveBtn.textContent = originalText;
          }}
        }});
      }}
    }})();
  </script>
</body>
</html>
"""
    return HTMLResponse(html)


@router.get("/debug/manual-page-image")
def manual_page_image(path: str = Query(...)):
    """Serve a manual-crop page image (PNG) from inside the debug tree.

    Only paths whose resolved location is under
    /Users/olly/aim2build-instruction/debug/ are allowed.
    """
    requested = Path(str(path or "").strip()).expanduser()
    allowed_root = Path("/Users/olly/aim2build-instruction/debug").resolve()
    try:
        resolved = requested.resolve()
    except Exception:
        raise HTTPException(status_code=404, detail="page image not found")
    if (
        allowed_root not in resolved.parents
        or not resolved.exists()
        or not resolved.is_file()
    ):
        raise HTTPException(status_code=404, detail="page image not found")
    return FileResponse(str(resolved), media_type="image/png")


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
    cache_hit = False
    try:
        temp_crop_path = _write_ai_snap_temp_crop_image(crop)
        if temp_crop_path is None:
            raise HTTPException(status_code=400, detail="crop image unavailable")
        # --- Disk cache check ---
        # Key = SHA1(raw crop bytes + canonical qty_token_boxes geometry).
        # A cache entry is discarded if any part_cutout_path file is gone.
        crop_image_bytes = temp_crop_path.read_bytes()
        _cache_key = _auto_mask_cache_key(crop_image_bytes, qty_token_boxes)
        _cached = _read_auto_mask_cache(_cache_key)
        if _cached is not None:
            result = _cached
            cache_hit = True
        else:
            result = create_shape_masks_for_callout_slots(
                str(temp_crop_path),
                qty_token_boxes,
                set_num=set_num,
                bag=int(bag),
                crop_id=crop_id,
                desktop_overlays=not fast_map_flag,
            )
            _write_auto_mask_cache(_cache_key, result)
    finally:
        if temp_crop_path is not None:
            try:
                temp_crop_path.unlink(missing_ok=True)
            except Exception:
                pass

    # Optional SAM refinement: runs only on fresh results (cache already applied it).
    if sam_refine_flag and not cache_hit:
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

    # Step-callout slot segmentation: run hybrid SAM2/colour segmentation on each slot window
    # using the better catalog-match pipeline (text-scrubbed crop → _hybrid_segment_crop).
    for _slot in list(result.get("slots") or []):
        _slot_win = _slot.get("slot_window") or _slot.get("slot_crop_box")
        if _slot_win and len(_slot_win) >= 4 and int(_slot_win[2]) > 0 and int(_slot_win[3]) > 0:
            _seg = _segment_step_callout_slot(
                crop, set_num, int(bag), crop_id,
                slot_index=int(_slot.get("slot_index", 0)),
                slot_window=list(_slot_win),
            )
            _slot["step_masked_path"] = str(_seg.get("masked_path", "")) if _seg.get("ok") else ""
            _slot["step_seg_method"]  = str(_seg.get("segmentation_method", ""))
            _slot["step_cache_hit"]   = bool(_seg.get("cache_hit", False))
        else:
            _slot["step_masked_path"] = ""
            _slot["step_seg_method"]  = ""
            _slot["step_cache_hit"]   = False

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
        "cache_hit": cache_hit,
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

    # Fast lookup: pool candidates by part_num::color_id for img_url / color_name enrichment.
    pool_lookup: Dict[str, Dict[str, Any]] = {
        _candidate_part_key(c.get("part_num"), c.get("color_id")): c
        for c in candidate_rows
        if str(c.get("part_num") or "").strip()
    }

    # --- Load confirmed-memory entries (cached by labels mtime) ---
    memory_entries = _confirmed_label_memory_cached(str(set_num), int(bag))

    # Extend color_bgr_by_id to cover memory candidate colour IDs that may not be
    # in candidate_rows (so the colour gate has LEGO catalogue BGR for every memory part).
    _mem_color_ids = list({int(e["color_id"]) for e in memory_entries if e.get("color_id") is not None})
    _missing_mem_ids = [cid for cid in _mem_color_ids if cid not in color_bgr_by_id]
    if _missing_mem_ids:
        for _ecr in _load_catalog_colors_for_ids(_missing_mem_ids):
            _ecid = _coerce_int(_ecr.get("color_id"))
            _ecbgr = _slot_mask_hex_to_bgr(_ecr.get("rgb"))
            if _ecid is not None and _ecbgr is not None:
                color_bgr_by_id[int(_ecid)] = _ecbgr

    # --- Shape-score memory entries (first pass — no colour gate yet) ---
    _scored_memory: List[Any] = []
    if memory_entries:
        for entry in memory_entries:
            score = _compare_slot_profiles(query_profile, entry["profile"])
            if score >= _CONFIRMED_MEMORY_SUGGEST_THRESHOLD:
                _scored_memory.append((score, entry))
        _scored_memory.sort(key=lambda x: x[0], reverse=True)

    # --- Normal catalog candidates ---
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
                "source": "catalog",
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

    # --- Consensus colour from catalog top-3 (computed before memory gate so we can use it) ---
    _catalog_top3 = ranked[:3]
    _color_freq: Dict[int, int] = {}
    for _cv in _catalog_top3:
        if _cv.get("color_id") is not None:
            _cid = int(_cv["color_id"])
            _color_freq[_cid] = _color_freq.get(_cid, 0) + 1
    if _color_freq:
        _consensus_color_id = max(_color_freq, key=lambda k: _color_freq[k])
        _consensus_count = _color_freq[_consensus_color_id]
        _consensus_color_name = next(
            (c["color_name"] for c in _catalog_top3
             if int(c.get("color_id", 0) or 0) == _consensus_color_id),
            str(_consensus_color_id),
        )
    else:
        _consensus_color_id, _consensus_color_name, _consensus_count = None, None, 0

    # --- Slot colour evidence for the compatibility gate ---
    # Prefer catalog consensus (more reliable than raw pixel sampling) when strong enough.
    # Fall back to the query profile's median BGR when consensus is weak or absent.
    _slot_evidence_bgr: Optional[np.ndarray] = None
    if _consensus_color_id is not None and _consensus_count >= 2:
        _slot_evidence_bgr = color_bgr_by_id.get(int(_consensus_color_id))
    if _slot_evidence_bgr is None:
        _qbgr = query_profile.get("bgr")
        if _qbgr is not None:
            _slot_evidence_bgr = np.asarray(_qbgr, dtype=np.float32)

    # --- Colour-compatibility gate: filter memory candidates ---
    # Only show memory badge when the confirmed colour is plausible for this slot.
    # No hard filter when colour evidence is unavailable (be lenient in ambiguous cases).
    memory_candidates: List[Dict[str, Any]] = []
    memory_seen: set = set()
    _memory_before_colour_filter = len(_scored_memory)
    _memory_colour_filtered = 0

    for score, entry in _scored_memory:
        key = _candidate_part_key(entry["part_num"], entry["color_id"])
        if key in memory_seen:
            continue  # deduplicate same part+colour across multiple confirmed slots
        if _slot_evidence_bgr is not None:
            _mem_bgr = color_bgr_by_id.get(int(entry["color_id"]))
            if _mem_bgr is not None:
                _colour_dist = float(np.linalg.norm(
                    np.asarray(_slot_evidence_bgr, dtype=np.float32)
                    - np.asarray(_mem_bgr, dtype=np.float32)
                ))
                if _colour_dist > _MEMORY_COLOUR_COMPAT_DISTANCE:
                    _memory_colour_filtered += 1
                    continue  # colour incompatible — exclude from memory badges
        memory_seen.add(key)
        pool_meta = pool_lookup.get(key, {})
        memory_candidates.append(
            {
                "part_num": entry["part_num"],
                "color_id": int(entry["color_id"]),
                # Always use the user-confirmed color_name, not auto-detected.
                "color_name": str(entry["color_name"] or pool_meta.get("color_name") or "n/a"),
                "element_id": str(entry["element_id"] or pool_meta.get("element_id") or ""),
                "image_url": str(pool_meta.get("img_url") or "").strip(),
                "image_path": str(_slot_mask_resolve_local_image_path(pool_meta.get("img_url")) or ""),
                "confidence": round(score, 4),
                "source": "confirmed_memory",
                "source_crop_id": str(entry.get("source_crop_id") or ""),
                "source_slot_index": entry.get("source_slot_index"),
                "score_breakdown": {
                    "colour": 0.0,
                    "aspect": 0.0,
                    "silhouette": 0.0,
                    "candidate_image_available": bool(pool_meta.get("img_url")),
                },
            }
        )

    _memory_after_colour_filter = len(memory_candidates)

    # Remove catalog entries whose part+colour is already in memory_candidates.
    ranked_deduped = [
        c for c in ranked
        if _candidate_part_key(c["part_num"], c["color_id"]) not in memory_seen
    ]
    top = memory_candidates + ranked_deduped[:clip_k]

    print(
        "[slot-mask-candidates] "
        f"set={set_num} bag={int(bag)} crop_id={crop_id} slot_index={int(slot_index)} "
        f"pool={pool_source} candidates={len(candidate_rows)} "
        f"memory_before_colour={_memory_before_colour_filter} "
        f"memory_after_colour={_memory_after_colour_filter} "
        f"colour_filtered={_memory_colour_filtered} "
        f"clip_k={clip_k} returned={len(top)} "
        f"consensus={_consensus_color_name}({_consensus_count}/3)"
    )
    return {
        "ok": True,
        "set_num": set_num,
        "bag": int(bag),
        "crop_id": crop_id,
        "slot_index": int(slot_index),
        "candidate_pool_source": pool_source,
        "candidate_count": len(candidate_rows),
        "memory_candidate_count": _memory_after_colour_filter,
        "memory_before_colour_filter": _memory_before_colour_filter,
        "memory_after_colour_filter": _memory_after_colour_filter,
        "memory_colour_filtered": _memory_colour_filtered,
        "ranked_candidates": top,
        "consensus_color_id": _consensus_color_id,
        "consensus_color_name": _consensus_color_name,
        "consensus_color_count": int(_consensus_count),
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
    bcs_step_masked_path  = str(data.get("step_masked_path") or "").strip()
    bcs_part_cutout_path  = str(data.get("part_cutout_path") or "").strip()
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
        _ensure_catalog_image_for_pair,
    )

    temp_crop_path: Optional[Path] = None
    try:
        temp_crop_path = _write_ai_snap_temp_crop_image(crop)
        if temp_crop_path is None:
            raise HTTPException(status_code=400, detail="crop image unavailable")

        rank_input_path = str(temp_crop_path)
        if bcs_step_masked_path and Path(bcs_step_masked_path).is_file():
            rank_input_path = bcs_step_masked_path
            print(
                "[buildability-clip-suggest] using step_masked_path as query image crop_id=%s slot_index=%s query_image_path=%s"
                % (str(crop_id), str(slot_index), bcs_step_masked_path)
            )
        elif bcs_part_cutout_path and Path(bcs_part_cutout_path).is_file():
            rank_input_path = bcs_part_cutout_path
            print(
                "[buildability-clip-suggest] using part_cutout_path as query image crop_id=%s slot_index=%s query_image_path=%s"
                % (str(crop_id), str(slot_index), bcs_part_cutout_path)
            )
        elif selected_qty_box is not None:
            normalized_result = normalize_slot_crop_from_qty(str(temp_crop_path), selected_qty_box)
            if bool(normalized_result.get("ok")) and str(normalized_result.get("normalized_path") or "").strip():
                rank_input_path = str(normalized_result.get("normalized_path") or "").strip()

        if not Path(rank_input_path).is_file():
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
            required_qty = int(part.get("set_required_qty", part.get("qty", 0)) or 0)
            assigned_qty = int(assigned_totals.get(key, 0) or 0)
            remaining_qty = required_qty - assigned_qty
            if remaining_qty <= 0:
                continue
            image_path = _ensure_catalog_image_for_pair(part_num, color_id)
            if image_path is None or not image_path.exists():
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
                }
            )

        if not candidate_rows:
            return {"ok": True, "mode": "clip-placeholder", "ranked_candidates": []}

        # Use open_clip ViT-B-32/laion2b_s34b_b79k — same model that generated
        # catalog_clip_embeddings. Never use a2b_clip_match_probe._load_clip_model()
        # here; that attempts openai/clip-vit-base-patch32 via transformers (wrong space).
        clip_ok, clip_err = _clip_load()
        if not clip_ok:
            return {"ok": False, "mode": "clip-unavailable", "error": clip_err, "ranked_candidates": []}

        try:
            query_vec = _clip_embed_image(rank_input_path)  # (512,) L2-normalised
        except Exception as _qe:
            return {"ok": False, "mode": "clip-unavailable", "error": f"query embed failed: {_qe}", "ranked_candidates": []}

        _valid_rows: List[Dict[str, Any]] = []
        _cand_vecs: List[Any] = []
        for _row in candidate_rows:
            try:
                _vec = _clip_embed_image(_row["image_path"])
                _valid_rows.append(_row)
                _cand_vecs.append(_vec)
            except Exception:
                continue
        candidate_rows = _valid_rows
        if not candidate_rows:
            return {"ok": True, "mode": "clip-placeholder", "ranked_candidates": []}

        candidate_vectors = np.stack(_cand_vecs).astype(np.float32)  # (M, 512)
        query_vectors = query_vec.reshape(1, -1).astype(np.float32)   # (1, 512)

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
                        try:
                            memory_vec = _clip_embed_image(normalized_saved_path)  # (512,)
                        except Exception:
                            continue
                        memory_vector_rows = memory_vec.reshape(1, -1).astype(np.float32)
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


@router.post("/debug/next-unfilled-crop")
async def next_unfilled_crop_endpoint(req: Request):
    """Return the crop_id of the next crop (after from_crop_id) where filled_slots < total_slots.

    Uses fresh labels from disk and the backend's _crop_qty_slot_state (which includes the
    qty-signature fallback for old-style saves without explicit selected_slot_index), so the
    result is authoritative and not affected by stale frontend state.

    Body: {set_num, bag, from_crop_id, crop_ids: [{crop_id, qty_numbers, qty_text}]}
    The crop_ids list must be in the same display order the frontend uses for navigation.
    """
    data = await req.json()
    set_num = str(data.get("set_num") or "70618").strip() or "70618"
    try:
        bag = int(data.get("bag") or 1)
    except Exception:
        bag = 1
    from_crop_id = str(data.get("from_crop_id") or "").strip()
    candidates = list(data.get("crop_ids") or [])

    path = _label_store_path(set_num, bag)
    labels_payload = _load_existing_labels(path)
    saved_crops = dict(labels_payload.get("crops") or {})

    ids = [str((c or {}).get("crop_id") or "") for c in candidates]
    try:
        start = ids.index(from_crop_id) if from_crop_id else -1
    except ValueError:
        start = -1

    for item in candidates[start + 1:]:
        crop_id = str((item or {}).get("crop_id") or "").strip()
        if not crop_id:
            continue
        qty_numbers = _coerce_int_list((item or {}).get("qty_numbers") or [])
        qty_text = _coerce_str_list((item or {}).get("qty_text") or [])
        saved = dict(saved_crops.get(crop_id) or {})
        # Fall back to saved qty when the frontend passed nothing (e.g. crop with no qty override)
        if not qty_numbers:
            qty_numbers = _coerce_int_list(saved.get("qty") or [])
        if not qty_text:
            qty_text = _coerce_str_list(saved.get("qty_text") or [])
        parts = list(saved.get("parts") or [])
        slot_state = _crop_qty_slot_state({"parts": parts}, qty_numbers, qty_text)
        if (
            not slot_state.get("no_qty_detected")
            and int(slot_state.get("total_slots") or 0) > 0
            and int(slot_state.get("filled_slots") or 0) < int(slot_state.get("total_slots") or 0)
        ):
            return {"found": True, "crop_id": crop_id}

    return {"found": False, "crop_id": None}


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
    show_slot_matches: Optional[int] = Query(0),
    strong_match_threshold: Optional[float] = Query(0.72),
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
    # Persistent disk cache for crop detection (OCR + step detection).
    # Only used for full-bag loads (no step/page filter, no AI mode, no rebuild).
    _do_rebuild = int(rebuild or 0) == 1
    _cache_eligible = not _do_rebuild and step is None and page is None and not (int(ai or 0) == 1)
    crops = None
    if _cache_eligible:
        crops = _load_crop_detection_cache(str(set_num), bag_number)
        if crops is not None:
            print(f"[crop-cache] hit set={set_num} bag={bag_number} crops={len(crops)}")
        else:
            print(f"[crop-cache] miss set={set_num} bag={bag_number}")
    if crops is None:
        crops = _build_instruction_callout_crops(
            str(set_num),
            bag_number,
            ai_enabled=int(ai or 0) == 1,
            step_filter=step,
            page_filter=page,
            rebuild=_do_rebuild,
        )
        if _cache_eligible:
            _write_crop_detection_cache(str(set_num), bag_number, crops)
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
                    "selected_slot_index": normalized_part["selected_slot_index"],
                    "selected_qty_label": normalized_part["qty_text"] or (str(normalized_part["qty"]) if normalized_part["qty"] is not None else "none"),
                    "part_bbox": normalized_part["part_bbox"],
                    "confidence": normalized_part["confidence"],
                    "ai_snap_input_path": normalized_part.get("ai_snap_input_path"),
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
                    "selected_slot_index": normalized_part["selected_slot_index"],
                    "selected_qty_label": normalized_part["qty_text"] or (str(normalized_part["qty"]) if normalized_part["qty"] is not None else "none"),
                    "part_bbox": normalized_part["part_bbox"],
                    "confidence": normalized_part["confidence"],
                    "ai_snap_input_path": normalized_part.get("ai_snap_input_path"),
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
              data-lazy-src="{escape('/debug/manual-page-image?path=' + _url_quote(str(page_item['image_path']), safe='') + '&v=1')}"
              class="manual-page-lazy"
              alt="Full page {int(page_item['page'])}"
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
    _show_slot_matches_flag = 1 if int(show_slot_matches or 0) == 1 else 0
    _strong_match_threshold_val = max(0.0, min(1.0, float(strong_match_threshold if strong_match_threshold is not None else 0.72)))
    sam_refine_json = json.dumps(_sam_refine_flag)
    clip_k_json = json.dumps(_clip_k_val)
    fast_map_json = json.dumps(_fast_map_flag)
    show_slot_matches_json = json.dumps(bool(_show_slot_matches_flag))
    strong_match_threshold_json = json.dumps(_strong_match_threshold_val)
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
        /* ── Tab navigation ──────────────────────────────────────── */
        .tab-bar {{
          display: flex;
          gap: 6px;
          margin-bottom: 14px;
          border-bottom: 2px solid #d6dee8;
          padding-bottom: 0;
        }}
        .tab-btn {{
          border: 1px solid transparent;
          border-bottom: none;
          background: #f4f7fb;
          color: #536576;
          border-radius: 10px 10px 0 0;
          padding: 8px 20px;
          font-size: 13px;
          font-weight: 700;
          cursor: pointer;
          position: relative;
          bottom: -2px;
          transition: background 0.1s, color 0.1s;
        }}
        .tab-btn:hover {{
          background: #eaf1fb;
          color: #2f4153;
        }}
        .tab-btn.active {{
          background: #fff;
          color: #1947a6;
          border-color: #d6dee8;
          border-bottom-color: #fff;
        }}
        .tab-panel[hidden] {{
          display: none;
        }}
        /* ── Manual-page lazy placeholder ────────────────────────── */
        .manual-page-lazy {{
          display: block;
          width: 100%;
          background: #f0f4f8;
          min-height: 120px;
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
        .picker-slot-confirmed {{
          display: block;
          margin-top: 4px;
          color: #155724;
          font-size: 11px;
          font-weight: 600;
        }}
        .picker-slot-confirmed-swatch {{
          display: inline-block;
          width: 10px;
          height: 10px;
          border: 1px solid #999;
          border-radius: 2px;
          vertical-align: middle;
          margin-right: 3px;
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
        .picker-slot-candidate.strong {{
          color: #155724;
        }}
        .picker-slot-candidate.weak {{
          color: #856404;
          opacity: 0.72;
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
        .picker-slot-candidate-score {{
          font-size: 8px;
          font-weight: 700;
        }}
        .picker-slot-candidate-memory {{
          border: 1px solid #b8d4f0;
          border-radius: 6px;
          background: #eef5fc;
          padding: 2px;
        }}
        .picker-slot-candidate-memory img {{
          border-color: #7ab3e0;
        }}
        .picker-slot-memory-badge {{
          display: inline-block;
          background: #1a73e8;
          color: #fff;
          font-size: 8px;
          font-weight: 700;
          padding: 1px 4px;
          border-radius: 3px;
          letter-spacing: 0.02em;
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
        .picker-slot-predicted-thumbs {{
          display: flex;
          gap: 4px;
          justify-content: center;
          margin-bottom: 3px;
        }}
        .picker-slot-ref-img {{
          width: 48px;
          height: 48px;
          object-fit: contain;
          border: 1px solid #ffc107;
          border-radius: 6px;
          background-color: #fff;
          background-image:
            linear-gradient(45deg, #edf1f5 25%, transparent 25%),
            linear-gradient(-45deg, #edf1f5 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #edf1f5 75%),
            linear-gradient(-45deg, transparent 75%, #edf1f5 75%);
          background-size: 12px 12px;
          background-position: 0 0, 0 6px, 6px -6px, -6px 0;
        }}
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

        <!-- Tab navigation bar -->
        <div class="tab-bar" role="tablist">
          <button
            type="button"
            class="tab-btn active"
            id="tab-btn-mapping"
            role="tab"
            aria-selected="true"
            aria-controls="tab-mapping"
            onclick="switchTab('mapping')"
          >Mapping</button>
          <button
            type="button"
            class="tab-btn"
            id="tab-btn-manual"
            role="tab"
            aria-selected="false"
            aria-controls="tab-manual"
            onclick="switchTab('manual')"
          >Manual Crop Mode</button>
        </div>

        <!-- Tab B: Manual Crop Mode (hidden until activated) -->
        <div id="tab-manual" class="tab-panel" hidden>
          <section class="panel">
            <h2>Manual Crop Mode</h2>
            <p class="save-note">Drag a rectangle on any page, enter the step number, then save the manual crop as training ground truth.</p>
            <div class="manual-pages-grid">
              {manual_pages_html}
            </div>
          </section>
        </div>

        <!-- Tab A: Detected Callout Crops + Part Library (default) -->
        <div id="tab-mapping" class="tab-panel">
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
        </div><!-- end #tab-mapping -->
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
        const SHOW_SLOT_MATCHES = {show_slot_matches_json};
        const SAM_REFINE = {sam_refine_json};
        const SLOT_MATCH_K = {clip_k_json};
        const FAST_MAP = {fast_map_json};
        const SLOT_MATCH_STRONG_THRESHOLD = {strong_match_threshold_json};
        const SET_NUM = {json.dumps(str(set_num))};
        const BAG_NUM = {bag_number};
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
        let manualTabLoaded = false;

        // ── Tab switching ──────────────────────────────────────────────────
        function activateManualTab() {{
          if (manualTabLoaded) return;
          let count = 0;
          document.querySelectorAll("img.manual-page-lazy").forEach(function(img) {{
            const lazySrc = img.dataset.lazySrc || "";
            if (lazySrc) {{
              img.src = lazySrc;
              count++;
            }}
          }});
          console.log("manual lazy images loaded", count);
          manualTabLoaded = true;
        }}

        function switchTab(name) {{
          const tabs = ["mapping", "manual"];
          tabs.forEach(function(t) {{
            const panel = document.getElementById("tab-" + t);
            const btn   = document.getElementById("tab-btn-" + t);
            if (!panel || !btn) return;
            const active = (t === name);
            if (active) {{
              panel.removeAttribute("hidden");
            }} else {{
              panel.setAttribute("hidden", "");
            }}
            btn.classList.toggle("active", active);
            btn.setAttribute("aria-selected", active ? "true" : "false");
          }});
          if (name === "manual") activateManualTab();
        }}

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

        // Return the RGB object {{r,g,b}} for a LEGO color_id, or null if not found.
        function legoColorRgb(colorId) {{
          const c = (window.legoColors || []).find((x) => x && Number(x.color_id) === Number(colorId));
          return c ? parseRgbHex(c.rgb) : null;
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

          // Slots filled = unique confirmed selected_slot_index values only.
          // Parts without an explicit selected_slot_index are not counted against a slot.
          const filledSlotIndices = new Set();
          parts.forEach((part) => {{
            const explicitSlotIndex = Number(part && part.selected_slot_index);
            if (Number.isInteger(explicitSlotIndex) && explicitSlotIndex >= 0 && explicitSlotIndex < sequence.length) {{
              filledSlotIndices.add(explicitSlotIndex);
            }}
          }});

          let nextSlot = null;
          let nextQtyIndex = sequence.length;
          for (let currentIndex = 0; currentIndex < sequence.length; currentIndex += 1) {{
            if (!filledSlotIndices.has(currentIndex)) {{
              nextSlot = sequence[currentIndex];
              nextQtyIndex = currentIndex;
              break;
            }}
          }}

          return {{
            totalSlots: sequence.length,
            filledSlots: filledSlotIndices.size,
            slotsFull: nextSlot === null,
            noQtyDetected: false,
            nextQtyLabel: nextSlot ? String(nextSlot.qty_text || nextSlot.qty || "none") : "filled",
            nextQtyValue: nextSlot && Number.isFinite(Number(nextSlot.qty)) ? Number(nextSlot.qty) : null,
            nextQtyIndex,
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
            && !confirmedPartForSlot(crop, explicitSlotIndex)
          ) {{
            return explicitSlotIndex;
          }}
          return Math.max(0, Math.min(Number(slotState.nextQtyIndex ?? slotState.filledSlots), slotState.totalSlots - 1));
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
          const memoryCount = candidates.filter((c) => c && c.source === "confirmed_memory").length;
          return '<span class="picker-slot-candidates">' + candidates.slice(0, SLOT_MATCH_K + memoryCount).map((candidate) => {{
            const imageUrl = String(candidate && (candidate.image_url || candidate.img_url || "") || "");
            const confidence = Number(candidate && candidate.confidence);
            const isMemory = candidate && candidate.source === "confirmed_memory";
            const label = String(candidate && candidate.part_num || "") + ":" + String(candidate && candidate.color_id || "");
            const isStrong = Number.isFinite(confidence) && confidence >= SLOT_MATCH_STRONG_THRESHOLD;
            const memoryBadge = isMemory ? '<span class="picker-slot-memory-badge" title="confirmed in this bag">memory</span>' : '';
            return '<span role="button" tabindex="0" class="picker-slot-candidate ' + (isStrong ? 'strong' : 'weak') + (isMemory ? ' picker-slot-candidate-memory' : '') + '" data-slot-suggestion="true" data-slot-index="' + escapeHtml(String(maskSlot.slot_index ?? "")) + '" data-part-num="' + escapeHtml(String(candidate && candidate.part_num || "")) + '" data-color-id="' + escapeHtml(String(candidate && candidate.color_id || 0)) + '" data-element-id="' + escapeHtml(String(candidate && candidate.element_id || "")) + '" data-color-name="' + escapeHtml(String(candidate && candidate.color_name || "")) + '" title="' + escapeHtml(label + (Number.isFinite(confidence) ? " " + confidence.toFixed(2) : "") + (isStrong ? " strong" : " weak") + (isMemory ? " [memory]" : "")) + '">'
              + (imageUrl ? '<img src="' + escapeHtml(imageUrl) + '" alt="' + escapeHtml(label) + '" loading="lazy" />' : '')
              + '<span>' + escapeHtml(label) + '</span>'
              + (Number.isFinite(confidence) ? '<span class="picker-slot-candidate-score">' + escapeHtml(confidence.toFixed(2)) + (isStrong ? ' strong' : '') + '</span>' : '')
              + memoryBadge
              + '</span>';
          }}).join("") + '</span>';
        }}

        function confirmedPartForSlot(crop, slotIndex) {{
          // Only restore by explicit selected_slot_index — never by array position.
          const parts = Array.isArray(crop && crop.parts) ? crop.parts : [];
          return parts.find((part) => Number(part && part.selected_slot_index) === Number(slotIndex)) || null;
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
            const slotStateFull = computeCropSlotState(crop);
            status.textContent = (slotStateFull.slotsFull && !slotStateFull.noQtyDetected)
              ? "All qty slots filled."
              : "No open slot available for AI Snap.";
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
              ${{(() => {{
                const cp = confirmedPartForSlot(crop, idx);
                if (!cp) return '';
                const cpRgb = legoColorRgb(Number(cp.color_id || 0));
                const swatchStyle = cpRgb ? 'background:rgb(' + cpRgb.r + ',' + cpRgb.g + ',' + cpRgb.b + ')' : 'background:#ccc';
                const cpColorLabel = cp.color_name || String(cp.color_id || '');
                return '<span class="picker-slot-confirmed">'
                  + '<span class="picker-slot-confirmed-swatch" style="' + swatchStyle + '"></span>'
                  + escapeHtml(String(cp.part_num || '')) + ' · ' + escapeHtml(cpColorLabel)
                  + '</span>';
              }})()}}
              ${{(() => {{
                const maskSlot = autoMaskSlots.get(Number(idx));
                const cutoutUrl = maskSlot ? aiSnapArtifactUrl(maskSlot.part_cutout_path, maskSlot.generated_at) : "";
                const slotOverlayUrl = maskSlot ? aiSnapArtifactUrl(maskSlot.slot_window_overlay_path, maskSlot.generated_at) : "";
                // Predicted badge — shown only when no confirmed part exists for this slot.
                const predSlot = _maskSlotPrediction(idx);
                const predictedHtml = predSlot
                  ? (() => {{
                      const refPath = String((predSlot && predSlot.prediction_reference_path) || "").trim();
                      const refUrl = refPath ? aiSnapArtifactUrl(refPath) : "";
                      const thumbHtml = refUrl
                        ? '<span class="picker-slot-predicted-thumbs">'
                          + '<img class="picker-slot-ref-img" src="' + escapeHtml(refUrl) + '" alt="reference" loading="lazy" />'
                          + '</span>'
                        : '';
                      return '<span class="picker-slot-predicted">'
                        + thumbHtml
                        + escapeHtml(String(predSlot.predicted_part.part_num || ""))
                        + ':' + escapeHtml(String(predSlot.predicted_part.color_id || ""))
                        + ' sim ' + escapeHtml(String((predSlot.prediction_similarity || 0).toFixed(2)))
                        + '<span class="picker-slot-predicted-actions">'
                        + '<button type="button" class="predicted-accept-btn" onclick="acceptPredictedSlot(event,' + idx + ')">Accept</button>'
                        + '<button type="button" class="predicted-reject-btn" onclick="rejectSlotPrediction(event,' + idx + ')">Reject</button>'
                        + '</span>'
                        + '</span>';
                    }})()
                  : '';
                let colorHtml = "";
                if (maskSlot && maskSlot.slot_rgb_median) {{
                  colorHtml = '<span class="picker-slot-color">'
                    + 'colour guess: experimental'
                    + '</span>';
                }}
                let lowAlphaHtml = "";
                if (maskSlot && String(maskSlot.status || "") === "needs_review_low_alpha") {{
                  lowAlphaHtml = '<span class="picker-slot-low-alpha">⚠ low alpha</span>';
                }}
                let confidenceHtml = "";
                if (maskSlot && maskSlot.slot_confidence) {{
                  // Downgrade to "medium" when shape is good but colour sampling is unreliable.
                  const rawConf = String(maskSlot.slot_confidence);
                  const colourConf = String(maskSlot.slot_colour_confidence || "");
                  const conf = (rawConf === "high" && colourConf === "low") ? "medium" : rawConf;
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
              if (!Number.isInteger(nextIndex) || nextIndex >= slotState.totalSlots || confirmedPartForSlot(crop, nextIndex)) {{
                return;
              }}
              crop.ai_snap_selected_slot_index = nextIndex;
              if (crop.ai_snap_result && Number(crop.ai_snap_result.slot_index) !== Number(nextIndex)) {{
                crop.ai_snap_result = null;
              }}
              crop.ai_snap_error = "";
              // Capture the maskSlot for this index before re-rendering (closure would use
              // the pre-render autoMaskSlots map, which is fine since objects are shared).
              const clickedMaskSlot = autoMaskSlots.get(nextIndex);
              renderBuildabilitySlots(crop.crop_id);
              renderSuggestedParts(crop.crop_id);
              renderAiSnapStatus(crop);
              // If this slot already has a cutout but no candidates yet, kick off matching.
              if (clickedMaskSlot && SHOW_SLOT_MATCHES) {{
                loadSlotMaskCandidates(crop, clickedMaskSlot);
              }}
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

        // Scroll the "Detected slots in crop" panel back into view after DOM updates.
        // Uses requestAnimationFrame so the browser has committed the re-render before
        // we reposition.  block:"nearest" means no scroll happens if panel is already visible.
        function scrollToActiveSlotPanel() {{
          requestAnimationFrame(function() {{
            const panel = document.querySelector(".picker-slots-panel");
            if (panel) {{
              const y = panel.getBoundingClientRect().top + window.scrollY - 120;
              window.scrollTo({{ top: Math.max(0, y), behavior: "smooth" }});
            }}
          }});
        }}

        async function goToNextUnfilledCrop() {{
          // Ask the backend for the next unfilled crop so we always use authoritative
          // saved-label state rather than potentially-stale in-memory frontend state.
          const visibleCrops = cropRecords
            .filter((c) => cropMap.has(c.crop_id) && isCropVisibleInCurrentView(c))
            .map((c) => ({{
              crop_id: c.crop_id,
              qty_numbers: Array.isArray(c.qty_numbers) ? c.qty_numbers : [],
              qty_text: Array.isArray(c.qty_text) ? c.qty_text : [],
            }}));
          let result;
          try {{
            const res = await fetch("/debug/next-unfilled-crop", {{
              method: "POST",
              headers: {{"Content-Type": "application/json"}},
              body: JSON.stringify({{
                set_num: SET_NUM,
                bag: BAG_NUM,
                from_crop_id: activeCropId || "",
                crop_ids: visibleCrops,
              }}),
            }});
            if (!res.ok) throw new Error("server error");
            result = await res.json();
          }} catch (_err) {{
            alert("Failed to find next unfilled crop.");
            return;
          }}
          if (result && result.found && result.crop_id) {{
            selectCrop(result.crop_id);
            scrollToActiveSlotPanel();
            setTimeout(scrollToActiveSlotPanel, 150);
          }} else {{
            alert("No unfilled crops remaining.");
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
          const _aiSnapMaskSlot = Array.isArray(crop.auto_mask_slots) ? (crop.auto_mask_slots[slotIndex] || null) : null;
          const payload = {{
            set_num: {json.dumps(str(set_num))},
            bag: {bag_number},
            crop_id: crop.crop_id,
            slot_index: slotIndex,
            manual_color_filter_id: Number.isFinite(Number(crop.manual_color_filter_id))
              ? Number(crop.manual_color_filter_id)
              : null,
            picked_rgb: crop.picked_rgb || null,
            step_masked_path: (_aiSnapMaskSlot && _aiSnapMaskSlot.step_masked_path) ? String(_aiSnapMaskSlot.step_masked_path) : "",
            part_cutout_path: (_aiSnapMaskSlot && _aiSnapMaskSlot.part_cutout_path) ? String(_aiSnapMaskSlot.part_cutout_path) : "",
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
          scrollToActiveSlotPanel();
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
          updateCropCardVisuals(crop.crop_id);
          scrollToActiveSlotPanel();
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
          updateCropCardVisuals(crop.crop_id);
          scrollToActiveSlotPanel();
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

          // Bind to the currently open slot so the backend stores selected_slot_index
          // and the frontend can restore by (crop_id, selected_slot_index) on refresh.
          const slotIndex = currentBuildabilitySlotIndex(crop);
          const sequence = buildClientQtySequence(crop);
          const seqSlot = (slotIndex !== null && Number.isInteger(slotIndex)) ? sequence[slotIndex] : null;
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
            selected_slot_index: slotIndex,
            qty: seqSlot ? (seqSlot.qty ?? null) : null,
            qty_text: seqSlot ? (seqSlot.qty_text || null) : null,
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
            renderBuildabilitySlots(crop.crop_id);
            updateCropCardVisuals(crop.crop_id);
            scrollToActiveSlotPanel();
            if (activeCropId === crop.crop_id) {{
              renderSuggestedParts(crop.crop_id);
              renderColourPicker(crop.crop_id);
              renderAiSnapStatus(crop);
            }}
            updatePartTileAssignmentState();
            status.textContent = "Saved " + crop.crop_id + " slot " + (slotIndex !== null ? slotIndex + 1 : "?") + " -> " + partNum + " / color " + colorId;
            const updatedSlotState = computeCropSlotState(crop);
            if (!updatedSlotState.noQtyDetected && updatedSlotState.slotsFull) {{
              const nextCropId = nextVisibleCropId(crop.crop_id);
              if (nextCropId) {{
                selectCrop(nextCropId);
                scrollToActiveSlotPanel();
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
    crops = _load_crop_detection_cache(str(set_num), bag_number)
    if crops is None:
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


# ─── Element Page Extractor ────────────────────────────────────────────────────


def _element_page_ocr(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Run pytesseract on the full image and return classified + raw tokens.

    Returns:
      qty_tokens      – {text, qty, x, y, w, h, cx, cy, conf}
      elem_tokens     – {element_id, raw_text, x, y, w, h, cx, cy, conf, digit_len}
      raw_ocr_tokens  – every non-empty token from every pass
                        {text, x, y, w, h, conf, pass}
      image_h, image_w
    """
    import pytesseract

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]

    scale = 2.0
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Inverted threshold – catches dark-background pages
    thresh_inv = cv2.bitwise_not(thresh)

    qty_tokens: List[Dict[str, Any]] = []
    elem_tokens: List[Dict[str, Any]] = []
    raw_ocr_tokens: List[Dict[str, Any]] = []
    seen_raw: set = set()
    seen_qty: set = set()
    seen_elem: set = set()

    # Each variant: (image, inv_scale, psm, config_extra, pass_label)
    variants = [
        (big,       1.0 / scale, 11, "-c tessedit_char_whitelist=0123456789xX", "big_psm11"),
        (thresh,    1.0 / scale, 11, "-c tessedit_char_whitelist=0123456789xX", "thresh_psm11"),
        (thresh_inv,1.0 / scale, 11, "-c tessedit_char_whitelist=0123456789xX", "thresh_inv_psm11"),
        (big,       1.0 / scale,  6, "-c tessedit_char_whitelist=0123456789xX", "big_psm6"),
        # Digit-only pass – best recall for pure numeric element IDs
        (big,       1.0 / scale, 11, "-c tessedit_char_whitelist=0123456789",   "big_digits_psm11"),
        (thresh,    1.0 / scale, 11, "-c tessedit_char_whitelist=0123456789",   "thresh_digits_psm11"),
    ]

    for variant_img, inv_scale, psm, cfg_extra, pass_label in variants:
        cfg = f"--psm {psm} {cfg_extra}"
        try:
            data = pytesseract.image_to_data(
                variant_img, config=cfg, output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue
        n = len(data.get("text") or [])
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            conf = float(data["conf"][i] or -1)
            bx = int(int(data["left"][i]  or 0) * inv_scale)
            by = int(int(data["top"][i]   or 0) * inv_scale)
            bw = int(int(data["width"][i] or 0) * inv_scale)
            bh = int(int(data["height"][i]or 0) * inv_scale)
            if bw <= 0 or bh <= 0:
                continue

            # ── raw token (deduplicated by position + text) ─────────────────
            raw_key = (text, int(bx // 6), int(by // 6))
            if raw_key not in seen_raw:
                seen_raw.add(raw_key)
                raw_ocr_tokens.append({
                    "text": text,
                    "x": bx, "y": by, "w": bw, "h": bh,
                    "conf": conf,
                    "pass": pass_label,
                })

            norm = re.sub(r"\s+", "", text.lower())

            # ── qty token: "3x" / "x3" ──────────────────────────────────────
            if re.fullmatch(r"\d+x|x\d+", norm):
                digits = re.sub(r"x", "", norm)
                try:
                    qty_val = int(digits)
                except ValueError:
                    qty_val = 1
                key = (int(bx // 8), int(by // 8))
                if key not in seen_qty:
                    seen_qty.add(key)
                    qty_tokens.append({
                        "text": norm,
                        "qty": qty_val,
                        "x": bx, "y": by, "w": bw, "h": bh,
                        "cx": bx + bw // 2,
                        "cy": by + bh // 2,
                        "conf": conf,
                    })
                continue

            # ── element candidate: 6-8 digit numeric token ──────────────────
            digits_only = re.sub(r"[^0-9]", "", text)
            dlen = len(digits_only)
            if 6 <= dlen <= 8 and conf > 10:
                key = (int(bx // 8), int(by // 8))
                if key not in seen_elem:
                    seen_elem.add(key)
                    elem_tokens.append({
                        "element_id": digits_only,
                        "raw_text": text,
                        "x": bx, "y": by, "w": bw, "h": bh,
                        "cx": bx + bw // 2,
                        "cy": by + bh // 2,
                        "conf": conf,
                        "digit_len": dlen,
                    })

    return {
        "qty_tokens": qty_tokens,
        "elem_tokens": elem_tokens,
        "raw_ocr_tokens": raw_ocr_tokens,
        "image_h": img_h,
        "image_w": img_w,
    }


def _find_qty_above_element(
    elem: Dict[str, Any],
    qty_tokens: List[Dict[str, Any]],
    col_half_w: int = 90,
    max_search_above_px: int = 300,
    min_thumb_gap_px: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    Return the qty token that is directly above this element number and in
    the same horizontal column.

    Criteria:
      • qty centre_x within col_half_w of element centre_x
      • qty bottom edge is above element top with at least min_thumb_gap_px
        gap (to ensure there is room for the thumbnail between them)
      • qty is within max_search_above_px of element top
      • Among candidates, pick the one with gap closest to the element
        (smallest vertical gap, but still >= min_thumb_gap_px).
    """
    elem_cx = elem["x"] + elem["w"] // 2
    elem_top = elem["y"]

    best_token = None
    best_gap = float("inf")

    for qt in qty_tokens:
        qt_cx = qt["x"] + qt["w"] // 2
        qt_bottom = qt["y"] + qt["h"]

        # Must be in the same column
        if abs(qt_cx - elem_cx) > col_half_w:
            continue
        # Must be above the element with enough room for a thumbnail
        gap = elem_top - qt_bottom
        if gap < min_thumb_gap_px or gap > max_search_above_px:
            continue

        if gap < best_gap:
            best_gap = gap
            best_token = qt

    return best_token


def _find_part_thumbnail_above(
    img_bgr: np.ndarray,
    elem_x: int,
    elem_y: int,
    elem_w: int,
    elem_h: int,
    search_top: Optional[int] = None,    # explicit top boundary (qty_bottom + 4)
    search_bottom: Optional[int] = None, # explicit bottom boundary (elem_y - 4)
    # Fallback window when qty bbox is absent
    fallback_max_gap_px: int = 120,
    max_thumb_w: int = 180,
    max_thumb_h: int = 160,
    min_area: int = 150,
    pad: int = 6,
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    """
    Return ((x, y, w, h), window_source) of the single part thumbnail.

    When search_top / search_bottom are provided (qty-bounded window):
      - The search region is exactly [search_top .. search_bottom] vertically
        and [elem_cx-90 .. elem_cx+90] horizontally.
      - Text boxes are already excluded by those boundaries; no post-trim needed.

    Fallback (no qty box found):
      - Search [elem_y - fallback_max_gap_px .. elem_y - 10].

    Returns the padded bbox of the best component, plus a string describing
    which window was used ('qty_bounded' or 'fallback').
    """
    img_h, img_w = img_bgr.shape[:2]
    elem_cx = elem_x + elem_w // 2
    half_w = 90

    band_x1 = max(0, elem_cx - half_w)
    band_x2 = min(img_w, elem_cx + half_w)

    if search_top is not None and search_bottom is not None:
        band_y1 = max(0, search_top)
        band_y2 = min(img_h, search_bottom)
        # If the bounded window is too small to contain a thumbnail, fall back
        if band_y2 - band_y1 < 15:
            band_y2 = max(0, elem_y - 10)
            band_y1 = max(0, elem_y - fallback_max_gap_px)
            window_source = "fallback_small_window"
        else:
            window_source = "qty_bounded"
    else:
        band_y2 = max(0, elem_y - 10)
        band_y1 = max(0, elem_y - fallback_max_gap_px)
        window_source = "fallback"

    if band_y1 >= band_y2 or band_x1 >= band_x2:
        return None, window_source

    region = img_bgr[band_y1:band_y2, band_x1:band_x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 228, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)

    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    best: Optional[Tuple[int, int, int, int]] = None
    best_score = float("inf")

    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        rx = int(stats[lbl, cv2.CC_STAT_LEFT])
        ry = int(stats[lbl, cv2.CC_STAT_TOP])
        rw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        rh = int(stats[lbl, cv2.CC_STAT_HEIGHT])

        if rw > max_thumb_w or rh > max_thumb_h:
            continue

        comp_bottom_abs = band_y1 + ry + rh
        comp_cx_abs = band_x1 + rx + rw // 2

        vert_dist = elem_y - comp_bottom_abs
        if vert_dist < 0:
            continue

        horiz_dist = abs(comp_cx_abs - elem_cx)
        score = vert_dist + horiz_dist * 0.3

        if score < best_score:
            best_score = score
            best = (band_x1 + rx, band_y1 + ry, rw, rh)

    if best is None:
        return None, window_source

    bx, by, bw, bh = best
    bx = max(0, bx - pad)
    by = max(0, by - pad)
    bx2 = min(img_w, bx + bw + pad * 2)
    by2 = min(img_h, by + bh + pad * 2)
    # In qty-bounded mode, don't let padding push the crop above the window top —
    # that would pull in the qty text we specifically excluded.
    if window_source == "qty_bounded":
        by = max(by, band_y1)
    return (bx, by, bx2 - bx, by2 - by), window_source


def _trim_crop_from_ocr_tokens(
    bx: int,
    by: int,
    bw: int,
    bh: int,
    ocr_tokens: List[Dict[str, Any]],
    pad: int = 5,
) -> Tuple[Tuple[int, int, int, int], bool, str]:
    """
    Given a crop bbox (bx, by, bw, bh) and a list of OCR tokens, trim any
    crop edge that overlaps a text box.

    Returns (trimmed_bbox, text_removed, trim_reason).

    Trimming rules
    --------------
    For each token that overlaps the crop:
      • Token centre in the top 40% of crop → trim the crop's top edge down
        to just below the token bottom.
      • Token centre in the bottom 40% of crop → trim the crop's bottom edge
        up to just above the token top.
      • Thin horizontal slivers outside those bands (centre in middle 20%)
        are ignored – they're probably part of the image itself.

    After trimming all tokens the crop is padded inward by `pad` px on any
    adjusted edge, then all four edges are padded outward by `pad` px.
    """
    bx2 = bx + bw
    by2 = by + bh
    reasons: List[str] = []

    for tok in ocr_tokens:
        tx, ty, tw, th = tok["x"], tok["y"], tok["w"], tok["h"]
        tx2, ty2 = tx + tw, ty + th

        # Quick overlap check
        if tx2 <= bx or tx < bx2 and ty2 <= by or ty >= by2:
            continue
        ox1 = max(bx, tx)
        oy1 = max(by, ty)
        ox2 = min(bx2, tx2)
        oy2 = min(by2, ty2)
        if ox2 <= ox1 or oy2 <= oy1:
            continue

        # Horizontal overlap fraction – only act if the text overlaps at
        # least 25% of the crop width (avoids trimming on tiny corner touches)
        horiz_overlap = (ox2 - ox1) / max(1, bw)
        if horiz_overlap < 0.20:
            continue

        # Token centre position relative to crop height (0 = top, 1 = bottom)
        tok_cy = (ty + ty2) / 2.0
        rel_y = (tok_cy - by) / max(1, by2 - by)

        label = tok.get("text", "?")
        if rel_y < 0.40:
            # Text is in the top portion → trim top edge down past token bottom
            new_top = ty2 + pad
            if new_top > by:
                by = new_top
                reasons.append(f"top↓ past '{label}' (rel={rel_y:.2f})")
        elif rel_y > 0.60:
            # Text is in the bottom portion → trim bottom edge up past token top
            new_bot = ty - pad
            if new_bot < by2:
                by2 = new_bot
                reasons.append(f"bot↑ past '{label}' (rel={rel_y:.2f})")
        # Middle 20%: leave it – likely part of the image itself

    # Sanity: ensure crop still has positive area
    if by >= by2 or bx >= bx2:
        return (bx, by, bw, bh), False, "trim_collapsed_reverted"

    # Add uniform pad outward on all edges (constrained to original crop)
    final_bx = max(bx - pad, bx)
    final_by = max(by - pad, by)
    final_bx2 = min(bx2 + pad, bx + bw)
    final_by2 = min(by2 + pad, by + bh)

    trimmed = (final_bx, final_by, final_bx2 - final_bx, final_by2 - final_by)
    text_removed = bool(reasons)
    return trimmed, text_removed, "; ".join(reasons)


def _detect_crop_bg_color(crop: np.ndarray) -> Tuple[int, int, int]:
    """
    Estimate the background colour of a crop by taking the median of its
    1-pixel border ring (top row, bottom row, left col, right col).
    Returns (B, G, R) in OpenCV channel order.
    Falls back to white when the crop is too small.
    """
    h, w = crop.shape[:2]
    if h < 3 or w < 3:
        return (255, 255, 255)
    border = np.concatenate([
        crop[0, :],          # top row
        crop[-1, :],         # bottom row
        crop[1:-1, 0],       # left column (excl. corners)
        crop[1:-1, -1],      # right column (excl. corners)
    ], axis=0)
    median = np.median(border, axis=0).astype(int)
    return (int(median[0]), int(median[1]), int(median[2]))


def _blank_text_regions_in_crop(
    crop: np.ndarray,
    crop_x: int,
    crop_y: int,
    tokens: List[Dict[str, Any]],
    bg_color: Tuple[int, int, int],
) -> Tuple[np.ndarray, bool, List[str]]:
    """
    Fill OCR token boxes that overlap the crop with bg_color.

    Parameters
    ----------
    crop     : the already-extracted crop image (numpy BGR array)
    crop_x, crop_y : top-left of the crop in the original image coordinate system
    tokens   : list of OCR token dicts with 'x','y','w','h','text' in
               original-image coordinates
    bg_color : (B, G, R) fill colour

    Returns
    -------
    (cleaned_crop, text_was_removed, list_of_removed_text_strings)
    """
    crop_h, crop_w = crop.shape[:2]
    out = crop.copy()
    removed: List[str] = []

    for tok in tokens:
        tx, ty, tw, th = tok["x"], tok["y"], tok["w"], tok["h"]
        # Intersection of token box and crop (in original-image coords)
        ix1 = max(tx, crop_x) - crop_x
        iy1 = max(ty, crop_y) - crop_y
        ix2 = min(tx + tw, crop_x + crop_w) - crop_x
        iy2 = min(ty + th, crop_y + crop_h) - crop_y
        if ix2 > ix1 and iy2 > iy1:
            out[iy1:iy2, ix1:ix2] = bg_color
            removed.append(str(tok.get("text") or ""))

    return out, bool(removed), removed


def _nearest_qty_for_element(
    elem: Dict[str, Any],
    qty_tokens: List[Dict[str, Any]],
    max_dist_px: int = 250,
) -> Optional[int]:
    """Return the qty value of the closest qty token to this element number."""
    ex, ey = float(elem["cx"]), float(elem["cy"])
    best_dist = float(max_dist_px)
    best_qty: Optional[int] = None
    for qt in qty_tokens:
        dx = float(qt["cx"]) - ex
        dy = float(qt["cy"]) - ey
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_qty = qt["qty"]
    return best_qty


# ── SAM2 helpers ──────────────────────────────────────────────────────────────

def _sam2_load() -> Tuple[bool, str]:
    """
    Lazy-load SAM2 processor + model via HuggingFace Transformers.
    Returns (ok, error_message).  Caches result in module-level globals.
    """
    global _sam2_processor, _sam2_model, _sam2_load_error
    if _sam2_model is not None:
        return True, ""
    if _sam2_load_error is not None:
        return False, _sam2_load_error
    try:
        from transformers import Sam2Processor, Sam2Model  # type: ignore
        _sam2_processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-tiny")
        _sam2_model = Sam2Model.from_pretrained("facebook/sam2-hiera-tiny")
        _sam2_model.eval()
        return True, ""
    except Exception as exc:
        _sam2_load_error = str(exc)
        return False, _sam2_load_error


def _sam2_segment_crop(
    img_bgr: np.ndarray,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Run SAM2 on an already-extracted BGR crop.

    Strategy
    --------
    Upscale to at least 512 px on the longest side.
    Prompt with a 5-point cross pattern centred on the image.

    Returns
    -------
    mask       : uint8 (H, W) array, 255 = foreground
    best_iou   : IOU score of the chosen mask
    all_scores : IOU scores for all candidate masks
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image as _Pil

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = _Pil.fromarray(img_rgb)
    h0, w0  = img_bgr.shape[:2]

    scale  = max(512 / max(w0, h0), 1.0)
    nw, nh = max(int(w0 * scale), 1), max(int(h0 * scale), 1)
    img_up = img_pil.resize((nw, nh), _Pil.LANCZOS)

    cx, cy = nw // 2, nh // 2
    pts    = [[cx, cy],
              [cx - nw // 4, cy], [cx + nw // 4, cy],
              [cx, cy - nh // 4], [cx, cy + nh // 4]]
    inputs = _sam2_processor(
        images=img_up,
        input_points=[[ pts ]],
        input_labels=[[ [1] * len(pts) ]],
        return_tensors="pt",
    )

    with torch.no_grad():
        out = _sam2_model(**inputs)

    pred       = out.pred_masks[0, 0]      # [num_masks, 256, 256]
    iou_t      = out.iou_scores[0, 0]      # [num_masks]
    all_scores = [round(float(s), 4) for s in iou_t.tolist()]
    best       = int(iou_t.argmax().item())
    best_iou   = float(iou_t[best].item())

    logits  = pred[best].unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(logits, size=(nh, nw),
                            mode="bilinear", align_corners=False)[0, 0]
    mask_up  = (resized > 0).numpy().astype(np.uint8) * 255
    from PIL import Image as _Pil2
    mask_out = np.array(
        _Pil2.fromarray(mask_up).resize((w0, h0), _Pil2.NEAREST)
    )
    return mask_out, best_iou, all_scores


def _cv_fallback_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    OpenCV fallback segmentation.
    Detects background colour from the border ring, thresholds by colour
    distance, then returns the largest connected foreground component.
    Returns uint8 (H, W) mask with 255 = foreground.
    """
    arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w = arr.shape[:2]
    border = np.concatenate([
        arr[0, :], arr[-1, :], arr[1:-1, 0], arr[1:-1, -1]
    ], axis=0)
    bg = np.median(border, axis=0)
    diff   = np.abs(arr - bg).max(axis=2)
    binary = (diff > 20).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return binary
    areas   = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, num)]
    largest = max(areas)[1]
    out     = np.zeros_like(binary)
    out[labels == largest] = 255
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(out, kernel, iterations=1)


def _make_masked_and_overlay(
    img_bgr: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build masked_crop (part on white bg) and overlay (green contour on original).
    Both returned as BGR uint8.
    """
    masked = img_bgr.copy()
    masked[mask == 0] = 255   # white background

    overlay = img_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 80), 1)
    return masked, overlay


def _mask_quality_metrics(mask: np.ndarray) -> Dict[str, Any]:
    """
    Compute quality metrics for a binary mask (uint8, 255 = foreground).

    Returns
    -------
    coverage_pct            : foreground pixels as % of total pixels
    mask_components         : number of connected foreground components
    largest_component_pct   : largest component area as % of ALL foreground pixels
                              (100 = single clean object; low = fragmented)
    mask_bbox               : [x, y, w, h] tight bounding box of all foreground
    """
    total_px  = int(mask.size)
    fg_px     = int((mask > 0).sum())
    coverage  = round(fg_px / max(total_px, 1) * 100, 2)

    if fg_px == 0:
        return {
            "coverage_pct":           0.0,
            "mask_components":        0,
            "largest_component_pct":  0.0,
            "mask_bbox":              [],
        }

    num, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    num_fg = num - 1   # exclude background label 0
    if num_fg == 0:
        return {
            "coverage_pct":           coverage,
            "mask_components":        0,
            "largest_component_pct":  0.0,
            "mask_bbox":              [],
        }

    largest_area = max(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num))
    largest_pct  = round(largest_area / fg_px * 100, 2)

    pts   = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(pts) if pts is not None else (0, 0, 0, 0)

    return {
        "coverage_pct":           coverage,
        "mask_components":        num_fg,
        "largest_component_pct":  largest_pct,
        "mask_bbox":              [int(x), int(y), int(w), int(h)],
    }


def _hybrid_segment_crop(
    img_bgr: np.ndarray,
    save_sam2_raw_prefix: Optional[str] = None,
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Hybrid segmentation pipeline for a single extracted crop.

    Step 1 — attempt SAM2 (5-point cross prompt on 512px upscale).
    Step 2 — evaluate quality with three guards:
              • coverage_pct  < 5   → missed part (SAM2 confused by tiny crop)
              • coverage_pct  > 85  → background flood
              • largest_component_pct < 50 → fragmented / wrong object picked
    Step 3 — if any guard fires, fall back to colour-threshold segmentation.

    Parameters
    ----------
    save_sam2_raw_prefix : optional path prefix (str).
        When given, the raw SAM2 mask and overlay are written to:
          <prefix>_sam2_raw_mask.png
          <prefix>_sam2_raw_overlay.png
        This happens BEFORE the quality guards are evaluated — used for
        diagnostic capture on the first N crops.

    Returns
    -------
    mask    : uint8 (H, W)  255 = foreground
    method  : 'sam2' | 'fallback_colour' | 'failed'
    info    : dict with coverage_pct, largest_component_pct, mask_components,
              mask_bbox, iou, all_scores, reject_reason,
              sam2_loaded, sam2_exception, sam2_mask_count,
              sam2_raw_coverage_pct, sam2_raw_largest_component_pct,
              sam2_reject_reason
    """
    info: Dict[str, Any] = {
        "iou":                              None,
        "all_scores":                       [],
        "reject_reason":                    None,
        "coverage_pct":                     0.0,
        "largest_component_pct":            0.0,
        "mask_components":                  0,
        "mask_bbox":                        [],
        # ── SAM2 diagnostics (pre-guard) ──────────────────────────────────
        "sam2_loaded":                      _sam2_model is not None,
        "sam2_exception":                   None,
        "sam2_mask_count":                  0,
        "sam2_raw_coverage_pct":            None,
        "sam2_raw_largest_component_pct":   None,
        "sam2_reject_reason":               None,
    }

    # ── Attempt SAM2 ──────────────────────────────────────────────────────────
    sam2_mask: Optional[np.ndarray] = None
    if _sam2_model is not None:
        try:
            sam2_mask, iou, all_scores = _sam2_segment_crop(img_bgr)
            info["iou"]            = round(iou, 4)
            info["all_scores"]     = all_scores
            info["sam2_mask_count"] = 1
        except Exception as exc:
            info["sam2_exception"] = str(exc)
            info["reject_reason"]  = f"sam2_error: {exc}"

    if sam2_mask is not None:
        q = _mask_quality_metrics(sam2_mask)

        # ── Record raw SAM2 quality BEFORE guards ─────────────────────────
        info["sam2_raw_coverage_pct"]          = q["coverage_pct"]
        info["sam2_raw_largest_component_pct"] = q["largest_component_pct"]

        # ── Optionally save raw SAM2 mask + overlay for diagnostics ───────
        if save_sam2_raw_prefix:
            try:
                cv2.imwrite(f"{save_sam2_raw_prefix}_sam2_raw_mask.png", sam2_mask)
                raw_ov = img_bgr.copy()
                cnts, _ = cv2.findContours(
                    sam2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(raw_ov, cnts, -1, (0, 100, 255), 1)
                cv2.imwrite(f"{save_sam2_raw_prefix}_sam2_raw_overlay.png", raw_ov)
            except Exception:
                pass  # diagnostic save failure must never break the pipeline

        # ── Quality guards ────────────────────────────────────────────────
        reject_reason: Optional[str] = None
        if q["coverage_pct"] < 5.0:
            reject_reason = f"coverage_too_low ({q['coverage_pct']:.1f}%)"
        elif q["coverage_pct"] > 85.0:
            reject_reason = f"coverage_too_high ({q['coverage_pct']:.1f}%)"
        elif q["largest_component_pct"] < 50.0:
            reject_reason = (
                f"fragmented (largest_cc={q['largest_component_pct']:.1f}%, "
                f"components={q['mask_components']})"
            )

        if reject_reason is None:
            # SAM2 accepted
            info.update(q)
            return sam2_mask, "sam2", info

        # SAM2 rejected — record why and fall through to colour fallback
        info["sam2_reject_reason"] = reject_reason
        info["reject_reason"]      = reject_reason

    # ── Colour-threshold fallback ──────────────────────────────────────────────
    try:
        fb_mask = _cv_fallback_mask(img_bgr)
        q2      = _mask_quality_metrics(fb_mask)
        info.update(q2)
        if q2["coverage_pct"] == 0:
            return fb_mask, "failed", info
        return fb_mask, "fallback_colour", info
    except Exception as exc:
        # Return a blank mask if everything failed
        blank = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        info["reject_reason"] = (info.get("reject_reason") or "") + f"; fallback_error: {exc}"
        return blank, "failed", info

