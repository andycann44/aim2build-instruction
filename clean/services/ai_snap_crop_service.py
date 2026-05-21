from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional

import cv2
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHAPE_MASK_DIR = _REPO_ROOT / "debug" / "ai_training" / "shape_masks"
_PART_CUTOUT_DIR = _REPO_ROOT / "debug" / "ai_training" / "part_cutouts"
_FULL_CROP_MASK_DIR = _REPO_ROOT / "debug" / "ai_training" / "full_crop_masks"
_FULL_CROP_MASK_OVERLAY_DIR = _REPO_ROOT / "debug" / "ai_training" / "full_crop_mask_overlays"
_SLOT_WINDOW_OVERLAY_DIR = _REPO_ROOT / "debug" / "ai_training" / "slot_window_overlays"
_DESKTOP_MASK_OVERLAY_DIR = Path.home() / "Desktop" / "aim2build-mask-overlays"


def _safe_slug(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    safe = "".join(ch for ch in text if ch.isalnum() or ch in "-_")
    return safe or fallback


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_basename(path_value: Any) -> str:
    text = str(path_value or "").strip()
    return Path(text).name if text else ""


def _page_step_from_crop_id(crop_id: Any) -> tuple[str, str]:
    text = str(crop_id or "")
    page = "pageunknown"
    step = "stepunknown"
    parts = text.split("_")
    for part in parts:
        if part.startswith("p") and part[1:].isdigit():
            page = f"page{part[1:]}"
        if part.startswith("s") and part[1:].isdigit():
            step = f"step{part[1:]}"
    return page, step


def _edge_background_bgr(img: Any) -> Optional[np.ndarray]:
    try:
        h, w = img.shape[:2]
        band = max(2, min(12, min(h, w) // 10))
        edge_pixels = np.concatenate(
            [
                img[:band, :, :].reshape(-1, 3),
                img[h - band :, :, :].reshape(-1, 3),
                img[:, :band, :].reshape(-1, 3),
                img[:, w - band :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        if edge_pixels.size == 0:
            return None
        return np.median(edge_pixels.astype(np.float32), axis=0)
    except Exception:
        return None


def _contour_circularity(contour: Any) -> float:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0.0 or perimeter <= 0.0:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def _rebuild_sharp_mask(mask: Any, min_area: float, max_components: int = 4) -> tuple[Any, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    cleaned = np.zeros((h, w), dtype=np.uint8)
    kept = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw <= 2 or ch <= 2:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = _contour_circularity(contour)
        if circularity >= 0.68 and min(cw, ch) <= 34:
            cv2.drawContours(cleaned, [contour], -1, 255, thickness=-1)
        else:
            epsilon = max(0.8, min(3.0, 0.006 * float(perimeter)))
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(cleaned, [simplified], -1, 255, thickness=-1)
        kept += 1
        if kept >= max_components:
            break
    return cleaned, kept


def _box_overlap_area(a: list[int], b: list[int]) -> int:
    ax, ay, aw, ah = [int(value) for value in a]
    bx, by, bw, bh = [int(value) for value in b]
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0
    return int((x2 - x1) * (y2 - y1))


def _select_component_near_slot_anchor(
    mask: Any,
    *,
    expected_part_center: Optional[list[int]] = None,
    reject_boxes: Optional[list[list[int]]] = None,
    plausible_box: Optional[list[int]] = None,
    min_area: float = 25.0,
) -> tuple[Any, int, str]:
    h, w = mask.shape[:2]
    if expected_part_center is None:
        return mask, 0, ""
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if labels_count <= 1:
        return mask, 0, "no_components"

    ax = float(expected_part_center[0])
    ay = float(expected_part_center[1])
    reject_boxes = reject_boxes or []
    px = py = pw = ph = None
    if plausible_box is not None:
        px, py, pw, ph = [int(value) for value in plausible_box]
    candidates: list[tuple[float, int]] = []
    max_reasonable_distance = max(22.0, float(max(w, h)) * 0.62)
    for label in range(1, labels_count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if cw <= 2 or ch <= 2:
            continue
        comp_box = [x, y, cw, ch]
        if px is not None and py is not None and pw is not None and ph is not None:
            overlap = _box_overlap_area(comp_box, [px, py, pw, ph])
            if overlap <= 0:
                continue
            if overlap / max(1.0, area) < 0.42:
                continue
        overlaps_text = False
        for reject_box in reject_boxes:
            overlap = _box_overlap_area(comp_box, reject_box)
            if overlap <= 0:
                continue
            if overlap / max(1.0, area) >= 0.08 or overlap >= 18:
                overlaps_text = True
                break
        if overlaps_text:
            continue
        cx = float(centroids[label][0])
        cy = float(centroids[label][1])
        distance = float(np.hypot(cx - ax, cy - ay))
        if distance > max_reasonable_distance:
            continue
        if cy > ay + max(8.0, float(h) * 0.08):
            continue
        horizontal_penalty = abs(cx - ax) * 0.85
        vertical_penalty = max(0.0, cy - ay) * 1.8 + max(0.0, ay - cy - float(h) * 0.58) * 0.7
        edge_penalty = 0.0
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            edge_penalty = min(area * 0.004, 32.0)
        oversized_penalty = 0.0
        if cw >= int(w * 0.86) or ch >= int(h * 0.90):
            oversized_penalty = 28.0
        score = (
            distance
            + horizontal_penalty
            + vertical_penalty
            + edge_penalty
            + oversized_penalty
            - min(area, float(w * h) * 0.16) * 0.0007
        )
        candidates.append((score, label))

    if not candidates:
        return np.zeros((h, w), dtype=np.uint8), 0, "no_plausible_component"
    _, best_label = min(candidates, key=lambda item: item[0])
    selected = np.zeros((h, w), dtype=np.uint8)
    selected[labels == best_label] = 255
    return selected, 1, "anchor_component"


def _token_rows(tokens: list[Dict[str, Any]]) -> list[list[Dict[str, Any]]]:
    rows: list[list[Dict[str, Any]]] = []
    for token in sorted(tokens, key=lambda item: (int(item.get("cy", 0) or 0), int(item.get("x", 0) or 0))):
        cy = int(token.get("cy", 0) or 0)
        placed = False
        for row in rows:
            row_cy = int(round(sum(int(item.get("cy", 0) or 0) for item in row) / max(1, len(row))))
            if abs(cy - row_cy) <= 22:
                row.append(token)
                placed = True
                break
        if not placed:
            rows.append([token])
    return [sorted(row, key=lambda item: int(item.get("x", 0) or 0)) for row in rows]


def _write_temp_slot_crop(img: Any, box: list[int]) -> Optional[Path]:
    x, y, w, h = [int(value) for value in box]
    crop = img[max(0, y) : max(0, y) + max(0, h), max(0, x) : max(0, x) + max(0, w)]
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    handle = tempfile.NamedTemporaryFile(prefix="ai_snap_slot_mask_", suffix=".png", delete=False)
    handle.close()
    path = Path(handle.name)
    ok = cv2.imwrite(str(path), crop)
    if not ok:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    return path


def _foreground_mask_for_image(img: Any) -> tuple[Optional[np.ndarray], str]:
    h, w = img.shape[:2]
    bg = _edge_background_bgr(img)
    if bg is None:
        return None, "background_estimate_failed"

    diff = np.linalg.norm(img.astype(np.float32) - bg.reshape(1, 1, 3), axis=2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 140)
    threshold = max(22.0, float(np.percentile(diff, 72)))
    mask = ((diff >= threshold) | (edges > 0)).astype(np.uint8) * 255

    border = max(1, min(4, min(h, w) // 24))
    mask[:border, :] = 0
    mask[h - border :, :] = 0
    mask[:, :border] = 0
    mask[:, w - border :] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return mask, ""


def create_full_crop_mask_debug(
    callout_crop_path: str,
    *,
    set_num: Optional[str] = None,
    bag: Optional[int] = None,
    crop_id: Optional[str] = None,
) -> Dict[str, Any]:
    source_path = Path(str(callout_crop_path or "").strip())
    if not source_path.exists():
        return {"ok": False, "full_crop_mask_path": "", "full_crop_mask_overlay_path": "", "error": "callout_crop_path_missing"}
    img = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if img is None or getattr(img, "size", 0) == 0:
        return {"ok": False, "full_crop_mask_path": "", "full_crop_mask_overlay_path": "", "error": "callout_crop_unreadable"}

    mask, error = _foreground_mask_for_image(img)
    if mask is None:
        return {"ok": False, "full_crop_mask_path": "", "full_crop_mask_overlay_path": "", "error": error or "full_crop_mask_failed"}

    _FULL_CROP_MASK_DIR.mkdir(parents=True, exist_ok=True)
    _FULL_CROP_MASK_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        stat = source_path.stat()
        digest_source = f"{source_path}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    except Exception:
        digest_source = str(source_path).encode("utf-8")
    digest = hashlib.sha1(digest_source).hexdigest()[:12]
    stem = "_".join(
        [
            _safe_slug(set_num, "set"),
            f"bag{int(bag or 0)}",
            _safe_slug(crop_id, "crop"),
            digest,
        ]
    )
    mask_path = _FULL_CROP_MASK_DIR / f"{stem}_full_mask.png"
    overlay_path = _FULL_CROP_MASK_OVERLAY_DIR / f"{stem}_full_mask_overlay.png"

    overlay = img.copy()
    tint = np.zeros_like(img)
    tint[:, :, 1] = 255
    keep = mask > 0
    overlay[keep] = cv2.addWeighted(img[keep], 0.58, tint[keep], 0.42, 0)

    ok_mask = cv2.imwrite(str(mask_path), mask)
    ok_overlay = cv2.imwrite(str(overlay_path), overlay)
    desktop_overlay_path = ""
    if ok_overlay:
        try:
            _DESKTOP_MASK_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
            page_part, step_part = _page_step_from_crop_id(crop_id)
            desktop_name = "_".join(
                [
                    _safe_slug(set_num, "set"),
                    page_part,
                    step_part,
                    _safe_slug(crop_id, "crop"),
                    digest,
                    "full_mask_overlay.png",
                ]
            )
            desktop_path = _DESKTOP_MASK_OVERLAY_DIR / desktop_name
            if cv2.imwrite(str(desktop_path), overlay):
                desktop_overlay_path = str(desktop_path)
        except Exception:
            desktop_overlay_path = ""
    if not ok_mask or not ok_overlay:
        return {
            "ok": False,
            "full_crop_mask_path": str(mask_path) if ok_mask else "",
            "full_crop_mask_overlay_path": str(overlay_path) if ok_overlay else "",
            "desktop_full_crop_mask_overlay_path": desktop_overlay_path,
            "error": "full_crop_mask_write_failed",
        }
    return {
        "ok": True,
        "full_crop_mask_path": str(mask_path),
        "full_crop_mask_overlay_path": str(overlay_path),
        "desktop_full_crop_mask_overlay_path": desktop_overlay_path,
        "error": "",
    }


def _alpha_stats(alpha: Any) -> Dict[str, Any]:
    if alpha is None or getattr(alpha, "size", 0) == 0:
        return {"alpha_pixel_count": 0, "non_transparent_bbox": None}
    keep = alpha > 0
    count = int(np.count_nonzero(keep))
    if count <= 0:
        return {"alpha_pixel_count": 0, "non_transparent_bbox": None}
    ys, xs = np.where(keep)
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return {
        "alpha_pixel_count": count,
        "non_transparent_bbox": [x1, y1, int(x2 - x1), int(y2 - y1)],
    }


def _write_slot_window_overlay(
    img: Any,
    slot_box: list[int],
    source_path: Path,
    *,
    set_num: Optional[str] = None,
    bag: Optional[int] = None,
    crop_id: Optional[str] = None,
    slot_index: Optional[int] = None,
) -> str:
    if img is None or getattr(img, "size", 0) == 0:
        return ""
    _SLOT_WINDOW_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        stat = source_path.stat()
        digest_source = f"{source_path}:{stat.st_mtime_ns}:{stat.st_size}:{slot_box}".encode("utf-8")
    except Exception:
        digest_source = f"{source_path}:{slot_box}".encode("utf-8")
    digest = hashlib.sha1(digest_source).hexdigest()[:12]
    stem = "_".join(
        [
            _safe_slug(set_num, "set"),
            f"bag{int(bag or 0)}",
            _safe_slug(crop_id, "crop"),
            f"slot{int(slot_index) if slot_index is not None else 0}",
            digest,
        ]
    )
    overlay_path = _SLOT_WINDOW_OVERLAY_DIR / f"{stem}_slot_window_overlay.png"
    overlay = img.copy()
    x, y, bw, bh = [int(value) for value in slot_box]
    x2 = max(0, min(overlay.shape[1] - 1, x + max(0, bw) - 1))
    y2 = max(0, min(overlay.shape[0] - 1, y + max(0, bh) - 1))
    x = max(0, min(overlay.shape[1] - 1, x))
    y = max(0, min(overlay.shape[0] - 1, y))
    tint = overlay.copy()
    cv2.rectangle(tint, (x, y), (x2, y2), (0, 255, 255), thickness=-1)
    overlay = cv2.addWeighted(overlay, 0.72, tint, 0.28, 0)
    cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 255), thickness=2)
    if not cv2.imwrite(str(overlay_path), overlay):
        return ""
    return str(overlay_path)


def _master_mask_components(mask: Any, qty_boxes: list[list[int]]) -> list[Dict[str, Any]]:
    if mask is None or getattr(mask, "size", 0) == 0:
        return []
    h, w = mask.shape[:2]
    source = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: list[Dict[str, Any]] = []
    min_area = max(18.0, float(h * w) * 0.00025)
    for contour_index, contour in enumerate(contours):
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw <= 2 or ch <= 2:
            continue
        local_contour = contour.copy()
        local_contour[:, 0, 0] -= int(x)
        local_contour[:, 0, 1] -= int(y)
        local_mask = np.zeros((int(ch), int(cw)), dtype=np.uint8)
        cv2.drawContours(local_mask, [local_contour], -1, 255, thickness=-1)
        local_source = source[y : y + ch, x : x + cw]
        local_mask = cv2.bitwise_and(local_mask, local_source)
        area = int(np.count_nonzero(local_mask))
        if area < min_area:
            continue
        box = [x, y, cw, ch]
        overlaps_qty = False
        for qty_box in qty_boxes:
            overlap = _box_overlap_area(box, qty_box)
            if overlap <= 0:
                continue
            if overlap >= 10 or overlap / max(1.0, float(area)) >= 0.05:
                overlaps_qty = True
                break
        if overlaps_qty:
            continue
        components.append(
            {
                "label": int(contour_index),
                "box": box,
                "area": int(area),
                "cx": float(x + cw / 2.0),
                "cy": float(y + ch / 2.0),
                "mask": local_mask,
            }
        )
    return components


def _mask_region_above_qty(mask: Any, token: Dict[str, Any]) -> tuple[Optional[list[int]], Optional[Any]]:
    if mask is None or getattr(mask, "size", 0) == 0:
        return None, None
    h, w = mask.shape[:2]
    qty_x = int(token.get("x", 0) or 0)
    qty_y = int(token.get("y", 0) or 0)
    qty_w = int(token.get("w", 0) or 0)
    qty_h = int(token.get("h", 0) or 0)
    qty_cx = int(token.get("cx", qty_x + qty_w // 2) or (qty_x + qty_w // 2))
    search_half_width = max(28, int(qty_w * 4.5), int(w * 0.08))
    x1 = max(0, qty_cx - search_half_width)
    x2 = min(w, qty_cx + search_half_width)
    y1 = max(0, qty_y - max(36, int(h * 0.45)))
    y2 = min(h, max(y1 + 1, qty_y - 1))
    if x2 <= x1 or y2 <= y1:
        return None, None
    region = mask[y1:y2, x1:x2].copy()
    # Remove the qty text area if the search window clips into it due to OCR geometry.
    local_qty_x = max(0, qty_x - x1)
    local_qty_y = max(0, qty_y - y1)
    local_qty_x2 = min(region.shape[1], local_qty_x + max(0, qty_w))
    local_qty_y2 = min(region.shape[0], local_qty_y + max(0, qty_h))
    if local_qty_x2 > local_qty_x and local_qty_y2 > local_qty_y:
        region[local_qty_y:local_qty_y2, local_qty_x:local_qty_x2] = 0
    if int(np.count_nonzero(region)) <= 0:
        return None, None
    ys, xs = np.where(region > 0)
    bx1 = int(xs.min())
    by1 = int(ys.min())
    bx2 = int(xs.max()) + 1
    by2 = int(ys.max()) + 1
    box = [int(x1 + bx1), int(y1 + by1), int(bx2 - bx1), int(by2 - by1)]
    cropped_mask = region[by1:by2, bx1:bx2]
    if cropped_mask is None or cropped_mask.size == 0 or int(np.count_nonzero(cropped_mask)) <= 0:
        return None, None
    return box, cropped_mask


def create_shape_mask_for_slot_crop(
    ai_snap_input_path: str,
    *,
    set_num: Optional[str] = None,
    bag: Optional[int] = None,
    crop_id: Optional[str] = None,
    slot_index: Optional[int] = None,
    expected_part_center: Optional[list[int]] = None,
    reject_boxes: Optional[list[list[int]]] = None,
    plausible_box: Optional[list[int]] = None,
) -> Dict[str, Any]:
    """Create debug-only shape mask/cutout images for an AI Snap input crop."""
    source_path = Path(str(ai_snap_input_path or "").strip())
    if not source_path.exists():
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "ai_snap_input_path_missing",
        }

    img = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if img is None or getattr(img, "size", 0) == 0:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "ai_snap_input_image_unreadable",
        }

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "ai_snap_input_image_empty",
        }

    mask, error = _foreground_mask_for_image(img)
    if mask is None:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": error or "foreground_mask_failed",
        }

    min_area = max(25.0, float(h * w) * 0.004)
    if expected_part_center is not None:
        mask, anchor_kept, anchor_reason = _select_component_near_slot_anchor(
            mask,
            expected_part_center=expected_part_center,
            reject_boxes=reject_boxes,
            plausible_box=plausible_box,
            min_area=min_area,
        )
    else:
        anchor_kept = 0
        anchor_reason = ""
    cleaned, kept = _rebuild_sharp_mask(mask, min_area, max_components=1 if expected_part_center is not None else 4)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    if expected_part_center is not None:
        cleaned, anchor_kept, anchor_reason = _select_component_near_slot_anchor(
            cleaned,
            expected_part_center=expected_part_center,
            reject_boxes=reject_boxes,
            plausible_box=plausible_box,
            min_area=min_area,
        )
    cleaned, kept = _rebuild_sharp_mask(cleaned, min_area, max_components=1 if expected_part_center is not None else 4)

    if int(cleaned.sum()) <= 0:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "shape_mask_empty",
        }

    ys, xs = np.where(cleaned > 0)
    if xs.size == 0 or ys.size == 0:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "shape_mask_empty",
        }
    pad = max(5, min(10, min(h, w) // 16))
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + pad + 1)
    y2 = min(h, int(ys.max()) + pad + 1)
    cropped_mask = cleaned[y1:y2, x1:x2]
    cropped_img = img[y1:y2, x1:x2]
    if cropped_mask is None or cropped_mask.size == 0 or cropped_img is None or cropped_img.size == 0:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "shape_mask_crop_empty",
        }

    _SHAPE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    _PART_CUTOUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        stat = source_path.stat()
        digest_source = f"{source_path}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    except Exception:
        digest_source = str(source_path).encode("utf-8")
    digest = hashlib.sha1(digest_source).hexdigest()[:12]
    stem = "_".join(
        [
            _safe_slug(set_num, "set"),
            f"bag{int(bag or 0)}",
            _safe_slug(crop_id, "crop"),
            f"slot{int(slot_index) if slot_index is not None else 0}",
            digest,
        ]
    )

    mask_path = _SHAPE_MASK_DIR / f"{stem}_mask.png"
    cutout_path = _PART_CUTOUT_DIR / f"{stem}_cutout.png"
    mask_existed_before = mask_path.exists()
    cutout_existed_before = cutout_path.exists()

    ok_mask = cv2.imwrite(str(mask_path), cropped_mask)
    bgra = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
    alpha = cropped_mask
    if min(alpha.shape[:2]) >= 12:
        feathered = cv2.GaussianBlur(alpha, (3, 3), 0)
        edge = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8), iterations=1)
        alpha = np.where(edge > 0, feathered, alpha).astype(np.uint8)
    bgra[:, :, 3] = alpha
    ok_cutout = cv2.imwrite(str(cutout_path), bgra)
    alpha_debug = _alpha_stats(bgra[:, :, 3])
    if not ok_mask:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": str(cutout_path) if ok_cutout else "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "shape_mask_write_failed",
        }
    return {
        "ok": True,
        "shape_mask_path": str(mask_path),
        "part_cutout_path": str(cutout_path) if ok_cutout else "",
        "mask_slot_index": slot_index,
        "cutout_slot_index": slot_index,
        "error": "",
        "mask_component_count": int(anchor_kept or kept),
        "mask_component_reason": anchor_reason,
        "mask_crop_box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
    }


def _write_slot_artifacts_from_master_mask(
    slot_img: Any,
    slot_mask: Any,
    source_path: Path,
    *,
    set_num: Optional[str] = None,
    bag: Optional[int] = None,
    crop_id: Optional[str] = None,
    slot_index: Optional[int] = None,
    component_count: int = 0,
    component_reason: str = "",
) -> Dict[str, Any]:
    h, w = slot_img.shape[:2]
    alpha_debug: Dict[str, Any] = {}
    if slot_mask is None or getattr(slot_mask, "size", 0) == 0 or int(slot_mask.sum()) <= 0:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "shape_mask_empty",
            "mask_component_reason": component_reason,
        }

    cropped_mask = slot_mask
    cropped_img = slot_img
    if cropped_mask is None or cropped_mask.size == 0 or cropped_img is None or cropped_img.size == 0:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "shape_mask_crop_empty",
            "mask_component_reason": component_reason,
        }

    _SHAPE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    _PART_CUTOUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        stat = source_path.stat()
        digest_source = f"{source_path}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    except Exception:
        digest_source = str(source_path).encode("utf-8")
    digest = hashlib.sha1(digest_source).hexdigest()[:12]
    stem = "_".join(
        [
            _safe_slug(set_num, "set"),
            f"bag{int(bag or 0)}",
            _safe_slug(crop_id, "crop"),
            f"slot{int(slot_index) if slot_index is not None else 0}",
            digest,
        ]
    )
    mask_path = _SHAPE_MASK_DIR / f"{stem}_mask.png"
    cutout_path = _PART_CUTOUT_DIR / f"{stem}_cutout.png"
    mask_existed_before = mask_path.exists()
    cutout_existed_before = cutout_path.exists()

    ok_mask = cv2.imwrite(str(mask_path), cropped_mask)
    bgra = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
    alpha = cropped_mask
    if min(alpha.shape[:2]) >= 12:
        feathered = cv2.GaussianBlur(alpha, (3, 3), 0)
        edge = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8), iterations=1)
        alpha = np.where(edge > 0, feathered, alpha).astype(np.uint8)
    bgra[:, :, 3] = alpha
    ok_cutout = cv2.imwrite(str(cutout_path), bgra)
    alpha_debug = _alpha_stats(bgra[:, :, 3])
    if not ok_mask:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": str(cutout_path) if ok_cutout else "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "shape_mask_write_failed",
            "mask_component_reason": component_reason,
            "function_path_used": "master_mask_slice",
            "existing_file_overwritten": bool(mask_existed_before or cutout_existed_before),
            "existing_file_reused": False,
            "actual_saved_cutout_size": [int(bgra.shape[1]), int(bgra.shape[0])] if ok_cutout else None,
            **alpha_debug,
        }
    return {
        "ok": True,
        "shape_mask_path": str(mask_path),
        "part_cutout_path": str(cutout_path) if ok_cutout else "",
        "mask_slot_index": slot_index,
        "cutout_slot_index": slot_index,
        "error": "",
        "mask_component_count": int(component_count),
        "mask_component_reason": component_reason,
        "mask_crop_box": [0, 0, int(w), int(h)],
        "function_path_used": "master_mask_slice",
        "existing_file_overwritten": bool(mask_existed_before or cutout_existed_before),
        "existing_file_reused": False,
        "actual_saved_cutout_size": [int(bgra.shape[1]), int(bgra.shape[0])],
        **alpha_debug,
    }


def _make_needs_review_slot(
    slot_index: int,
    source_path: Path,
    source_mask_path: str,
    source_mask_basename: str,
    slot_box: list[int],
    token: Dict[str, Any],
    generated_at: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "slot_index": int(slot_index),
        "status": "needs_review",
        "shape_mask_path": "",
        "part_cutout_path": "",
        "reason": reason,
        "function_path_used": reason,
        "source_crop_path": str(source_path),
        "source_crop_basename": _path_basename(source_path),
        "master_mask_path": source_mask_path,
        "master_mask_basename": source_mask_basename,
        "slot_crop_box": slot_box,
        "slot_window": slot_box,
        "component_box": None,
        "component_area": 0,
        "component_assignment_score": None,
        "slot_window_overlay_path": "",
        "slot_window_overlay_basename": "",
        "qty_box": dict(token),
        "qty_token_box": dict(token),
        "mask_component_reason": reason,
        "cutout_basename": "",
        "generated_at": generated_at,
        "using_master_mask": True,
        "source_mask_path": source_mask_path,
        "source_mask_basename": source_mask_basename,
        "actual_saved_cutout_size": None,
        "non_transparent_bbox": None,
        "alpha_pixel_count": 0,
        "existing_file_overwritten": False,
        "existing_file_reused": False,
        "candidate_matching_started_before_cutout_save": False,
        "candidate_matching_started_after_cutout_save": None,
    }


def create_shape_masks_for_callout_slots(
    callout_crop_path: str,
    qty_token_boxes: list[Dict[str, Any]],
    *,
    set_num: Optional[str] = None,
    bag: Optional[int] = None,
    crop_id: Optional[str] = None,
) -> Dict[str, Any]:
    generated_at = _utc_timestamp()
    source_path = Path(str(callout_crop_path or "").strip())
    if not source_path.exists():
        return {"ok": False, "slots": [], "error": "callout_crop_path_missing"}
    img = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if img is None or getattr(img, "size", 0) == 0:
        return {"ok": False, "slots": [], "error": "callout_crop_unreadable"}

    full_mask_debug = create_full_crop_mask_debug(
        str(source_path),
        set_num=set_num,
        bag=bag,
        crop_id=crop_id,
    )

    h, w = img.shape[:2]
    master_mask, master_mask_error = _foreground_mask_for_image(img)
    source_mask_path = str(full_mask_debug.get("full_crop_mask_path") or "")
    source_mask_basename = Path(source_mask_path).name if source_mask_path else ""

    # Validate and normalise token coordinates.
    tokens: list[Dict[str, Any]] = []
    for token in qty_token_boxes or []:
        if not isinstance(token, dict):
            continue
        tx = int(token.get("x", 0) or 0)
        ty = int(token.get("y", 0) or 0)
        tw = int(token.get("w", 0) or 0)
        th = int(token.get("h", 0) or 0)
        if tw <= 0 or th <= 0:
            continue
        if tx < 0 or ty < 0 or tx + tw > w or ty + th > h:
            continue
        tokens.append(
            {
                **token,
                "x": tx,
                "y": ty,
                "w": tw,
                "h": th,
                "cx": int(token.get("cx", tx + tw // 2) or (tx + tw // 2)),
                "cy": int(token.get("cy", ty + th // 2) or (ty + th // 2)),
            }
        )

    rows = _token_rows(tokens)
    ordered_tokens = [token for row in rows for token in row]
    slots: list[Dict[str, Any]] = []

    # Stage 1 — build clean_mask: erase all qty token glyphs globally before any
    # component analysis so OCR text pixels never contaminate part extractions.
    clean_mask: Optional[np.ndarray] = None
    if master_mask is not None:
        clean_mask = master_mask.copy()
        pad = 2
        for tok in tokens:
            tx1 = max(0, int(tok["x"]) - pad)
            ty1 = max(0, int(tok["y"]) - pad)
            tx2 = min(w, int(tok["x"]) + int(tok["w"]) + pad)
            ty2 = min(h, int(tok["y"]) + int(tok["h"]) + pad)
            if tx2 > tx1 and ty2 > ty1:
                clean_mask[ty1:ty2, tx1:tx2] = 0

    row_bottoms = [
        max(int(item.get("y", 0) or 0) + int(item.get("h", 0) or 0) for item in row)
        for row in rows
    ]

    # Stage 2 — per-token local extraction.
    # Each slot is extracted independently from its own column/row ROI.
    # No global competition, no scoring across slots, no component merging.
    slot_index = 0
    for row_index, row in enumerate(rows):
        previous_qty_bottom = row_bottoms[row_index - 1] if row_index > 0 else None
        row_top_bound = 0 if previous_qty_bottom is None else max(0, int(previous_qty_bottom) + 4)

        for col_index, token in enumerate(row):
            prev_token = row[col_index - 1] if col_index > 0 else None
            next_token = row[col_index + 1] if col_index + 1 < len(row) else None
            cx = int(token.get("cx", 0) or 0)
            token_y = int(token.get("y", 0) or 0)

            # Column x-bounds: midpoints between adjacent qty centres (or image edge).
            x_left = 0 if prev_token is None else int((int(prev_token.get("cx", 0) or 0) + cx) / 2)
            x_right = w if next_token is None else int((cx + int(next_token.get("cx", 0) or 0)) / 2)
            x_left = max(0, x_left - 10)
            x_right = min(w, x_right + 10)

            # Row y-bottom: just above the qty token text.
            y_bottom = max(0, token_y - 1)
            slot_box = [
                int(x_left),
                int(row_top_bound),
                int(max(0, x_right - x_left)),
                int(max(0, y_bottom - row_top_bound)),
            ]

            if clean_mask is None or x_right <= x_left:
                slots.append(_make_needs_review_slot(
                    slot_index, source_path, source_mask_path, source_mask_basename,
                    slot_box, token, generated_at, master_mask_error or "no_clean_mask",
                ))
                slot_index += 1
                continue

            def _pick_local_component(
                y_top: int,
                _cm: np.ndarray = clean_mask,  # type: ignore[assignment]
                _xl: int = x_left,
                _xr: int = x_right,
                _ty: int = token_y,
                _cx: int = cx,
            ) -> Optional[tuple]:
                """Return (extraction_box, component_mask, area) from clean_mask ROI, or None."""
                y_bot = max(0, _ty - 1)
                if y_bot <= y_top or _xr <= _xl:
                    return None
                roi = _cm[y_top:y_bot, _xl:_xr]
                if roi.size == 0 or int(np.count_nonzero(roi)) == 0:
                    return None
                min_area = max(16.0, float(roi.shape[0] * roi.shape[1]) * 0.003)
                lc, ll, ls, lcentroids = cv2.connectedComponentsWithStats(
                    (roi > 0).astype(np.uint8), 8
                )
                if lc <= 1:
                    return None
                # Thin bottom band used to discard text-fragment slivers.
                bottom_band = max(4, int(roi.shape[0] * 0.06))
                candidates: list[Dict[str, Any]] = []
                for lab in range(1, lc):
                    area = int(ls[lab, cv2.CC_STAT_AREA])
                    if area < min_area:
                        continue
                    lx_ = int(ls[lab, cv2.CC_STAT_LEFT])
                    ly_ = int(ls[lab, cv2.CC_STAT_TOP])
                    lw_ = int(ls[lab, cv2.CC_STAT_WIDTH])
                    lh_ = int(ls[lab, cv2.CC_STAT_HEIGHT])
                    if lw_ <= 2 or lh_ <= 2:
                        continue
                    # Skip tiny slivers at the very bottom of the ROI.
                    if ly_ + lh_ >= roi.shape[0] - bottom_band and lh_ <= bottom_band:
                        continue
                    comp_cx = float(lcentroids[lab][0])
                    hdist = abs(comp_cx - float(_cx - _xl))
                    candidates.append({
                        "lab": int(lab),
                        "area": int(area),
                        "lx": lx_,
                        "ly": ly_,
                        "lw": lw_,
                        "lh": lh_,
                        "hdist": float(hdist),
                    })
                if not candidates:
                    return None
                # Largest component wins; tiebreak by horizontal proximity to qty centre.
                best = max(candidates, key=lambda c: (c["area"], -c["hdist"]))
                comp_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
                comp_mask_roi[ll == best["lab"]] = 255
                abs_x = _xl + best["lx"]
                abs_y = y_top + best["ly"]
                component_mask = comp_mask_roi[
                    best["ly"]: best["ly"] + best["lh"],
                    best["lx"]: best["lx"] + best["lw"],
                ].copy()
                extraction_box = [int(abs_x), int(abs_y), int(best["lw"]), int(best["lh"])]
                return extraction_box, component_mask, best["area"]

            # Primary: ROI from row_top_bound to just above qty label.
            result_tuple = _pick_local_component(row_top_bound)
            extraction_reason = "local_roi_largest_component"

            # Fallback: expand y_top to 0 to catch parts that bleed above the row boundary.
            if result_tuple is None and row_top_bound > 0:
                result_tuple = _pick_local_component(0)
                if result_tuple is not None:
                    extraction_reason = "local_roi_expanded_to_top"

            if result_tuple is None:
                slots.append(_make_needs_review_slot(
                    slot_index, source_path, source_mask_path, source_mask_basename,
                    slot_box, token, generated_at, "needs_review_no_local_component",
                ))
                print(
                    "[auto-mask-slot] "
                    f"crop_id={crop_id} slot_index={slot_index} "
                    f"function_path=needs_review_no_local_component "
                    f"source_crop={source_path} slot_window={slot_box} "
                    f"component_box=None generated_at={generated_at}"
                )
                slot_index += 1
                continue

            extraction_box, component_mask, component_area = result_tuple
            ex, ey, ew, eh = [int(v) for v in extraction_box]
            slot_img = img[ey: ey + eh, ex: ex + ew]

            artifact_result = _write_slot_artifacts_from_master_mask(
                slot_img,
                component_mask,
                source_path,
                set_num=set_num,
                bag=bag,
                crop_id=crop_id,
                slot_index=slot_index,
                component_count=1,
                component_reason=extraction_reason,
            )

            slots.append(
                {
                    "slot_index": int(slot_index),
                    "status": "masked" if bool(artifact_result.get("ok")) else "needs_review",
                    "shape_mask_path": str(artifact_result.get("shape_mask_path") or ""),
                    "part_cutout_path": str(artifact_result.get("part_cutout_path") or ""),
                    "reason": str(artifact_result.get("error") or ""),
                    "function_path_used": extraction_reason,
                    "source_crop_path": str(source_path),
                    "source_crop_basename": _path_basename(source_path),
                    "master_mask_path": source_mask_path,
                    "master_mask_basename": source_mask_basename,
                    "slot_crop_box": slot_box,
                    "slot_window": slot_box,
                    "component_box": extraction_box,
                    "component_area": int(component_area),
                    "component_assignment_score": None,
                    "slot_window_overlay_path": "",
                    "slot_window_overlay_basename": "",
                    "qty_box": dict(token),
                    "qty_token_box": dict(token),
                    "mask_component_reason": str(artifact_result.get("mask_component_reason") or ""),
                    "cutout_basename": Path(str(artifact_result.get("part_cutout_path") or "")).name,
                    "generated_at": generated_at,
                    "using_master_mask": True,
                    "source_mask_path": source_mask_path,
                    "source_mask_basename": source_mask_basename,
                    "actual_saved_cutout_size": artifact_result.get("actual_saved_cutout_size"),
                    "non_transparent_bbox": artifact_result.get("non_transparent_bbox"),
                    "alpha_pixel_count": int(artifact_result.get("alpha_pixel_count", 0) or 0),
                    "existing_file_overwritten": bool(artifact_result.get("existing_file_overwritten")),
                    "existing_file_reused": bool(artifact_result.get("existing_file_reused")),
                    "candidate_matching_started_before_cutout_save": False,
                    "candidate_matching_started_after_cutout_save": None,
                }
            )
            print(
                "[auto-mask-slot] "
                f"crop_id={crop_id} slot_index={slot_index} function_path={extraction_reason} "
                f"source_crop={source_path} slot_window={slot_box} "
                f"component_box={extraction_box} component_area={component_area} "
                f"cutout={artifact_result.get('part_cutout_path') or ''} "
                f"size={artifact_result.get('actual_saved_cutout_size')} "
                f"alpha_pixels={int(artifact_result.get('alpha_pixel_count', 0) or 0)} "
                f"alpha_bbox={artifact_result.get('non_transparent_bbox')} "
                f"generated_at={generated_at} "
                f"overwritten={bool(artifact_result.get('existing_file_overwritten'))} "
                f"reused={bool(artifact_result.get('existing_file_reused'))} "
                f"candidate_before_save=false overlay="
            )
            slot_index += 1

    return {
        "ok": True,
        "slots": slots,
        "slot_count": len(ordered_tokens),
        "full_crop_mask_path": str(full_mask_debug.get("full_crop_mask_path") or ""),
        "full_crop_mask_overlay_path": str(full_mask_debug.get("full_crop_mask_overlay_path") or ""),
        "desktop_full_crop_mask_overlay_path": str(full_mask_debug.get("desktop_full_crop_mask_overlay_path") or ""),
        "full_crop_mask_error": str(full_mask_debug.get("error") or ""),
        "generated_at": generated_at,
        "error": "",
    }
