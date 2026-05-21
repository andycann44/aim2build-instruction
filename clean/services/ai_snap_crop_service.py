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
_SAM_REFINED_DIR = Path.home() / "aim2build-data" / "instruction-training" / "sam_refined"
_MIN_ALPHA_PIXELS = 30
_MIN_ALPHA_RATIO = 0.04   # 4 % of the slot ROI bounding box


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


def _recover_grey_foreground(
    mask: np.ndarray,
    img: np.ndarray,
    bg: np.ndarray,
) -> np.ndarray:
    """Conservative expansion pass for low-saturation grey/light parts.

    Grey parts on a coloured (e.g. light-blue) background are under-detected by the
    colour-diff threshold because their L2 distance from the background is smaller
    than dark parts.  When the 72nd-percentile threshold is pulled up by high-contrast
    dark parts, grey pixels can fall below it and leave large internal holes.

    Strategy
    --------
    1. Grey-candidate pixels: HSV-saturation < 80 (rules out the coloured background)
       AND at least a small colour diff from background (rules out pure bg).
    2. Restrict candidates to the dilation zone of already-found blobs — never expands
       into regions far from existing foreground.
    3. Morphological close fills internal holes; open removes isolated specks.

    The saturation guard is the primary safety: the light-blue background has
    S ≈ 80-150 in OpenCV HSV, so ``S < 80`` excludes it reliably while admitting
    neutral greys (S ≈ 0-40).
    """
    if mask is None or int(np.count_nonzero(mask)) == 0:
        return mask

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)  # 0-255
    diff = np.linalg.norm(img.astype(np.float32) - bg.reshape(1, 1, 3), axis=2)

    # Grey-candidate: low saturation (not background-coloured) and at least slightly
    # different from background in colour distance.  Threshold 10 is deliberately low
    # — the saturation guard carries the safety burden, not the diff floor.
    grey_cand = ((s < 80) & (diff > 10)).astype(np.uint8) * 255

    # Dilation zone: only accept grey candidates within this neighbourhood.
    zone_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    zone = cv2.dilate(mask, zone_k, iterations=1)
    in_zone = cv2.bitwise_and(grey_cand, zone)

    expanded = cv2.bitwise_or(mask, in_zone)

    # Close internal holes, then drop isolated specks smaller than the open kernel.
    expanded = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    expanded = cv2.morphologyEx(expanded, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Re-apply border zeroing.
    h, w = mask.shape[:2]
    border = max(1, min(4, min(h, w) // 24))
    expanded[:border, :] = 0
    expanded[h - border:, :] = 0
    expanded[:, :border] = 0
    expanded[:, w - border:] = 0

    return expanded


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
    mask = _recover_grey_foreground(mask, img, bg)
    return mask, ""


def create_full_crop_mask_debug(
    callout_crop_path: str,
    *,
    set_num: Optional[str] = None,
    bag: Optional[int] = None,
    crop_id: Optional[str] = None,
    desktop_overlays: bool = True,
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
    if ok_overlay and desktop_overlays:
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


def _rgb_stats_from_bgra(bgra: np.ndarray) -> Dict[str, Any]:
    """Return median and mean RGB (not BGR) of opaque pixels in a BGRA image."""
    alpha = bgra[:, :, 3]
    mask = alpha > 0
    if not mask.any():
        return {"slot_rgb_median": None, "slot_rgb_avg": None}
    b, g, r = bgra[:, :, 0][mask], bgra[:, :, 1][mask], bgra[:, :, 2][mask]
    med = [int(np.median(r)), int(np.median(g)), int(np.median(b))]
    avg = [int(np.mean(r)),   int(np.mean(g)),   int(np.mean(b))]
    return {"slot_rgb_median": med, "slot_rgb_avg": avg}


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


def _enhance_slot_crop(
    slot_img: np.ndarray,
    slot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Per-slot CLAHE + unsharp + grey-fill recovery pass.

    Parameters
    ----------
    slot_img  : BGR image cropped to the component bounding box.
    slot_mask : Binary (0/255) mask of the same spatial extent.

    Returns
    -------
    enhanced_img         : BGR with CLAHE and mild unsharp applied.
    refined_mask         : Binary mask after conservative grey-fill recovery.
    overexpanded         : True when refined area > 135 % of original area
                           (caller should mark the slot needs_review).

    Design notes
    ------------
    - CLAHE and unsharp are applied only to the saved pixel data; they do not
      affect ownership or the global mask used for slot assignment.
    - Grey-fill recovery is restricted to a small dilation zone around existing
      foreground and gated by HSV saturation < 80, so the coloured background
      (S ≈ 80-150 for light-blue) cannot leak in.
    - If the crop is too small or any step raises an exception the function
      returns the inputs unchanged.
    """
    h, w = slot_img.shape[:2]
    if h < 8 or w < 8 or slot_img.ndim != 3:
        return slot_img, slot_mask, False

    original_area = int(np.count_nonzero(slot_mask))

    # ── 1. CLAHE on L* channel ────────────────────────────────────────────
    try:
        lab = cv2.cvtColor(slot_img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        tile_w = max(2, min(8, w // 8))
        tile_h = max(2, min(8, h // 8))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_w, tile_h))
        l_ch = clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    except Exception:
        enhanced = slot_img.copy()

    # ── 2. Mild unsharp mask ──────────────────────────────────────────────
    try:
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0.8)
        enhanced = cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)
    except Exception:
        pass

    # ── 3. Grey-fill recovery near existing foreground ───────────────────
    refined_mask = slot_mask.copy()
    if original_area > 0 and h >= 12 and w >= 12:
        try:
            hsv = cv2.cvtColor(slot_img, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1].astype(np.float32)

            local_bg = _edge_background_bgr(slot_img)
            if local_bg is not None:
                diff = np.linalg.norm(
                    slot_img.astype(np.float32) - local_bg.reshape(1, 1, 3), axis=2
                )
                grey_cand = ((s < 80) & (diff > 8)).astype(np.uint8) * 255
            else:
                # No reliable local background — use saturation alone with a
                # stricter threshold so we don't over-expand on ambiguous crops.
                grey_cand = (s < 50).astype(np.uint8) * 255

            # Limit expansion to a 9×9 ellipse neighbourhood of existing blobs.
            zone_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            zone = cv2.dilate(refined_mask, zone_k, iterations=1)
            in_zone = cv2.bitwise_and(grey_cand, zone)

            refined_mask = cv2.bitwise_or(refined_mask, in_zone)
            refined_mask = cv2.morphologyEx(
                refined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
            )
            refined_mask = cv2.morphologyEx(
                refined_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
            )
        except Exception:
            refined_mask = slot_mask.copy()

    # ── 4. Area-growth guard ──────────────────────────────────────────────
    refined_area = int(np.count_nonzero(refined_mask))
    overexpanded = original_area > 0 and refined_area > int(original_area * 1.35)

    return enhanced, refined_mask, overexpanded


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

    # Per-slot enhancement: CLAHE, unsharp mask, grey-fill recovery.
    # enhanced_img replaces cropped_img for the saved RGBA; refined_mask replaces
    # cropped_mask.  Neither feeds back into ownership or global mask assignment.
    _original_area = int(np.count_nonzero(cropped_mask))
    _enhanced_img, _refined_mask, _enhancement_overexpanded = _enhance_slot_crop(
        cropped_img, cropped_mask
    )
    _refined_area = int(np.count_nonzero(_refined_mask))
    _enhancement_expanded_pct = (
        float((_refined_area - _original_area) / _original_area)
        if _original_area > 0 else 0.0
    )

    ok_mask = cv2.imwrite(str(mask_path), _refined_mask)
    bgra = cv2.cvtColor(_enhanced_img, cv2.COLOR_BGR2BGRA)
    alpha = _refined_mask
    if min(alpha.shape[:2]) >= 12:
        feathered = cv2.GaussianBlur(alpha, (3, 3), 0)
        edge = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8), iterations=1)
        alpha = np.where(edge > 0, feathered, alpha).astype(np.uint8)
    bgra[:, :, 3] = alpha
    alpha_debug = _alpha_stats(bgra[:, :, 3])
    rgb_stats = _rgb_stats_from_bgra(bgra)
    _alpha_count = int(alpha_debug.get("alpha_pixel_count", 0) or 0)
    _ah, _aw = bgra.shape[:2]
    if _alpha_count < _MIN_ALPHA_PIXELS or _alpha_count < _ah * _aw * _MIN_ALPHA_RATIO:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "needs_review_low_alpha",
            "mask_component_reason": component_reason,
            "alpha_pixel_count": _alpha_count,
            "enhancement_overexpanded": bool(_enhancement_overexpanded),
            "enhancement_expanded_pct": _enhancement_expanded_pct,
            **rgb_stats,
        }
    ok_cutout = cv2.imwrite(str(cutout_path), bgra)
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
            "enhancement_overexpanded": bool(_enhancement_overexpanded),
            "enhancement_expanded_pct": _enhancement_expanded_pct,
            **alpha_debug,
            **rgb_stats,
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
        "enhancement_overexpanded": bool(_enhancement_overexpanded),
        "enhancement_expanded_pct": _enhancement_expanded_pct,
        **alpha_debug,
        **rgb_stats,
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
    desktop_overlays: bool = True,
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
        desktop_overlays=desktop_overlays,
    )

    h, w = img.shape[:2]
    source_mask_path = str(full_mask_debug.get("full_crop_mask_path") or "")
    source_mask_basename = Path(source_mask_path).name if source_mask_path else ""

    # Primary: load the binary mask written by create_full_crop_mask_debug directly.
    # Threshold > 0 to recover the binary foreground mask.
    # The overlay image is for human viewing only and is never used for extraction.
    # Fallback: recompute via _foreground_mask_for_image if the mask file is missing/unreadable.
    master_mask: Optional[np.ndarray] = None
    master_mask_error = ""
    _mask_file_path = str(full_mask_debug.get("full_crop_mask_path") or "")
    if _mask_file_path and Path(_mask_file_path).exists():
        _gray = cv2.imread(_mask_file_path, cv2.IMREAD_GRAYSCALE)
        if _gray is not None and _gray.size > 0:
            master_mask = (_gray > 0).astype(np.uint8) * 255
    if master_mask is None:
        master_mask, master_mask_error = _foreground_mask_for_image(img)

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

    # Precompute column x-bounds per token (midpoint between adjacent centres,
    # or image edge).  Stored by object id to avoid mutating the token dicts.
    _col_bounds: Dict[int, tuple[int, int]] = {}
    for row in rows:
        for col_index, token in enumerate(row):
            _prev = row[col_index - 1] if col_index > 0 else None
            _next = row[col_index + 1] if col_index + 1 < len(row) else None
            _tcx = int(token.get("cx", 0) or 0)
            _xl = 0 if _prev is None else int((int(_prev.get("cx", 0) or 0) + _tcx) / 2)
            _xr = w if _next is None else int((_tcx + int(_next.get("cx", 0) or 0)) / 2)
            _col_bounds[id(token)] = (max(0, _xl - 10), min(w, _xr + 10))

    # Stage 2 — global component extraction.
    # Run connectedComponentsWithStats on the full clean_mask once, then assign
    # each component to the best-matching qty token using column x-bounds and the
    # constraint that the component centroid must be above the qty label.
    # This eliminates the row_top_bound hard clip that sliced vertically stacked parts.
    slot_index = 0
    global_comps: list[Dict[str, Any]] = []
    _gl: Optional[np.ndarray] = None  # labels array, shared across all token iterations
    if clean_mask is not None:
        _min_comp_area = max(16.0, float(h * w) * 0.0008)
        _glc, _gl, _gls, _glcentroids = cv2.connectedComponentsWithStats(
            (clean_mask > 0).astype(np.uint8), 8
        )
        for _lab in range(1, _glc):
            _garea = int(_gls[_lab, cv2.CC_STAT_AREA])
            _glx = int(_gls[_lab, cv2.CC_STAT_LEFT])
            _gly = int(_gls[_lab, cv2.CC_STAT_TOP])
            _glw = int(_gls[_lab, cv2.CC_STAT_WIDTH])
            _glh = int(_gls[_lab, cv2.CC_STAT_HEIGHT])
            if _glw <= 2 or _glh <= 2 or _garea < _min_comp_area:
                continue
            global_comps.append({
                "lab": _lab,
                "area": _garea,
                "lx": _glx, "ly": _gly, "lw": _glw, "lh": _glh,
                "gcx": float(_glcentroids[_lab][0]),
                "gcy": float(_glcentroids[_lab][1]),
                "used": False,
            })

    for token in ordered_tokens:
        token_y = int(token.get("y", 0) or 0)
        cx_tok = int(token.get("cx", 0) or 0)
        x_left, x_right = _col_bounds.get(id(token), (0, w))
        slot_box = [int(x_left), 0, int(max(0, x_right - x_left)), int(token_y)]

        if clean_mask is None or _gl is None or x_right <= x_left:
            slots.append(_make_needs_review_slot(
                slot_index, source_path, source_mask_path, source_mask_basename,
                slot_box, token, generated_at, master_mask_error or "no_clean_mask",
            ))
            slot_index += 1
            continue

        # Score every unused global component against this token.
        # Hard constraints: centroid above qty label; centroid within column band.
        # Rank: largest area first; tiebreak by horizontal closeness to token centre.
        candidates = []
        for comp in global_comps:
            if comp["used"]:
                continue
            if comp["gcy"] >= token_y:
                continue
            if not (x_left <= comp["gcx"] < x_right):
                continue
            hdist = abs(comp["gcx"] - float(cx_tok))
            candidates.append((comp, hdist))

        if not candidates:
            slots.append(_make_needs_review_slot(
                slot_index, source_path, source_mask_path, source_mask_basename,
                slot_box, token, generated_at, "needs_review_no_component",
            ))
            print(
                "[auto-mask-slot] "
                f"crop_id={crop_id} slot_index={slot_index} "
                f"function_path=needs_review_no_component "
                f"source_crop={source_path} slot_window={slot_box} "
                f"component_box=None generated_at={generated_at}"
            )
            slot_index += 1
            continue

        best_comp = max(candidates, key=lambda item: (item[0]["area"], -item[1]))[0]
        best_comp["used"] = True
        extraction_reason = "global_mask_component"

        ex = best_comp["lx"]
        ey_val = best_comp["ly"]
        ew = best_comp["lw"]
        eh = best_comp["lh"]
        extraction_box = [ex, ey_val, ew, eh]
        slot_img = img[ey_val: ey_val + eh, ex: ex + ew]
        component_mask = (_gl[ey_val: ey_val + eh, ex: ex + ew] == best_comp["lab"]).astype(np.uint8) * 255

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

        _artifact_error = str(artifact_result.get("error") or "")
        _enhancement_overexpanded = bool(artifact_result.get("enhancement_overexpanded"))
        _slot_status = (
            "needs_review_low_alpha" if _artifact_error == "needs_review_low_alpha"
            else "needs_review" if _enhancement_overexpanded
            else "masked" if bool(artifact_result.get("ok"))
            else "needs_review"
        )
        _alpha_px = int(artifact_result.get("alpha_pixel_count") or 0)
        _expanded_pct = float(artifact_result.get("enhancement_expanded_pct") or 0.0)
        _slot_confidence = (
            "low" if _slot_status != "masked" or _alpha_px < 150
            else "medium" if _expanded_pct > 0.10
            else "high"
        )
        slots.append(
            {
                "slot_index": int(slot_index),
                "status": _slot_status,
                "shape_mask_path": str(artifact_result.get("shape_mask_path") or ""),
                "part_cutout_path": str(artifact_result.get("part_cutout_path") or ""),
                "reason": _artifact_error,
                "function_path_used": extraction_reason,
                "source_crop_path": str(source_path),
                "source_crop_basename": _path_basename(source_path),
                "master_mask_path": source_mask_path,
                "master_mask_basename": source_mask_basename,
                "slot_crop_box": slot_box,
                "slot_window": slot_box,
                "component_box": extraction_box,
                "component_area": int(best_comp["area"]),
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
                "slot_rgb_median": artifact_result.get("slot_rgb_median"),
                "slot_rgb_avg": artifact_result.get("slot_rgb_avg"),
                "enhancement_overexpanded": _enhancement_overexpanded,
                "enhancement_expanded_pct": _expanded_pct,
                "slot_confidence": _slot_confidence,
            }
        )
        print(
            "[auto-mask-slot] "
            f"crop_id={crop_id} slot_index={slot_index} function_path={extraction_reason} "
            f"source_crop={source_path} slot_window={slot_box} "
            f"component_box={extraction_box} component_area={best_comp['area']} "
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


# ---------------------------------------------------------------------------
# Optional SAM refinement — upgrades alpha-mask edges when SAM is installed.
# The core extraction pipeline is not affected; this is a purely additive step.
# ---------------------------------------------------------------------------

def refine_slot_cutout_with_sam(
    part_cutout_path: str,
    *,
    set_num: Optional[str] = None,
    bag: Optional[int] = None,
    crop_id: Optional[str] = None,
    slot_index: int = 0,
) -> Dict[str, Any]:
    """Refine the alpha mask of an existing RGBA slot cutout using SAM/SAM2.

    Returns a dict with keys:
        ok               – bool
        sam_refined_path – str  (absolute path to refined PNG, or "")
        sam_refine_status – "ok" | "unavailable" | "failed[: reason]"

    The original cutout is never modified.  If SAM is not installed, or its
    model weights are absent, returns status="unavailable" immediately.
    """
    _FAIL = lambda status: {"ok": False, "sam_refined_path": "", "sam_refine_status": status}

    cutout_path = Path(str(part_cutout_path or "").strip())
    if not cutout_path.exists():
        return _FAIL("failed_no_cutout")

    # ------------------------------------------------------------------
    # 1. Try to import SAM (segment_anything preferred; fall back to sam2)
    # ------------------------------------------------------------------
    _sam_version: int = 0
    _sam_module: Any = None
    try:
        import segment_anything as _sam_module  # type: ignore[import]
        _sam_version = 1
    except ImportError:
        pass

    if _sam_version == 0:
        try:
            import sam2 as _sam_module  # type: ignore[import]  # noqa: F401
            _sam_version = 2
        except ImportError:
            pass

    if _sam_version == 0:
        return _FAIL("unavailable")

    # ------------------------------------------------------------------
    # 2. Locate a model checkpoint in the standard data directory
    # ------------------------------------------------------------------
    model_dir = Path.home() / "aim2build-data" / "instruction-training" / "models"
    checkpoint: Optional[Path] = None

    if _sam_version == 1:
        for name in (
            "sam_vit_b_01ec64.pth",
            "sam_vit_l_0b3195.pth",
            "sam_vit_h_4b8939.pth",
        ):
            cand = model_dir / name
            if cand.exists():
                checkpoint = cand
                break
    else:  # SAM2
        for name in (
            "sam2_hiera_base_plus.pt",
            "sam2_hiera_large.pt",
            "sam2_hiera_small.pt",
            "sam2_hiera_tiny.pt",
        ):
            cand = model_dir / name
            if cand.exists():
                checkpoint = cand
                break

    if checkpoint is None:
        return _FAIL("unavailable")

    # ------------------------------------------------------------------
    # 3. Load the existing RGBA cutout and derive the SAM box prompt
    # ------------------------------------------------------------------
    try:
        rgba = cv2.imread(str(cutout_path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim < 3 or rgba.shape[2] != 4:
            return _FAIL("failed_read_cutout")

        alpha_ch = rgba[:, :, 3]
        ys, xs = np.where(alpha_ch > 10)
        if ys.size == 0:
            return _FAIL("failed_empty_alpha")

        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        box_xyxy = np.array([x1, y1, x2, y2], dtype=float)

        # SAM expects an RGB image (H, W, 3) in uint8
        rgb_img = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2RGB)

        # ------------------------------------------------------------------
        # 4. Run SAM prediction
        # ------------------------------------------------------------------
        refined_mask: Optional[np.ndarray] = None

        if _sam_version == 1:
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore[import]

            if "vit_b" in checkpoint.name:
                model_type = "vit_b"
            elif "vit_l" in checkpoint.name:
                model_type = "vit_l"
            else:
                model_type = "vit_h"

            sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint))
            sam_model.eval()
            predictor = SamPredictor(sam_model)
            predictor.set_image(rgb_img)
            masks, _, _ = predictor.predict(
                box=box_xyxy,
                multimask_output=False,
            )
            refined_mask = np.asarray(masks[0], dtype=bool)

        else:  # SAM2
            import torch  # type: ignore[import]
            from sam2.build_sam import build_sam2  # type: ignore[import]
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import]

            _cfg_map = {
                "sam2_hiera_base_plus.pt": "sam2_hiera_b+.yaml",
                "sam2_hiera_large.pt": "sam2_hiera_l.yaml",
                "sam2_hiera_small.pt": "sam2_hiera_s.yaml",
                "sam2_hiera_tiny.pt": "sam2_hiera_t.yaml",
            }
            cfg = _cfg_map.get(checkpoint.name, "sam2_hiera_b+.yaml")
            sam2_model = build_sam2(cfg, str(checkpoint))
            predictor_s2 = SAM2ImagePredictor(sam2_model)
            with torch.inference_mode():
                predictor_s2.set_image(rgb_img)
                masks, _, _ = predictor_s2.predict(
                    box=box_xyxy,
                    multimask_output=False,
                )
            refined_mask = np.asarray(masks[0], dtype=bool)

        if refined_mask is None or refined_mask.shape[:2] != rgba.shape[:2]:
            return _FAIL("failed_mask_shape_mismatch")

        # ------------------------------------------------------------------
        # 5. Apply refined mask and save beside the original
        # ------------------------------------------------------------------
        refined_rgba = rgba.copy()
        refined_rgba[:, :, 3] = np.where(refined_mask, 255, 0).astype(np.uint8)

        _SAM_REFINED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _SAM_REFINED_DIR / f"{cutout_path.stem}_sam_refined.png"
        cv2.imwrite(str(out_path), refined_rgba)

        return {
            "ok": True,
            "sam_refined_path": str(out_path),
            "sam_refine_status": "ok",
        }

    except Exception as exc:  # noqa: BLE001
        return _FAIL(f"failed: {exc!s}")
