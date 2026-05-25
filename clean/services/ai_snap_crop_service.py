from __future__ import annotations

import hashlib
import json
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
_BG_MASK_DEBUG_DIR = _REPO_ROOT / "debug" / "ai_training" / "bg_masks"
_DESKTOP_MASK_OVERLAY_DIR = Path.home() / "Desktop" / "aim2build-mask-overlays"
_SAM_REFINED_DIR = Path.home() / "aim2build-data" / "instruction-training" / "sam_refined"
_MIN_ALPHA_PIXELS = 30
_MIN_ALPHA_RATIO = 0.04   # 4 % of the slot ROI bounding box
_MIN_ALPHA_FOR_COLOUR = 127
_MAX_COLOUR_BRIGHTNESS = 210
_MIN_COLOUR_PIXEL_COUNT = 20

# Feature flag: when True, uses qty-anchor proximity to assign components to slots
# instead of the column-band + _force_window_fallback approach.
# Set to False to revert to the previous column-band path without changing any logic.
_USE_QTY_ANCHOR_PROXIMITY: bool = True


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


def _estimate_background_bgr(img: Any) -> Optional[np.ndarray]:
    """Estimate the dominant background colour by combining border samples with
    smooth (low-gradient) interior regions.

    More robust than a pure border median when the background fills large interior
    areas or has a colour that differs from the border stripe (e.g. a white interior
    with a thin coloured border, or vice-versa).

    Algorithm
    ---------
    1. Collect border-band pixels (top / bottom / left / right, same band width as
       ``_edge_background_bgr``).  Border pixels are included three times to give
       them higher weight.
    2. Find smooth interior pixels: Sobel gradient magnitude < 8 (very flat regions,
       not edges or text).  Filter out dark pixels (luma < 35) which are likely text,
       shadows, or dark parts rather than background.
    3. Round all samples to the nearest-8 colour bucket and pick the most-common
       bucket (mode via ``np.unique``).
    4. Return the median of all pixels within BGR distance 30 of that bucket centre.
    """
    try:
        h, w = img.shape[:2]
        band = max(2, min(12, min(h, w) // 10))

        # Border pixels (weighted 3×)
        border_px = np.concatenate(
            [
                img[:band, :].reshape(-1, 3),
                img[h - band:, :].reshape(-1, 3),
                img[:, :band].reshape(-1, 3),
                img[:, w - band:].reshape(-1, 3),
            ],
            axis=0,
        )

        # Smooth interior pixels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sx * sx + sy * sy)
        smooth_mask = (grad_mag < 8) & (gray.astype(np.float32) > 35)
        smooth_px = img[smooth_mask].reshape(-1, 3)

        combined = (
            np.concatenate([border_px, border_px, border_px, smooth_px], axis=0)
            if smooth_px.shape[0] > 0
            else np.concatenate([border_px, border_px, border_px], axis=0)
        )
        if combined.shape[0] == 0:
            return None

        # Dominant-colour bucket (quantise to multiples of 8)
        q = (combined.astype(np.int32) // 8) * 8
        keys = (
            (q[:, 0].astype(np.int64) << 16)
            | (q[:, 1].astype(np.int64) << 8)
            | q[:, 2].astype(np.int64)
        )
        unique_keys, counts = np.unique(keys, return_counts=True)
        best_key = int(unique_keys[int(np.argmax(counts))])
        center = np.array(
            [(best_key >> 16) & 0xFF, (best_key >> 8) & 0xFF, best_key & 0xFF],
            dtype=np.float32,
        )

        # Median of pixels close to the dominant bucket
        dists = np.linalg.norm(combined.astype(np.float32) - center.reshape(1, 3), axis=1)
        cluster = combined[dists < 30]
        if cluster.shape[0] == 0:
            cluster = combined
        return np.median(cluster.astype(np.float32), axis=0)
    except Exception:
        return None


def _lab_dist(img: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel L2 distance in CIELAB space between every pixel of *img* and the
    scalar background colour *bg_bgr* (shape (3,), float32 BGR).

    CIELAB is perceptually uniform, so a threshold of ~6–18 units maps directly to
    a "just noticeable difference" regardless of the background hue.
    """
    img_lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_u8 = np.clip(np.round(bg_bgr), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    bg_lab = cv2.cvtColor(bg_u8, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(3)
    return np.linalg.norm(img_lab - bg_lab.reshape(1, 1, 3), axis=2)


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

    Grey parts can be under-detected by the colour-diff threshold when their
    distance from the background is smaller than high-contrast dark parts (the
    72nd-percentile threshold gets pulled up by the dark parts, leaving grey pixels
    below it with internal holes).

    Strategy
    --------
    1. Grey-candidate pixels: LAB distance from the estimated background > 8
       (perceptibly different from the background regardless of its hue).
    2. Restrict candidates to the dilation zone of already-found blobs — never
       expands into regions far from existing foreground.
    3. Morphological close fills internal holes; open removes isolated specks.

    The LAB-distance gate replaces the old hard-coded HSV-saturation < 80 gate
    (which was calibrated for light-blue backgrounds).  A threshold of 8 LAB units
    corresponds roughly to a "just noticeable difference", keeping background pixels
    (distance ≈ 0–5) excluded while admitting grey/white parts that differ even
    slightly from whatever the detected background colour is.
    """
    if mask is None or int(np.count_nonzero(mask)) == 0:
        return mask

    lab_d = _lab_dist(img, bg)
    # Grey-candidate: perceptibly different from the estimated background in LAB space.
    grey_cand = (lab_d > 8).astype(np.uint8) * 255

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


def _recover_light_foreground(
    mask: np.ndarray,
    img: np.ndarray,
    bg: np.ndarray,
) -> np.ndarray:
    """Recovery pass for white, near-white, and transparent LEGO parts.

    Light-coloured and transparent parts have near-zero colour distance from a
    white or very light background, so the 72nd-percentile diff threshold in
    ``_foreground_mask_for_image`` may leave them entirely undetected (seed == 0)
    or heavily under-segmented.  This pass attempts to recover them without
    disturbing an existing mask if the part is dark.

    Pixel criterion
    ---------------
    S < 55  — low-saturation pixels only (not vividly coloured).
    V > 185 — restricts to genuinely bright pixels; dark/brown/red/grey parts all
              have V ≪ 185 so this guard completely protects Bag 1 dark-part paths.
    diff > 6 — excludes pixels matching background almost exactly (fast BGR gate).
    LAB distance < 6 → excluded — background-colour suppression that works for any
              background hue, replacing the old hard-coded blue/cyan HSV range.

    Extra safety
    ------------
    * Soft-edge reinforcement (Canny 20/80) is applied only in the zone-restricted
      path (seed ≥ 50 px) where the dilation zone prevents runaway expansion.
    * Area-growth guard: if the existing seed is already substantial (> 100 px)
      and the result would more than double it, the expansion is discarded.
    * Minimum-pixel guard: if the final mask has fewer than _MIN_ALPHA_PIXELS
      pixels, the original mask is returned unchanged to avoid reporting a
      spurious near-empty light blob.

    Zone-restricted only
    ---------------------
    Requires seed ≥ 50 px from the primary threshold before doing anything.
    Without a reliable anchor the candidate map cannot be safely bounded, so
    we return the unchanged mask and let downstream mark the slot needs_review.
    seed ≥ 50  — expand within a 39×39 dilation zone only, augmented with
                 soft Canny edges (20/80) to catch transparent-part outlines.
    """
    if mask is None or int(np.count_nonzero(mask)) < 50:
        return mask

    img_h, img_w = mask.shape[:2]
    seed_count = int(np.count_nonzero(mask))

    diff = np.linalg.norm(img.astype(np.float32) - bg.reshape(1, 1, 3), axis=2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1].astype(np.float32)   # 0–255
    v_ch = hsv[:, :, 2].astype(np.float32)   # 0–255

    # Light/white candidate pixels: low saturation, high brightness, non-background.
    light_cand = (s_ch < 55) & (v_ch > 185) & (diff > 6)

    # Suppress pixels too close to the detected background colour in LAB space.
    # This replaces the old hard-coded blue/cyan HSV range (H 95–135, S ≥ 40) and
    # works correctly for any background hue.
    too_close_to_bg = _lab_dist(img, bg) < 6
    light_cand = light_cand & ~too_close_to_bg

    light_cand_u8 = light_cand.astype(np.uint8) * 255

    border = max(1, min(4, min(img_h, img_w) // 24))

    # Zone-restricted path: expand from the existing seed with soft edges.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_soft = cv2.Canny(gray, 20, 80)
    cand_with_edges = cv2.bitwise_or(light_cand_u8, edges_soft)
    zone_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 39))
    zone = cv2.dilate(mask, zone_k, iterations=1)
    in_zone = cv2.bitwise_and(cand_with_edges, zone)
    expanded = cv2.bitwise_or(mask, in_zone)

    # Border zeroing.
    expanded[:border, :] = 0
    expanded[img_h - border:, :] = 0
    expanded[:, :border] = 0
    expanded[:, img_w - border:] = 0

    # Consolidate: close fills internal gaps; open removes isolated specks.
    expanded = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    expanded = cv2.morphologyEx(expanded, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)

    # Re-apply border zeroing after morphological ops.
    expanded[:border, :] = 0
    expanded[img_h - border:, :] = 0
    expanded[:, :border] = 0
    expanded[:, img_w - border:] = 0

    new_count = int(np.count_nonzero(expanded))

    # Area-growth guard: if seed was substantial and result explodes, discard expansion.
    if seed_count > 100 and new_count > seed_count * 2:
        return mask

    # Minimum-pixel guard: discard noise results below the low-alpha threshold.
    if new_count < _MIN_ALPHA_PIXELS:
        return mask

    return expanded


def _part_mask_for_callout_crop(img: Any) -> tuple[Optional[np.ndarray], str]:
    """Binary foreground mask for a full callout-crop image.

    Why not _foreground_mask_for_image?
    ------------------------------------
    _foreground_mask_for_image follows its initial threshold with two aggressive
    dilation-based recovery passes (_recover_grey_foreground uses a 19-px dilation
    zone; _recover_light_foreground uses a 39-px zone).  For a callout-crop image
    the rounded-rectangle frame border is detected as foreground, then the 19-px
    dilation bridges that frame to every interior part blob, collapsing all 10
    separate components into 1 giant blob that floods the whole interior.

    This function keeps identical initial detection (LAB colour distance + Canny)
    but stops morphological processing at a single (3×3)×1 close/open pair that
    is small enough to solidify individual part bodies without bridging neighbours.
    A lightweight component-level filter then removes:
      • the callout-frame outline  (large span, fill < 0.45, touches ≥ 3 borders)
      • solid background blobs     (area > 18 % of image, fill > 0.65)
      • text-strip artefacts       (aspect > 5, very short, very small)
    The result is separate white blobs per visible part.

    Verified on p22_s26_c1: 10 components before filter → 1 FRAME + 3 TEXT
    rejected → 6 clean part blobs retained.
    """
    h, w = img.shape[:2]
    image_area = float(h * w)

    bg = _estimate_background_bgr(img)
    if bg is None:
        bg = _edge_background_bgr(img)
    if bg is None:
        return None, "background_estimate_failed"

    diff  = _lab_dist(img, bg)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 140)
    threshold = max(22.0, float(np.percentile(diff, 72)))
    mask = ((diff >= threshold) | (edges > 0)).astype(np.uint8) * 255

    border = max(1, min(4, min(h, w) // 24))
    mask[:border, :] = 0
    mask[h - border:, :] = 0
    mask[:, :border] = 0
    mask[:, w - border:] = 0

    # Minimal morphological pass: fill sub-pixel gaps within part bodies; remove
    # isolated noise specks.  Keep kernel small so separate parts stay separate.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((2, 2), np.uint8), iterations=1)
    # No _recover_grey_foreground / _recover_light_foreground — those passes use
    # 19–39 px dilation zones that bridge the callout-frame to interior parts.

    # Component-level filtering: reject non-part shapes.
    bm       = max(3, min(h, w) // 16)
    min_area = max(16.0, image_area * 0.002)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = np.zeros((h, w), dtype=np.uint8)
    for _lab in range(1, n):
        _a  = int(stats[_lab, cv2.CC_STAT_AREA])
        _lx = int(stats[_lab, cv2.CC_STAT_LEFT])
        _ly = int(stats[_lab, cv2.CC_STAT_TOP])
        _lw = int(stats[_lab, cv2.CC_STAT_WIDTH])
        _lh = int(stats[_lab, cv2.CC_STAT_HEIGHT])
        if _a < min_area:
            continue
        _fill = float(_a) / float(max(1, _lw * _lh))
        # Reject frame outlines: spans most of image extent, thin fill.
        # 3-of-4 border touch handles callout boxes that don't reach all 4 edges.
        _touches = sum([
            _lx <= bm,
            _ly <= bm,
            _lx + _lw >= w - bm,
            _ly + _lh >= h - bm,
        ])
        if _touches >= 3 and _fill < 0.45:
            continue
        # Reject large solid background blobs.
        if _a > image_area * 0.18 and _fill > 0.65:
            continue
        # Reject text strips: wide aspect, very short, small.
        _asp = float(_lw) / float(max(1, _lh))
        if _asp > 5.0 and _lh < h * 0.10 and _a < image_area * 0.012:
            continue
        result[labels == _lab] = 255

    return result, ""


def _foreground_mask_for_image(img: Any) -> tuple[Optional[np.ndarray], str]:
    h, w = img.shape[:2]
    bg = _estimate_background_bgr(img)
    if bg is None:
        bg = _edge_background_bgr(img)   # fallback to simple border median
    if bg is None:
        return None, "background_estimate_failed"

    diff = _lab_dist(img, bg)
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
    mask = _recover_light_foreground(mask, img, bg)
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
    print(f"[full-mask-debug-enter] crop_id={crop_id} stem={stem}")
    if not source_path.exists():
        print("[full-mask-debug-skip] reason=callout_crop_path_missing")
        return {"ok": False, "full_crop_mask_path": "", "full_crop_mask_overlay_path": "", "error": "callout_crop_path_missing"}
    img = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if img is None or getattr(img, "size", 0) == 0:
        print("[full-mask-debug-skip] reason=callout_crop_unreadable")
        return {"ok": False, "full_crop_mask_path": "", "full_crop_mask_overlay_path": "", "error": "callout_crop_unreadable"}

    mask, error = _part_mask_for_callout_crop(img)
    if mask is None:
        print(f"[full-mask-debug-skip] reason={error or 'full_crop_mask_failed'}")
        return {"ok": False, "full_crop_mask_path": "", "full_crop_mask_overlay_path": "", "error": error or "full_crop_mask_failed"}

    _FULL_CROP_MASK_DIR.mkdir(parents=True, exist_ok=True)
    _FULL_CROP_MASK_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = _FULL_CROP_MASK_DIR / f"{stem}_full_mask.png"
    overlay_path = _FULL_CROP_MASK_OVERLAY_DIR / f"{stem}_full_mask_overlay.png"

    overlay = img.copy()
    tint = np.zeros_like(img)
    tint[:, :, 1] = 255
    keep = mask > 0
    overlay[keep] = cv2.addWeighted(img[keep], 0.58, tint[keep], 0.42, 0)

    ok_mask = cv2.imwrite(str(mask_path), mask)
    if ok_mask:
        print(f"[full-mask-debug-write] path={mask_path}")
    else:
        print(f"[full-mask-debug-skip] reason=full_crop_mask_write_failed path={mask_path}")
    ok_overlay = cv2.imwrite(str(overlay_path), overlay)
    if ok_overlay:
        print(f"[full-mask-debug-write] path={overlay_path}")
    else:
        print(f"[full-mask-debug-skip] reason=full_crop_overlay_write_failed path={overlay_path}")
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
                print(f"[full-mask-debug-write] path={desktop_path}")
            else:
                print(f"[full-mask-debug-skip] reason=desktop_overlay_write_failed path={desktop_path}")
        except Exception:
            print("[full-mask-debug-skip] reason=desktop_overlay_exception")
            desktop_overlay_path = ""
    # --- Background debug artefacts (bg_estimate.json, bg_mask.png, fg_overlay.png) ---
    bg_estimate_path = ""
    bg_mask_out_path = ""
    fg_overlay_path = ""
    try:
        _bg = _estimate_background_bgr(img)
        if _bg is None:
            _bg = _edge_background_bgr(img)
        if _bg is not None:
            _BG_MASK_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            _bg_u8 = np.clip(np.round(_bg), 0, 255).astype(np.uint8)
            _bg_b, _bg_g, _bg_r = int(_bg_u8[0]), int(_bg_u8[1]), int(_bg_u8[2])
            _bg_pixel = np.array([[[_bg_b, _bg_g, _bg_r]]], dtype=np.uint8)
            _bg_lab = cv2.cvtColor(_bg_pixel, cv2.COLOR_BGR2LAB)[0, 0]
            _bg_est_data = {
                "bgr": [_bg_b, _bg_g, _bg_r],
                "lab": [int(_bg_lab[0]), int(_bg_lab[1]), int(_bg_lab[2])],
                "hex": f"#{_bg_r:02x}{_bg_g:02x}{_bg_b:02x}",
            }
            _bg_est_file = _BG_MASK_DEBUG_DIR / f"{stem}_bg_estimate.json"
            _bg_est_file.write_text(json.dumps(_bg_est_data, indent=2), encoding="utf-8")
            bg_estimate_path = str(_bg_est_file)

            _bg_dist = _lab_dist(img, _bg)
            _bg_mask_img = (_bg_dist < 18).astype(np.uint8) * 255
            _bg_mask_file = _BG_MASK_DEBUG_DIR / f"{stem}_bg_mask.png"
            if cv2.imwrite(str(_bg_mask_file), _bg_mask_img):
                bg_mask_out_path = str(_bg_mask_file)

            _fg_ov = img.copy()
            _red_tint = np.zeros_like(img)
            _red_tint[:, :, 2] = 255
            _fg_px = mask > 0
            _fg_ov[_fg_px] = cv2.addWeighted(img[_fg_px], 0.58, _red_tint[_fg_px], 0.42, 0)
            _fg_ov_file = _BG_MASK_DEBUG_DIR / f"{stem}_fg_overlay.png"
            if cv2.imwrite(str(_fg_ov_file), _fg_ov):
                fg_overlay_path = str(_fg_ov_file)
    except Exception:
        pass

    if not ok_mask or not ok_overlay:
        return {
            "ok": False,
            "full_crop_mask_path": str(mask_path) if ok_mask else "",
            "full_crop_mask_overlay_path": str(overlay_path) if ok_overlay else "",
            "desktop_full_crop_mask_overlay_path": desktop_overlay_path,
            "bg_estimate_path": bg_estimate_path,
            "bg_mask_path": bg_mask_out_path,
            "fg_overlay_path": fg_overlay_path,
            "error": "full_crop_mask_write_failed",
        }
    return {
        "ok": True,
        "full_crop_mask_path": str(mask_path),
        "full_crop_mask_overlay_path": str(overlay_path),
        "desktop_full_crop_mask_overlay_path": desktop_overlay_path,
        "bg_estimate_path": bg_estimate_path,
        "bg_mask_path": bg_mask_out_path,
        "fg_overlay_path": fg_overlay_path,
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
    b_ch = bgra[:, :, 0].astype(np.float32)
    g_ch = bgra[:, :, 1].astype(np.float32)
    r_ch = bgra[:, :, 2].astype(np.float32)
    brightness = (b_ch + g_ch + r_ch) / 3.0
    mask = (alpha > _MIN_ALPHA_FOR_COLOUR) & (brightness < _MAX_COLOUR_BRIGHTNESS)
    if int(np.count_nonzero(mask)) < _MIN_COLOUR_PIXEL_COUNT:
        mask = alpha > 50
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
    crop_origin: Optional[list[int]] = None,
    qty_token_boxes: Optional[list[Dict[str, Any]]] = None,
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

    mask_pixels = slot_mask > 0
    ys, xs = np.where(mask_pixels)
    if xs.size == 0 or ys.size == 0:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "mask_slot_index": slot_index,
            "cutout_slot_index": slot_index,
            "error": "shape_mask_empty",
            "mask_component_reason": component_reason,
        }
    pad = 4
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(int(w), int(xs.max()) + pad + 1)
    y2 = min(int(h), int(ys.max()) + pad + 1)
    cropped_mask = slot_mask[y1:y2, x1:x2]
    cropped_img = slot_img[y1:y2, x1:x2]
    if crop_origin is not None and qty_token_boxes:
        origin_x = int(crop_origin[0]) + int(x1)
        origin_y = int(crop_origin[1]) + int(y1)
        crop_h, crop_w = cropped_mask.shape[:2]
        cropped_mask = cropped_mask.copy()
        cropped_img = cropped_img.copy()
        for qty_box in qty_token_boxes:
            qx = int(qty_box.get("x", 0) or 0)
            qy = int(qty_box.get("y", 0) or 0)
            qw = int(qty_box.get("w", 0) or 0)
            qh = int(qty_box.get("h", 0) or 0)
            if qw <= 0 or qh <= 0:
                continue
            scrub_pad = 2
            lx1 = max(0, qx - scrub_pad - origin_x)
            ly1 = max(0, qy - scrub_pad - origin_y)
            lx2 = min(crop_w, qx + qw + scrub_pad - origin_x)
            ly2 = min(crop_h, qy + qh + scrub_pad - origin_y)
            if lx2 > lx1 and ly2 > ly1:
                cropped_mask[ly1:ly2, lx1:lx2] = 0
                cropped_img[ly1:ly2, lx1:lx2] = 0
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
    bgra[bgra[:, :, 3] == 0, :3] = 0
    alpha_debug = _alpha_stats(bgra[:, :, 3])
    rgb_stats = _rgb_stats_from_bgra(bgra)
    _alpha_count = int(alpha_debug.get("alpha_pixel_count", 0) or 0)
    _ah, _aw = bgra.shape[:2]
    # Colour reliability is independent of shape quality.
    # Small cutouts, expanded masks, or missing medians produce unreliable RGB.
    _colour_confidence = (
        "low"
        if (rgb_stats.get("slot_rgb_median") is None
            or _alpha_count < 200
            or _enhancement_expanded_pct > 0.15)
        else "high"
    )
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
            "colour_confidence": _colour_confidence,
            **rgb_stats,
        }
    if crop_origin is not None and qty_token_boxes:
        final_origin_x = int(crop_origin[0]) + int(x1)
        final_origin_y = int(crop_origin[1]) + int(y1)
        final_h, final_w = bgra.shape[:2]
        final_crop_box = [final_origin_x, final_origin_y, int(final_w), int(final_h)]
        for qty_box in qty_token_boxes:
            qx = int(qty_box.get("x", 0) or 0)
            qy = int(qty_box.get("y", 0) or 0)
            qw = int(qty_box.get("w", 0) or 0)
            qh = int(qty_box.get("h", 0) or 0)
            if qw <= 0 or qh <= 0:
                continue
            scrub_pad = 2
            full_box = [qx, qy, qw, qh]
            qx1 = qx - scrub_pad
            qy1 = qy - scrub_pad
            qx2 = qx + qw + scrub_pad
            qy2 = qy + qh + scrub_pad
            cx1 = final_origin_x
            cy1 = final_origin_y
            cx2 = final_origin_x + final_w
            cy2 = final_origin_y + final_h
            ix1 = max(qx1, cx1)
            iy1 = max(qy1, cy1)
            ix2 = min(qx2, cx2)
            iy2 = min(qy2, cy2)
            if ix2 > ix1 and iy2 > iy1:
                lx1 = int(ix1 - final_origin_x)
                ly1 = int(iy1 - final_origin_y)
                lx2 = int(ix2 - final_origin_x)
                ly2 = int(iy2 - final_origin_y)
                bgra[ly1:ly2, lx1:lx2, :] = 0
            else:
                lx1 = ly1 = lx2 = ly2 = 0
            local_box = [int(lx1), int(ly1), int(max(0, lx2 - lx1)), int(max(0, ly2 - ly1))]
            applied_pixels = int(local_box[2] * local_box[3])
            print(
                "[qty-scrub] "
                f"slot={slot_index} "
                f"full_box={full_box} "
                f"local_box={local_box} "
                f"crop_box={final_crop_box} "
                f"applied_pixels={applied_pixels}"
            )
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
            "colour_confidence": _colour_confidence,
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
        "colour_confidence": _colour_confidence,
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


def _anchor_proximity_score(
    comp: Dict[str, Any],
    token: Dict[str, Any],
    img_w: int,
    img_h: int,
) -> Optional[float]:
    """Score how well a foreground component matches a qty-label anchor.

    Lower score = better match.  Returns None for hard-rejected components.

    Reference point
    ---------------
    The top-centre of the qty label: ``(token["cx"], token["y"])``.
    Parts sit above or beside this point; we bias toward above.

    Hard rejects (return None)
    --------------------------
    1. Centroid below qty-label top + small tolerance — belongs to a row below.
    2. Very wide thin strip (aspect > 4 and small area) — likely a text artifact.
    3. Euclidean distance from qty top-centre > 65 % of max(img_w, img_h).

    Score
    -----
    raw_euclidean_dist
      + abs(dx) * 0.30          # mild horizontal penalty; directly-above preferred
      - min(area, w*h*0.12) * 0.0004   # tiny area bonus for larger blobs
    """
    qty_cx = float(token.get("cx", 0) or 0)
    qty_top = float(token.get("y", 0) or 0)     # top edge of qty bbox
    comp_cx = float(comp["gcx"])
    comp_cy = float(comp["gcy"])
    comp_lw = int(comp["lw"])
    comp_lh = int(comp["lh"])
    comp_area = float(comp["area"])

    # Hard reject: centroid well below qty-label top.
    # A tiny epsilon keeps near-boundary top-row parts from being rejected by
    # sub-pixel centroid noise.
    tolerance = max(8.0, comp_lh * 0.25)
    tolerance_epsilon = 1.0
    if comp_cy > qty_top + tolerance + tolerance_epsilon:
        return None

    # Hard reject: text-shaped strip (very wide, small area)
    if comp_lh > 0 and (comp_lw / comp_lh) > 4.0 and comp_area < img_w * img_h * 0.008:
        return None

    # Hard reject: unreachably far from qty anchor
    dx = comp_cx - qty_cx
    dy = comp_cy - qty_top      # negative = above (preferred)
    raw_dist = (dx * dx + dy * dy) ** 0.5
    if raw_dist > max(img_w, img_h) * 0.65:
        return None

    area_bonus = min(comp_area, float(img_w * img_h) * 0.12) * 0.0004
    return raw_dist + abs(dx) * 0.30 - area_bonus


def _anchor_proximity_debug_reason(
    comp: Dict[str, Any],
    token: Dict[str, Any],
    img_w: int,
    img_h: int,
) -> tuple[Optional[float], str]:
    qty_cx = float(token.get("cx", 0) or 0)
    qty_top = float(token.get("y", 0) or 0)
    comp_cx = float(comp["gcx"])
    comp_cy = float(comp["gcy"])
    comp_lw = int(comp["lw"])
    comp_lh = int(comp["lh"])
    comp_area = float(comp["area"])

    tolerance = max(8.0, comp_lh * 0.25)
    tolerance_epsilon = 1.0
    if comp_cy > qty_top + tolerance + tolerance_epsilon:
        return None, (
            "centroid_below_qty_top"
            f"(gcy={comp_cy:.1f}>qty_top+tolerance+epsilon={qty_top + tolerance + tolerance_epsilon:.1f})"
        )

    if comp_lh > 0 and (comp_lw / comp_lh) > 4.0 and comp_area < img_w * img_h * 0.008:
        return None, (
            "text_shaped_strip"
            f"(aspect={float(comp_lw) / float(comp_lh):.2f},area={int(comp_area)})"
        )

    dx = comp_cx - qty_cx
    dy = comp_cy - qty_top
    raw_dist = (dx * dx + dy * dy) ** 0.5
    max_dist = max(img_w, img_h) * 0.65
    if raw_dist > max_dist:
        return None, f"too_far(raw_dist={raw_dist:.1f}>max={max_dist:.1f})"

    area_bonus = min(comp_area, float(img_w * img_h) * 0.12) * 0.0004
    return raw_dist + abs(dx) * 0.30 - area_bonus, "pass"


def _is_assignment_frame_ring(comp: Dict[str, Any], img_w: int, img_h: int) -> bool:
    area = float(comp["area"])
    lx = int(comp["lx"])
    ly = int(comp["ly"])
    lw = int(comp["lw"])
    lh = int(comp["lh"])
    fill = area / float(max(1, lw * lh))
    if lw > img_w * 0.70 or lh > img_h * 0.70:
        return True
    margin = max(3, int(min(img_w, img_h) * 0.08))
    touches = sum(
        [
            lx <= margin,
            ly <= margin,
            lx + lw >= img_w - margin,
            ly + lh >= img_h - margin,
        ]
    )
    return fill < 0.12 and touches >= 3


def _component_slice_from_full_mask(
    img: np.ndarray,
    full_component_mask: np.ndarray,
    comp: Dict[str, Any],
    *,
    pad: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    img_h, img_w = img.shape[:2]
    x1 = max(0, int(comp["lx"]) - int(pad))
    y1 = max(0, int(comp["ly"]) - int(pad))
    x2 = min(img_w, int(comp["lx"]) + int(comp["lw"]) + int(pad))
    y2 = min(img_h, int(comp["ly"]) + int(comp["lh"]) + int(pad))
    return (
        img[y1:y2, x1:x2],
        full_component_mask[y1:y2, x1:x2],
        [x1, y1, int(x2 - x1), int(y2 - y1)],
    )


def _split_large_component_watershed(
    source_mask: np.ndarray,
    component_mask: np.ndarray,
    component_bbox: list[int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Attempt watershed splitting of a large foreground component.

    Returns (accepted_child_masks, rejected_child_masks) where both lists
    contain full-image binary masks.  Accepted masks pass the minimum-area
    gate (0.2 % of image); rejected masks do not.
    Returns ([], []) when splitting yields only one region or fails.
    """
    H, W = source_mask.shape[:2]
    image_area = float(H * W)
    min_child_area = max(16, image_area * 0.002)   # 0.2 % of image

    lx, ly, lw, lh = [int(v) for v in component_bbox]
    if lw <= 0 or lh <= 0:
        return [], []

    roi_bin = (component_mask[ly:ly + lh, lx:lx + lw] > 0).astype(np.uint8) * 255
    if roi_bin.size == 0 or not roi_bin.any():
        return [], []

    # Morphology open — break thin bridges between sub-parts
    k_size = max(3, min(lw, lh) // 10)
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    opened = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, kernel, iterations=2)

    # Distance transform on opened mask
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    max_dist = float(dist.max())
    if max_dist < 1.0:
        return [], []

    # Sure foreground: peaks above 50 % of max distance
    _, sure_fg = cv2.threshold(dist, 0.5 * max_dist, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # connectedComponents markers
    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers <= 2:
        return [], []

    # Watershed on distance-transform proxy
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    dist_bgr = cv2.cvtColor(dist_u8, cv2.COLOR_GRAY2BGR)
    markers_ws = markers.astype(np.int32)
    unknown = cv2.subtract(opened, sure_fg)
    markers_ws[unknown > 0] = 0

    try:
        cv2.watershed(dist_bgr, markers_ws)
    except Exception:
        return [], []

    # Reference pixel count for parent-likeness comparisons (pre-open ROI)
    parent_px = int(np.count_nonzero(roi_bin))

    # Build child masks; apply size and parent-likeness filters
    accepted: list[np.ndarray] = []
    rejected: list[np.ndarray] = []
    for region_id in range(1, n_markers):
        child_roi = (markers_ws == region_id).astype(np.uint8) * 255
        child_area = int(np.count_nonzero(child_roi))
        full_child = np.zeros((H, W), dtype=np.uint8)
        full_child[ly:ly + lh, lx:lx + lw] = child_roi

        # Reject: below minimum size (0.2 % of image)
        if child_area < min_child_area:
            rejected.append(full_child)
            continue

        # Compute child bounding box in ROI-local coordinates
        c_ys, c_xs = np.where(child_roi > 0)
        c_bw = int(c_xs.max()) - int(c_xs.min()) + 1
        c_bh = int(c_ys.max()) - int(c_ys.min()) + 1

        # Reject: child carries >= 65 % of parent pixels — no real split
        if child_area >= parent_px * 0.65:
            rejected.append(full_child)
            continue

        # Reject: child bbox spans >= 85 % of parent bbox in both axes
        if c_bw >= lw * 0.85 and c_bh >= lh * 0.85:
            rejected.append(full_child)
            continue

        # Reject: dense fill AND bbox still dominates parent (belt-and-suspenders)
        c_fill = float(child_area) / float(max(1, c_bw * c_bh))
        if c_fill > 0.75 and c_bw >= lw * 0.80 and c_bh >= lh * 0.80:
            rejected.append(full_child)
            continue

        accepted.append(full_child)

    # Sort accepted children by top-left pixel position (row-major: y then x)
    def _cm_topleft(cm: np.ndarray) -> tuple[int, int]:
        ys, xs = np.where(cm > 0)
        return (int(ys.min()) if ys.size else 0, int(xs.min()) if xs.size else 0)

    accepted.sort(key=_cm_topleft)

    return accepted, rejected


def _scrub_non_part_components(
    mask: np.ndarray,
    img_h: int,
    img_w: int,
    crop_id: str = "",
    overlay: Optional[np.ndarray] = None,
) -> tuple[int, list[Dict[str, Any]]]:
    """Zero pixels belonging to non-part structures in *mask* (in-place).

    Runs its own connectedComponentsWithStats pass and classifies each
    component using three geometric heuristics:

    frame_outline
        Component bbox touches all four edges within border_margin ``bm``
        AND fill-ratio < 0.35 — thin perimeter frame / callout border.

    large_block
        area > 18 % of image AND fill-ratio > 0.65 — solid background block.
        Watershed splitting is attempted; accepted children are returned as
        independent component records (NOT restored to mask) so they surface
        as separate labels in global_comps downstream.

    text_strip
        aspect ratio (lw/lh) > 5.0 AND height < 10 % of image height AND
        area < 1.2 % of image — narrow horizontal text strip.

    Returns (total_zeroed_px, split_children) where split_children is a list
    of component dicts ready to extend global_comps; each carries a
    "split_mask" key with the full-image binary mask for extraction.
    """
    if mask is None or mask.size == 0:
        return 0, []

    split_children: list[Dict[str, Any]] = []
    H, W = img_h, img_w
    image_area = float(H * W)
    bm = max(3, min(H, W) // 16)   # border margin for frame detection

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    total_zeroed = 0

    for lab in range(1, n_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        lx = int(stats[lab, cv2.CC_STAT_LEFT])
        ly = int(stats[lab, cv2.CC_STAT_TOP])
        lw = int(stats[lab, cv2.CC_STAT_WIDTH])
        lh = int(stats[lab, cv2.CC_STAT_HEIGHT])

        if lw <= 0 or lh <= 0:
            continue

        bbox_area = lw * lh
        fill_ratio = float(area) / float(bbox_area)
        verdict = None

        # 1. Frame outline: bbox spans the full image within bm, but sparse fill
        touches_all_edges = (lx <= bm and ly <= bm and lx + lw >= W - bm and ly + lh >= H - bm)
        if touches_all_edges and fill_ratio < 0.35:
            verdict = "REJECT_frame_outline"

        # 2. Large solid block: covers >18 % of image and fill-ratio >65 %
        if verdict is None and area > image_area * 0.18 and fill_ratio > 0.65:
            verdict = "REJECT_large_block"

        # 3. Text strip: very wide, short, and small area
        if verdict is None:
            aspect = float(lw) / float(lh)
            if aspect > 5.0 and lh < H * 0.10 and area < image_area * 0.012:
                verdict = "REJECT_text_strip"

        print(
            f"[auto-mask-scrub] crop_id={crop_id} lab={lab} area={area} "
            f"bbox=[{lx},{ly},{lw},{lh}] fill={fill_ratio:.2f} "
            f"verdict={verdict or 'ok'}"
        )

        print(
            f"[auto-mask-scrub] "
            f"crop_id={crop_id} "
            f"lab={lab} "
            f"area={area} "
            f"fill={fill_ratio:.2f} "
            f"verdict={verdict}"
        )

        if verdict is not None:
            if verdict == "REJECT_large_block":
                comp_mask_full = (labels == lab).astype(np.uint8) * 255
                child_accepted, child_rejected = _split_large_component_watershed(
                    mask, comp_mask_full, [lx, ly, lw, lh]
                )
                n_total = len(child_accepted) + len(child_rejected)
                print(
                    f"[auto-mask-split] crop_id={crop_id} "
                    f"parent_area={area} "
                    f"child_count={n_total} "
                    f"accepted={len(child_accepted)} "
                    f"rejected={len(child_rejected)}"
                )
                # Zero the parent — always removed from mask
                mask[labels == lab] = 0
                total_zeroed += area
                if overlay is not None:
                    overlay[labels == lab] = (0, 0, 255)   # red = rejected parent
                # Register accepted children as independent component records.
                # NOT written back to mask — each gets its own split_mask so
                # connectedComponents never merges them into a single lab.
                for _ci, cm in enumerate(child_accepted):
                    _cm_area = int(np.count_nonzero(cm))
                    _ys, _xs = np.where(cm > 0)
                    _clx = int(_xs.min())
                    _cly = int(_ys.min())
                    _clw = int(_xs.max()) - _clx + 1
                    _clh = int(_ys.max()) - _cly + 1
                    _cgcx = float(_xs.mean())
                    _cgcy = float(_ys.mean())
                    split_children.append({
                        "lab": -1,
                        "area": _cm_area,
                        "lx": _clx, "ly": _cly, "lw": _clw, "lh": _clh,
                        "gcx": _cgcx,
                        "gcy": _cgcy,
                        "used": False,
                        "split_mask": cm,
                    })
                    print(
                        f"[auto-mask-split-child] child_index={_ci} "
                        f"area={_cm_area} "
                        f"bbox=[{_clx},{_cly},{_clw},{_clh}]"
                    )
                    if overlay is not None:
                        overlay[cm > 0] = (0, 255, 0)      # green = accepted children
                if overlay is not None:
                    for cm in child_rejected:
                        overlay[cm > 0] = (0, 165, 255)    # orange = rejected children
            else:
                mask[labels == lab] = 0
                total_zeroed += area

    return total_zeroed, split_children


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
        master_mask, master_mask_error = _part_mask_for_callout_crop(img)

    # ── Raw master_mask debug ────────────────────────────────────────────────
    # Written immediately after master_mask is finalised, before any
    # scrub / split / zeroing / rejection.  Three files per crop:
    #   <stem>_raw_master_mask.png        — binary mask as-is
    #   <stem>_raw_master_overlay.png     — img with mask pixels tinted red
    #   <stem>_master_island_overlay.png  — img with per-island bbox + index
    if master_mask is not None:
        # Use the structured stem written by create_full_crop_mask_debug so all
        # debug files for the same crop share the same base name.  Falls back to
        # the temp-file stem when the mask path is unavailable.
        _dbg_mask_path = str(full_mask_debug.get("full_crop_mask_path") or "")
        if _dbg_mask_path:
            _dbg_stem = Path(_dbg_mask_path).stem.removesuffix("_full_mask")
        else:
            _dbg_stem = source_path.stem
        _FULL_CROP_MASK_DIR.mkdir(parents=True, exist_ok=True)
        _FULL_CROP_MASK_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
        # 1. Raw binary mask
        _raw_master_mask_path = _FULL_CROP_MASK_DIR / f"{_dbg_stem}_raw_master_mask.png"
        if cv2.imwrite(str(_raw_master_mask_path), master_mask):
            print(f"[full-mask-debug-write] path={_raw_master_mask_path}")
        else:
            print(f"[full-mask-debug-skip] reason=raw_master_mask_write_failed path={_raw_master_mask_path}")
        # 2. Raw overlay: tint mask pixels red
        _dbg_raw_ov = img.copy()
        _dbg_raw_px = master_mask > 0
        _dbg_raw_ov[_dbg_raw_px] = (
            _dbg_raw_ov[_dbg_raw_px].astype(np.float32) * 0.5
            + np.array([0, 0, 180], dtype=np.float32)
        ).clip(0, 255).astype(np.uint8)
        _raw_master_overlay_path = _FULL_CROP_MASK_OVERLAY_DIR / f"{_dbg_stem}_raw_master_overlay.png"
        if cv2.imwrite(str(_raw_master_overlay_path), _dbg_raw_ov):
            print(f"[full-mask-debug-write] path={_raw_master_overlay_path}")
        else:
            print(f"[full-mask-debug-skip] reason=raw_master_overlay_write_failed path={_raw_master_overlay_path}")
        # 3. Island overlay: connectedComponents on raw mask, numbered bboxes
        _dbg_n, _dbg_labs, _dbg_stats, _dbg_cents = cv2.connectedComponentsWithStats(
            (master_mask > 0).astype(np.uint8), 8
        )
        _dbg_island_ov = img.copy()
        print(f"[raw-master-mask] crop_id={crop_id} master_island_count={_dbg_n - 1}")
        for _dbg_i in range(1, _dbg_n):
            _dbg_lx = int(_dbg_stats[_dbg_i, cv2.CC_STAT_LEFT])
            _dbg_ly = int(_dbg_stats[_dbg_i, cv2.CC_STAT_TOP])
            _dbg_lw = int(_dbg_stats[_dbg_i, cv2.CC_STAT_WIDTH])
            _dbg_lh = int(_dbg_stats[_dbg_i, cv2.CC_STAT_HEIGHT])
            _dbg_area = int(_dbg_stats[_dbg_i, cv2.CC_STAT_AREA])
            _dbg_fill = float(_dbg_area) / float(max(1, _dbg_lw * _dbg_lh))
            print(
                f"[raw-master-island] crop_id={crop_id} idx={_dbg_i} "
                f"area={_dbg_area} "
                f"bbox=[{_dbg_lx},{_dbg_ly},{_dbg_lw},{_dbg_lh}] "
                f"fill={_dbg_fill:.2f}"
            )
            cv2.rectangle(
                _dbg_island_ov,
                (_dbg_lx, _dbg_ly),
                (_dbg_lx + _dbg_lw - 1, _dbg_ly + _dbg_lh - 1),
                (0, 200, 0),
                1,
            )
            cv2.putText(
                _dbg_island_ov,
                str(_dbg_i),
                (_dbg_lx + 2, max(_dbg_ly + 12, _dbg_ly + _dbg_lh // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        _master_island_overlay_path = _FULL_CROP_MASK_OVERLAY_DIR / f"{_dbg_stem}_master_island_overlay.png"
        if cv2.imwrite(str(_master_island_overlay_path), _dbg_island_ov):
            print(f"[full-mask-debug-write] path={_master_island_overlay_path}")
        else:
            print(f"[full-mask-debug-skip] reason=master_island_overlay_write_failed path={_master_island_overlay_path}")
    else:
        print(f"[full-mask-debug-skip] reason=master_mask_missing crop_id={crop_id}")
    # ── end raw master_mask debug ────────────────────────────────────────────

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
    _slot_assignment_debug = "p22_s26_c1" in str(crop_id or "")
    _slot5_debug = "p22_s26_c1" in str(crop_id or "")
    if _slot5_debug:
        for _dbg_slot_index, _dbg_token in enumerate(ordered_tokens):
            print(
                "[slot5-debug-anchor] "
                f"crop_id={crop_id} slot_index={_dbg_slot_index} "
                f"anchor=({int(_dbg_token.get('cx', 0) or 0)},{int(_dbg_token.get('y', 0) or 0)}) "
                f"qty_box=[{int(_dbg_token.get('x', 0) or 0)},{int(_dbg_token.get('y', 0) or 0)},"
                f"{int(_dbg_token.get('w', 0) or 0)},{int(_dbg_token.get('h', 0) or 0)}] "
                f"text={str(_dbg_token.get('text') or '')}"
            )

    def _nearest_qty_anchor_for_debug(comp: Dict[str, Any]) -> tuple[Optional[int], Optional[Dict[str, Any]], float]:
        nearest_slot: Optional[int] = None
        nearest_token: Optional[Dict[str, Any]] = None
        nearest_dist = float("inf")
        for _slot_i, _tok in enumerate(ordered_tokens):
            _dx = float(comp["gcx"]) - float(_tok.get("cx", 0) or 0)
            _dy = float(comp["gcy"]) - float(_tok.get("y", 0) or 0)
            _dist = (_dx * _dx + _dy * _dy) ** 0.5
            if _dist < nearest_dist:
                nearest_dist = _dist
                nearest_slot = _slot_i
                nearest_token = _tok
        return nearest_slot, nearest_token, nearest_dist

    # If 3+ tokens share the same qty label the global connected component tends
    # to be one merged blob spanning all slots.  Use per-slot window fallback instead.
    _qty_text_freq: Dict[str, int] = {}
    for _t in ordered_tokens:
        _k = str(_t.get("text") or "").strip()
        _qty_text_freq[_k] = _qty_text_freq.get(_k, 0) + 1
    _force_window_fallback = bool(_qty_text_freq and max(_qty_text_freq.values()) >= 3)
    if _force_window_fallback:
        print(f"[auto-mask] crop_id={crop_id} disabling window_fallback for repeated qty labels x{max(_qty_text_freq.values())}")

    slots: list[Dict[str, Any]] = []

    # Stage 1 — build clean_mask: erase all qty token glyphs globally before any
    # component analysis so OCR text pixels never contaminate part extractions.
    clean_mask: Optional[np.ndarray] = None
    pre_scrub_mask: Optional[np.ndarray] = None   # snapshot before scrub — slot fallback
    _split_children: list[Dict[str, Any]] = []
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

        # Snapshot before scrub: REJECT_large_block zeros pixels destructively;
        # pre_scrub_mask lets unresolved slots fall back to the original pixels
        # without exposing the scrubbed mask as the only candidate source.
        pre_scrub_mask = clean_mask.copy()
        # Scrub frame / block / text-strip components before slot ownership.
        if clean_mask is not None:
            _scrubbed_px, _split_children = _scrub_non_part_components(clean_mask, h, w, crop_id=crop_id)
            if _scrubbed_px > 0:
                print(f"[auto-mask-scrub] crop_id={crop_id} zeroed {_scrubbed_px} non-part pixels")

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
        _min_comp_area = max(16.0, float(h * w) * 0.0004)
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
            _global_comp = {
                "lab": _lab,
                "area": _garea,
                "lx": _glx, "ly": _gly, "lw": _glw, "lh": _glh,
                "gcx": float(_glcentroids[_lab][0]),
                "gcy": float(_glcentroids[_lab][1]),
                "used": False,
            }
            if _is_assignment_frame_ring(_global_comp, w, h):
                if _slot_assignment_debug:
                    _gfill = float(_garea) / float(max(1, _glw * _glh))
                    print(
                        "[slot-assign-debug-component] "
                        f"crop_id={crop_id} pool=global component_index={_lab} "
                        f"lab={_lab} bbox=[{_glx},{_gly},{_glw},{_glh}] "
                        f"area={_garea} centroid=({_global_comp['gcx']:.1f},{_global_comp['gcy']:.1f}) "
                        f"assigned_slot_index=None rejection_reason=frame_ring(fill={_gfill:.2f})"
                    )
                continue
            global_comps.append(_global_comp)
        # Watershed split children bypassed connectedComponentsWithStats entirely
        # (they were never written back to clean_mask).  Inject them now so the
        # slot-assignment loop sees them as separate, independent components.
        global_comps.extend(_split_children)
        for _debug_comp_index, _debug_comp in enumerate(global_comps, start=1):
            _debug_comp["debug_comp_index"] = _debug_comp_index
            if _slot_assignment_debug:
                _nearest_slot, _nearest_tok, _nearest_dist = _nearest_qty_anchor_for_debug(_debug_comp)
                _anchor_text = (
                    f"({int(_nearest_tok.get('cx', 0) or 0)},{int(_nearest_tok.get('y', 0) or 0)})"
                    if _nearest_tok is not None
                    else "None"
                )
                _nearest_reject = "no_qty_anchor"
                _nearest_score: Optional[float] = None
                if _nearest_tok is not None:
                    _nearest_score, _nearest_reject = _anchor_proximity_debug_reason(_debug_comp, _nearest_tok, w, h)
                print(
                    "[slot-assign-debug-component] "
                    f"crop_id={crop_id} pool=global component_index={_debug_comp_index} "
                    f"lab={_debug_comp.get('lab')} "
                    f"bbox=[{_debug_comp['lx']},{_debug_comp['ly']},{_debug_comp['lw']},{_debug_comp['lh']}] "
                    f"area={_debug_comp['area']} "
                    f"centroid=({_debug_comp['gcx']:.1f},{_debug_comp['gcy']:.1f}) "
                    f"nearest_slot_index={_nearest_slot} nearest_qty_anchor={_anchor_text} "
                    f"nearest_dist={_nearest_dist:.1f} "
                    f"nearest_score={_nearest_score if _nearest_score is not None else 'None'} "
                    f"nearest_rejection_reason={_nearest_reject}"
                )

    # ── Master-island component pool ─────────────────────────────────────────────
    # Built from the original foreground mask (before OCR erasure or scrubbing).
    # Used as a last-resort fallback when global_comps has no candidate for a slot.
    # Applies the same rejection filters as _scrub_non_part_components so large
    # blocks, frame outlines, and text strips are excluded; only genuine part-shaped
    # islands survive.  Each entry carries its own pixel mask for extraction so the
    # slot loop never needs to crop a rectangular window.
    _master_island_comps: list[Dict[str, Any]] = []
    if master_mask is not None:
        _mi_image_area = float(h * w)
        _mi_bm = max(3, min(h, w) // 16)
        _mi_min_area = max(16.0, _mi_image_area * 0.002)
        _mi_glc, _mi_gl, _mi_gls, _mi_gcents = cv2.connectedComponentsWithStats(
            (master_mask > 0).astype(np.uint8), 8
        )
        for _mi_lab in range(1, _mi_glc):
            _mi_a = int(_mi_gls[_mi_lab, cv2.CC_STAT_AREA])
            _mi_lx = int(_mi_gls[_mi_lab, cv2.CC_STAT_LEFT])
            _mi_ly = int(_mi_gls[_mi_lab, cv2.CC_STAT_TOP])
            _mi_lw = int(_mi_gls[_mi_lab, cv2.CC_STAT_WIDTH])
            _mi_lh = int(_mi_gls[_mi_lab, cv2.CC_STAT_HEIGHT])
            _mi_reject_reason = ""
            _slot5_rgb_median = None
            try:
                _mi_px = img[_mi_gl == _mi_lab]
                if _mi_px.size > 0:
                    _slot5_rgb_median = [
                        int(np.median(_mi_px[:, 2])),
                        int(np.median(_mi_px[:, 1])),
                        int(np.median(_mi_px[:, 0])),
                    ]
            except Exception:
                _slot5_rgb_median = None
            if _slot5_debug:
                print(
                    "[slot5-debug-master-island] "
                    f"crop_id={crop_id} component_index={_mi_lab} lab={_mi_lab} "
                    f"bbox=[{_mi_lx},{_mi_ly},{_mi_lw},{_mi_lh}] area={_mi_a} "
                    f"centroid=({_mi_gcents[_mi_lab][0]:.1f},{_mi_gcents[_mi_lab][1]:.1f}) "
                    f"rgb_median={_slot5_rgb_median}"
                )
            if _mi_lw <= 2 or _mi_lh <= 2 or _mi_a < _mi_min_area:
                if _slot_assignment_debug:
                    _mi_reject_reason = (
                        f"too_small_or_thin(lw={_mi_lw},lh={_mi_lh},"
                        f"area={_mi_a},min_area={_mi_min_area:.1f})"
                    )
                    print(
                        "[slot-assign-debug-master-island] "
                        f"crop_id={crop_id} component_index={_mi_lab} lab={_mi_lab} "
                        f"bbox=[{_mi_lx},{_mi_ly},{_mi_lw},{_mi_lh}] area={_mi_a} "
                        f"centroid=({_mi_gcents[_mi_lab][0]:.1f},{_mi_gcents[_mi_lab][1]:.1f}) "
                        f"assigned_slot_index=None rejection_reason={_mi_reject_reason}"
                    )
                continue
            _mi_fill = float(_mi_a) / float(max(1, _mi_lw * _mi_lh))
            _mi_candidate_comp = {
                "area": _mi_a,
                "lx": _mi_lx, "ly": _mi_ly, "lw": _mi_lw, "lh": _mi_lh,
                "gcx": float(_mi_gcents[_mi_lab][0]),
                "gcy": float(_mi_gcents[_mi_lab][1]),
            }
            if _is_assignment_frame_ring(_mi_candidate_comp, w, h):
                if _slot_assignment_debug:
                    print(
                        "[slot-assign-debug-master-island] "
                        f"crop_id={crop_id} component_index={_mi_lab} lab={_mi_lab} "
                        f"bbox=[{_mi_lx},{_mi_ly},{_mi_lw},{_mi_lh}] area={_mi_a} "
                        f"centroid=({_mi_gcents[_mi_lab][0]:.1f},{_mi_gcents[_mi_lab][1]:.1f}) "
                        f"assigned_slot_index=None rejection_reason=frame_ring(fill={_mi_fill:.2f})"
                    )
                continue
            # Reject frame outlines
            _mi_touches = (
                _mi_lx <= _mi_bm and _mi_ly <= _mi_bm
                and _mi_lx + _mi_lw >= w - _mi_bm
                and _mi_ly + _mi_lh >= h - _mi_bm
            )
            if _mi_touches and _mi_fill < 0.35:
                if _slot_assignment_debug:
                    _mi_reject_reason = f"frame_outline(fill={_mi_fill:.2f})"
                    print(
                        "[slot-assign-debug-master-island] "
                        f"crop_id={crop_id} component_index={_mi_lab} lab={_mi_lab} "
                        f"bbox=[{_mi_lx},{_mi_ly},{_mi_lw},{_mi_lh}] area={_mi_a} "
                        f"centroid=({_mi_gcents[_mi_lab][0]:.1f},{_mi_gcents[_mi_lab][1]:.1f}) "
                        f"assigned_slot_index=None rejection_reason={_mi_reject_reason}"
                    )
                continue
            # Reject large solid blocks
            if _mi_a > _mi_image_area * 0.18 and _mi_fill > 0.65:
                if _slot_assignment_debug:
                    _mi_reject_reason = f"large_solid_block(fill={_mi_fill:.2f})"
                    print(
                        "[slot-assign-debug-master-island] "
                        f"crop_id={crop_id} component_index={_mi_lab} lab={_mi_lab} "
                        f"bbox=[{_mi_lx},{_mi_ly},{_mi_lw},{_mi_lh}] area={_mi_a} "
                        f"centroid=({_mi_gcents[_mi_lab][0]:.1f},{_mi_gcents[_mi_lab][1]:.1f}) "
                        f"assigned_slot_index=None rejection_reason={_mi_reject_reason}"
                    )
                continue
            # Reject text strips
            _mi_aspect = float(_mi_lw) / float(max(1, _mi_lh))
            if _mi_aspect > 5.0 and _mi_lh < h * 0.10 and _mi_a < _mi_image_area * 0.012:
                if _slot_assignment_debug:
                    _mi_reject_reason = f"text_strip(aspect={_mi_aspect:.2f})"
                    print(
                        "[slot-assign-debug-master-island] "
                        f"crop_id={crop_id} component_index={_mi_lab} lab={_mi_lab} "
                        f"bbox=[{_mi_lx},{_mi_ly},{_mi_lw},{_mi_lh}] area={_mi_a} "
                        f"centroid=({_mi_gcents[_mi_lab][0]:.1f},{_mi_gcents[_mi_lab][1]:.1f}) "
                        f"assigned_slot_index=None rejection_reason={_mi_reject_reason}"
                    )
                continue
            # Store full-image pixel mask so extraction never needs _gl or a window
            _mi_cm = (_mi_gl == _mi_lab).astype(np.uint8) * 255
            _mi_comp = {
                "lab": -2,
                "debug_comp_index": _mi_lab,
                "area": _mi_a,
                "lx": _mi_lx, "ly": _mi_ly, "lw": _mi_lw, "lh": _mi_lh,
                "gcx": float(_mi_gcents[_mi_lab][0]),
                "gcy": float(_mi_gcents[_mi_lab][1]),
                "used": False,
                "split_mask": _mi_cm,
            }
            _master_island_comps.append(_mi_comp)
            if _slot_assignment_debug:
                _nearest_slot, _nearest_tok, _nearest_dist = _nearest_qty_anchor_for_debug(_mi_comp)
                _anchor_text = (
                    f"({int(_nearest_tok.get('cx', 0) or 0)},{int(_nearest_tok.get('y', 0) or 0)})"
                    if _nearest_tok is not None
                    else "None"
                )
                _nearest_score: Optional[float] = None
                _nearest_reject = "no_qty_anchor"
                if _nearest_tok is not None:
                    _nearest_score, _nearest_reject = _anchor_proximity_debug_reason(_mi_comp, _nearest_tok, w, h)
                print(
                    "[slot-assign-debug-master-island] "
                    f"crop_id={crop_id} component_index={_mi_lab} lab={_mi_lab} "
                    f"bbox=[{_mi_lx},{_mi_ly},{_mi_lw},{_mi_lh}] area={_mi_a} "
                    f"centroid=({_mi_comp['gcx']:.1f},{_mi_comp['gcy']:.1f}) "
                    f"assigned_slot_index=None nearest_slot_index={_nearest_slot} "
                    f"nearest_qty_anchor={_anchor_text} nearest_dist={_nearest_dist:.1f} "
                    f"nearest_score={_nearest_score if _nearest_score is not None else 'None'} "
                    f"rejection_reason={_nearest_reject if _nearest_reject != 'pass' else 'none'}"
                )

    # Use the full/raw master-mask islands as the assignment source of truth.
    # The scrubbed global pool above is retained only for diagnostics/history;
    # slot crops must be selected and exported from these white islands.
    global_comps = list(_master_island_comps)
    for _debug_comp_index, _debug_comp in enumerate(global_comps, start=1):
        _debug_comp["debug_comp_index"] = _debug_comp_index
    if _slot5_debug:
        _slot5_token = ordered_tokens[5] if len(ordered_tokens) > 5 else None
        if _slot5_token is None:
            print(
                "[slot5-debug-score] "
                f"crop_id={crop_id} slot_index=5 reason=slot5_anchor_missing "
                f"anchor_count={len(ordered_tokens)}"
            )
        else:
            _slot5_anchor = (
                int(_slot5_token.get("cx", 0) or 0),
                int(_slot5_token.get("y", 0) or 0),
            )
            for _slot5_comp in global_comps:
                _slot5_score, _slot5_reason = _anchor_proximity_debug_reason(_slot5_comp, _slot5_token, w, h)
                _slot5_nearest, _, _slot5_nearest_dist = _nearest_qty_anchor_for_debug(_slot5_comp)
                print(
                    "[slot5-debug-score] "
                    f"crop_id={crop_id} slot_index=5 "
                    f"component_index={_slot5_comp.get('debug_comp_index')} lab={_slot5_comp.get('lab')} "
                    f"bbox=[{_slot5_comp['lx']},{_slot5_comp['ly']},{_slot5_comp['lw']},{_slot5_comp['lh']}] "
                    f"area={_slot5_comp['area']} centroid=({_slot5_comp['gcx']:.1f},{_slot5_comp['gcy']:.1f}) "
                    f"slot5_anchor={_slot5_anchor} nearest_slot_index={_slot5_nearest} "
                    f"nearest_dist={_slot5_nearest_dist:.1f} "
                    f"score={_slot5_score if _slot5_score is not None else 'None'} "
                    f"reason={_slot5_reason if _slot5_reason != 'pass' else 'pass'} "
                    f"brown_brick_candidate={_slot5_nearest == 5}"
                )

    _slot_assignment_debug_assigned_comp_ids: set[int] = set()

    for token in ordered_tokens:
        token_y = int(token.get("y", 0) or 0)
        token_h = int(token.get("h", 0) or 0)
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

        _token_bottom = token_y + token_h

        if _USE_QTY_ANCHOR_PROXIMITY:
            # ── Qty-anchor proximity ownership ──────────────────────────────────
            # Primary: score every unused global component against this qty anchor.
            # Lower score = closer/better.  See _anchor_proximity_score for rules.
            _dbg = f"[auto-mask-comp] crop_id={crop_id} slot={slot_index} qty_anchor=({cx_tok},{token_y})"
            candidates_prox: list[tuple[float, Dict[str, Any]]] = []
            for comp in global_comps:
                _debug_score: Optional[float] = None
                _debug_reason = ""
                if _slot_assignment_debug:
                    _debug_score, _debug_reason = _anchor_proximity_debug_reason(comp, token, w, h)
                if comp["used"]:
                    print(f"{_dbg} lab={comp['lab']} verdict=SKIP_used")
                    if _slot_assignment_debug:
                        print(
                            "[slot-assign-debug-score] "
                            f"crop_id={crop_id} slot_index={slot_index} "
                            f"component_index={comp.get('debug_comp_index')} lab={comp.get('lab')} "
                            f"bbox=[{comp['lx']},{comp['ly']},{comp['lw']},{comp['lh']}] "
                            f"area={comp['area']} centroid=({comp['gcx']:.1f},{comp['gcy']:.1f}) "
                            f"qty_anchor=({cx_tok},{token_y}) assigned_slot_index=None "
                            f"rejection_reason=already_used"
                        )
                    continue
                _prox = _anchor_proximity_score(comp, token, w, h)
                if _prox is None:
                    print(f"{_dbg} lab={comp['lab']} area={comp['area']} bbox=[{comp['lx']},{comp['ly']},{comp['lw']},{comp['lh']}] centroid=({comp['gcx']:.1f},{comp['gcy']:.1f}) verdict=SKIP_proximity")
                    if _slot_assignment_debug:
                        print(
                            "[slot-assign-debug-score] "
                            f"crop_id={crop_id} slot_index={slot_index} "
                            f"component_index={comp.get('debug_comp_index')} lab={comp.get('lab')} "
                            f"bbox=[{comp['lx']},{comp['ly']},{comp['lw']},{comp['lh']}] "
                            f"area={comp['area']} centroid=({comp['gcx']:.1f},{comp['gcy']:.1f}) "
                            f"qty_anchor=({cx_tok},{token_y}) assigned_slot_index=None "
                            f"rejection_reason={_debug_reason or 'SKIP_proximity'}"
                        )
                    continue
                print(f"{_dbg} lab={comp['lab']} area={comp['area']} bbox=[{comp['lx']},{comp['ly']},{comp['lw']},{comp['lh']}] centroid=({comp['gcx']:.1f},{comp['gcy']:.1f}) score={_prox:.1f} verdict=PASS")
                if _slot_assignment_debug:
                    print(
                        "[slot-assign-debug-score] "
                        f"crop_id={crop_id} slot_index={slot_index} "
                        f"component_index={comp.get('debug_comp_index')} lab={comp.get('lab')} "
                        f"bbox=[{comp['lx']},{comp['ly']},{comp['lw']},{comp['lh']}] "
                        f"area={comp['area']} centroid=({comp['gcx']:.1f},{comp['gcy']:.1f}) "
                        f"qty_anchor=({cx_tok},{token_y}) assigned_slot_index=None "
                        f"score={_debug_score if _debug_score is not None else _prox:.1f} "
                        f"rejection_reason=none"
                    )
                candidates_prox.append((_prox, comp))

            if not candidates_prox:
                if _slot5_debug and slot_index == 5:
                    print(
                        "[slot5-debug-not-written] "
                        f"crop_id={crop_id} slot_index=5 reason=no_candidate_passed_scoring "
                        f"candidate_count={len(global_comps)}"
                    )
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

            else:
                # Pick global component with lowest proximity score (closest to anchor).
                best_comp = min(candidates_prox, key=lambda item: item[0])[1]
                best_comp["used"] = True
                _slot_assignment_debug_assigned_comp_ids.add(id(best_comp))
                if _slot5_debug and slot_index == 5:
                    _slot5_best_score = min(candidates_prox, key=lambda item: item[0])[0]
                    print(
                        "[slot5-debug-selected] "
                        f"crop_id={crop_id} slot_index=5 "
                        f"component_index={best_comp.get('debug_comp_index')} lab={best_comp.get('lab')} "
                        f"bbox=[{best_comp['lx']},{best_comp['ly']},{best_comp['lw']},{best_comp['lh']}] "
                        f"area={best_comp['area']} centroid=({best_comp['gcx']:.1f},{best_comp['gcy']:.1f}) "
                        f"score={_slot5_best_score:.1f}"
                    )
                if _slot_assignment_debug:
                    print(
                        "[slot-assign-debug-assigned] "
                        f"crop_id={crop_id} assigned_slot_index={slot_index} "
                        f"component_index={best_comp.get('debug_comp_index')} lab={best_comp.get('lab')} "
                        f"bbox=[{best_comp['lx']},{best_comp['ly']},{best_comp['lw']},{best_comp['lh']}] "
                        f"area={best_comp['area']} centroid=({best_comp['gcx']:.1f},{best_comp['gcy']:.1f}) "
                        f"assignment_source=global_mask_component"
                    )
                extraction_reason = "global_mask_component"
                ex = best_comp["lx"]
                ey_val = best_comp["ly"]
                ew = best_comp["lw"]
                eh = best_comp["lh"]
                if best_comp.get("split_mask") is not None:
                    slot_img, component_mask, extraction_box = _component_slice_from_full_mask(
                        img,
                        best_comp["split_mask"],
                        best_comp,
                        pad=4,
                    )
                else:
                    extraction_box = [ex, ey_val, ew, eh]
                    slot_img = img[ey_val: ey_val + eh, ex: ex + ew]
                    component_mask = (_gl[ey_val: ey_val + eh, ex: ex + ew] == best_comp["lab"]).astype(np.uint8) * 255

        else:
            # ── Legacy column-band ownership (kept for reversibility) ────────────
            # Activate by setting _USE_QTY_ANCHOR_PROXIMITY = False at the top of
            # this file.  No logic has been changed from the previous version.
            _dbg = f"[auto-mask-comp] crop_id={crop_id} slot={slot_index} token_y={token_y} token_bottom={_token_bottom} x=[{x_left},{x_right}]"
            candidates = []
            if not _force_window_fallback:
                for comp in global_comps:
                    if comp["used"]:
                        print(f"{_dbg} lab={comp['lab']} area={comp['area']} bbox=[{comp['lx']},{comp['ly']},{comp['lw']},{comp['lh']}] centroid=({comp['gcx']:.1f},{comp['gcy']:.1f}) verdict=SKIP_used")
                        continue
                    if comp["gcy"] >= _token_bottom:
                        print(f"{_dbg} lab={comp['lab']} area={comp['area']} bbox=[{comp['lx']},{comp['ly']},{comp['lw']},{comp['lh']}] centroid=({comp['gcx']:.1f},{comp['gcy']:.1f}) verdict=SKIP_centroid_below(gcy={comp['gcy']:.1f}>={_token_bottom})")
                        continue
                    _comp_x2 = comp["lx"] + comp["lw"]
                    _overlap_w = max(0, min(_comp_x2, x_right) - max(comp["lx"], x_left))
                    if _overlap_w == 0:
                        print(f"{_dbg} lab={comp['lab']} area={comp['area']} bbox=[{comp['lx']},{comp['ly']},{comp['lw']},{comp['lh']}] centroid=({comp['gcx']:.1f},{comp['gcy']:.1f}) verdict=SKIP_no_x_overlap")
                        continue
                    _overlap_frac = _overlap_w / max(1, comp["lw"])
                    hdist = abs(comp["gcx"] - float(cx_tok))
                    print(f"{_dbg} lab={comp['lab']} area={comp['area']} bbox=[{comp['lx']},{comp['ly']},{comp['lw']},{comp['lh']}] centroid=({comp['gcx']:.1f},{comp['gcy']:.1f}) overlap_frac={_overlap_frac:.2f} verdict=PASS")
                    candidates.append((comp, hdist, _overlap_frac))

            if not candidates:
                _wf_best_label = None
                _wf_stats_arr = None
                _wf_labels_arr = None
                if not _force_window_fallback and clean_mask is not None and _token_bottom > 0 and x_right > x_left:
                    _win_mask = clean_mask[0:_token_bottom, int(x_left):int(x_right)]
                    if _win_mask.any():
                        _wf_n, _wf_labels_arr, _wf_stats_arr, _ = cv2.connectedComponentsWithStats(
                            _win_mask, connectivity=8
                        )
                        _wf_best_area = 0
                        for _li in range(1, _wf_n):
                            _a = int(_wf_stats_arr[_li, cv2.CC_STAT_AREA])
                            if _a > _wf_best_area and _a >= _min_comp_area:
                                _wf_best_area = _a
                                _wf_best_label = _li

                # Fallback 2: master-island proximity (replaces pre_scrub_window_fallback).
                _mi_best_comp: Optional[Dict[str, Any]] = None
                if _wf_best_label is None and _master_island_comps:
                    _mi_scored = []
                    for _mic in _master_island_comps:
                        _ms = _anchor_proximity_score(_mic, token, w, h)
                        if _ms is not None:
                            _mi_scored.append((_ms, _mic))
                    if _mi_scored:
                        _mi_best_comp = min(_mi_scored, key=lambda _x: _x[0])[1]

                if _wf_best_label is None and _mi_best_comp is None:
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

                if _mi_best_comp is not None:
                    # Extract directly from island's full-image split_mask — no window.
                    ex = _mi_best_comp["lx"]
                    ey_val = _mi_best_comp["ly"]
                    ew = _mi_best_comp["lw"]
                    eh = _mi_best_comp["lh"]
                    extraction_box = [ex, ey_val, ew, eh]
                    extraction_reason = "master_island_fallback"
                    slot_img = img[ey_val: ey_val + eh, ex: ex + ew]
                    component_mask = _mi_best_comp["split_mask"][ey_val: ey_val + eh, ex: ex + ew]
                    best_comp = _mi_best_comp
                    print(
                        "[auto-mask-slot] "
                        f"crop_id={crop_id} slot_index={slot_index} "
                        f"function_path=master_island_fallback "
                        f"source_crop={source_path} slot_window={slot_box} "
                        f"component_box={extraction_box} generated_at={generated_at}"
                    )
                else:
                    _lx_loc = int(_wf_stats_arr[_wf_best_label, cv2.CC_STAT_LEFT])
                    _ly_loc = int(_wf_stats_arr[_wf_best_label, cv2.CC_STAT_TOP])
                    _lw_loc = int(_wf_stats_arr[_wf_best_label, cv2.CC_STAT_WIDTH])
                    _lh_loc = int(_wf_stats_arr[_wf_best_label, cv2.CC_STAT_HEIGHT])
                    ex = int(x_left) + _lx_loc
                    ey_val = _ly_loc
                    ew = _lw_loc
                    eh = _lh_loc
                    extraction_box = [ex, ey_val, ew, eh]
                    extraction_reason = "window_fallback"
                    slot_img = img[ey_val: ey_val + eh, ex: ex + ew]
                    component_mask = (
                        _wf_labels_arr[_ly_loc: _ly_loc + eh, _lx_loc: _lx_loc + ew] == _wf_best_label
                    ).astype(np.uint8) * 255
                    _local_comp = {"area": _wf_best_area}
                    best_comp = _local_comp
                    print(
                        "[auto-mask-slot] "
                        f"crop_id={crop_id} slot_index={slot_index} "
                        f"function_path=window_fallback "
                        f"source_crop={source_path} slot_window={slot_box} "
                        f"component_box={extraction_box} generated_at={generated_at}"
                    )

            else:
                best_comp = max(candidates, key=lambda item: (item[0]["area"] * item[2], -item[1]))[0]
                best_comp["used"] = True
                extraction_reason = "global_mask_component"

                ex = best_comp["lx"]
                ey_val = best_comp["ly"]
                ew = best_comp["lw"]
                eh = best_comp["lh"]
                extraction_box = [ex, ey_val, ew, eh]
                slot_img = img[ey_val: ey_val + eh, ex: ex + ew]
                if best_comp.get("split_mask") is not None:
                    component_mask = best_comp["split_mask"][ey_val: ey_val + eh, ex: ex + ew]
                else:
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
            crop_origin=[int(extraction_box[0]), int(extraction_box[1])],
            qty_token_boxes=tokens,
        )
        if _slot5_debug and slot_index == 5:
            print(
                "[slot5-debug-write] "
                f"crop_id={crop_id} slot_index=5 ok={bool(artifact_result.get('ok'))} "
                f"shape_mask_path={artifact_result.get('shape_mask_path') or ''} "
                f"part_cutout_path={artifact_result.get('part_cutout_path') or ''} "
                f"error={artifact_result.get('error') or ''} "
                f"alpha_pixels={int(artifact_result.get('alpha_pixel_count', 0) or 0)} "
                f"size={artifact_result.get('actual_saved_cutout_size')} "
                f"component_box={extraction_box} "
                f"component_index={best_comp.get('debug_comp_index')} "
                f"component_area={best_comp.get('area')}"
            )
            if not bool(artifact_result.get("ok")):
                print(
                    "[slot5-debug-not-written] "
                    f"crop_id={crop_id} slot_index=5 "
                    f"reason={artifact_result.get('error') or 'write_not_ok'} "
                    f"shape_mask_path={artifact_result.get('shape_mask_path') or ''} "
                    f"part_cutout_path={artifact_result.get('part_cutout_path') or ''}"
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
                "slot_colour_confidence": str(artifact_result.get("colour_confidence") or "low"),
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

    if _slot_assignment_debug:
        for _debug_comp in global_comps:
            if _debug_comp.get("used") or id(_debug_comp) in _slot_assignment_debug_assigned_comp_ids:
                continue
            _nearest_slot, _nearest_tok, _nearest_dist = _nearest_qty_anchor_for_debug(_debug_comp)
            _anchor_text = (
                f"({int(_nearest_tok.get('cx', 0) or 0)},{int(_nearest_tok.get('y', 0) or 0)})"
                if _nearest_tok is not None
                else "None"
            )
            _nearest_score: Optional[float] = None
            _nearest_reject = "no_qty_anchor"
            if _nearest_tok is not None:
                _nearest_score, _nearest_reject = _anchor_proximity_debug_reason(_debug_comp, _nearest_tok, w, h)
            print(
                "[slot-assign-debug-unassigned] "
                f"crop_id={crop_id} pool=global "
                f"component_index={_debug_comp.get('debug_comp_index')} lab={_debug_comp.get('lab')} "
                f"bbox=[{_debug_comp['lx']},{_debug_comp['ly']},{_debug_comp['lw']},{_debug_comp['lh']}] "
                f"area={_debug_comp['area']} centroid=({_debug_comp['gcx']:.1f},{_debug_comp['gcy']:.1f}) "
                f"assigned_slot_index=None nearest_slot_index={_nearest_slot} "
                f"nearest_qty_anchor={_anchor_text} nearest_dist={_nearest_dist:.1f} "
                f"nearest_score={_nearest_score if _nearest_score is not None else 'None'} "
                f"rejection_reason={_nearest_reject if _nearest_reject != 'pass' else 'passed_but_not_selected'}"
            )
        for _debug_comp in _master_island_comps:
            if _debug_comp.get("used") or id(_debug_comp) in _slot_assignment_debug_assigned_comp_ids:
                continue
            _nearest_slot, _nearest_tok, _nearest_dist = _nearest_qty_anchor_for_debug(_debug_comp)
            _anchor_text = (
                f"({int(_nearest_tok.get('cx', 0) or 0)},{int(_nearest_tok.get('y', 0) or 0)})"
                if _nearest_tok is not None
                else "None"
            )
            _nearest_score: Optional[float] = None
            _nearest_reject = "no_qty_anchor"
            if _nearest_tok is not None:
                _nearest_score, _nearest_reject = _anchor_proximity_debug_reason(_debug_comp, _nearest_tok, w, h)
            print(
                "[slot-assign-debug-unassigned] "
                f"crop_id={crop_id} pool=master_island "
                f"component_index={_debug_comp.get('debug_comp_index')} lab={_debug_comp.get('lab')} "
                f"bbox=[{_debug_comp['lx']},{_debug_comp['ly']},{_debug_comp['lw']},{_debug_comp['lh']}] "
                f"area={_debug_comp['area']} centroid=({_debug_comp['gcx']:.1f},{_debug_comp['gcy']:.1f}) "
                f"assigned_slot_index=None nearest_slot_index={_nearest_slot} "
                f"nearest_qty_anchor={_anchor_text} nearest_dist={_nearest_dist:.1f} "
                f"nearest_score={_nearest_score if _nearest_score is not None else 'None'} "
                f"rejection_reason={_nearest_reject if _nearest_reject != 'pass' else 'fallback_not_reached_or_not_selected'}"
            )

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
