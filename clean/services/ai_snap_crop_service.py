from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHAPE_MASK_DIR = _REPO_ROOT / "debug" / "ai_training" / "shape_masks"
_PART_CUTOUT_DIR = _REPO_ROOT / "debug" / "ai_training" / "part_cutouts"


def _safe_slug(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    safe = "".join(ch for ch in text if ch.isalnum() or ch in "-_")
    return safe or fallback


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


def create_shape_mask_for_slot_crop(
    ai_snap_input_path: str,
    *,
    set_num: Optional[str] = None,
    bag: Optional[int] = None,
    crop_id: Optional[str] = None,
    slot_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Create debug-only shape mask/cutout images for an AI Snap input crop."""
    source_path = Path(str(ai_snap_input_path or "").strip())
    if not source_path.exists():
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "error": "ai_snap_input_path_missing",
        }

    img = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if img is None or getattr(img, "size", 0) == 0:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "error": "ai_snap_input_image_unreadable",
        }

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "error": "ai_snap_input_image_empty",
        }

    bg = _edge_background_bgr(img)
    if bg is None:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "error": "background_estimate_failed",
        }

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

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros((h, w), dtype=np.uint8)
    min_area = max(25.0, float(h * w) * 0.004)
    kept = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw <= 2 or ch <= 2:
            continue
        cv2.drawContours(cleaned, [contour], -1, 255, thickness=-1)
        kept += 1
        if kept >= 4:
            break

    if int(cleaned.sum()) <= 0:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": "",
            "error": "shape_mask_empty",
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

    ok_mask = cv2.imwrite(str(mask_path), cleaned)
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = cleaned
    ok_cutout = cv2.imwrite(str(cutout_path), bgra)
    if not ok_mask:
        return {
            "ok": False,
            "shape_mask_path": "",
            "part_cutout_path": str(cutout_path) if ok_cutout else "",
            "error": "shape_mask_write_failed",
        }
    return {
        "ok": True,
        "shape_mask_path": str(mask_path),
        "part_cutout_path": str(cutout_path) if ok_cutout else "",
        "error": "",
        "mask_component_count": kept,
    }
