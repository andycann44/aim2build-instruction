"""Shared helpers for full-crop mask artifacts (mask-review + bag review fallbacks)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_CROP_MASK_DIR = _REPO_ROOT / "debug" / "ai_training" / "full_crop_masks"
FULL_CROP_MASK_OVERLAY_DIR = _REPO_ROOT / "debug" / "ai_training" / "full_crop_mask_overlays"
ISLAND_SLOT_CUTOUT_DIR = _REPO_ROOT / "debug" / "ai_training" / "island_slot_cutouts"
ISLAND_SLOT_MASK_DIR = _REPO_ROOT / "debug" / "ai_training" / "island_slot_masks"

# Drop tiny CC specks when inferring fallback slots from raw_master_mask.
ISLAND_MIN_AREA = 400
ISLAND_MIN_AREA_FRACTION = 0.08
MAX_AUTO_ISLAND_SLOTS = 3


def find_full_mask_stem(set_num: str, bag: int, crop_id: str) -> Optional[str]:
    """Return mask file stem (no suffix) for the latest full_crop_mask for this crop."""
    safe_set = "".join(ch for ch in str(set_num) if ch.isalnum() or ch in "-_")
    safe_crop = "".join(ch for ch in str(crop_id) if ch.isalnum() or ch in "-_")
    pattern = f"{safe_set}_bag{int(bag)}_{safe_crop}_*_full_mask.png"
    matches = sorted(FULL_CROP_MASK_DIR.glob(pattern))
    if not matches:
        return None
    latest = max(matches, key=lambda path: path.stat().st_mtime)
    return latest.stem.removesuffix("_full_mask")


def raw_master_mask_path(stem: str) -> Path:
    return FULL_CROP_MASK_DIR / f"{stem}_raw_master_mask.png"


def master_island_overlay_path(stem: str) -> Path:
    return FULL_CROP_MASK_OVERLAY_DIR / f"{stem}_master_island_overlay.png"


def master_islands_from_mask(mask_path: str) -> List[Dict[str, Any]]:
    mask_file = Path(str(mask_path or "").strip())
    if not mask_file.is_file():
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
                "centroid": [
                    round(float(centroids[label][0]), 2),
                    round(float(centroids[label][1]), 2),
                ],
            }
        )
    return islands


def sort_islands_for_slots(islands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order islands left-to-right by bbox center x."""
    return sorted(
        list(islands or []),
        key=lambda item: (
            (float((item.get("bbox") or [0, 0, 0, 0])[0]) + float((item.get("bbox") or [0, 0, 0, 0])[2]))
            / 2.0
        ),
    )


def island_slot_cutout_cache_path(stem: str, slot_index: int, island_label: int) -> Path:
    return ISLAND_SLOT_CUTOUT_DIR / f"{stem}_slot{int(slot_index)}_island{int(island_label)}.png"


def island_slot_mask_cache_path(stem: str, slot_index: int, island_label: int) -> Path:
    return ISLAND_SLOT_MASK_DIR / f"{stem}_slot{int(slot_index)}_island{int(island_label)}_mask.png"


def filter_significant_islands(
    islands: List[Dict[str, Any]],
    *,
    min_area: int = ISLAND_MIN_AREA,
    min_fraction: float = ISLAND_MIN_AREA_FRACTION,
) -> List[Dict[str, Any]]:
    """Keep CC islands large enough to be visible parts; drop OCR/edge specks."""
    ordered = sort_islands_for_slots(list(islands or []))
    if not ordered:
        return []
    max_area = max(int(item.get("area") or 0) for item in ordered)
    threshold = max(int(min_area), int(round(float(min_fraction) * float(max_area))))
    return [item for item in ordered if int(item.get("area") or 0) >= threshold]


def _islands_from_label_map(
    labels_count: int,
    stats: Any,
    centroids: Any,
) -> List[Dict[str, Any]]:
    islands: List[Dict[str, Any]] = []
    for label in range(1, int(labels_count)):
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
                "centroid": [
                    round(float(centroids[label][0]), 2),
                    round(float(centroids[label][1]), 2),
                ],
            }
        )
    return islands


def ensure_island_slot_cutout_path(
    *,
    stem: str,
    slot_index: int,
    crop_image_path: str,
    crop_box: List[int],
    island_label: Optional[int] = None,
) -> Optional[Path]:
    """
    Build (or return cached) RGBA cutout for one master-mask island mapped to slot_index.

    Uses raw_master_mask connected-component label geometry only — not
    master_island_overlay visualization boxes. Alpha is restricted to the single
    island label so overlapping CC bboxes cannot merge neighboring parts.
    """
    raw_path = raw_master_mask_path(stem)
    if not raw_path.is_file():
        return None

    mask = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or getattr(mask, "size", 0) == 0:
        return None

    labels_count, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        8,
    )
    if island_label is None:
        return None
    resolved_label = int(island_label)
    label_ids = {
        int(isl["label"])
        for isl in _islands_from_label_map(labels_count, stats, centroids)
    }
    if resolved_label not in label_ids:
        return None

    cache_path = island_slot_cutout_cache_path(stem, slot_index, resolved_label)
    if cache_path.is_file():
        return cache_path

    if len(crop_box) < 4:
        return None
    cx, cy, cw, ch = [int(v) for v in crop_box[:4]]
    if cw <= 0 or ch <= 0:
        return None

    page = cv2.imread(str(crop_image_path))
    if page is None or getattr(page, "size", 0) == 0:
        return None
    ih, iw = page.shape[:2]
    cx = max(0, min(cx, iw - 1))
    cy = max(0, min(cy, ih - 1))
    cw = min(cw, iw - cx)
    ch = min(ch, ih - cy)
    if cw <= 0 or ch <= 0:
        return None
    callout = page[cy : cy + ch, cx : cx + cw]
    if callout is None or callout.size == 0:
        return None
    if mask.shape[:2] != callout.shape[:2]:
        return None

    island_mask = ((label_map == resolved_label).astype(np.uint8)) * 255
    ys, xs = np.where(island_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    pad = 4
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(cw, int(xs.max()) + 1 + pad)
    y2 = min(ch, int(ys.max()) + 1 + pad)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = callout[y1:y2, x1:x2].copy()
    roi_mask = island_mask[y1:y2, x1:x2]
    bgra = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = ((roi_mask > 0).astype(np.uint8)) * 255
    bgra[bgra[:, :, 3] == 0, :3] = 0

    ISLAND_SLOT_CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(cache_path), bgra):
        return cache_path
    return None


def ensure_island_slot_mask_path(
    *,
    stem: str,
    slot_index: int,
    crop_image_path: str,
    crop_box: List[int],
    island_label: Optional[int] = None,
) -> Optional[Path]:
    """Build (or return cached) RGB slot-mask preview for one island (step-seg style)."""
    raw_path = raw_master_mask_path(stem)
    if not raw_path.is_file():
        return None

    mask = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or getattr(mask, "size", 0) == 0:
        return None

    labels_count, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        8,
    )
    if island_label is None:
        return None
    resolved_label = int(island_label)
    label_ids = {
        int(isl["label"])
        for isl in _islands_from_label_map(labels_count, stats, centroids)
    }
    if resolved_label not in label_ids:
        return None

    cache_path = island_slot_mask_cache_path(stem, slot_index, resolved_label)
    if cache_path.is_file():
        return cache_path

    cutout_path = ensure_island_slot_cutout_path(
        stem=stem,
        slot_index=int(slot_index),
        crop_image_path=crop_image_path,
        crop_box=crop_box,
        island_label=resolved_label,
    )
    if cutout_path is None or not cutout_path.is_file():
        return None

    bgra = cv2.imread(str(cutout_path), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.size == 0:
        return None
    if bgra.ndim == 2:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGRA)
    elif bgra.shape[2] == 3:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)

    alpha = bgra[:, :, 3] if bgra.shape[2] == 4 else np.full(bgra.shape[:2], 255, dtype=np.uint8)
    preview = np.full((bgra.shape[0], bgra.shape[1], 3), (251, 247, 244), dtype=np.uint8)
    fg = alpha > 0
    preview[fg] = bgra[fg, :3]

    ISLAND_SLOT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(cache_path), preview):
        return cache_path
    return None
