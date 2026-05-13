import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


def _encode_png(path: Path, image: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")
    return str(path)


def _error_result(original_path: Path, message: str, **extra: Any) -> Dict[str, Any]:
    payload = {
        "ok": False,
        "original_path": str(original_path),
        "normalized_path": "",
        "mask_path": "",
        "box": None,
        "debug": {"error": message},
    }
    payload.update(extra)
    return payload


def _load_bgr_image(image_path: str) -> Optional[Any]:
    original_path = Path(str(image_path or "").strip())
    if not original_path.exists() or not original_path.is_file():
        return None
    img = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if img is None or getattr(img, "size", 0) == 0:
        return None
    return img


def _resolve_output_dir(output_dir: Optional[str], prefix: str) -> Path:
    if output_dir:
        out_dir = Path(str(output_dir)).expanduser()
    else:
        out_dir = Path(tempfile.mkdtemp(prefix=prefix))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _foreground_mask_from_instruction_crop(img: Any) -> Dict[str, Any]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bgr = img

    pale_blue_mask = (
        (hsv[:, :, 0] >= 80)
        & (hsv[:, :, 0] <= 130)
        & (hsv[:, :, 1] <= 150)
        & (hsv[:, :, 2] >= 135)
    )
    extra_pale_blue_mask = (
        (bgr[:, :, 0] >= 150)
        & (bgr[:, :, 1] >= 150)
        & (bgr[:, :, 2] >= 130)
        & ((bgr[:, :, 0].astype(np.int16) - bgr[:, :, 2].astype(np.int16)) >= 8)
    )
    bg_mask = pale_blue_mask | extra_pale_blue_mask
    fg_mask = (~bg_mask).astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return {
        "mask": fg_mask,
        "bg_mask": bg_mask.astype(np.uint8) * 255,
    }


def _component_stats(mask: Any) -> Dict[str, Any]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = int(mask.shape[0] * mask.shape[1])
    min_component_area = max(40, int(image_area * 0.0025))
    kept_components = []
    component_mask = np.zeros_like(mask)
    for label in range(1, int(num_labels)):
        area = int(stats[label, cv2.CC_STAT_AREA] or 0)
        if area < min_component_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        kept_components.append(
            {
                "label": int(label),
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": float(cx),
                "cy": float(cy),
            }
        )
        component_mask[labels == label] = 255
    if int(component_mask.max()) == 0:
        component_mask = mask.copy()
    return {
        "labels": labels,
        "component_mask": component_mask,
        "kept_components": kept_components,
        "min_component_area": min_component_area,
    }


def _tight_box_from_mask(mask: Any) -> Optional[list[int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _padded_box(box: list[int], image_w: int, image_h: int, pad_ratio: float = 0.12, min_pad: int = 8) -> list[int]:
    x, y, w, h = [int(value) for value in box]
    pad = max(int(min_pad), int(round(max(w, h) * float(pad_ratio))))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(int(image_w), x + w + pad)
    y2 = min(int(image_h), y + h + pad)
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def _crop_and_square(img: Any, mask: Any, box: list[int]) -> Dict[str, Any]:
    x, y, w, h = [int(value) for value in box]
    crop_img = img[y : y + h, x : x + w]
    crop_mask = mask[y : y + h, x : x + w]
    masked_crop = np.full_like(crop_img, 255)
    masked_crop[crop_mask > 0] = crop_img[crop_mask > 0]

    side = max(masked_crop.shape[0], masked_crop.shape[1])
    square_side = max(96, int(side))
    square_canvas = np.full((square_side, square_side, 3), 255, dtype=np.uint8)
    square_mask = np.zeros((square_side, square_side), dtype=np.uint8)
    offset_y = max(0, (square_side - masked_crop.shape[0]) // 2)
    offset_x = max(0, (square_side - masked_crop.shape[1]) // 2)
    square_canvas[offset_y : offset_y + masked_crop.shape[0], offset_x : offset_x + masked_crop.shape[1]] = masked_crop
    square_mask[offset_y : offset_y + crop_mask.shape[0], offset_x : offset_x + crop_mask.shape[1]] = crop_mask
    return {
        "masked_crop": masked_crop,
        "square_canvas": square_canvas,
        "square_mask": square_mask,
        "square_side": square_side,
    }


def normalize_part_crop(image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    original_path = Path(str(image_path or "").strip())
    img = _load_bgr_image(image_path)
    if img is None:
        return _error_result(original_path, "image_path not found or could not load image")

    out_dir = _resolve_output_dir(output_dir, "part_crop_normalize_")
    mask_info = _foreground_mask_from_instruction_crop(img)
    fg_mask = mask_info["mask"]
    component_info = _component_stats(fg_mask)
    component_mask = component_info["component_mask"]
    kept_components = component_info["kept_components"]

    tight_box = _tight_box_from_mask(component_mask)
    if tight_box is None:
        tight_box = [0, 0, img.shape[1], img.shape[0]]
    padded_box = _padded_box(tight_box, img.shape[1], img.shape[0])
    square_info = _crop_and_square(img, component_mask, padded_box)

    stem = original_path.stem or "crop"
    normalized_path = out_dir / f"{stem}_normalized.png"
    mask_path = out_dir / f"{stem}_mask.png"
    _encode_png(normalized_path, square_info["square_canvas"])
    _encode_png(mask_path, square_info["square_mask"])

    return {
        "ok": True,
        "original_path": str(original_path),
        "normalized_path": str(normalized_path),
        "mask_path": str(mask_path),
        "box": [int(value) for value in padded_box],
        "debug": {
            "original_shape": [int(img.shape[1]), int(img.shape[0])],
            "tight_shape": [int(square_info["masked_crop"].shape[1]), int(square_info["masked_crop"].shape[0])],
            "square_side": int(square_info["square_side"]),
            "foreground_pixels": int(np.count_nonzero(component_mask)),
            "kept_component_count": len(kept_components),
            "kept_components": kept_components,
            "min_component_area": int(component_info["min_component_area"]),
            "output_dir": str(out_dir),
        },
    }


def normalize_slot_crop_from_qty(
    image_path: str,
    qty_box: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    original_path = Path(str(image_path or "").strip())
    img = _load_bgr_image(image_path)
    if img is None:
        return {
            "ok": False,
            "original_path": str(original_path),
            "mask_path": "",
            "component_path": "",
            "normalized_path": "",
            "selected_box": None,
            "qty_box": None,
            "debug": {"error": "image_path not found or could not load image"},
        }

    qx = qty_box.get("x")
    qy = qty_box.get("y")
    qw = qty_box.get("w")
    qh = qty_box.get("h")
    try:
        qx_i = int(qx)
        qy_i = int(qy)
        qw_i = int(qw)
        qh_i = int(qh)
    except (TypeError, ValueError):
        return {
            "ok": False,
            "original_path": str(original_path),
            "mask_path": "",
            "component_path": "",
            "normalized_path": "",
            "selected_box": None,
            "qty_box": None,
            "debug": {"error": "invalid qty_box"},
        }
    if qw_i <= 0 or qh_i <= 0:
        return {
            "ok": False,
            "original_path": str(original_path),
            "mask_path": "",
            "component_path": "",
            "normalized_path": "",
            "selected_box": None,
            "qty_box": None,
            "debug": {"error": "qty_box must have positive size"},
        }

    out_dir = _resolve_output_dir(output_dir, "slot_crop_normalize_")
    mask_info = _foreground_mask_from_instruction_crop(img)
    fg_mask = mask_info["mask"]
    qty_pad = max(4, int(round(max(qw_i, qh_i) * 0.18)))
    mask_x1 = max(0, qx_i - qty_pad)
    mask_y1 = max(0, qy_i - qty_pad)
    mask_x2 = min(img.shape[1], qx_i + qw_i + qty_pad)
    mask_y2 = min(img.shape[0], qy_i + qh_i + qty_pad)
    fg_mask[mask_y1:mask_y2, mask_x1:mask_x2] = 0

    component_info = _component_stats(fg_mask)
    labels = component_info["labels"]
    kept_components = component_info["kept_components"]
    component_mask = component_info["component_mask"]
    qty_cx = qx_i + (qw_i / 2.0)
    qty_cy = qy_i + (qh_i / 2.0)

    selected_component = None
    best_score = None
    scored_components = []
    for component in kept_components:
        comp_cx = float(component["cx"])
        comp_cy = float(component["cy"])
        vertical_priority = 0 if comp_cy <= qty_cy else 1
        dx = abs(comp_cx - qty_cx)
        dy = abs(comp_cy - qty_cy)
        distance_score = (dy * 2.2) + dx
        score = (vertical_priority * 1_000_000.0) + distance_score - (float(component["area"]) * 0.02)
        scored_components.append(
            {
                "label": int(component["label"]),
                "score": round(score, 3),
                "vertical_priority": vertical_priority,
                "distance_score": round(distance_score, 3),
                "area": int(component["area"]),
                "x": int(component["x"]),
                "y": int(component["y"]),
                "w": int(component["w"]),
                "h": int(component["h"]),
                "cx": round(comp_cx, 2),
                "cy": round(comp_cy, 2),
            }
        )
        if best_score is None or score < best_score:
            best_score = score
            selected_component = component

    if selected_component is None:
        return {
            "ok": False,
            "original_path": str(original_path),
            "mask_path": "",
            "component_path": "",
            "normalized_path": "",
            "selected_box": None,
            "qty_box": {"x": qx_i, "y": qy_i, "w": qw_i, "h": qh_i},
            "debug": {
                "error": "no component selected",
                "kept_component_count": len(kept_components),
            },
        }

    selected_box = [
        int(selected_component["x"]),
        int(selected_component["y"]),
        int(selected_component["w"]),
        int(selected_component["h"]),
    ]
    selected_mask = np.zeros_like(component_mask)
    selected_mask[labels == int(selected_component["label"])] = 255
    padded_box = _padded_box(selected_box, img.shape[1], img.shape[0], pad_ratio=0.16, min_pad=10)
    square_info = _crop_and_square(img, selected_mask, padded_box)

    x, y, w, h = [int(value) for value in padded_box]
    component_crop = square_info["masked_crop"]
    stem = original_path.stem or "crop"
    normalized_path = out_dir / f"{stem}_slot_normalized.png"
    mask_path = out_dir / f"{stem}_slot_mask.png"
    component_path = out_dir / f"{stem}_slot_component.png"
    _encode_png(normalized_path, square_info["square_canvas"])
    _encode_png(mask_path, selected_mask)
    _encode_png(component_path, component_crop)

    return {
        "ok": True,
        "original_path": str(original_path),
        "mask_path": str(mask_path),
        "component_path": str(component_path),
        "normalized_path": str(normalized_path),
        "selected_box": selected_box,
        "qty_box": {"x": qx_i, "y": qy_i, "w": qw_i, "h": qh_i},
        "debug": {
            "original_shape": [int(img.shape[1]), int(img.shape[0])],
            "padded_box": padded_box,
            "qty_mask_box": [int(mask_x1), int(mask_y1), int(mask_x2 - mask_x1), int(mask_y2 - mask_y1)],
            "foreground_pixels": int(np.count_nonzero(selected_mask)),
            "square_side": int(square_info["square_side"]),
            "kept_component_count": len(kept_components),
            "selected_component": {
                "label": int(selected_component["label"]),
                "area": int(selected_component["area"]),
                "x": int(selected_component["x"]),
                "y": int(selected_component["y"]),
                "w": int(selected_component["w"]),
                "h": int(selected_component["h"]),
                "cx": round(float(selected_component["cx"]), 2),
                "cy": round(float(selected_component["cy"]), 2),
            },
            "component_scores": scored_components,
            "output_dir": str(out_dir),
        },
    }


__all__ = ["normalize_part_crop", "normalize_slot_crop_from_qty"]
