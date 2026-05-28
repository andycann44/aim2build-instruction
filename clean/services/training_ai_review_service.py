from __future__ import annotations

import base64
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from clean.services.training_bundle_index_service import (
    get_bundle,
    update_ai_analysis,
    update_split_candidate_status,
    update_split_candidates,
)
from clean.services.training_store_service import _REPO_ROOT


_AZURE_REQUIRED_KEYS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"]


def _read_local_env_file() -> Dict[str, str]:
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists() or not env_path.is_file():
        return {}
    values: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _config_value(key: str, local_env: Dict[str, str]) -> str:
    return str(os.environ.get(key) or local_env.get(key) or "").strip()


def _analysis_client_and_model() -> Dict[str, Any]:
    local_env = _read_local_env_file()
    azure_values = {key: _config_value(key, local_env) for key in _AZURE_REQUIRED_KEYS}
    if all(azure_values.values()):
        from openai import AzureOpenAI

        return {
            "client": AzureOpenAI(
                azure_endpoint=azure_values["AZURE_OPENAI_ENDPOINT"],
                api_key=azure_values["AZURE_OPENAI_API_KEY"],
                api_version=_config_value("AZURE_OPENAI_API_VERSION", local_env) or "2025-03-01-preview",
            ),
            "model": azure_values["AZURE_OPENAI_DEPLOYMENT"],
            "provider": "azure_openai",
        }

    openai_key = _config_value("OPENAI_API_KEY", local_env)
    if openai_key:
        from openai import OpenAI

        return {
            "client": OpenAI(api_key=openai_key),
            "model": _config_value("OPENAI_VISION_MODEL", local_env) or "gpt-4.1",
            "provider": "openai",
        }
    raise ValueError("Azure/OpenAI vision config is not available")


def _mime_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _image_data_url(path_value: Any) -> str:
    path = Path(str(path_value or "").strip())
    if not path.exists() or not path.is_file():
        return ""
    raw = path.read_bytes()
    return "data:%s;base64,%s" % (
        _mime_type_for_path(path),
        base64.b64encode(raw).decode("ascii"),
    )


def _metadata_path_for_row(row: Dict[str, Any]) -> Path:
    manifest_path = Path(str(row.get("manifest_path") or "").strip())
    if manifest_path.exists() and manifest_path.is_file():
        return manifest_path
    bundle_id = str(row.get("bundle_id") or "").strip()
    return _REPO_ROOT / "debug" / "ai_training" / "analysis_bundles" / bundle_id / "metadata.json"


def _read_bundle_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    path = _metadata_path_for_row(row)
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_box(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    out: List[float] = []
    for item in list(value)[:4]:
        try:
            out.append(float(item))
        except Exception:
            return []
    return out


def _clip_box(box: List[float], width: int, height: int) -> List[int]:
    if len(box) != 4:
        return []
    x, y, w, h = box
    if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
        x *= width
        w *= width
        y *= height
        h *= height
    x1 = max(0, min(int(width), int(x)))
    y1 = max(0, min(int(height), int(y)))
    x2 = max(0, min(int(width), int(x + max(0, w))))
    y2 = max(0, min(int(height), int(y + max(0, h))))
    if x2 <= x1 or y2 <= y1:
        return []
    return [x1, y1, x2 - x1, y2 - y1]


def _analysis_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("ai_analysis_json")
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str) and payload.strip():
        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _boxes_intersect(a: List[int], b: List[int]) -> bool:
    if len(a) != 4 or len(b) != 4:
        return False
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


def _box_intersection_area(a: List[int], b: List[int]) -> int:
    if len(a) != 4 or len(b) != 4:
        return 0
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _point_in_box(px: float, py: float, box: List[int]) -> bool:
    if len(box) != 4:
        return False
    x, y, w, h = box
    return float(x) <= px <= float(x + w) and float(y) <= py <= float(y + h)


def _box_iou(a: List[int], b: List[int]) -> float:
    if not _boxes_intersect(a, b):
        return 0.0
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = max(1, aw * ah + bw * bh - inter)
    return float(inter) / float(union)


def _scrub_qty_boxes_from_bgra(
    bgra: np.ndarray,
    crop_box: List[int],
    qty_token_boxes: List[Dict[str, Any]],
    pad: int = 2,
) -> None:
    if bgra is None or len(crop_box) != 4:
        return
    crop_x, crop_y, crop_w, crop_h = crop_box
    for token in qty_token_boxes:
        token_box = _clip_box(
            [
                float(token.get("x", 0) or 0) - pad,
                float(token.get("y", 0) or 0) - pad,
                float(token.get("w", 0) or 0) + pad * 2,
                float(token.get("h", 0) or 0) + pad * 2,
            ],
            crop_x + crop_w,
            crop_y + crop_h,
        )
        if not token_box or not _boxes_intersect(crop_box, token_box):
            continue
        tx, ty, tw, th = token_box
        ix1 = max(crop_x, tx)
        iy1 = max(crop_y, ty)
        ix2 = min(crop_x + crop_w, tx + tw)
        iy2 = min(crop_y + crop_h, ty + th)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        lx1, ly1 = ix1 - crop_x, iy1 - crop_y
        lx2, ly2 = ix2 - crop_x, iy2 - crop_y
        bgra[ly1:ly2, lx1:lx2, :] = 0


def _candidate_qty_tokens(
    crop_box: List[int],
    qty_token_boxes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    if len(crop_box) != 4:
        return matched
    for token in qty_token_boxes:
        token_box = _clip_box(
            [
                float(token.get("x", 0) or 0),
                float(token.get("y", 0) or 0),
                float(token.get("w", 0) or 0),
                float(token.get("h", 0) or 0),
            ],
            crop_box[0] + crop_box[2],
            crop_box[1] + crop_box[3],
        )
        if token_box and _boxes_intersect(crop_box, token_box):
            matched.append(dict(token))
    return matched


def _qty_values_from_tokens(tokens: List[Dict[str, Any]]) -> List[int]:
    values: List[int] = []
    for token in tokens:
        text = str(token.get("text") or token.get("value") or "").strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            continue
        try:
            value = int(digits)
        except Exception:
            continue
        if value not in values:
            values.append(value)
    return values


def _blue_callout_background_mask(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return np.zeros((0, 0), dtype=bool)
    bgr = image_bgr.astype(np.int16)
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]
    return (
        (b >= 135)
        & (g >= 115)
        & (b >= r + 16)
        & (g >= r + 4)
    )


def _expanded_export_alpha(original_bgr: np.ndarray, component_mask: np.ndarray) -> np.ndarray:
    base = (component_mask > 0).astype(np.uint8)
    if base is None or getattr(base, "size", 0) == 0 or int(np.count_nonzero(base)) == 0:
        return np.zeros_like(base, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    background = _blue_callout_background_mask(original_bgr)
    if background.shape != dilated.shape:
        background = np.zeros_like(dilated, dtype=bool)
    expanded = ((dilated > 0) & ~background) | (base > 0)
    return expanded.astype(np.uint8) * 255


def _token_center(token: Dict[str, Any]) -> List[float]:
    try:
        x = float(token.get("x", 0) or 0)
        y = float(token.get("y", 0) or 0)
        w = float(token.get("w", 0) or 0)
        h = float(token.get("h", 0) or 0)
    except Exception:
        return [0.0, 0.0]
    return [x + (w / 2.0), y + (h / 2.0)]


def _box_center(box: List[int]) -> List[float]:
    if len(box) != 4:
        return [0.0, 0.0]
    x, y, w, h = box
    return [float(x) + (float(w) / 2.0), float(y) + (float(h) / 2.0)]


def _distance_from_point_to_box(px: float, py: float, box: List[int]) -> float:
    if len(box) != 4:
        return 999999.0
    x, y, w, h = box
    x2, y2 = x + w, y + h
    dx = 0.0
    if px < x:
        dx = float(x) - px
    elif px > x2:
        dx = px - float(x2)
    dy = 0.0
    if py < y:
        dy = float(y) - py
    elif py > y2:
        dy = py - float(y2)
    return float((dx * dx + dy * dy) ** 0.5)


def _assign_qty_tokens_to_components(
    components: List[Dict[str, Any]],
    qty_token_boxes: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    assignments: Dict[int, Dict[str, Any]] = {}
    used_labels = set()
    ordered_tokens = sorted(
        [
            dict(token, _token_order=index)
            for index, token in enumerate(list(qty_token_boxes or []))
            if isinstance(token, dict)
        ],
        key=lambda token: (
            float(token.get("y", 0) or 0),
            float(token.get("x", 0) or 0),
        ),
    )
    for token_order, token in enumerate(ordered_tokens):
        tcx, tcy = _token_center(token)
        scored: List[Dict[str, Any]] = []
        for component in components:
            label = int(component.get("component_label", -1))
            if label in used_labels:
                continue
            box = list(component.get("box") or [])
            if len(box) != 4:
                continue
            ccx, ccy = _box_center(box)
            x, y, w, h = box
            above_or_near = ccy <= tcy or y <= tcy + max(12.0, float(h) * 0.25)
            if not above_or_near:
                continue
            point_distance = _distance_from_point_to_box(tcx, tcy, box)
            center_distance = float(((ccx - tcx) ** 2 + (ccy - tcy) ** 2) ** 0.5)
            vertical_gap = max(0.0, float(y) - tcy)
            below_penalty = 0.0 if ccy <= tcy else 80.0
            score = point_distance + (center_distance * 0.18) + (vertical_gap * 1.5) + below_penalty
            scored.append(
                {
                    "component_label": label,
                    "score": score,
                    "point_distance": point_distance,
                    "center_distance": center_distance,
                    "component_center": [ccx, ccy],
                    "component_box": box,
                    "token_center": [tcx, tcy],
                }
            )
        if not scored:
            continue
        best = sorted(scored, key=lambda item: item["score"])[0]
        label = int(best["component_label"])
        used_labels.add(label)
        token_copy = dict(token)
        token_copy["anchor_order"] = token_order
        assignments[label] = {
            "qty_token": token_copy,
            "qty_values": _qty_values_from_tokens([token_copy]),
            "anchor_order": token_order,
            "assignment_score": best,
            "candidate_scores": scored,
        }
    return assignments


def _normalised_qty_tokens_from_metadata(row: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    tokens = [
        dict(item)
        for item in list(metadata.get("direct_qty_ocr_boxes") or metadata.get("qty_token_boxes") or [])
        if isinstance(item, dict)
    ]
    if not tokens:
        return []
    copied_files = metadata.get("copied_files") if isinstance(metadata.get("copied_files"), dict) else {}
    original_path = Path(str(copied_files.get("original_crop") or "").strip())
    crop_width = crop_height = 0
    if original_path.exists() and original_path.is_file():
        img = cv2.imread(str(original_path), cv2.IMREAD_UNCHANGED)
        if img is not None and getattr(img, "size", 0) != 0:
            crop_height, crop_width = img.shape[:2]
    source_crop = metadata.get("source_crop") if isinstance(metadata.get("source_crop"), dict) else {}
    crop_box = source_crop.get("crop_box") if isinstance(source_crop.get("crop_box"), list) else [0, 0, crop_width, crop_height]
    try:
        crop_x, crop_y, crop_w, crop_h = [int(float(value)) for value in list(crop_box)[:4]]
    except Exception:
        crop_x, crop_y, crop_w, crop_h = 0, 0, crop_width, crop_height
    if crop_width <= 0:
        crop_width = crop_w
    if crop_height <= 0:
        crop_height = crop_h

    def clip(x: float, y: float, w: float, h: float) -> List[int]:
        x1 = max(0, min(crop_width, int(round(x))))
        y1 = max(0, min(crop_height, int(round(y))))
        x2 = max(0, min(crop_width, int(round(x + max(0, w)))))
        y2 = max(0, min(crop_height, int(round(y + max(0, h)))))
        if x2 <= x1 or y2 <= y1:
            return []
        return [x1, y1, x2 - x1, y2 - y1]

    out: List[Dict[str, Any]] = []
    for token in tokens:
        if isinstance(token.get("normalized_box"), list) and len(token.get("normalized_box") or []) >= 4:
            box = [int(float(value)) for value in list(token.get("normalized_box") or [])[:4]]
            item = dict(token)
            item["x"], item["y"], item["w"], item["h"] = box
            out.append(item)
            continue
        try:
            x = float(token.get("x", 0) or 0)
            y = float(token.get("y", 0) or 0)
            w = float(token.get("w", 0) or 0)
            h = float(token.get("h", 0) or 0)
        except Exception:
            continue
        source = "crop_local_xywh"
        box = clip(x, y, w, h)
        if not box and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
            source = "crop_local_normalized_xywh"
            box = clip(x * crop_width, y * crop_height, w * crop_width, h * crop_height)
        if not box and x >= crop_x and y >= crop_y:
            source = "page_xywh_minus_crop_origin"
            box = clip(x - crop_x, y - crop_y, w, h)
        if not box:
            continue
        item = dict(token)
        item["raw_box"] = [token.get("x"), token.get("y"), token.get("w"), token.get("h")]
        item["normalized_box"] = box
        item["coordinate_source"] = source
        item["x"], item["y"], item["w"], item["h"] = box
        out.append(item)
    return out


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    try:
        payload = response.model_dump()
    except Exception:
        payload = {}
    texts: List[str] = []
    for item in list((payload or {}).get("output") or []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in list(item.get("content") or []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                texts.append(str(content.get("text") or "").strip())
    return "\n".join(text for text in texts if text)


def _normalise_analysis_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    confidence_scores = payload.get("confidence_scores") if isinstance(payload.get("confidence_scores"), dict) else {}
    return {
        "ai_summary": str(payload.get("ai_summary") or ""),
        "segmentation_issues": list(payload.get("segmentation_issues") or []),
        "suggested_split_regions": list(payload.get("suggested_split_regions") or []),
        "qty_text_regions": list(payload.get("qty_text_regions") or []),
        "cleanup_recommendations": list(payload.get("cleanup_recommendations") or []),
        "confidence_scores": {
            "clean_single_part_extraction": float(confidence_scores.get("clean_single_part_extraction", 0) or 0),
            "mask_quality": float(confidence_scores.get("mask_quality", 0) or 0),
            "split_quality": float(confidence_scores.get("split_quality", 0) or 0),
            "qty_text_detection": float(confidence_scores.get("qty_text_detection", 0) or 0),
        },
    }


def _analysis_schema() -> Dict[str, Any]:
    region = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string"},
            "reason": {"type": "string"},
            "box": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
            "confidence": {"type": "number"},
        },
        "required": ["label", "reason", "box", "confidence"],
    }
    recommendation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target": {"type": "string"},
            "recommendation": {"type": "string"},
            "box": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
            "confidence": {"type": "number"},
        },
        "required": ["target", "recommendation", "box", "confidence"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ai_summary": {"type": "string"},
            "segmentation_issues": {"type": "array", "items": region},
            "suggested_split_regions": {"type": "array", "items": region},
            "qty_text_regions": {"type": "array", "items": region},
            "cleanup_recommendations": {"type": "array", "items": recommendation},
            "confidence_scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "clean_single_part_extraction": {"type": "number"},
                    "mask_quality": {"type": "number"},
                    "split_quality": {"type": "number"},
                    "qty_text_detection": {"type": "number"},
                },
                "required": [
                    "clean_single_part_extraction",
                    "mask_quality",
                    "split_quality",
                    "qty_text_detection",
                ],
            },
        },
        "required": [
            "ai_summary",
            "segmentation_issues",
            "suggested_split_regions",
            "qty_text_regions",
            "cleanup_recommendations",
            "confidence_scores",
        ],
    }


def analyse_reviewed_bundle(bundle_id: str) -> Dict[str, Any]:
    row = dict(get_bundle(bundle_id).get("row") or {})
    status = str(row.get("review_status") or "pending")
    if status == "pending":
        raise ValueError("bundle must be manually reviewed before AI analysis")

    metadata = _read_bundle_metadata(row)
    copied_files = metadata.get("copied_files") if isinstance(metadata.get("copied_files"), dict) else {}
    slot_cutouts = list(copied_files.get("slot_cutouts") or metadata.get("cutout_paths") or [])
    image_refs = [
        ("original_crop", copied_files.get("original_crop")),
        ("full_mask_overlay", copied_files.get("full_mask_overlay")),
        ("raw_master_mask", copied_files.get("raw_master_mask")),
    ]
    for index, path_value in enumerate(slot_cutouts):
        image_refs.append((f"slot_cutout_{index}", path_value))

    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "You are reviewing a LEGO instruction training bundle. Return strict JSON only. "
                "Analyse why segmentation may have failed, where masks should split, where qty text exists, "
                "what background should be removed, suggested bounding boxes/islands, and confidence scores. "
                "All boxes must be [x, y, w, h] in the coordinate space of the most relevant provided image. "
                "Do not suggest applying fixes automatically."
            ),
        },
        {
            "type": "input_text",
            "text": json.dumps(
                {
                    "bundle_index_row": row,
                    "review_metadata": {
                        "review_status": row.get("review_status"),
                        "review_notes": row.get("review_notes"),
                        "mask_quality": row.get("mask_quality"),
                        "split_quality": row.get("split_quality"),
                        "qty_text_present": row.get("qty_text_present"),
                        "multi_part_merge": row.get("multi_part_merge"),
                        "reviewed_by": row.get("reviewed_by"),
                    },
                    "bundle_metadata": {
                        "set_num": metadata.get("set_num"),
                        "bag": metadata.get("bag"),
                        "crop_id": metadata.get("crop_id"),
                        "source_crop": metadata.get("source_crop"),
                        "qty_token_boxes": metadata.get("qty_token_boxes"),
                        "master_islands": metadata.get("master_islands"),
                        "slot_assignments": metadata.get("slot_assignments"),
                    },
                    "image_order": [name for name, _path in image_refs],
                },
                ensure_ascii=True,
            ),
        },
    ]
    included_images: List[str] = []
    for name, path_value in image_refs:
        image_url = _image_data_url(path_value)
        if not image_url:
            continue
        content.append({"type": "input_text", "text": f"Image: {name}"})
        content.append({"type": "input_image", "image_url": image_url, "detail": "high"})
        included_images.append(name)

    if not included_images:
        raise ValueError("bundle has no readable images for AI analysis")

    client_info = _analysis_client_and_model()
    response = client_info["client"].responses.create(
        model=client_info["model"],
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "training_bundle_segmentation_review",
                "strict": True,
                "schema": _analysis_schema(),
            }
        },
    )
    raw_text = _response_text(response)
    try:
        payload = json.loads(raw_text)
    except Exception:
        payload = {}
    analysis = _normalise_analysis_payload(payload)
    analysis["included_images"] = included_images
    analysis["bundle_id"] = str(row.get("bundle_id") or bundle_id)
    analysis["review_status"] = status

    stored = update_ai_analysis(
        str(row.get("bundle_id") or bundle_id),
        ai_analysis_json=analysis,
        ai_model=str(client_info.get("model") or ""),
    )
    return {
        "ok": True,
        "bundle_id": str(row.get("bundle_id") or bundle_id),
        "ai_model": str(client_info.get("model") or ""),
        "provider": str(client_info.get("provider") or ""),
        "analysis": analysis,
        "row": stored.get("row"),
    }


def generate_split_candidates(bundle_id: str) -> Dict[str, Any]:
    row = dict(get_bundle(bundle_id).get("row") or {})
    analysis = _analysis_dict(row)
    suggested_regions = [
        dict(item)
        for item in list(analysis.get("suggested_split_regions") or [])
        if isinstance(item, dict)
    ]

    metadata = _read_bundle_metadata(row)
    copied_files = metadata.get("copied_files") if isinstance(metadata.get("copied_files"), dict) else {}
    qty_token_boxes = _normalised_qty_tokens_from_metadata(row, metadata)
    original_path = Path(str(copied_files.get("original_crop") or "").strip())
    mask_path = Path(str(copied_files.get("raw_master_mask") or "").strip())
    overlay_path = Path(str(copied_files.get("full_mask_overlay") or "").strip())
    if not original_path.exists() or not original_path.is_file():
        raise FileNotFoundError("original crop not found")
    if not mask_path.exists() or not mask_path.is_file():
        raise FileNotFoundError("raw master mask not found")

    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR) if overlay_path.exists() else None
    if original is None or getattr(original, "size", 0) == 0:
        raise ValueError("original crop is unreadable")
    if raw_mask is None or getattr(raw_mask, "size", 0) == 0:
        raise ValueError("raw master mask is unreadable")

    height, width = original.shape[:2]
    if raw_mask.shape[:2] != (height, width):
        raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if overlay is None or getattr(overlay, "size", 0) == 0:
        preview = original.copy()
    else:
        if overlay.shape[:2] != (height, width):
            overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)
        preview = overlay.copy()

    bundle_dir = _metadata_path_for_row(row).parent
    bundle_dir.mkdir(parents=True, exist_ok=True)
    baseline_candidates: List[Dict[str, Any]] = []
    ai_candidates: List[Dict[str, Any]] = []
    rejected_components: List[Dict[str, Any]] = []
    palette = [(0, 80, 255), (0, 180, 80), (255, 100, 0), (200, 0, 200), (0, 180, 220)]
    component_source_mask = (raw_mask > 0).astype(np.uint8) * 255
    qty_reject_pad = 2
    padded_qty_boxes: List[List[int]] = []
    for token in qty_token_boxes:
        token_box = _clip_box(
            [
                float(token.get("x", 0) or 0) - qty_reject_pad,
                float(token.get("y", 0) or 0) - qty_reject_pad,
                float(token.get("w", 0) or 0) + qty_reject_pad * 2,
                float(token.get("h", 0) or 0) + qty_reject_pad * 2,
            ],
            width,
            height,
        )
        if token_box:
            padded_qty_boxes.append(token_box)
    component_count, component_labels, component_stats, component_centroids = cv2.connectedComponentsWithStats(
        component_source_mask,
        8,
    )
    min_component_area = max(8, int(round((width * height) * 0.0015)))
    components: List[Dict[str, Any]] = []
    for component_label in range(1, int(component_count)):
        x = int(component_stats[component_label, cv2.CC_STAT_LEFT])
        y = int(component_stats[component_label, cv2.CC_STAT_TOP])
        w = int(component_stats[component_label, cv2.CC_STAT_WIDTH])
        h = int(component_stats[component_label, cv2.CC_STAT_HEIGHT])
        area = int(component_stats[component_label, cv2.CC_STAT_AREA])
        component = {
            "component_label": int(component_label),
            "box": [x, y, w, h],
            "area": area,
            "centroid": [
                float(component_centroids[component_label][0]),
                float(component_centroids[component_label][1]),
            ],
        }
        rejected_reason = ""
        qty_overlap_ratio = 0.0
        bbox_overlap_ratio = 0.0
        center_inside_qty = False
        if area <= 0:
            rejected_reason = "no_mask_pixels"
        elif area < min_component_area:
            rejected_reason = "component_area_too_small"
        elif w < 6:
            rejected_reason = "component_width_too_small"
        elif h < 6:
            rejected_reason = "component_height_too_small"
        else:
            component_box = [x, y, w, h]
            component_pixel_area = max(1, area)
            component_overlap_pixels = 0
            if padded_qty_boxes:
                component_pixels = component_labels == component_label
                for qty_box in padded_qty_boxes:
                    qx, qy, qw, qh = qty_box
                    ix1, iy1 = max(x, qx), max(y, qy)
                    ix2, iy2 = min(x + w, qx + qw), min(y + h, qy + qh)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    component_overlap_pixels = max(
                        component_overlap_pixels,
                        int(np.count_nonzero(component_pixels[iy1:iy2, ix1:ix2])),
                    )
            qty_overlap_ratio = float(component_overlap_pixels) / float(component_pixel_area)
            center_inside_qty = any(
                _point_in_box(float(component["centroid"][0]), float(component["centroid"][1]), qty_box)
                for qty_box in padded_qty_boxes
            )
            bbox_overlap_ratio = max(
                (_box_intersection_area(component_box, qty_box) / float(max(1, w * h)) for qty_box in padded_qty_boxes),
                default=0.0,
            )
            if qty_overlap_ratio > 0.5 or center_inside_qty:
                rejected_reason = "qty_text_component"
        if rejected_reason:
            rejected_components.append({
                **component,
                "rejected_reason": rejected_reason,
                "qty_overlap_ratio": qty_overlap_ratio,
                "bbox_overlap_ratio": bbox_overlap_ratio,
                "center_inside_qty": center_inside_qty,
            })
            continue
        components.append(component)
    components.sort(key=lambda item: (int(item["box"][1]), int(item["box"][0])))
    qty_component_assignments = _assign_qty_tokens_to_components(components, qty_token_boxes)
    component_order = {
        int(component.get("component_label")): index
        for index, component in enumerate(components)
    }
    ordered_components = sorted(
        components,
        key=lambda component: (
            qty_component_assignments.get(int(component.get("component_label")), {}).get(
                "anchor_order",
                10000 + component_order.get(int(component.get("component_label")), 0),
            ),
            int(component.get("box", [0, 0, 0, 0])[1]),
            int(component.get("box", [0, 0, 0, 0])[0]),
        ),
    )

    def write_candidate(
        *,
        group: str,
        display_index: int,
        internal_index: int,
        source_box: List[int],
        component_label: int = 0,
        component_mask: np.ndarray,
        assigned_qty: Dict[str, Any],
        label: str,
        reason: str,
        confidence: float,
    ) -> Dict[str, Any]:
        x, y, w, h = source_box
        scoped_component_mask = np.zeros_like(component_mask, dtype=np.uint8)
        scoped_component_mask[y : y + h, x : x + w] = (component_mask[y : y + h, x : x + w] > 0).astype(np.uint8) * 255
        local_mask = (scoped_component_mask[y : y + h, x : x + w] > 0).astype(np.uint8)
        if local_mask is None or getattr(local_mask, "size", 0) == 0 or int(np.count_nonzero(local_mask)) == 0:
            return {}
        nz = cv2.findNonZero(local_mask)
        if nz is None:
            return {}
        bx, by, bw, bh = cv2.boundingRect(nz)
        pad = 6
        raw_tight_x = max(0, x + bx - pad)
        raw_tight_y = max(0, y + by - pad)
        raw_tight_x2 = min(width, x + bx + bw + pad)
        raw_tight_y2 = min(height, y + by + bh + pad)
        export_alpha_full = _expanded_export_alpha(original, scoped_component_mask)
        export_nz = cv2.findNonZero((export_alpha_full > 0).astype(np.uint8))
        if export_nz is not None:
            ex, ey, ew, eh = cv2.boundingRect(export_nz)
            tight_x = max(0, ex - pad)
            tight_y = max(0, ey - pad)
            tight_x2 = min(width, ex + ew + pad)
            tight_y2 = min(height, ey + eh + pad)
        else:
            tight_x = raw_tight_x
            tight_y = raw_tight_y
            tight_x2 = raw_tight_x2
            tight_y2 = raw_tight_y2
        tight_box = [tight_x, tight_y, tight_x2 - tight_x, tight_y2 - tight_y]
        tx, ty, tw, th = tight_box
        tight_original = original[ty : ty + th, tx : tx + tw]
        raw_tight_mask = (scoped_component_mask[ty : ty + th, tx : tx + tw] > 0).astype(np.uint8) * 255
        tight_mask = export_alpha_full[ty : ty + th, tx : tx + tw]
        if tight_original is None or getattr(tight_original, "size", 0) == 0:
            return {}
        bgra = cv2.cvtColor(tight_original, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = tight_mask
        bgra[tight_mask == 0, 0:3] = 0
        assigned_token = assigned_qty.get("qty_token") if isinstance(assigned_qty.get("qty_token"), dict) else {}
        candidate_qty_tokens = [dict(assigned_token)] if assigned_token else []
        qty_values = list(assigned_qty.get("qty_values") or [])

        prefix = "baseline_slot_candidate" if group == "baseline_slot" else "ai_split_candidate"
        candidate_path = bundle_dir / f"{prefix}_{internal_index}.png"
        mask_candidate_path = bundle_dir / f"{prefix}_{internal_index}_mask.png"
        raw_mask_candidate_path = bundle_dir / f"{prefix}_{internal_index}_raw_mask.png"
        cv2.imwrite(str(candidate_path), bgra)
        cv2.imwrite(str(mask_candidate_path), bgra[:, :, 3])
        cv2.imwrite(str(raw_mask_candidate_path), raw_tight_mask)
        clean_candidate_path = ""
        clean_mask_candidate_path = ""
        qty_scrub_status = "not_needed"
        if candidate_qty_tokens:
            clean_bgra = bgra.copy()
            _scrub_qty_boxes_from_bgra(clean_bgra, tight_box, candidate_qty_tokens, pad=3)
            clean_path = bundle_dir / f"{prefix}_{internal_index}_qty_scrubbed.png"
            clean_mask_path = bundle_dir / f"{prefix}_{internal_index}_qty_scrubbed_mask.png"
            cv2.imwrite(str(clean_path), clean_bgra)
            cv2.imwrite(str(clean_mask_path), clean_bgra[:, :, 3])
            clean_candidate_path = str(clean_path)
            clean_mask_candidate_path = str(clean_mask_path)
            qty_scrub_status = "auto_scrubbed"
        return {
            "index": internal_index,
            "display_index": display_index,
            "group": group,
            "status": "pending",
            "box": tight_box,
            "source_box": source_box,
            "component_label": int(component_label),
            "component_area": int(np.count_nonzero(scoped_component_mask > 0)),
            "label": label,
            "reason": reason,
            "confidence": confidence,
            "candidate_path": str(candidate_path),
            "mask_path": str(mask_candidate_path),
            "raw_mask_path": str(raw_mask_candidate_path),
            "alpha_expansion": {
                "mode": "dilate_close_non_blue_background",
                "dilate_px": 2,
                "close_kernel": "3x3",
                "raw_alpha_area": int(np.count_nonzero(raw_tight_mask > 0)),
                "expanded_alpha_area": int(np.count_nonzero(tight_mask > 0)),
            },
            "base_candidate_path": str(candidate_path),
            "current_candidate_path": clean_candidate_path or str(candidate_path),
            "current_alpha_path": clean_mask_candidate_path or str(mask_candidate_path),
            "qty_detected": bool(candidate_qty_tokens),
            "qty_values": qty_values,
            "qty_token_boxes": candidate_qty_tokens,
            "qty_anchor_assignment": assigned_qty,
            "qty_scrubbed_path": clean_candidate_path,
            "qty_scrubbed_mask_path": clean_mask_candidate_path,
            "thumbnail_path": clean_candidate_path or str(candidate_path),
            "qty_scrub_status": qty_scrub_status,
            "qty_text_state": qty_scrub_status,
            "mask_review_state": "",
            "cleanup_history": [],
        }

    combined_index = 0
    for component_pos, component in enumerate(ordered_components):
        box = list(component.get("box") or [])
        component_label = int(component.get("component_label") or 0)
        if len(box) != 4:
            continue
        component_mask = (component_labels == component_label).astype(np.uint8) * 255
        color = palette[component_pos % len(palette)]
        x, y, w, h = box
        cv2.rectangle(preview, (x, y), (x + w - 1, y + h - 1), color, 2)
        assigned_qty = qty_component_assignments.get(component_label, {})
        qty_values = list(assigned_qty.get("qty_values") or [])
        label_text = f"Candidate {component_pos + 1}"
        if qty_values:
            label_text = f"{label_text} ({'/'.join(str(value) + 'x' for value in qty_values)})"
        cv2.putText(
            preview,
            label_text,
            (x, max(14, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        candidate = write_candidate(
            group="baseline_slot",
            display_index=component_pos + 1,
            internal_index=combined_index,
            source_box=box,
            component_label=component_label,
            component_mask=component_mask,
            assigned_qty=assigned_qty,
            label=f"Mask Component Candidate {component_pos + 1}",
            reason="raw_master_mask connected component with OCR qty anchor metadata",
            confidence=1.0,
        )
        if candidate:
            baseline_candidates.append(candidate)
            combined_index += 1

    ai_internal_index = 0
    for region_index, region in enumerate(suggested_regions):
        box = _clip_box(_coerce_box(region.get("box")), width, height)
        if not box:
            continue
        if any(_box_iou(box, candidate.get("source_box") or candidate.get("box") or []) >= 0.55 for candidate in baseline_candidates):
            continue
        x, y, w, h = box
        color = palette[(len(baseline_candidates) + region_index) % len(palette)]
        cv2.rectangle(preview, (x, y), (x + w - 1, y + h - 1), color, 2)
        cv2.putText(
            preview,
            f"ai {ai_internal_index + 1}",
            (x, max(14, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        candidate = write_candidate(
            group="ai_suggested",
            display_index=ai_internal_index + 1,
            internal_index=combined_index,
            source_box=box,
            component_label=0,
            component_mask=raw_mask,
            assigned_qty={},
            label=str(region.get("label") or f"AI Suggested Candidate {ai_internal_index + 1}"),
            reason=str(region.get("reason") or ""),
            confidence=float(region.get("confidence", 0) or 0),
        )
        if candidate:
            ai_candidates.append(candidate)
            ai_internal_index += 1
            combined_index += 1

    split_overlay_path = bundle_dir / "split_candidate_overlay.png"
    cv2.imwrite(str(split_overlay_path), preview)
    all_candidates = baseline_candidates + ai_candidates
    paths = {
        "overlay_path": str(split_overlay_path),
        "source_original_crop": str(original_path),
        "source_raw_master_mask": str(mask_path),
        "source_overlay": str(overlay_path) if overlay_path.exists() else "",
        "candidate_source": "raw_master_mask_connected_components",
        "qty_anchor_source": "direct_qty_ocr_boxes",
        "qty_anchor_rule": "OCR boxes assign qty/order to nearest above/near mask component; crops always come from component masks",
        "mask_component_count": int(component_count) - 1,
        "accepted_component_count": len(components),
        "rejected_component_count": len(rejected_components),
        "rejected_components": rejected_components,
        "qty_component_assignments": {
            str(label): assignment
            for label, assignment in qty_component_assignments.items()
        },
        "baseline_slot_candidates": baseline_candidates,
        "ai_suggested_candidates": ai_candidates,
        "candidates": all_candidates,
    }
    stored = update_split_candidates(str(row.get("bundle_id") or bundle_id), split_candidate_paths=paths)
    return {
        "ok": True,
        "bundle_id": str(row.get("bundle_id") or bundle_id),
        "baseline_slot_candidate_count": len(baseline_candidates),
        "ai_suggested_candidate_count": len(ai_candidates),
        "split_candidate_count": len(all_candidates),
        "mask_component_count": int(component_count) - 1,
        "accepted_component_count": len(components),
        "rejected_component_count": len(rejected_components),
        "qty_component_assignments": paths["qty_component_assignments"],
        "split_candidate_paths": paths,
        "row": stored.get("row"),
    }


def mark_split_candidate(bundle_id: str, candidate_index: int, status: str) -> Dict[str, Any]:
    row = dict(get_bundle(bundle_id).get("row") or {})
    paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
    candidates = list(paths.get("candidates") or [])
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise ValueError("candidate_index is out of range")
    candidate = dict(candidates[candidate_index]) if isinstance(candidates[candidate_index], dict) else {}
    _qty_clean_states = {"auto_scrubbed", "not_needed", "manual_mark_clean", "not_detected", "scrubbed"}
    _qty_is_clean = (
        not bool(candidate.get("qty_detected"))
        or str(candidate.get("qty_text_state") or "") in _qty_clean_states
        or bool(str(candidate.get("qty_scrubbed_path") or "").strip())
    )
    if status == "accepted" and not _qty_is_clean:
        raise ValueError("candidate has detected qty text; scrub or mark qty clean before accepting")
    v2_mask_path = ""
    if status == "accepted":
        mask_path = Path(str(candidate.get("mask_path") or "").strip())
        if mask_path.exists() and mask_path.is_file():
            v2_path = mask_path.with_name(f"split_candidate_{candidate_index}_accepted_v2_mask.png")
            shutil.copy2(str(mask_path), str(v2_path))
            v2_mask_path = str(v2_path)
    stored = update_split_candidate_status(
        str(row.get("bundle_id") or bundle_id),
        candidate_index=int(candidate_index),
        status=status,
        v2_mask_path=v2_mask_path,
    )
    return {
        "ok": True,
        "bundle_id": str(row.get("bundle_id") or bundle_id),
        "candidate_index": int(candidate_index),
        "status": status,
        "v2_mask_path": v2_mask_path,
        "row": stored.get("row"),
    }


def set_split_candidate_review_state(bundle_id: str, candidate_index: int, review_state: str) -> Dict[str, Any]:
    allowed_states = {"", "needs_mask_expand", "needs_ocr_review", "needs_manual_crop"}
    state = str(review_state or "").strip()
    if state not in allowed_states:
        raise ValueError("review_state must be needs_mask_expand, needs_ocr_review, needs_manual_crop, or empty")
    row = dict(get_bundle(bundle_id).get("row") or {})
    paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
    candidates = list(paths.get("candidates") or [])
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise ValueError("candidate_index is out of range")

    def update_candidate(raw_candidate: Any) -> Dict[str, Any]:
        item = dict(raw_candidate) if isinstance(raw_candidate, dict) else {}
        try:
            item_index = int(item.get("index"))
        except Exception:
            item_index = None
        if item_index != int(candidate_index):
            return item
        item["review_state"] = state
        if state and str(item.get("status") or "") == "accepted":
            item["status"] = "pending"
        return item

    paths["candidates"] = [update_candidate(item) for item in list(paths.get("candidates") or [])]
    paths["baseline_slot_candidates"] = [update_candidate(item) for item in list(paths.get("baseline_slot_candidates") or [])]
    paths["ai_suggested_candidates"] = [update_candidate(item) for item in list(paths.get("ai_suggested_candidates") or [])]
    stored = update_split_candidates(str(row.get("bundle_id") or bundle_id), split_candidate_paths=paths)
    return {
        "ok": True,
        "bundle_id": str(row.get("bundle_id") or bundle_id),
        "candidate_index": int(candidate_index),
        "review_state": state,
        "row": stored.get("row"),
    }


def scrub_candidate_qty(bundle_id: str, candidate_index: int) -> Dict[str, Any]:
    row = dict(get_bundle(bundle_id).get("row") or {})
    metadata = _read_bundle_metadata(row)
    qty_token_boxes = _normalised_qty_tokens_from_metadata(row, metadata)
    paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
    candidates = list(paths.get("candidates") or [])
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise ValueError("candidate_index is out of range")
    candidate = dict(candidates[candidate_index]) if isinstance(candidates[candidate_index], dict) else {}
    candidate_path = Path(str(
        candidate.get("current_candidate_path")
        or candidate.get("candidate_path")
        or ""
    ).strip())
    if not candidate_path.exists() or not candidate_path.is_file():
        raise FileNotFoundError("candidate image not found")
    crop_box = [int(value) for value in list(candidate.get("box") or [])[:4]]
    if len(crop_box) != 4:
        raise ValueError("candidate box missing")
    image = cv2.imread(str(candidate_path), cv2.IMREAD_UNCHANGED)
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("candidate image is unreadable")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    matched_tokens = _candidate_qty_tokens(crop_box, qty_token_boxes)
    _scrub_qty_boxes_from_bgra(image, crop_box, qty_token_boxes, pad=3)
    scrubbed_path = candidate_path.with_name(f"{candidate_path.stem}_qty_scrubbed.png")
    scrubbed_mask_path = candidate_path.with_name(f"{candidate_path.stem}_qty_scrubbed_mask.png")
    cv2.imwrite(str(scrubbed_path), image)
    cv2.imwrite(str(scrubbed_mask_path), image[:, :, 3])

    def update_candidate(raw_candidate: Any) -> Dict[str, Any]:
        item = dict(raw_candidate) if isinstance(raw_candidate, dict) else {}
        item_index = item.get("index")
        try:
            parsed_index = int(item_index)
        except Exception:
            parsed_index = -1
        if parsed_index != int(candidate_index):
            return item
        item["qty_detected"] = bool(matched_tokens)
        item["qty_values"] = _qty_values_from_tokens(matched_tokens)
        item["qty_token_boxes"] = matched_tokens
        item["qty_scrubbed_path"] = str(scrubbed_path)
        item["qty_scrubbed_mask_path"] = str(scrubbed_mask_path)
        item["qty_scrub_status"] = "scrubbed"
        item["qty_text_state"] = "auto_scrubbed"
        item["current_candidate_path"] = str(scrubbed_path)
        item["current_alpha_path"] = str(scrubbed_mask_path)
        history = list(item.get("cleanup_history") or [])
        history.append({"op": "qty_scrub", "path": str(scrubbed_path)})
        item["cleanup_history"] = history
        return item

    paths["candidates"] = [update_candidate(item) for item in list(paths.get("candidates") or [])]
    paths["baseline_slot_candidates"] = [update_candidate(item) for item in list(paths.get("baseline_slot_candidates") or [])]
    paths["ai_suggested_candidates"] = [update_candidate(item) for item in list(paths.get("ai_suggested_candidates") or [])]
    stored = update_split_candidates(str(row.get("bundle_id") or bundle_id), split_candidate_paths=paths)
    return {
        "ok": True,
        "bundle_id": str(row.get("bundle_id") or bundle_id),
        "candidate_index": int(candidate_index),
        "qty_detected": bool(matched_tokens),
        "qty_values": _qty_values_from_tokens(matched_tokens),
        "qty_scrubbed_path": str(scrubbed_path),
        "qty_scrubbed_mask_path": str(scrubbed_mask_path),
        "row": stored.get("row"),
    }
