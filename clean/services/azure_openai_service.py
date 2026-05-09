import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from clean.routers.debug import _require_openai_vision_client_debug, _response_text_to_json_debug


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _encode_debug_image_data_uri(img: Any) -> Optional[str]:
    if img is None or getattr(img, "size", 0) == 0:
        return None
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


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


def rank_crop_candidates(
    crop: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    deduped_candidates = _dedupe_candidate_rows(list(candidates or []))
    crop_image_info = _resolve_crop_image_for_ai_rank(crop)
    candidate_img_urls = [
        str(candidate.get("img_url") or "").strip()
        for candidate in deduped_candidates
        if str(candidate.get("img_url") or "").strip()
    ]
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    
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
    }
    
    crop_image_url = _input_image_url_for_debug_ref(crop_image_info.get("image_url"))
    if not crop_image_url:
        result["reason"] = "Crop image missing."
        return result
    result["crop_image_included"] = True

    candidate_rows: List[Dict[str, Any]] = []
    prompt_text = (
        "Choose the best LEGO part candidate for this callout crop. "
        "Use only the provided crop image and up to 5 candidate element images. "
        "Return JSON only with best_part_num, best_color_id, best_element_id, confidence, reason. "
        "Do not invent candidates outside the provided list."
    )
    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": prompt_text},
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
        result["reason"] = "No candidate images available."
        return result
    
    if not _azure_ai_rank_available():
        result["reason"] = "Azure AI rank unavailable."
        return result

    try:
        client = _require_crop_ai_client()
        response = client.responses.create(
            model=_crop_ai_model_name(),
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
        raw_text = _response_output_text_debug(response)
        result["raw_response_text"] = raw_text
        payload: Dict[str, Any] = {}
        try:
            payload = _response_text_to_json_debug(response) or {}
        except Exception:
            payload = {}
        result["parsed_response_json"] = payload or None
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
        result["exception_type"] = exc.__class__.__name__
        result["exception_message"] = str(exc)
        result["reason"] = f"Azure rank failed: {str(exc)}"
        return result
