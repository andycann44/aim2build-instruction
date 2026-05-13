import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from clean.routers.debug import _require_openai_vision_client_debug, _response_text_to_json_debug


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    azure_ready = bool(
        os.getenv("AZURE_OPENAI_ENDPOINT")
        and os.getenv("AZURE_OPENAI_API_KEY")
        and os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )
    openai_ready = bool(os.getenv("OPENAI_API_KEY"))
    return azure_ready or openai_ready


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


def _mime_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _response_output_text_debug(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    try:
        response_dict = response.model_dump()
    except Exception:
        response_dict = None

    texts: List[str] = []
    if isinstance(response_dict, dict):
        for item in response_dict.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    texts.append(str(content.get("text") or "").strip())
    return "\n".join(text for text in texts if text)


def _input_image_url_for_debug_ref(image_ref: Any) -> Optional[str]:
    text = str(image_ref or "").strip()
    if not text:
        return None
    if text.startswith("data:image/"):
        return text
    if text.startswith(("http://", "https://")):
        return text
    if text.startswith("file://"):
        text = text[7:]
    path = Path(text)
    if not path.exists():
        repo_path = _repo_root() / text
        if repo_path.exists():
            path = repo_path
    if not path.exists() or not path.is_file():
        return None
    raw = path.read_bytes()
    return "data:%s;base64,%s" % (
        _mime_type_for_path(path),
        base64.b64encode(raw).decode("ascii"),
    )


def _resolve_crop_image_for_ai_rank(crop: Dict[str, Any]) -> Dict[str, Any]:
    crop_dict = crop if isinstance(crop, dict) else {}
    data_uri = str(crop_dict.get("data_uri") or "").strip()
    if data_uri.startswith("data:image/"):
        return {
            "image_url": data_uri,
            "exists": True,
            "source": "data_uri",
        }

    image_ref = str(crop_dict.get("image_url") or "").strip()
    if image_ref:
        resolved = _input_image_url_for_debug_ref(image_ref)
        return {
            "image_url": resolved or image_ref,
            "exists": bool(resolved or image_ref),
            "source": "image_url",
        }

    crop_image_path = str(crop_dict.get("crop_image_path") or "").strip()
    crop_box = _coerce_box_list(crop_dict.get("crop_box"))
    if crop_image_path:
        path = Path(crop_image_path)
        if not path.exists():
            repo_path = _repo_root() / crop_image_path
            if repo_path.exists():
                path = repo_path
        if path.exists() and path.is_file():
            if crop_box is not None:
                img = cv2.imread(str(path))
                if img is not None and getattr(img, "size", 0) != 0:
                    x, y, w, h = [int(value) for value in crop_box]
                    crop_img = img[max(0, y) : max(0, y) + max(0, h), max(0, x) : max(0, x) + max(0, w)]
                    encoded = _encode_debug_image_data_uri(crop_img)
                    if encoded:
                        return {
                            "image_url": encoded,
                            "exists": True,
                            "source": "crop_box_from_page",
                        }
            return {
                "image_url": str(path),
                "exists": True,
                "source": "local_path",
            }

    return {
        "image_url": "",
        "exists": False,
        "source": "missing",
    }


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
        "ranked_candidates": [],
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
        "model": _crop_ai_model_name(),
    }
    
    crop_image_url = _input_image_url_for_debug_ref(crop_image_info.get("image_url"))
    if not crop_image_url:
        result["reason"] = "Crop image missing."
        return result
    result["crop_image_included"] = True

    candidate_rows: List[Dict[str, Any]] = []
    slot_qty_text = str(crop.get("slot_qty_text") or crop.get("next_qty_label") or "").strip()
    manual_color_filter_id = _coerce_int(crop.get("manual_color_filter_id"))
    manual_color_name = str(crop.get("manual_color_name") or "").strip()
    prompt_text = (
        "Rank the provided LEGO part candidates for this single instruction crop slot. "
        "Use only the provided crop image and candidate images. "
        "Choose only from the indexed candidates provided here. "
        "Prefer visible shape first and use colour as secondary evidence. "
        "Return strict JSON only with ranked_candidates and summary. "
        "Do not invent candidates outside the provided list."
    )
    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": prompt_text},
        {
            "type": "input_text",
            "text": json.dumps(
                {
                    "crop_id": str(crop.get("crop_id") or "").strip(),
                    "page": _coerce_int(crop.get("page")),
                    "step": _coerce_int(crop.get("step")),
                    "slot_index": _coerce_int(crop.get("slot_index")),
                    "slot_qty_text": slot_qty_text or None,
                    "manual_color_filter_id": manual_color_filter_id,
                    "manual_color_name": manual_color_name or None,
                }
            ),
        },
        {"type": "input_image", "image_url": crop_image_url, "detail": "high"},
    ]
    result["prompt_text"] = prompt_text
    for idx, candidate in enumerate(deduped_candidates[:8], start=1):
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
            "remaining_qty": _coerce_int(candidate.get("remaining_qty")),
            "color_name": str(candidate.get("color_name") or "").strip() or None,
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
                            "ranked_candidates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "confidence": {"type": "number"},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["index", "confidence", "reason"],
                                },
                            },
                            "summary": {"type": "string"},
                        },
                        "required": ["ranked_candidates", "summary"],
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
        ranked_candidates: List[Dict[str, Any]] = []
        for rank, ranked_item in enumerate(list(payload.get("ranked_candidates", []) or []), start=1):
            candidate_index = _coerce_int((ranked_item or {}).get("index"))
            if candidate_index is None or candidate_index < 1 or candidate_index > len(candidate_rows):
                continue
            candidate_row = dict(candidate_rows[candidate_index - 1] or {})
            ranked_candidates.append(
                {
                    "rank": rank,
                    "index": candidate_index,
                    "part_num": str(candidate_row.get("part_num") or "").strip(),
                    "color_id": int(candidate_row.get("color_id", 0) or 0),
                    "element_id": candidate_row.get("element_id"),
                    "img_url": str(candidate_row.get("img_url") or ""),
                    "candidate_source": str(candidate_row.get("candidate_source") or ""),
                    "score": float(candidate_row.get("score", 0.0) or 0.0),
                    "remaining_qty": _coerce_int(candidate_row.get("remaining_qty")),
                    "color_name": str(candidate_row.get("color_name") or "").strip() or None,
                    "confidence": _coerce_float((ranked_item or {}).get("confidence")),
                    "reason": str((ranked_item or {}).get("reason") or "").strip(),
                }
            )
        best_candidate = ranked_candidates[0] if ranked_candidates else {}
        result.update(
            {
                "enabled": bool(ranked_candidates),
                "ranked_candidates": ranked_candidates,
                "best_part_num": str(best_candidate.get("part_num") or "").strip() or None,
                "best_color_id": _coerce_int(best_candidate.get("color_id")),
                "best_element_id": str(best_candidate.get("element_id") or "").strip() or None,
                "confidence": _coerce_float(best_candidate.get("confidence")),
                "reason": str(payload.get("summary") or "").strip()
                or str(best_candidate.get("reason") or "").strip()
                or (raw_text[:300] if raw_text else ""),
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
