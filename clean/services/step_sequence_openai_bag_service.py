import json
import os
from pathlib import Path
from typing import Any, Dict, List

from clean.services import debug_service, step_sequence_bag_service


PROJECT_ROOT = Path("/Users/olly/aim2build-instruction")
OPENAI_VERIFY_ROOT = PROJECT_ROOT / "debug" / "openai_bag_verify"
DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")


def _load_cached_bag_verify(set_num: str, page: int) -> Dict[str, Any]:
    cache_path = OPENAI_VERIFY_ROOT / str(set_num) / ("page_%d.json" % int(page))
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _require_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI Python SDK is not installed. Install `openai` in this environment first."
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required in the environment.")

    return OpenAI(api_key=api_key)


def _encode_image_as_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime_type = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    else:
        mime_type = "application/octet-stream"

    raw = image_path.read_bytes()
    import base64

    return "data:%s;base64,%s" % (
        mime_type,
        base64.b64encode(raw).decode("ascii"),
    )


def _response_text_to_json_payload(response: Any) -> Dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return json.loads(output_text)

    try:
        response_dict = response.model_dump()
    except Exception:
        response_dict = None

    if isinstance(response_dict, dict):
        for item in response_dict.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return json.loads(content["text"])

    raise RuntimeError("Could not extract JSON text from OpenAI response.")


def _verification_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_real_bag_start": {"type": "boolean"},
            "main_steps": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "previous_page_main_steps": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": [
            "is_real_bag_start",
            "main_steps",
            "previous_page_main_steps",
            "reason",
            "confidence",
        ],
    }


def _verification_prompt(set_num: str, page: int, previous_page: int) -> str:
    return (
        "You are comparing two LEGO instruction manual pages using visible structure only. "
        "You will see the candidate page first, then the previous page. "
        "Return strict JSON only matching the schema. "
        "Do not invent LEGO part IDs or unseen text. "
        "Do not mention sorting bags, opening bags, bag context, or hidden meaning. "
        "Do not infer intent beyond what is visibly shown. "
        "A page is a real bag start ONLY if all of these are visibly true: "
        "the main visible step number resets to 1, the previous page has a higher visible main step number of at least 10, "
        "the candidate page is a normal instruction page rather than an intro, cover, parts, or overview page, "
        "and the candidate page clearly shows a new build section such as a new base, a new structure, or a clear visual break. "
        "If any of those are not visibly clear, set is_real_bag_start to false. "
        "main_steps must contain only visible large main step numbers on the candidate page. "
        "previous_page_main_steps must contain only visible large main step numbers on the previous page. "
        "reason must describe only visible layout facts. "
        "Set number: %s. Candidate page: %d. Previous page: %d."
    ) % (set_num, page, previous_page)


def _save_verify_debug_json(payload: Dict[str, Any], set_num: str, page: int) -> None:
    out_dir = OPENAI_VERIFY_ROOT / str(set_num)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("page_%d.json" % int(page))
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _verify_candidate_page(
    set_num: str,
    page: int,
    model: str,
) -> Dict[str, Any]:
    cached = _load_cached_bag_verify(set_num, page)
    if cached:
        return cached

    current_image_path = debug_service.resolve_page_image_path(set_num, page)
    if current_image_path is None:
        raise RuntimeError("Rendered page image not found for page %d." % int(page))

    previous_page = max(1, int(page) - 1)
    previous_image_path = debug_service.resolve_page_image_path(set_num, previous_page)

    client = _require_openai_client()
    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": _verification_prompt(set_num, int(page), int(previous_page)),
        },
        {
            "type": "input_image",
            "image_url": _encode_image_as_data_url(current_image_path),
            "detail": "high",
        },
    ]
    if previous_image_path is not None:
        content.append(
            {
                "type": "input_image",
                "image_url": _encode_image_as_data_url(previous_image_path),
                "detail": "high",
            }
        )

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "lego_bag_verify",
                "strict": True,
                "schema": _verification_schema(),
            }
        },
    )

    payload = _response_text_to_json_payload(response)
    _save_verify_debug_json(payload, set_num, page)
    return payload


def scan_step_bag_sequence_openai_verify(
    set_num: str,
    start_page: int,
    end_page: int,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    local_scan = step_sequence_bag_service.scan_step_bag_sequence(
        set_num=set_num,
        start_page=int(start_page),
        end_page=int(end_page),
    )
    local_candidates = local_scan.get("candidate_bag_starts", [])
    page_steps = local_scan.get("page_steps", [])
    page_step_map = {
        int(row.get("page", 0) or 0): row
        for row in page_steps
    }
    verified_bag_starts: List[Dict[str, Any]] = []
    skipped_candidates: List[Dict[str, Any]] = []

    for candidate in local_candidates:
        page = int(candidate.get("page", 0) or 0)
        previous_page = max(1, int(page) - 1)
        previous_row = page_step_map.get(previous_page, {}) or {}
        previous_main_steps = [
            int(item.get("value", 0) or 0)
            for item in (previous_row.get("main_steps", []) or [])
        ]
        page_data = page_step_map.get(page, {}) or {}
        main_steps = [
            int(item.get("value", 0) or 0)
            for item in (page_data.get("main_steps", []) or [])
        ]
        sub_steps = [int(v) for v in (page_data.get("sub_steps", []) or [])]
        main_min = min(main_steps) if main_steps else None
        has_substep_one = 1 in sub_steps
        allow_candidate = (
            (main_min == 1)
            or (main_min is not None and main_min >= 10 and has_substep_one)
        )

        print(f"[verify] page={page} main_steps={main_steps} sub_steps={sub_steps}")

        if page <= 3:
            skipped_candidates.append(
                {"page": int(page), "reason": "skipped early intro page"}
            )
            continue

        if not previous_main_steps:
            skipped_candidates.append(
                {"page": int(page), "reason": "skipped previous page has no valid main steps"}
            )
            continue

        previous_max_step = max(previous_main_steps)
        current_min_step = main_min
        step_drop = (
            int(previous_max_step) - int(current_min_step)
            if current_min_step is not None
            else None
        )

        if previous_max_step < 10:
            skipped_candidates.append(
                {"page": int(page), "reason": "skipped previous max step < 10"}
            )
            continue

        if not allow_candidate:
            skipped_candidates.append(
                {
                    "page": int(page),
                    "reason": "skipped no valid step-1 start (neither main nor substep)",
                }
            )
            continue

        if step_drop is None or step_drop < 10:
            skipped_candidates.append(
                {"page": int(page), "reason": "skipped step drop < 10"}
            )
            continue

        payload = _verify_candidate_page(set_num=set_num, page=page, model=model)
        if bool(payload.get("is_real_bag_start")):
            verified_bag_starts.append(
                {
                    "page": int(page),
                    "is_real_bag_start": bool(payload.get("is_real_bag_start")),
                    "main_steps": payload.get("main_steps", []),
                    "previous_page_main_steps": payload.get("previous_page_main_steps", []),
                    "reason": str(payload.get("reason", "")),
                    "confidence": float(payload.get("confidence", 0.0) or 0.0),
                }
            )

    verified_bag_starts.sort(key=lambda item: int(item.get("page", 0) or 0))

    return {
        "set_num": str(set_num),
        "local_candidates": local_candidates,
        "verified_bag_starts": verified_bag_starts,
        "skipped_candidates": skipped_candidates,
    }
