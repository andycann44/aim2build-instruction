#!/usr/bin/env python3

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path("/Users/olly/aim2build-instruction")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean.services.debug_service import resolve_page_image_path  # noqa: E402


DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")
OUTPUT_ROOT = PROJECT_ROOT / "debug" / "openai_page_structure"


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
    b64 = base64.b64encode(raw).decode("ascii")
    return "data:%s;base64,%s" % (mime_type, b64)


def _response_text_to_json_payload(response: Any) -> Dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return json.loads(output_text)

    try:
        response_dict = response.model_dump()
    except Exception:
        response_dict = None

    if isinstance(response_dict, dict):
        output = response_dict.get("output", [])
        for item in output:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return json.loads(content["text"])

    raise RuntimeError("Could not extract JSON text from OpenAI response.")


def _build_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "set_num": {"type": "string"},
            "page": {"type": "integer"},
            "page_type": {
                "type": "string",
                "enum": ["bag_start", "step_page", "mixed", "unknown"],
            },
            "page_number": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "value": {"type": ["integer", "null"]},
                    "side": {
                        "type": "string",
                        "enum": ["left", "right", "unknown"],
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["value", "side", "confidence"],
            },
            "bag_start": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "is_bag_start": {"type": "boolean"},
                    "bag_number": {"type": ["integer", "null"]},
                    "confidence": {"type": "number"},
                },
                "required": ["is_bag_start", "bag_number", "confidence"],
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "step_number": {"type": ["integer", "null"]},
                        "bbox_hint": {
                            "type": "string",
                            "enum": [
                                "top-left",
                                "top-right",
                                "middle",
                                "bottom-left",
                                "bottom-right",
                                "unknown",
                            ],
                        },
                        "x_markers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "part_callouts_visible": {"type": "boolean"},
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "step_number",
                        "bbox_hint",
                        "x_markers",
                        "part_callouts_visible",
                        "notes",
                    ],
                },
            },
            "callouts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "bbox_hint": {
                            "type": "string",
                            "enum": [
                                "top-left",
                                "top-right",
                                "middle",
                                "bottom-left",
                                "bottom-right",
                                "unknown",
                            ],
                        },
                        "likely_parts_count": {"type": ["integer", "null"]},
                    },
                    "required": ["text", "bbox_hint", "likely_parts_count"],
                },
            },
            "observations": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "set_num",
            "page",
            "page_type",
            "page_number",
            "bag_start",
            "steps",
            "callouts",
            "observations",
        ],
    }


def _build_prompt(set_num: str, page: int) -> str:
    return (
        "Analyze this LEGO instruction manual page and return strict JSON only. "
        "Describe visible page structure, not hidden intent. "
        "Do not invent LEGO part IDs or unseen text. "
        "If uncertain, use null, empty arrays, or lower confidence values. "
        "Classify whether the page looks like a bag-start page, a step page, mixed, or unknown. "
        "Identify likely footer page number side if visible. "
        "List visible step numbers, quantity markers like 2x or 24x, and part callouts only when clearly visible. "
        "Use coarse bbox hints only from this set: top-left, top-right, middle, bottom-left, bottom-right, unknown. "
        "The requested set number is %s and the requested page is %d."
    ) % (set_num, page)


def analyze_page_structure(set_num: str, page: int, model: str) -> Dict[str, Any]:
    image_path = resolve_page_image_path(set_num, page)
    if image_path is None:
        raise RuntimeError(
            "Rendered page image not found for set %s page %d." % (set_num, page)
        )

    client = _require_openai_client()
    data_url = _encode_image_as_data_url(image_path)
    schema = _build_schema()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _build_prompt(set_num, page),
                    },
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "high",
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "lego_page_structure",
                "strict": True,
                "schema": schema,
            }
        },
    )

    payload = _response_text_to_json_payload(response)
    payload["set_num"] = str(set_num)
    payload["page"] = int(page)
    return payload


def _save_output(payload: Dict[str, Any], set_num: str, page: int) -> Path:
    out_dir = OUTPUT_ROOT / str(set_num)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("page_%d.json" % int(page))
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimental OpenAI page-structure extraction for a rendered LEGO manual page."
    )
    parser.add_argument("--set", dest="set_num", required=True, help="LEGO set number")
    parser.add_argument("--page", dest="page", required=True, type=int, help="Rendered page number")
    parser.add_argument(
        "--model",
        dest="model",
        default=DEFAULT_MODEL,
        help="Vision-capable OpenAI model to use (default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        payload = analyze_page_structure(args.set_num, int(args.page), args.model)
        out_path = _save_output(payload, args.set_num, int(args.page))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "set_num": str(args.set_num),
                    "page": int(args.page),
                    "error": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(payload, indent=2, sort_keys=False))
    print("Saved to %s" % out_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
