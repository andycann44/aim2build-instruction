import base64
from html import escape
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from clean.routers.debug import (
    _build_material_crop_candidates,
    _contact_sheet_step_boxes_from_detected,
    _encode_contact_sheet_crop,
    _encode_debug_image_data_uri,
    _extract_detected_qty_details_from_crop,
    _extract_qty_tokens_from_image,
    _resolve_bag_page_range,
    _response_text_to_json_debug,
)
from clean.services.azure_openai_service import (
    rank_crop_candidates,
)
from clean.services.part_candidate_service import get_part_candidates_for_crop
from clean.services import debug_service, step_detector_service
from clean.services.instruction_buildability_source import load_instruction_set_parts

router = APIRouter()

# Other functions remain unchanged...

@router.post("/debug/ai-rank-crop")
async def ai_rank_crop(req: Request):
    data = await req.json()
    crop = data.get("crop") if isinstance(data.get("crop"), dict) else {}
    candidates = data.get("top_part_candidates") if isinstance(data.get("top_part_candidates"), list) else []
    
    ai_rank_result = rank_crop_candidates(crop, candidates)
    
    return {
        "ok": True,
        "top_part_candidates": ai_rank_result.get("top_part_candidates"),
        "ai_rank_best_candidate": ai_rank_result.get("ai_rank_best_candidate"),
        "ai_rank_confidence": ai_rank_result.get("ai_rank_confidence"),
        "ai_rank_reason": ai_rank_result.get("ai_rank_reason"),
        "ai_rank_candidate": ai_rank_result.get("ai_rank_candidate"),
        "ai_rank_debug": ai_rank_result.get("ai_rank_debug"),
    }

# Other functions remain unchanged...
