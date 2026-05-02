import json
import re
import sqlite3
from typing import Any, Dict, List, Optional

import cv2
import pytesseract
from pytesseract import Output

from clean.services import debug_service

ELEMENT_ID_RE = re.compile(r"^\d{6,7}$")
QTY_RE = re.compile(r"^(?:[xX](\d{1,3})|(\d{1,3})[xX]?)$")
INVENTORY_PAGE_MIN_ELEMENT_IDS = 12
OCR_SCALE = 2.5
CATALOG_DB_PATH = debug_service.PROJECT_ROOT / "bag_inspector.db"
CATALOG_MAPS_DIR = debug_service.DEBUG_ROOT / "catalog_maps"
_CATALOG_JSON_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _load_catalog_table_names() -> set[str]:
    if not CATALOG_DB_PATH.exists():
        return set()

    try:
        conn = sqlite3.connect(str(CATALOG_DB_PATH))
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return set()

    return {str(row[0]) for row in rows}


CATALOG_TABLE_NAMES = _load_catalog_table_names()
CATALOG_JOIN_AVAILABLE = {
    "elements",
    "inventory_parts",
    "inventories",
}.issubset(CATALOG_TABLE_NAMES)


def _catalog_map_path(set_num: str):
    return CATALOG_MAPS_DIR / f"{str(set_num)}_element_map.json"


def _load_catalog_json_map(set_num: str) -> Dict[str, Dict[str, Any]]:
    set_key = str(set_num)
    if set_key in _CATALOG_JSON_CACHE:
        return _CATALOG_JSON_CACHE[set_key]

    path = _catalog_map_path(set_key)
    if not path.exists():
        _CATALOG_JSON_CACHE[set_key] = {}
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _CATALOG_JSON_CACHE[set_key] = {}
        return {}

    if not isinstance(payload, dict):
        _CATALOG_JSON_CACHE[set_key] = {}
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for element_id, row in payload.items():
        if not isinstance(row, dict):
            continue
        normalized[str(element_id)] = {
            "part_num": row.get("part_num"),
            "color_id": row.get("color_id"),
            "qty": row.get("qty"),
        }

    _CATALOG_JSON_CACHE[set_key] = normalized
    return normalized


def _extract_qty_value(text: str) -> Optional[int]:
    match = QTY_RE.fullmatch(str(text or "").strip())
    if not match:
        return None

    raw = match.group(1) or match.group(2)
    if raw is None:
        return None

    value = int(raw)
    if value <= 0 or value > 500:
        return None
    return value


def _ocr_inventory_tokens(image_path: str) -> List[Dict[str, Any]]:
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    scaled = cv2.resize(
        img,
        None,
        fx=OCR_SCALE,
        fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(
        gray,
        config="--psm 11 -c tessedit_char_whitelist=0123456789xX",
        output_type=Output.DICT,
    )

    tokens: List[Dict[str, Any]] = []
    token_count = len(data.get("text", []))
    for i in range(token_count):
        text = str((data.get("text", [""])[i] or "")).strip()
        if not text:
            continue

        try:
            conf = float(data.get("conf", ["-1"])[i])
        except Exception:
            conf = -1.0

        left = int(data.get("left", [0])[i] or 0)
        top = int(data.get("top", [0])[i] or 0)
        width = int(data.get("width", [0])[i] or 0)
        height = int(data.get("height", [0])[i] or 0)
        if width <= 0 or height <= 0:
            continue

        x = int(round(left / OCR_SCALE))
        y = int(round(top / OCR_SCALE))
        w = int(round(width / OCR_SCALE))
        h = int(round(height / OCR_SCALE))
        tokens.append(
            {
                "text": text,
                "conf": conf,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": x + (w / 2.0),
                "cy": y + (h / 2.0),
            }
        )

    return tokens


def _lookup_catalog_match(set_num: str, element_id: str) -> Dict[str, Any]:
    match_payload = {
        "matched_catalog_part_num": None,
        "matched_catalog_color_id": None,
        "matched_catalog_qty": None,
    }
    if CATALOG_JOIN_AVAILABLE:
        query = """
            SELECT
                e.part_num,
                e.color_id,
                ip.quantity
            FROM elements e
            LEFT JOIN inventory_parts ip
                ON ip.part_num = e.part_num
                AND ip.color_id = e.color_id
            LEFT JOIN inventories i
                ON i.id = ip.inventory_id
            WHERE e.element_id = ?
              AND (i.set_num = ? OR i.set_num IS NULL)
            ORDER BY
                CASE WHEN i.set_num = ? THEN 0 ELSE 1 END,
                ip.quantity DESC
            LIMIT 1
        """

        try:
            conn = sqlite3.connect(str(CATALOG_DB_PATH))
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    query,
                    (str(element_id), str(set_num), str(set_num)),
                ).fetchone()
            finally:
                conn.close()
        except Exception:
            row = None

        if row is not None:
            return {
                "matched_catalog_part_num": row["part_num"],
                "matched_catalog_color_id": row["color_id"],
                "matched_catalog_qty": row["quantity"],
            }

    json_map = _load_catalog_json_map(set_num)
    mapped = json_map.get(str(element_id))
    if not mapped:
        return match_payload

    return {
        "matched_catalog_part_num": mapped.get("part_num"),
        "matched_catalog_color_id": mapped.get("color_id"),
        "matched_catalog_qty": mapped.get("qty"),
    }


def _find_nearby_qty_candidates(
    element_token: Dict[str, Any],
    qty_tokens: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates: List[tuple[tuple[int, float, float], Dict[str, Any]]] = []
    for qty_token in qty_tokens:
        qty_value = _extract_qty_value(str(qty_token.get("text", "")))
        if qty_value is None:
            continue

        dy = float(element_token["y"]) - float(qty_token["y"])
        dx = abs(float(element_token["cx"]) - float(qty_token["cx"]))
        if dy < -8 or dy > 60:
            continue
        if dx > max(70.0, float(element_token["w"]) * 3.0):
            continue

        score = (
            0 if "x" in str(qty_token.get("text", "")).lower() else 1,
            abs(dy - 14.0),
            dx,
        )
        candidates.append((score, qty_token))

    candidates.sort(key=lambda item: item[0])
    output: List[Dict[str, Any]] = []
    for _, qty_token in candidates[:3]:
        output.append(
            {
                "text": str(qty_token.get("text", "")),
                "qty": _extract_qty_value(str(qty_token.get("text", ""))),
                "position": {
                    "x": int(qty_token.get("x", 0) or 0),
                    "y": int(qty_token.get("y", 0) or 0),
                    "w": int(qty_token.get("w", 0) or 0),
                    "h": int(qty_token.get("h", 0) or 0),
                },
            }
        )
    return output


def _find_nearby_qty_token(
    element_token: Dict[str, Any],
    qty_tokens: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    candidates = _find_nearby_qty_candidates(element_token, qty_tokens)
    if not candidates:
        return None
    first = candidates[0]
    position = first.get("position") or {}
    return {
        "text": first.get("text"),
        "x": int(position.get("x", 0) or 0),
        "y": int(position.get("y", 0) or 0),
        "w": int(position.get("w", 0) or 0),
        "h": int(position.get("h", 0) or 0),
    }


def _scan_inventory_page_details(
    set_num: str,
    page: int,
) -> Dict[str, Any]:
    image_path = debug_service.resolve_page_image_path(set_num, int(page))
    if image_path is None:
        return {
            "page": int(page),
            "image_path": None,
            "inventory_page_detected": False,
            "items": [],
            "ocr_candidates": [],
        }

    tokens = _ocr_inventory_tokens(str(image_path))
    element_tokens = [
        token
        for token in tokens
        if ELEMENT_ID_RE.fullmatch(str(token.get("text", "")))
    ]
    qty_tokens = [
        token
        for token in tokens
        if _extract_qty_value(str(token.get("text", ""))) is not None
    ]
    inventory_page_detected = len(element_tokens) >= INVENTORY_PAGE_MIN_ELEMENT_IDS

    items: List[Dict[str, Any]] = []
    accepted_keys = set()
    ocr_candidates: List[Dict[str, Any]] = []
    for element_token in sorted(
        element_tokens,
        key=lambda token: (int(token.get("y", 0) or 0), int(token.get("x", 0) or 0)),
    ):
        nearby_qty_candidates = _find_nearby_qty_candidates(element_token, qty_tokens)
        accepted = bool(inventory_page_detected)
        rejection_reason = None if accepted else "page_not_inventory_like"
        qty_token = nearby_qty_candidates[0] if nearby_qty_candidates else None

        candidate_row: Dict[str, Any] = {
            "page": int(page),
            "element_id": str(element_token["text"]),
            "position": {
                "x": int(element_token["x"]),
                "y": int(element_token["y"]),
                "w": int(element_token["w"]),
                "h": int(element_token["h"]),
            },
            "confidence": float(element_token.get("conf", -1.0) or -1.0),
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "nearby_qty_candidates": nearby_qty_candidates,
        }
        if qty_token is not None:
            candidate_row["matched_qty_text"] = qty_token.get("text")
        ocr_candidates.append(candidate_row)

        if not accepted:
            continue

        qty = (
            _extract_qty_value(str(qty_token.get("text", "")))
            if qty_token is not None
            else None
        )
        item: Dict[str, Any] = {
            "page": int(page),
            "element_id": str(element_token["text"]),
            "qty": qty,
            "position": {
                "x": int(element_token["x"]),
                "y": int(element_token["y"]),
                "w": int(element_token["w"]),
                "h": int(element_token["h"]),
            },
            "matched_catalog_part_num": None,
            "matched_catalog_color_id": None,
            "matched_catalog_qty": None,
        }
        item.update(_lookup_catalog_match(set_num, str(element_token["text"])))

        if qty_token is not None:
            qty_position = qty_token.get("position") or {}
            item["qty_position"] = {
                "x": int(qty_position.get("x", 0) or 0),
                "y": int(qty_position.get("y", 0) or 0),
                "w": int(qty_position.get("w", 0) or 0),
                "h": int(qty_position.get("h", 0) or 0),
            }

        items.append(item)
        accepted_keys.add(
            (
                str(element_token["text"]),
                int(element_token["x"]),
                int(element_token["y"]),
            )
        )

    return {
        "page": int(page),
        "image_path": str(image_path),
        "inventory_page_detected": bool(inventory_page_detected),
        "items": items,
        "ocr_candidates": ocr_candidates,
    }


def extract_inventory_items_from_page(
    set_num: str,
    page: int,
) -> List[Dict[str, Any]]:
    return list(_scan_inventory_page_details(set_num, int(page)).get("items", []) or [])


def scan_instruction_inventory(
    set_num: str,
    start: int,
    end: int,
) -> Dict[str, Any]:
    if int(end) < int(start):
        raise RuntimeError("end must be >= start")

    pages_scanned = 0
    inventory_pages: List[int] = []
    items: List[Dict[str, Any]] = []
    ocr_candidates: List[Dict[str, Any]] = []

    for page in range(int(start), int(end) + 1):
        page_details = _scan_inventory_page_details(set_num, int(page))
        page_items = list(page_details.get("items", []) or [])
        pages_scanned += 1
        ocr_candidates.extend(list(page_details.get("ocr_candidates", []) or []))
        if not page_items:
            continue
        inventory_pages.append(int(page))
        items.extend(page_items)

    return {
        "set_num": str(set_num),
        "start": int(start),
        "end": int(end),
        "pages_scanned": int(pages_scanned),
        "inventory_pages": inventory_pages,
        "catalog_join_available": bool(CATALOG_JOIN_AVAILABLE),
        "catalog_json_fallback_available": bool(_load_catalog_json_map(set_num)),
        "ocr_candidates": ocr_candidates,
        "items": items,
    }
