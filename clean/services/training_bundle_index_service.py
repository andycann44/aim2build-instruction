from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from clean.services.training_store_service import _REPO_ROOT


_DATABASE_URL_KEY = "AZURE_DATABASE_URL"


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


def _database_url() -> str:
    env_value = str(os.environ.get(_DATABASE_URL_KEY) or "").strip()
    if env_value:
        return env_value
    file_value = str(_read_local_env_file().get(_DATABASE_URL_KEY) or "").strip()
    if file_value:
        return file_value
    raise ValueError(f"{_DATABASE_URL_KEY} is not configured")


def _connect():
    return psycopg.connect(_database_url(), row_factory=dict_row)


def ensure_schema() -> Dict[str, Any]:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS training_bundle_index (
                    id SERIAL PRIMARY KEY,
                    bundle_id TEXT UNIQUE NOT NULL,
                    set_num TEXT,
                    bag_num INTEGER,
                    page_num INTEGER,
                    step_num INTEGER,
                    crop_num INTEGER,
                    slot_count INTEGER,
                    approved BOOLEAN DEFAULT FALSE,
                    r2_prefix TEXT,
                    manifest_path TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                ALTER TABLE training_bundle_index
                    ADD COLUMN IF NOT EXISTS review_status TEXT DEFAULT 'pending',
                    ADD COLUMN IF NOT EXISTS review_notes TEXT DEFAULT '',
                    ADD COLUMN IF NOT EXISTS mask_quality INTEGER,
                    ADD COLUMN IF NOT EXISTS split_quality INTEGER,
                    ADD COLUMN IF NOT EXISTS qty_text_present BOOLEAN DEFAULT FALSE,
                    ADD COLUMN IF NOT EXISTS multi_part_merge BOOLEAN DEFAULT FALSE,
                    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP,
                    ADD COLUMN IF NOT EXISTS reviewed_by TEXT DEFAULT '',
                    ADD COLUMN IF NOT EXISTS ai_analysis_json JSONB,
                    ADD COLUMN IF NOT EXISTS ai_reviewed_at TIMESTAMP,
                    ADD COLUMN IF NOT EXISTS ai_model TEXT DEFAULT '',
                    ADD COLUMN IF NOT EXISTS split_candidate_count INTEGER DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS split_candidate_paths JSONB;
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_training_examples (
                    id SERIAL PRIMARY KEY,
                    bundle_id TEXT NOT NULL,
                    candidate_index INTEGER NOT NULL,
                    part_num TEXT,
                    color_id INTEGER,
                    element_id TEXT,
                    qty INTEGER,
                    thumbnail_path TEXT,
                    r2_path TEXT,
                    confirmed_by TEXT,
                    confirmed_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE (bundle_id, candidate_index)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bag_review_exports (
                    id SERIAL PRIMARY KEY,
                    set_num TEXT NOT NULL,
                    bag_num INTEGER NOT NULL,
                    schema_version TEXT,
                    exported_at TIMESTAMP,
                    manifest_path TEXT,
                    r2_prefix TEXT,
                    progress_json JSONB,
                    metadata_json JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE (set_num, bag_num)
                );
                """
            )
        conn.commit()
    return {
        "ok": True,
        "tables": ["training_bundle_index", "candidate_training_examples", "bag_review_exports"],
    }


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _crop_num_from_crop_id(crop_id: str) -> Optional[int]:
    match = re.search(r"(?:^|_)c(\d+)(?:_|$)", str(crop_id or ""))
    return int(match.group(1)) if match else None


def _metadata_from_manifest_path(manifest_path: str) -> Dict[str, Any]:
    path = Path(str(manifest_path or "").strip())
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _ids_from_bundle_id(bundle_id: str) -> Dict[str, Any]:
    text = str(bundle_id or "")
    match = re.search(r"^(?P<set>[^_]+)_bag(?P<bag>\d+)_p(?P<page>\d+)_s(?P<step>\d+)_c(?P<crop>\d+)", text)
    if not match:
        return {"set_num": "", "bag_num": None, "page_num": None, "step_num": None, "crop_num": None}
    return {
        "set_num": str(match.group("set") or ""),
        "bag_num": _coerce_int(match.group("bag")),
        "page_num": _coerce_int(match.group("page")),
        "step_num": _coerce_int(match.group("step")),
        "crop_num": _coerce_int(match.group("crop")),
    }


def _json_safe_row(row: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, value in dict(row or {}).items():
        if hasattr(value, "isoformat"):
            safe[key] = value.isoformat()
        else:
            safe[key] = value
    return safe


def _row_with_backfilled_ids(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row or {})
    parsed = _ids_from_bundle_id(str(out.get("bundle_id") or ""))
    if not str(out.get("set_num") or "").strip() and parsed.get("set_num"):
        out["set_num"] = parsed.get("set_num")
    for key in ("bag_num", "page_num", "step_num", "crop_num"):
        if _coerce_int(out.get(key)) is None and parsed.get(key) is not None:
            out[key] = parsed.get(key)
    return out


def register_bundle(
    bundle_entry: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    r2_prefix: str = "",
    manifest_path: str = "",
) -> Dict[str, Any]:
    ensure_schema()
    artifact_paths = bundle_entry.get("artifact_paths") if isinstance(bundle_entry.get("artifact_paths"), dict) else {}
    resolved_manifest_path = str(manifest_path or artifact_paths.get("metadata") or "")
    metadata = metadata or _metadata_from_manifest_path(resolved_manifest_path)
    source_crop = metadata.get("source_crop") if isinstance(metadata.get("source_crop"), dict) else {}
    slot_assignments = metadata.get("slot_assignments") if isinstance(metadata.get("slot_assignments"), list) else []
    cutout_paths = metadata.get("cutout_paths") if isinstance(metadata.get("cutout_paths"), list) else []
    copied_files = metadata.get("copied_files") if isinstance(metadata.get("copied_files"), dict) else {}
    copied_slot_cutouts = copied_files.get("slot_cutouts") if isinstance(copied_files.get("slot_cutouts"), list) else []

    bundle_id = str(bundle_entry.get("bundle_id") or "").strip()
    if not bundle_id:
        raise ValueError("bundle_id is required")
    crop_id = str(bundle_entry.get("crop_id") or metadata.get("crop_id") or "").strip()
    parsed_ids = _ids_from_bundle_id(bundle_id)
    resolved_manifest_path = str(resolved_manifest_path or metadata.get("metadata_path") or "")
    row = {
        "bundle_id": bundle_id,
        "set_num": str(bundle_entry.get("set_num") or metadata.get("set_num") or parsed_ids.get("set_num") or ""),
        "bag_num": _coerce_int(bundle_entry.get("bag")) or _coerce_int(metadata.get("bag")) or parsed_ids.get("bag_num"),
        "page_num": _coerce_int(source_crop.get("page")) or parsed_ids.get("page_num"),
        "step_num": _coerce_int(source_crop.get("step")) or parsed_ids.get("step_num"),
        "crop_num": _crop_num_from_crop_id(crop_id) or parsed_ids.get("crop_num"),
        "slot_count": len(slot_assignments) if slot_assignments else (len(cutout_paths) if cutout_paths else (len(copied_slot_cutouts) if copied_slot_cutouts else None)),
        "approved": str(bundle_entry.get("review_status") or "") == "approved",
        "r2_prefix": str(r2_prefix or ""),
        "manifest_path": resolved_manifest_path,
    }

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_bundle_index (
                    bundle_id, set_num, bag_num, page_num, step_num, crop_num,
                    slot_count, approved, r2_prefix, manifest_path
                ) VALUES (
                    %(bundle_id)s, %(set_num)s, %(bag_num)s, %(page_num)s, %(step_num)s, %(crop_num)s,
                    %(slot_count)s, %(approved)s, %(r2_prefix)s, %(manifest_path)s
                )
                ON CONFLICT (bundle_id) DO UPDATE SET
                    set_num = EXCLUDED.set_num,
                    bag_num = EXCLUDED.bag_num,
                    page_num = EXCLUDED.page_num,
                    step_num = EXCLUDED.step_num,
                    crop_num = EXCLUDED.crop_num,
                    slot_count = EXCLUDED.slot_count,
                    approved = EXCLUDED.approved,
                    r2_prefix = EXCLUDED.r2_prefix,
                    manifest_path = EXCLUDED.manifest_path,
                    review_status = COALESCE(NULLIF(training_bundle_index.review_status, ''), 'pending'),
                    updated_at = NOW()
                RETURNING *;
                """,
                row,
            )
            result = cur.fetchone()
        conn.commit()
    return {"ok": True, "row": _json_safe_row(dict(result or {}))}


def get_bundle(bundle_id: str) -> Dict[str, Any]:
    ensure_schema()
    bundle_id = str(bundle_id or "").strip()
    if not bundle_id:
        raise ValueError("bundle_id is required")
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM training_bundle_index WHERE bundle_id = %s;",
                (bundle_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise FileNotFoundError(f"bundle_id not found: {bundle_id}")
    backfilled = _row_with_backfilled_ids(dict(row))
    if (
        backfilled.get("set_num") != row.get("set_num")
        or backfilled.get("bag_num") != row.get("bag_num")
        or backfilled.get("page_num") != row.get("page_num")
        or backfilled.get("step_num") != row.get("step_num")
        or backfilled.get("crop_num") != row.get("crop_num")
    ):
        _backfill_bundle_ids(str(backfilled.get("bundle_id") or bundle_id), backfilled)
    return {"ok": True, "row": _json_safe_row(backfilled)}


def _backfill_bundle_ids(bundle_id: str, row: Dict[str, Any]) -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_bundle_index
                SET
                    set_num = COALESCE(NULLIF(set_num, ''), %(set_num)s),
                    bag_num = COALESCE(bag_num, %(bag_num)s),
                    page_num = COALESCE(page_num, %(page_num)s),
                    step_num = COALESCE(step_num, %(step_num)s),
                    crop_num = COALESCE(crop_num, %(crop_num)s),
                    updated_at = NOW()
                WHERE bundle_id = %(bundle_id)s;
                """,
                {
                    "bundle_id": bundle_id,
                    "set_num": str(row.get("set_num") or ""),
                    "bag_num": _coerce_int(row.get("bag_num")),
                    "page_num": _coerce_int(row.get("page_num")),
                    "step_num": _coerce_int(row.get("step_num")),
                    "crop_num": _coerce_int(row.get("crop_num")),
                },
            )
        conn.commit()


def _coerce_quality(value: Any, field_name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    quality = _coerce_int(value)
    if quality is None or quality < 1 or quality > 5:
        raise ValueError(f"{field_name} must be between 1 and 5")
    return quality


def update_review(
    bundle_id: str,
    *,
    review_status: str,
    review_notes: str = "",
    mask_quality: Any = None,
    split_quality: Any = None,
    qty_text_present: bool = False,
    multi_part_merge: bool = False,
    reviewed_by: str = "",
) -> Dict[str, Any]:
    ensure_schema()
    bundle_id = str(bundle_id or "").strip()
    if not bundle_id:
        raise ValueError("bundle_id is required")
    status = str(review_status or "").strip()
    if status not in {"approved", "rejected", "needs_split_fix", "bad_mask"}:
        raise ValueError("review_status must be approved, rejected, needs_split_fix, or bad_mask")

    row = {
        "bundle_id": bundle_id,
        "review_status": status,
        "review_notes": str(review_notes or ""),
        "mask_quality": _coerce_quality(mask_quality, "mask_quality"),
        "split_quality": _coerce_quality(split_quality, "split_quality"),
        "qty_text_present": bool(qty_text_present),
        "multi_part_merge": bool(multi_part_merge),
        "reviewed_by": str(reviewed_by or ""),
    }
    parsed_ids = _ids_from_bundle_id(bundle_id)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_bundle_index
                SET
                    review_status = %(review_status)s,
                    review_notes = %(review_notes)s,
                    mask_quality = %(mask_quality)s,
                    split_quality = %(split_quality)s,
                    qty_text_present = %(qty_text_present)s,
                    multi_part_merge = %(multi_part_merge)s,
                    reviewed_by = %(reviewed_by)s,
                    set_num = COALESCE(NULLIF(set_num, ''), %(set_num)s),
                    bag_num = COALESCE(bag_num, %(bag_num)s),
                    page_num = COALESCE(page_num, %(page_num)s),
                    step_num = COALESCE(step_num, %(step_num)s),
                    crop_num = COALESCE(crop_num, %(crop_num)s),
                    reviewed_at = NOW(),
                    updated_at = NOW()
                WHERE bundle_id = %(bundle_id)s
                RETURNING *;
                """,
                {
                    **row,
                    "set_num": str(parsed_ids.get("set_num") or ""),
                    "bag_num": parsed_ids.get("bag_num"),
                    "page_num": parsed_ids.get("page_num"),
                    "step_num": parsed_ids.get("step_num"),
                    "crop_num": parsed_ids.get("crop_num"),
                },
            )
            result = cur.fetchone()
        conn.commit()
    if result is None:
        raise FileNotFoundError(f"bundle_id not found: {bundle_id}")
    return {"ok": True, "row": _json_safe_row(dict(result))}


def list_review_queue(
    *,
    review_status: str = "",
    set_num: str = "",
    bag_num: Any = None,
    limit: Any = 100,
) -> Dict[str, Any]:
    ensure_schema()
    clauses: List[str] = []
    params: Dict[str, Any] = {}
    status = str(review_status or "").strip()
    if status:
        clauses.append("review_status = %(review_status)s")
        params["review_status"] = status
    set_filter = str(set_num or "").strip()
    if set_filter:
        clauses.append("set_num = %(set_num)s")
        params["set_num"] = set_filter
    bag_filter = _coerce_int(bag_num)
    if bag_num not in {None, ""} and bag_filter is not None:
        clauses.append("bag_num = %(bag_num)s")
        params["bag_num"] = bag_filter
    limit_value = _coerce_int(limit) or 100
    limit_value = max(1, min(limit_value, 500))
    params["limit"] = limit_value
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT *
                FROM training_bundle_index
                {where_sql}
                ORDER BY created_at ASC, bundle_id ASC
                LIMIT %(limit)s;
                """,
                params,
            )
            rows = cur.fetchall()
    backfilled_rows: List[Dict[str, Any]] = []
    for raw_row in rows:
        original = dict(raw_row)
        row = _row_with_backfilled_ids(original)
        backfilled_rows.append(row)
        if (
            row.get("set_num") != original.get("set_num")
            or row.get("bag_num") != original.get("bag_num")
            or row.get("page_num") != original.get("page_num")
            or row.get("step_num") != original.get("step_num")
            or row.get("crop_num") != original.get("crop_num")
        ):
            _backfill_bundle_ids(str(row.get("bundle_id") or ""), row)
    return {
        "ok": True,
        "count": len(backfilled_rows),
        "rows": [_json_safe_row(row) for row in backfilled_rows],
        "filters": {
            "review_status": status,
            "set_num": set_filter,
            "bag_num": bag_filter,
            "limit": limit_value,
        },
    }


def get_review_stats() -> Dict[str, Any]:
    ensure_schema()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE COALESCE(review_status, 'pending') = 'pending') AS pending,
                    COUNT(*) FILTER (WHERE review_status = 'approved') AS approved,
                    COUNT(*) FILTER (WHERE review_status = 'rejected') AS rejected,
                    COUNT(*) FILTER (WHERE review_status = 'bad_mask') AS bad_mask,
                    COUNT(*) FILTER (WHERE review_status = 'needs_split_fix') AS needs_split_fix
                FROM training_bundle_index;
                """
            )
            row = cur.fetchone() or {}
    return {"ok": True, "stats": {key: int((row or {}).get(key, 0) or 0) for key in ["pending", "approved", "rejected", "bad_mask", "needs_split_fix"]}}


def update_ai_analysis(
    bundle_id: str,
    *,
    ai_analysis_json: Dict[str, Any],
    ai_model: str,
) -> Dict[str, Any]:
    ensure_schema()
    bundle_id = str(bundle_id or "").strip()
    if not bundle_id:
        raise ValueError("bundle_id is required")
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_bundle_index
                SET
                    ai_analysis_json = %s::jsonb,
                    ai_reviewed_at = NOW(),
                    ai_model = %s,
                    updated_at = NOW()
                WHERE bundle_id = %s
                RETURNING *;
                """,
                (json.dumps(ai_analysis_json, ensure_ascii=True), str(ai_model or ""), bundle_id),
            )
            row = cur.fetchone()
        conn.commit()
    if row is None:
        raise FileNotFoundError(f"bundle_id not found: {bundle_id}")
    return {"ok": True, "row": _json_safe_row(dict(row))}


def update_split_candidates(
    bundle_id: str,
    *,
    split_candidate_paths: Dict[str, Any],
) -> Dict[str, Any]:
    ensure_schema()
    bundle_id = str(bundle_id or "").strip()
    if not bundle_id:
        raise ValueError("bundle_id is required")
    candidates = split_candidate_paths.get("candidates") if isinstance(split_candidate_paths.get("candidates"), list) else []
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_bundle_index
                SET
                    split_candidate_count = %s,
                    split_candidate_paths = %s::jsonb,
                    updated_at = NOW()
                WHERE bundle_id = %s
                RETURNING *;
                """,
                (
                    len(candidates),
                    json.dumps(split_candidate_paths, ensure_ascii=True),
                    bundle_id,
                ),
            )
            row = cur.fetchone()
        conn.commit()
    if row is None:
        raise FileNotFoundError(f"bundle_id not found: {bundle_id}")
    return {"ok": True, "row": _json_safe_row(dict(row))}


def update_split_candidate_status(
    bundle_id: str,
    *,
    candidate_index: int,
    status: str,
    v2_mask_path: str = "",
) -> Dict[str, Any]:
    ensure_schema()
    bundle_id = str(bundle_id or "").strip()
    if not bundle_id:
        raise ValueError("bundle_id is required")
    if status not in {"accepted", "rejected"}:
        raise ValueError("status must be accepted or rejected")
    if candidate_index < 0:
        raise ValueError("candidate_index must be >= 0")
    row = dict(get_bundle(bundle_id).get("row") or {})
    paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
    candidates = list(paths.get("candidates") or [])
    target_pos = -1
    for pos, raw_candidate in enumerate(candidates):
        candidate = raw_candidate if isinstance(raw_candidate, dict) else {}
        if _coerce_int(candidate.get("index")) == int(candidate_index):
            target_pos = pos
            break
    if target_pos < 0 and candidate_index < len(candidates):
        target_pos = candidate_index
    if target_pos < 0 or target_pos >= len(candidates):
        raise ValueError("candidate_index is out of range")
    candidate = dict(candidates[target_pos]) if isinstance(candidates[target_pos], dict) else {}
    candidate["status"] = status
    if status == "accepted":
        candidate["review_state"] = ""
    if v2_mask_path:
        candidate["v2_mask_path"] = str(v2_mask_path)
    candidates[target_pos] = candidate
    def update_group(raw_items: Any) -> List[Dict[str, Any]]:
        updated: List[Dict[str, Any]] = []
        for raw_item in list(raw_items or []):
            item = dict(raw_item) if isinstance(raw_item, dict) else {}
            if _coerce_int(item.get("index")) == int(candidate_index):
                item.update(candidate)
            updated.append(item)
        return updated

    paths["candidates"] = candidates
    paths["baseline_slot_candidates"] = update_group(paths.get("baseline_slot_candidates"))
    paths["ai_suggested_candidates"] = update_group(paths.get("ai_suggested_candidates"))
    return update_split_candidates(bundle_id, split_candidate_paths=paths)


def auto_promote_bundle_review_status(bundle_id: str) -> Dict[str, Any]:
    """Promote bundle review_status to 'approved' when all candidates have been resolved,
    but only if the bundle is not already in a terminal state (approved / rejected).

    This is a targeted update — only the review_status column is touched.
    Existing review_notes, quality scores, and other review metadata are preserved.

    Returns:
        {"ok": True, "promoted": True}  — status was advanced to 'approved'
        {"ok": True, "promoted": False} — already approved/rejected, no change made
    """
    ensure_schema()
    bundle_id = str(bundle_id or "").strip()
    if not bundle_id:
        return {"ok": False, "promoted": False, "error": "bundle_id is required"}
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_bundle_index
                SET review_status = 'approved', updated_at = NOW()
                WHERE bundle_id = %s
                  AND COALESCE(review_status, '') NOT IN ('approved', 'rejected')
                RETURNING bundle_id;
                """,
                (bundle_id,),
            )
            row = cur.fetchone()
        conn.commit()
    promoted = row is not None
    return {"ok": True, "promoted": promoted}


def confirm_candidate_part(
    *,
    bundle_id: str,
    candidate_index: Any,
    part_num: str,
    color_id: Any,
    element_id: str = "",
    qty: Any = None,
    thumbnail_path: str = "",
    r2_path: str = "",
    confirmed_by: str = "",
) -> Dict[str, Any]:
    ensure_schema()
    bundle_id = str(bundle_id or "").strip()
    if not bundle_id:
        raise ValueError("bundle_id is required")
    parsed_candidate_index = _coerce_int(candidate_index)
    if parsed_candidate_index is None or parsed_candidate_index < 0:
        raise ValueError("candidate_index must be >= 0")
    resolved_part_num = str(part_num or "").strip()
    if not resolved_part_num:
        raise ValueError("part_num is required")
    parsed_color_id = _coerce_int(color_id)
    if parsed_color_id is None:
        raise ValueError("color_id is required")
    parsed_qty = _coerce_int(qty)

    row = {
        "bundle_id": bundle_id,
        "candidate_index": parsed_candidate_index,
        "part_num": resolved_part_num,
        "color_id": parsed_color_id,
        "element_id": str(element_id or "").strip(),
        "qty": parsed_qty,
        "thumbnail_path": str(thumbnail_path or "").strip(),
        "r2_path": str(r2_path or "").strip(),
        "confirmed_by": str(confirmed_by or "").strip(),
    }
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO candidate_training_examples (
                    bundle_id, candidate_index, part_num, color_id, element_id,
                    qty, thumbnail_path, r2_path, confirmed_by, confirmed_at
                ) VALUES (
                    %(bundle_id)s, %(candidate_index)s, %(part_num)s, %(color_id)s, %(element_id)s,
                    %(qty)s, %(thumbnail_path)s, %(r2_path)s, %(confirmed_by)s, NOW()
                )
                ON CONFLICT (bundle_id, candidate_index) DO UPDATE SET
                    part_num = EXCLUDED.part_num,
                    color_id = EXCLUDED.color_id,
                    element_id = EXCLUDED.element_id,
                    qty = EXCLUDED.qty,
                    thumbnail_path = EXCLUDED.thumbnail_path,
                    r2_path = EXCLUDED.r2_path,
                    confirmed_by = EXCLUDED.confirmed_by,
                    confirmed_at = NOW()
                RETURNING *;
                """,
                row,
            )
            result = cur.fetchone()
        conn.commit()
    return {"ok": True, "row": _json_safe_row(dict(result or {}))}


def list_candidate_training_examples(bundle_id: str) -> Dict[str, Any]:
    ensure_schema()
    bundle_id = str(bundle_id or "").strip()
    if not bundle_id:
        raise ValueError("bundle_id is required")
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM candidate_training_examples
                WHERE bundle_id = %s
                ORDER BY candidate_index ASC, id ASC;
                """,
                (bundle_id,),
            )
            rows = cur.fetchall()
    return {
        "ok": True,
        "bundle_id": bundle_id,
        "rows": [_json_safe_row(dict(row or {})) for row in list(rows or [])],
    }


def list_confirmed_part_usage(
    *,
    set_num: str,
    part_num: str,
    color_id: Any,
    element_id: str = "",
    required_qty: Any = None,
) -> Dict[str, Any]:
    ensure_schema()
    resolved_set_num = str(set_num or "").strip()
    if not resolved_set_num:
        raise ValueError("set_num is required")
    resolved_part_num = str(part_num or "").strip()
    if not resolved_part_num:
        raise ValueError("part_num is required")
    parsed_color_id = _coerce_int(color_id)
    if parsed_color_id is None:
        raise ValueError("color_id is required")
    resolved_element_id = str(element_id or "").strip()

    params: Dict[str, Any] = {
        "set_num": resolved_set_num,
        "part_num": resolved_part_num,
        "color_id": parsed_color_id,
    }
    element_clause = ""
    if resolved_element_id:
        element_clause = "AND COALESCE(e.element_id, '') = %(element_id)s"
        params["element_id"] = resolved_element_id

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    e.bundle_id,
                    e.candidate_index,
                    e.part_num,
                    e.color_id,
                    e.element_id,
                    e.qty,
                    e.confirmed_by,
                    e.confirmed_at AS created_at,
                    e.thumbnail_path,
                    e.r2_path,
                    b.set_num,
                    b.bag_num,
                    b.page_num,
                    b.step_num,
                    b.crop_num,
                    b.review_status
                FROM candidate_training_examples e
                JOIN training_bundle_index b ON b.bundle_id = e.bundle_id
                WHERE b.set_num = %(set_num)s
                  AND e.part_num = %(part_num)s
                  AND e.color_id = %(color_id)s
                  {element_clause}
                ORDER BY e.confirmed_at ASC, e.bundle_id ASC, e.candidate_index ASC;
                """,
                params,
            )
            rows = cur.fetchall()

    safe_rows = [_json_safe_row(dict(row or {})) for row in list(rows or [])]
    total_confirmed_qty = 0
    for row in safe_rows:
        row_qty = _coerce_int(row.get("qty"))
        total_confirmed_qty += int(row_qty) if row_qty is not None and int(row_qty) > 0 else 1
    parsed_required_qty = _coerce_int(required_qty)
    effective_remaining_qty = (
        max(0, int(parsed_required_qty) - int(total_confirmed_qty))
        if parsed_required_qty is not None
        else None
    )
    over_confirmed_by = (
        max(0, int(total_confirmed_qty) - int(parsed_required_qty))
        if parsed_required_qty is not None
        else None
    )
    return {
        "ok": True,
        "set_num": resolved_set_num,
        "part_num": resolved_part_num,
        "color_id": parsed_color_id,
        "element_id": resolved_element_id,
        "required_qty": parsed_required_qty,
        "confirmed_qty": total_confirmed_qty,
        "total_confirmed_qty": total_confirmed_qty,
        "effective_remaining_qty": effective_remaining_qty,
        "over_confirmed_by": over_confirmed_by,
        "rows": safe_rows,
        "count": len(safe_rows),
    }


def list_confirmed_part_totals_for_set(set_num: str) -> Dict[str, Any]:
    ensure_schema()
    resolved_set_num = str(set_num or "").strip()
    if not resolved_set_num:
        raise ValueError("set_num is required")
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.part_num,
                    e.color_id,
                    SUM(CASE WHEN COALESCE(e.qty, 0) > 0 THEN e.qty ELSE 1 END) AS confirmed_qty,
                    COUNT(*) AS confirmation_count
                FROM candidate_training_examples e
                JOIN training_bundle_index b ON b.bundle_id = e.bundle_id
                WHERE b.set_num = %s
                GROUP BY e.part_num, e.color_id
                ORDER BY e.part_num ASC, e.color_id ASC;
                """,
                (resolved_set_num,),
            )
            rows = cur.fetchall()
    return {
        "ok": True,
        "set_num": resolved_set_num,
        "rows": [_json_safe_row(dict(row or {})) for row in list(rows or [])],
    }


def unconfirm_candidate_part(*, bundle_id: str, candidate_index: Any) -> Dict[str, Any]:
    ensure_schema()
    bundle_id = str(bundle_id or "").strip()
    if not bundle_id:
        raise ValueError("bundle_id is required")
    parsed_candidate_index = _coerce_int(candidate_index)
    if parsed_candidate_index is None or parsed_candidate_index < 0:
        raise ValueError("candidate_index must be >= 0")
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM candidate_training_examples
                WHERE bundle_id = %s AND candidate_index = %s
                RETURNING *;
                """,
                (bundle_id, parsed_candidate_index),
            )
            row = cur.fetchone()
        conn.commit()
    return {
        "ok": True,
        "bundle_id": bundle_id,
        "candidate_index": parsed_candidate_index,
        "removed": bool(row),
        "row": _json_safe_row(dict(row or {})) if row else {},
    }


def reset_bag_index_rows(*, set_num: str, bag_num: Any) -> Dict[str, Any]:
    ensure_schema()
    safe_set_num = str(set_num or "").strip()
    parsed_bag_num = _coerce_int(bag_num)
    if not safe_set_num:
        raise ValueError("set_num is required")
    if parsed_bag_num is None:
        raise ValueError("bag_num is required")
    bundle_prefix = f"{safe_set_num}_bag{parsed_bag_num}_"
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM candidate_training_examples
                WHERE bundle_id LIKE %s
                RETURNING bundle_id;
                """,
                (bundle_prefix + "%",),
            )
            candidate_rows = cur.fetchall()
            cur.execute(
                """
                DELETE FROM training_bundle_index
                WHERE set_num = %s AND bag_num = %s
                RETURNING bundle_id;
                """,
                (safe_set_num, parsed_bag_num),
            )
            index_rows = cur.fetchall()
        conn.commit()
    return {
        "ok": True,
        "set_num": safe_set_num,
        "bag_num": parsed_bag_num,
        "bundle_prefix": bundle_prefix,
        "candidate_training_examples_deleted_count": len(candidate_rows or []),
        "candidate_training_example_bundle_ids": sorted({str(row.get("bundle_id") or "") for row in list(candidate_rows or [])}),
        "training_bundle_index_deleted_count": len(index_rows or []),
        "training_bundle_index_bundle_ids": sorted(str(row.get("bundle_id") or "") for row in list(index_rows or [])),
    }


def upsert_bag_review_export(
    *,
    set_num: str,
    bag_num: int,
    metadata: Dict[str, Any],
    manifest_path: str,
) -> Dict[str, Any]:
    """Persist reviewed bag metadata to Azure-hosted Postgres."""
    ensure_schema()
    set_text = str(set_num or "").strip()
    bag_number = int(bag_num or 0)
    if not set_text:
        raise ValueError("set_num is required")
    if bag_number < 1:
        raise ValueError("bag_num must be >= 1")

    progress = metadata.get("progress") if isinstance(metadata.get("progress"), dict) else {}
    row = {
        "set_num": set_text,
        "bag_num": bag_number,
        "schema_version": str(metadata.get("schema_version") or ""),
        "exported_at": str(metadata.get("exported_at") or ""),
        "manifest_path": str(manifest_path or ""),
        "r2_prefix": str(metadata.get("r2_prefix") or ""),
        "progress_json": json.dumps(progress),
        "metadata_json": json.dumps(metadata),
    }

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bag_review_exports (
                    set_num, bag_num, schema_version, exported_at,
                    manifest_path, r2_prefix, progress_json, metadata_json
                ) VALUES (
                    %(set_num)s, %(bag_num)s, %(schema_version)s, %(exported_at)s,
                    %(manifest_path)s, %(r2_prefix)s, %(progress_json)s::jsonb, %(metadata_json)s::jsonb
                )
                ON CONFLICT (set_num, bag_num) DO UPDATE SET
                    schema_version = EXCLUDED.schema_version,
                    exported_at = EXCLUDED.exported_at,
                    manifest_path = EXCLUDED.manifest_path,
                    r2_prefix = EXCLUDED.r2_prefix,
                    progress_json = EXCLUDED.progress_json,
                    metadata_json = EXCLUDED.metadata_json,
                    updated_at = NOW()
                RETURNING *;
                """,
                row,
            )
            result = cur.fetchone()
        conn.commit()
    return {"ok": True, "row": _json_safe_row(dict(result or {}))}
