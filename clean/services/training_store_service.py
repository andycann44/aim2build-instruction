from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANALYSIS_BUNDLES_DIR = _REPO_ROOT / "debug" / "ai_training" / "analysis_bundles"
_TRAINING_STORE_DIR = _REPO_ROOT / "debug" / "ai_training" / "training_store"
_TRAINING_STORE_INDEX = _TRAINING_STORE_DIR / "index.json"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_bundle_id(bundle_id: str) -> str:
    value = str(bundle_id or "").strip()
    if not value:
        raise ValueError("bundle_id is required")
    if any(part in {"", ".", ".."} for part in Path(value).parts):
        raise ValueError("invalid bundle_id")
    safe = "".join(ch for ch in value if ch.isalnum() or ch in "-_." )
    if safe != value:
        raise ValueError("invalid bundle_id")
    return value


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(str(tmp_path), str(path))


def _artifact_paths_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    copied_files = metadata.get("copied_files") if isinstance(metadata.get("copied_files"), dict) else {}
    source_artifacts = metadata.get("source_artifacts") if isinstance(metadata.get("source_artifacts"), dict) else {}
    return {
        "original_crop": str(copied_files.get("original_crop") or ""),
        "full_mask_overlay": str(copied_files.get("full_mask_overlay") or ""),
        "raw_master_mask": str(copied_files.get("raw_master_mask") or ""),
        "master_island_overlay": str(copied_files.get("master_island_overlay") or ""),
        "slot_cutouts": [
            str(item)
            for item in list(copied_files.get("slot_cutouts") or [])
            if str(item or "").strip()
        ],
        "metadata": str(metadata.get("metadata_path") or ""),
        "source_artifacts": source_artifacts,
    }


def _normalize_slot_indexes(slot_indexes: Optional[List[int]]) -> List[int]:
    normalized: List[int] = []
    for value in list(slot_indexes or []):
        try:
            slot_index = int(value)
        except Exception:
            continue
        if slot_index < 0 or slot_index in normalized:
            continue
        normalized.append(slot_index)
    normalized.sort()
    return normalized


def _review_defaults(existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    existing = existing or {}
    return {
        "review_status": str(existing.get("review_status") or "unreviewed"),
        "approved_slots": _normalize_slot_indexes(existing.get("approved_slots") if isinstance(existing.get("approved_slots"), list) else []),
        "rejected_slots": _normalize_slot_indexes(existing.get("rejected_slots") if isinstance(existing.get("rejected_slots"), list) else []),
        "notes": str(existing.get("notes") or ""),
        "reviewed_at": str(existing.get("reviewed_at") or ""),
        "reviewer": str(existing.get("reviewer") or ""),
    }


def register_analysis_bundle(bundle_id: str) -> Dict[str, Any]:
    safe_bundle_id = _safe_bundle_id(bundle_id)
    bundle_path = (_ANALYSIS_BUNDLES_DIR / safe_bundle_id).resolve()
    allowed_root = _ANALYSIS_BUNDLES_DIR.resolve()
    if allowed_root not in bundle_path.parents:
        raise ValueError("bundle path escapes analysis bundle directory")

    metadata_path = bundle_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found for bundle_id={safe_bundle_id}")
    metadata = _read_json_file(metadata_path)
    if not metadata:
        raise ValueError(f"metadata.json is empty or invalid for bundle_id={safe_bundle_id}")

    created_at = str(metadata.get("generated_at") or _utc_timestamp())
    entry = {
        "bundle_id": safe_bundle_id,
        "set_num": str(metadata.get("set_num") or ""),
        "bag": int(metadata.get("bag", 0) or 0),
        "crop_id": str(metadata.get("crop_id") or ""),
        "created_at": created_at,
        "local_bundle_path": str(bundle_path),
        "artifact_paths": _artifact_paths_from_metadata(metadata),
        "review_status": "unreviewed",
        "cloud_status": "local_only",
    }

    index = _read_json_file(_TRAINING_STORE_INDEX)
    entries = index.get("entries") if isinstance(index.get("entries"), list) else []
    existing_entry = next(
        (
            dict(item)
            for item in entries
            if isinstance(item, dict) and str(item.get("bundle_id") or "") == safe_bundle_id
        ),
        {},
    )
    updated_entries = [
        dict(item)
        for item in entries
        if isinstance(item, dict) and str(item.get("bundle_id") or "") != safe_bundle_id
    ]
    entry.update(_review_defaults(existing_entry))
    updated_entries.append(entry)
    updated_entries.sort(key=lambda item: str(item.get("bundle_id") or ""))

    new_index = {
        "schema_version": "1.0",
        "updated_at": _utc_timestamp(),
        "entries": updated_entries,
    }
    _write_json_atomic(_TRAINING_STORE_INDEX, new_index)

    return {
        "ok": True,
        "entry": entry,
        "index_path": str(_TRAINING_STORE_INDEX),
        "entry_count": len(updated_entries),
    }


def update_bundle_review(
    bundle_id: str,
    review_status: str,
    *,
    slot_indexes: Optional[List[int]] = None,
    notes: str = "",
    reviewer: str = "",
) -> Dict[str, Any]:
    safe_bundle_id = _safe_bundle_id(bundle_id)
    if review_status not in {"approved", "rejected"}:
        raise ValueError("review_status must be approved or rejected")

    index = _read_json_file(_TRAINING_STORE_INDEX)
    entries = index.get("entries") if isinstance(index.get("entries"), list) else []
    updated_entries: List[Dict[str, Any]] = []
    target_entry: Optional[Dict[str, Any]] = None
    normalized_slots = _normalize_slot_indexes(slot_indexes)

    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        if str(entry.get("bundle_id") or "") == safe_bundle_id:
            entry.update(_review_defaults(entry))
            entry["review_status"] = review_status
            if review_status == "approved":
                entry["approved_slots"] = normalized_slots
                entry["rejected_slots"] = []
            else:
                entry["rejected_slots"] = normalized_slots
                entry["approved_slots"] = []
            entry["notes"] = str(notes or "")
            entry["reviewed_at"] = _utc_timestamp()
            entry["reviewer"] = str(reviewer or "")
            target_entry = entry
        updated_entries.append(entry)

    if target_entry is None:
        raise FileNotFoundError(f"bundle_id not registered: {safe_bundle_id}")

    updated_entries.sort(key=lambda item: str(item.get("bundle_id") or ""))
    new_index = {
        "schema_version": str(index.get("schema_version") or "1.0"),
        "updated_at": _utc_timestamp(),
        "entries": updated_entries,
    }
    _write_json_atomic(_TRAINING_STORE_INDEX, new_index)
    return {
        "ok": True,
        "entry": target_entry,
        "index_path": str(_TRAINING_STORE_INDEX),
        "entry_count": len(updated_entries),
    }
