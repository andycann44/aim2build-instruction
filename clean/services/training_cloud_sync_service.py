from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from clean.services.training_store_service import (
    _REPO_ROOT,
    _TRAINING_STORE_INDEX,
    _read_json_file,
    _safe_bundle_id,
    _utc_timestamp,
    _write_json_atomic,
)
from clean.services.training_bundle_index_service import get_bundle, register_bundle, update_split_candidates


_R2_REQUIRED_KEYS = ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"]
_R2_OPTIONAL_KEYS = ["R2_PUBLIC_BASE_URL"]


def _approved_entry(bundle_id: str) -> Dict[str, Any]:
    safe_bundle_id = _safe_bundle_id(bundle_id)
    index = _read_json_file(_TRAINING_STORE_INDEX)
    entries = index.get("entries") if isinstance(index.get("entries"), list) else []
    for item in entries:
        if not isinstance(item, dict):
            continue
        if str(item.get("bundle_id") or "") != safe_bundle_id:
            continue
        if str(item.get("review_status") or "") != "approved":
            raise ValueError(f"bundle_id is not approved: {safe_bundle_id}")
        return dict(item)
    raise FileNotFoundError(f"bundle_id not registered: {safe_bundle_id}")


def _flatten_artifacts(artifact_paths: Dict[str, Any]) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []

    def add_file(role: str, path_value: Any) -> None:
        path_text = str(path_value or "").strip()
        if not path_text:
            return
        path = Path(path_text)
        files.append(
            {
                "role": role,
                "local_path": path_text,
                "exists": path.exists() and path.is_file(),
                "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
            }
        )

    add_file("original_crop", artifact_paths.get("original_crop"))
    add_file("full_mask_overlay", artifact_paths.get("full_mask_overlay"))
    add_file("raw_master_mask", artifact_paths.get("raw_master_mask"))
    add_file("master_island_overlay", artifact_paths.get("master_island_overlay"))
    add_file("metadata", artifact_paths.get("metadata"))
    for index, path_value in enumerate(list(artifact_paths.get("slot_cutouts") or [])):
        add_file(f"slot_cutout_{index}", path_value)
    return files


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


def _load_r2_config() -> Dict[str, Any]:
    local_env = _read_local_env_file()
    values: Dict[str, str] = {}
    sources: Dict[str, str] = {}
    for key in _R2_REQUIRED_KEYS + _R2_OPTIONAL_KEYS:
        env_value = str(os.environ.get(key) or "").strip()
        if env_value:
            values[key] = env_value
            sources[key] = "environment"
            continue
        file_value = str(local_env.get(key) or "").strip()
        if file_value:
            values[key] = file_value
            sources[key] = ".env"
    missing = [key for key in _R2_REQUIRED_KEYS if not values.get(key)]
    return {
        "values": values,
        "missing": missing,
        "present": {key: bool(values.get(key)) for key in _R2_REQUIRED_KEYS + _R2_OPTIONAL_KEYS},
        "sources": sources,
    }


def _r2_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "present": dict(config.get("present") or {}),
        "missing": list(config.get("missing") or []),
        "sources": dict(config.get("sources") or {}),
    }


def _update_r2_upload_status(bundle_id: str, status: str, r2_paths: Dict[str, str]) -> None:
    index = _read_json_file(_TRAINING_STORE_INDEX)
    entries = index.get("entries") if isinstance(index.get("entries"), list) else []
    updated_entries: List[Dict[str, Any]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        if str(entry.get("bundle_id") or "") == str(bundle_id):
            entry["r2_status"] = status
            entry["r2_uploaded_at"] = _utc_timestamp() if status == "uploaded" else ""
            entry["r2_paths"] = dict(r2_paths)
        updated_entries.append(entry)
    _write_json_atomic(
        _TRAINING_STORE_INDEX,
        {
            "schema_version": str(index.get("schema_version") or "1.0"),
            "updated_at": _utc_timestamp(),
            "entries": updated_entries,
        },
    )


def _public_or_key(values: Dict[str, str], r2_key: str) -> str:
    public_base = str(values.get("R2_PUBLIC_BASE_URL") or "").rstrip("/")
    return f"{public_base}/{r2_key}" if public_base else r2_key


def _safe_r2_segment(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return safe.strip("._") or fallback


def _candidate_upload_timestamp(status: str) -> str:
    return _utc_timestamp() if status in {"uploaded", "upload_failed"} else ""


def _candidate_assets_dir(bundle_id: str, candidate_index: int, fallback_path: str = "") -> Path:
    fallback = Path(str(fallback_path or "").strip())
    if fallback.parent and str(fallback.parent) not in {"", "."}:
        return fallback.parent
    return _REPO_ROOT / "debug" / "ai_training" / str(bundle_id) / "confirmed_candidates" / str(candidate_index)


def _candidate_asset_file(role: str, path_value: Any, r2_key: str) -> Dict[str, Any]:
    path_text = str(path_value or "").strip()
    path = Path(path_text) if path_text else Path("")
    exists = bool(path_text and path.exists() and path.is_file())
    return {
        "role": role,
        "local_path": path_text,
        "exists": exists,
        "size_bytes": int(path.stat().st_size) if exists else 0,
        "r2_key": r2_key,
    }


def _write_candidate_confirmation_metadata(
    *,
    row: Dict[str, Any],
    candidate: Dict[str, Any],
    confirmation: Dict[str, Any],
    metadata_path: Path,
    image_path: str,
    mask_path: str,
) -> Dict[str, Any]:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "kind": "confirmed_candidate_part",
        "bundle_id": str(row.get("bundle_id") or confirmation.get("bundle_id") or ""),
        "set_num": str(row.get("set_num") or ""),
        "bag_num": row.get("bag_num"),
        "page_num": row.get("page_num"),
        "step_num": row.get("step_num"),
        "crop_num": row.get("crop_num"),
        "candidate_index": confirmation.get("candidate_index"),
        "part_num": str(confirmation.get("part_num") or ""),
        "color_id": confirmation.get("color_id"),
        "element_id": str(confirmation.get("element_id") or ""),
        "qty": confirmation.get("qty"),
        "confirmed_by": str(confirmation.get("confirmed_by") or ""),
        "confirmed_at": str(confirmation.get("confirmed_at") or ""),
        "candidate_status": str(candidate.get("status") or ""),
        "candidate_review_state": str(candidate.get("review_state") or ""),
        "local_paths": {
            "thumbnail": str(image_path or ""),
            "mask": str(mask_path or ""),
        },
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload


def _set_candidate_r2_status(
    *,
    bundle_id: str,
    candidate_index: int,
    status: str,
    r2_paths: Dict[str, str],
    metadata_path: str = "",
    error: str = "",
) -> Dict[str, Any]:
    row = dict(get_bundle(bundle_id).get("row") or {})
    paths = row.get("split_candidate_paths") if isinstance(row.get("split_candidate_paths"), dict) else {}
    candidates = list(paths.get("candidates") or [])
    target_pos = -1
    for pos, raw_candidate in enumerate(candidates):
        candidate = raw_candidate if isinstance(raw_candidate, dict) else {}
        try:
            candidate_idx = int(candidate.get("index"))
        except Exception:
            candidate_idx = pos
        if candidate_idx == int(candidate_index):
            target_pos = pos
            break
    if target_pos < 0 and 0 <= int(candidate_index) < len(candidates):
        target_pos = int(candidate_index)
    if target_pos < 0 or target_pos >= len(candidates):
        raise ValueError("candidate_index is out of range")

    candidate = dict(candidates[target_pos]) if isinstance(candidates[target_pos], dict) else {}
    candidate["r2_status"] = status
    candidate["r2_uploaded_at"] = _candidate_upload_timestamp(status)
    candidate["r2_paths"] = dict(r2_paths)
    if metadata_path:
        candidate["confirmation_metadata_path"] = str(metadata_path)
    if error:
        candidate["r2_error"] = str(error)
    elif "r2_error" in candidate:
        candidate.pop("r2_error", None)
    candidates[target_pos] = candidate

    def update_group(raw_items: Any) -> List[Dict[str, Any]]:
        updated: List[Dict[str, Any]] = []
        for pos, raw_item in enumerate(list(raw_items or [])):
            item = dict(raw_item) if isinstance(raw_item, dict) else {}
            try:
                item_idx = int(item.get("index"))
            except Exception:
                item_idx = pos
            if item_idx == int(candidate_index):
                item.update(candidate)
            updated.append(item)
        return updated

    paths["candidates"] = candidates
    paths["baseline_slot_candidates"] = update_group(paths.get("baseline_slot_candidates"))
    paths["ai_suggested_candidates"] = update_group(paths.get("ai_suggested_candidates"))
    paths["r2_status"] = status
    paths["r2_uploaded_at"] = candidate["r2_uploaded_at"]
    paths["r2_paths"] = dict(r2_paths)
    stored = update_split_candidates(bundle_id, split_candidate_paths=paths)
    return {"ok": True, "row": stored.get("row"), "candidate": candidate}


def _update_bundle_metadata_candidate_upload(
    *,
    row: Dict[str, Any],
    candidate_index: int,
    status: str,
    r2_paths: Dict[str, str],
    metadata_path: str = "",
    error: str = "",
) -> None:
    manifest_path = Path(str(row.get("manifest_path") or "").strip())
    if not manifest_path.exists() or not manifest_path.is_file():
        return
    payload = _read_json_file(manifest_path)
    if not isinstance(payload, dict):
        payload = {}
    uploads = payload.get("confirmed_candidate_uploads")
    if not isinstance(uploads, dict):
        uploads = {}
    upload_row = {
        "candidate_index": int(candidate_index),
        "r2_status": status,
        "r2_uploaded_at": _candidate_upload_timestamp(status),
        "r2_paths": dict(r2_paths),
        "confirmation_metadata_path": str(metadata_path or ""),
    }
    if error:
        upload_row["r2_error"] = str(error)
    uploads[str(int(candidate_index))] = upload_row
    payload["confirmed_candidate_uploads"] = uploads
    payload["r2_status"] = status
    payload["r2_uploaded_at"] = upload_row["r2_uploaded_at"]
    payload["r2_paths"] = dict(r2_paths)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def upload_confirmed_candidate_assets(
    *,
    bundle_id: str,
    candidate_index: int,
    candidate: Dict[str, Any],
    confirmation: Dict[str, Any],
    row: Dict[str, Any] | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    row = dict(row or get_bundle(bundle_id).get("row") or {})
    safe_bundle_id = str(row.get("bundle_id") or bundle_id or "").strip()
    if not safe_bundle_id:
        raise ValueError("bundle_id is required")
    review_state = str(candidate.get("review_state") or "").strip()
    status = str(candidate.get("status") or "").strip()
    blocked_states = {"needs_mask_expand", "needs_ocr_review", "needs_manual_crop"}
    if status == "rejected" or review_state in blocked_states:
        return {
            "ok": True,
            "skipped": True,
            "dry_run": bool(dry_run),
            "reason": "candidate_not_clean",
            "candidate_status": status,
            "candidate_review_state": review_state,
        }
    if status != "accepted":
        return {
            "ok": True,
            "skipped": True,
            "dry_run": bool(dry_run),
            "reason": "candidate_not_accepted",
            "candidate_status": status,
            "candidate_review_state": review_state,
        }

    image_path = str(
        confirmation.get("thumbnail_path")
        or candidate.get("qty_scrubbed_path")
        or candidate.get("thumbnail_path")
        or candidate.get("candidate_path")
        or ""
    ).strip()
    mask_path = str(
        candidate.get("qty_scrubbed_mask_path")
        or candidate.get("v2_mask_path")
        or candidate.get("mask_path")
        or ""
    ).strip()

    metadata_dir = _candidate_assets_dir(safe_bundle_id, int(candidate_index), image_path)
    metadata_path = metadata_dir / f"candidate_{int(candidate_index)}_confirmation_metadata.json"
    if not image_path or not Path(image_path).exists():
        error = "candidate_asset_missing"
        if dry_run:
            return {
                "ok": False,
                "bundle_id": safe_bundle_id,
                "candidate_index": int(candidate_index),
                "dry_run": True,
                "uploaded": False,
                "would_upload": False,
                "error": error,
            }
        _set_candidate_r2_status(
            bundle_id=safe_bundle_id,
            candidate_index=int(candidate_index),
            status="upload_failed",
            r2_paths={},
            metadata_path=str(metadata_path),
            error=error,
        )
        _update_bundle_metadata_candidate_upload(
            row=row,
            candidate_index=int(candidate_index),
            status="upload_failed",
            r2_paths={},
            metadata_path=str(metadata_path),
            error=error,
        )
        return {
            "ok": False,
            "bundle_id": safe_bundle_id,
            "candidate_index": int(candidate_index),
            "dry_run": bool(dry_run),
            "uploaded": False,
            "would_upload": False,
            "error": error,
        }
    metadata_payload = _write_candidate_confirmation_metadata(
        row=row,
        candidate=candidate,
        confirmation=confirmation,
        metadata_path=metadata_path,
        image_path=image_path,
        mask_path=mask_path,
    )

    prefix = (
        f"confirmed-candidates/{_safe_r2_segment(safe_bundle_id, 'bundle')}/"
        f"candidate_{int(candidate_index)}_"
        f"{_safe_r2_segment(confirmation.get('part_num'), 'part')}_"
        f"{_safe_r2_segment(confirmation.get('color_id'), 'color')}"
    )
    files = [
        _candidate_asset_file("thumbnail", image_path, f"{prefix}/{Path(image_path).name}"),
        _candidate_asset_file("metadata", str(metadata_path), f"{prefix}/{metadata_path.name}"),
    ]
    if mask_path:
        files.append(_candidate_asset_file("mask", mask_path, f"{prefix}/{Path(mask_path).name}"))
    manifest = {
        "provider": "cloudflare_r2",
        "bundle_id": safe_bundle_id,
        "candidate_index": int(candidate_index),
        "part_num": str(confirmation.get("part_num") or ""),
        "color_id": confirmation.get("color_id"),
        "element_id": str(confirmation.get("element_id") or ""),
        "files": files,
        "metadata": metadata_payload,
    }
    config = _load_r2_config()
    prepared = {
        "ok": True,
        "bundle_id": safe_bundle_id,
        "candidate_index": int(candidate_index),
        "dry_run": bool(dry_run),
        "would_upload": True,
        "manifest": manifest,
        "r2_config": _r2_config_summary(config),
    }
    if dry_run:
        return prepared
    if config.get("missing"):
        error = "missing_r2_config"
        _set_candidate_r2_status(
            bundle_id=safe_bundle_id,
            candidate_index=int(candidate_index),
            status="upload_failed",
            r2_paths={},
            metadata_path=str(metadata_path),
            error=error,
        )
        _update_bundle_metadata_candidate_upload(
            row=row,
            candidate_index=int(candidate_index),
            status="upload_failed",
            r2_paths={},
            metadata_path=str(metadata_path),
            error=error,
        )
        return {**prepared, "ok": False, "uploaded": False, "would_upload": False, "error": error}

    try:
        import boto3  # type: ignore
    except Exception:
        error = "boto3_unavailable"
        _set_candidate_r2_status(
            bundle_id=safe_bundle_id,
            candidate_index=int(candidate_index),
            status="upload_failed",
            r2_paths={},
            metadata_path=str(metadata_path),
            error=error,
        )
        _update_bundle_metadata_candidate_upload(
            row=row,
            candidate_index=int(candidate_index),
            status="upload_failed",
            r2_paths={},
            metadata_path=str(metadata_path),
            error=error,
        )
        return {**prepared, "ok": False, "uploaded": False, "would_upload": False, "error": error}

    values = dict(config.get("values") or {})
    endpoint_url = f"https://{values['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=values["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=values["R2_SECRET_ACCESS_KEY"],
    )
    uploaded_paths: Dict[str, str] = {}
    try:
        for item in files:
            local_path = str(item.get("local_path") or "")
            r2_key = str(item.get("r2_key") or "")
            if not local_path or not r2_key or not bool(item.get("exists")):
                continue
            client.upload_file(local_path, values["R2_BUCKET"], r2_key)
            uploaded_paths[str(item.get("role") or r2_key)] = _public_or_key(values, r2_key)
    except Exception as exc:
        error = type(exc).__name__
        _set_candidate_r2_status(
            bundle_id=safe_bundle_id,
            candidate_index=int(candidate_index),
            status="upload_failed",
            r2_paths=uploaded_paths,
            metadata_path=str(metadata_path),
            error=error,
        )
        _update_bundle_metadata_candidate_upload(
            row=row,
            candidate_index=int(candidate_index),
            status="upload_failed",
            r2_paths=uploaded_paths,
            metadata_path=str(metadata_path),
            error=error,
        )
        return {
            **prepared,
            "ok": False,
            "uploaded": False,
            "would_upload": False,
            "r2_paths": uploaded_paths,
            "error": error,
        }

    _set_candidate_r2_status(
        bundle_id=safe_bundle_id,
        candidate_index=int(candidate_index),
        status="uploaded",
        r2_paths=uploaded_paths,
        metadata_path=str(metadata_path),
    )
    _update_bundle_metadata_candidate_upload(
        row=row,
        candidate_index=int(candidate_index),
        status="uploaded",
        r2_paths=uploaded_paths,
        metadata_path=str(metadata_path),
    )
    return {
        **prepared,
        "ok": True,
        "uploaded": True,
        "would_upload": False,
        "r2_status": "uploaded",
        "r2_paths": uploaded_paths,
    }


def prepare_bundle_for_r2(bundle_id: str) -> Dict[str, Any]:
    entry = _approved_entry(bundle_id)
    artifact_paths = entry.get("artifact_paths") if isinstance(entry.get("artifact_paths"), dict) else {}
    files = _flatten_artifacts(artifact_paths)
    safe_bundle_id = str(entry.get("bundle_id") or "")
    manifest_files = [
        {
            **item,
            "r2_key": f"training-bundles/{safe_bundle_id}/{Path(str(item.get('local_path') or '')).name}",
        }
        for item in files
    ]
    return {
        "ok": True,
        "mode": "dry_run",
        "bundle_id": safe_bundle_id,
        "review_status": str(entry.get("review_status") or ""),
        "r2_status": str(entry.get("r2_status") or "pending"),
        "would_upload": True,
        "manifest": {
            "provider": "cloudflare_r2",
            "bundle_id": safe_bundle_id,
            "files": manifest_files,
        },
    }


def upload_bundle_to_r2(bundle_id: str, *, dry_run: bool = True) -> Dict[str, Any]:
    prepared = prepare_bundle_for_r2(bundle_id)
    config = _load_r2_config()
    prepared["r2_config"] = _r2_config_summary(config)
    prepared["dry_run"] = bool(dry_run)
    if dry_run:
        prepared["would_upload"] = True
        return prepared

    if config.get("missing"):
        return {
            **prepared,
            "ok": False,
            "would_upload": False,
            "error": "missing_r2_config",
        }

    try:
        import boto3  # type: ignore
    except Exception:
        return {
            **prepared,
            "ok": False,
            "would_upload": False,
            "error": "boto3_unavailable",
        }

    values = dict(config.get("values") or {})
    endpoint_url = f"https://{values['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=values["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=values["R2_SECRET_ACCESS_KEY"],
    )
    uploaded_paths: Dict[str, str] = {}
    try:
        for item in list((prepared.get("manifest") or {}).get("files") or []):
            local_path = str(item.get("local_path") or "")
            r2_key = str(item.get("r2_key") or "")
            if not local_path or not r2_key or not bool(item.get("exists")):
                continue
            client.upload_file(local_path, values["R2_BUCKET"], r2_key)
            public_base = str(values.get("R2_PUBLIC_BASE_URL") or "").rstrip("/")
            uploaded_paths[str(item.get("role") or r2_key)] = (
                f"{public_base}/{r2_key}" if public_base else r2_key
            )
    except Exception as exc:
        _update_r2_upload_status(str(prepared.get("bundle_id") or bundle_id), "failed", uploaded_paths)
        return {
            **prepared,
            "ok": False,
            "would_upload": False,
            "uploaded": False,
            "r2_paths": uploaded_paths,
            "error": type(exc).__name__,
        }

    uploaded_bundle_id = str(prepared.get("bundle_id") or bundle_id)
    _update_r2_upload_status(uploaded_bundle_id, "uploaded", uploaded_paths)
    entry = _approved_entry(uploaded_bundle_id)
    metadata_path = str(((entry.get("artifact_paths") or {}).get("metadata") if isinstance(entry.get("artifact_paths"), dict) else "") or "")
    metadata = _read_json_file(Path(metadata_path)) if metadata_path else {}
    r2_prefix = f"training-bundles/{uploaded_bundle_id}/"
    azure_registration: Dict[str, Any]
    try:
        azure_registration = register_bundle(
            entry,
            metadata=metadata,
            r2_prefix=r2_prefix,
            manifest_path=metadata_path,
        )
    except Exception as exc:
        azure_registration = {
            "ok": False,
            "error": type(exc).__name__,
        }
    return {
        **prepared,
        "ok": True,
        "would_upload": False,
        "uploaded": True,
        "r2_paths": uploaded_paths,
        "azure_postgres_registration": azure_registration,
    }


def prepare_bundle_for_azure(bundle_id: str) -> Dict[str, Any]:
    entry = _approved_entry(bundle_id)
    artifact_paths = entry.get("artifact_paths") if isinstance(entry.get("artifact_paths"), dict) else {}
    files = _flatten_artifacts(artifact_paths)
    safe_bundle_id = str(entry.get("bundle_id") or "")
    return {
        "ok": True,
        "mode": "dry_run",
        "bundle_id": safe_bundle_id,
        "review_status": str(entry.get("review_status") or ""),
        "azure_status": str(entry.get("azure_status") or "pending"),
        "would_index": True,
        "manifest": {
            "provider": "azure",
            "bundle_id": safe_bundle_id,
            "set_num": str(entry.get("set_num") or ""),
            "bag": int(entry.get("bag", 0) or 0),
            "crop_id": str(entry.get("crop_id") or ""),
            "local_bundle_path": str(entry.get("local_bundle_path") or ""),
            "files": files,
        },
    }


_AZURE_BLOB_CONNECTION_KEY = "AZURE_STORAGE_CONNECTION_STRING"
_AZURE_BLOB_ACCOUNT_NAME_KEY = "AZURE_STORAGE_ACCOUNT_NAME"
_AZURE_BLOB_ACCOUNT_KEY_KEY = "AZURE_STORAGE_ACCOUNT_KEY"
_AZURE_BLOB_CONTAINER_KEY = "AZURE_STORAGE_CONTAINER"
_AZURE_BLOB_OPTIONAL_KEYS = [_AZURE_BLOB_CONTAINER_KEY]
_AZURE_BLOB_DEFAULT_CONTAINER = "bag-review-metadata"


def _azure_blob_config_keys() -> List[str]:
    return [
        _AZURE_BLOB_CONNECTION_KEY,
        _AZURE_BLOB_ACCOUNT_NAME_KEY,
        _AZURE_BLOB_ACCOUNT_KEY_KEY,
        *_AZURE_BLOB_OPTIONAL_KEYS,
    ]


def _azure_blob_config_complete(values: Dict[str, str]) -> bool:
    if str(values.get(_AZURE_BLOB_CONNECTION_KEY) or "").strip():
        return True
    return bool(
        str(values.get(_AZURE_BLOB_ACCOUNT_NAME_KEY) or "").strip()
        and str(values.get(_AZURE_BLOB_ACCOUNT_KEY_KEY) or "").strip()
    )


def _load_azure_blob_config() -> Dict[str, Any]:
    local_env = _read_local_env_file()
    values: Dict[str, str] = {}
    sources: Dict[str, str] = {}
    for key in _azure_blob_config_keys():
        env_value = str(os.environ.get(key) or "").strip()
        if env_value:
            values[key] = env_value
            sources[key] = "environment"
            continue
        file_value = str(local_env.get(key) or "").strip()
        if file_value:
            values[key] = file_value
            sources[key] = ".env"
    missing: List[str] = []
    if not _azure_blob_config_complete(values):
        if not str(values.get(_AZURE_BLOB_CONNECTION_KEY) or "").strip():
            missing.append(_AZURE_BLOB_CONNECTION_KEY)
        if not (
            str(values.get(_AZURE_BLOB_ACCOUNT_NAME_KEY) or "").strip()
            and str(values.get(_AZURE_BLOB_ACCOUNT_KEY_KEY) or "").strip()
        ):
            missing.extend(
                key
                for key in (_AZURE_BLOB_ACCOUNT_NAME_KEY, _AZURE_BLOB_ACCOUNT_KEY_KEY)
                if not str(values.get(key) or "").strip()
            )
    present = {key: bool(str(values.get(key) or "").strip()) for key in _azure_blob_config_keys()}
    present["azure_blob_auth"] = _azure_blob_config_complete(values)
    return {
        "values": values,
        "missing": missing,
        "present": present,
        "sources": sources,
    }


def _azure_blob_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "present": dict(config.get("present") or {}),
        "missing": list(config.get("missing") or []),
        "sources": dict(config.get("sources") or {}),
    }


def _azure_storage_account_name(values: Dict[str, str]) -> str:
    connection_string = str(values.get(_AZURE_BLOB_CONNECTION_KEY) or "").strip()
    for part in connection_string.split(";"):
        segment = part.strip()
        if segment.lower().startswith("accountname="):
            return segment.split("=", 1)[1].strip()
    return str(values.get(_AZURE_BLOB_ACCOUNT_NAME_KEY) or "").strip()


def _azure_blob_container_name(values: Dict[str, str]) -> str:
    return str(values.get(_AZURE_BLOB_CONTAINER_KEY) or _AZURE_BLOB_DEFAULT_CONTAINER).strip() or _AZURE_BLOB_DEFAULT_CONTAINER


def azure_blob_path_for_bag_review_metadata(set_num: str, bag: int) -> str:
    safe_set = "".join(ch for ch in str(set_num or "").strip() if ch.isalnum() or ch in "-_") or "unknown"
    return f"{safe_set}/bag{max(1, int(bag or 1))}/metadata.json"


def azure_blob_url_for_path(blob_path: str, values: Dict[str, str]) -> str:
    account_name = _azure_storage_account_name(values)
    container_name = _azure_blob_container_name(values)
    blob_path_text = str(blob_path or "").lstrip("/")
    if not account_name or not container_name or not blob_path_text:
        return ""
    return f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_path_text}"


def upload_file_to_azure_blob(
    local_path: str,
    blob_path: str,
    *,
    dry_run: bool = True,
    content_type: str = "application/json",
) -> Dict[str, Any]:
    path = Path(str(local_path or "").strip())
    blob_path_text = str(blob_path or "").lstrip("/")
    config = _load_azure_blob_config()
    values = dict(config.get("values") or {})
    container_name = _azure_blob_container_name(values)
    blob_url = azure_blob_url_for_path(blob_path_text, values)
    summary = _azure_blob_config_summary(config)

    base = {
        "ok": True,
        "uploaded": False,
        "dry_run": bool(dry_run),
        "would_upload": True,
        "container": container_name,
        "blob_path": blob_path_text,
        "blob_url": blob_url,
        "local_path": str(path),
        "azure_blob_config": summary,
        "error": None,
    }

    if not path.is_file():
        return {
            **base,
            "ok": False,
            "would_upload": False,
            "error": "local_file_missing",
        }

    if dry_run:
        return base

    if not _azure_blob_config_complete(values):
        return {
            **base,
            "ok": True,
            "dry_run": True,
            "error": "missing_azure_blob_config",
        }

    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except Exception:
        return {
            **base,
            "ok": False,
            "would_upload": False,
            "dry_run": False,
            "error": "azure_storage_blob_unavailable",
        }

    connection_string = str(values.get(_AZURE_BLOB_CONNECTION_KEY) or "").strip()
    account_name = str(values.get(_AZURE_BLOB_ACCOUNT_NAME_KEY) or "").strip()
    account_key = str(values.get(_AZURE_BLOB_ACCOUNT_KEY_KEY) or "").strip()
    try:
        if connection_string:
            client = BlobServiceClient.from_connection_string(connection_string)
        else:
            client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=account_key,
            )
        blob_client = client.get_blob_client(container=container_name, blob=blob_path_text)
        with path.open("rb") as handle:
            blob_client.upload_blob(handle, overwrite=True, content_type=content_type)
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "would_upload": False,
            "dry_run": False,
            "error": type(exc).__name__,
        }

    return {
        **base,
        "ok": True,
        "uploaded": True,
        "would_upload": False,
        "dry_run": False,
    }
