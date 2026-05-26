from __future__ import annotations

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
from clean.services.training_bundle_index_service import register_bundle


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
