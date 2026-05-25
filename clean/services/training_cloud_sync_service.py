from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from clean.services.training_store_service import _TRAINING_STORE_INDEX, _read_json_file, _safe_bundle_id


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
