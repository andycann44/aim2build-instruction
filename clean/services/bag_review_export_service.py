from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from clean.services.bag_review_service import build_review_model
from clean.services.training_store_service import _REPO_ROOT
from clean.services.training_cloud_sync_service import (
    _load_azure_blob_config,
    _load_r2_config,
    azure_blob_path_for_bag_review_metadata,
    azure_blob_url_for_path,
    _public_or_key,
    _r2_config_summary,
    _safe_r2_segment,
    upload_file_to_azure_blob,
)

_EXPORT_DIR = _REPO_ROOT / "debug" / "bag_review_exports"
_SCHEMA_VERSION = "bag_review_export/1.1"
_ALLOWED_R2_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_BLOCKED_R2_PATH_MARKERS = ("bag_review_exports", "training_labels", "training_data")
_BLOCKED_R2_NAME_TOKENS = ("metadata", "labels", "progress", "truth")


def _azure_metadata_upload_response(
    *,
    enabled: bool,
    uploaded: bool,
    dry_run: bool,
    blob_path: str,
    blob_url: str,
) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "uploaded": bool(uploaded),
        "dry_run": bool(dry_run),
        "blob_path": str(blob_path or ""),
        "blob_url": str(blob_url or ""),
    }


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_set_num(set_num: str) -> str:
    safe = "".join(ch for ch in str(set_num or "").strip() if ch.isalnum() or ch in "-_")
    return safe or "unknown"


def _metadata_path(set_num: str, bag: int) -> Path:
    return _EXPORT_DIR / f"{_safe_set_num(set_num)}_bag{int(bag)}_metadata.json"


def _r2_prefix(set_num: str, bag: int) -> str:
    return f"bag-review/{_safe_set_num(set_num)}/bag{int(bag)}/"


def _slot_review_status(slot: Dict[str, Any]) -> str:
    if bool(slot.get("ignored")):
        return "ignored"
    if bool(slot.get("unknown")):
        return "unknown"
    label = slot.get("saved_label") if isinstance(slot.get("saved_label"), dict) else {}
    if str(label.get("part_num") or "").strip():
        return "confirmed"
    return "needs_review"


def _build_progress(slots: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {
        "confirmed": 0,
        "ignored": 0,
        "unknown": 0,
        "needs_review": 0,
    }
    for slot in slots:
        status = str(slot.get("review_status") or "needs_review")
        if status not in counts:
            status = "needs_review"
        counts[status] += 1
    total = len(slots)
    complete = total - counts["needs_review"]
    return {
        "total_slots": total,
        "confirmed_slots": counts["confirmed"],
        "ignored_slots": counts["ignored"],
        "unknown_slots": counts["unknown"],
        "needs_review_slots": counts["needs_review"],
        "percent_complete": int(round((complete / total) * 100)) if total else 0,
    }


def build_review_progress_from_review_crops(review_crops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mutually-exclusive bag review progress for UI/export parity."""
    slot_rows: List[Dict[str, Any]] = []
    for crop in list(review_crops or []):
        if not isinstance(crop, dict):
            continue
        for slot in list(crop.get("slots") or []):
            if not isinstance(slot, dict):
                continue
            slot_rows.append({"review_status": _slot_review_status(slot)})
    progress = _build_progress(slot_rows)
    return {
        "total_slots": int(progress.get("total_slots", 0) or 0),
        "confirmed_only": int(progress.get("confirmed_slots", 0) or 0),
        "ignored_only": int(progress.get("ignored_slots", 0) or 0),
        "unknown_only": int(progress.get("unknown_slots", 0) or 0),
        "unresolved_only": int(progress.get("needs_review_slots", 0) or 0),
        "percent_complete": int(progress.get("percent_complete", 0) or 0),
    }


def build_bag_review_progress(set_num: str, bag: int) -> Dict[str, Any]:
    review_model = build_review_model(str(set_num), int(bag))
    return build_review_progress_from_review_crops(list(review_model.get("crops") or []))


def _mask_local_path_for_slot(slot: Dict[str, Any]) -> str:
    for key in ("island_slot_mask_path", "step_masked_path"):
        path_text = str(slot.get(key) or "").strip()
        if path_text and Path(path_text).is_file():
            return path_text
    return ""


def _empty_r2_asset_fields() -> Dict[str, str]:
    return {
        "crop_image_r2_key": "",
        "crop_image_r2_url": "",
        "island_cutout_r2_key": "",
        "island_cutout_r2_url": "",
        "mask_r2_key": "",
        "mask_r2_url": "",
        "local_crop_image_path": "",
        "local_island_cutout_path": "",
        "local_mask_path": "",
    }


def _build_slot_record(
    set_num: str,
    bag: int,
    crop: Dict[str, Any],
    slot: Dict[str, Any],
) -> Dict[str, Any]:
    label = slot.get("saved_label") if isinstance(slot.get("saved_label"), dict) else {}
    has_label = bool(str(label.get("part_num") or "").strip())
    return {
        "set_num": str(set_num),
        "bag": int(bag),
        "crop_id": str(crop.get("crop_id") or ""),
        "page": int(crop.get("page", 0) or 0),
        "step": int(crop.get("step", 0) or 0),
        "slot_index": int(slot.get("slot_index", 0) or 0),
        "qty_text": str(slot.get("qty_text") or ""),
        "qty": slot.get("qty"),
        "review_status": _slot_review_status(slot),
        "part_num": str(label.get("part_num") or "") if has_label else None,
        "color_id": int(label.get("color_id", 0) or 0) if has_label else None,
        "element_id": str(label.get("element_id") or "") if has_label else None,
        "island_label": slot.get("island_label"),
        **_empty_r2_asset_fields(),
        "local_crop_image_path": str(crop.get("crop_image_path") or ""),
        "local_island_cutout_path": str(slot.get("island_cutout_path") or ""),
        "local_mask_path": _mask_local_path_for_slot(slot),
    }


def _r2_key_for_asset(
    r2_prefix: str,
    crop_id: str,
    slot_index: int,
    role: str,
    local_path: str,
) -> str:
    suffix = Path(local_path).suffix.lower() or ".png"
    safe_crop_id = _safe_r2_segment(crop_id, "crop")
    if role == "crop_image":
        return f"{r2_prefix}{safe_crop_id}/crop_image{suffix}"
    return f"{r2_prefix}{safe_crop_id}/slot{int(slot_index)}_{role}{suffix}"


def _predict_r2_url(r2_key: str, r2_values: Dict[str, str]) -> str:
    key = str(r2_key or "").strip()
    if not key:
        return ""
    public_base = str(r2_values.get("R2_PUBLIC_BASE_URL") or "").rstrip("/")
    if not public_base:
        return ""
    return f"{public_base}/{key}"


def _enrich_slots_with_r2_assets(
    slots: List[Dict[str, Any]],
    upload_plan: List[Dict[str, Any]],
    r2_paths: Dict[str, str],
    *,
    r2_prefix: str,
    upload_r2: bool,
    dry_run: bool,
) -> None:
    r2_values = dict((_load_r2_config().get("values") or {})) if upload_r2 or dry_run else {}
    plan_by_local_path = {
        str(item.get("local_path") or ""): item
        for item in upload_plan
        if str(item.get("local_path") or "")
    }

    for slot in slots:
        crop_id = str(slot.get("crop_id") or "")
        slot_index = int(slot.get("slot_index", 0) or 0)
        for role, local_field, key_field, url_field in (
            ("crop_image", "local_crop_image_path", "crop_image_r2_key", "crop_image_r2_url"),
            ("island_cutout", "local_island_cutout_path", "island_cutout_r2_key", "island_cutout_r2_url"),
            ("mask", "local_mask_path", "mask_r2_key", "mask_r2_url"),
        ):
            local_path = str(slot.get(local_field) or "").strip()
            if not local_path or not Path(local_path).is_file():
                slot[key_field] = ""
                slot[url_field] = ""
                continue

            plan_item = plan_by_local_path.get(local_path)
            r2_key = str((plan_item or {}).get("r2_key") or "").strip()
            if not r2_key:
                r2_key = _r2_key_for_asset(
                    r2_prefix,
                    crop_id,
                    slot_index,
                    role,
                    local_path,
                )
            slot[key_field] = r2_key

            uploaded_url = str(r2_paths.get(local_path) or "").strip()
            if uploaded_url:
                slot[url_field] = uploaded_url
            elif upload_r2 or dry_run:
                slot[url_field] = _predict_r2_url(r2_key, r2_values)
            else:
                slot[url_field] = ""


def _is_r2_image_asset(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in _ALLOWED_R2_IMAGE_SUFFIXES:
        return False
    lower_name = path.name.lower()
    if any(token in lower_name for token in _BLOCKED_R2_NAME_TOKENS):
        return False
    if any(marker in path.parts for marker in _BLOCKED_R2_PATH_MARKERS):
        return False
    return True


def _asset_upload_plan(
    slots: List[Dict[str, Any]],
    r2_prefix: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Image assets only; metadata JSON is never uploaded to R2."""
    plan: List[Dict[str, Any]] = []
    seen: set[str] = set()
    skipped_non_image_count = 0
    for slot in slots:
        crop_id = _safe_r2_segment(slot.get("crop_id"), "crop")
        slot_index = int(slot.get("slot_index", 0) or 0)
        for role, field in (
            ("crop_image", "local_crop_image_path"),
            ("island_cutout", "local_island_cutout_path"),
            ("mask", "local_mask_path"),
        ):
            local_path = str(slot.get(field) or "").strip()
            if not local_path or local_path in seen:
                continue
            path = Path(local_path)
            if not path.is_file():
                continue
            if not _is_r2_image_asset(path):
                skipped_non_image_count += 1
                continue
            seen.add(local_path)
            r2_key = _r2_key_for_asset(
                r2_prefix,
                str(slot.get("crop_id") or ""),
                slot_index,
                role,
                local_path,
            )
            plan.append(
                {
                    "role": role,
                    "local_path": local_path,
                    "r2_key": r2_key,
                    "crop_id": str(slot.get("crop_id") or ""),
                    "slot_index": slot_index,
                }
            )
    return plan, skipped_non_image_count


def _upload_image_assets_to_r2(
    upload_plan: List[Dict[str, Any]],
    *,
    dry_run: bool,
) -> Tuple[int, Dict[str, str], Optional[str]]:
    if not upload_plan:
        return 0, {}, None

    if dry_run:
        return len(upload_plan), {}, None

    config = _load_r2_config()
    if config.get("missing"):
        return 0, {}, "missing_r2_config"

    try:
        import boto3  # type: ignore
    except Exception:
        return 0, {}, "boto3_unavailable"

    values = dict(config.get("values") or {})
    endpoint_url = f"https://{values['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=values["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=values["R2_SECRET_ACCESS_KEY"],
    )
    uploaded_paths: Dict[str, str] = {}
    uploaded_count = 0
    try:
        for item in upload_plan:
            local_path = str(item.get("local_path") or "")
            r2_key = str(item.get("r2_key") or "")
            if not local_path or not r2_key:
                continue
            client.upload_file(local_path, values["R2_BUCKET"], r2_key)
            uploaded_paths[local_path] = _public_or_key(values, r2_key)
            uploaded_count += 1
    except Exception as exc:
        return uploaded_count, uploaded_paths, type(exc).__name__

    return uploaded_count, uploaded_paths, None


def upload_bag_review_metadata_to_azure(
    metadata: Dict[str, Any],
    metadata_path: str,
    *,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Upload reviewed bag metadata JSON to Azure Blob Storage."""
    set_text = str(metadata.get("set_num") or "").strip()
    bag_number = int(metadata.get("bag", 0) or 0)
    blob_path = azure_blob_path_for_bag_review_metadata(set_text, bag_number)
    blob_config = _load_azure_blob_config()
    blob_values = dict(blob_config.get("values") or {})
    blob_url = azure_blob_url_for_path(blob_path, blob_values)

    upload_result = upload_file_to_azure_blob(
        str(metadata_path or ""),
        blob_path,
        dry_run=dry_run,
        content_type="application/json",
    )

    return {
        "ok": bool(upload_result.get("ok")),
        "enabled": True,
        "uploaded": bool(upload_result.get("uploaded")),
        "dry_run": bool(upload_result.get("dry_run")),
        "blob_path": str(upload_result.get("blob_path") or blob_path),
        "blob_url": str(upload_result.get("blob_url") or blob_url),
        "container": str(upload_result.get("container") or ""),
        "local_path": str(upload_result.get("local_path") or metadata_path or ""),
        "azure_blob_config": upload_result.get("azure_blob_config"),
        "error": upload_result.get("error"),
    }


def _write_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(path)


def export_bag_review(
    set_num: str,
    bag: int,
    *,
    upload_r2: bool = False,
    upload_azure: bool = False,
    dry_run: bool = True,
) -> Dict[str, Any]:
    set_text = str(set_num or "").strip() or "70618"
    bag_number = max(1, int(bag or 1))
    review_model = build_review_model(set_text, bag_number)

    slots: List[Dict[str, Any]] = []
    for crop in list(review_model.get("crops") or []):
        if not isinstance(crop, dict):
            continue
        for slot in list(crop.get("slots") or []):
            if not isinstance(slot, dict):
                continue
            slots.append(_build_slot_record(set_text, bag_number, crop, slot))

    progress = _build_progress(slots)
    r2_prefix = _r2_prefix(set_text, bag_number)
    metadata_path = _metadata_path(set_text, bag_number)
    azure_blob_path = azure_blob_path_for_bag_review_metadata(set_text, bag_number)
    azure_blob_url = azure_blob_url_for_path(
        azure_blob_path,
        dict(_load_azure_blob_config().get("values") or {}),
    )
    source_label_path = str(
        _REPO_ROOT / "debug" / "training_labels" / f"{_safe_set_num(set_text)}_bag{bag_number}.json"
    )

    upload_plan, skipped_non_image_count = _asset_upload_plan(slots, r2_prefix)
    asset_count = len(upload_plan)

    r2_asset_upload_count = 0
    r2_paths: Dict[str, str] = {}
    r2_error: Optional[str] = None
    r2_config_summary: Optional[Dict[str, Any]] = None

    if upload_r2:
        r2_config_summary = _r2_config_summary(_load_r2_config())
        r2_asset_upload_count, r2_paths, r2_error = _upload_image_assets_to_r2(
            upload_plan,
            dry_run=dry_run,
        )

    _enrich_slots_with_r2_assets(
        slots,
        upload_plan,
        r2_paths,
        r2_prefix=r2_prefix,
        upload_r2=upload_r2,
        dry_run=dry_run,
    )

    metadata: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "set_num": set_text,
        "bag": bag_number,
        "exported_at": _iso_now(),
        "local_source_label_path": source_label_path,
        "r2_prefix": r2_prefix,
        "progress": progress,
        "slots": slots,
    }
    if upload_r2:
        metadata["r2_assets"] = {
            "upload_r2": True,
            "dry_run": bool(dry_run),
            "asset_count": asset_count,
            "skipped_non_image_count": skipped_non_image_count,
            "r2_asset_upload_count": r2_asset_upload_count,
            "r2_config": r2_config_summary,
            "r2_error": r2_error,
            "upload_plan": upload_plan,
            "r2_paths": r2_paths,
        }

    _write_metadata(metadata_path, metadata)

    azure_result: Optional[Dict[str, Any]] = None
    if upload_azure:
        azure_result = upload_bag_review_metadata_to_azure(
            metadata,
            str(metadata_path),
            dry_run=dry_run,
        )
        metadata["azure_metadata"] = {
            "enabled": True,
            "uploaded": bool(azure_result.get("uploaded")),
            "dry_run": bool(azure_result.get("dry_run")),
            "blob_path": str(azure_result.get("blob_path") or azure_blob_path),
            "blob_url": str(azure_result.get("blob_url") or azure_blob_url),
            "container": str(azure_result.get("container") or ""),
            "error": azure_result.get("error"),
        }
        _write_metadata(metadata_path, metadata)

    azure_metadata_upload = _azure_metadata_upload_response(
        enabled=bool(upload_azure),
        uploaded=bool((azure_result or {}).get("uploaded")),
        dry_run=bool((azure_result or {}).get("dry_run")) if upload_azure else bool(dry_run),
        blob_path=str((azure_result or {}).get("blob_path") or azure_blob_path),
        blob_url=str((azure_result or {}).get("blob_url") or azure_blob_url),
    )

    return {
        "ok": r2_error is None and (azure_result is None or bool(azure_result.get("ok", True))),
        "metadata_path": str(metadata_path),
        "progress": progress,
        "asset_count": asset_count,
        "skipped_non_image_count": skipped_non_image_count,
        "r2_asset_upload_count": r2_asset_upload_count,
        "azure_metadata_upload": azure_metadata_upload,
        "dry_run": bool(dry_run),
        "upload_r2": bool(upload_r2),
        "upload_azure": bool(upload_azure),
        "r2_error": r2_error,
        "azure_error": (azure_result or {}).get("error"),
    }
