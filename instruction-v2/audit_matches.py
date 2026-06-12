import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from paths import INDEXES_DIR


MATCH_MANIFEST_PATH = INDEXES_DIR / "09_match_manifest.json"
MANUAL_CONFIG_PATH = INDEXES_DIR / "11_manual_match_config.json"
OUT_PATH = INDEXES_DIR / "10_match_audit.json"
RANDOM_SEED = 20260602
AUDIT_BUCKET_SIZE = 10
DEFAULT_MANUAL_CONFIG = {
    "force_part_num": None,
    "force_color_id": None,
    "reject_candidate_part_nums": [],
    "reject_candidate_color_ids": [],
    "notes": "",
    "status": "pending",
}


def top_score(entry: Dict[str, Any]) -> float:
    candidates = entry.get("top_candidates") or []
    if not candidates:
        return -1.0
    score = candidates[0].get("score")
    if isinstance(score, (int, float)):
        return float(score)
    return -1.0


def catalog_img_url(candidate: Optional[Dict[str, Any]]) -> Optional[str]:
    if not candidate:
        return None
    part_num = candidate.get("part_num")
    color_id = candidate.get("color_id")
    if part_num is None or color_id is None:
        return None
    return f"https://img.aim2build.co.uk/static/element_images/{part_num}/{part_num}__{color_id}.jpg"


def overlay_path_from_cutout(cutout_path: Any) -> Optional[str]:
    if not isinstance(cutout_path, str) or not cutout_path:
        return None
    if cutout_path.endswith("_cutout.png"):
        return cutout_path[:-11] + "_overlay.png"
    return None


def manual_key(crop_id: Any, segment_index: Any) -> tuple[Any, Any]:
    return (crop_id, segment_index)


def load_manual_config() -> Dict[tuple[Any, Any], Dict[str, Any]]:
    if not MANUAL_CONFIG_PATH.exists():
        return {}

    payload = json.loads(MANUAL_CONFIG_PATH.read_text(encoding="utf-8"))
    overrides = payload.get("overrides") or []
    if not isinstance(overrides, list):
        raise RuntimeError("11_manual_match_config.json overrides must be a list")

    config_by_key = {}
    for index, override in enumerate(overrides):
        if not isinstance(override, dict):
            raise RuntimeError(f"Manual override entry {index} must be an object")
        crop_id = override.get("crop_id")
        segment_index = override.get("segment_index")
        if crop_id is None or segment_index is None:
            raise RuntimeError(f"Manual override entry {index} must include crop_id and segment_index")
        config_by_key[manual_key(crop_id, segment_index)] = {
            "force_part_num": override.get("force_part_num"),
            "force_color_id": override.get("force_color_id"),
            "reject_candidate_part_nums": override.get("reject_candidate_part_nums") or [],
            "reject_candidate_color_ids": override.get("reject_candidate_color_ids") or [],
            "notes": override.get("notes") or "",
            "status": override.get("status") or "pending",
        }
    return config_by_key


def manual_config_for(entry: Dict[str, Any], config_by_key: Dict[tuple[Any, Any], Dict[str, Any]]) -> Dict[str, Any]:
    config = DEFAULT_MANUAL_CONFIG.copy()
    override = config_by_key.get(manual_key(entry.get("crop_id"), entry.get("segment_index")))
    if override:
        config.update(override)
    return config


def audit_item(entry: Dict[str, Any], reason_bucket: str, config_by_key: Dict[tuple[Any, Any], Dict[str, Any]]) -> Dict[str, Any]:
    top_candidate = (entry.get("top_candidates") or [None])[0]
    return {
        "bag": entry.get("bag"),
        "page": entry.get("page"),
        "step": entry.get("step"),
        "crop_id": entry.get("crop_id"),
        "segment_index": entry.get("segment_index"),
        "cutout_path": entry.get("cutout_path"),
        "overlay_path": overlay_path_from_cutout(entry.get("cutout_path")),
        "top_candidate_part_num": top_candidate.get("part_num") if top_candidate else None,
        "top_candidate_color_id": top_candidate.get("color_id") if top_candidate else None,
        "top_candidate_score": top_candidate.get("score") if top_candidate else None,
        "expected_in_set": top_candidate.get("expected_in_set") if top_candidate else entry.get("expected_in_set"),
        "catalog_img_url": catalog_img_url(top_candidate),
        "manual_config": manual_config_for(entry, config_by_key),
        "reason_bucket": reason_bucket,
    }


def entry_key(entry: Dict[str, Any]) -> tuple[Any, Any, Any]:
    return (entry.get("crop_id"), entry.get("segment_index"), entry.get("cutout_path"))


def select_audit_entries(entries: List[Dict[str, Any]], config_by_key: Dict[tuple[Any, Any], Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = [entry for entry in entries if entry.get("top_candidates")]
    highest = sorted(scored, key=top_score, reverse=True)[:AUDIT_BUCKET_SIZE]
    lowest = sorted(scored, key=top_score)[:AUDIT_BUCKET_SIZE]

    used = {entry_key(entry) for entry in highest + lowest}
    random_pool = [entry for entry in scored if entry_key(entry) not in used]
    rng = random.Random(RANDOM_SEED)
    if len(random_pool) > AUDIT_BUCKET_SIZE:
        random_entries = rng.sample(random_pool, AUDIT_BUCKET_SIZE)
    else:
        random_entries = random_pool

    audit_entries = []
    audit_entries.extend(audit_item(entry, "highest", config_by_key) for entry in highest)
    audit_entries.extend(audit_item(entry, "lowest", config_by_key) for entry in lowest)
    audit_entries.extend(audit_item(entry, "random", config_by_key) for entry in random_entries)
    return audit_entries


def build_match_audit() -> Dict[str, Any]:
    if not MATCH_MANIFEST_PATH.exists():
        raise RuntimeError(f"Missing match manifest: {MATCH_MANIFEST_PATH}")

    match_manifest = json.loads(MATCH_MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = match_manifest.get("entries") or []
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("09_match_manifest.json has no match entries to audit")

    config_by_key = load_manual_config()
    audit_entries = select_audit_entries(entries, config_by_key)
    payload = {
        "stage": 9,
        "name": "match_audit",
        "input_manifests": [
            "indexes/09_match_manifest.json",
            "indexes/11_manual_match_config.json",
        ],
        "selection": {
            "highest_count": AUDIT_BUCKET_SIZE,
            "lowest_count": AUDIT_BUCKET_SIZE,
            "random_count": AUDIT_BUCKET_SIZE,
            "random_seed": RANDOM_SEED,
        },
        "entry_count": len(audit_entries),
        "entries": audit_entries,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    payload = build_match_audit()
    print(
        json.dumps(
            {
                "ok": True,
                "entry_count": payload.get("entry_count"),
                "out": str(OUT_PATH.relative_to(INDEXES_DIR.parent)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
