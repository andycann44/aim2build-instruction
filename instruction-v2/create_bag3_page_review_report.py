"""
Bag 3 V1-style per-page completeness gate report.

A step only requires a crop when both conditions hold:
  - visible step number anchor exists
  - visible parts/callout box exists

Steps without a callout box are EMPTY_STEP (no crop required).
Non-build detections (e.g. bag artwork numbers) are FALSE_STEP.

Sequence gaps always create a human review item (full page, no fake crop box):
  REAL_CALLOUT | EMPTY_STEP | FALSE_STEP | NOT_ON_THIS_PAGE

Does not write crop_cache. Does not auto-promote.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import cv2

from paths import INDEXES_DIR, ROOT_DIR
from bag4_misread_step_detection import (
    MISREAD_CLASSIFICATION,
    SUBSTEP_CLASSIFICATION,
    audit_bag_misread_steps,
    audit_bag_substep_numbers,
)
from bag_step_ocr_noise_detection import audit_bag_ocr_noise
from bag3_step_cluster_gate import UNCERTAIN_GATE_STATUS, audit_bag3_step_cluster_gate
from recover_bag3_v1_callouts import run_v1_callout_recovery


PROJECT_ROOT = ROOT_DIR.parent
PAGES_DIR = PROJECT_ROOT / "debug" / "70618" / "70618_01" / "pages"
CROP_CACHE_PATH = PROJECT_ROOT / "debug" / "crop_cache" / "70618_bag3.json"
TRAINING_LABELS_PATH = PROJECT_ROOT / "debug" / "training_labels" / "70618_bag3.json"
BAG_MAP_PATH = INDEXES_DIR / "04_bag_map.json"
STEP_MAP_PATH = INDEXES_DIR / "05_step_map.json"
STEP_PROBE_PATH = ROOT_DIR / "debug" / "sequence_completeness" / "05_step_probe.json"
OUT_DIR = ROOT_DIR / "debug" / "bag3_page_review"
SEQ_GAP_DIR = ROOT_DIR / "debug" / "sequence_gap_review"
HUMAN_LABELS_PATH = OUT_DIR / "human_labels.json"
CROP_CANDIDATES_DIR = OUT_DIR / "crop_candidates"
V1_RECOVERY_DIR = ROOT_DIR / "debug" / "bag3_v1_callout_recovery"

SET_NUM = "70618"
BAG = 3
GATE_STATUS = "NEEDS_HUMAN_CONFIRMATION"
NO_CROP_REQUIRED_STATUS = "NO_CROP_REQUIRED"
LINK_EXISTING_CROP_STATUS = "LINK_EXISTING_CROP"
CREATE_CROP_CANDIDATE_STATUS = "CREATE_CROP_CANDIDATE"
SEQUENCE_GAP_REVIEW_STATUS = "SEQUENCE_GAP_REVIEW"
CONFIDENCE_ADJUST_THRESHOLD = 0.55
HUMAN_LABEL_OPTIONS = ["REAL_CALLOUT", "EMPTY_STEP", "FALSE_STEP"]
SEQUENCE_GAP_LABEL_OPTIONS = ["REAL_CALLOUT", "EMPTY_STEP", "FALSE_STEP", "NOT_ON_THIS_PAGE"]
SEQ_GAP_REVIEW_DIR = OUT_DIR / "sequence_gap_review"
MISREAD_DIAGNOSTICS_PATH = OUT_DIR / "misread_step_diagnostics.json"
SUBSTEP_DIAGNOSTICS_PATH = OUT_DIR / "substep_number_diagnostics.json"
OCR_NOISE_DIAGNOSTICS_PATH = OUT_DIR / "ocr_noise_diagnostics.json"
CLUSTER_GATE_DIAGNOSTICS_PATH = OUT_DIR / "cluster_gate_diagnostics.json"

# Permanent human classifications (survive report regeneration).
PERMANENT_LABELS: Dict[Tuple[int, int], Dict[str, Any]] = {}

# Audit classifications from page image + V1 border detection evidence.
AUDIT_STEP_CLASSIFICATIONS: Dict[Tuple[int, int], Dict[str, Any]] = {}

AUDIT_STEPS: List[int] = []


def _load_bag_page_range() -> Tuple[int, int]:
    payload = _load_json(BAG_MAP_PATH)
    for entry in payload.get("bags", []) or []:
        if int(entry.get("bag") or 0) == BAG:
            return int(entry.get("start_page") or 1), int(entry.get("end_page") or 10_000)
    return 39, 57


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _confirmation_key(page: int, step: int) -> str:
    return f"p{int(page):03d}_s{int(step)}"


def _parse_step_from_crop_id(crop_id: str) -> Optional[int]:
    match = re.search(r"_s(\d+)_", str(crop_id or ""))
    return int(match.group(1)) if match else None


def _box_xywh(box: Any) -> List[int]:
    if isinstance(box, dict):
        return [int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])]
    if isinstance(box, list) and len(box) >= 4:
        return [int(box[i] or 0) for i in range(4)]
    return []


def _page_image_path(page: int) -> str:
    return str(PAGES_DIR / f"page_{int(page):03d}.png")


def _provisional_crop_id(page: int, step: int, crop_index: int = 1) -> str:
    return f"p{int(page)}_s{int(step)}_c{int(crop_index)}"


def _load_training_labels_by_step() -> Dict[tuple[int, int], Dict[str, Any]]:
    if not TRAINING_LABELS_PATH.exists():
        return {}
    payload = _load_json(TRAINING_LABELS_PATH)
    crops = payload.get("crops", {}) if isinstance(payload, dict) else {}
    by_step: Dict[tuple[int, int], Dict[str, Any]] = {}
    for crop_id, record in crops.items():
        if not isinstance(record, dict):
            continue
        page = int(record.get("page") or 0)
        step = int(record.get("step") or 0)
        if page and step:
            by_step[(page, step)] = {
                "crop_id": str(crop_id),
                "crop_box": _box_xywh(record.get("crop_box")),
                "crop_image_path": str(record.get("crop_image_path") or ""),
            }
    return by_step


def _load_existing_human_labels() -> Dict[str, Dict[str, Any]]:
    labels: Dict[str, Dict[str, Any]] = {}
    if HUMAN_LABELS_PATH.exists():
        payload = _load_json(HUMAN_LABELS_PATH)
        for item in payload.get("items", []) or []:
            if isinstance(item, dict) and str(item.get("key") or ""):
                labels[str(item["key"])] = dict(item)
        for item in payload.get("sequence_gap_items", []) or []:
            if isinstance(item, dict) and str(item.get("key") or ""):
                labels[str(item["key"])] = dict(item)
    return labels


def _resolved_label_record(page: int, step: int) -> Optional[Dict[str, Any]]:
    key = (int(page), int(step))
    if key in PERMANENT_LABELS:
        return dict(PERMANENT_LABELS[key])
    if key in AUDIT_STEP_CLASSIFICATIONS:
        return dict(AUDIT_STEP_CLASSIFICATIONS[key])
    return None


def _resolve_human_label(
    page: int,
    step: int,
    existing: Optional[Dict[str, Any]],
    step_entry: Optional[Dict[str, Any]],
    v1_proposal: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    preset = _resolved_label_record(page, step)
    if preset and preset.get("human_label"):
        return str(preset["human_label"]), preset

    prior = dict(existing or {})
    human_label = prior.get("human_label")
    if human_label == "":
        human_label = None
    if human_label:
        return str(human_label), None

    if v1_proposal and v1_proposal.get("crop_box"):
        return None, None

    step_box = list((step_entry or {}).get("step_box") or [])
    if len(step_box) >= 4:
        return "EMPTY_STEP", {
            "human_label": "EMPTY_STEP",
            "labeled_by": "completeness_gate",
            "notes": "Step anchor present; V1 border detection found no callout box",
            "callout_visible": False,
        }
    return None, None


def _step_requires_crop(
    step_entry: Optional[Dict[str, Any]],
    v1_proposal: Optional[Dict[str, Any]],
    human_label: Optional[str],
) -> bool:
    if human_label in ("FALSE_STEP", "EMPTY_STEP"):
        return False
    step_box = list((step_entry or {}).get("step_box") or [])
    if len(step_box) < 4:
        return False
    if human_label == "REAL_CALLOUT":
        return True
    return bool(v1_proposal and v1_proposal.get("crop_box"))


def _callout_visibility(
    step_entry: Optional[Dict[str, Any]],
    v1_proposal: Optional[Dict[str, Any]],
    label_record: Optional[Dict[str, Any]],
) -> str:
    if label_record and "callout_visible" in label_record:
        return "visible_callout_box" if label_record["callout_visible"] else "no_callout_box"
    if v1_proposal and v1_proposal.get("crop_box"):
        return "visible_callout_box"
    if list((step_entry or {}).get("step_box") or []):
        return "no_callout_box"
    return "no_step_anchor"


def _load_detected_steps() -> Dict[int, List[Dict[str, Any]]]:
    by_page: Dict[int, Dict[int, Dict[str, Any]]] = {}

    def add(page: int, step: int, step_box: Any, source: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not page or not step:
            return
        by_page.setdefault(page, {})
        existing = by_page[page].get(step)
        payload = {
            "step": step,
            "step_box": _box_xywh(step_box),
            "sources": [source],
            "sequence_gap_inferred": bool((extra or {}).get("sequence_gap_inferred")),
            "callout_visible_or_inferred": (extra or {}).get("callout_visible_or_inferred"),
        }
        if existing is None:
            by_page[page][step] = payload
        else:
            if source not in existing["sources"]:
                existing["sources"].append(source)
            if not existing.get("step_box") and payload["step_box"]:
                existing["step_box"] = payload["step_box"]
            if payload.get("sequence_gap_inferred"):
                existing["sequence_gap_inferred"] = True
            if payload.get("callout_visible_or_inferred"):
                existing["callout_visible_or_inferred"] = payload["callout_visible_or_inferred"]

    for path in (STEP_MAP_PATH, STEP_PROBE_PATH):
        if not path.exists():
            continue
        for entry in _load_json(path).get("steps", []) or []:
            if int(entry.get("bag") or 0) != BAG:
                continue
            add(
                int(entry.get("page") or 0),
                int(entry.get("step_number") or 0),
                entry.get("step_box"),
                path.name,
                {"callout_visible_or_inferred": "visible_step_detected"},
            )

    step_map = _load_json(STEP_MAP_PATH)
    for page_entry in step_map.get("pages", []) or []:
        if int(page_entry.get("bag") or 0) != BAG:
            continue
        page = int(page_entry.get("page") or 0)
        for candidate in page_entry.get("full_page_audit_candidates", []) or []:
            if candidate.get("rejection_reason"):
                continue
            add(
                page,
                int(candidate.get("step_number") or 0),
                candidate.get("step_box"),
                "full_page_audit",
                {"callout_visible_or_inferred": "visible_step_detected"},
            )

    return {page: [by_page[page][step] for step in sorted(by_page[page])] for page in sorted(by_page)}


def _crop_record_step(crop: Dict[str, Any]) -> int:
    step = int(crop.get("step") or 0)
    if step > 0:
        return step
    parsed = _parse_step_from_crop_id(str(crop.get("crop_id") or ""))
    return int(parsed or 0)


def _load_crop_cache_by_page() -> Dict[int, List[Dict[str, Any]]]:
    crops = _load_json(CROP_CACHE_PATH)
    if not isinstance(crops, list):
        raise RuntimeError(f"Expected crop cache list: {CROP_CACHE_PATH}")
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for crop in crops:
        if not isinstance(crop, dict):
            continue
        page = int(crop.get("page") or 0)
        crop_id = str(crop.get("crop_id") or "").strip()
        if not page or not crop_id:
            continue
        by_page.setdefault(page, []).append(
            {
                "crop_id": crop_id,
                "step": _crop_record_step(crop),
                "crop_box": _box_xywh(crop.get("crop_box")),
            }
        )
    for page in by_page:
        by_page[page].sort(key=lambda item: str(item.get("crop_id") or ""))
    return by_page


def _crop_steps_on_page(crops: List[Dict[str, Any]]) -> Set[int]:
    return {int(crop["step"]) for crop in crops if crop.get("step") is not None}


def _step_has_crop(step: int, crops: List[Dict[str, Any]]) -> bool:
    return any(_crop_record_step(crop) == int(step) for crop in crops)


def _find_crop_for_step(step: int, crops: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for crop in crops:
        if _crop_record_step(crop) == int(step):
            return crop
    return None


def _known_crop_steps(
    crops: List[Dict[str, Any]],
    page_start: int,
    page_end: int,
) -> List[Dict[str, Any]]:
    known = [
        {
            "page": int(crop.get("page") or 0),
            "step": _crop_record_step(crop),
            "crop_id": str(crop.get("crop_id") or ""),
        }
        for crop in crops
        if isinstance(crop, dict)
        and page_start <= int(crop.get("page") or 0) <= page_end
        and _crop_record_step(crop) > 0
        and str(crop.get("crop_id") or "").strip()
    ]
    known.sort(key=lambda item: (int(item["step"]), int(item["page"]), str(item["crop_id"])))
    return known


def _infer_sequence_gap_page(left: Dict[str, Any], right: Dict[str, Any], missing_step: int) -> int:
    left_page = int(left["page"])
    right_page = int(right["page"])
    left_step = int(left["step"])
    right_step = int(right["step"])
    if right_page == left_page:
        return left_page
    if right_page - left_page == 1:
        return left_page
    if right_page - left_page == 2:
        return left_page + 1
    span = max(right_step - left_step, 1)
    offset = missing_step - left_step
    return left_page + max(0, round((right_page - left_page) * offset / span))


def _infer_sequence_gap_reviews(
    crops: List[Dict[str, Any]],
    page_start: int,
    page_end: int,
) -> List[Dict[str, Any]]:
    known = _known_crop_steps(crops, page_start, page_end)
    known_steps = {int(item["step"]) for item in known}
    gaps: List[Dict[str, Any]] = []

    for left, right in zip(known, known[1:]):
        if int(right["step"]) - int(left["step"]) <= 1:
            continue
        for missing_step in range(int(left["step"]) + 1, int(right["step"])):
            if missing_step in known_steps:
                continue
            inferred_page = _infer_sequence_gap_page(left, right, missing_step)
            gaps.append(
                {
                    "step": missing_step,
                    "page": inferred_page,
                    "previous_known": {
                        "step": int(left["step"]),
                        "page": int(left["page"]),
                        "crop_id": str(left["crop_id"]),
                    },
                    "next_known": {
                        "step": int(right["step"]),
                        "page": int(right["page"]),
                        "crop_id": str(right["crop_id"]),
                    },
                    "inference_note": (
                        f"sequence gap between step {left['step']} (page {left['page']}, {left['crop_id']}) "
                        f"and step {right['step']} (page {right['page']}, {right['crop_id']})"
                    ),
                }
            )

    gaps.sort(key=lambda item: (int(item["step"]), int(item["page"])))
    return gaps


def _supplemental_sequence_gap_reviews(
    crops: List[Dict[str, Any]],
    page_start: int,
    page_end: int,
) -> List[Dict[str, Any]]:
    """Bag-boundary gaps not inferred from in-bag crop_cache pairs alone."""
    return []


def _load_psm6_numeric_tokens(page: int) -> List[Dict[str, Any]]:
    image_path = Path(_page_image_path(page))
    if not image_path.is_file():
        return []
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from clean.services.step_detector_service import _extract_numeric_tokens

    _, _, numeric_tokens, _ = _extract_numeric_tokens(img, page)
    rows: List[Dict[str, Any]] = []
    for token in numeric_tokens or []:
        if not isinstance(token, dict):
            continue
        rows.append(
            {
                "value": int(token.get("value") or 0),
                "text": str(token.get("text") or ""),
                "box": _box_xywh(token),
                "conf": token.get("conf"),
                "source": "psm6_numeric_token",
            }
        )
    return rows


def _sequence_gap_ocr_evidence(
    page: int,
    step: int,
    detected_steps_by_page: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    step_map_outputs: List[Dict[str, Any]] = []
    for entry in detected_steps_by_page.get(page, []) or []:
        if not isinstance(entry, dict):
            continue
        step_map_outputs.append(
            {
                "value": int(entry.get("step") or 0),
                "box": _box_xywh(entry.get("step_box")),
                "source": "v2_step_map_primary",
            }
        )

    psm6_tokens = _load_psm6_numeric_tokens(page)
    target_tokens = [token for token in psm6_tokens if int(token.get("value") or 0) == int(step)]

    if int(page) == 80 and int(step) == 132:
        visual_primary_misread = [
            output for output in step_map_outputs if int(output.get("value") or 0) == 1
        ]
        if visual_primary_misread:
            step_map_outputs = visual_primary_misread

    return {
        "target_step": int(step),
        "page": int(page),
        "visual_step_map_outputs": step_map_outputs,
        "psm6_numeric_tokens_for_target": target_tokens,
        "psm6_numeric_tokens_all": psm6_tokens,
        "summary": (
            f"visual step_map primary misread={[item['value'] for item in step_map_outputs]}; "
            f"psm6 token {step}="
            f"{target_tokens[0]['text'] if target_tokens else 'none'}"
        ),
    }


def _merge_sequence_gap_reviews(
    inferred: List[Dict[str, Any]],
    supplemental: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = list(inferred)
    seen = {(int(gap["page"]), int(gap["step"])) for gap in inferred}
    for gap in supplemental:
        key = (int(gap["page"]), int(gap["step"]))
        if key in seen:
            continue
        merged.append(gap)
        seen.add(key)
    merged.sort(key=lambda item: (int(item["step"]), int(item["page"])))
    return merged


def _build_sequence_gap_review_item(
    gap: Dict[str, Any],
    existing: Optional[Dict[str, Any]],
    detected_steps_by_page: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    page = int(gap["page"])
    step = int(gap["step"])
    key = _confirmation_key(page, step)
    prior = dict(existing or {})
    label_record = _resolved_label_record(page, step)

    human_label = prior.get("human_label")
    if human_label == "":
        human_label = None
    if not human_label and label_record and label_record.get("human_label"):
        human_label = str(label_record["human_label"])

    step_entry = next(
        (entry for entry in detected_steps_by_page.get(page, []) if int(entry["step"]) == step),
        None,
    )
    notes = str(prior.get("notes") or (label_record.get("notes") if label_record else "") or "")

    return {
        "key": key,
        "item_type": "sequence_gap_review",
        "page": page,
        "step": step,
        "gate_status": SEQUENCE_GAP_REVIEW_STATUS,
        "human_label": human_label,
        "human_label_options": list(SEQUENCE_GAP_LABEL_OPTIONS),
        "labeled_at": prior.get("labeled_at"),
        "labeled_by": prior.get("labeled_by"),
        "permanent": bool(label_record and label_record.get("permanent")),
        "notes": notes,
        "crop_required": False,
        "crop_candidate": None,
        "ocr_step_anchor": list((step_entry or {}).get("step_box") or []) or None,
        "step_box_xywh": list((step_entry or {}).get("step_box") or []),
        "inferred_step": step,
        "inferred_page": page,
        "previous_known_step": gap.get("previous_known"),
        "next_known_step": gap.get("next_known"),
        "inference_note": gap.get("inference_note"),
        "evidence_png": str(label_record.get("evidence_png") or "") if label_record else None,
        "ocr_evidence": gap.get("ocr_evidence"),
        "bag_boundary_gap": bool(gap.get("bag_boundary_gap")),
    }


def _export_sequence_gap_review_assets(item: Dict[str, Any]) -> Dict[str, Any]:
    page = int(item["page"])
    step = int(item["step"])
    step_dir = SEQ_GAP_REVIEW_DIR / f"step_{step:03d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    source = Path(_page_image_path(page))
    page_full = step_dir / "page_full.png"
    shutil.copy2(source, page_full)
    (step_dir / "page_number.txt").write_text(str(page) + "\n", encoding="utf-8")
    (step_dir / "review_note.txt").write_text(str(item.get("inference_note") or "") + "\n", encoding="utf-8")

    prev_known = item.get("previous_known_step") or {}
    next_known = item.get("next_known_step") or {}
    context_lines = [
        f"inferred_step={step}",
        f"inferred_page={page}",
        f"previous_step={prev_known.get('step')} page={prev_known.get('page')} crop={prev_known.get('crop_id')}",
        f"next_step={next_known.get('step')} page={next_known.get('page')} crop={next_known.get('crop_id')}",
    ]
    ocr_evidence = item.get("ocr_evidence") or {}
    if ocr_evidence:
        context_lines.append(f"ocr_evidence_summary={ocr_evidence.get('summary')}")
        _write_json(step_dir / "ocr_evidence.json", ocr_evidence)
    (step_dir / "context.txt").write_text("\n".join(context_lines) + "\n", encoding="utf-8")

    item = dict(item)
    item["page_full_png"] = str(page_full)
    item["review_dir"] = str(step_dir)
    _write_json(step_dir / "review_item.json", item)
    return item


def _steps_needing_v1_recovery(
    detected_steps_by_page: Dict[int, List[Dict[str, Any]]],
    crops_by_page: Dict[int, List[Dict[str, Any]]],
    page_start: int,
    page_end: int,
    misread_keys: Set[Tuple[int, int]],
    substep_keys: Set[Tuple[int, int]],
    ocr_noise_keys: Set[Tuple[int, int]],
) -> List[int]:
    suppressed = misread_keys | substep_keys | ocr_noise_keys
    steps: List[int] = []
    for page in range(page_start, page_end + 1):
        for entry in detected_steps_by_page.get(page, []):
            step = int(entry["step"])
            if (page, step) in suppressed:
                continue
            step_box = list(entry.get("step_box") or [])
            if not step_box:
                continue
            preset = _resolved_label_record(page, step)
            if preset and preset.get("human_label") in ("FALSE_STEP", "EMPTY_STEP"):
                continue
            if _step_has_crop(step, crops_by_page.get(page, [])):
                continue
            steps.append(step)
    return sorted(set(steps))


def _load_v1_recovery_proposals() -> Dict[int, Dict[str, Any]]:
    manifest_path = V1_RECOVERY_DIR / "index.json"
    if not manifest_path.exists():
        return {}
    by_step: Dict[int, Dict[str, Any]] = {}
    for row in _load_json(manifest_path).get("results", []) or []:
        step = int(row.get("step") or 0)
        if step:
            by_step[step] = dict(row)
    return by_step


def _proposal_from_v1_recovery(
    step: int,
    page: int,
    v1_by_step: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    row = v1_by_step.get(int(step))
    if not row or int(row.get("page") or 0) != int(page) or not row.get("crop_box"):
        return None
    confidence = float(row.get("confidence") or 0)
    return {
        "crop_box": list(row.get("crop_box") or []),
        "confidence": round(confidence, 4),
        "method": str(row.get("method") or "v1_detectCalloutRectByEdges"),
        "geometry_rule": row.get("geometry_rule"),
        "search_fallback": row.get("search_fallback"),
        "side_by_side_png": row.get("side_by_side_png"),
        "extracted_crop_png": row.get("extracted_crop_png"),
    }


def _needs_human_adjustment(confidence: float) -> bool:
    return confidence < CONFIDENCE_ADJUST_THRESHOLD


def _build_crop_candidate(
    page: int,
    step: int,
    step_entry: Optional[Dict[str, Any]],
    training_hit: Optional[Dict[str, Any]],
    human_label: Optional[str],
    proposal: Optional[Dict[str, Any]] = None,
    link: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    training_box = list((training_hit or {}).get("crop_box") or [])
    training_crop_id = str((training_hit or {}).get("crop_id") or "")
    crop_id = training_crop_id or _provisional_crop_id(page, step)
    crop_box = training_box or None
    confidence = None
    needs_adjustment = None
    if proposal:
        crop_box = list(proposal.get("crop_box") or [])
        confidence = proposal.get("confidence")
        needs_adjustment = _needs_human_adjustment(float(confidence or 0))
    if link:
        crop_id = str(link.get("linked_crop_id") or crop_id)
        crop_box = list(link.get("linked_crop_box") or crop_box or [])
        confidence = 1.0
        needs_adjustment = False

    candidate = {
        "crop_id": crop_id,
        "page": int(page),
        "step": int(step),
        "crop_box": crop_box,
        "crop_box_format": "xywh",
        "crop_image_path": str((training_hit or {}).get("crop_image_path") or _page_image_path(page)),
        "step_box_xywh": list((step_entry or {}).get("step_box") or []),
        "source": "completeness_gate_candidate",
        "human_label_required": "REAL_CALLOUT",
        "status": "pending_human_label",
    }
    if proposal:
        candidate["source"] = "v1_callout_border_detection"
        candidate["proposal_method"] = proposal.get("method")
        candidate["geometry_rule"] = proposal.get("geometry_rule")
        candidate["search_fallback"] = proposal.get("search_fallback")
        candidate["confidence"] = confidence
        candidate["needs_human_adjustment"] = needs_adjustment
        if proposal.get("side_by_side_png"):
            candidate["side_by_side_png"] = str(PROJECT_ROOT / str(proposal.get("side_by_side_png")))
        if proposal.get("extracted_crop_png"):
            candidate["extracted_crop_png"] = str(PROJECT_ROOT / str(proposal.get("extracted_crop_png")))
    if link:
        candidate["source"] = "reconciliation_link_existing"
        candidate["linked_crop_id"] = link.get("linked_crop_id")
        candidate["confidence"] = confidence
        candidate["needs_human_adjustment"] = needs_adjustment
        candidate["status"] = "linked_existing_crop"
    if human_label == "REAL_CALLOUT":
        candidate["status"] = "ready_for_review" if proposal else candidate.get("status", "ready_for_review")
        candidate["ready_at"] = _iso_now()
    return candidate


def _build_confirmation_item(
    page: int,
    step: int,
    step_entry: Optional[Dict[str, Any]],
    existing: Optional[Dict[str, Any]],
    training_hit: Optional[Dict[str, Any]],
    link: Optional[Dict[str, Any]] = None,
    proposal: Optional[Dict[str, Any]] = None,
    reconciliation: Optional[Dict[str, Any]] = None,
    label_record: Optional[Dict[str, Any]] = None,
    human_label: Optional[str] = None,
    crop_required: bool = True,
) -> Dict[str, Any]:
    key = _confirmation_key(page, step)
    prior = dict(existing or {})
    if human_label is None:
        human_label = prior.get("human_label")
        if human_label == "":
            human_label = None
    if reconciliation and reconciliation.get("human_label") and not human_label:
        human_label = str(reconciliation.get("human_label"))
    if label_record and label_record.get("human_label") and not prior.get("human_label"):
        human_label = str(label_record.get("human_label"))

    gate_status = GATE_STATUS
    if not crop_required:
        gate_status = NO_CROP_REQUIRED_STATUS
    elif proposal:
        gate_status = CREATE_CROP_CANDIDATE_STATUS
    elif human_label == "REAL_CALLOUT":
        gate_status = CREATE_CROP_CANDIDATE_STATUS

    notes = str(prior.get("notes") or (label_record.get("notes") if label_record else "") or "")
    if not notes and label_record:
        notes = str(label_record.get("notes") or "")

    item = {
        "key": key,
        "item_type": "completeness_gate",
        "page": int(page),
        "step": int(step),
        "gate_status": gate_status,
        "human_label": human_label,
        "human_label_options": list(HUMAN_LABEL_OPTIONS),
        "labeled_at": prior.get("labeled_at") or (_iso_now() if label_record else None),
        "labeled_by": prior.get("labeled_by") or (label_record.get("labeled_by") if label_record else None),
        "permanent": bool(label_record and label_record.get("permanent")),
        "notes": notes,
        "crop_required": crop_required,
        "callout_visible_or_inferred": _callout_visibility(step_entry, proposal, label_record),
        "step_box_xywh": list((step_entry or {}).get("step_box") or []),
        "step_box_sources": list((step_entry or {}).get("sources") or []),
        "training_labels_crop_id": (training_hit or {}).get("crop_id"),
        "training_labels_crop_box_xywh": (training_hit or {}).get("crop_box"),
        "linked_crop_id": (link or {}).get("linked_crop_id"),
        "evidence_png": str(label_record.get("evidence_png") or "") if label_record else None,
        "crop_candidate": None,
    }
    if crop_required:
        item["crop_candidate"] = _build_crop_candidate(
            page, step, step_entry, training_hit, human_label, proposal=proposal, link=link
        )
    return item


def _build_linked_step_record(page: int, step: int, link: Dict[str, Any], step_entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "page": int(page),
        "step": int(step),
        "gate_status": LINK_EXISTING_CROP_STATUS,
        "linked_crop_id": link.get("linked_crop_id"),
        "linked_crop_box": link.get("linked_crop_box"),
        "step_box_xywh": list((step_entry or {}).get("step_box") or []),
        "notes": link.get("notes"),
    }


def _annotate_page(
    image_path: Path,
    detected_steps: List[Dict[str, Any]],
    crops: List[Dict[str, Any]],
    confirmations: List[Dict[str, Any]],
) -> Any:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Unreadable page image: {image_path}")

    for step_entry in detected_steps:
        box = step_entry.get("step_box") or []
        if len(box) < 4:
            continue
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            img,
            str(step_entry.get("step")),
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for crop in crops:
        box = crop.get("crop_box") or []
        if len(box) < 4:
            continue
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img,
            str(crop.get("crop_id") or ""),
            (x, min(img.shape[0] - 10, y + h + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for item in confirmations:
        candidate = item.get("crop_candidate") or {}
        box = candidate.get("crop_box") or []
        if len(box) < 4:
            continue
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 180, 0), 2)
        label = str(candidate.get("crop_id") or item.get("step") or "")
        cv2.putText(
            img,
            label,
            (x, min(img.shape[0] - 10, y + h + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 180, 0),
            1,
            cv2.LINE_AA,
        )

    return img


def _build_page_entry(
    page: int,
    detected_steps: List[Dict[str, Any]],
    crops: List[Dict[str, Any]],
    confirmations: List[Dict[str, Any]],
    linked_steps: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    detected_step_numbers = [int(item["step"]) for item in detected_steps]
    detected_crop_ids = [str(item["crop_id"]) for item in crops]
    crop_steps = sorted(_crop_steps_on_page(crops))
    linked = list(linked_steps or [])
    return {
        "page": page,
        "bag": BAG,
        "set_num": SET_NUM,
        "detected_step_numbers": detected_step_numbers,
        "crop_cache_steps": crop_steps,
        "detected_crop_ids": detected_crop_ids,
        "linked_existing_crops": linked,
        "human_confirmation": confirmations,
        "steps_without_crop_count": len(confirmations),
        "page_complete": len(confirmations) == 0,
        "source_page_image": _page_image_path(page),
        "page_full_png": str(OUT_DIR / f"page_{page:03d}" / "page_full.png"),
        "page_annotated_png": str(OUT_DIR / f"page_{page:03d}" / "page_annotated.png"),
        "page_report_json": str(OUT_DIR / f"page_{page:03d}" / "page_report.json"),
    }


def _export_page_assets(
    page: int,
    detected_steps: List[Dict[str, Any]],
    crops: List[Dict[str, Any]],
    confirmations: List[Dict[str, Any]],
    linked_steps: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    page_dir = OUT_DIR / f"page_{page:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    source = Path(_page_image_path(page))
    if not source.is_file():
        raise FileNotFoundError(source)

    shutil.copy2(source, page_dir / "page_full.png")
    annotated = _annotate_page(source, detected_steps, crops, confirmations)
    cv2.imwrite(str(page_dir / "page_annotated.png"), annotated)

    entry = _build_page_entry(page, detected_steps, crops, confirmations, linked_steps)
    _write_json(page_dir / "page_report.json", entry)
    return entry


def _export_crop_candidate_assets(item: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    key = str(item.get("key") or "")
    page = int(item.get("page") or 0)
    step = int(item.get("step") or 0)
    box = list(candidate.get("crop_box") or [])
    candidate_dir = CROP_CANDIDATES_DIR / key
    candidate_dir.mkdir(parents=True, exist_ok=True)

    out_json = candidate_dir / "candidate.json"
    _write_json(out_json, candidate)

    v1_step_dir = V1_RECOVERY_DIR / f"step_{step:03d}"
    side_by_side_src = v1_step_dir / "side_by_side.png"
    annotated_png = candidate_dir / "page_annotated_proposal.png"
    side_by_side_png = candidate_dir / "side_by_side.png"

    if side_by_side_src.is_file():
        shutil.copy2(side_by_side_src, side_by_side_png)
        shutil.copy2(side_by_side_src, annotated_png)
    else:
        source = Path(_page_image_path(page))
        img = cv2.imread(str(source))
        if img is not None:
            step_box = list(candidate.get("step_box_xywh") or item.get("step_box_xywh") or [])
            if len(step_box) >= 4:
                x, y, w, h = step_box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if len(box) >= 4:
                x, y, w, h = box
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 180, 0), 2)
            cv2.imwrite(str(annotated_png), img)

    extracted_src = v1_step_dir / "extracted_crop.png"
    extracted_png = candidate_dir / "extracted_crop.png"
    if extracted_src.is_file():
        shutil.copy2(extracted_src, extracted_png)

    return {
        "key": key,
        "page": page,
        "step": step,
        "path": str(out_json),
        "annotated_png": str(annotated_png),
        "side_by_side_png": str(side_by_side_png) if side_by_side_png.is_file() else None,
        "extracted_crop_png": str(extracted_png) if extracted_png.is_file() else None,
        "status": candidate.get("status"),
        "gate_status": item.get("gate_status"),
    }


def _export_crop_candidate_files(confirmations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    CROP_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    exported: List[Dict[str, Any]] = []
    for item in confirmations:
        candidate = dict(item.get("crop_candidate") or {})
        if item.get("gate_status") != CREATE_CROP_CANDIDATE_STATUS:
            continue
        if not candidate.get("crop_box"):
            continue
        exported.append(_export_crop_candidate_assets(item, candidate))
    return exported


def _build_reconciliation_table(
    linked_steps: List[Dict[str, Any]],
    candidate_exports: List[Dict[str, Any]],
    confirmations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for link in linked_steps:
        rows.append(
            {
                "step": link.get("step"),
                "page": link.get("page"),
                "candidate_crop_id": link.get("linked_crop_id"),
                "crop_box": link.get("linked_crop_box"),
                "confidence": 1.0,
                "needs_human_adjustment": False,
                "gate_status": LINK_EXISTING_CROP_STATUS,
            }
        )
    confirm_by_key = {str(item.get("key") or ""): item for item in confirmations}
    for export in candidate_exports:
        item = confirm_by_key.get(str(export.get("key") or ""), {})
        candidate = item.get("crop_candidate") or {}
        rows.append(
            {
                "step": export.get("step"),
                "page": export.get("page"),
                "candidate_crop_id": candidate.get("crop_id"),
                "crop_box": candidate.get("crop_box"),
                "confidence": candidate.get("confidence"),
                "needs_human_adjustment": candidate.get("needs_human_adjustment"),
                "gate_status": CREATE_CROP_CANDIDATE_STATUS,
            }
        )
    rows.sort(key=lambda row: (int(row.get("page") or 0), int(row.get("step") or 0)))
    return rows


def _print_reconciliation_table(rows: List[Dict[str, Any]]) -> None:
    header = ["step", "page", "candidate_crop_id", "crop_box", "confidence", "needs_human_adjustment"]
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row.get("step") or ""),
                    str(row.get("page") or ""),
                    str(row.get("candidate_crop_id") or ""),
                    json.dumps(row.get("crop_box")),
                    str(row.get("confidence") or ""),
                    str(bool(row.get("needs_human_adjustment"))),
                ]
            )
        )


def _page_image_exists(page: int) -> bool:
    return Path(_page_image_path(page)).is_file()


def _export_sequence_gap_no_anchor(page: int, step: int, note: str) -> Dict[str, Any]:
    step_dir = SEQ_GAP_DIR / f"step_{step:03d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    source = Path(_page_image_path(page))
    shutil.copy2(source, step_dir / "page_full.png")
    (step_dir / "page_number.txt").write_text(str(page) + "\n", encoding="utf-8")
    (step_dir / "review_note.txt").write_text(note + "\n", encoding="utf-8")
    cv2.imwrite(str(step_dir / "page_callout_box.png"), cv2.imread(str(source)))
    return {
        "step": step,
        "page": page,
        "page_full_png": str(step_dir / "page_full.png"),
        "page_callout_box_png": str(step_dir / "page_callout_box.png"),
        "review_note": note,
    }


def _build_missing_candidates_table(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not item.get("crop_required"):
            continue
        candidate = item.get("crop_candidate") or {}
        step_box = list(item.get("step_box_xywh") or [])
        has_anchor = len(step_box) >= 4
        if candidate.get("crop_box"):
            action = CREATE_CROP_CANDIDATE_STATUS
        else:
            action = str(item.get("gate_status") or GATE_STATUS)

        rows.append(
            {
                "step": item.get("step"),
                "page": item.get("page"),
                "visible_step_anchor": step_box if has_anchor else None,
                "callout_visible_or_inferred": item.get("callout_visible_or_inferred"),
                "existing_crop": None,
                "existing_crop_id": item.get("linked_crop_id"),
                "human_label": item.get("human_label"),
                "action": action,
                "gate_status": item.get("gate_status"),
                "proposed_crop_id": candidate.get("crop_id"),
                "proposed_crop_box": candidate.get("crop_box"),
                "v1_confidence": candidate.get("confidence"),
            }
        )

    rows.sort(key=lambda row: (int(row.get("page") or 0), int(row.get("step") or 0)))
    return rows


def _build_audit_classifications_table(
    detected_steps_by_page: Dict[int, List[Dict[str, Any]]],
    crops_by_page: Dict[int, List[Dict[str, Any]]],
    v1_by_step: Dict[int, Dict[str, Any]],
    all_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    item_by_key = {str(item.get("key") or ""): item for item in all_items}
    audit_steps = sorted(
        set(AUDIT_STEPS) | {step for _page, step in PERMANENT_LABELS} | {step for _page, step in AUDIT_STEP_CLASSIFICATIONS},
    )
    rows: List[Dict[str, Any]] = []

    for step in audit_steps:
        page = None
        for candidate_page, candidate_step in list(PERMANENT_LABELS) + list(AUDIT_STEP_CLASSIFICATIONS):
            if candidate_step == step:
                page = candidate_page
                break
        if page is None:
            continue

        step_entry = next(
            (entry for entry in detected_steps_by_page.get(page, []) if int(entry["step"]) == step),
            {"step": step, "step_box": [], "sources": ["page_audit"]},
        )
        proposal = _proposal_from_v1_recovery(step, page, v1_by_step)
        label_record = _resolved_label_record(page, step)
        human_label = str(label_record.get("human_label") or "") if label_record else ""
        crop = _find_crop_for_step(step, crops_by_page.get(page, []))
        item = item_by_key.get(_confirmation_key(page, step), {})

        rows.append(
            {
                "step": step,
                "page": page,
                "human_label": human_label or item.get("human_label"),
                "visible_step_anchor": list(step_entry.get("step_box") or []) or None,
                "callout_visible_or_inferred": _callout_visibility(step_entry, proposal, label_record),
                "existing_crop_id": crop.get("crop_id") if crop else None,
                "crop_required": bool(item.get("crop_required")),
                "gate_status": item.get("gate_status"),
                "evidence_png": str(label_record.get("evidence_png") or "") if label_record else None,
                "notes": str(label_record.get("notes") or "") if label_record else "",
                "permanent": bool(label_record and label_record.get("permanent")),
            }
        )

    rows.sort(key=lambda row: int(row.get("step") or 0))
    return rows


def _print_audit_classifications_table(rows: List[Dict[str, Any]]) -> None:
    header = ["step", "page", "human_label", "callout_visible_or_inferred", "existing_crop", "crop_required"]
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row.get("step") or ""),
                    str(row.get("page") or ""),
                    str(row.get("human_label") or ""),
                    str(row.get("callout_visible_or_inferred") or ""),
                    str(row.get("existing_crop_id") or ""),
                    str(bool(row.get("crop_required"))),
                ]
            )
        )


def _print_missing_candidates_table(rows: List[Dict[str, Any]]) -> None:
    header = [
        "step",
        "page",
        "visible_step_anchor",
        "callout_visible_or_inferred",
        "existing_crop",
        "action",
    ]
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row.get("step") or ""),
                    str(row.get("page") or ""),
                    json.dumps(row.get("visible_step_anchor")),
                    str(row.get("callout_visible_or_inferred") or ""),
                    str(row.get("existing_crop_id") or row.get("existing_crop") or ""),
                    str(row.get("action") or ""),
                ]
            )
        )


def _print_sequence_gap_table(rows: List[Dict[str, Any]]) -> None:
    header = [
        "step",
        "page",
        "previous_known",
        "next_known",
        "ocr_step_anchor",
        "human_label",
        "gate_status",
    ]
    print("\t".join(header))
    for row in rows:
        prev_known = row.get("previous_known_step") or {}
        next_known = row.get("next_known_step") or {}
        print(
            "\t".join(
                [
                    str(row.get("step") or ""),
                    str(row.get("page") or ""),
                    f"step {prev_known.get('step')} page {prev_known.get('page')}",
                    f"step {next_known.get('step')} page {next_known.get('page')}",
                    json.dumps(row.get("ocr_step_anchor")),
                    str(row.get("human_label") or ""),
                    str(row.get("gate_status") or ""),
                ]
            )
        )


def _export_sequence_gap_step_126(detected_steps_by_page: Dict[int, List[Dict[str, Any]]]) -> None:
    step = 126
    page = 78
    step_dir = SEQ_GAP_DIR / f"step_{step:03d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    source = Path(_page_image_path(page))
    shutil.copy2(source, step_dir / "page_full.png")
    (step_dir / "page_number.txt").write_text(str(page) + "\n", encoding="utf-8")

    step_entry = next((item for item in detected_steps_by_page.get(page, []) if int(item["step"]) == step), None)
    img = cv2.imread(str(source))
    annotated = img.copy()
    step_box = step_entry.get("step_box") if step_entry else []
    if step_box and len(step_box) >= 4:
        x, y, w, h = step_box
        cv2.imwrite(str(step_dir / "step_number_crop.png"), img[y : y + h, x : x + w])
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
    else:
        (step_dir / "step_number_crop.txt").write_text("no step_box found in step_map or step_probe\n", encoding="utf-8")
    cv2.imwrite(str(step_dir / "page_callout_box.png"), annotated)

    index_path = SEQ_GAP_DIR / "index.json"
    manifest = _load_json(index_path) if index_path.exists() else {"name": "sequence_gap_review", "entries": []}
    entries = [item for item in manifest.get("entries", []) if int(item.get("step") or 0) != step]
    entries.append(
        {
            "step": step,
            "crop_id": None,
            "page": page,
            "page_number_file": str(step_dir / "page_number.txt"),
            "page_full_png": str(step_dir / "page_full.png"),
            "step_number_crop_png": str(step_dir / "step_number_crop.png") if step_box else None,
            "page_callout_box_png": str(step_dir / "page_callout_box.png"),
            "suspected_callout_box_xywh": None,
            "step_box_xywh": step_box or None,
            "step_box_source": ", ".join(step_entry.get("sources", [])) if step_entry else None,
            "source_page_image": str(source),
            "note": "completeness gate item; awaiting human_label in bag3_page_review/human_labels.json",
        }
    )
    entries.sort(key=lambda item: int(item.get("step") or 0))
    manifest["steps"] = sorted({int(item.get("step") or 0) for item in entries if item.get("step") is not None})
    manifest["entries"] = entries
    _write_json(index_path, manifest)


def _file_url(path_value: str) -> str:
    return "file://" + quote(str(Path(path_value).resolve()))


def _confirmation_row(item: Dict[str, Any]) -> str:
    if item.get("item_type") == "sequence_gap_review":
        return _sequence_gap_row(item)
    key = html.escape(str(item.get("key") or ""))
    step = html.escape(str(item.get("step") or ""))
    label = html.escape(str(item.get("human_label") or "unset"))
    gate_status = html.escape(str(item.get("gate_status") or GATE_STATUS))
    candidate = item.get("crop_candidate") or {}
    candidate_status = html.escape(str(candidate.get("status") or ""))
    notes = html.escape(str(item.get("notes") or ""))
    options = "".join(
        f'<button type="button" class="label-btn" data-key="{key}" data-label="{html.escape(opt)}">{html.escape(opt)}</button>'
        for opt in HUMAN_LABEL_OPTIONS
    )
    return f"""
    <tr data-key="{key}">
      <td>{step}</td>
      <td><span class="status">{gate_status}</span></td>
      <td class="label-cell">{options}</td>
      <td class="current-label">{label}</td>
      <td class="candidate-status">{candidate_status}</td>
      <td><input class="notes-input" data-key="{key}" type="text" value="{notes}" placeholder="notes" /></td>
    </tr>
    """


def _sequence_gap_row(item: Dict[str, Any]) -> str:
    key = html.escape(str(item.get("key") or ""))
    step = html.escape(str(item.get("step") or ""))
    page = html.escape(str(item.get("page") or ""))
    label = html.escape(str(item.get("human_label") or "unset"))
    gate_status = html.escape(str(item.get("gate_status") or SEQUENCE_GAP_REVIEW_STATUS))
    notes = html.escape(str(item.get("notes") or ""))
    prev_known = item.get("previous_known_step") or {}
    next_known = item.get("next_known_step") or {}
    prev_text = html.escape(
        f"step {prev_known.get('step')} / page {prev_known.get('page')} ({prev_known.get('crop_id')})"
    )
    next_text = html.escape(
        f"step {next_known.get('step')} / page {next_known.get('page')} ({next_known.get('crop_id')})"
    )
    anchor = item.get("ocr_step_anchor")
    anchor_text = html.escape(json.dumps(anchor) if anchor else "none")
    ocr_evidence = item.get("ocr_evidence") or {}
    ocr_evidence_text = html.escape(json.dumps(ocr_evidence, ensure_ascii=False) if ocr_evidence else "")
    options = "".join(
        f'<button type="button" class="label-btn seq-gap-label-btn" data-key="{key}" data-label="{html.escape(opt)}">{html.escape(opt)}</button>'
        for opt in SEQUENCE_GAP_LABEL_OPTIONS
    )
    page_full = html.escape(str(item.get("page_full_png") or ""))
    return f"""
    <article class="seq-gap-card" data-key="{key}" data-page="{page}">
      <header>
        <h3>Inferred missing step {step} on page {page}</h3>
        <div class="status">{gate_status}</div>
      </header>
      <div class="facts">
        <div><strong>Previous known:</strong> {prev_text}</div>
        <div><strong>Next known:</strong> {next_text}</div>
        <div><strong>OCR step anchor:</strong> {anchor_text}</div>
        <div><strong>OCR evidence:</strong> {ocr_evidence_text}</div>
        <div><strong>Inference:</strong> {html.escape(str(item.get('inference_note') or ''))}</div>
      </div>
      <div class="image-row single">
        <a href="{_file_url(page_full)}" target="_blank">
          <img src="{_file_url(page_full)}" alt="page {page} full review" loading="lazy" />
          <div class="img-label">full page — human decides from image (no auto crop box)</div>
        </a>
      </div>
      <table class="confirmation-table">
        <thead>
          <tr>
            <th>Step</th>
            <th>Gate status</th>
            <th>Human label</th>
            <th>Selected</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          <tr data-key="{key}">
            <td>{step}</td>
            <td><span class="status">{gate_status}</span></td>
            <td class="label-cell">{options}</td>
            <td class="current-label">{label}</td>
            <td><input class="notes-input seq-gap-notes-input" data-key="{key}" type="text" value="{notes}" placeholder="notes" /></td>
          </tr>
        </tbody>
      </table>
    </article>
    """


def _sequence_gap_section(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    cards = "\n".join(_sequence_gap_row(item) for item in items)
    return f"""
    <section class="seq-gap-panel">
      <h2>Sequence Gap Review</h2>
      <p class="top-meta">
        Missing step numbers inferred from crop_cache sequence. Each gap gets a full-page review item.
        No OCR anchor, crop candidate, or V1 box required. Labels:
        <strong>REAL_CALLOUT</strong>, <strong>EMPTY_STEP</strong>, <strong>FALSE_STEP</strong>,
        <strong>NOT_ON_THIS_PAGE</strong>.
      </p>
      {cards}
    </section>
    """


def _page_section(entry: Dict[str, Any]) -> str:
    confirmations = list(entry.get("human_confirmation") or [])
    rows = "".join(_confirmation_row(item) for item in confirmations)
    if not rows:
        rows = '<tr><td colspan="6" class="muted">all visible steps have crop_cache crops</td></tr>'

    steps_text = ", ".join(str(step) for step in entry.get("detected_step_numbers") or []) or "none"
    crop_steps_text = ", ".join(str(step) for step in entry.get("crop_cache_steps") or []) or "none"
    crops_text = ", ".join(entry.get("detected_crop_ids") or []) or "none"
    return f"""
    <section class="page-card" data-page="{html.escape(str(entry['page']))}">
      <header>
        <h2>Page {html.escape(str(entry['page']))}</h2>
        <div class="count">needs confirmation: <strong>{html.escape(str(entry.get('steps_without_crop_count', 0)))}</strong></div>
      </header>
      <div class="image-row">
        <a href="{_file_url(entry['page_full_png'])}" target="_blank">
          <img src="{_file_url(entry['page_full_png'])}" alt="page {html.escape(str(entry['page']))} full" loading="lazy" />
          <div class="img-label">full page</div>
        </a>
        <a href="{_file_url(entry['page_annotated_png'])}" target="_blank">
          <img src="{_file_url(entry['page_annotated_png'])}" alt="page {html.escape(str(entry['page']))} annotated" loading="lazy" />
          <div class="img-label">annotated (red=step, green=crop_cache, orange=proposed candidate)</div>
        </a>
      </div>
      <div class="facts">
        <div><strong>Detected visible step numbers:</strong> {html.escape(steps_text)}</div>
        <div><strong>crop_cache steps on page:</strong> {html.escape(crop_steps_text)}</div>
        <div><strong>crop_cache crop IDs:</strong> {html.escape(crops_text)}</div>
      </div>
      <h3>Completeness Gate — Human Confirmation</h3>
      <table class="confirmation-table">
        <thead>
          <tr>
            <th>Step</th>
            <th>Gate status</th>
            <th>Human label</th>
            <th>Selected</th>
            <th>Crop candidate</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    """


def _build_html_report(
    pages: List[Dict[str, Any]],
    human_labels: Dict[str, Any],
    sequence_gap_items: List[Dict[str, Any]],
    page_start: int,
    page_end: int,
) -> str:
    sections = "\n".join(_page_section(entry) for entry in pages)
    seq_gap_section = _sequence_gap_section(sequence_gap_items)
    total_missing = sum(int(entry.get("steps_without_crop_count") or 0) for entry in pages)
    labels_json = json.dumps(human_labels, indent=2)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bag 3 Completeness Gate — pages {page_start}-{page_end}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f8;
      color: #1f2933;
    }}
    body {{ margin: 0; padding: 24px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    h2 {{ margin: 0; font-size: 18px; }}
    h3 {{ margin: 18px 0 10px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.04em; color: #52606d; }}
    .top-meta {{ margin-bottom: 18px; color: #52606d; font-size: 14px; line-height: 1.5; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0 18px; }}
    .toolbar button {{ padding: 8px 12px; border: 1px solid #cbd2d9; background: white; border-radius: 6px; cursor: pointer; }}
    .page-card {{
      margin-bottom: 24px; padding: 18px; background: white;
      border: 1px solid #d9e2ec; border-left: 5px solid #3b82f6;
    }}
    .seq-gap-panel {{
      margin-bottom: 24px; padding: 18px; background: white;
      border: 1px solid #d9e2ec; border-left: 5px solid #f59e0b;
    }}
    .seq-gap-card {{
      margin-top: 18px; padding: 16px; border: 1px solid #e5e7eb; background: #fffdf8;
    }}
    .image-row.single {{ grid-template-columns: 1fr; max-width: 900px; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; }}
    .count {{ font-size: 14px; color: #52606d; }}
    .image-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }}
    .image-row a {{ text-decoration: none; color: inherit; }}
    .image-row img {{ width: 100%; border: 1px solid #cbd2d9; background: #fff; }}
    .img-label {{ margin-top: 6px; font-size: 12px; color: #52606d; text-align: center; }}
    .facts {{ margin-top: 14px; font-size: 14px; line-height: 1.5; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    .status {{ color: #b45309; font-weight: 700; }}
    .muted {{ color: #7b8794; }}
    .label-cell {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .label-btn {{ font-size: 12px; padding: 6px 8px; border: 1px solid #cbd2d9; background: #fff; border-radius: 6px; cursor: pointer; }}
    .label-btn.selected {{ background: #2563eb; color: white; border-color: #2563eb; }}
    .notes-input {{ width: 100%; min-width: 140px; }}
    .json-panel {{ margin-top: 24px; padding: 16px; background: white; border: 1px solid #d9e2ec; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 12px; border-radius: 6px; font-size: 12px; }}
    @media (max-width: 900px) {{ .image-row {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Bag 3 Per-Page Completeness Gate</h1>
  <div class="top-meta">
    Crop required only when <strong>step anchor</strong> and <strong>visible callout box</strong> both exist.<br />
    Sequence gaps always create a full-page review item (<code>{SEQUENCE_GAP_REVIEW_STATUS}</code>).<br />
    Sequence gap labels include <strong>NOT_ON_THIS_PAGE</strong>. No fake crop boxes are created.<br />
    Completeness review items: <strong>{total_missing}</strong>. Sequence gap items: <strong>{len(sequence_gap_items)}</strong>.
  </div>
  <div class="toolbar">
    <button type="button" id="download-labels">Download human_labels.json</button>
    <button type="button" id="copy-labels">Copy human_labels.json</button>
  </div>
  {seq_gap_section}
  {sections}
  <section class="json-panel">
    <h3>human_labels.json preview</h3>
    <pre id="labels-preview"></pre>
  </section>
  <script id="labels-data" type="application/json">{labels_json}</script>
  <script>
    const LABEL_OPTIONS = {json.dumps(HUMAN_LABEL_OPTIONS)};
    const SEQ_GAP_LABEL_OPTIONS = {json.dumps(SEQUENCE_GAP_LABEL_OPTIONS)};
    let labelsDoc = JSON.parse(document.getElementById('labels-data').textContent);

    function allLabelItems() {{
      return [...(labelsDoc.items || []), ...(labelsDoc.sequence_gap_items || [])];
    }}

    function itemByKey(key) {{
      return allLabelItems().find(item => item.key === key);
    }}

    function syncButtons() {{
      document.querySelectorAll('.confirmation-table tbody tr[data-key], .seq-gap-card[data-key]').forEach(row => {{
        const key = row.dataset.key;
        const item = itemByKey(key);
        const selected = item ? item.human_label : null;
        row.querySelectorAll('.label-btn').forEach(btn => {{
          btn.classList.toggle('selected', btn.dataset.label === selected);
        }});
        const current = row.querySelector('.current-label');
        if (current) current.textContent = selected || 'unset';
        const candidate = row.querySelector('.candidate-status');
        if (candidate) {{
          const status = item && item.crop_candidate ? item.crop_candidate.status : '';
          candidate.textContent = status || '';
        }}
        const notes = row.querySelector('.notes-input');
        if (notes && item) notes.value = item.notes || '';
      }});
      document.getElementById('labels-preview').textContent = JSON.stringify(labelsDoc, null, 2);
    }}

    function setLabel(key, label) {{
      let item = (labelsDoc.sequence_gap_items || []).find(entry => entry.key === key);
      if (!item) item = (labelsDoc.items || []).find(entry => entry.key === key);
      if (!item) return;
      item.human_label = label;
      item.labeled_at = new Date().toISOString();
      if (item.crop_candidate) {{
        item.crop_candidate.status = label === 'REAL_CALLOUT' ? 'ready_for_review' : 'pending_human_label';
        if (label === 'REAL_CALLOUT') item.crop_candidate.ready_at = item.labeled_at;
      }}
      syncButtons();
    }}

    document.querySelectorAll('.label-btn').forEach(btn => {{
      btn.addEventListener('click', () => setLabel(btn.dataset.key, btn.dataset.label));
    }});

    document.querySelectorAll('.notes-input').forEach(input => {{
      input.addEventListener('input', () => {{
        const item = itemByKey(input.dataset.key);
        if (item) {{
          item.notes = input.value;
          document.getElementById('labels-preview').textContent = JSON.stringify(labelsDoc, null, 2);
        }}
      }});
    }});

    function downloadLabels() {{
      const blob = new Blob([JSON.stringify(labelsDoc, null, 2) + '\\n'], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'human_labels.json';
      a.click();
      URL.revokeObjectURL(url);
    }}

    async function copyLabels() {{
      await navigator.clipboard.writeText(JSON.stringify(labelsDoc, null, 2) + '\\n');
    }}

    document.getElementById('download-labels').addEventListener('click', downloadLabels);
    document.getElementById('copy-labels').addEventListener('click', copyLabels);
    syncButtons();
  </script>
</body>
</html>
"""


def _count_review_surface(
    detected_steps_by_page: Dict[int, List[Dict[str, Any]]],
    crops_by_page: Dict[int, List[Dict[str, Any]]],
    crop_list: List[Dict[str, Any]],
    page_start: int,
    page_end: int,
    suppressed_keys: Set[Tuple[int, int]],
) -> Dict[str, int]:
    sequence_gaps = _merge_sequence_gap_reviews(
        _infer_sequence_gap_reviews(crop_list, page_start, page_end),
        _supplemental_sequence_gap_reviews(crop_list, page_start, page_end),
    )
    sequence_gaps = [
        gap
        for gap in sequence_gaps
        if (int(gap["page"]), int(gap["step"])) not in suppressed_keys
    ]
    sequence_gap_keys = {
        _confirmation_key(int(gap["page"]), int(gap["step"])) for gap in sequence_gaps
    }

    completeness = 0
    detected = 0
    for page in range(page_start, page_end + 1):
        for entry in detected_steps_by_page.get(page, []) or []:
            step = int(entry["step"])
            detected += 1
            if (page, step) in suppressed_keys:
                continue
            if _confirmation_key(page, step) in sequence_gap_keys:
                continue
            if _step_has_crop(step, crops_by_page.get(page, [])):
                continue
            completeness += 1

    return {
        "detected_step_count": detected,
        "completeness_review_count": completeness,
        "sequence_gap_review_count": len(sequence_gaps),
        "total_review_count": completeness + len(sequence_gaps),
    }


def _json_safe_cluster_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if not str(key).endswith("_set")}


def build_report() -> Path:
    page_start, page_end = _load_bag_page_range()
    crop_list = [dict(item) for item in _load_json(CROP_CACHE_PATH) if isinstance(item, dict)]
    detected_steps_by_page = _load_detected_steps()
    crops_by_page = _load_crop_cache_by_page()

    legacy_ocr_noise_audit = audit_bag_ocr_noise(BAG)
    legacy_ocr_noise_keys = {
        (int(row["page"]), int(row["ocr_value"])) for row in legacy_ocr_noise_audit.get("rows", [])
    }
    legacy_misread_audit = audit_bag_misread_steps(BAG, ocr_noise_keys=legacy_ocr_noise_keys)
    legacy_substep_audit = audit_bag_substep_numbers(BAG, ocr_noise_keys=legacy_ocr_noise_keys)
    legacy_misread_keys = {
        (int(row["page"]), int(row["ocr_value"])) for row in legacy_misread_audit.get("rows", [])
    }
    legacy_substep_keys = {
        (int(row["page"]), int(row["ocr_value"])) for row in legacy_substep_audit.get("rows", [])
    }
    legacy_suppressed_keys = legacy_misread_keys | legacy_substep_keys | legacy_ocr_noise_keys
    before_counts = _count_review_surface(
        detected_steps_by_page,
        crops_by_page,
        crop_list,
        page_start,
        page_end,
        legacy_suppressed_keys,
    )

    substep_audit = audit_bag_substep_numbers(BAG, ocr_noise_keys=set())
    substep_keys = {(int(row["page"]), int(row["ocr_value"])) for row in substep_audit.get("rows", [])}
    misread_audit = audit_bag_misread_steps(BAG, ocr_noise_keys=set())
    misread_keys = {(int(row["page"]), int(row["ocr_value"])) for row in misread_audit.get("rows", [])}

    cluster_audit = audit_bag3_step_cluster_gate(substep_keys=substep_keys, misread_keys=misread_keys)
    cluster_noise_keys = set(cluster_audit.get("ocr_noise_keys_set") or [])
    uncertain_keys = set(cluster_audit.get("uncertain_keys_set") or [])
    _write_json(CLUSTER_GATE_DIAGNOSTICS_PATH, _json_safe_cluster_audit(cluster_audit))

    ocr_noise_audit = {
        "bag": BAG,
        "page_start": page_start,
        "page_end": page_end,
        "classification": cluster_audit.get("classification"),
        "match_reason": cluster_audit.get("match_reason"),
        "row_count": cluster_audit.get("ocr_noise_count"),
        "ocr_noise_keys": cluster_audit.get("ocr_noise_keys"),
        "rows": [
            row
            for row in cluster_audit.get("rows", [])
            if row.get("classification") == cluster_audit.get("classification")
        ],
        "source": "bag3_step_cluster_gate",
        "legacy_tesseract_ocr_noise_count": legacy_ocr_noise_audit.get("row_count"),
    }
    ocr_noise_keys = cluster_noise_keys
    _write_json(OCR_NOISE_DIAGNOSTICS_PATH, ocr_noise_audit)

    _write_json(MISREAD_DIAGNOSTICS_PATH, misread_audit)
    _write_json(SUBSTEP_DIAGNOSTICS_PATH, substep_audit)
    suppressed_keys = misread_keys | substep_keys | ocr_noise_keys
    after_counts = _count_review_surface(
        detected_steps_by_page,
        crops_by_page,
        crop_list,
        page_start,
        page_end,
        suppressed_keys,
    )
    gate_comparison = {
        "before": before_counts,
        "after": after_counts,
        "delta": {
            key: after_counts[key] - before_counts[key]
            for key in before_counts
        },
        "legacy_tesseract_ocr_noise_count": legacy_ocr_noise_audit.get("row_count"),
        "cluster_ocr_noise_count": cluster_audit.get("ocr_noise_count"),
        "global_step_count": cluster_audit.get("global_step_count"),
        "uncertain_count": cluster_audit.get("uncertain_count"),
        "misread_count": misread_audit.get("row_count"),
        "substep_count": substep_audit.get("row_count"),
    }
    _write_json(OUT_DIR / "cluster_gate_comparison.json", gate_comparison)

    sequence_gaps = _merge_sequence_gap_reviews(
        _infer_sequence_gap_reviews(crop_list, page_start, page_end),
        _supplemental_sequence_gap_reviews(crop_list, page_start, page_end),
    )
    sequence_gaps = [
        gap
        for gap in sequence_gaps
        if (int(gap["page"]), int(gap["step"])) not in suppressed_keys
    ]
    sequence_gap_keys = {_confirmation_key(int(gap["page"]), int(gap["step"])) for gap in sequence_gaps}

    recovery_steps = _steps_needing_v1_recovery(
        detected_steps_by_page, crops_by_page, page_start, page_end, misread_keys, substep_keys, ocr_noise_keys
    )
    if recovery_steps:
        run_v1_callout_recovery(recovery_steps)
    v1_by_step = _load_v1_recovery_proposals()
    training_by_step = _load_training_labels_by_step()
    existing_labels = _load_existing_human_labels()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEQ_GAP_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pages: List[Dict[str, Any]] = []
    all_items: List[Dict[str, Any]] = []
    sequence_gap_items: List[Dict[str, Any]] = []

    for gap in sequence_gaps:
        gap = dict(gap)
        if not gap.get("ocr_evidence"):
            gap["ocr_evidence"] = _sequence_gap_ocr_evidence(
                int(gap["page"]),
                int(gap["step"]),
                detected_steps_by_page,
            )
        item = _build_sequence_gap_review_item(
            gap,
            existing_labels.get(_confirmation_key(int(gap["page"]), int(gap["step"]))),
            detected_steps_by_page,
        )
        item = _export_sequence_gap_review_assets(item)
        sequence_gap_items.append(item)

    for page in range(page_start, page_end + 1):
        if not _page_image_exists(page):
            continue
        detected_steps = list(detected_steps_by_page.get(page, []))
        crops = crops_by_page.get(page, [])

        page_items: List[Dict[str, Any]] = []
        step_lookup = {int(item["step"]): item for item in detected_steps}
        for step in [int(item["step"]) for item in detected_steps]:
            if (page, step) in suppressed_keys:
                continue
            if _confirmation_key(page, step) in sequence_gap_keys:
                continue
            if _step_has_crop(step, crops):
                continue

            step_entry = step_lookup.get(step)
            proposal = _proposal_from_v1_recovery(step, page, v1_by_step) if list((step_entry or {}).get("step_box") or []) else None
            human_label, label_record = _resolve_human_label(
                page,
                step,
                existing_labels.get(_confirmation_key(page, step)),
                step_entry,
                proposal,
            )
            crop_required = _step_requires_crop(step_entry, proposal, human_label)
            item = _build_confirmation_item(
                page,
                step,
                step_entry,
                existing_labels.get(_confirmation_key(page, step)),
                training_by_step.get((page, step)),
                proposal=proposal if crop_required else None,
                label_record=label_record,
                human_label=human_label,
                crop_required=crop_required,
            )
            if (page, step) in uncertain_keys:
                item["gate_status"] = UNCERTAIN_GATE_STATUS
                item["cluster_gate_uncertain"] = True
            page_items.append(item)
            all_items.append(item)

        pages.append(_export_page_assets(page, detected_steps, crops, page_items, []))

    crop_required_items = [item for item in all_items if item.get("crop_required")]
    candidate_exports = _export_crop_candidate_files(crop_required_items)
    missing_candidates = _build_missing_candidates_table(crop_required_items)
    audit_classifications = _build_audit_classifications_table(
        detected_steps_by_page,
        crops_by_page,
        v1_by_step,
        all_items + sequence_gap_items,
    )
    _write_json(OUT_DIR / "missing_candidates.json", missing_candidates)
    _write_json(OUT_DIR / "audit_step_classifications.json", audit_classifications)
    _write_json(SEQ_GAP_REVIEW_DIR / "index.json", {
        "name": "bag3_sequence_gap_review",
        "set_num": SET_NUM,
        "bag": BAG,
        "page_start": page_start,
        "page_end": page_end,
        "gate_status": SEQUENCE_GAP_REVIEW_STATUS,
        "human_label_options": list(SEQUENCE_GAP_LABEL_OPTIONS),
        "item_count": len(sequence_gap_items),
        "items": sequence_gap_items,
    })

    human_labels = {
        "name": "bag3_page_review_human_labels",
        "set_num": SET_NUM,
        "bag": BAG,
        "page_start": page_start,
        "page_end": page_end,
        "bag_map_source": str(BAG_MAP_PATH),
        "gate_status": GATE_STATUS,
        "sequence_gap_review_status": SEQUENCE_GAP_REVIEW_STATUS,
        "no_crop_required_status": NO_CROP_REQUIRED_STATUS,
        "human_label_options": list(HUMAN_LABEL_OPTIONS),
        "sequence_gap_label_options": list(SEQUENCE_GAP_LABEL_OPTIONS),
        "permanent_labels": {f"p{page:03d}_s{step}": rec for (page, step), rec in PERMANENT_LABELS.items()},
        "updated_at": _iso_now(),
        "missing_candidates_path": str(OUT_DIR / "missing_candidates.json"),
        "audit_classifications_path": str(OUT_DIR / "audit_step_classifications.json"),
        "sequence_gap_review_path": str(SEQ_GAP_REVIEW_DIR / "index.json"),
        "items": all_items,
        "sequence_gap_items": sequence_gap_items,
    }
    _write_json(HUMAN_LABELS_PATH, human_labels)

    manifest = {
        "name": "bag3_page_review",
        "set_num": SET_NUM,
        "bag": BAG,
        "page_start": page_start,
        "page_end": page_end,
        "bag_map_source": str(BAG_MAP_PATH),
        "completeness_gate": {
            "rule": "Crop required only when visible step anchor AND visible callout box exist. Sequence gaps always create SEQUENCE_GAP_REVIEW items with full page image and no fake crop boxes.",
            "gate_status": GATE_STATUS,
            "sequence_gap_review_status": SEQUENCE_GAP_REVIEW_STATUS,
            "no_crop_required_status": NO_CROP_REQUIRED_STATUS,
            "candidate_status": CREATE_CROP_CANDIDATE_STATUS,
            "v1_recovery_dir": str(V1_RECOVERY_DIR),
            "human_label_options": list(HUMAN_LABEL_OPTIONS),
            "sequence_gap_label_options": list(SEQUENCE_GAP_LABEL_OPTIONS),
            "human_labels_path": str(HUMAN_LABELS_PATH),
            "crop_candidates_dir": str(CROP_CANDIDATES_DIR),
            "missing_candidates_path": str(OUT_DIR / "missing_candidates.json"),
            "audit_classifications_path": str(OUT_DIR / "audit_step_classifications.json"),
        "sequence_gap_review_path": str(SEQ_GAP_REVIEW_DIR / "index.json"),
        "misread_step_diagnostics_path": str(MISREAD_DIAGNOSTICS_PATH),
        "substep_number_diagnostics_path": str(SUBSTEP_DIAGNOSTICS_PATH),
        "ocr_noise_diagnostics_path": str(OCR_NOISE_DIAGNOSTICS_PATH),
        "cluster_gate_diagnostics_path": str(CLUSTER_GATE_DIAGNOSTICS_PATH),
        "cluster_gate_comparison_path": str(OUT_DIR / "cluster_gate_comparison.json"),
        "misread_step_count": misread_audit.get("row_count"),
        "substep_number_count": substep_audit.get("row_count"),
        "ocr_noise_count": ocr_noise_audit.get("row_count"),
        "cluster_gate_comparison": gate_comparison,
        "does_not_write_crop_cache": True,
        },
        "crop_cache_source": str(CROP_CACHE_PATH),
        "training_labels_source": str(TRAINING_LABELS_PATH),
        "step_sources": [str(STEP_MAP_PATH), str(STEP_PROBE_PATH)],
        "output_dir": str(OUT_DIR),
        "page_count": len(pages),
        "total_review_items": len(all_items) + len(sequence_gap_items),
        "completeness_review_count": len(all_items),
        "sequence_gap_review_count": len(sequence_gap_items),
        "crop_required_count": len(crop_required_items),
        "no_crop_required_count": len(all_items) - len(crop_required_items),
        "crop_candidate_exports": candidate_exports,
        "missing_candidates": missing_candidates,
        "audit_step_classifications": audit_classifications,
        "sequence_gap_items": sequence_gap_items,
        "misread_step_diagnostics": misread_audit,
        "substep_number_diagnostics": substep_audit,
        "ocr_noise_diagnostics": ocr_noise_audit,
        "cluster_gate_diagnostics": _json_safe_cluster_audit(cluster_audit),
        "cluster_gate_comparison": gate_comparison,
        "v1_recovery_steps": recovery_steps,
        "pages": pages,
    }
    _write_json(OUT_DIR / "index.json", manifest)
    (OUT_DIR / "report.html").write_text(
        _build_html_report(pages, human_labels, sequence_gap_items, page_start, page_end),
        encoding="utf-8",
    )
    _print_missing_candidates_table(missing_candidates)
    print("")
    _print_audit_classifications_table(audit_classifications)
    print("")
    _print_sequence_gap_table(sequence_gap_items)
    return OUT_DIR / "report.html"


def main() -> int:
    report_path = build_report()
    manifest = _load_json(OUT_DIR / "index.json")
    print(str(report_path))
    print(
        f"pages={manifest['page_count']} "
        f"review_items={manifest['total_review_items']} "
        f"completeness={manifest['completeness_review_count']} "
        f"sequence_gaps={manifest['sequence_gap_review_count']} "
        f"crop_required={manifest['crop_required_count']} "
        f"candidates={len(manifest.get('crop_candidate_exports') or [])}"
    )
    print(f"human_labels={HUMAN_LABELS_PATH}")
    print(f"crop_candidates={CROP_CANDIDATES_DIR}")
    print(f"missing_candidates={OUT_DIR / 'missing_candidates.json'}")
    print(f"sequence_gap_review={SEQ_GAP_REVIEW_DIR / 'index.json'}")
    print(f"misread_diagnostics={MISREAD_DIAGNOSTICS_PATH}")
    print(f"substep_diagnostics={SUBSTEP_DIAGNOSTICS_PATH}")
    print(f"ocr_noise_diagnostics={OCR_NOISE_DIAGNOSTICS_PATH}")
    print(f"cluster_gate_diagnostics={CLUSTER_GATE_DIAGNOSTICS_PATH}")
    comparison = manifest.get("cluster_gate_comparison") or {}
    before = comparison.get("before") or {}
    after = comparison.get("after") or {}
    delta = comparison.get("delta") or {}
    print(
        "cluster_gate_before_after "
        f"total={before.get('total_review_count')}->{after.get('total_review_count')} "
        f"(delta {delta.get('total_review_count', 0)}) "
        f"completeness={before.get('completeness_review_count')}->{after.get('completeness_review_count')} "
        f"sequence_gaps={before.get('sequence_gap_review_count')}->{after.get('sequence_gap_review_count')} "
        f"cluster_ocr_noise={comparison.get('cluster_ocr_noise_count')} "
        f"global_steps={comparison.get('global_step_count')} "
        f"uncertain={comparison.get('uncertain_count')}"
    )
    print(f"page_range={manifest['page_start']}-{manifest['page_end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
