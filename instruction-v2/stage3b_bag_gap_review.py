import json
import subprocess
import tempfile
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

from paths import INDEXES_DIR, ROOT_DIR


INPUT_PATH = INDEXES_DIR / "03_bag_candidates.json"
PAGE_INDEX_PATH = INDEXES_DIR / "02_page_index.json"
STEP_MAP_PATH = INDEXES_DIR / "05_step_map.json"
OUT_PATH = INDEXES_DIR / "03b_bag_gap_review.json"
NODE_BIN = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
NODE_MODULES = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")
CLUSTER_MAX_GAP = 6
FRONT_MATTER_MAX_PAGE = 5
MIN_GAP_PAGE_SPAN = 20
LARGE_GAP_MULTIPLIER = 1.5
PRESERVED_REVIEW_FIELDS = (
    "status",
    "accepted_page",
    "observed_bag_number",
    "review_source",
    "notes",
)


def _candidate_page(candidate: Dict[str, Any]) -> int:
    return int(candidate.get("page") or 0)


def _candidate_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("score") or 0.0)


def _cluster_pages(candidates: List[Dict[str, Any]], max_gap: int = CLUSTER_MAX_GAP) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (_candidate_page(item), -_candidate_score(item))):
        page = _candidate_page(candidate)
        if not clusters or page - int(clusters[-1]["end_page"]) > max_gap:
            clusters.append({"start_page": page, "end_page": page, "candidates": [candidate]})
            continue
        clusters[-1]["end_page"] = page
        clusters[-1]["candidates"].append(candidate)
    return clusters


def _cluster_confidence(cluster: Dict[str, Any]) -> float:
    scores = [_candidate_score(candidate) for candidate in cluster.get("candidates", [])]
    if not scores:
        return 0.0
    confidence = sum(scores) / len(scores)
    if len(scores) >= 2:
        confidence = min(1.0, confidence + 0.05)
    return round(confidence, 4)


def _cluster_evidence(cluster: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = []
    for candidate in cluster.get("candidates", []):
        evidence.append(
            {
                "page": _candidate_page(candidate),
                "score": _candidate_score(candidate),
                "debug_image_path": candidate.get("debug_image_path"),
                "signals": candidate.get("signals") or {},
            }
        )
    return evidence


def _known_sequence(build_clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sequence = []
    for index, cluster in enumerate(build_clusters, start=1):
        sequence.append(
            {
                "bag": index,
                "start_page": int(cluster["start_page"]),
                "end_page": int(cluster["end_page"]),
                "confidence": _cluster_confidence(cluster),
                "evidence_pages": [_candidate_page(candidate) for candidate in cluster.get("candidates", [])],
            }
        )
    return sequence


def _gap_threshold(build_clusters: List[Dict[str, Any]]) -> int:
    gaps = [
        int(next_cluster["start_page"]) - int(cluster["start_page"])
        for cluster, next_cluster in zip(build_clusters, build_clusters[1:])
    ]
    if not gaps:
        return MIN_GAP_PAGE_SPAN
    return max(MIN_GAP_PAGE_SPAN, int(round(median(gaps) * LARGE_GAP_MULTIPLIER)))


def _candidate_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "page": _candidate_page(candidate),
        "score": _candidate_score(candidate),
        "debug_image_path": candidate.get("debug_image_path"),
        "signals": candidate.get("signals") or {},
        "detected_bag_number": candidate.get("detected_bag_number"),
        "detected_bag_number_confidence": candidate.get("detected_bag_number_confidence"),
        "detected_bag_number_source": candidate.get("detected_bag_number_source"),
        "detected_bag_number_box": candidate.get("detected_bag_number_box"),
        "detected_bag_number_ocr_raw": candidate.get("detected_bag_number_ocr_raw"),
    }


def _existing_gap_reviews() -> Dict[str, Dict[str, Any]]:
    if not OUT_PATH.exists():
        return {}
    try:
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        str(gap.get("window_id")): gap
        for gap in existing.get("gap_windows", []) or []
        if gap.get("window_id")
    }


def _merge_existing_review(window: Dict[str, Any], existing_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    existing = existing_by_id.get(str(window.get("window_id")))
    if not existing:
        return window
    merged = dict(window)
    if existing.get("status") in {"accepted", "rejected"}:
        for field in PRESERVED_REVIEW_FIELDS:
            if field in existing:
                merged[field] = existing[field]
    return merged


def _scan_gap_windows(gap_windows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    if not gap_windows:
        return {}
    if not PAGE_INDEX_PATH.exists():
        raise RuntimeError(f"Missing page index manifest: {PAGE_INDEX_PATH}")
    if not NODE_BIN.exists() or not NODE_MODULES.exists():
        raise RuntimeError("Missing bundled Node runtime/dependencies for gap window scan")

    with tempfile.TemporaryDirectory(prefix="a2p_v2_gap_scan_") as tmp:
        windows_path = Path(tmp) / "gap_windows.json"
        out_path = Path(tmp) / "gap_window_scan.json"
        windows_path.write_text(json.dumps({"gap_windows": gap_windows}, indent=2) + "\n", encoding="utf-8")
        subprocess.run(
            [
                str(NODE_BIN),
                str(ROOT_DIR / "gap_window_scan.mjs"),
                "--page-index",
                str(PAGE_INDEX_PATH),
                "--windows",
                str(windows_path),
                "--repo-root",
                str(ROOT_DIR),
                "--out",
                str(out_path),
                "--node-modules",
                str(NODE_MODULES),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    return payload.get("scan_by_window") or {}


def _scan_step_window(start_page: int, end_page: int) -> Dict[str, Any]:
    if not PAGE_INDEX_PATH.exists():
        raise RuntimeError(f"Missing page index manifest: {PAGE_INDEX_PATH}")
    if not NODE_BIN.exists() or not NODE_MODULES.exists():
        raise RuntimeError("Missing bundled Node runtime/dependencies for bootstrap step scan")

    with tempfile.TemporaryDirectory(prefix="a2p_v2_bag1_bootstrap_steps_") as tmp:
        tmp_path = Path(tmp)
        bag_map_path = tmp_path / "bootstrap_bag_map.json"
        out_path = tmp_path / "bootstrap_step_map.json"
        debug_dir = tmp_path / "step_map"
        bag_map_path.write_text(
            json.dumps(
                {
                    "bags": [
                        {
                            "bag": 1,
                            "start_page": int(start_page),
                            "end_page": int(end_page),
                        }
                    ]
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(NODE_BIN),
                str(ROOT_DIR / "step_map_scan.mjs"),
                "--page-index",
                str(PAGE_INDEX_PATH),
                "--bag-map",
                str(bag_map_path),
                "--repo-root",
                str(ROOT_DIR),
                "--out",
                str(out_path),
                "--debug-dir",
                str(debug_dir),
                "--node-modules",
                str(NODE_MODULES),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(out_path.read_text(encoding="utf-8"))


def _load_existing_step_map() -> Dict[str, Any]:
    if not STEP_MAP_PATH.exists():
        return {}
    try:
        return json.loads(STEP_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _earliest_step_1_page_from_steps(steps: List[Dict[str, Any]], min_page: int = 1) -> Optional[int]:
    pages = [
        int(step.get("page") or 0)
        for step in steps
        if int(step.get("page") or 0) >= min_page and step.get("step_number") == 1
    ]
    return min(pages) if pages else None


def _infer_early_step_1_page(scanned_steps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    build_steps = [
        step
        for step in scanned_steps
        if int(step.get("page") or 0) > FRONT_MATTER_MAX_PAGE
    ]
    detected_step_1_page = _earliest_step_1_page_from_steps(build_steps, min_page=FRONT_MATTER_MAX_PAGE + 1)
    if detected_step_1_page is not None:
        return {
            "page": detected_step_1_page,
            "source": "direct_bootstrap_step_scan",
            "reason": "Bootstrap step scan found printed step_number 1 after front matter.",
        }

    numbered_steps = [
        step
        for step in build_steps
        if isinstance(step.get("step_number"), int) and int(step.get("step_number")) > 1
    ]
    numbered_steps.sort(key=lambda item: (int(item.get("page") or 0), int(item.get("step_number") or 0)))
    first_numbered = numbered_steps[0] if numbered_steps else None
    if not first_numbered:
        return None

    first_numbered_page = int(first_numbered.get("page") or 0)
    candidate_pages = sorted(
        {
            int(step.get("page") or 0)
            for step in build_steps
            if int(step.get("page") or 0) < first_numbered_page
        }
    )
    if not candidate_pages:
        return None

    inferred_page = candidate_pages[-1]
    return {
        "page": inferred_page,
        "source": "sequence_gap_inference",
        "reason": (
            f"Bootstrap scan did not OCR step 1, but page {inferred_page} has step-box evidence "
            f"immediately before page {first_numbered_page}, where printed step {first_numbered.get('step_number')} was detected."
        ),
    }


def _build_bag_1_bootstrap_review(build_clusters: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not build_clusters:
        return None

    current_bag_1_start = int(build_clusters[0]["start_page"])
    if current_bag_1_start <= FRONT_MATTER_MAX_PAGE + 1:
        return None

    existing_step_map = _load_existing_step_map()
    existing_steps = list(existing_step_map.get("steps") or [])
    manifest_step_1_page = _earliest_step_1_page_from_steps(existing_steps)
    bootstrap_scan = _scan_step_window(1, current_bag_1_start - 1)
    bootstrap_steps = list(bootstrap_scan.get("steps") or [])
    inferred_step_1 = _infer_early_step_1_page(bootstrap_steps)
    if not inferred_step_1:
        return {
            "rule": "earliest_step_1_previous_page",
            "current_bag_1_start_page": current_bag_1_start,
            "earliest_step_1_page": manifest_step_1_page,
            "proposed_bag_1_start_page": None,
            "status": "pending",
            "evidence_reason": "No reliable early step 1 page could be found before the current Bag 1 start page.",
            "review_source": "instruction_v2_bootstrap_rule",
        }

    earliest_step_1_page = int(inferred_step_1["page"])
    proposed_start = earliest_step_1_page - 1 if earliest_step_1_page > 1 else None
    evidence_scan = {}
    page_evidence = None
    if proposed_start is not None:
        bootstrap_windows = [
            {
                "window_id": "bag_1_bootstrap_previous_page",
                "window": {"start_page": proposed_start, "end_page": proposed_start},
            }
        ]
        evidence_scan = _scan_gap_windows(bootstrap_windows)
        candidates = evidence_scan.get("bag_1_bootstrap_previous_page") or []
        page_evidence = candidates[0] if candidates else None

    strong_evidence = bool(
        page_evidence
        and float(page_evidence.get("score") or 0) >= 50
        and (page_evidence.get("signals") or {}).get("orange_arrow_component_count", 0) >= 1
        and (page_evidence.get("signals") or {}).get("large_number_group_count", 0) >= 1
    )

    status = "accepted" if strong_evidence else "pending"
    evidence_reason = (
        f"Page {earliest_step_1_page} is the earliest build-page step-1 candidate by {inferred_step_1['source']}; "
        f"previous page {proposed_start} has material/start-page evidence"
        if strong_evidence
        else f"Page {earliest_step_1_page} is a step-1 candidate, but previous page material/start evidence is not strong enough to auto-accept."
    )

    return {
        "rule": "earliest_step_1_previous_page",
        "current_bag_1_start_page": current_bag_1_start,
        "manifest_earliest_step_1_page": manifest_step_1_page,
        "earliest_step_1_page": earliest_step_1_page,
        "earliest_step_1_source": inferred_step_1["source"],
        "proposed_bag_1_start_page": proposed_start,
        "status": status,
        "review_source": "instruction_v2_bootstrap_rule",
        "requires_human_review": True,
        "confidence": float(page_evidence.get("confidence") or 0.0) if page_evidence else 0.0,
        "evidence_reason": evidence_reason,
        "step_1_detection_reason": inferred_step_1["reason"],
        "previous_page_evidence": page_evidence,
    }


def _build_gap_windows(build_clusters: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    threshold = _gap_threshold(build_clusters)
    windows = []

    for window_index, (left, right) in enumerate(zip(build_clusters, build_clusters[1:]), start=1):
        left_bag = window_index
        right_bag = window_index + 1
        gap_start = int(left["end_page"]) + 1
        gap_end = int(right["start_page"]) - 1
        page_span = gap_end - gap_start + 1
        start_delta = int(right["start_page"]) - int(left["start_page"])

        if page_span <= 0 or start_delta < threshold:
            continue

        existing_stage2_candidates = [
            _candidate_summary(candidate)
            for candidate in candidates
            if gap_start <= _candidate_page(candidate) <= gap_end
        ]
        confidence = min(0.99, round(start_delta / max(1, threshold), 4))
        windows.append(
            {
                "window_id": f"gap_after_bag_{left_bag:02d}_before_bag_{right_bag:02d}",
                "between_known_bags": [left_bag, right_bag],
                "previous_known_start_page": int(left["start_page"]),
                "next_known_start_page": int(right["start_page"]),
                "window": {"start_page": gap_start, "end_page": gap_end, "page_span": page_span},
                "candidate_pages": existing_stage2_candidates,
                "stage2_candidate_pages_inside_window": existing_stage2_candidates,
                "confidence": confidence,
                "status": "pending",
                "notes": [
                    f"Start-page delta {start_delta} exceeds gap threshold {threshold}.",
                    "Human review should decide whether a missing bag start exists inside this window before regenerating 04_bag_map.json.",
                ],
            }
        )

    return windows


def build_bag_gap_review() -> Dict[str, Any]:
    if not INPUT_PATH.exists():
        raise RuntimeError(f"Missing bag candidates manifest: {INPUT_PATH}")

    existing_by_id = _existing_gap_reviews()
    candidate_manifest = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    candidates = list(candidate_manifest.get("candidates") or [])
    clusters = _cluster_pages(candidates)
    build_clusters = [cluster for cluster in clusters if int(cluster["start_page"]) > FRONT_MATTER_MAX_PAGE]
    front_matter_clusters = [cluster for cluster in clusters if int(cluster["start_page"]) <= FRONT_MATTER_MAX_PAGE]
    gap_windows = [
        _merge_existing_review(window, existing_by_id)
        for window in _build_gap_windows(build_clusters, candidates)
    ]
    scan_by_window = _scan_gap_windows(gap_windows)
    for window in gap_windows:
        scored_candidates = scan_by_window.get(str(window.get("window_id")), [])
        window["candidate_pages"] = scored_candidates
        window["candidate_page_count"] = len(scored_candidates)
        window["top_candidate_page"] = scored_candidates[0] if scored_candidates else None
        window["candidate_scan_source"] = "instruction_v2_gap_window_scan_v1"
    bag_1_bootstrap_review = _build_bag_1_bootstrap_review(build_clusters)

    payload = {
        "stage": "3b",
        "name": "bag_gap_review",
        "input_manifests": [
            "indexes/03_bag_candidates.json",
            "indexes/02_page_index.json",
        ],
        "method": "candidate_cluster_gap_review_with_window_scoring_v2",
        "expected_bag_sequence": {
            "available": False,
            "source": None,
            "bags": [],
            "notes": "No explicit expected bag sequence is available from 03_bag_candidates.json.",
        },
        "known_bag_sequence": {
            "available": bool(build_clusters),
            "source": "inferred_from_03_bag_candidates_candidate_clusters",
            "bags": _known_sequence(build_clusters),
        },
        "detected_bag_candidate_pages": [_candidate_summary(candidate) for candidate in sorted(candidates, key=_candidate_page)],
        "front_matter_candidate_clusters": [
            {
                "start_page": int(cluster["start_page"]),
                "end_page": int(cluster["end_page"]),
                "evidence_pages": [_candidate_page(candidate) for candidate in cluster.get("candidates", [])],
                "confidence": _cluster_confidence(cluster),
            }
            for cluster in front_matter_clusters
        ],
        "bag_1_bootstrap_review": bag_1_bootstrap_review,
        "gap_threshold_pages": _gap_threshold(build_clusters),
        "gap_window_count": len(gap_windows),
        "gap_windows": gap_windows,
        "status": "pending",
        "notes": [
            "This manifest is a human review checkpoint before 04_bag_map.json.",
            "It does not modify or regenerate the final bag map.",
            "Statuses are pending/accepted/rejected and should be resolved by human review.",
        ],
    }

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    payload = build_bag_gap_review()
    print(
        json.dumps(
            {
                "ok": True,
                "gap_window_count": payload["gap_window_count"],
                "out": str(OUT_PATH.relative_to(INDEXES_DIR.parent)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
