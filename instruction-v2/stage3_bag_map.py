import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from paths import INDEXES_DIR, ROOT_DIR


INPUT_PATH = INDEXES_DIR / "03_bag_candidates.json"
GAP_REVIEW_PATH = INDEXES_DIR / "03b_bag_gap_review.json"
V1_BAG_TRUTH_PATH = INDEXES_DIR / "03d_v1_bag_truth_import.json"
PAGE_INDEX_PATH = INDEXES_DIR / "02_page_index.json"
OUT_PATH = INDEXES_DIR / "04_bag_map.json"
DEBUG_DIR = ROOT_DIR / "debug" / "bag_map"


def _candidate_sort_key(candidate: Dict[str, Any]) -> tuple:
    return int(candidate.get("page") or 0), -float(candidate.get("score") or 0)


def _cluster_pages(candidates: List[Dict[str, Any]], max_gap: int = 6) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=_candidate_sort_key):
        page = int(candidate["page"])
        if not clusters or page - int(clusters[-1]["end_page"]) > max_gap:
            clusters.append(
                {
                    "start_page": page,
                    "end_page": page,
                    "candidates": [candidate],
                }
            )
            continue
        clusters[-1]["end_page"] = page
        clusters[-1]["candidates"].append(candidate)
    return clusters


def _cluster_score(cluster: Dict[str, Any]) -> float:
    scores = [float(item.get("score") or 0) for item in cluster["candidates"]]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def _is_front_matter_cluster(cluster: Dict[str, Any]) -> bool:
    return int(cluster["start_page"]) <= 5


def _page_image_paths() -> Dict[int, str]:
    if not PAGE_INDEX_PATH.exists():
        return {}

    payload = json.loads(PAGE_INDEX_PATH.read_text(encoding="utf-8"))
    paths: Dict[int, str] = {}
    for entry in payload.get("pages", []) or []:
        page = entry.get("page")
        image_path = entry.get("image_path")
        if page is None or not image_path:
            continue
        paths[int(page)] = str(image_path)
    return paths


def _accepted_gap_corrections() -> List[Dict[str, Any]]:
    if not GAP_REVIEW_PATH.exists():
        return []

    gap_review = json.loads(GAP_REVIEW_PATH.read_text(encoding="utf-8"))
    corrections = []
    for gap in gap_review.get("gap_windows", []) or []:
        if gap.get("status") != "accepted":
            continue
        accepted_page = gap.get("accepted_page")
        observed_bag_number = gap.get("observed_bag_number")
        if accepted_page is None or observed_bag_number is None:
            continue
        corrections.append(
            {
                "page": int(accepted_page),
                "observed_bag_number": int(observed_bag_number),
                "window_id": gap.get("window_id"),
                "review_source": gap.get("review_source") or "human",
                "notes": gap.get("notes") or [],
                "confidence": float(gap.get("confidence") or 1.0),
            }
        )
    return corrections


def _accepted_bag_1_bootstrap() -> Optional[Dict[str, Any]]:
    if not GAP_REVIEW_PATH.exists():
        return None

    gap_review = json.loads(GAP_REVIEW_PATH.read_text(encoding="utf-8"))
    bootstrap = gap_review.get("bag_1_bootstrap_review") or {}
    if bootstrap.get("status") != "accepted":
        return None

    proposed_page = bootstrap.get("proposed_bag_1_start_page")
    earliest_step_1_page = bootstrap.get("earliest_step_1_page")
    if proposed_page is None or earliest_step_1_page is None:
        return None

    return {
        "page": int(proposed_page),
        "earliest_step_1_page": int(earliest_step_1_page),
        "rule": bootstrap.get("rule") or "earliest_step_1_previous_page",
        "review_source": bootstrap.get("review_source") or "instruction_v2_bootstrap_rule",
        "confidence": float(bootstrap.get("confidence") or 0.0),
        "evidence_reason": bootstrap.get("evidence_reason"),
        "step_1_detection_reason": bootstrap.get("step_1_detection_reason"),
        "previous_page_evidence": bootstrap.get("previous_page_evidence"),
        "requires_human_review": bool(bootstrap.get("requires_human_review", True)),
    }


def _v1_bag_truth_entries() -> List[Dict[str, Any]]:
    if not V1_BAG_TRUTH_PATH.exists():
        return []

    payload = json.loads(V1_BAG_TRUTH_PATH.read_text(encoding="utf-8"))
    entries: List[Dict[str, Any]] = []
    for item in payload.get("bags", []) or []:
        bag_number = item.get("bag_number", item.get("bag"))
        start_page = item.get("start_page")
        if bag_number is None or start_page is None:
            continue
        entries.append(
            {
                "bag": int(bag_number),
                "start_page": int(start_page),
                "source": "v1_bag_truth_db",
                "confidence": float(item.get("confidence") or 0.0),
                "v1_source": item.get("v1_source"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
            }
        )
    return sorted(entries, key=lambda item: (int(item["bag"]), int(item["start_page"])))


def _build_anchors(clusters: List[Dict[str, Any]], corrections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    anchors = []
    provisional_index = 0
    for cluster in clusters:
        if _is_front_matter_cluster(cluster):
            continue
        provisional_index += 1
        anchors.append(
            {
                "start_page": int(cluster["start_page"]),
                "source": "candidate_cluster",
                "cluster": cluster,
                "provisional_bag_index": provisional_index,
            }
        )
    for correction in corrections:
        anchors.append(
            {
                "start_page": int(correction["page"]),
                "source": "human_gap_review",
                "correction": correction,
                "observed_bag_number": int(correction["observed_bag_number"]),
            }
        )
    return sorted(anchors, key=lambda item: (int(item["start_page"]), 0 if item["source"] == "human_gap_review" else 1))


def _build_priority_anchors(
    clusters: List[Dict[str, Any]],
    corrections: List[Dict[str, Any]],
    v1_truth_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    used_bags = set()
    used_pages = set()

    for entry in v1_truth_entries:
        bag = int(entry["bag"])
        page = int(entry["start_page"])
        anchors.append(
            {
                "bag": bag,
                "start_page": page,
                "source": "v1_bag_truth_db",
                "bag_number_source": "v1_bag_truth_db",
                "v1_truth": entry,
            }
        )
        used_bags.add(bag)
        used_pages.add(page)

    for correction in corrections:
        bag = int(correction["observed_bag_number"])
        page = int(correction["page"])
        if bag in used_bags or page in used_pages:
            continue
        anchors.append(
            {
                "bag": bag,
                "start_page": page,
                "source": "human_gap_review",
                "bag_number_source": "observed_human_gap_review",
                "correction": correction,
                "observed_bag_number": bag,
            }
        )
        used_bags.add(bag)
        used_pages.add(page)

    if not anchors:
        return _build_anchors(clusters, corrections)

    if v1_truth_entries:
        return sorted(anchors, key=lambda item: (int(item["start_page"]), int(item.get("bag") or 0)))

    next_bag = max(used_bags) + 1 if used_bags else 1
    for cluster in clusters:
        if _is_front_matter_cluster(cluster):
            continue
        page = int(cluster["start_page"])
        if page in used_pages:
            continue
        anchors.append(
            {
                "bag": next_bag,
                "start_page": page,
                "source": "candidate_cluster",
                "bag_number_source": "inferred_sequence_after_priority_sources",
                "cluster": cluster,
                "provisional_bag_index": next_bag,
            }
        )
        used_pages.add(page)
        next_bag += 1

    return sorted(anchors, key=lambda item: (int(item["start_page"]), int(item.get("bag") or 0)))


def _assign_bag_numbers(anchors: List[Dict[str, Any]]) -> None:
    next_bag = 1
    for anchor in anchors:
        observed = anchor.get("observed_bag_number")
        if observed is not None:
            anchor["bag"] = int(observed)
            anchor["bag_number_source"] = "observed_human_gap_review"
            next_bag = int(observed) + 1
            continue
        anchor["bag"] = next_bag
        anchor["bag_number_source"] = "inferred_sequence"
        next_bag += 1


def _build_ranges(
    clusters: List[Dict[str, Any]],
    page_count: int,
    corrections: List[Dict[str, Any]],
    v1_truth_entries: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    v1_truth_entries = list(v1_truth_entries or [])
    page_image_paths = _page_image_paths()
    anchors = _build_priority_anchors(clusters, corrections, v1_truth_entries)
    if not v1_truth_entries:
        _assign_bag_numbers(anchors)
    ranges: List[Dict[str, Any]] = []

    for idx, anchor in enumerate(anchors):
        next_anchor = anchors[idx + 1] if idx + 1 < len(anchors) else None
        start_page = int(anchor["start_page"])
        end_page = int(next_anchor["start_page"]) - 1 if next_anchor else int(page_count)

        if anchor["source"] == "v1_bag_truth_db":
            truth = anchor["v1_truth"]
            confidence = float(truth.get("confidence") or 0.0)
            evidence_pages = [int(truth["start_page"])]
            evidence = [
                {
                    "page": int(truth["start_page"]),
                    "score": confidence,
                    "source": "v1_bag_truth_db",
                    "debug_image_path": page_image_paths.get(int(truth["start_page"])),
                    "v1_source": truth.get("v1_source"),
                    "created_at": truth.get("created_at"),
                    "updated_at": truth.get("updated_at"),
                }
            ]
        elif anchor["source"] == "human_gap_review":
            correction = anchor["correction"]
            evidence = [
                {
                    "page": int(correction["page"]),
                    "score": float(correction["confidence"]),
                    "source": "human_gap_review",
                    "window_id": correction.get("window_id"),
                    "review_source": correction.get("review_source") or "human",
                    "observed_bag_number": int(correction["observed_bag_number"]),
                    "notes": correction.get("notes") or [],
                }
            ]
            confidence = float(correction["confidence"])
            evidence_pages = [int(correction["page"])]
        else:
            cluster = anchor["cluster"]
            confidence = _cluster_score(cluster)
            if len(cluster["candidates"]) >= 2:
                confidence = min(1.0, round(confidence + 0.05, 4))
            evidence_pages = [int(item["page"]) for item in cluster["candidates"]]
            evidence = [
                {
                    "page": int(item["page"]),
                    "score": float(item.get("score") or 0),
                    "source": "bag_candidate_cluster",
                    "debug_image_path": item.get("debug_image_path"),
                    "signals": item.get("signals") or {},
                }
                for item in cluster["candidates"]
            ]

        ranges.append(
            {
                "bag": int(anchor["bag"]),
                "start_page": start_page,
                "end_page": end_page,
                "confidence": round(confidence, 4),
                "source": anchor["source"],
                "bag_number_source": anchor["bag_number_source"],
                "observed_bag_number": anchor.get("observed_bag_number"),
                "v1_source": (anchor.get("v1_truth") or {}).get("v1_source"),
                "evidence_pages": evidence_pages,
                "evidence": evidence,
            }
        )

    return ranges


def _apply_bag_1_bootstrap(bag_ranges: List[Dict[str, Any]], bootstrap: Optional[Dict[str, Any]]) -> None:
    if not bootstrap:
        return

    proposed_start = int(bootstrap["page"])
    for bag_range in bag_ranges:
        if int(bag_range.get("bag") or 0) != 1:
            continue
        current_start = int(bag_range.get("start_page") or 0)
        if proposed_start >= current_start:
            return

        evidence = {
            "page": proposed_start,
            "score": float(bootstrap.get("confidence") or 0.0),
            "source": "bag_1_bootstrap_rule",
            "debug_image_path": (bootstrap.get("previous_page_evidence") or {}).get("image_path"),
            "rule": bootstrap.get("rule") or "earliest_step_1_previous_page",
            "review_source": bootstrap.get("review_source") or "instruction_v2_bootstrap_rule",
            "earliest_step_1_page": int(bootstrap.get("earliest_step_1_page") or 0),
            "evidence_reason": bootstrap.get("evidence_reason"),
            "step_1_detection_reason": bootstrap.get("step_1_detection_reason"),
            "previous_page_evidence": bootstrap.get("previous_page_evidence"),
            "requires_human_review": bool(bootstrap.get("requires_human_review", True)),
        }

        bag_range["previous_start_page"] = current_start
        bag_range["start_page"] = proposed_start
        bag_range["source"] = "bag_1_bootstrap_rule"
        bag_range["bag_number_source"] = "earliest_step_1_previous_page"
        bag_range["confidence"] = round(max(float(bag_range.get("confidence") or 0.0), float(bootstrap.get("confidence") or 0.0)), 4)
        bag_range["evidence_pages"] = sorted({proposed_start, *[int(page) for page in bag_range.get("evidence_pages", [])]})
        bag_range.setdefault("evidence", []).insert(0, evidence)
        bag_range["bootstrap_review"] = evidence
        return


def build_bag_map() -> Dict[str, Any]:
    if not INPUT_PATH.exists():
        raise RuntimeError(f"Missing bag candidate manifest: {INPUT_PATH}")

    candidate_manifest = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    candidates = list(candidate_manifest.get("candidates") or [])
    clusters = _cluster_pages(candidates)
    accepted_corrections = _accepted_gap_corrections()
    accepted_bootstrap = _accepted_bag_1_bootstrap()
    v1_truth_entries = _v1_bag_truth_entries()
    bag_ranges = _build_ranges(clusters, int(candidate_manifest.get("page_count") or 0), accepted_corrections, v1_truth_entries)
    if not v1_truth_entries:
        _apply_bag_1_bootstrap(bag_ranges, accepted_bootstrap)

    front_matter = [
        {
            "start_page": int(cluster["start_page"]),
            "end_page": int(cluster["end_page"]),
            "evidence_pages": [int(item["page"]) for item in cluster["candidates"]],
            "reason": "candidate cluster before first build range; retained as overview/front-matter evidence",
        }
        for cluster in clusters
        if _is_front_matter_cluster(cluster)
    ]

    payload = {
        "stage": 3,
        "name": "bag_page_ranges",
        "input_manifests": [
            "indexes/03d_v1_bag_truth_import.json",
            "indexes/03_bag_candidates.json",
            "indexes/03b_bag_gap_review.json",
        ],
        "method": "priority_v1_bag_truth_then_human_gap_review_then_candidates",
        "page_count": int(candidate_manifest.get("page_count") or 0),
        "candidate_count": int(candidate_manifest.get("candidate_count") or len(candidates)),
        "v1_bag_truth_count": len(v1_truth_entries),
        "v1_bag_truth_source": str(V1_BAG_TRUTH_PATH.relative_to(ROOT_DIR)) if V1_BAG_TRUTH_PATH.exists() else None,
        "accepted_gap_correction_count": len(accepted_corrections),
        "accepted_gap_corrections": accepted_corrections,
        "accepted_bag_1_bootstrap": accepted_bootstrap,
        "bag_count": len(bag_ranges),
        "front_matter_candidate_clusters": front_matter,
        "bags": bag_ranges,
        "notes": [
            "Source priority: 03d V1 bag truth import, then accepted 03b human gap-review corrections, then 03 bag candidates.",
            "It does not read debug/, training labels, clean routes, OCR, or step detection.",
            "The V1 truth import manifest is read-only evidence produced from debug/bag_truth.db by Stage 3d.",
            "Accepted human gap-review pages preserve review_source and observed_bag_number evidence.",
            "Accepted Bag 1 bootstrap evidence is only applied when no V1 bag truth import is available.",
        ],
    }

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    payload = build_bag_map()
    print(
        json.dumps(
            {
                "ok": True,
                "bag_count": payload["bag_count"],
                "out": str(OUT_PATH.relative_to(ROOT_DIR)),
                "debug_dir": str(DEBUG_DIR.relative_to(ROOT_DIR)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
