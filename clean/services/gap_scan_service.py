import json
from functools import cmp_to_key
from typing import Optional

from clean.services import debug_service, precheck_service, sequence_service, truth_service

try:
    from clean.services import analyzer_scan_service
except Exception:
    analyzer_scan_service = None


_GREEN_STEP_BOX_CACHE = {}


def _get_green_step_box_count(set_num: str, page: int) -> int:
    cache = _GREEN_STEP_BOX_CACHE.get(str(set_num))
    if cache is None:
        cache = {}
        path = debug_service.DEBUG_ROOT / str(set_num) / "green_step_boxes_7_38.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text())
                cache = {
                    int(item.get("page", 0) or 0): int(item.get("green_box_count", 0) or 0)
                    for item in (payload.get("pages", []) or [])
                }
            except Exception:
                cache = {}
        _GREEN_STEP_BOX_CACHE[str(set_num)] = cache
    return int(cache.get(int(page), 0) or 0)


def _compare_candidates(a: dict, b: dict) -> int:
    score_a = float(a.get("score", 0.0) or 0.0)
    score_b = float(b.get("score", 0.0) or 0.0)
    if abs(score_a - score_b) < 5.0:
        return -1 if int(a.get("page", 0) or 0) < int(b.get("page", 0) or 0) else 1
    return -1 if score_a > score_b else 1 if score_a < score_b else 0


def _bounded_window_from_missing_window(window: dict):
    if not window:
        return None

    prev_page = window.get("previous_confirmed_page")
    next_page = window.get("next_confirmed_page")

    if prev_page is None or next_page is None:
        return None

    start_page = int(prev_page) + 1
    end_page = int(next_page) - 1

    if end_page < start_page:
        return None

    return start_page, end_page


def _normalize_analysis_row(row: dict, page: int, precheck: dict) -> dict:
    normalized = dict(row or {})
    normalized["page"] = int(normalized.get("page", page) or page)
    normalized["panel_found"] = bool(normalized.get("panel_found"))
    normalized["shell_found"] = bool(normalized.get("shell_found"))
    normalized["grey_bag_found"] = bool(normalized.get("grey_bag_found"))
    normalized["number_box_found"] = bool(
        normalized.get("number_box_found", normalized.get("number_found"))
    )
    normalized["bag_number"] = normalized.get("bag_number")
    normalized["confidence"] = float(normalized.get("confidence", 0.0) or 0.0)
    normalized["overview_page"] = bool(
        normalized.get("overview_page")
        or normalized.get("panel_source") == "overview_page"
    )
    normalized["strong_structure"] = bool(
        normalized.get("strong_structure")
        or (
            normalized["panel_found"]
            and normalized["shell_found"]
            and normalized["grey_bag_found"]
        )
    )
    normalized["cache_hit"] = bool(normalized.get("cache_hit"))
    normalized["precheck"] = dict(precheck or {})
    normalized["page_kind"] = normalized["precheck"].get("page_kind", "other")
    return normalized


def _get_cached_window_row(set_num: str, page: int) -> Optional[dict]:
    if analyzer_scan_service is None:
        raise RuntimeError("clean.services.analyzer_scan_service is not importable")

    if not hasattr(analyzer_scan_service, "get_cached_sequence_scan_row"):
        raise RuntimeError(
            "analyzer_scan_service does not expose get_cached_sequence_scan_row"
        )

    precheck = precheck_service.get_page_precheck(set_num, int(page))
    row = analyzer_scan_service.get_cached_sequence_scan_row(set_num, int(page))
    if row is None:
        return None
    return _normalize_analysis_row(row, int(page), precheck)


def _get_or_build_window_row(set_num: str, page: int) -> dict:
    if analyzer_scan_service is None:
        raise RuntimeError("clean.services.analyzer_scan_service is not importable")

    if not hasattr(analyzer_scan_service, "get_or_build_sequence_scan_row"):
        raise RuntimeError(
            "analyzer_scan_service does not expose get_or_build_sequence_scan_row"
        )

    precheck = precheck_service.get_page_precheck(set_num, int(page))
    row = analyzer_scan_service.get_or_build_sequence_scan_row(set_num, int(page))
    return _normalize_analysis_row(row, int(page), precheck)


def _score_window_candidate(
    set_num: str,
    row: dict,
    bag_number: int,
    start_page: int,
    end_page: int,
):
    score = 0.0
    reasons = []

    page = int(row["page"])
    confidence = float(row.get("confidence", 0.0) or 0.0)
    detected_number = row.get("bag_number")
    page_kind = row.get("page_kind", "other")
    green_box_count = _get_green_step_box_count(set_num, page)
    no_bag_structure_number_only = bool(
        row.get("number_box_found")
        and not row.get("strong_structure")
        and not row.get("shell_found")
        and not row.get("grey_bag_found")
    )

    if row.get("number_box_found"):
        score += 40.0
        reasons.append("+ number_box_found")
    else:
        score -= 16.0
        reasons.append("- no_number_detected")

    if detected_number == bag_number:
        if no_bag_structure_number_only:
            score += 12.0
            reasons.append(f"+ exact_bag_number_match_weak:{bag_number}")
            reasons.append("- no_bag_structure_number_only")
        else:
            score += 85.0
            reasons.append(f"+ exact_bag_number_match:{bag_number}")
    elif detected_number is not None:
        jump = abs(int(detected_number) - int(bag_number))
        penalty = 25.0 + min(30.0, float(jump) * 8.0)
        score -= penalty
        reasons.append(f"- conflicting_bag_number:{detected_number}")
    else:
        score -= 10.0
        reasons.append("- no_bag_number")

    if row.get("panel_found"):
        score += 18.0
        reasons.append("+ panel_found")
    else:
        score -= 8.0
        reasons.append("- no_panel")

    if row.get("shell_found"):
        score += 10.0
        reasons.append("+ shell_found")

    if row.get("grey_bag_found"):
        score += 12.0
        reasons.append("+ grey_bag_found")

    if row.get("strong_structure"):
        score += 12.0
        reasons.append("+ strong_structure")
    else:
        score -= 12.0
        reasons.append("- weak_structure")

    score += confidence * 35.0
    if confidence >= 0.75:
        reasons.append(f"+ high_confidence:{confidence:.2f}")
    elif confidence <= 0.30:
        score -= 14.0
        reasons.append(f"- low_confidence:{confidence:.2f}")

    span = max(1, int(end_page) - int(start_page))
    distance_from_start = max(0, page - int(start_page))
    early_bias = 1.0 - (float(distance_from_start) / float(span))
    score += early_bias * 14.0
    if early_bias >= 0.66:
        reasons.append("+ near_window_start")
    elif early_bias <= 0.20:
        score -= 8.0
        reasons.append("- late_in_window")

    if row.get("overview_page"):
        score -= 22.0
        reasons.append("- overview_like_page")

    if page_kind != "other":
        score -= 18.0
        reasons.append(f"- precheck_page_kind:{page_kind}")

    if no_bag_structure_number_only:
        score -= 150.0
        reasons.append("- no_bag_structure_number_only")

    if green_box_count >= 4:
        score -= 90.0
        reasons.append("- multi_step_green_boxes")
    elif green_box_count >= 2:
        score -= 55.0
        reasons.append("- multi_step_green_boxes")

    if (
        page_kind == "other"
        and not row.get("number_box_found")
        and not row.get("strong_structure")
        and detected_number is None
    ):
        score -= 15.0
        reasons.append("- looks_like_normal_instruction_page")

    return round(score, 3), reasons


def _build_candidate(set_num: str, row: dict, bag_number: int, start_page: int, end_page: int) -> dict:
    score, reasons = _score_window_candidate(set_num, row, bag_number, start_page, end_page)

    return {
        "page": int(row["page"]),
        "score": score,
        "reasons": reasons,
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "bag_number": row.get("bag_number"),
        "panel_found": bool(row.get("panel_found")),
        "panel_source": row.get("panel_source"),
        "shell_found": bool(row.get("shell_found")),
        "shell_method": row.get("shell_method"),
        "grey_bag_found": bool(row.get("grey_bag_found")),
        "number_box_found": bool(row.get("number_box_found")),
        "overview_page": bool(row.get("overview_page")),
        "strong_structure": bool(row.get("strong_structure")),
        "page_kind": row.get("page_kind", "other"),
        "ocr_raw": row.get("ocr_raw", "") or "",
        "analysis_source": "cached_analyzer_row" if row.get("cache_hit") else "window_page_analysis",
    }


def _scan_window(set_num: str, bag_number: int, window: dict, fast: bool = False) -> dict:
    bounded_window = _bounded_window_from_missing_window(window)
    if bounded_window is None:
        return {
            "bag": int(bag_number),
            "status": "unbounded",
            "window": None,
            "candidates": [],
            "previous_anchor": {
                "bag": window.get("previous_confirmed_bag"),
                "page": window.get("previous_confirmed_page"),
            },
            "next_anchor": {
                "bag": window.get("next_confirmed_bag"),
                "page": window.get("next_confirmed_page"),
            },
            "reason": "missing bounded anchor pages",
        }

    start_page, end_page = bounded_window
    confirmed_pages = set(int(page) for page in truth_service.get_all_confirmed_pages(set_num))

    rows = []
    pages_considered = []
    pages_missing = []
    rows_missing = []
    reused_rows = 0
    analyzed_rows = 0

    for page in range(int(start_page), int(end_page) + 1):
        if page in confirmed_pages:
            continue

        if debug_service.resolve_page_image_path(set_num, int(page)) is None:
            pages_missing.append(int(page))
            continue

        normalized = _get_cached_window_row(set_num, int(page)) if fast else _get_or_build_window_row(set_num, int(page))
        if normalized is None:
            rows_missing.append(int(page))
            continue

        rows.append(normalized)
        pages_considered.append(int(page))

        if normalized.get("cache_hit"):
            reused_rows += 1
        else:
            analyzed_rows += 1

    candidates = [
        _build_candidate(set_num, row, int(bag_number), int(start_page), int(end_page))
        for row in rows
    ]
    candidates.sort(key=cmp_to_key(_compare_candidates))

    return {
        "bag": int(bag_number),
        "status": "ok",
        "window": [int(start_page), int(end_page)],
        "candidates": candidates,
        "previous_anchor": {
            "bag": window.get("previous_confirmed_bag"),
            "page": window.get("previous_confirmed_page"),
        },
        "next_anchor": {
            "bag": window.get("next_confirmed_bag"),
            "page": window.get("next_confirmed_page"),
        },
        "pages_considered": pages_considered,
        "pages_missing": pages_missing,
        "rows_missing": rows_missing,
        "rows_reused": int(reused_rows),
        "rows_analyzed": int(analyzed_rows),
    }


def _build_gap_scan_item(set_num: str, bag_number: int, window: dict, fast: bool = False) -> dict:
    confirmed_page = truth_service.get_confirmed_page_for_bag(set_num, int(bag_number))
    if confirmed_page is not None:
        return {
            "bag": int(bag_number),
            "status": "already_confirmed",
            "window": [int(confirmed_page), int(confirmed_page)],
            "candidates": [],
            "confirmed_page": int(confirmed_page),
        }

    return _scan_window(set_num, int(bag_number), window, fast=fast)


def scan_gaps(set_num: str, bag_number: Optional[int] = None, fast: bool = False):
    sequence = sequence_service.run_sequence_scan(set_num)
    missing_windows = sequence.get("missing_bag_windows", [])

    if bag_number is not None:
        target_bag = int(bag_number)
        missing_windows = [
            window
            for window in missing_windows
            if int(window.get("bag_number", 0) or 0) == target_bag
        ]

        if not missing_windows:
            confirmed_page = truth_service.get_confirmed_page_for_bag(set_num, target_bag)
            if confirmed_page is not None:
                gaps = [
                    {
                        "bag": int(target_bag),
                        "status": "already_confirmed",
                        "window": [int(confirmed_page), int(confirmed_page)],
                        "candidates": [],
                        "confirmed_page": int(confirmed_page),
                    }
                ]
            else:
                gaps = [
                    {
                        "bag": int(target_bag),
                        "status": "not_missing",
                        "window": None,
                        "candidates": [],
                    }
                ]
        else:
            gaps = [
                _build_gap_scan_item(set_num, int(window["bag_number"]), window, fast=fast)
                for window in missing_windows
            ]
    else:
        gaps = [
            _build_gap_scan_item(set_num, int(window["bag_number"]), window, fast=fast)
            for window in missing_windows
        ]

    return {
        "set_num": str(set_num),
        "gaps": gaps,
    }


def scan_gap_for_bag(set_num: str, bag_number: int, fast: bool = False):
    payload = scan_gaps(set_num, bag_number=int(bag_number), fast=fast)
    gap = payload.get("gaps", [])
    if not gap:
        return {
            "set_num": str(set_num),
            "bag_number": int(bag_number),
            "status": "not_missing",
            "window": None,
            "analysis_rows": [],
            "analysis_count": 0,
        }

    item = gap[0]
    window = item.get("window")

    if item.get("status") == "already_confirmed":
        confirmed_page = int(item.get("confirmed_page"))
        return {
            "set_num": str(set_num),
            "bag_number": int(bag_number),
            "status": "already_confirmed",
            "message": "bag already exists in DB truth",
            "accepted_page": confirmed_page,
            "window": {
                "previous_confirmed_bag": None,
                "previous_confirmed_page": None,
                "next_confirmed_bag": None,
                "next_confirmed_page": None,
            },
            "scan_start_page": confirmed_page,
            "scan_end_page": confirmed_page,
            "pages_considered": [],
            "pages_skipped_confirmed": [],
            "analysis_rows": [],
            "analysis_count": 0,
            "top_candidate": None,
        }

    if item.get("status") != "ok":
        return {
            "set_num": str(set_num),
            "bag_number": int(bag_number),
            "status": str(item.get("status")),
            "message": item.get("reason", "gap scan unavailable"),
            "window": {
                "previous_confirmed_bag": item.get("previous_anchor", {}).get("bag"),
                "previous_confirmed_page": item.get("previous_anchor", {}).get("page"),
                "next_confirmed_bag": item.get("next_anchor", {}).get("bag"),
                "next_confirmed_page": item.get("next_anchor", {}).get("page"),
            },
            "scan_start_page": None if not window else int(window[0]),
            "scan_end_page": None if not window else int(window[1]),
            "pages_considered": item.get("pages_considered", []),
            "pages_missing": item.get("pages_missing", []),
            "rows_missing": item.get("rows_missing", []),
            "pages_skipped_confirmed": [],
            "analysis_rows": item.get("candidates", []),
            "analysis_count": len(item.get("candidates", [])),
            "top_candidate": None,
        }

    return {
        "set_num": str(set_num),
        "bag_number": int(bag_number),
        "status": "ok",
        "window": {
            "previous_confirmed_bag": item.get("previous_anchor", {}).get("bag"),
            "previous_confirmed_page": item.get("previous_anchor", {}).get("page"),
            "next_confirmed_bag": item.get("next_anchor", {}).get("bag"),
            "next_confirmed_page": item.get("next_anchor", {}).get("page"),
        },
        "scan_start_page": int(window[0]),
        "scan_end_page": int(window[1]),
        "pages_considered": item.get("pages_considered", []),
        "pages_missing": item.get("pages_missing", []),
        "rows_missing": item.get("rows_missing", []),
        "pages_skipped_confirmed": [],
        "analysis_rows": item.get("candidates", []),
        "analysis_count": len(item.get("candidates", [])),
        "top_candidate": item.get("candidates", [None])[0],
        "rows_reused": int(item.get("rows_reused", 0) or 0),
        "rows_analyzed": int(item.get("rows_analyzed", 0) or 0),
    }
