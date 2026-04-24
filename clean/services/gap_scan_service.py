from clean.services import truth_service, sequence_service

try:
    from clean.services import analyzer_scan_service
except Exception:
    analyzer_scan_service = None


def _derive_scan_range(window: dict):
    if not window:
        return None, None

    prev_page = window.get("previous_confirmed_page")
    next_page = window.get("next_confirmed_page")

    start_page = None
    end_page = None

    if prev_page is not None:
        start_page = int(prev_page) + 1

    if next_page is not None:
        end_page = int(next_page) - 1

    return start_page, end_page


def _row_has_candidate_signal(row: dict) -> bool:
    return bool(
        row.get("panel_found")
        or row.get("number_box_found")
        or row.get("bag_number") is not None
        or float(row.get("confidence", 0.0) or 0.0) > 0.2
    )


def _summarize_gap_candidate(row: dict) -> dict:
    return {
        "page": int(row.get("page", 0) or 0),
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "panel_found": bool(row.get("panel_found")),
        "panel_source": row.get("panel_source"),
        "shell_found": bool(row.get("shell_found")),
        "shell_method": row.get("shell_method"),
        "grey_bag_found": bool(row.get("grey_bag_found")),
        "number_box_found": bool(row.get("number_box_found")),
        "bag_number": row.get("bag_number"),
        "ocr_raw": row.get("ocr_raw", ""),
        "overview_page": bool(
            row.get("overview_page")
            or row.get("panel_source") == "overview_page"
        ),
        "strong_structure": bool(
            row.get("strong_structure")
            or (
                bool(row.get("panel_found"))
                and bool(row.get("shell_found"))
                and bool(row.get("grey_bag_found"))
            )
        ),
    }


def _score_gap_candidate(candidate: dict, bag_number: int, start_page: int, end_page: int):
    score = 0.0
    reasons = []

    detected_number = candidate.get("bag_number")
    confidence = float(candidate.get("confidence", 0.0) or 0.0)
    page = int(candidate.get("page", 0) or 0)

    if detected_number == bag_number:
        score += 150.0
        reasons.append("exact_bag_number")
    elif detected_number is not None:
        jump = abs(int(detected_number) - int(bag_number))
        penalty = min(80.0, float(jump) * 12.0)
        score -= penalty
        reasons.append(f"number_jump_{jump}")

    score += confidence * 30.0
    if confidence >= 0.70:
        reasons.append("high_confidence")

    if candidate.get("panel_found"):
        score += 25.0
        reasons.append("panel_found")

    if candidate.get("panel_found") and not candidate.get("number_box_found"):
        score -= 40.0
        reasons.append("panel_without_number_penalty")

    if candidate.get("panel_source") == "strict_top_left":
        score += 15.0
        reasons.append("strict_top_left_panel")

    if candidate.get("number_box_found"):
        score += 35.0
        reasons.append("number_box_found")

    if candidate.get("strong_structure"):
        score += 18.0
        reasons.append("strong_structure")
    else:
        if candidate.get("shell_found"):
            score += 4.0
        if candidate.get("grey_bag_found"):
            score += 6.0

    if candidate.get("overview_page"):
        score -= 35.0
        reasons.append("overview_penalty")

    if start_page is not None and end_page is not None and end_page >= start_page:
        span = max(1, end_page - start_page)
        pos = float(page - start_page) / float(span)
        score += max(0.0, 3.0 - (pos * 3.0))

    return score, reasons


def _rank_gap_candidates(rows, bag_number: int, start_page: int, end_page: int):
    candidates = []

    for row in rows:
        if not _row_has_candidate_signal(row):
            continue

        candidate = _summarize_gap_candidate(row)
        score, reasons = _score_gap_candidate(candidate, bag_number, start_page, end_page)
        candidate["score"] = round(score, 3)
        candidate["why"] = reasons
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            -float(item.get("confidence", 0.0) or 0.0),
            int(item.get("page", 0) or 0),
        )
    )
    return candidates


def _get_analysis_row_for_page(set_num: str, page: int) -> dict:
    if analyzer_scan_service is None:
        raise RuntimeError("clean.services.analyzer_scan_service is not importable")

    if hasattr(analyzer_scan_service, "get_or_build_sequence_scan_row"):
        row = analyzer_scan_service.get_or_build_sequence_scan_row(set_num, page)
    else:
        raise RuntimeError(
            "analyzer_scan_service does not expose get_or_build_sequence_scan_row"
        )

    normalized = dict(row)
    normalized["overview_page"] = bool(
        normalized.get("overview_page")
        or normalized.get("panel_source") == "overview_page"
    )
    normalized["strong_structure"] = bool(
        normalized.get("strong_structure")
        or (
            bool(normalized.get("panel_found"))
            and bool(normalized.get("shell_found"))
            and bool(normalized.get("grey_bag_found"))
        )
    )
    normalized["page"] = int(normalized.get("page", page) or page)
    return normalized


def _collect_window_rows(set_num: str, pages_considered):
    rows = []
    for page in pages_considered:
        try:
            rows.append(_get_analysis_row_for_page(set_num, int(page)))
        except Exception as exc:
            rows.append(
                {
                    "page": int(page),
                    "confidence": 0.0,
                    "panel_found": False,
                    "panel_source": None,
                    "shell_found": False,
                    "shell_method": None,
                    "grey_bag_found": False,
                    "number_box_found": False,
                    "bag_number": None,
                    "ocr_raw": f"ERROR: {exc}",
                    "overview_page": False,
                    "strong_structure": False,
                    "analysis_error": str(exc),
                }
            )
    return rows


def scan_gap_for_bag(set_num: str, bag_number: int):
    bag_number = int(bag_number)

    confirmed_page = truth_service.get_confirmed_page_for_bag(set_num, bag_number)
    if confirmed_page is not None:
        return {
            "set_num": set_num,
            "bag_number": bag_number,
            "status": "already_confirmed",
            "message": "bag already exists in DB truth",
            "accepted_page": int(confirmed_page),
            "analysis_rows": [],
            "analysis_count": 0,
        }

    window = sequence_service.get_missing_window(set_num, bag_number)
    if not window:
        return {
            "set_num": set_num,
            "bag_number": bag_number,
            "status": "no_window",
            "message": "could not build missing bag window from truth",
            "analysis_rows": [],
            "analysis_count": 0,
        }

    start_page, end_page = _derive_scan_range(window)

    if start_page is None and end_page is None:
        return {
            "set_num": set_num,
            "bag_number": bag_number,
            "status": "unbounded",
            "window": window,
            "message": "no bounded page range available yet",
            "analysis_rows": [],
            "analysis_count": 0,
        }

    if start_page is None:
        start_page = 1

    if end_page is None:
        return {
            "set_num": set_num,
            "bag_number": bag_number,
            "status": "unbounded",
            "window": window,
            "scan_start_page": int(start_page),
            "scan_end_page": None,
            "message": "missing next confirmed anchor; bounded scan not safe yet",
            "analysis_rows": [],
            "analysis_count": 0,
        }

    if end_page < start_page:
        end_page = start_page

    confirmed_pages = set(int(p) for p in truth_service.get_all_confirmed_pages(set_num))

    pages_considered = []
    pages_skipped_confirmed = []

    for page in range(int(start_page), int(end_page) + 1):
        if page in confirmed_pages:
            pages_skipped_confirmed.append(page)
            continue
        pages_considered.append(page)

    raw_rows = _collect_window_rows(set_num, pages_considered)

    # DEBUG (add this)
    print(f"[gap] pages={len(pages_considered)} rows={len(raw_rows)}")

    ranked_rows = _rank_gap_candidates(raw_rows, bag_number, int(start_page), int(end_page))

    return {
        "set_num": set_num,
        "bag_number": bag_number,
        "status": "ok",
        "window": window,
        "scan_start_page": int(start_page),
        "scan_end_page": int(end_page),
        "pages_considered": pages_considered,
        "pages_skipped_confirmed": pages_skipped_confirmed,
        "analysis_rows": ranked_rows,
        "analysis_count": len(ranked_rows),
        "top_candidate": ranked_rows[0] if ranked_rows else None,
    }
