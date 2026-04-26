from clean.services import analyzer_scan_service, truth_service


MIN_AUTO_CONFIRM_CONFIDENCE = 0.85


def _coerce_positive_int(value):
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _coerce_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _is_strong_candidate(row):
    if row.get("skipped"):
        return False

    summary = row.get("analyzer_summary", {})
    confidence = _coerce_float(summary.get("confidence"))
    bag_number = _coerce_positive_int(summary.get("bag_number"))

    return (
        bool(summary.get("grey_bag_found"))
        and bool(summary.get("number_found"))
        and confidence is not None
        and confidence >= MIN_AUTO_CONFIRM_CONFIDENCE
        and bag_number is not None
    )


def _candidate_from_row(row):
    summary = row.get("analyzer_summary", {})
    return {
        "bag_number": int(summary["bag_number"]),
        "start_page": int(row["page"]),
        "confidence": float(summary["confidence"]),
        "grey_bag_found": bool(summary.get("grey_bag_found")),
        "number_found": bool(summary.get("number_found")),
    }


def auto_confirm_top_candidates(set_num: str):
    scan = analyzer_scan_service.scan_set_with_analyzer(set_num, include_all=False)
    if not scan.get("ok"):
        return {
            "set_num": set_num,
            "ok": False,
            "error": scan.get("error", "analyzer scan failed"),
            "candidates_considered": 0,
            "auto_confirmed_count": 0,
            "auto_confirmed_rows": [],
            "skipped_due_to_existing_truth": [],
            "duplicate_candidates_discarded": [],
        }

    best_by_bag = {}
    duplicate_candidates_discarded = []
    candidates_considered = 0

    for row in scan.get("rows", []):
        if row.get("skipped"):
            continue

        candidates_considered += 1

        if not _is_strong_candidate(row):
            continue

        candidate = _candidate_from_row(row)
        bag_number = int(candidate["bag_number"])
        current = best_by_bag.get(bag_number)

        if current is None:
            best_by_bag[bag_number] = candidate
            continue

        candidate_key = (-candidate["confidence"], candidate["start_page"])
        current_key = (-current["confidence"], current["start_page"])
        if candidate_key < current_key:
            duplicate_candidates_discarded.append(current)
            best_by_bag[bag_number] = candidate
        else:
            duplicate_candidates_discarded.append(candidate)

    auto_confirmed_rows = []
    skipped_due_to_existing_truth = []

    for bag_number in sorted(best_by_bag):
        candidate = best_by_bag[bag_number]
        existing_page = truth_service.get_confirmed_page_for_bag(set_num, bag_number)

        if existing_page is not None and int(existing_page) != int(candidate["start_page"]):
            skipped_due_to_existing_truth.append(
                {
                    "bag_number": bag_number,
                    "existing_page": int(existing_page),
                    "candidate_page": int(candidate["start_page"]),
                    "confidence": float(candidate["confidence"]),
                }
            )
            continue

        truth_service.save_confirmed_bag_truth(
            set_num=set_num,
            bag_number=bag_number,
            start_page=int(candidate["start_page"]),
        )
        auto_confirmed_rows.append(candidate)

    return {
        "set_num": set_num,
        "ok": True,
        "candidates_considered": candidates_considered,
        "auto_confirmed_count": len(auto_confirmed_rows),
        "auto_confirmed_rows": auto_confirmed_rows,
        "skipped_due_to_existing_truth": skipped_due_to_existing_truth,
        "duplicate_candidates_discarded": duplicate_candidates_discarded,
    }
