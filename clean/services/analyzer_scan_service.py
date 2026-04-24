from clean.services import page_analyzer, debug_service, precheck_service


_SKIP_PAGE_KINDS = {"cover_page", "intro_or_legal", "parts_page"}
_ANALYZER_SCAN_CACHE = {}
_SHORTLIST_LIMIT = 40


def _list_rendered_pages(set_num: str):
    pages_dir = debug_service._find_latest_pages_dir_for_set(set_num)
    if pages_dir is None:
        return None, []

    pages = []
    for path in sorted(pages_dir.glob("page_*.png")):
        stem = path.stem
        try:
            page_num = int(stem.replace("page_", ""))
        except ValueError:
            continue
        pages.append(page_num)

    return pages_dir, pages


def _build_skipped_row(page: int, precheck: dict):
    page_kind = precheck.get("page_kind", "other")
    return {
        "page": int(page),
        "precheck_kind": page_kind,
        "skipped": True,
        "skip_reason": page_kind,
        "precheck": precheck,
        "analyzer_summary": {
            "panel_found": False,
            "panel_source": None,
            "shell_found": False,
            "grey_bag_found": False,
            "number_found": False,
            "bag_number": None,
            "confidence": 0.0,
        },
        "panel_found": False,
        "panel_source": None,
        "shell_found": False,
        "grey_bag_found": False,
        "number_found": False,
        "number_box_found": False,
        "bag_number": None,
        "confidence": 0.0,
        "ocr_raw": "",
        "analysis_available": False,
        "cache_hit": False,
    }


def _build_sequence_scan_row(page: int):
    result = page_analyzer.analyze_page(page, include_image=False)
    ocr_raw = result.get("ocr_raw", "") or ""

    row = {
        "page": int(page),
        "precheck_kind": result.get("page_kind", "other"),
        "skipped": False,
        "skip_reason": "",
        "precheck": {},
        "analyzer_summary": {
            "panel_found": result.get("panel_found"),
            "panel_source": result.get("panel_source"),
            "shell_found": result.get("shell_found"),
            "grey_bag_found": result.get("grey_bag_found"),
            "number_found": result.get("number_found"),
            "bag_number": result.get("bag_number"),
            "confidence": result.get("confidence"),
        },
        "panel_found": bool(result.get("panel_found")),
        "panel_source": result.get("panel_source"),
        "shell_found": bool(result.get("shell_found")),
        "shell_method": result.get("shell_method"),
        "grey_bag_found": bool(result.get("grey_bag_found")),
        "number_found": bool(result.get("number_found")),
        "number_box_found": bool(result.get("number_box_found", result.get("number_found"))),
        "bag_number": result.get("bag_number"),
        "confidence": result.get("confidence"),
        "ocr_raw": ocr_raw,
        "overview_page": bool(
            result.get("overview_page")
            or result.get("panel_source") == "overview_page"
        ),
        "strong_structure": bool(
            result.get("strong_structure")
            or (
                bool(result.get("panel_found"))
                and bool(result.get("shell_found"))
                and bool(result.get("grey_bag_found"))
            )
        ),
        "analysis_available": True,
        "cache_hit": False,
    }
    return row


def get_or_build_sequence_scan_row(set_num: str, page: int):
    pages_dir, _ = _list_rendered_pages(set_num)
    if pages_dir is None:
        raise RuntimeError(f"no rendered pages found for set {set_num}")

    page_analyzer.configure_pages_dir(str(pages_dir))
    return _get_cached_or_scan(set_num, int(page))


def _get_cached_or_scan(set_num: str, page: int):
    cache_key = (str(set_num), int(page))

    cached = _ANALYZER_SCAN_CACHE.get(cache_key)
    if cached is not None:
        row = dict(cached)
        row["cache_hit"] = True
        return row

    row = _build_sequence_scan_row(page)

    _ANALYZER_SCAN_CACHE[cache_key] = dict(row)
    return row


def _row_sort_key(row):
    summary = row.get("analyzer_summary", {})
    return (
        1 if row.get("skipped") else 0,
        -int(bool(summary.get("grey_bag_found"))),
        -int(bool(summary.get("number_found"))),
        -int(bool(summary.get("shell_found"))),
        -int(bool(summary.get("panel_found"))),
        -float(summary.get("confidence", 0.0) or 0.0),
        int(row.get("page", 0)),
    )

def _shortlist_sort_key(row):
    summary = row.get("analyzer_summary", {})
    return (
        -float(summary.get("confidence", 0.0) or 0.0),
        int(row.get("page", 0)),
    )

def _is_shortlist_candidate(row):
    if row.get("skipped"):
        return False

    summary = row.get("analyzer_summary", {})
    confidence = float(summary.get("confidence", 0.0) or 0.0)
    number_found = bool(summary.get("number_found"))
    grey_bag_found = bool(summary.get("grey_bag_found"))
    panel_found = bool(summary.get("panel_found"))

    if confidence < 0.50:
        return False

    if not number_found and not grey_bag_found:
        return False

    return (
        summary.get("bag_number") is not None
        or confidence >= 0.60
        or (panel_found and number_found)
    )

def scan_set_with_analyzer(set_num: str, include_all: bool = True):
    pages_dir, pages = _list_rendered_pages(set_num)

    if pages_dir is None:
        return {
            "ok": False,
            "set_num": set_num,
            "error": "no rendered pages found for set",
            "pages_dir": None,
            "rows": [],
        }

    # 🔑 IMPORTANT: set pages_dir for analyzer
    page_analyzer.configure_pages_dir(str(pages_dir))

    prechecked_pages = []
    skipped_page_count = 0
    shortlist_page_count = 0

    for page in pages:
        precheck = precheck_service.get_page_precheck(set_num, page)
        page_kind = precheck.get("page_kind", "other")

        skipped = page_kind in _SKIP_PAGE_KINDS

        prechecked_pages.append(
            {
                "page": int(page),
                "precheck": precheck,
                "skipped": skipped,
            }
        )

        if skipped:
            skipped_page_count += 1
        else:
            shortlist_page_count += 1

    rows = []
    cache_hits = 0
    analyzed_page_count = 0

    for entry in prechecked_pages:
        page = int(entry["page"])
        precheck = entry["precheck"]

        if entry["skipped"]:
            row = _build_skipped_row(page, precheck)
        else:
            row = _get_cached_or_scan(set_num, page)
            row["precheck"] = precheck

            if row.get("cache_hit"):
                cache_hits += 1
            else:
                analyzed_page_count += 1

        if include_all or _is_shortlist_candidate(row):
            rows.append(row)

    if include_all:
        rows.sort(key=_row_sort_key)
    else:
        rows.sort(key=_shortlist_sort_key)
        rows = rows[:_SHORTLIST_LIMIT]

    return {
        "ok": True,
        "set_num": set_num,
        "pages_dir": str(pages_dir),
        "page_count": len(pages),
        "shortlist_page_count": shortlist_page_count,
        "skipped_page_count": skipped_page_count,
        "analyzed_page_count": analyzed_page_count,
        "cache_hits": cache_hits,
        "row_count": len(rows),
        "rows": rows,
    }
