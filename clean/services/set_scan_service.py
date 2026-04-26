from clean.services import debug_service, precheck_service


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


def scan_set_pages(set_num: str, include_all: bool = True):
    pages_dir, pages = _list_rendered_pages(set_num)

    if pages_dir is None:
        return {
            "ok": False,
            "set_num": set_num,
            "error": "no rendered pages found for set",
            "pages_dir": None,
            "rows": [],
        }

    rows = []
    for page in pages:
        precheck = precheck_service.text_heavy_precheck(set_num, page)

        row = {
            "page": page,
            "page_kind": precheck.get("page_kind", "other"),
            "bag_start_score": precheck.get("bag_start_score", 0.0),
            "bag_start_label": precheck.get("bag_start_label", "disabled"),
            "numeric_token_count": precheck.get("numeric_token_count", 0),
            "is_cover_page": precheck.get("is_cover_page", False),
            "is_likely_build_page": precheck.get("is_likely_build_page", False),
            "is_possible_bag_candidate": precheck.get("is_possible_bag_candidate", False),
            "word_count": precheck.get("word_count", 0),
            "line_count": precheck.get("line_count", 0),
            "bright_ratio": precheck.get("bright_ratio", 0.0),
            "edge_density": precheck.get("edge_density", 0.0),
            "large_box_count": precheck.get("large_box_count", 0),
            "largest_box_area_ratio": precheck.get("largest_box_area_ratio", 0.0),
        }

        if include_all:
            rows.append(row)
        else:
            if row["page_kind"] == "other":
                rows.append(row)

    rows.sort(key=lambda r: int(r["page"]))

    return {
        "ok": True,
        "set_num": set_num,
        "pages_dir": str(pages_dir),
        "page_count": len(pages),
        "row_count": len(rows),
        "rows": rows,
    }
