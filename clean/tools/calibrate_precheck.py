import sqlite3
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path("/Users/olly/aim2build-instruction")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean.services import precheck_service

DB_PATH = PROJECT_ROOT / "bag_inspector.db"

SIGNAL_KEYS = [
    "word_count",
    "numeric_token_count",
    "line_count",
    "bright_ratio",
    "edge_density",
    "large_box_count",
    "largest_box_area_ratio",
    "bag_start_score",
]


def _fetch_confirmed_bag_pages(set_num: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT bag_number, start_page
            FROM bag_truth
            WHERE set_num = ? AND confirmed = 1
            ORDER BY bag_number ASC, start_page ASC
            """,
            (set_num,),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "bag_number": int(row["bag_number"]),
            "start_page": int(row["start_page"]),
        }
        for row in rows
    ]


def _median(values):
    return statistics.median(values) if values else 0.0


def _summarize_rows(title: str, rows):
    print(f"\n=== {title} ===")
    print(f"rows: {len(rows)}")
    if not rows:
        return

    for key in SIGNAL_KEYS:
        values = [float(row[key]) for row in rows]
        print(
            f"{key}: min={min(values):.4f} med={_median(values):.4f} max={max(values):.4f}"
        )


def _print_rows(title: str, rows):
    print(f"\n--- {title} ---")
    for row in rows:
        summary = {
            "page": row["page"],
            "page_kind": row["page_kind"],
            "bag_start_score": row["bag_start_score"],
            "bag_start_label": row["bag_start_label"],
            "word_count": row["word_count"],
            "numeric_token_count": row["numeric_token_count"],
            "line_count": row["line_count"],
            "bright_ratio": row["bright_ratio"],
            "edge_density": row["edge_density"],
            "large_box_count": row["large_box_count"],
            "largest_box_area_ratio": row["largest_box_area_ratio"],
        }
        print(summary)


def _known_70618_classes():
    return {
        1: "cover_page",
        2: "intro_or_legal",
        3: "intro_or_legal",
        4: "intro_or_legal",
        5: "intro_or_legal",
        6: "bag_candidate",
        7: "build_page",
        8: "build_page",
        9: "build_page",
        10: "build_page",
        11: "build_page",
        12: "build_page",
        13: "build_page",
        14: "build_page",
        15: "build_page",
        16: "build_page",
        17: "build_page",
        18: "build_page",
        19: "build_page",
        20: "build_page",
        21: "build_page",
        22: "bag_candidate",
        23: "build_page",
        24: "build_page",
        25: "build_page",
    }


def main():
    confirmed_21330 = _fetch_confirmed_bag_pages("21330")
    confirmed_rows = []
    for item in confirmed_21330:
        page = int(item["start_page"])
        row = precheck_service.text_heavy_precheck("21330", page)
        row["bag_number"] = int(item["bag_number"])
        confirmed_rows.append(row)

    _print_rows("21330 confirmed bag starts", confirmed_rows)
    _summarize_rows("21330 confirmed bag starts", confirmed_rows)

    pages_70618 = []
    expected_classes = _known_70618_classes()
    for page in range(1, 26):
        row = precheck_service.text_heavy_precheck("70618", page)
        row["expected_class"] = expected_classes.get(page, "unknown")
        pages_70618.append(row)

    _print_rows("70618 pages 1-25", pages_70618)

    grouped = {}
    for row in pages_70618:
        key = row["expected_class"]
        grouped.setdefault(key, []).append(row)

    for class_name in sorted(grouped):
        _summarize_rows(f"70618 expected {class_name}", grouped[class_name])

    false_positives = [
        row
        for row in pages_70618
        if row["page_kind"] != "bag_candidate" and float(row["bag_start_score"]) >= 0.60
    ]
    _print_rows("70618 false positives currently scored high", false_positives)
    _summarize_rows("70618 false positives currently scored high", false_positives)


if __name__ == "__main__":
    main()
