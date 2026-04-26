import os
import sqlite3


BAG_INSPECTOR_DB = "/Users/olly/aim2build-instruction/bag_inspector.db"


def _connect():
    if not os.path.exists(BAG_INSPECTOR_DB):
        return None
    conn = sqlite3.connect(BAG_INSPECTOR_DB)
    conn.row_factory = sqlite3.Row
    return conn


def get_all_confirmed_bags(set_num: str):
    conn = _connect()
    if not conn:
        return []

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT bag_number, start_page, confidence, source, confirmed
            FROM bag_truth
            WHERE set_num = ?
              AND confirmed = 1
            ORDER BY bag_number ASC, start_page ASC
            """,
            (str(set_num),),
        )
        rows = cur.fetchall()

        out = []
        for row in rows:
            out.append(
                {
                    "bag_number": int(row["bag_number"]),
                    "page": int(row["start_page"]),
                    "start_page": int(row["start_page"]),
                    "confidence": float(row["confidence"] or 0.0),
                    "source": row["source"],
                    "confirmed": int(row["confirmed"] or 0),
                }
            )
        return out

    except Exception as e:
        return [{"_error": str(e)}]

    finally:
        conn.close()


def get_confirmed_bag_map(set_num: str):
    rows = get_all_confirmed_bags(set_num)
    out = {}

    for row in rows:
        if "_error" in row:
            continue
        n = int(row["bag_number"])
        p = int(row["start_page"])
        out.setdefault(n, []).append(p)

    for n in out:
        out[n] = sorted(set(out[n]))
    return out


def get_confirmed_page_for_bag(set_num: str, bag_number: int):
    bag_map = get_confirmed_bag_map(set_num)
    pages = bag_map.get(int(bag_number), [])
    return pages[0] if pages else None


def get_all_confirmed_pages(set_num: str):
    rows = get_all_confirmed_bags(set_num)
    pages = []
    for row in rows:
        if "_error" in row:
            continue
        pages.append(int(row["start_page"]))
    return sorted(set(pages))


def is_page_confirmed(set_num: str, page: int):
    return int(page) in set(get_all_confirmed_pages(set_num))


def save_confirmed_bag_truth(set_num: str, bag_number: int, start_page: int):
    conn = _connect()
    if not conn:
        raise RuntimeError("bag_inspector.db not found")

    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO bag_truth
            (set_num, bag_number, start_page, confidence, source, confirmed)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(set_num), int(bag_number), int(start_page), 1.0, "manual", 1),
        )
        conn.commit()
    finally:
        conn.close()


def get_truth_for_set(set_num: str):
    rows = get_all_confirmed_bags(set_num)

    out = []
    for row in rows:
        if "_error" in row:
            continue
        out.append(
            {
                "bag_number": int(row["bag_number"]),
                "page": int(row["start_page"]),
                "start_page": int(row["start_page"]),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "source": row.get("source"),
                "confirmed": int(row.get("confirmed", 0) or 0),
            }
        )
    return out