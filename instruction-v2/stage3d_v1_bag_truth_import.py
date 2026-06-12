import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from paths import INDEXES_DIR, ROOT_DIR


PROJECT_ROOT = ROOT_DIR.parent
V1_BAG_TRUTH_DB = PROJECT_ROOT / "debug" / "bag_truth.db"
OUT_PATH = INDEXES_DIR / "03d_v1_bag_truth_import.json"


def _row_to_entry(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "set_num": str(row["set_num"]),
        "bag": int(row["bag_number"]),
        "bag_number": int(row["bag_number"]),
        "start_page": int(row["start_page"]),
        "source": "v1_bag_truth_db",
        "v1_source": row["source"],
        "confidence": row["confidence"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def import_v1_bag_truth(set_num: str) -> Dict[str, Any]:
    if not V1_BAG_TRUTH_DB.exists():
        raise FileNotFoundError(f"V1 bag truth DB not found: {V1_BAG_TRUTH_DB}")

    conn = sqlite3.connect(str(V1_BAG_TRUTH_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT set_num, bag_number, start_page, source, confidence, created_at, updated_at
            FROM bag_truth
            WHERE set_num = ?
            ORDER BY bag_number ASC, start_page ASC
            """,
            (str(set_num),),
        ).fetchall()
    finally:
        conn.close()

    entries: List[Dict[str, Any]] = [_row_to_entry(row) for row in rows]
    return {
        "stage": "03d",
        "name": "v1_bag_truth_import",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "set_num": str(set_num),
        "source": "v1_bag_truth_db",
        "source_db": str(V1_BAG_TRUTH_DB),
        "read_only": True,
        "entry_count": len(entries),
        "bags": entries,
    }


def main() -> int:
    payload = import_v1_bag_truth("70618")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(str(OUT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
