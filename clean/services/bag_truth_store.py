import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from clean.services import debug_service


def _db_path():
    return debug_service.DEBUG_ROOT / "bag_truth.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bag_truth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                set_num TEXT NOT NULL,
                bag_number INTEGER NOT NULL,
                start_page INTEGER NOT NULL,
                source TEXT DEFAULT 'detector',
                confidence REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(set_num, bag_number)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bag_truth_conflicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                set_num TEXT NOT NULL,
                bag_number INTEGER NOT NULL,
                existing_page INTEGER,
                candidate_page INTEGER,
                source TEXT,
                confidence REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def get_bag_truth(set_num: str) -> List[Dict[str, Any]]:
    init_db()
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT id, set_num, bag_number, start_page, source, confidence, created_at, updated_at
            FROM bag_truth
            WHERE set_num = ?
            ORDER BY bag_number ASC, start_page ASC
            """,
            (str(set_num),),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_conflicts(set_num: str) -> List[Dict[str, Any]]:
    init_db()
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT id, set_num, bag_number, existing_page, candidate_page, source, confidence, created_at
            FROM bag_truth_conflicts
            WHERE set_num = ?
            ORDER BY id ASC
            """,
            (str(set_num),),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def upsert_bag_truth(
    set_num: str,
    bag_number: int,
    start_page: int,
    source: str = "detector",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    init_db()
    now = _utc_now()
    set_num = str(set_num)
    bag_number = int(bag_number)
    start_page = int(start_page)
    confidence_value = None if confidence is None else float(confidence)

    with closing(_connect()) as conn:
        existing = conn.execute(
            """
            SELECT id, set_num, bag_number, start_page, source, confidence, created_at, updated_at
            FROM bag_truth
            WHERE set_num = ? AND bag_number = ?
            """,
            (set_num, bag_number),
        ).fetchone()

        if existing is None:
            conn.execute(
                """
                INSERT INTO bag_truth (
                    set_num, bag_number, start_page, source, confidence, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    set_num,
                    bag_number,
                    start_page,
                    str(source),
                    confidence_value,
                    now,
                    now,
                ),
            )
            conn.commit()
            saved = conn.execute(
                """
                SELECT id, set_num, bag_number, start_page, source, confidence, created_at, updated_at
                FROM bag_truth
                WHERE set_num = ? AND bag_number = ?
                """,
                (set_num, bag_number),
            ).fetchone()
            return {
                "status": "inserted",
                "row": _row_to_dict(saved) if saved is not None else None,
                "conflict": None,
            }

        existing_dict = _row_to_dict(existing)
        if int(existing["start_page"]) == start_page:
            return {
                "status": "exists",
                "row": existing_dict,
                "conflict": None,
            }

        conn.execute(
            """
            INSERT INTO bag_truth_conflicts (
                set_num, bag_number, existing_page, candidate_page, source, confidence, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                set_num,
                bag_number,
                int(existing["start_page"]),
                start_page,
                str(source),
                confidence_value,
                now,
            ),
        )
        conn.commit()
        conflict = conn.execute(
            """
            SELECT id, set_num, bag_number, existing_page, candidate_page, source, confidence, created_at
            FROM bag_truth_conflicts
            WHERE rowid = last_insert_rowid()
            """
        ).fetchone()
        return {
            "status": "conflict",
            "row": existing_dict,
            "conflict": _row_to_dict(conflict) if conflict is not None else None,
        }


def save_many_bag_starts(
    set_num: str,
    bag_starts: List[Dict[str, Any]],
    source: str = "detector",
) -> Dict[str, Any]:
    init_db()
    operations: List[Dict[str, Any]] = []

    for item in bag_starts or []:
        bag_number = item.get("bag_number")
        start_page = item.get("start_page", item.get("page"))
        if bag_number is None or start_page is None:
            continue
        operation = upsert_bag_truth(
            set_num=set_num,
            bag_number=int(bag_number),
            start_page=int(start_page),
            source=str(source),
            confidence=item.get("confidence"),
        )
        operations.append(operation)

    return {
        "saved_truth": get_bag_truth(set_num),
        "conflicts": get_conflicts(set_num),
        "operations": operations,
    }
