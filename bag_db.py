import sqlite3

DEFAULT_DB_PATH = "bag_inspector.db"


def get_conn(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    with get_conn(db_path) as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bag_truth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                set_num TEXT NOT NULL,
                bag_number INTEGER NOT NULL,
                start_page INTEGER NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                confirmed INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_set_bag
            ON bag_truth(set_num, bag_number);
            """
        )

        conn.commit()


def insert_or_replace_bag_truth(
    set_num: str,
    bag_number: int,
    start_page: int,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    with get_conn(db_path) as conn:
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO bag_truth (
                set_num,
                bag_number,
                start_page
            )
            VALUES (?, ?, ?)
            ON CONFLICT(set_num, bag_number)
            DO UPDATE SET
                start_page = excluded.start_page
            """,
            (set_num, bag_number, start_page),
        )

        conn.commit()


def get_bag_truth(
    set_num: str,
    db_path: str = DEFAULT_DB_PATH,
) -> list[sqlite3.Row]:
    with get_conn(db_path) as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                id,
                set_num,
                bag_number,
                start_page,
                confidence,
                source,
                confirmed,
                created_at
            FROM bag_truth
            WHERE set_num = ?
            ORDER BY bag_number
            """,
            (set_num,),
        )

        return cur.fetchall()
