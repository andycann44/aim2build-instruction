import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

DB_PATH = Path("/Users/olly/aim2build-instruction/debug/server_catalog/lego_catalog.db")


def _normalize_set_id(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return s
    if "-" not in s:
        return f"{s}-1"
    return s


def _connect():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def load_instruction_set_parts(set_num: str) -> Dict:
    set_id = _normalize_set_id(set_num)

    con = _connect()
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # --- latest inventory ---
    row = cur.execute(
        """
        SELECT inventory_id
        FROM inventories
        WHERE set_num = ?
        ORDER BY version DESC, inventory_id DESC
        LIMIT 1
        """,
        (set_id,),
    ).fetchone()

    if not row:
        return {"set_num": set_id, "parts": []}

    inventory_id = int(row["inventory_id"])

    # --- parts ---
    parts = cur.execute(
        """
        SELECT part_num, color_id, SUM(quantity) AS quantity
        FROM inventory_parts
        WHERE inventory_id = ?
          AND COALESCE(is_spare, 0) = 0
        GROUP BY part_num, color_id
        HAVING COALESCE(SUM(quantity), 0) > 0
        """,
        (inventory_id,),
    ).fetchall()

    out: List[Dict] = []

    for p in parts:
        part_num = str(p["part_num"])
        color_id = int(p["color_id"])
        qty = int(p["quantity"])

        # element_id
        e = cur.execute(
            """
            SELECT MIN(element_id) AS element_id
            FROM elements
            WHERE part_num = ? AND color_id = ?
            """,
            (part_num, color_id),
        ).fetchone()

        element_id = e["element_id"] if e else None

        # image
        img = cur.execute(
            """
            SELECT img_url
            FROM element_images
            WHERE part_num = ? AND color_id = ?
            LIMIT 1
            """,
            (part_num, color_id),
        ).fetchone()

        img_url = img["img_url"] if img else None

        # color info
        c = cur.execute(
            """
            SELECT name, rgb
            FROM colors
            WHERE color_id = ?
            """,
            (color_id,),
        ).fetchone()

        color_name = c["name"] if c else None
        rgb = c["rgb"] if c else None

        out.append(
            {
                "part_num": part_num,
                "color_id": color_id,
                "color_name": color_name,
                "rgb": rgb,
                "element_id": element_id,
                "qty": qty,
                "img_url": img_url,
                "needs_image": img_url is None,
                "shape_key": part_num,
                "color_key": color_id,
            }
        )

    con.close()

    return {
        "set_num": set_id,
        "inventory_id": inventory_id,
        "parts": out,
    }
