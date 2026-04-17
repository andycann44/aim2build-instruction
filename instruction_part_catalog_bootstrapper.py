#!/usr/bin/env python3
"""
Instruction Part Catalog Bootstrapper (local tool)

This script helps you build a folder + JSON manifest structure for your own
custom brick images, using your existing Rebrickable-style SQLite database.

What it does
------------
- Scans a folder of instruction PDFs (for YOUR reference only).
- Guesses a LEGO set number from each PDF filename.
- Looks up that set in your Rebrickable-style SQLite DB
  (e.g. lego_catalog.db) to pull the parts list.
- Creates an output structure like:

  out_dir/
    parts_index.json         # global index of all (part_num, color_id)
    assets/
      parts/
        3001-5.png           # where YOU will place your own art
    sets/
      21330-1/
        manifest.json        # set metadata + parts list
        21330-1.pdf.link     # text file pointing to your original PDF

It does NOT:
- Extract or copy any official LEGO images.
- Modify your PDFs.
- Talk to any external APIs.

You stay fully in control of the artwork. The PNGs in assets/parts are YOUR
images (hand-drawn, AI, LDraw-based renders, etc.).

Typical usage
-------------
    python instruction_part_catalog_bootstrapper.py \
        --pdf-dir /path/to/instruction_pdfs \
        --db-path ./backend/app/data/lego_catalog.db \
        --out-dir ./backend/app/data/instruction_catalog \
        --create-placeholders

Later, your FastAPI image router can serve files from the same out-dir.

NOTE: This script assumes a Rebrickable-style schema with these tables:
    - sets (set_num TEXT PRIMARY KEY, name TEXT, year INT, ...)
    - inventories (id INT PK, set_num TEXT, ...)
    - inventory_parts (id INT PK, inventory_id INT FK, part_num TEXT,
                       color_id INT, quantity INT, is_spare INT)

It ignores spare parts (is_spare = 1) if that column exists.
"""

import argparse
import json
import logging
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional pillow import for placeholder thumbnails
try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:  # pragma: no cover - pillow is optional
    Image = None  # type: ignore


LOG = logging.getLogger("instruction_catalog_bootstrapper")


@dataclass
class PartEntry:
    part_num: str
    color_id: int
    quantity: int


@dataclass
class SetManifest:
    set_num: str
    name: Optional[str]
    year: Optional[int]
    source_pdf: Dict[str, str]
    parts: List[PartEntry]


def guess_set_num_from_filename(name: str) -> Optional[str]:
    """
    Guess a LEGO set number from a filename.

    Strategy:
    - Look for patterns like '21330-1' first.
    - If none, look for a plain number '21330' (3–7 digits) and normalise
      to '21330-1' as a default.
    """
    stem = Path(name).stem

    # Prefer pattern with dash: 1234-1
    m_dash = re.search(r"(\d{3,7}-\d+)", stem)
    if m_dash:
        return m_dash.group(1)

    # Fallback: plain number
    m_plain = re.search(r"(\d{3,7})", stem)
    if m_plain:
        digits = m_plain.group(1)
        # Default Rebrickable style uses -1 as suffix for main release
        return f"{digits}-1"

    return None


def connect_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise SystemExit(f"DB not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def load_set_meta(conn: sqlite3.Connection, set_num: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Load basic set metadata (name, year) from the sets table.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT name, year FROM sets WHERE set_num = ?",
        (set_num,),
    )
    row = cur.fetchone()
    if not row:
        LOG.warning("No metadata found for set_num=%s", set_num)
        return None, None
    return row["name"], row["year"]


def load_set_parts(conn: sqlite3.Connection, set_num: str) -> List[PartEntry]:
    """
    Load parts for a given set_num using a Rebrickable-style schema.

    We auto-detect the join columns between inventories and inventory_parts,
    because different dumps may use different PK/FK names (id vs inventory_id).
    """
    cur = conn.cursor()

    # Detect inventories PK column
    cur.execute("PRAGMA table_info(inventories)")
    inv_rows = cur.fetchall()
    if not inv_rows:
        raise RuntimeError("Table 'inventories' not found in DB")

    inv_cols = [r["name"] for r in inv_rows]
    if "id" in inv_cols:
        inv_pk = "id"
    elif "inventory_id" in inv_cols:
        inv_pk = "inventory_id"
    else:
        inv_pk = inv_cols[0]

    # Detect inventory_parts FK column
    cur.execute("PRAGMA table_info(inventory_parts)")
    ip_rows = cur.fetchall()
    if not ip_rows:
        raise RuntimeError("Table 'inventory_parts' not found in DB")

    ip_cols = [r["name"] for r in ip_rows]
    if "inventory_id" in ip_cols:
        ip_fk = "inventory_id"
    elif "inventoryid" in ip_cols:
        ip_fk = "inventoryid"
    else:
        ip_fk = ip_cols[0]

    # Build base query using detected columns
    base_sql = f"""
        SELECT ip.part_num, ip.color_id, ip.quantity
        FROM inventories i
        JOIN inventory_parts ip ON i.{inv_pk} = ip.{ip_fk}
        WHERE i.set_num = ?
    """

    # If there's an is_spare column, ignore spares
    if "is_spare" in ip_cols:
        base_sql += " AND (ip.is_spare = 0 OR ip.is_spare IS NULL)"

    cur.execute(base_sql, (set_num,))
    parts: List[PartEntry] = []
    for row in cur.fetchall():
        parts.append(
            PartEntry(
                part_num=row["part_num"],
                color_id=int(row["color_id"]),
                quantity=int(row["quantity"]),
            )
        )

    if not parts:
        LOG.warning("No parts found for set_num=%s", set_num)
    return parts




def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_placeholder_png(path: Path, text: str) -> None:
    """
    Create a simple placeholder PNG with the given text.
    Uses Pillow if available; otherwise, does nothing.
    """
    if Image is None:
        LOG.debug("Pillow not available; skipping placeholder for %s", path)
        return

    size = (256, 256)
    img = Image.new("RGB", size, (32, 32, 32))
    draw = ImageDraw.Draw(img)

    try:
        # Use a simple default font; system-dependent
        font = ImageFont.load_default()
    except Exception:
        font = None  # type: ignore

    text = text[:10]  # keep it short
    w, h = draw.textsize(text, font=font)
    pos = ((size[0] - w) // 2, (size[1] - h) // 2)
    draw.text(pos, text, fill=(220, 220, 220), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), format="PNG")


def build_instruction_catalog(
    pdf_dir: Path,
    db_path: Path,
    out_dir: Path,
    create_placeholders: bool = False,
) -> None:
    """
    Main pipeline:
    - Scan PDFs
    - Resolve set numbers
    - Read parts from DB
    - Write per-set manifests
    - Build global parts_index.json
    - Optionally create placeholder thumbnails
    """
    ensure_dir(out_dir)
    assets_parts_dir = out_dir / "assets" / "parts"
    sets_dir = out_dir / "sets"
    ensure_dir(assets_parts_dir)
    ensure_dir(sets_dir)

    conn = connect_db(db_path)

    # Collect global index: (part_num, color_id) -> {total_qty, sets: [..]}
    parts_index: Dict[Tuple[str, int], Dict[str, object]] = defaultdict(
        lambda: {"total_qty": 0, "sets": []}
    )

    pdf_files = sorted(
        [p for p in pdf_dir.glob("**/*") if p.is_file() and p.suffix.lower() == ".pdf"]
    )
    if not pdf_files:
        LOG.warning("No PDFs found under %s", pdf_dir)

    for pdf_path in pdf_files:
        rel_pdf = pdf_path.relative_to(pdf_dir)
        LOG.info("Processing PDF: %s", rel_pdf)

        set_num = guess_set_num_from_filename(pdf_path.name)
        if not set_num:
            LOG.warning("Could not guess set number from %s", pdf_path.name)
            continue

        name, year = load_set_meta(conn, set_num)
        parts = load_set_parts(conn, set_num)
        if not parts:
            LOG.warning("Skipping set %s because it has no parts", set_num)
            continue

        # Build manifest
        manifest = SetManifest(
            set_num=set_num,
            name=name,
            year=year,
            source_pdf={
                "filename": pdf_path.name,
                "relative_path_from_pdf_dir": str(rel_pdf),
                "absolute_path": str(pdf_path.resolve()),
            },
            parts=parts,
        )

        set_dir = sets_dir / set_num
        ensure_dir(set_dir)

        manifest_path = set_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "set_num": manifest.set_num,
                    "name": manifest.name,
                    "year": manifest.year,
                    "source_pdf": manifest.source_pdf,
                    "parts": [asdict(p) for p in manifest.parts],
                },
                f,
                indent=2,
                sort_keys=True,
            )

        # Also store a tiny .link file pointing to original PDF
        link_path = set_dir / f"{set_num}.pdf.link"
        with link_path.open("w", encoding="utf-8") as f:
            f.write(str(pdf_path.resolve()))

        LOG.info("Wrote manifest for %s to %s", set_num, manifest_path)

        # Update global index
        for p in parts:
            key = (p.part_num, p.color_id)
            entry = parts_index[key]
            entry["total_qty"] = int(entry["total_qty"]) + int(p.quantity)
            entry["sets"].append(
                {"set_num": set_num, "quantity": int(p.quantity)}
            )

            if create_placeholders:
                img_path = assets_parts_dir / f"{p.part_num}-{p.color_id}.png"
                if not img_path.exists():
                    create_placeholder_png(img_path, f"{p.part_num}:{p.color_id}")

    # Write global parts_index.json
    index_path = out_dir / "parts_index.json"
    parts_list = []
    for (part_num, color_id), data in sorted(
        parts_index.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        parts_list.append(
            {
                "part_num": part_num,
                "color_id": color_id,
                "total_qty": data["total_qty"],
                "sets": data["sets"],
            }
        )

    with index_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_by": "instruction_part_catalog_bootstrapper",
                "parts": parts_list,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    LOG.info("Wrote global parts_index.json to %s", index_path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an instruction part catalog + folder structure for custom images."
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="Folder containing instruction PDFs (scanned recursively).",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to Rebrickable-style SQLite DB (e.g. lego_catalog.db).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output folder for catalog (parts_index.json, assets/parts, sets/*).",
    )
    parser.add_argument(
        "--create-placeholders",
        action="store_true",
        help="If set, creates simple placeholder PNGs for missing part thumbnails.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv).",
    )
    return parser.parse_args(argv)


def configure_logging(verbosity: int) -> None:
    level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 0:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    LOG.info("PDF dir: %s", pdf_dir)
    LOG.info("DB path: %s", db_path)
    LOG.info("Out dir: %s", out_dir)

    build_instruction_catalog(
        pdf_dir=pdf_dir,
        db_path=db_path,
        out_dir=out_dir,
        create_placeholders=bool(args.create_placeholders),
    )


if __name__ == "__main__":
    main()
