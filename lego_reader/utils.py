from __future__ import annotations

import json
import logging
import re
from pathlib import Path


LOG = logging.getLogger("lego_reader")
SET_NUM_RE = re.compile(r"^\d{3,7}(?:-\d+)?$")


def normalize_set_num(raw_set_num: str) -> str:
    candidate = raw_set_num.strip()
    if not SET_NUM_RE.fullmatch(candidate):
        raise ValueError(f"Invalid LEGO set number: {raw_set_num!r}")
    return candidate.split("-", 1)[0]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
