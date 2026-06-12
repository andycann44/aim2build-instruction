import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from paths import INDEXES_DIR, ROOT_DIR


CALLOUT_MAP_PATH = INDEXES_DIR / "06_callout_crop_box_map.json"
OUT_PATH = INDEXES_DIR / "07_qty_ocr_map.json"
TMP_DIR = ROOT_DIR / "debug" / "qty_ocr_tmp"
NODE_BIN = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
NODE_MODULES = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")


def build_qty_ocr_map() -> Dict[str, Any]:
    if not CALLOUT_MAP_PATH.exists():
        raise RuntimeError(f"Missing callout crop box map: {CALLOUT_MAP_PATH}")
    if not NODE_BIN.exists() or not NODE_MODULES.exists():
        raise RuntimeError("Missing bundled Node runtime/dependencies for OCR preprocessing")

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                str(NODE_BIN),
                str(ROOT_DIR / "qty_ocr_scan.mjs"),
                "--callout-map",
                str(CALLOUT_MAP_PATH),
                "--repo-root",
                str(ROOT_DIR),
                "--out",
                str(OUT_PATH),
                "--tmp-dir",
                str(TMP_DIR),
                "--node-modules",
                str(NODE_MODULES),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    return json.loads(OUT_PATH.read_text(encoding="utf-8"))


def main() -> None:
    payload = build_qty_ocr_map()
    print(
        json.dumps(
            {
                "ok": True,
                "entry_count": payload.get("entry_count"),
                "out": str(OUT_PATH.relative_to(ROOT_DIR)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
