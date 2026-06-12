import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from paths import INDEXES_DIR, ROOT_DIR


CALLOUT_MAP_PATH = INDEXES_DIR / "06_callout_crop_box_map.json"
QTY_MAP_PATH = INDEXES_DIR / "07_qty_ocr_map.json"
OUT_PATH = INDEXES_DIR / "08_part_segmentation_map.json"
DEBUG_DIR = ROOT_DIR / "debug" / "part_segmentation"
NODE_BIN = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
NODE_MODULES = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")


def build_part_segmentation_map() -> Dict[str, Any]:
    if not CALLOUT_MAP_PATH.exists():
        raise RuntimeError(f"Missing callout crop box map: {CALLOUT_MAP_PATH}")
    if not QTY_MAP_PATH.exists():
        raise RuntimeError(f"Missing qty OCR map: {QTY_MAP_PATH}")
    if not NODE_BIN.exists() or not NODE_MODULES.exists():
        raise RuntimeError("Missing bundled Node runtime/dependencies for segmentation")

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            str(NODE_BIN),
            str(ROOT_DIR / "part_segmentation_scan.mjs"),
            "--callout-map",
            str(CALLOUT_MAP_PATH),
            "--qty-map",
            str(QTY_MAP_PATH),
            "--repo-root",
            str(ROOT_DIR),
            "--out",
            str(OUT_PATH),
            "--debug-dir",
            str(DEBUG_DIR),
            "--node-modules",
            str(NODE_MODULES),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return json.loads(OUT_PATH.read_text(encoding="utf-8"))


def main() -> None:
    payload = build_part_segmentation_map()
    print(
        json.dumps(
            {
                "ok": True,
                "entry_count": payload.get("entry_count"),
                "out": str(OUT_PATH.relative_to(ROOT_DIR)),
                "debug_dir": str(DEBUG_DIR.relative_to(ROOT_DIR)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
