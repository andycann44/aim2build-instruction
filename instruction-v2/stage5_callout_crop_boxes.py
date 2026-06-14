import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from paths import INDEXES_DIR, ROOT_DIR


PAGE_INDEX_PATH = INDEXES_DIR / "02_page_index.json"
STEP_MAP_PATH = (
    INDEXES_DIR / "05_step_box_map.json"
    if (INDEXES_DIR / "05_step_box_map.json").exists()
    else INDEXES_DIR / "05_step_map.json"
)
V1_CROP_CACHE_PATH = INDEXES_DIR / "05c_v1_crop_cache_import.json"
OUT_PATH = INDEXES_DIR / "06_callout_crop_box_map.json"
DEBUG_DIR = ROOT_DIR / "debug" / "callout_crop_boxes"
NODE_BIN = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
NODE_MODULES = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")


def build_callout_crop_box_map(*, bag_only: Optional[int] = None) -> Dict[str, Any]:
    if not PAGE_INDEX_PATH.exists():
        raise RuntimeError(f"Missing page index: {PAGE_INDEX_PATH}")
    if not STEP_MAP_PATH.exists():
        raise RuntimeError(f"Missing step map: {STEP_MAP_PATH}")
    if not NODE_BIN.exists() or not NODE_MODULES.exists():
        raise RuntimeError("Missing bundled Node runtime/dependencies for page image analysis")

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(NODE_BIN),
        str(ROOT_DIR / "callout_crop_box_scan.mjs"),
        "--page-index",
        str(PAGE_INDEX_PATH),
        "--step-map",
        str(STEP_MAP_PATH),
        "--repo-root",
        str(ROOT_DIR),
        "--out",
        str(OUT_PATH),
        "--debug-dir",
        str(DEBUG_DIR),
        "--node-modules",
        str(NODE_MODULES),
        "--v1-crop-cache",
        str(V1_CROP_CACHE_PATH),
    ]
    if bag_only is not None:
        cmd.extend(["--bag-only", str(int(bag_only))])

    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    return json.loads(OUT_PATH.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 5 callout crop box map.")
    parser.add_argument("--bag-only", type=int, default=None, help="Regenerate one bag and merge into existing manifest.")
    args = parser.parse_args()
    payload = build_callout_crop_box_map(bag_only=args.bag_only)
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
