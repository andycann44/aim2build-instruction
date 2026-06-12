import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from paths import INDEXES_DIR, ROOT_DIR


PAGE_INDEX_PATH = INDEXES_DIR / "02_page_index.json"
OUT_PATH = INDEXES_DIR / "03_bag_candidates.json"
DEBUG_DIR = ROOT_DIR / "debug" / "bag_candidates"
NODE_BIN = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
NODE_MODULES = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")


def build_bag_candidates() -> Dict[str, Any]:
    if not PAGE_INDEX_PATH.exists():
        raise RuntimeError(f"Missing page index: {PAGE_INDEX_PATH}")
    if not NODE_BIN.exists() or not NODE_MODULES.exists():
        raise RuntimeError("Missing bundled Node runtime/dependencies for page image analysis")

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            str(NODE_BIN),
            str(ROOT_DIR / "bag_candidate_scan.mjs"),
            "--page-index",
            str(PAGE_INDEX_PATH),
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
    payload = build_bag_candidates()
    print(
        json.dumps(
            {
                "ok": True,
                "candidate_count": payload.get("candidate_count"),
                "out": str(OUT_PATH.relative_to(ROOT_DIR)),
                "debug_dir": str(DEBUG_DIR.relative_to(ROOT_DIR)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
