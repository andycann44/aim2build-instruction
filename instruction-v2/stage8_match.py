import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from paths import INDEXES_DIR, ROOT_DIR


SET_CONTEXT_PATH = INDEXES_DIR / "00_set_context.json"
SEGMENTATION_PATH = INDEXES_DIR / "08_part_segmentation_map.json"
OUT_PATH = INDEXES_DIR / "09_match_manifest.json"
NODE_BIN = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
NODE_MODULES = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")


def run_validator() -> None:
    subprocess.run(
        [sys.executable, str(ROOT_DIR / "validate_pipeline.py")],
        check=True,
        cwd=str(ROOT_DIR.parent),
    )


def build_match_manifest() -> Dict[str, Any]:
    run_validator()

    if not SET_CONTEXT_PATH.exists():
        raise RuntimeError(f"Missing set context: {SET_CONTEXT_PATH}")
    if not SEGMENTATION_PATH.exists():
        raise RuntimeError(f"Missing part segmentation map: {SEGMENTATION_PATH}")
    if not NODE_BIN.exists() or not NODE_MODULES.exists():
        raise RuntimeError("Missing bundled Node runtime/dependencies for matching")

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(NODE_BIN),
            str(ROOT_DIR / "match_segments_scan.mjs"),
            "--set-context",
            str(SET_CONTEXT_PATH),
            "--segmentation-map",
            str(SEGMENTATION_PATH),
            "--repo-root",
            str(ROOT_DIR),
            "--out",
            str(OUT_PATH),
            "--node-modules",
            str(NODE_MODULES),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return json.loads(OUT_PATH.read_text(encoding="utf-8"))


def main() -> None:
    payload = build_match_manifest()
    print(
        json.dumps(
            {
                "ok": True,
                "entry_count": payload.get("entry_count"),
                "candidate_pool_count": payload.get("candidate_pool_count"),
                "out": str(OUT_PATH.relative_to(ROOT_DIR)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
