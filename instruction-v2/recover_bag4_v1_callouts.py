"""Run V1 callout border detection for Bag 4 recovery steps.

Uses the same detectCalloutRectByEdges logic as callout_crop_box_scan.mjs
(stage5 / V1 callout_crop_lab flow). Does not write to crop_cache.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from paths import INDEXES_DIR, ROOT_DIR


PROJECT_ROOT = ROOT_DIR.parent
PAGES_DIR = PROJECT_ROOT / "debug" / "70618" / "70618_01" / "pages"
PAGE_INDEX_PATH = INDEXES_DIR / "02_page_index.json"
STEP_MAP_PATH = INDEXES_DIR / "05_step_map.json"
OUT_DIR = ROOT_DIR / "debug" / "bag4_v1_callout_recovery"
RECOVERY_MJS = ROOT_DIR / "bag4_callout_recovery.mjs"
NODE_BIN = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
NODE_MODULES = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")

RECOVERY_STEPS = [110, 112, 113, 115, 116, 117, 119, 126, 131]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _page_index_for_recovery() -> Path:
    payload = _load_json(PAGE_INDEX_PATH)
    pages = []
    for entry in payload.get("pages", []) or []:
        page = int(entry.get("page") or 0)
        if not page:
            continue
        pages.append(
            {
                **entry,
                "image_path": f"debug/70618/70618_01/pages/page_{page:03d}.png",
            }
        )
    out_path = OUT_DIR / "_page_index_recovery.json"
    _write_json(out_path, {"pages": pages})
    return out_path


def _build_contact_sheet(manifest: Dict[str, Any]) -> Path:
    rows = []
    for row in manifest.get("results", []) or []:
        step = int(row.get("step") or 0)
        side_by_side = OUT_DIR / f"step_{step:03d}" / "side_by_side.png"
        if side_by_side.is_file():
            rows.append((step, side_by_side))
    if not rows:
        return OUT_DIR / "contact_sheet.png"

    label_h = 36
    panels = []
    max_w = 0
    max_h = 0
    for step, path in rows:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        header = np.full((label_h, w, 3), 255, dtype=np.uint8)
        cv2.putText(
            header,
            f"step {step}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        panel = np.vstack([header, img])
        panels.append(panel)
        max_w = max(max_w, panel.shape[1])
        max_h = max(max_h, panel.shape[0])

    sheet_h = sum(panel.shape[0] for panel in panels) + max(0, len(panels) - 1) * 8
    sheet = np.full((sheet_h, max_w, 3), 245, dtype=np.uint8)
    y = 0
    for panel in panels:
        sheet[y : y + panel.shape[0], : panel.shape[1]] = panel
        y += panel.shape[0] + 8

    out_path = OUT_DIR / "contact_sheet.png"
    cv2.imwrite(str(out_path), sheet)
    return out_path


def run_v1_callout_recovery(steps: List[int] | None = None) -> Dict[str, Any]:
    if not NODE_BIN.exists() or not NODE_MODULES.exists():
        raise RuntimeError("Missing bundled Node runtime for V1 callout detection")
    if not RECOVERY_MJS.is_file():
        raise RuntimeError(f"Missing recovery script: {RECOVERY_MJS}")

    step_list = list(steps or RECOVERY_STEPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page_index_path = _page_index_for_recovery()

    subprocess.run(
        [
            str(NODE_BIN),
            str(RECOVERY_MJS),
            "--repo-root",
            str(PROJECT_ROOT),
            "--page-index",
            str(page_index_path),
            "--step-map",
            str(STEP_MAP_PATH),
            "--out-dir",
            str(OUT_DIR),
            "--node-modules",
            str(NODE_MODULES),
            "--steps",
            ",".join(str(step) for step in step_list),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = _load_json(OUT_DIR / "index.json")
    contact_sheet = _build_contact_sheet(manifest)
    manifest["contact_sheet_png"] = str(contact_sheet)
    _write_json(OUT_DIR / "index.json", manifest)
    return manifest


def _print_table(manifest: Dict[str, Any]) -> None:
    print("step\tpage\tcandidate_crop_id\tcrop_box\tconfidence\tneeds_human_adjustment")
    for row in manifest.get("results", []) or []:
        box = row.get("crop_box")
        conf = row.get("confidence")
        failed = bool(row.get("detection_failed"))
        needs = failed or conf is None or float(conf or 0) < 0.55
        print(
            "\t".join(
                [
                    str(row.get("step") or ""),
                    str(row.get("page") or ""),
                    str(row.get("candidate_crop_id") or ""),
                    json.dumps(box),
                    str(conf if conf is not None else ""),
                    str(needs),
                ]
            )
        )


def main() -> int:
    manifest = run_v1_callout_recovery()
    _print_table(manifest)
    print(str(OUT_DIR / "index.json"))
    print(f"contact_sheet={manifest.get('contact_sheet_png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
