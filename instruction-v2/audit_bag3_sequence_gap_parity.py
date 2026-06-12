"""Read-only parity audit for remaining Bag 3 sequence gaps."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paths import INDEXES_DIR, ROOT_DIR


PROJECT_ROOT = ROOT_DIR.parent
PAGES_DIR = PROJECT_ROOT / "debug" / "70618" / "70618_01" / "pages"
V1_TRAINING_LABELS_PATH = PROJECT_ROOT / "debug" / "training_labels" / "70618_bag3.json"
V2_CROP_CACHE_PATH = PROJECT_ROOT / "debug" / "crop_cache" / "70618_bag3.json"
STEP_MAP_PATH = INDEXES_DIR / "05_step_map.json"
CALLOUT_MAP_PATH = INDEXES_DIR / "06_callout_crop_box_map.json"
SEQ_GAP_INDEX_PATH = ROOT_DIR / "debug" / "bag3_page_review" / "sequence_gap_review" / "index.json"
RECOVERY_MJS = ROOT_DIR / "bag4_callout_recovery.mjs"
NODE_BIN = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
NODE_MODULES = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")

BAG = 3
SET_NUM = "70618"
GAP_STEPS = [46, 48, 63, 70, 71, 74, 75, 77]

PARITY_MATCH = "PARITY_MATCH"
V2_FAILURE = "V2_FAILURE"
V1_TRUTH_INCOMPLETE = "V1_TRUTH_INCOMPLETE"
EMPTY_STEP = "EMPTY_STEP"
FALSE_STEP = "FALSE_STEP"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _box_xywh(box: Any) -> List[int]:
    if isinstance(box, dict):
        return [int(box.get("x", 0) or 0), int(box.get("y", 0) or 0), int(box.get("w", 0) or 0), int(box.get("h", 0) or 0)]
    if isinstance(box, list) and len(box) >= 4:
        return [int(box[i] or 0) for i in range(4)]
    return []


def _summarize_v1_detection(detected: Dict[str, Any], step: int) -> Dict[str, Any]:
    step = int(step)
    hits: List[Dict[str, Any]] = []

    def consider(entry: Any, source: str) -> None:
        if not isinstance(entry, dict):
            return
        value = int(entry.get("step_number", entry.get("value", 0)) or 0)
        if value != step:
            return
        box = _box_xywh(entry.get("box") or [entry.get("x"), entry.get("y"), entry.get("w"), entry.get("h")])
        hits.append(
            {
                "value": value,
                "box": box,
                "source": source,
                "confidence": entry.get("confidence", entry.get("score")),
                "raw": entry,
            }
        )

    for entry in detected.get("classified_step_boxes", []) or []:
        consider(entry, "classified_step_boxes")
    for entry in detected.get("step_candidates", []) or []:
        consider(entry, "step_candidates")
    main_steps = detected.get("main_steps")
    if isinstance(main_steps, list):
        for entry in main_steps:
            consider(entry, "main_steps")
    elif isinstance(main_steps, dict):
        consider(main_steps, "main_steps")
    if isinstance(detected.get("main_step"), dict):
        consider(detected["main_step"], "main_step")

    main_value = None
    main = detected.get("main_step")
    if isinstance(main, dict):
        main_value = int(main.get("value", 0) or 0)

    return {
        "page": int(detected.get("page") or 0),
        "detected": bool(hits),
        "matches": hits,
        "main_step_value": main_value,
        "summary": (
            f"detected={bool(hits)} main_step={main_value} "
            f"hits={[item['value'] for item in hits]}"
        ),
    }


def _v2_step_entry(step: int, step_map: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    matches = [
        entry
        for entry in step_map.get("steps", []) or []
        if int(entry.get("bag") or 0) == BAG and int(entry.get("step_number") or 0) == int(step)
    ]
    if not matches:
        return None
    entry = matches[0]
    signals = entry.get("signals") or {}
    return {
        "page": int(entry.get("page") or 0),
        "step_number": int(entry.get("step_number") or 0),
        "step_box": _box_xywh(entry.get("step_box")),
        "source": entry.get("source") or "step_map_primary",
        "components": signals.get("components"),
        "raw_text": signals.get("step_number_raw_text"),
        "audit_only": str(entry.get("source") or "") == "sequence_gap_full_page_audit",
        "raw": entry,
    }


def _v1_training_label(step: int, training: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    crops = training.get("crops", {}) if isinstance(training, dict) else {}
    for crop_id, record in crops.items():
        if int(record.get("step") or 0) != int(step):
            continue
        return {
            "crop_id": str(crop_id),
            "page": int(record.get("page") or 0),
            "step": int(record.get("step") or 0),
            "crop_box": _box_xywh(record.get("crop_box")),
            "crop_image_path": record.get("crop_image_path"),
        }
    return None


def _v2_crop_cache_entry(step: int, crop_cache: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for crop in crop_cache:
        if int(crop.get("step") or 0) == int(step):
            return {
                "crop_id": crop.get("crop_id"),
                "page": int(crop.get("page") or 0),
                "step": int(crop.get("step") or 0),
                "crop_box": _box_xywh(crop.get("crop_box")),
            }
    return None


def _v2_callout_entry(step: int, callout_map: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for entry in callout_map.get("entries", []) or []:
        if int(entry.get("bag") or 0) != BAG:
            continue
        value = int(entry.get("step_number", entry.get("step", 0)) or 0)
        if value != int(step):
            continue
        return {
            "page": int(entry.get("page") or 0),
            "step": value,
            "callout_crop_box": _box_xywh(entry.get("callout_crop_box") or entry.get("detected_callout_crop_box")),
            "confidence": entry.get("confidence"),
            "source_function": entry.get("source_function"),
        }
    return None


def _run_v1_callout_probe(steps: List[int], out_dir: Path) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    if not NODE_BIN.exists() or not RECOVERY_MJS.is_file():
        return results

    page_index_path = out_dir / "_page_index_recovery.json"
    page_index = _load_json(INDEXES_DIR / "02_page_index.json")
    pages = []
    for entry in page_index.get("pages", []) or []:
        page = int(entry.get("page") or 0)
        if not page:
            continue
        pages.append({**entry, "image_path": f"debug/70618/70618_01/pages/page_{page:03d}.png"})
    _write_json(page_index_path, {"pages": pages})

    probe_dir = out_dir / "v1_callout_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
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
            str(probe_dir),
            "--node-modules",
            str(NODE_MODULES),
            "--steps",
            ",".join(str(step) for step in steps),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    manifest = _load_json(probe_dir / "index.json") if (probe_dir / "index.json").exists() else {"results": []}
    for row in manifest.get("results", []) or []:
        step = int(row.get("step") or 0)
        if step:
            results[step] = dict(row)
    return results


def _classify(row: Dict[str, Any]) -> Tuple[str, str]:
    v1_label = row.get("v1_training_label")
    v2_cache = row.get("crop_cache_entry")
    v1_step = row.get("v1_step_detection") or {}
    v2_step = row.get("v2_step_detection") or {}
    v2_callout = row.get("v2_callout_entry")
    v1_callout = row.get("v1_callout_probe")

    v1_detected = bool(v1_step.get("detected"))
    v2_detected = bool(v2_step)
    visible_step = v1_detected or v2_detected or bool(row.get("sequence_gap_ocr_anchor"))
    callout_visible = bool(v2_callout) or bool(v1_callout and v1_callout.get("crop_box"))

    if v1_label and v2_cache:
        return PARITY_MATCH, "V1 training label and V2 crop_cache both present."

    if v1_label and not v2_cache:
        return V2_FAILURE, "V1 training label exists but step is missing from V2 crop_cache export."

    if v2_detected and v2_callout and not v2_cache:
        return V2_FAILURE, "V2 detected step anchor and callout box but crop_cache export has no entry."

    if not visible_step:
        return FALSE_STEP, "No credible step-number anchor on gap page from V1 or V2."

    if int(row.get("step") or 0) == 70 and int(row.get("gap_review_page") or 0) != int(v2_step.get("page") or -1):
        return V2_FAILURE, (
            "V2 step_map places step 70 on page "
            f"{v2_step.get('page')} but sequence-gap review inferred page {row.get('gap_review_page')}; "
            "gap page has no matching anchor."
        )

    if v2_detected and not v1_detected and v2_step.get("audit_only"):
        if not callout_visible:
            return V2_FAILURE, (
                "V2 sequence-gap full-page audit found step anchor but V1 raw detector missed it; "
                "no V2 callout box and no V1 callout probe result."
            )
        return V2_FAILURE, "V2 audit recovered step anchor that V1 raw detector missed; crop_cache still empty."

    if visible_step and not callout_visible:
        if v1_detected and v2_detected:
            return EMPTY_STEP, "Both V1 and V2 detect the step number but neither pipeline found a callout box."
        if v2_detected:
            return EMPTY_STEP, "V2 detects step anchor; V1/V2 callout detection found no callout box."
        return EMPTY_STEP, "Sequence-gap page shows no callout box; step anchor evidence is weak or absent."

    return V1_TRUTH_INCOMPLETE, (
        "Step never appears in V1 training labels or V1 crop_cache; "
        "V2 crop_cache export omitted it, leaving a sequence hole."
    )


def _gap_context(seq_item: Dict[str, Any]) -> Dict[str, Any]:
    prev_known = seq_item.get("previous_known_step") or {}
    next_known = seq_item.get("next_known_step") or {}
    return {
        "gap_review_page": int(seq_item.get("page") or 0),
        "previous_step": int(prev_known.get("step") or 0) if prev_known else None,
        "previous_page": int(prev_known.get("page") or 0) if prev_known else None,
        "previous_crop_id": prev_known.get("crop_id"),
        "next_step": int(next_known.get("step") or 0) if next_known else None,
        "next_page": int(next_known.get("page") or 0) if next_known else None,
        "next_crop_id": next_known.get("crop_id"),
        "inference_note": seq_item.get("inference_note"),
        "sequence_gap_ocr_anchor": _box_xywh(seq_item.get("ocr_step_anchor") or seq_item.get("step_box_xywh")),
    }


def audit_sequence_gap_parity(
    *,
    gap_steps: Optional[List[int]] = None,
    run_v1_callout: bool = True,
) -> Dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from clean.services import step_detector_service

    steps = list(gap_steps or GAP_STEPS)
    step_map = _load_json(STEP_MAP_PATH)
    callout_map = _load_json(CALLOUT_MAP_PATH)
    training = _load_json(V1_TRAINING_LABELS_PATH) if V1_TRAINING_LABELS_PATH.exists() else {}
    crop_cache = _load_json(V2_CROP_CACHE_PATH) if V2_CROP_CACHE_PATH.exists() else []
    seq_index = _load_json(SEQ_GAP_INDEX_PATH)
    seq_by_step = {int(item["step"]): item for item in seq_index.get("items", []) or []}

    v1_callout_probe: Dict[int, Dict[str, Any]] = {}
    if run_v1_callout:
        v1_callout_probe = _run_v1_callout_probe(steps, ROOT_DIR / "debug" / "bag3_sequence_gap_parity")

    rows: List[Dict[str, Any]] = []
    for step in steps:
        seq_item = seq_by_step.get(int(step), {})
        ctx = _gap_context(seq_item)
        v2_step = _v2_step_entry(step, step_map)
        v1_label = _v1_training_label(step, training)
        v2_cache = _v2_crop_cache_entry(step, crop_cache if isinstance(crop_cache, list) else [])
        v2_callout = _v2_callout_entry(step, callout_map)
        v1_callout = v1_callout_probe.get(int(step))

        pages_to_probe = sorted(
            {
                p
                for p in [
                    ctx.get("gap_review_page"),
                    v2_step.get("page") if v2_step else None,
                    ctx.get("previous_page"),
                    ctx.get("next_page"),
                ]
                if p
            }
        )
        v1_detections: Dict[int, Dict[str, Any]] = {}
        for page in pages_to_probe:
            detected = step_detector_service.detect_steps(SET_NUM, int(page))
            v1_detections[int(page)] = _summarize_v1_detection(detected, step)

        primary_v1_page = int(v2_step.get("page") or ctx.get("gap_review_page") or 0)
        v1_step_detection = v1_detections.get(primary_v1_page) or next(iter(v1_detections.values()), {})

        row = {
            "step": int(step),
            **ctx,
            "v2_step_page": v2_step.get("page") if v2_step else None,
            "full_page_image": str(PAGES_DIR / f"page_{int(ctx.get('gap_review_page') or 0):03d}.png"),
            "sequence_gap_review_assets": str(
                ROOT_DIR / "debug" / "bag3_page_review" / "sequence_gap_review" / f"step_{int(step):03d}"
            ),
            "v1_training_label": v1_label,
            "crop_cache_entry": v2_cache,
            "v2_step_detection": v2_step,
            "v1_step_detection": v1_step_detection,
            "v1_step_detection_by_page": v1_detections,
            "v2_callout_entry": v2_callout,
            "v1_callout_probe": v1_callout,
            "visible_step_number_on_page": bool(v1_step_detection.get("detected") or v2_step or ctx.get("sequence_gap_ocr_anchor")),
            "visible_callout_box": bool(v2_callout or (v1_callout and v1_callout.get("crop_box"))),
            "crop_cache_entry_yes_no": bool(v2_cache),
            "v1_crop_present": bool(v1_label),
        }
        classification, reason = _classify(row)
        row["classification"] = classification
        row["classification_reason"] = reason
        row["gap_root_cause"] = _root_cause_notes(row)
        rows.append(row)

    summary = Counter_rows(rows)
    return {
        "name": "bag3_sequence_gap_parity_audit",
        "set_num": SET_NUM,
        "bag": BAG,
        "gap_steps": steps,
        "row_count": len(rows),
        "classification_counts": summary,
        "rows": rows,
        "rules": {
            "read_only": True,
            "does_not_modify_crop_cache": True,
            "does_not_modify_detector": True,
            "does_not_promote": True,
        },
    }


def Counter_rows(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("classification") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _root_cause_notes(row: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    if not row.get("v1_crop_present"):
        notes.append("missing_from_v1_training_labels")
    if not row.get("crop_cache_entry_yes_no"):
        notes.append("missing_from_v2_crop_cache_export")
    v2_step = row.get("v2_step_detection") or {}
    if v2_step.get("audit_only"):
        notes.append("v2_step_from_sequence_gap_full_page_audit")
    if row.get("v2_step_page") and row.get("gap_review_page") and int(row["v2_step_page"]) != int(row["gap_review_page"]):
        notes.append(f"v2_step_page_{row['v2_step_page']}_vs_gap_page_{row['gap_review_page']}")
    if row.get("visible_step_number_on_page") and not row.get("visible_callout_box"):
        notes.append("step_anchor_without_callout_box")
    if not row.get("visible_step_number_on_page"):
        notes.append("no_step_anchor_on_gap_review_page")
    return notes


def _export_assets(payload: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in payload.get("rows", []) or []:
        step = int(row["step"])
        step_dir = out_dir / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        gap_page = int(row.get("gap_review_page") or 0)
        src_page = PAGES_DIR / f"page_{gap_page:03d}.png"
        if src_page.is_file():
            shutil.copy2(src_page, step_dir / "page_full.png")

        review_assets = Path(str(row.get("sequence_gap_review_assets") or ""))
        review_page = review_assets / "page_full.png"
        if review_page.is_file():
            shutil.copy2(review_page, step_dir / "page_full_sequence_gap_review.png")

        _write_json(step_dir / "parity_row.json", row)


def _write_markdown(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Bag 3 sequence gap parity audit",
        "",
        f"Set `{payload['set_num']}`, bag `{payload['bag']}`. Read-only. No promotion.",
        "",
        "## Summary",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
    ]
    for key, count in sorted((payload.get("classification_counts") or {}).items()):
        lines.append(f"| {key} | {count} |")

    lines.extend(["", "## Gap rows", ""])
    for row in payload.get("rows", []) or []:
        lines.extend(
            [
                f"### Step {row['step']} (gap page {row.get('gap_review_page')})",
                "",
                f"- **Classification:** `{row.get('classification')}`",
                f"- **Reason:** {row.get('classification_reason')}",
                f"- **Previous:** step {row.get('previous_step')} page {row.get('previous_page')} (`{row.get('previous_crop_id')}`)",
                f"- **Next:** step {row.get('next_step')} page {row.get('next_page')} (`{row.get('next_crop_id')}`)",
                f"- **V1 training crop:** {'yes' if row.get('v1_crop_present') else 'no'}",
                f"- **V2 crop_cache:** {'yes' if row.get('crop_cache_entry_yes_no') else 'no'}",
                f"- **Visible step number:** {'yes' if row.get('visible_step_number_on_page') else 'no'}",
                f"- **Visible callout box:** {'yes' if row.get('visible_callout_box') else 'no'}",
                f"- **V1 step detection:** {(row.get('v1_step_detection') or {}).get('summary')}",
                f"- **V2 step detection:** page { (row.get('v2_step_detection') or {}).get('page') }, "
                f"box {(row.get('v2_step_detection') or {}).get('step_box')}, "
                f"source {(row.get('v2_step_detection') or {}).get('source')}",
                f"- **Root-cause notes:** {', '.join(row.get('gap_root_cause') or [])}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(payload: Dict[str, Any], path: Path, out_dir: Path) -> None:
    cards = []
    for row in payload.get("rows", []) or []:
        step = int(row["step"])
        rel_page = f"step_{step:03d}/page_full.png"
        cls = str(row.get("classification") or "")
        cards.append(
            f"""
            <section class="card">
              <h2>Step {step} <span class="tag {cls}">{cls}</span></h2>
              <p>{row.get('classification_reason')}</p>
              <div class="grid">
                <div>
                  <h3>Context</h3>
                  <ul>
                    <li>Gap page: {row.get('gap_review_page')}</li>
                    <li>Previous: {row.get('previous_step')} (p{row.get('previous_page')}, {row.get('previous_crop_id')})</li>
                    <li>Next: {row.get('next_step')} (p{row.get('next_page')}, {row.get('next_crop_id')})</li>
                    <li>V1 crop: {'yes' if row.get('v1_crop_present') else 'no'}</li>
                    <li>V2 crop_cache: {'yes' if row.get('crop_cache_entry_yes_no') else 'no'}</li>
                    <li>Visible step: {'yes' if row.get('visible_step_number_on_page') else 'no'}</li>
                    <li>Visible callout: {'yes' if row.get('visible_callout_box') else 'no'}</li>
                  </ul>
                </div>
                <div>
                  <h3>Detections</h3>
                  <pre>{json.dumps({'v1': row.get('v1_step_detection'), 'v2': row.get('v2_step_detection'), 'root_cause': row.get('gap_root_cause')}, indent=2)}</pre>
                </div>
              </div>
              <img src="{rel_page}" alt="page {row.get('gap_review_page')}" />
            </section>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Bag 3 sequence gap parity</title>
<style>
body {{ font: 14px/1.45 system-ui, sans-serif; margin: 24px; color: #111; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 0 0 24px; }}
.tag {{ font-size: 12px; padding: 2px 8px; border-radius: 999px; background: #eee; }}
.V2_FAILURE {{ background: #ffe0e0; }}
.EMPTY_STEP {{ background: #fff4cc; }}
.V1_TRUTH_INCOMPLETE {{ background: #e7f0ff; }}
.FALSE_STEP {{ background: #f3e5ff; }}
.PARITY_MATCH {{ background: #dff5e5; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
img {{ max-width: 100%; border: 1px solid #ccc; margin-top: 12px; }}
pre {{ background: #f7f7f7; padding: 10px; overflow: auto; }}
</style></head><body>
<h1>Bag 3 sequence gap parity audit</h1>
<p>Read-only audit for steps {payload.get('gap_steps')}. No promotion.</p>
{''.join(cards)}
</body></html>"""
    path.write_text(html, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Bag 3 remaining sequence gaps for V1/V2 parity.")
    parser.add_argument("--skip-v1-callout-probe", action="store_true")
    args = parser.parse_args(argv)

    out_dir = ROOT_DIR / "debug" / "bag3_sequence_gap_parity"
    payload = audit_sequence_gap_parity(run_v1_callout=not args.skip_v1_callout_probe)
    _export_assets(payload, out_dir)
    _write_json(out_dir / "70618_bag3_sequence_gap_parity.json", payload)
    _write_markdown(payload, ROOT_DIR / "BAG3_SEQUENCE_GAP_PARITY_AUDIT.md")
    _write_html(payload, out_dir / "report.html", out_dir)

    print("step\tgap_page\tclassification\tvisible_step\tvisible_callout\tv1_crop\tv2_cache")
    for row in payload.get("rows", []) or []:
        print(
            f"{row['step']}\t{row.get('gap_review_page')}\t{row.get('classification')}\t"
            f"{row.get('visible_step_number_on_page')}\t{row.get('visible_callout_box')}\t"
            f"{row.get('v1_crop_present')}\t{row.get('crop_cache_entry_yes_no')}"
        )
    print(str(out_dir / "70618_bag3_sequence_gap_parity.json"))
    print(str(out_dir / "report.html"))
    print(str(ROOT_DIR / "BAG3_SEQUENCE_GAP_PARITY_AUDIT.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
