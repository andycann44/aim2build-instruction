import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from paths import INDEXES_DIR, ROOT_DIR


PAGE_INDEX_PATH = INDEXES_DIR / "02_page_index.json"
BAG_MAP_PATH = INDEXES_DIR / "04_bag_map.json"
CALLOUT_MAP_PATH = INDEXES_DIR / "06_callout_crop_box_map.json"
V1_CROP_CACHE_PATH = INDEXES_DIR / "05c_v1_crop_cache_import.json"
OUT_PATH = INDEXES_DIR / "06a_sequence_completeness_diagnostics.json"
DEBUG_DIR = ROOT_DIR / "debug" / "sequence_completeness"
STEP_PROBE_PATH = DEBUG_DIR / "05_step_probe.json"
STEP_PROBE_BAG_MAP_PATH = DEBUG_DIR / "04_bag_map_probe.json"
STEP_PROBE_DEBUG_DIR = DEBUG_DIR / "step_probe"
NODE_BIN = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
NODE_MODULES = Path("/Users/andy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_pages(raw: Optional[str]) -> Optional[Set[int]]:
    if not raw:
        return None
    pages: Set[int] = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start_raw, end_raw = piece.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            pages.update(range(min(start, end), max(start, end) + 1))
        else:
            pages.add(int(piece))
    return pages


def _build_probe_bag_map(page_filter: Optional[Set[int]]) -> Path:
    source = _read_json(BAG_MAP_PATH)
    if not page_filter:
        STEP_PROBE_BAG_MAP_PATH.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        return STEP_PROBE_BAG_MAP_PATH

    bags: List[Dict[str, Any]] = []
    for bag in source.get("bags", []):
        bag_num = int(bag.get("bag"))
        start = int(bag.get("start_page"))
        end = int(bag.get("end_page"))
        selected = sorted(page for page in page_filter if start <= page <= end)
        if not selected:
            continue

        runs: List[List[int]] = []
        for page in selected:
            if not runs or page != runs[-1][-1] + 1:
                runs.append([page])
            else:
                runs[-1].append(page)

        for run in runs:
            bags.append(
                {
                    "bag": bag_num,
                    "start_page": run[0],
                    "end_page": run[-1],
                    "confidence": bag.get("confidence", 1),
                    "source": "sequence_completeness_diagnostic_page_filter",
                    "evidence_pages": bag.get("evidence_pages", []),
                }
            )

    payload = {
        "stage": "diagnostic",
        "name": "sequence_completeness_probe_bag_map",
        "method": "filtered_from_indexes_04_bag_map",
        "page_filter": sorted(page_filter),
        "bags": bags,
    }
    STEP_PROBE_BAG_MAP_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return STEP_PROBE_BAG_MAP_PATH


def _run_step_probe(page_filter: Optional[Set[int]]) -> Dict[str, Any]:
    if not PAGE_INDEX_PATH.exists():
        raise RuntimeError(f"Missing page index: {PAGE_INDEX_PATH}")
    if not BAG_MAP_PATH.exists():
        raise RuntimeError(f"Missing bag map: {BAG_MAP_PATH}")
    if not CALLOUT_MAP_PATH.exists():
        raise RuntimeError(f"Missing callout map: {CALLOUT_MAP_PATH}")
    if not NODE_BIN.exists() or not NODE_MODULES.exists():
        raise RuntimeError("Missing bundled Node runtime/dependencies for page image analysis")

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    STEP_PROBE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    probe_bag_map = _build_probe_bag_map(page_filter)

    subprocess.run(
        [
            str(NODE_BIN),
            str(ROOT_DIR / "step_map_scan.mjs"),
            "--page-index",
            str(PAGE_INDEX_PATH),
            "--bag-map",
            str(probe_bag_map),
            "--repo-root",
            str(ROOT_DIR),
            "--out",
            str(STEP_PROBE_PATH),
            "--debug-dir",
            str(STEP_PROBE_DEBUG_DIR),
            "--node-modules",
            str(NODE_MODULES),
            "--v1-crop-cache",
            str(V1_CROP_CACHE_PATH),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return _read_json(STEP_PROBE_PATH)


def _valid_numbers(values: Iterable[Any]) -> List[int]:
    numbers: Set[int] = set()
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            numbers.add(number)
    return sorted(numbers)


def _consecutive_runs(numbers: Sequence[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for number in sorted(set(numbers)):
        if not runs or number != runs[-1][-1] + 1:
            runs.append([number])
        else:
            runs[-1].append(number)
    return runs


def _visible_sequences(visible_numbers: Sequence[int], emitted_steps: Sequence[int]) -> List[List[int]]:
    emitted = set(emitted_steps)
    sequences: List[List[int]] = []
    for run in _consecutive_runs(visible_numbers):
        if len(run) < 3:
            continue
        if emitted and not emitted.intersection(run):
            continue
        sequences.append(run)
    return sequences


def _callouts_by_page(callout_map: Dict[str, Any]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    by_page: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for entry in callout_map.get("entries", []):
        try:
            bag = int(entry.get("bag"))
            page = int(entry.get("page"))
        except (TypeError, ValueError):
            continue
        by_page.setdefault((bag, page), []).append(entry)
    return by_page


def build_sequence_completeness_diagnostics(page_filter: Optional[Set[int]] = None) -> Dict[str, Any]:
    step_probe = _run_step_probe(page_filter)
    callout_map = _read_json(CALLOUT_MAP_PATH)
    callouts = _callouts_by_page(callout_map)

    entries: List[Dict[str, Any]] = []
    missing_report_count = 0
    for page_record in step_probe.get("pages", []):
        try:
            bag = int(page_record.get("bag"))
            page = int(page_record.get("page"))
        except (TypeError, ValueError):
            continue
        if page_filter and page not in page_filter:
            continue

        visible_numbers = _valid_numbers(
            candidate.get("step_number")
            for candidate in page_record.get("full_page_audit_candidates", [])
        )
        emitted_entries = callouts.get((bag, page), [])
        emitted_steps = _valid_numbers(entry.get("step") for entry in emitted_entries)
        sequences = _visible_sequences(visible_numbers, emitted_steps)
        sequence_reports = []
        for sequence in sequences:
            missing = [number for number in sequence if number not in set(emitted_steps)]
            if missing:
                missing_report_count += 1
            sequence_reports.append(
                {
                    "visible_sequence": sequence,
                    "emitted_steps_in_sequence": [number for number in sequence if number in set(emitted_steps)],
                    "missing_steps": missing,
                    "status": "missing_steps_reported" if missing else "complete",
                }
            )

        entries.append(
            {
                "bag": bag,
                "page": page,
                "visible_step_numbers": visible_numbers,
                "emitted_callout_steps": emitted_steps,
                "sequence_reports": sequence_reports,
                "missing_steps": sorted(
                    {
                        step
                        for report in sequence_reports
                        for step in report.get("missing_steps", [])
                    }
                ),
                "diagnostic_only": True,
                "debug_overlay_path": page_record.get("debug_overlay_path"),
                "source": "page_level_step_number_probe_vs_06_callout_crop_box_map",
            }
        )

    payload = {
        "stage": "diagnostic",
        "name": "sequence_completeness_diagnostics",
        "input_manifests": [
            "indexes/04_bag_map.json",
            "indexes/02_page_index.json",
            "indexes/06_callout_crop_box_map.json",
        ],
        "method": "diagnostic_only_visible_step_probe_vs_emitted_callout_steps",
        "rules": [
            "Do not auto-create crops.",
            "Do not promote probe output.",
            "Every number in a monotonic visible sequence must either have an emitted crop or appear in missing_steps.",
        ],
        "page_filter": sorted(page_filter) if page_filter else None,
        "entry_count": len(entries),
        "missing_report_count": missing_report_count,
        "step_probe_path": str(STEP_PROBE_PATH.relative_to(ROOT_DIR)),
        "entries": entries,
    }
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic-only sequence completeness guardrail.")
    parser.add_argument("--pages", help="Optional comma/range page filter, e.g. 78,79 or 78-79")
    args = parser.parse_args()
    payload = build_sequence_completeness_diagnostics(_parse_pages(args.pages))
    print(
        json.dumps(
            {
                "ok": True,
                "entry_count": payload.get("entry_count"),
                "missing_report_count": payload.get("missing_report_count"),
                "out": str(OUT_PATH.relative_to(ROOT_DIR)),
                "step_probe_path": payload.get("step_probe_path"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
