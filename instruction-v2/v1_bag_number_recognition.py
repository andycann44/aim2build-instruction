#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from paths import INDEXES_DIR
from v1_page_loading import _find_latest_pages_dir_for_set


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
OUT_PATH = INDEXES_DIR / "03c_v1_bag_number_recognition.json"

SOURCE_FUNCTIONS = [
    "clean/services/page_analyzer.py::configure_pages_dir",
    "clean/services/page_analyzer.py::analyze_page",
    "clean/services/analyzer_scan_service.py::_build_sequence_scan_row",
    "clean/services/gap_scan_service.py::_normalize_analysis_row",
    "clean/services/gap_scan_service.py::_score_window_candidate",
    "clean/services/gap_scan_service.py::_build_candidate",
]


def _ensure_project_on_path() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _load_v1_modules():
    _ensure_project_on_path()
    from clean.services import analyzer_scan_service, gap_scan_service, page_analyzer, precheck_service

    return page_analyzer, analyzer_scan_service, gap_scan_service, precheck_service


def _parse_page_spec(spec: str) -> Tuple[int, Optional[int]]:
    if ":" not in spec:
        return int(spec), None
    page_text, expected_text = spec.split(":", 1)
    return int(page_text), int(expected_text) if expected_text else None


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    fields = [
        "page",
        "page_kind",
        "panel_found",
        "panel_source",
        "shell_found",
        "shell_method",
        "grey_bag_found",
        "number_found",
        "number_box_found",
        "bag_number",
        "confidence",
        "overview_page",
        "strong_structure",
        "multi_step_green_boxes",
        "ocr_raw",
        "analysis_available",
        "cache_hit",
    ]
    return {key: row.get(key) for key in fields if key in row}


def analyze_bag_number_with_v1(
    set_num: str,
    page: int,
    *,
    expected_bag_number: Optional[int] = None,
    window: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    V2 wrapper only: use exact current V1 recognizer/scan/scoring functions.
    This function intentionally does not call instruction-v2/bag_number_recognizer.mjs.
    """
    pages_dir = _find_latest_pages_dir_for_set(str(set_num))
    if pages_dir is None:
        raise RuntimeError(f"no V1 rendered pages directory found for set {set_num}")

    page_analyzer, analyzer_scan_service, gap_scan_service, precheck_service = _load_v1_modules()
    page_analyzer.configure_pages_dir(str(pages_dir))

    analysis = page_analyzer.analyze_page(int(page), include_image=False)
    scan_row = analyzer_scan_service._build_sequence_scan_row(int(page))
    precheck = precheck_service.get_page_precheck(str(set_num), int(page))
    normalized = gap_scan_service._normalize_analysis_row(scan_row, int(page), precheck)

    candidate = None
    if expected_bag_number is not None:
        start_page, end_page = window or (int(page), int(page))
        candidate = gap_scan_service._build_candidate(
            str(set_num),
            normalized,
            int(expected_bag_number),
            int(start_page),
            int(end_page),
        )

    return {
        "set_num": str(set_num),
        "page": int(page),
        "source": "exact_v1_page_analyzer_sequence_gap_wrapper",
        "source_functions": SOURCE_FUNCTIONS,
        "pages_dir": str(pages_dir),
        "uses_previous_v2_bag_number_recognizer_mjs": False,
        "expected_bag_number": expected_bag_number,
        "detected_bag_number": normalized.get("bag_number"),
        "confidence": float(normalized.get("confidence", 0.0) or 0.0),
        "number_box_found": bool(normalized.get("number_box_found")),
        "panel_found": bool(normalized.get("panel_found")),
        "panel_source": normalized.get("panel_source"),
        "shell_found": bool(normalized.get("shell_found")),
        "grey_bag_found": bool(normalized.get("grey_bag_found")),
        "strong_structure": bool(normalized.get("strong_structure")),
        "page_kind": normalized.get("page_kind", "other"),
        "ocr_raw": normalized.get("ocr_raw", "") or "",
        "analysis": _compact_row(analysis),
        "scan_row": _compact_row(scan_row),
        "normalized_row": _compact_row(normalized),
        "v1_gap_candidate": candidate,
    }


def build_manifest(
    set_num: str,
    page_specs: Iterable[Tuple[int, Optional[int]]],
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for page, expected_bag_number in page_specs:
        try:
            entries.append(
                analyze_bag_number_with_v1(
                    set_num,
                    page,
                    expected_bag_number=expected_bag_number,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "page": int(page),
                    "expected_bag_number": expected_bag_number,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "stage": "03c",
        "name": "v1_bag_number_recognition",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "set_num": str(set_num),
        "source": "exact_current_v1_wrapped_by_instruction_v2",
        "source_functions": SOURCE_FUNCTIONS,
        "uses_previous_v2_bag_number_recognizer_mjs": False,
        "entry_count": len(entries),
        "error_count": len(errors),
        "entries": entries,
        "errors": errors,
    }


def save_manifest(payload: Dict[str, Any], out_path: Path = OUT_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exact V1 bag-number recognition from V2.")
    parser.add_argument("--set-num", default="70618")
    parser.add_argument(
        "--page",
        action="append",
        required=True,
        help="Page to analyze, optionally page:expected_bag_number. May be repeated.",
    )
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    page_specs = [_parse_page_spec(spec) for spec in args.page]
    payload = build_manifest(args.set_num, page_specs)
    save_manifest(payload, Path(args.out))

    for entry in payload["entries"]:
        candidate = entry.get("v1_gap_candidate") or {}
        print(
            "page={page} expected={expected} detected={detected} "
            "confidence={confidence:.3f} number_box={number_box} score={score}".format(
                page=entry.get("page"),
                expected=entry.get("expected_bag_number"),
                detected=entry.get("detected_bag_number"),
                confidence=float(entry.get("confidence", 0.0) or 0.0),
                number_box=entry.get("number_box_found"),
                score=candidate.get("score"),
            )
        )

    for error in payload["errors"]:
        print(
            "ERROR page={page} expected={expected}: {error}".format(
                page=error.get("page"),
                expected=error.get("expected_bag_number"),
                error=error.get("error"),
            )
        )

    return 1 if payload["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
