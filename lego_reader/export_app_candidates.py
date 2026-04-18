from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from .utils import ensure_dir, write_json


DEFAULT_DEBUG_ROOT = "debug"
DEFAULT_LABELS_ROOT = "manual_labels"


def _parse_bool(raw_value: str) -> bool:
    value = raw_value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {raw_value!r}")


def _candidate_stem(page_number: int) -> str:
    return f"candidate_page_{page_number:03d}"


def _load_manual_labels(labels_path: Path) -> dict[str, Any]:
    if not labels_path.is_file():
        raise SystemExit(f"Manual labels file not found: {labels_path}")

    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid manual labels format: {labels_path}")

    raw_labels = payload.get("labels", {})
    if not isinstance(raw_labels, dict):
        raw_labels = {}

    raw_unsure = payload.get("unsure_pages", [])
    if not isinstance(raw_unsure, list):
        raw_unsure = []

    return {
        "set_num": str(payload.get("set_num", "")).strip(),
        "pdf_name": str(payload.get("pdf_name", "")).strip(),
        "labels": {
            int(str(page)): str(label).strip()
            for page, label in raw_labels.items()
            if str(page).isdigit() and isinstance(label, str) and label.strip()
        },
        "unsure_pages": {
            int(str(page))
            for page in raw_unsure
            if str(page).isdigit()
        },
    }


def _load_app_module(repo_root: Path):
    repo_root_str = repo_root.as_posix()
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return importlib.import_module("app")


def _configure_app_module(app_module, pages_dir: Path, reference_page: int) -> None:
    app_module.BASE = pages_dir.as_posix()
    app_module.REFERENCE_PAGE = reference_page
    app_module.SHELL_TEMPLATE = app_module.make_shell_template()


def _lookup_by_page(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        page = row.get("page")
        if isinstance(page, int):
            output[page] = row
    return output


def _sequence_lookups(sequence_data: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    accepted_by_page: dict[int, dict[str, Any]] = {}
    for item in sequence_data.get("accepted_sequence", []):
        page = item.get("page")
        if isinstance(page, int):
            accepted_by_page[page] = item

    deferred_by_page: dict[int, dict[str, Any]] = {}
    for item in sequence_data.get("deferred_candidates", []):
        page = item.get("page")
        if isinstance(page, int) and page not in deferred_by_page:
            deferred_by_page[page] = item

    page_debug_by_page: dict[int, dict[str, Any]] = {}
    for item in sequence_data.get("page_debug_log", []):
        page = item.get("page")
        if isinstance(page, int):
            page_debug_by_page[page] = item

    return accepted_by_page, deferred_by_page, page_debug_by_page


def _strong_detector_signal(row: dict[str, Any], min_confidence: float) -> bool:
    return bool(
        float(row.get("confidence", 0.0) or 0.0) >= min_confidence
        and row.get("panel_found")
        and row.get("shell_found")
        and row.get("grey_bag_found")
    )


def _disagreement_types(
    *,
    manual_label: str,
    row: dict[str, Any],
    accepted_item: dict[str, Any] | None,
    min_confidence: float,
) -> list[str]:
    disagreement_types: list[str] = []

    bag_number = row.get("bag_number")
    confidence = float(row.get("confidence", 0.0) or 0.0)

    if manual_label == "true_bag_start":
        if accepted_item is None:
            disagreement_types.append("manual_true_bag_start_not_accepted")
        if bag_number is None:
            disagreement_types.append("manual_true_bag_start_no_number_detected")
        if confidence < min_confidence:
            disagreement_types.append("manual_true_bag_start_low_confidence")
    elif manual_label in {"normal_step", "sticker_or_callout"}:
        if accepted_item is not None:
            disagreement_types.append("manual_non_bag_accepted_by_sequence")
        if bag_number is not None:
            disagreement_types.append("manual_non_bag_bag_number_detected")
        if _strong_detector_signal(row, min_confidence):
            disagreement_types.append("manual_non_bag_strong_detector_signal")

    return sorted(set(disagreement_types))


def _write_annotated_image(app_module, page_number: int, destination_path: Path) -> None:
    annotated_result = app_module.analyze_page(page_number, include_image=True)
    destination_path.write_bytes(annotated_result["image_bytes"])


def _write_number_crop(app_module, page_image_path: Path, number_box: list[int] | None, destination_path: Path) -> bool:
    if not number_box:
        return False

    image = cv2.imread(page_image_path.as_posix())
    if image is None:
        return False

    crop = app_module.crop(image, tuple(number_box))
    if crop is None or crop.size == 0:
        return False

    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return False

    destination_path.write_bytes(buf.tobytes())
    return True


def export_app_candidates(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    debug_root = Path(args.debug_root).expanduser().resolve()
    labels_root = Path(args.labels_root).expanduser().resolve()

    debug_pdf_dir = debug_root / args.set_num / args.pdf_name
    pages_dir = debug_pdf_dir / "pages"
    if not pages_dir.is_dir():
        raise SystemExit(f"Page images directory not found: {pages_dir}")

    labels_path = labels_root / f"{args.set_num}_{args.pdf_name}.json"
    labels_payload = _load_manual_labels(labels_path)

    app_module = _load_app_module(repo_root)
    _configure_app_module(app_module, pages_dir, args.reference_page)

    sequence_data = app_module.run_sequential_scan(
        start_number=args.start_number,
        end_number=args.end_number,
        min_confidence=args.min_confidence,
        allow_structure_only_start=args.allow_structure_only_start,
    )

    rows_by_page = _lookup_by_page(sequence_data.get("rows", []))
    accepted_by_page, deferred_by_page, page_debug_by_page = _sequence_lookups(sequence_data)

    app_candidates_dir = ensure_dir(debug_pdf_dir / "app_candidates")

    exported_count = 0
    skipped_unsure = 0
    skipped_missing_page = 0
    disagreement_counter: Counter[str] = Counter()

    for page_number, manual_label in sorted(labels_payload["labels"].items()):
        if page_number in labels_payload["unsure_pages"]:
            skipped_unsure += 1
            continue

        row = rows_by_page.get(page_number)
        if row is None:
            skipped_missing_page += 1
            continue

        accepted_item = accepted_by_page.get(page_number)
        disagreement_types = _disagreement_types(
            manual_label=manual_label,
            row=row,
            accepted_item=accepted_item,
            min_confidence=args.min_confidence,
        )
        if not disagreement_types:
            continue

        page_debug = page_debug_by_page.get(page_number, {})
        deferred_item = deferred_by_page.get(page_number)
        page_image_path = Path(app_module.page_path(page_number))
        if not page_image_path.is_file():
            skipped_missing_page += 1
            continue

        stem = _candidate_stem(page_number)
        full_page_path = app_candidates_dir / f"{stem}_full.png"
        annotated_path = app_candidates_dir / f"{stem}_annotated.png"
        number_crop_path = app_candidates_dir / f"{stem}_number.png"
        sidecar_path = app_candidates_dir / f"{stem}.json"

        shutil.copy2(page_image_path, full_page_path)
        _write_annotated_image(app_module, page_number, annotated_path)

        number_crop_saved = _write_number_crop(
            app_module,
            page_image_path,
            row.get("number_box"),
            number_crop_path,
        )
        if not number_crop_saved and number_crop_path.exists():
            number_crop_path.unlink()

        payload = {
            "export_version": 1,
            "set_num": args.set_num,
            "pdf_name": args.pdf_name,
            "page_number": page_number,
            "disagreement_types": disagreement_types,
            "source_paths": {
                "page_image": page_image_path.as_posix(),
            },
            "saved_paths": {
                "full_page_image": full_page_path.as_posix(),
                "annotated_image": annotated_path.as_posix(),
                "number_crop_image": number_crop_path.as_posix() if number_crop_saved else None,
            },
            "manual_truth": {
                "label": manual_label,
                "is_unsure": False,
            },
            "detector": row,
            "sequence": {
                "accepted": accepted_item is not None,
                "accepted_number": accepted_item.get("number") if accepted_item else None,
                "accepted_reason": accepted_item.get("reason") if accepted_item else None,
                "deferred": deferred_item is not None,
                "deferred_number": deferred_item.get("number") if deferred_item else None,
                "deferred_reason": deferred_item.get("reason") if deferred_item else None,
                "page_debug_action": page_debug.get("action"),
                "expected_next_before": page_debug.get("expected_next_before"),
                "expected_next_after": page_debug.get("expected_next_after"),
                "strong_structure": bool(row.get("strong_structure")),
            },
            "review": {
                "review_status": None,
                "review_label": None,
                "reviewed_bag_number": None,
                "review_notes": None,
            },
        }
        write_json(sidecar_path, payload)

        exported_count += 1
        disagreement_counter.update(disagreement_types)

    print(
        json.dumps(
            {
                "set_num": args.set_num,
                "pdf_name": args.pdf_name,
                "labels_file": labels_path.as_posix(),
                "pages_in_labels": len(labels_payload["labels"]),
                "skipped_unsure_pages": skipped_unsure,
                "skipped_missing_pages": skipped_missing_page,
                "exported_disagreement_pages": exported_count,
                "output_dir": app_candidates_dir.as_posix(),
                "disagreement_counts": dict(sorted(disagreement_counter.items())),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export app.py detector disagreements against manual_labels into debug/<set>/<pdf>/app_candidates/."
    )
    parser.add_argument("--set-num", required=True, help="LEGO set number, for example 21330")
    parser.add_argument("--pdf-name", required=True, help="PDF folder name under debug/<set>, for example 21330_01")
    parser.add_argument("--debug-root", default=DEFAULT_DEBUG_ROOT, help="Root directory containing rendered debug pages.")
    parser.add_argument("--labels-root", default=DEFAULT_LABELS_ROOT, help="Directory containing manual label JSON files.")
    parser.add_argument("--start-number", type=int, default=1, help="First bag number for sequence scan.")
    parser.add_argument("--end-number", type=int, default=24, help="Last bag number for sequence scan.")
    parser.add_argument("--min-confidence", type=float, default=0.70, help="Minimum confidence used for disagreement checks.")
    parser.add_argument(
        "--allow-structure-only-start",
        type=_parse_bool,
        default=True,
        help="Whether to allow structure-only bag 1 candidates in sequence scan (true/false).",
    )
    parser.add_argument("--reference-page", type=int, default=28, help="Reference page used to rebuild the shell template.")
    parser.set_defaults(func=export_app_candidates)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
