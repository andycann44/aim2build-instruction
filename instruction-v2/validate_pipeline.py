#!/usr/bin/env python3
"""Validate the instruction-v2 manifest chain before matching runs."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INDEXES = ROOT / "indexes"


@dataclass(frozen=True)
class StageSpec:
    stage: int
    label: str
    file_names: tuple[str, ...]
    required_fields: tuple[str, ...]
    count_field: str | None
    list_field: str | None


STAGES = (
    StageSpec(0, "set context", ("00_set_context.json",), ("stage", "name", "set_num", "parts"), None, "parts"),
    StageSpec(1, "pdf manifest", ("01_pdf_manifest.json",), ("stage", "name", "pdf"), None, None),
    StageSpec(2, "page index", ("02_page_index.json",), ("stage", "name", "page_count", "pages"), "page_count", "pages"),
    StageSpec(3, "bag candidates", ("03_bag_candidates.json",), ("stage", "name", "candidate_count", "candidates"), "candidate_count", "candidates"),
    StageSpec(4, "bag map", ("04_bag_map.json",), ("stage", "name", "bag_count", "bags"), "bag_count", "bags"),
    StageSpec(
        5,
        "step box map",
        ("05_step_box_map.json", "05_step_map.json"),
        ("stage", "name", "step_count", "steps"),
        "step_count",
        "steps",
    ),
    StageSpec(
        6,
        "callout crop box map",
        ("06_callout_crop_box_map.json",),
        ("stage", "name", "entry_count", "entries"),
        "entry_count",
        "entries",
    ),
    StageSpec(7, "qty ocr", ("07_qty_ocr_map.json",), ("stage", "name", "entry_count", "entries"), "entry_count", "entries"),
    StageSpec(
        8,
        "part segmentation",
        ("08_part_segmentation_map.json",),
        ("stage", "name", "entry_count", "entries"),
        "entry_count",
        "entries",
    ),
)


def fail(stage: StageSpec, message: str) -> str:
    return f"FAIL stage {stage.stage} {stage.label}: {message}"


def pass_line(stage: StageSpec) -> str:
    return f"PASS stage {stage.stage} {stage.label}"


def load_manifest(stage: StageSpec) -> tuple[Path | None, dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    selected = next((INDEXES / name for name in stage.file_names if (INDEXES / name).exists()), None)
    if selected is None:
        return None, None, [fail(stage, f"missing manifest {' or '.join(stage.file_names)}")]

    try:
        data = json.loads(selected.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return selected, None, [fail(stage, f"invalid JSON in {selected.name}: {exc.msg} at line {exc.lineno}")]

    if not isinstance(data, dict):
        return selected, None, [fail(stage, f"{selected.name} must contain a JSON object")]

    return selected, data, errors


def repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def check_path(stage: StageSpec, entry_label: str, field: str, value: Any) -> list[str]:
    if not isinstance(value, str) or not value:
        return [fail(stage, f"missing {field} on {entry_label}")]

    path = Path(value)
    if path.is_absolute() and not path.is_relative_to(ROOT):
        return [fail(stage, f"{field} on {entry_label} points outside instruction-v2: {value}")]

    full_path = repo_path(value)
    if not full_path.exists():
        return [fail(stage, f"missing referenced file {field} on {entry_label}: {value}")]

    return []


def check_required_fields(stage: StageSpec, data: dict[str, Any]) -> list[str]:
    errors = []
    for field in stage.required_fields:
        if field not in data:
            errors.append(fail(stage, f"missing top-level field {field}"))
    return errors


def check_nonzero_count(stage: StageSpec, data: dict[str, Any]) -> list[str]:
    errors = []
    if stage.count_field:
        count = data.get(stage.count_field)
        if not isinstance(count, int) or count <= 0:
            errors.append(fail(stage, f"{stage.count_field} must be a non-zero integer"))

    if stage.list_field:
        entries = data.get(stage.list_field)
        if not isinstance(entries, list) or not entries:
            errors.append(fail(stage, f"{stage.list_field} must be a non-empty list"))

    return errors


def check_manifest_paths(stage: StageSpec, data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if stage.stage == 1:
        pdf = data.get("pdf")
        if not isinstance(pdf, dict):
            return [fail(stage, "pdf must be an object")]
        errors.extend(check_path(stage, "pdf", "saved_path", pdf.get("saved_path")))

    if stage.stage == 2:
        for index, page in enumerate(data.get("pages", [])):
            errors.extend(check_path(stage, f"page entry {index}", "image_path", page.get("image_path")))

    if stage.stage == 3:
        for index, candidate in enumerate(data.get("candidates", [])):
            errors.extend(check_path(stage, f"candidate entry {index}", "image_path", candidate.get("image_path")))
            errors.extend(check_path(stage, f"candidate entry {index}", "debug_image_path", candidate.get("debug_image_path")))

    if stage.stage == 4:
        for bag_index, bag in enumerate(data.get("bags", [])):
            for evidence_index, evidence in enumerate(bag.get("evidence", [])):
                if evidence.get("source") == "human_gap_review":
                    continue
                entry = f"bag entry {bag_index} evidence {evidence_index}"
                errors.extend(check_path(stage, entry, "debug_image_path", evidence.get("debug_image_path")))

    if stage.stage == 5:
        for index, step in enumerate(data.get("steps", [])):
            errors.extend(check_path(stage, f"step entry {index}", "debug_overlay_path", step.get("debug_overlay_path")))

    if stage.stage == 6:
        for index, entry in enumerate(data.get("entries", [])):
            entry_label = f"entry {index}"
            errors.extend(check_path(stage, entry_label, "crop_image_path", entry.get("crop_image_path")))
            errors.extend(check_path(stage, entry_label, "debug_overlay_path", entry.get("debug_overlay_path")))

    if stage.stage == 7:
        for index, entry in enumerate(data.get("entries", [])):
            errors.extend(check_path(stage, f"entry {index}", "crop_image_path", entry.get("crop_image_path")))

    if stage.stage == 8:
        for index, entry in enumerate(data.get("entries", [])):
            entry_label = f"entry {index}"
            for field in ("crop_image_path", "mask_path", "cutout_path", "overlay_path"):
                errors.extend(check_path(stage, entry_label, field, entry.get(field)))

    return errors


def validate_stage_order(present_stages: list[int]) -> list[str]:
    errors = []
    if present_stages != list(range(len(STAGES))):
        errors.append("FAIL stage order: manifest chain skipped a required stage")
    return errors


def validate() -> int:
    all_errors: list[str] = []
    present_stages: list[int] = []

    for stage in STAGES:
        _path, data, errors = load_manifest(stage)
        if errors:
            all_errors.extend(errors)
            continue
        assert data is not None
        present_stages.append(stage.stage)

        stage_errors = []
        stage_errors.extend(check_required_fields(stage, data))
        stage_errors.extend(check_nonzero_count(stage, data))
        stage_errors.extend(check_manifest_paths(stage, data))

        if stage_errors:
            all_errors.extend(stage_errors)
        else:
            print(pass_line(stage))

    order_errors = validate_stage_order(present_stages)
    all_errors.extend(order_errors)

    for error in all_errors:
        print(error)

    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(validate())
