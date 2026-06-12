"""Audit Bag 4 OCR step misreads and write diagnostics report."""

from __future__ import annotations

from pathlib import Path

from bag4_misread_step_detection import (
    MISREAD_CLASSIFICATION,
    SUBSTEP_CLASSIFICATION,
    audit_bag_misread_steps,
    audit_bag_substep_numbers,
)
from paths import ROOT_DIR


OUT_DIR = ROOT_DIR / "debug" / "bag4_misread_step_audit"
OUT_JSON = OUT_DIR / "70618_bag4_misread_steps.json"
OUT_TABLE = OUT_DIR / "70618_bag4_misread_steps.tsv"
SUBSTEP_JSON = OUT_DIR / "70618_bag4_substep_numbers.json"
SUBSTEP_TABLE = OUT_DIR / "70618_bag4_substep_numbers.tsv"


def _write_table(rows: list) -> None:
    header = ["page", "ocr_value", "corrected_value", "classification", "match_reason"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row.get("page") or ""),
                    str(row.get("ocr_value") or ""),
                    str(row.get("corrected_value") or ""),
                    str(row.get("classification") or MISREAD_CLASSIFICATION),
                    str(row.get("match_reason") or ""),
                ]
            )
        )
    OUT_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_substep_table(rows: list) -> None:
    header = ["page", "ocr_value", "parent_step", "classification", "match_reason"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row.get("page") or ""),
                    str(row.get("ocr_value") or ""),
                    str(row.get("parent_step") or ""),
                    str(row.get("classification") or SUBSTEP_CLASSIFICATION),
                    str(row.get("match_reason") or ""),
                ]
            )
        )
    SUBSTEP_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    payload = audit_bag_misread_steps(bag=4)
    substep_payload = audit_bag_substep_numbers(bag=4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        __import__("json").dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    SUBSTEP_JSON.write_text(
        __import__("json").dumps(substep_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_table(payload.get("rows", []))
    _write_substep_table(substep_payload.get("rows", []))

    print("page\tocr_value\tcorrected_value\tclassification")
    for row in payload.get("rows", []):
        print(
            f"{row.get('page')}\t{row.get('ocr_value')}\t{row.get('corrected_value')}\t{row.get('classification')}"
        )
    print("")
    print("page\tocr_value\tparent_step\tclassification")
    for row in substep_payload.get("rows", []):
        print(
            f"{row.get('page')}\t{row.get('ocr_value')}\t{row.get('parent_step')}\t{row.get('classification')}"
        )
    print(str(OUT_JSON))
    print(str(SUBSTEP_JSON))
    print(str(OUT_TABLE))
    print(str(SUBSTEP_TABLE))
    print(f"misread_count={payload.get('row_count')}")
    print(f"substep_count={substep_payload.get('row_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
