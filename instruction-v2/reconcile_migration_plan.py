#!/usr/bin/env python3
"""Regenerate reconciled migration plan from migration_verdicts.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = ROOT / "debug" / "spatial_qty_island_binding_audit"
MIGRATION = AUDIT_DIR / "migration" / "migration_audit.json"
VERDICTS = AUDIT_DIR / "migration_verdicts.json"
OUT_JSON = AUDIT_DIR / "migration_plan_reconciled.json"
OUT_MD = ROOT / "instruction-v2" / "SPATIAL_QTY_ISLAND_MIGRATION_PLAN_RECONCILED.md"
ORIG_PLAN = AUDIT_DIR / "migration_plan.json"


def clip_signal(slot: Dict[str, Any]) -> str:
    rc, rp = slot.get("saved_rank_current"), slot.get("saved_rank_proposed")
    if rc and not rp:
        return "current_better"
    if rp and not rc:
        return "proposed_better"
    return "neither_top5"


def clip_tier(signal: str) -> str:
    if signal == "proposed_better":
        return "AUTO_MIGRATE"
    if signal == "current_better":
        return "HOLD"
    return "MANUAL_REVIEW"


def load_verdict_map() -> Dict[Tuple[str, int], Dict[str, Any]]:
    payload = json.loads(VERDICTS.read_text())
    mapping: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for entry in payload["verdicts"]:
        key = (entry["crop_id"], int(entry["slot_index"]))
        if key in mapping:
            raise ValueError(f"duplicate verdict for {key}")
        mapping[key] = entry
    return mapping


def verdict_group(verdict: str, verdict_to_group: Dict[str, str]) -> str:
    group = verdict_to_group.get(verdict)
    if not group:
        raise ValueError(f"unknown verdict: {verdict}")
    return group


def build_reconciled() -> Dict[str, Any]:
    verdict_payload = json.loads(VERDICTS.read_text())
    verdict_map = load_verdict_map()
    verdict_to_group = verdict_payload["verdict_to_group"]
    mig = json.loads(MIGRATION.read_text())
    orig_counts = json.loads(ORIG_PLAN.read_text())["counts"] if ORIG_PLAN.exists() else None

    slots: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []

    for crop in mig["crops"]:
        for slot in crop["slots"]:
            key = (crop["crop_id"], int(slot["slot_index"]))
            signal = clip_signal(slot)
            clip_group = clip_tier(signal)
            verdict_entry = verdict_map.get(key)
            if verdict_entry:
                final_group = verdict_group(verdict_entry["verdict"], verdict_to_group)
                human_verdict = verdict_entry["verdict"]
                human_source = verdict_entry["source"]
                human_note = verdict_entry.get("note")
            else:
                final_group = clip_group
                human_verdict = None
                human_source = None
                human_note = None

            row = {
                "bag": crop["bag"],
                "crop_id": crop["crop_id"],
                "slot_index": slot["slot_index"],
                "saved_label": slot["saved_label"],
                "current_island": slot["current_island"],
                "coord_island": slot["proposed_island"],
                "clip_signal": signal,
                "clip_group": clip_group,
                "final_group": final_group,
                "human_verdict": human_verdict,
                "human_verdict_source": human_source,
                "human_note": human_note,
                "contact_sheet": slot["contact_sheet"],
            }
            if verdict_entry and clip_group != final_group:
                row["conflict"] = True
                conflicts.append(row)
            else:
                row["conflict"] = False
            slots.append(row)

    groups: Dict[str, List[Dict[str, Any]]] = {
        "AUTO_MIGRATE": [],
        "MANUAL_REVIEW": [],
        "HOLD": [],
    }
    for row in slots:
        groups[row["final_group"]].append(row)
    for group in groups:
        groups[group].sort(key=lambda r: (r["bag"], r["crop_id"], r["slot_index"]))

    reviewed = sum(1 for r in slots if r["human_verdict"])
    pending = len(slots) - reviewed

    return {
        "set_num": mig["set_num"],
        "status": "reconciled_plan_only",
        "binding_rule_future": "qty_centre \u2192 nearest significant island bbox centre",
        "verdict_registry": str(VERDICTS.relative_to(ROOT)),
        "reconciliation_note": "Final group from migration_verdicts.json; unreviewed slots fall back to CLIP tier",
        "counts_before": orig_counts,
        "counts": {g: len(groups[g]) for g in groups},
        "reviewed_slots": reviewed,
        "pending_review_slots": pending,
        "conflicts": conflicts,
        "groups": groups,
    }


def render_markdown(plan: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Spatial Qty \u2194 Island Migration Plan (Reconciled)",
        "",
        "**Set:** 70618 \u2014 19 reviewed slots  ",
        "**Status:** Reconciled from `migration_verdicts.json` \u2014 plan only, no rebinds  ",
        "",
        f"**Verdict registry:** `{plan['verdict_registry']}`  ",
        f"**Reviewed:** {plan['reviewed_slots']}/19 \u00b7 **Pending:** {plan['pending_review_slots']}/19",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Group | Count |",
        "|-------|------:|",
    ]
    for g in ("AUTO_MIGRATE", "MANUAL_REVIEW", "HOLD"):
        lines.append(f"| **{g}** | {plan['counts'][g]} |")
    if plan.get("counts_before"):
        before = plan["counts_before"]
        lines += [
            "",
            "| Group | CLIP plan | Reconciled | \u0394 |",
            "|-------|----------:|-----------:|---:|",
        ]
        for g in ("AUTO_MIGRATE", "MANUAL_REVIEW", "HOLD"):
            delta = plan["counts"][g] - before[g]
            sign = f"{delta:+d}" if delta else "\u2014"
            lines.append(f"| {g} | {before[g]} | {plan['counts'][g]} | {sign} |")

    lines += ["", "---", "", "## Conflicts (CLIP tier vs human verdict)", ""]
    if plan["conflicts"]:
        lines += [
            "| crop_id | Slot | Saved | CLIP tier | Human verdict | Final |",
            "|---------|-----:|-------|-----------|---------------|-------|",
        ]
        for s in plan["conflicts"]:
            lines.append(
                f"| `{s['crop_id']}` | {s['slot_index']} | {s['saved_label']} "
                f"| {s['clip_group']} | {s['human_verdict']} | **{s['final_group']}** |"
            )
    else:
        lines.append("None.")

    for gname, title in [
        ("AUTO_MIGRATE", "AUTO_MIGRATE"),
        ("MANUAL_REVIEW", "MANUAL_REVIEW"),
        ("HOLD", "HOLD"),
    ]:
        rows = plan["groups"][gname]
        lines += ["", "---", "", f"## {title} ({len(rows)} slots)", ""]
        lines += [
            "| Bag | crop_id | Slot | Saved | I_current \u2192 I_coord | CLIP | Verdict | Source |",
            "|-----|---------|-----:|-------|---------------------|------|---------|--------|",
        ]
        for s in rows:
            hv = s["human_verdict"] or "pending"
            src = s["human_verdict_source"] or "\u2014"
            lines.append(
                f"| {s['bag']} | `{s['crop_id']}` | {s['slot_index']} | {s['saved_label']} "
                f"| I{s['current_island']} \u2192 I{s['coord_island']} | {s['clip_signal']} "
                f"| {hv} | {src} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Regenerate",
        "",
        "```bash",
        "cd instruction-v2 && python3 reconcile_migration_plan.py",
        "```",
        "",
        "## Constraints",
        "",
        "- No migration execution",
        "- No label changes",
        "- No rebinds",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    plan = build_reconciled()
    OUT_JSON.write_text(json.dumps(plan, indent=2) + "\n")
    OUT_MD.write_text(render_markdown(plan))
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    for g in plan["groups"]:
        print(f"  {g}: {plan['counts'][g]}")


if __name__ == "__main__":
    main()
