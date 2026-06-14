#!/usr/bin/env python3
"""Post-implementation verification for mandatory island_label binding."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.services.bag_review_service import build_review_model, load_review_state, _load_crop_cache
from clean.services.island_binding import (
    order_island_label,
    resolve_island_label_for_slot,
    resolve_nearest_significant_island_label,
    significant_islands_for_stem,
    qty_token_center,
)
from clean.services.full_crop_mask_paths import find_full_mask_stem

VERDICTS = ROOT / "debug" / "spatial_qty_island_binding_audit" / "migration_verdicts.json"
BINDING_AUDIT = ROOT / "debug" / "spatial_qty_island_binding_audit" / "binding_audit.json"
OUT = ROOT / "debug" / "spatial_qty_island_binding_audit" / "implementation_verification.json"
OUT_MD = ROOT / "instruction-v2" / "ISLAND_BINDING_IMPLEMENTATION_VERIFICATION.md"

SET_NUM = "70618"
BAG = 1

EXPECTED = {
    "p11_s10_c2": {
        0: {"island_label": 1, "saved": "3023/4", "verdict": "KEEP_CURRENT"},
        1: {"island_label": 1, "saved": "3039/72", "verdict": "MIGRATE_TO_PROPOSED"},
        2: {"island_label": 2, "saved": "3021/70", "verdict": "MIGRATE_TO_PROPOSED"},
    },
    "p12_s11_c1": {
        0: {"island_label": 2, "saved": "2431/308", "verdict": "MIGRATE_TO_PROPOSED"},
        1: {"island_label": 1, "saved": "3003/308", "verdict": "MIGRATE_TO_PROPOSED"},
        2: {"island_label": 3, "saved": "3069b/297", "verdict": None},
    },
}

HOLD_SLOTS = [
    ("p11_s10_c2", 0, 1, 1),
    ("p7_s1_c1", 1, 2, 1),
    ("p7_s1_c1", 2, 3, 1),
    ("p8_s3_c1", 1, 2, 1),
    ("p9_s5_c1", 1, 2, 1),
    ("p9_s6_c2", 1, 2, 1),
    ("p71_s114_c1", 0, 1, 4),
    ("p74_s120_c1", 1, 2, 4),
    ("p74_s120_c1", 2, 3, 4),
]


def _file_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _saved_label_str(saved: Any) -> str:
    if not isinstance(saved, dict):
        return ""
    pn = str(saved.get("part_num") or "").strip()
    cid = saved.get("color_id")
    if not pn:
        return ""
    return f"{pn}/{cid}"


def _load_verdict_map() -> Dict[Tuple[str, int], Dict[str, Any]]:
    payload = json.loads(VERDICTS.read_text(encoding="utf-8"))
    return {
        (e["crop_id"], int(e["slot_index"])): e
        for e in payload.get("verdicts") or []
    }


def _git_status_training_labels() -> str:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "debug/training_labels"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip()
    except Exception as exc:
        return f"error: {exc}"


def verify() -> Dict[str, Any]:
    labels_before = {
        str(p.relative_to(ROOT)): _file_hash(p)
        for p in sorted((ROOT / "debug" / "training_labels").glob("*.json"))
    }

    model = build_review_model(SET_NUM, BAG)
    verdict_map = _load_verdict_map()

    p11_p12: List[Dict[str, Any]] = []
    all_ok = True
    for crop_id, slots in EXPECTED.items():
        crop = next(c for c in model["crops"] if c["crop_id"] == crop_id)
        cache_tokens = None
        for c in crop.get("slots") or []:
            pass
        cache = next(c for c in _load_crop_cache(SET_NUM, BAG) if c["crop_id"] == crop_id)
        tokens = list(cache.get("qty_token_boxes") or [])
        stem = find_full_mask_stem(SET_NUM, BAG, crop_id)
        sig = significant_islands_for_stem(stem) if stem else []

        for slot_index, spec in slots.items():
            slot = next(s for s in crop["slots"] if s["slot_index"] == slot_index)
            saved = _saved_label_str(slot.get("saved_label"))
            island = slot.get("island_label")
            part_cutout = slot.get("part_cutout_path") or ""
            island_cutout = slot.get("island_cutout_path") or ""
            display = part_cutout or island_cutout

            coord = None
            order = order_island_label(sig, slot_index)
            if 0 <= slot_index < len(tokens):
                coord = resolve_nearest_significant_island_label(
                    qty_token_center(tokens[slot_index]), sig
                )

            ok = island == spec["island_label"] and saved == spec["saved"]
            if not ok:
                all_ok = False
            p11_p12.append(
                {
                    "crop_id": crop_id,
                    "slot_index": slot_index,
                    "saved_label": saved,
                    "expected_island_label": spec["island_label"],
                    "actual_island_label": island,
                    "order_island": order,
                    "coord_island": coord,
                    "verdict": spec.get("verdict"),
                    "part_cutout_primary": bool(part_cutout),
                    "display_path_kind": "part_cutout" if part_cutout else "island_cutout",
                    "match": ok,
                }
            )

    hold_rows: List[Dict[str, Any]] = []
    hold_ok = True
    binding_current = {}
    if BINDING_AUDIT.is_file():
        ba = json.loads(BINDING_AUDIT.read_text())
        for c in ba.get("crops") or []:
            for ch in c.get("changes") or []:
                if ch.get("reviewed"):
                    binding_current[(c["crop_id"], int(ch["slot_index"]))] = int(ch["current_island"])

    for crop_id, slot_index, expected_hold, bag in HOLD_SLOTS:
        cache = next(c for c in _load_crop_cache(SET_NUM, bag) if c["crop_id"] == crop_id)
        tokens = list(cache.get("qty_token_boxes") or [])
        resolved = resolve_island_label_for_slot(
            set_num=SET_NUM,
            bag=bag,
            crop_id=crop_id,
            slot_index=slot_index,
            slot_source="qty",
            qty_token_boxes=tokens,
        )
        current = binding_current.get((crop_id, slot_index), expected_hold)

        m = build_review_model(SET_NUM, bag)
        crop = next((c for c in m["crops"] if c["crop_id"] == crop_id), None)
        slot = next((s for s in (crop or {}).get("slots") or [] if s["slot_index"] == slot_index), None)
        model_island = slot.get("island_label") if slot else None

        ok = resolved == current
        if not ok:
            hold_ok = False
        hold_rows.append(
            {
                "crop_id": crop_id,
                "bag": bag,
                "slot_index": slot_index,
                "expected_hold_island": current,
                "resolver_island_label": resolved,
                "model_island_label": model_island,
                "model_slot_present": slot is not None,
                "verdict": verdict_map.get((crop_id, slot_index), {}).get("verdict"),
                "match": ok,
            }
        )

    labels_after = {
        str(p.relative_to(ROOT)): _file_hash(p)
        for p in sorted((ROOT / "debug" / "training_labels").glob("*.json"))
    }
    labels_unchanged = labels_before == labels_after and not _git_status_training_labels()

    # Reconcile migration audit: production should match verdict-driven binding
    mig_slots = []
    verdict_ok = True
    if (ROOT / "debug/spatial_qty_island_binding_audit/migration/migration_audit.json").is_file():
        mig = json.loads(
            (ROOT / "debug/spatial_qty_island_binding_audit/migration/migration_audit.json").read_text()
        )
        for crop in mig["crops"]:
            bag = int(crop["bag"])
            cache = next(c for c in _load_crop_cache(SET_NUM, bag) if c["crop_id"] == crop["crop_id"])
            tokens = list(cache.get("qty_token_boxes") or [])
            m = build_review_model(SET_NUM, bag)
            mc = next((c for c in m["crops"] if c["crop_id"] == crop["crop_id"]), None)
            for s in crop["slots"]:
                slot_index = int(s["slot_index"])
                verdict = verdict_map.get((crop["crop_id"], slot_index), {})
                verdict_name = str(verdict.get("verdict") or "")
                if verdict_name == "MIGRATE_TO_PROPOSED":
                    expected = int(s["proposed_island"])
                elif verdict_name == "KEEP_CURRENT":
                    expected = int(s["current_island"])
                elif verdict_name == "RELABEL_MANUAL":
                    expected = int(s["proposed_island"])
                else:
                    expected = int(s["proposed_island"])
                resolved = resolve_island_label_for_slot(
                    set_num=SET_NUM,
                    bag=bag,
                    crop_id=crop["crop_id"],
                    slot_index=slot_index,
                    slot_source="qty",
                    qty_token_boxes=tokens,
                )
                slot = next((x for x in (mc or {}).get("slots") or [] if x["slot_index"] == slot_index), None)
                model_island = slot.get("island_label") if slot else None
                matches = resolved == expected
                if not matches:
                    verdict_ok = False
                mig_slots.append(
                    {
                        "crop_id": crop["crop_id"],
                        "slot_index": slot_index,
                        "saved_label": s["saved_label"],
                        "verdict": verdict_name or None,
                        "audit_current": s["current_island"],
                        "audit_proposed": s["proposed_island"],
                        "expected_island_label": expected,
                        "resolver_island_label": resolved,
                        "model_island_label": model_island,
                        "matches_verdict": matches,
                    }
                )

    report = {
        "set_num": SET_NUM,
        "status": "pass" if all_ok and hold_ok and labels_unchanged and verdict_ok else "fail",
        "p11_p12_examples": p11_p12,
        "p11_p12_all_match": all_ok,
        "hold_slots": hold_rows,
        "hold_all_unchanged": hold_ok,
        "training_labels_unchanged": labels_unchanged,
        "training_labels_git_status": _git_status_training_labels(),
        "migration_audit_reconciliation": mig_slots,
        "migration_verdict_match_count": sum(1 for r in mig_slots if r["matches_verdict"]),
        "migration_verdict_total": len(mig_slots),
        "migration_verdict_all_match": verdict_ok,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Island Binding Implementation Verification",
        "",
        f"**Status:** {report['status'].upper()}",
        "",
        "## 1. p11 / p12 documented examples",
        "",
        "| crop | slot | saved | expected I | actual I | order | coord | verdict | display | match |",
        "|------|-----:|-------|-------------:|---------:|------:|------:|---------|---------|------:|",
    ]
    for r in report["p11_p12_examples"]:
        lines.append(
            f"| `{r['crop_id']}` | {r['slot_index']} | {r['saved_label']} | I{r['expected_island_label']} "
            f"| I{r['actual_island_label']} | I{r['order_island']} | I{r['coord_island']} "
            f"| {r.get('verdict') or '—'} | {r['display_path_kind']} | {'✓' if r['match'] else '✗'} |"
        )
    lines += [
        "",
        f"**All match:** {report['p11_p12_all_match']}",
        "",
        "## 2. HOLD slots unchanged",
        "",
        "| crop | slot | expected hold I | resolver I | model I | slot in model | verdict | match |",
        "|------|-----:|----------------:|-----------:|--------:|:-------------:|---------|------:|",
    ]
    for r in report["hold_slots"]:
        lines.append(
            f"| `{r['crop_id']}` | {r['slot_index']} | I{r['expected_hold_island']} "
            f"| I{r['resolver_island_label']} | I{r.get('model_island_label') or '—'} "
            f"| {'yes' if r.get('model_slot_present') else 'no'} "
            f"| {r.get('verdict') or '—'} | {'✓' if r['match'] else '✗'} |"
        )
    lines += [
        "",
        f"**All HOLD unchanged:** {report['hold_all_unchanged']}",
        "",
        "## 3. training_labels unchanged",
        "",
        f"- **Unchanged:** {report['training_labels_unchanged']}",
        f"- **git status:** `{report['training_labels_git_status'] or '(clean)'}`",
        "",
        "## 4. Migration audit reconciliation (verdict-driven binding)",
        "",
        f"Resolver island_label matches verdict expectation on "
        f"{report['migration_verdict_match_count']}/{report['migration_verdict_total']} migration slots.",
        "",
        "| crop | slot | verdict | audit current | audit proposed | expected I | resolver I | model I | match |",
        "|------|-----:|---------|--------------:|---------------:|-----------:|-----------:|--------:|------:|",
    ]
    for r in report["migration_audit_reconciliation"]:
        lines.append(
            f"| `{r['crop_id']}` | {r['slot_index']} | {r.get('verdict') or '—'} "
            f"| I{r['audit_current']} | I{r['audit_proposed']} | I{r['expected_island_label']} "
            f"| I{r['resolver_island_label']} | I{r.get('model_island_label') or '—'} "
            f"| {'✓' if r['matches_verdict'] else '✗'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = verify()
    print(json.dumps(report, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    if report["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
