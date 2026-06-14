# Spatial Qty ↔ Island Migration Plan (Reconciled)

**Set:** 70618 — 19 reviewed slots  
**Status:** Reconciled from `migration_verdicts.json` — plan only, no rebinds  

**Verdict registry:** `debug/spatial_qty_island_binding_audit/migration_verdicts.json`  
**Reviewed:** 15/19 · **Pending:** 4/19

---

## Summary

| Group | Count |
|-------|------:|
| **AUTO_MIGRATE** | 7 |
| **MANUAL_REVIEW** | 3 |
| **HOLD** | 9 |

| Group | CLIP plan | Reconciled | Δ |
|-------|----------:|-----------:|---:|
| AUTO_MIGRATE | 7 | 7 | — |
| MANUAL_REVIEW | 9 | 3 | -6 |
| HOLD | 3 | 9 | +6 |

---

## Conflicts (CLIP tier vs human verdict)

| crop_id | Slot | Saved | CLIP tier | Human verdict | Final |
|---------|-----:|-------|-----------|---------------|-------|
| `p11_s10_c2` | 0 | 3023/4 | AUTO_MIGRATE | KEEP_CURRENT | **HOLD** |
| `p11_s10_c2` | 1 | 3039/72 | MANUAL_REVIEW | MIGRATE_TO_PROPOSED | **AUTO_MIGRATE** |
| `p7_s1_c1` | 1 | 4274/71 | MANUAL_REVIEW | KEEP_CURRENT | **HOLD** |
| `p7_s1_c1` | 2 | 32532/0 | MANUAL_REVIEW | KEEP_CURRENT | **HOLD** |
| `p8_s3_c1` | 1 | 32532/0 | MANUAL_REVIEW | KEEP_CURRENT | **HOLD** |
| `p9_s5_c1` | 1 | 32532/0 | MANUAL_REVIEW | KEEP_CURRENT | **HOLD** |
| `p9_s6_c2` | 1 | 32532/0 | MANUAL_REVIEW | KEEP_CURRENT | **HOLD** |
| `p74_s120_c1` | 1 | 64567/71 | MANUAL_REVIEW | KEEP_CURRENT | **HOLD** |

---

## AUTO_MIGRATE (7 slots)

| Bag | crop_id | Slot | Saved | I_current → I_coord | CLIP | Verdict | Source |
|-----|---------|-----:|-------|---------------------|------|---------|--------|
| 1 | `p11_s10_c2` | 1 | 3039/72 | I2 → I1 | neither_top5 | MIGRATE_TO_PROPOSED | human_review |
| 1 | `p11_s10_c2` | 2 | 3021/70 | I3 → I2 | proposed_better | MIGRATE_TO_PROPOSED | human_review |
| 1 | `p12_s11_c1` | 0 | 2431/308 | I1 → I2 | proposed_better | MIGRATE_TO_PROPOSED | human_review |
| 1 | `p12_s11_c1` | 1 | 3003/308 | I2 → I1 | proposed_better | MIGRATE_TO_PROPOSED | human_review |
| 1 | `p17_s20_c1` | 1 | 87087/0 | I2 → I3 | proposed_better | MIGRATE_TO_PROPOSED | human_review |
| 2 | `p22_s26_c1` | 3 | 87580/320 | I4 → I6 | proposed_better | pending | — |
| 2 | `p22_s26_c1` | 4 | 3002/70 | I5 → I4 | proposed_better | pending | — |

---

## MANUAL_REVIEW (3 slots)

| Bag | crop_id | Slot | Saved | I_current → I_coord | CLIP | Verdict | Source |
|-----|---------|-----:|-------|---------------------|------|---------|--------|
| 1 | `p17_s20_c1` | 0 | 60481/0 | I1 → I2 | neither_top5 | RELABEL_MANUAL | human_review |
| 1 | `p17_s20_c1` | 2 | 87618/71 | I3 → I1 | neither_top5 | pending | — |
| 2 | `p22_s26_c1` | 2 | 18962/19 | I3 → I5 | neither_top5 | pending | — |

---

## HOLD (9 slots)

| Bag | crop_id | Slot | Saved | I_current → I_coord | CLIP | Verdict | Source |
|-----|---------|-----:|-------|---------------------|------|---------|--------|
| 1 | `p11_s10_c2` | 0 | 3023/4 | I1 → I3 | proposed_better | KEEP_CURRENT | human_review |
| 1 | `p7_s1_c1` | 1 | 4274/71 | I2 → I1 | neither_top5 | KEEP_CURRENT | human_review |
| 1 | `p7_s1_c1` | 2 | 32532/0 | I3 → I1 | neither_top5 | KEEP_CURRENT | human_review |
| 1 | `p8_s3_c1` | 1 | 32532/0 | I2 → I1 | neither_top5 | KEEP_CURRENT | human_review |
| 1 | `p9_s5_c1` | 1 | 32532/0 | I2 → I1 | neither_top5 | KEEP_CURRENT | human_review |
| 1 | `p9_s6_c2` | 1 | 32532/0 | I2 → I1 | neither_top5 | KEEP_CURRENT | human_review |
| 4 | `p71_s114_c1` | 0 | 18653/70 | I1 → I2 | current_better | KEEP_CURRENT | audit_hold |
| 4 | `p74_s120_c1` | 1 | 64567/71 | I2 → I4 | neither_top5 | KEEP_CURRENT | audit_hold |
| 4 | `p74_s120_c1` | 2 | 30154/72 | I3 → I5 | current_better | KEEP_CURRENT | audit_hold |

---

## Regenerate

```bash
cd instruction-v2 && python3 reconcile_migration_plan.py
```

## Constraints

- No migration execution
- No label changes
- No rebinds
