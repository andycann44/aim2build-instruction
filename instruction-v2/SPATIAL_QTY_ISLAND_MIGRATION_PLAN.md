# Spatial Qty ↔ Island Migration Plan

**Set:** 70618 — 19 reviewed slots across 10 crops  
**Status:** Plan only — no implementation, no label changes  
**Binding rule (future):** qty centre → nearest significant island centre  
**Retired:** `islands[slot_index]`, qty array order  
**Not adding:** below-part / x-containment / x-overlap pre-filters (tested, no benefit)

---

## Summary

| Group | Slots | Action |
|-------|------:|--------|
| **AUTO_MIGRATE** | 7 | Rebind to coord island after signoff |
| **MANUAL_REVIEW** | 9 | Visual review on contact sheet first |
| **HOLD** | 3 | Do not rebind — preserve current island |

Machine-readable index: `debug/spatial_qty_island_binding_audit/migration_plan.json`

---

## AUTO_MIGRATE (7 slots)

CLIP **proposed_better** — saved label in proposed Top 5 only. Coordinate binding agrees. Safe candidates for island rebind after human signoff.

| Bag | crop_id | Slot | Saved | Current → Coord | CLIP rank cur | CLIP rank prop | Contact sheet |
|-----|---------|-----:|-------|----------------|--------------:|---------------:|-----------------|
| 1 | `p11_s10_c2` | 0 | 3023/4 | I1 → **I3** | — | 1 | `debug/spatial_qty_island_binding_audit/migration/bag1_p11_s10_c2/slot0_migration.png` |
| 1 | `p11_s10_c2` | 2 | 3021/70 | I3 → **I2** | — | 1 | `debug/spatial_qty_island_binding_audit/migration/bag1_p11_s10_c2/slot2_migration.png` |
| 1 | `p12_s11_c1` | 0 | 2431/308 | I1 → **I2** | — | 1 | `debug/spatial_qty_island_binding_audit/migration/bag1_p12_s11_c1/slot0_migration.png` |
| 1 | `p12_s11_c1` | 1 | 3003/308 | I2 → **I1** | — | 1 | `debug/spatial_qty_island_binding_audit/migration/bag1_p12_s11_c1/slot1_migration.png` |
| 1 | `p17_s20_c1` | 1 | 87087/0 | I2 → **I3** | — | 2 | `debug/spatial_qty_island_binding_audit/migration/bag1_p17_s20_c1/slot1_migration.png` |
| 2 | `p22_s26_c1` | 3 | 87580/320 | I4 → **I6** | — | 1 | `debug/spatial_qty_island_binding_audit/migration/bag2_p22_s26_c1/slot3_migration.png` |
| 2 | `p22_s26_c1` | 4 | 3002/70 | I5 → **I4** | — | 1 | `debug/spatial_qty_island_binding_audit/migration/bag2_p22_s26_c1/slot4_migration.png` |

**Per-slot action:** Rebind `island_label` from current to coord island. Regenerate cutout/mask paths. Saved label unchanged.

**Review before executing:** Confirm contact sheet shows saved label on proposed cutout.

## MANUAL_REVIEW (9 slots)

CLIP **neither_top5** — saved label absent from both cutout Top 5. Visual review required before any rebind decision.

| Bag | crop_id | Slot | Saved | Current → Coord | CLIP rank cur | CLIP rank prop | Contact sheet |
|-----|---------|-----:|-------|----------------|--------------:|---------------:|-----------------|
| 1 | `p11_s10_c2` | 1 | 3039/72 | I2 → **I1** | — | — | `debug/spatial_qty_island_binding_audit/migration/bag1_p11_s10_c2/slot1_migration.png` |
| 1 | `p17_s20_c1` | 0 | 60481/0 | I1 → **I2** | — | — | `debug/spatial_qty_island_binding_audit/migration/bag1_p17_s20_c1/slot0_migration.png` |
| 1 | `p17_s20_c1` | 2 | 87618/71 | I3 → **I1** | — | — | `debug/spatial_qty_island_binding_audit/migration/bag1_p17_s20_c1/slot2_migration.png` |
| 1 | `p7_s1_c1` | 1 | 4274/71 | I2 → **I1** | — | — | `debug/spatial_qty_island_binding_audit/migration/bag1_p7_s1_c1/slot1_migration.png` |
| 1 | `p7_s1_c1` | 2 | 32532/0 | I3 → **I1** | — | — | `debug/spatial_qty_island_binding_audit/migration/bag1_p7_s1_c1/slot2_migration.png` |
| 1 | `p8_s3_c1` | 1 | 32532/0 | I2 → **I1** | — | — | `debug/spatial_qty_island_binding_audit/migration/bag1_p8_s3_c1/slot1_migration.png` |
| 1 | `p9_s5_c1` | 1 | 32532/0 | I2 → **I1** | — | — | `debug/spatial_qty_island_binding_audit/migration/bag1_p9_s5_c1/slot1_migration.png` |
| 1 | `p9_s6_c2` | 1 | 32532/0 | I2 → **I1** | — | — | `debug/spatial_qty_island_binding_audit/migration/bag1_p9_s6_c2/slot1_migration.png` |
| 2 | `p22_s26_c1` | 2 | 18962/19 | I3 → **I5** | — | — | `debug/spatial_qty_island_binding_audit/migration/bag2_p22_s26_c1/slot2_migration.png` |

**Per-slot action:** Record verdict: `MIGRATE` | `KEEP_CURRENT` | `RELABEL_MANUAL`.

**Note:** Five slots share saved label `32532/0` (p7/p8/p9) — review as a batch; likely same root cause.

## HOLD (3 slots)

Do not auto-rebind. Current island preserved regardless of coordinate proposal.

| Bag | crop_id | Slot | Saved | Current → Coord | CLIP rank cur | CLIP rank prop | Contact sheet |
|-----|---------|-----:|-------|----------------|--------------:|---------------:|-----------------|
| 4 | `p71_s114_c1` | 0 | 18653/70 | I1 → **I2** | 1 | — | `debug/spatial_qty_island_binding_audit/migration/bag4_p71_s114_c1/slot0_migration.png` |
| 4 | `p74_s120_c1` | 1 | 64567/71 | I2 → **I4** | — | — | `debug/spatial_qty_island_binding_audit/migration/bag4_p74_s120_c1/slot1_migration.png` |
| 4 | `p74_s120_c1` | 2 | 30154/72 | I3 → **I5** | 1 | — | `debug/spatial_qty_island_binding_audit/migration/bag4_p74_s120_c1/slot2_migration.png` |

- **`p71_s114_c1` slot 0** — CLIP current_better — saved label ranks on current island only
- **`p74_s120_c1` slot 1** — Slot↔qty index mismatch — coord picks I4, saved label on I2; requires qty-remap review
- **`p74_s120_c1` slot 2** — CLIP current_better — saved label ranks on current island only

**Per-slot action:** Keep current island binding. Coordinate rule would assign different island — do not apply.

---

## Contact sheet folders by group

### AUTO_MIGRATE

- `debug/spatial_qty_island_binding_audit/migration/bag1_p11_s10_c2/` — slot0, slot2
- `debug/spatial_qty_island_binding_audit/migration/bag1_p12_s11_c1/` — slot0, slot1
- `debug/spatial_qty_island_binding_audit/migration/bag1_p17_s20_c1/` — slot1
- `debug/spatial_qty_island_binding_audit/migration/bag2_p22_s26_c1/` — slot3, slot4

### MANUAL_REVIEW

- `debug/spatial_qty_island_binding_audit/migration/bag1_p11_s10_c2/` — slot1
- `debug/spatial_qty_island_binding_audit/migration/bag1_p17_s20_c1/` — slot0, slot2
- `debug/spatial_qty_island_binding_audit/migration/bag1_p7_s1_c1/` — slot1, slot2
- `debug/spatial_qty_island_binding_audit/migration/bag1_p8_s3_c1/` — slot1
- `debug/spatial_qty_island_binding_audit/migration/bag1_p9_s5_c1/` — slot1
- `debug/spatial_qty_island_binding_audit/migration/bag1_p9_s6_c2/` — slot1
- `debug/spatial_qty_island_binding_audit/migration/bag2_p22_s26_c1/` — slot2

### HOLD

- `debug/spatial_qty_island_binding_audit/migration/bag4_p71_s114_c1/` — slot0
- `debug/spatial_qty_island_binding_audit/migration/bag4_p74_s120_c1/` — slot1, slot2

---

## Execution order (future — not now)

1. **HOLD** — skip; no changes.
2. **MANUAL_REVIEW** — complete human verdicts on contact sheets.
3. **AUTO_MIGRATE** — rebind after spot-check of 7 contact sheets.
4. Implement coordinate binding in slot generation (`island_label` at creation).
5. Regenerate affected cutouts/masks only for migrated slots.

## Constraints

- No production code changes until this plan is signed off
- No `saved_label` / `training_labels` changes without per-slot verdict
- Bag 4 exposure audit must remain **0 exceptions**
