# Spatial Qty ↔ Island Migration Audit

**Set:** 70618  
**Scope:** 10 affected crops, 19 reviewed slots  
**Status:** Audit only — no labels, code, or training data modified  
**Purpose:** Human truth before any migration

---

## Deliverables

| Asset | Path |
|-------|------|
| Machine-readable index | `debug/spatial_qty_island_binding_audit/migration/migration_audit.json` |
| Per-slot contact sheets | `debug/spatial_qty_island_binding_audit/migration/bag{N}_{crop_id}/slot{N}_migration.png` |
| Audit generator (read-only) | `instruction-v2/audit_spatial_binding_migration.py` |

Each contact sheet shows: original crop, island overlay, **current** island cutout, **proposed** island cutout, CLIP Top 5 for each.

---

## CLIP pre-read (not a verdict — human review required)

Automated signal: does saved label appear in Top 5 for current vs proposed island cutout?

| Signal | Slots | Meaning |
|--------|------:|---------|
| **proposed_better** | 7 | Saved label in proposed Top 5 only |
| **neither_top5** | 10 | Saved label absent from both Top 5 |
| **current_better** | 2 | Saved label in current Top 5 only |

**Preliminary read:** In 7/19 slots CLIP ranks the saved label on the **proposed** island but not current — consistent with **label on wrong island today**. In 10/19 neither ranks it (ambiguous / color mismatch / CLIP limit). **2 slots (p71 s0, p74 s2)** rank saved label on **current** island only — proposed rebinding may be wrong there.

---

## All 19 reviewed slots

| Bag | crop_id | Slot | Saved label | Current → Proposed | Rank cur | Rank prop | CLIP signal |
|-----|---------|-----:|-------------|-------------------:|---------:|----------:|-------------|
| 1 | p11_s10_c2 | 0 | 3023/4 | I1 → I3 | — | **1** | proposed_better |
| 1 | p11_s10_c2 | 1 | 3039/72 | I2 → I1 | — | — | neither_top5 |
| 1 | p11_s10_c2 | 2 | 3021/70 | I3 → I2 | — | **1** | proposed_better |
| 1 | p12_s11_c1 | 0 | 2431/308 | I1 → I2 | — | **1** | proposed_better |
| 1 | p12_s11_c1 | 1 | 3003/308 | I2 → I1 | — | **1** | proposed_better |
| 1 | p17_s20_c1 | 0 | 60481/0 | I1 → I2 | — | — | neither_top5 |
| 1 | p17_s20_c1 | 1 | 87087/0 | I2 → I3 | — | **2** | proposed_better |
| 1 | p17_s20_c1 | 2 | 87618/71 | I3 → I1 | — | — | neither_top5 |
| 1 | p7_s1_c1 | 1 | 4274/71 | I2 → I1 | — | — | neither_top5 |
| 1 | p7_s1_c1 | 2 | 32532/0 | I3 → I1 | — | — | neither_top5 |
| 1 | p8_s3_c1 | 1 | 32532/0 | I2 → I1 | — | — | neither_top5 |
| 1 | p9_s5_c1 | 1 | 32532/0 | I2 → I1 | — | — | neither_top5 |
| 1 | p9_s6_c2 | 1 | 32532/0 | I2 → I1 | — | — | neither_top5 |
| 2 | p22_s26_c1 | 2 | 18962/19 | I3 → I5 | — | — | neither_top5 |
| 2 | p22_s26_c1 | 3 | 87580/320 | I4 → I6 | — | **1** | proposed_better |
| 2 | p22_s26_c1 | 4 | 3002/70 | I5 → I4 | — | **1** | proposed_better |
| 4 | p71_s114_c1 | 0 | 18653/70 | I1 → I2 | **1** | — | **current_better** |
| 4 | p74_s120_c1 | 1 | 64567/71 | I2 → I4 | — | — | neither_top5 |
| 4 | p74_s120_c1 | 2 | 30154/72 | I3 → I5 | **1** | — | **current_better** |

---

## Per-crop review folders

| Bag | crop_id | Slots | Folder |
|-----|---------|------:|--------|
| 1 | p11_s10_c2 | 3 | `migration/bag1_p11_s10_c2/` |
| 1 | p12_s11_c1 | 2 | `migration/bag1_p12_s11_c1/` |
| 1 | p17_s20_c1 | 3 | `migration/bag1_p17_s20_c1/` |
| 1 | p7_s1_c1 | 2 | `migration/bag1_p7_s1_c1/` |
| 1 | p8_s3_c1 | 1 | `migration/bag1_p8_s3_c1/` |
| 1 | p9_s5_c1 | 1 | `migration/bag1_p9_s5_c1/` |
| 1 | p9_s6_c2 | 1 | `migration/bag1_p9_s6_c2/` |
| 2 | p22_s26_c1 | 3 | `migration/bag2_p22_s26_c1/` |
| 4 | p71_s114_c1 | 1 | `migration/bag4_p71_s114_c1/` |
| 4 | p74_s120_c1 | 2 | `migration/bag4_p74_s120_c1/` |

---

## Human review workflow

For each `slot{N}_migration.png`:

1. Confirm saved label matches **visual part** under qty token (original crop).
2. Compare current vs proposed cutout — which island is the qty token anchored to?
3. Use CLIP panels as tie-breaker only; color_id mismatches (e.g. 64567/71 vs 64567/0) may suppress exact Top-5 match.
4. Record per slot: `KEEP_CURRENT` | `MIGRATE_TO_PROPOSED` | `RELABEL_MANUAL`

---

## Notable cases for human attention

**p74_s120_c1 slot 1** — saved 64567/71; current I2 cutout ranks 64567/**0** #1 (0.695); proposed I4 does not rank 64567 at all. Visual: current cutout is tall cylinder under qty token. **Likely current binding correct; proposed wrong.**

**p74_s120_c1 slot 2** — saved 30154/72; current I3 ranks 30154/72 **#1** (0.672); proposed I5 does not. **Likely current binding correct.**

**p71_s114_c1 slot 0** — saved 18653/70; current I1 ranks **#1** (0.864); proposed I2 shows different part (plate). **Likely current binding correct; do not migrate.**

**p11_s10_c2 / p12_s11_c1** — multiple slots where proposed island ranks saved label #1 and current does not. **Strong candidates for wrong-island-today.**

**p7/p8/p9 32532/0 slots** — neither Top 5; all propose I2→I1. Requires visual review (single-island crops with OCR noise).

---

## Constraints (unchanged)

- No nearest-island binding implementation
- No label / training data migration
- Bag 4 exposure audit: **0 exceptions**

---

## Regenerate

```bash
cd instruction-v2 && python3 audit_spatial_binding_migration.py
```

Requires port 8000 server for CLIP (`/debug/buildability-clip-suggest`).
