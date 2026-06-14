# Constrained Coordinate Slot Binding Audit

**Set:** 70618 — 19 migration slots (10 crops) + focus crops `p71_s114_c1`, `p74_s120_c1`  
**Status:** Design audit only — no implementation, no label or training data changes  
**Output:** `debug/spatial_qty_island_binding_audit/constrained_coordinate_binding_audit.json`

---

## Question

Does a **constrained** coordinate rule beat pure nearest-island and order-based binding?

**Proposed rule (evaluated, not implemented):**

```
qty token centre
  → significant islands only
  → prefer islands vertically above / aligned with qty token
  → distance tie-break (horizontal |Δx| or 2D)
```

---

## Methods tested

All methods use **significant islands** (same filter as review geometry).  
Qty anchor: **token bbox centre** in crop coordinates.

| ID | Constraint chain (first non-empty wins) | Tie-break |
|----|-------------------------------------------|-----------|
| **order_islands_slot_index** | *(baseline)* `islands[slot_index]` | — |
| **pure_2d** | all sig islands | min 2D distance |
| **pure_hx_below** | below-part → all sig | min \|Δx\| |
| **qtybox_x_hx_below** | qty-x-overlap + below → below → all | min \|Δx\| |
| **xcontain_hx_below** | qty-centre-in-bbox + below → below → all | min \|Δx\| |
| **xcontain_hx_below_tol8** | same, below tolerance 8 px | min \|Δx\| |
| **xcontain_qtybox_x_hx** | x-contain → qty-x-overlap + below → below → all | min \|Δx\| |
| **xcontain_2d_below** | x-contain + below → below → all | min 2D |
| **qtybox_x_only_hx** | qty-x-overlap → all | min \|Δx\| |

**Below-part:** island bbox bottom ≤ qty box top + tolerance (12 px default, 8 px variant).  
**Qty-x-overlap:** qty bbox x-range overlaps island bbox x-range (±2 px pad).  
**X-contain:** qty centre x falls inside island bbox x-range (±2 px pad).

---

## Verdict proxies

| Proxy | Slots | Source |
|-------|------:|--------|
| **Human notes** | 10 | 7 migrate-to-proposed + 3 keep-current from migration audit |
| **CLIP oracle** | 9 | Saved label in Top 5 on exactly one island |
| **Migration proposed** | 19 | Prior nearest-island binding audit |
| **KEEP-3** | 3 | `p71_s114_c1` s0, `p74_s120_c1` s1/s2 — do-not-migrate cases |

---

## Summary results

| Method | Human | CLIP oracle | KEEP-3 | = pure | = order |
|--------|------:|------------:|-------:|-------:|--------:|
| **pure_2d** (nearest sig) | **7/10 (70%)** | **7/9 (78%)** | 0/3 | 19/19 | 0/19 |
| xcontain_2d_below | 7/10 (70%) | 7/9 (78%) | 0/3 | 18/19 | 1/19 |
| pure_hx_below | 6/10 (60%) | 6/9 (67%) | 0/3 | 17/19 | 1/19 |
| qtybox_x_hx_below | 6/10 (60%) | 6/9 (67%) | 0/3 | 17/19 | 1/19 |
| xcontain_hx_below | 6/10 (60%) | 6/9 (67%) | 0/3 | 17/19 | 1/19 |
| xcontain_hx_below_tol8 | 6/10 (60%) | 6/9 (67%) | 0/3 | 17/19 | 1/19 |
| xcontain_qtybox_x_hx | 6/10 (60%) | 6/9 (67%) | 0/3 | 17/19 | 1/19 |
| qtybox_x_only_hx | 6/10 (60%) | 6/9 (67%) | 0/3 | 18/19 | 0/19 |
| **order_islands_slot_index** | 3/10 (30%) | 2/9 (22%) | **3/3** | 0/19 | 19/19 |

### Headline

1. **Coordinate binding (pure nearest) beats order-based** on human notes (70% vs 30%) and CLIP oracle (78% vs 22%). Confirms prior coordinate audit.
2. **No constrained variant fixes the 3 KEEP slots.** Every tested constraint chain still assigns the same wrong island as pure nearest on `p71_s114_c1` s0 and `p74_s120_c1` s1/s2.
3. **Constraints that diverge from pure nearest regress**, not improve: `p17_s20_c1` s2 and `p22_s26_c1` s4 flip to wrong islands under x-contain / hx-below chains.
4. **Best constrained rule = no constraint.** `xcontain_2d_below` ties pure on human/CLIP but adds 1 regression (`p17_s20_c1` s2); all hx-below chains lose 1 human + 1 CLIP vs pure.

---

## Comparison vs pure nearest

| Outcome | Count | Slots |
|---------|------:|-------|
| Same as pure_2d | 17–19/19 | Most slots unchanged |
| **Regression** (pure correct → constrained wrong) | 1–2 | `p17_s20_c1` s2 (pure→I1, constrained→I3); `p22_s26_c1` s4 (pure→I4, hx-below→I2) |
| **Improvement** over pure | **0** | — |

Pure nearest already matches migration proposed on **19/19** slots. Constrained rules only **break** previously correct assignments.

---

## Comparison vs order-based

| Method | Agrees with order | Disagrees (coord wins on CLIP/human) |
|--------|------------------:|-------------------------------------:|
| pure_2d | 0/19 | 19/19 |
| xcontain_hx_below | 1/19 (`p17_s20_c1` s2) | 18/19 |
| order | 19/19 | — |

Order-based is wrong on 7/9 CLIP-oracle slots (same list as prior coordinate audit). Constrained coordinate does not recover order's 3 KEEP slots while keeping coord's 7 wins.

---

## Focus crops: why constraints fail

### `p71_s114_c1` slot 0 — saved 18653/70

| | Detail |
|---|--------|
| Qty | **One token only** (Q0 centre ≈ 63.5, 91.5) — OCR undercount; crop has 2 parts |
| Islands | I1 bracket centre x≈226; I2 plate centre x≈97 |
| Order | I1 — saved label ranks **#1** on I1 cutout (CLIP current_better) |
| Pure / all constrained | **I2** — qty sits under left plate (x-overlap, x-contain, nearest all agree) |

**Root cause:** Binding rule cannot pair a single qty token with two parts. Geometry says I2; saved truth says I1. Constraint filters cannot disambiguate — both islands pass below-part, only I2 passes x-overlap/x-contain.

### `p74_s120_c1` slot 1 — saved 64567/71

| | Detail |
|---|--------|
| Qty | Q1 centre ≈ (268.5, 72) |
| Human keep | I2 (order-based) |
| Pure / constrained | **I4** — qty-x-overlap + nearest (hx=5.5 vs I2 hx=111.5) |

**Root cause:** Slot index 1 uses `qty_token_boxes[1]` (rightmost qty) but order maps to I2 (middle stud). Geometry correctly places Q1 under I4 (right cylinder). "Keep I2" preserves a **label↔island association**, not qty-under-part geometry. Constraints enforce geometry → I4, not I2.

### `p74_s120_c1` slot 2 — saved 30154/72

| | Detail |
|---|--------|
| Qty | Q2 centre ≈ (32.5, 74.5) |
| Human keep | I3 (order-based) |
| Pure / constrained | **I5** — qty-x-overlap + nearest (hx=4.5 vs I3 hx=65.5) |

**Root cause:** Q2 is leftmost qty under I5 (left cylinder). Human/CLIP favor I3 (hinge at x≈98) because saved label ranks on I3 cutout. Q3 at x≈83 is the qty actually under I3 — **slot↔qty index mismatch**, not fixable by island distance alone.

### `p74_s120_c1` qty ↔ island spatial truth

| Qty | Centre x | Geometric island | Order island for same index |
|-----|---------:|-----------------|------------------------------:|
| Q0 | 204 | I1 | I1 ✓ |
| Q1 | 268 | **I4** | I2 ✗ |
| Q2 | 32 | **I5** | I3 ✗ |
| Q3 | 83 | I3 | — |
| Q4 | 154 | I2 | — |

Five qty tokens, five sig islands, but **slot_index → qty[i] → island[i]** scrambles s1 and s2. Pure nearest fixes island-for-qty but exposes that **saved labels on s1/s2 may belong to different qty tokens**.

---

## Per-slot assignment (19 slots)

| crop | slot | saved | order | pure | xcontain_hx | human | CLIP |
|------|-----:|-------|------:|-----:|------------:|------:|-----:|
| p11_s10_c2 | 0 | 3023/4 | I1 | **I3** | I3 | I3 | I3 |
| p11_s10_c2 | 1 | 3039/72 | I2 | I1 | I1 | — | — |
| p11_s10_c2 | 2 | 3021/70 | I3 | **I2** | I2 | I2 | I2 |
| p12_s11_c1 | 0 | 2431/308 | I1 | **I2** | I2 | I2 | I2 |
| p12_s11_c1 | 1 | 3003/308 | I2 | **I1** | I1 | I1 | I1 |
| p17_s20_c1 | 0 | 60481/0 | I1 | I2 | I2 | — | — |
| p17_s20_c1 | 1 | 87087/0 | I2 | **I3** | I3 | I3 | I3 |
| p17_s20_c1 | 2 | 87618/71 | I3 | I1 | **I3** | — | — |
| p7_s1_c1 | 1 | 4274/71 | I2 | I1 | I1 | — | — |
| p7_s1_c1 | 2 | 32532/0 | I3 | I1 | I1 | — | — |
| p8_s3_c1 | 1 | 32532/0 | I2 | I1 | I1 | — | — |
| p9_s5_c1 | 1 | 32532/0 | I2 | I1 | I1 | — | — |
| p9_s6_c2 | 1 | 32532/0 | I2 | I1 | I1 | — | — |
| p22_s26_c1 | 2 | 18962/19 | I3 | I5 | I5 | — | — |
| p22_s26_c1 | 3 | 87580/320 | I4 | **I6** | I6 | I6 | I6 |
| p22_s26_c1 | 4 | 3002/70 | I5 | **I4** | I2 | I4 | I4 |
| p71_s114_c1 | 0 | 18653/70 | **I1** | I2 | I2 | I1 | I1 |
| p74_s120_c1 | 1 | 64567/71 | **I2** | I4 | I4 | I2 | — |
| p74_s120_c1 | 2 | 30154/72 | **I3** | I5 | I5 | I3 | I3 |

**Bold** = matches human or CLIP oracle where defined.

---

## Safest future binding rule (recommendation)

### Do not implement yet

Historical truth (`saved_label`, `training_labels`) must not move until per-slot human verdicts are recorded on migration contact sheets.

### Recommended binding logic (when implemented)

```
PRIMARY (default for new qty slots):
  qty_centre → nearest significant island bbox centre

GUARDRAILS (before changing existing labels):
  1. Never use islands[slot_index] or qty array order as binding.
  2. Persist island_label on slot at creation time.
  3. If coord binding ≠ current island AND CLIP signal = current_better → HOLD, require human.
  4. If coord binding ≠ current island AND crop has qty_count ≠ sig_island_count → HOLD (p71 class).
  5. If coord binding ≠ current island AND qty[i] x-overlap island differs from order island → HOLD + flag slot↔qty remap review (p74 class).

DO NOT ADD (tested, no benefit):
  - below-part-only filter (all sig islands pass — qty row sits at crop bottom)
  - x-contain / qty-x-overlap as hard filters before distance (regresses p17/p22; still fails p71/p74)
  - horizontal-only tie-break without guardrails (same failures as pure nearest on KEEP slots)
```

### Why not constrained coordinate as primary?

| Goal | pure_2d | best constrained (xcontain_2d_below) |
|------|---------|----------------------------------------|
| CLIP oracle | 78% | 78% (tie) |
| Human notes | 70% | 70% (tie) |
| KEEP-3 | 0/3 | 0/3 |
| Regressions vs pure | — | 1 slot (p17 s2) |
| Matches migration proposed | 19/19 | 18/19 |

Constraints add complexity without improving accuracy and introduce regressions. The failures on `p71` and `p74` are **structural** (qty undercount, slot↔qty↔island index mismatch), not fixable by tighter geometric filters.

### Migration strategy (after human signoff)

| Tier | Slots | Action |
|------|------:|--------|
| **A — migrate** | 7 | CLIP proposed_better; pure coord = proposed; human migrate notes agree |
| **B — review** | 10 | neither_top5; coord ≠ order; visual contact sheet required |
| **C — hold** | 3 | current_better + KEEP notes; do not auto-rebind (`p71` s0, `p74` s2; `p74` s1 needs qty-remap review) |

---

## Constraints (unchanged)

- No binding implementation in production code
- No label / training data migration
- Bag 4 exposure audit: **0 exceptions**

---

## Artifacts

| File | Contents |
|------|----------|
| `debug/spatial_qty_island_binding_audit/constrained_coordinate_binding_audit.json` | Full per-slot assignments + summary |
| `debug/spatial_qty_island_binding_audit/coordinate_binding_methods_audit.json` | Pure coordinate method comparison |
| `debug/spatial_qty_island_binding_audit/migration/migration_audit.json` | CLIP + contact sheet index |
| `instruction-v2/SPATIAL_QTY_ISLAND_COORDINATE_BINDING_AUDIT.md` | Prior pure-coordinate audit |
| `instruction-v2/SPATIAL_QTY_ISLAND_MIGRATION_AUDIT.md` | Human review workflow |
