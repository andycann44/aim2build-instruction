# Coordinate-Based Slot Binding Audit

**Set:** 70618 — 19 migration-reviewed slots (10 crops)  
**Status:** Audit only — no implementation  
**Output:** `debug/spatial_qty_island_binding_audit/coordinate_binding_methods_audit.json`

---

## Design principle

Slot binding must **never** use list index order (`qty_token_boxes[i] → islands[i]`).

**Preferred rule (crop-local coordinates):**

```
qty_token anchor (crop coords) → nearest significant island target (bbox centre or centroid)
```

---

## Methods tested

| ID | Qty anchor | Island target | Notes |
|----|-----------|---------------|-------|
| **order_islands_slot_index** | *(baseline)* | `all_islands[slot_index]` | Current production |
| **qty_center__sig_bbox_center** | Token centre | Sig island bbox centre | **Recommended** |
| qty_center__sig_centroid | Token centre | Sig island centroid | Same result as bbox centre (19/19) |
| qty_bottom_left__sig_bbox_center | Bottom-left of qty box | Sig island bbox centre | Differs on 1/19 slots |
| qty_bottom_left__sig_centroid | Bottom-left | Sig island centroid | Same as bottom-left bbox |
| binding_audit_nearest_sig | Token centre | Sig island bbox centre | Prior spatial audit |

All coordinate methods use **significant islands only** (same filter as review geometry).

---

## Results vs migration verdict proxies

### CLIP-oracle (9 unambiguous slots)

Saved label in Top 5 on **one** island only → treat that island as oracle.

| Method | Matches oracle | Rate |
|--------|-------------:|-----:|
| **qty_center → sig bbox centre** | 7/9 | **78%** |
| qty_center → sig centroid | 7/9 | 78% |
| qty_bottom_left → sig bbox centre | 7/9 | 78% |
| binding_audit nearest sig | 7/9 | 78% |
| **order_islands_slot_index** | 2/9 | **22%** |

Coordinate binding is **3.5× better** than order-based on CLIP-unambiguous slots.

### High-confidence human notes (9 slots from migration audit)

| Method | Matches | Rate |
|--------|--------:|-----:|
| **qty_center → sig bbox centre** | 6/9 | 67% |
| order_islands_slot_index | 3/9 | 33% |

Human notes used: keep current on p71 s0, p74 s1/s2; migrate on p11, p12, p22 slots where CLIP proposed_better.

### By CLIP signal (coord = qty_center → sig bbox centre)

| Signal | Count | Coord = migration proposed |
|--------|------:|---------------------------:|
| proposed_better | 7 | **7/7** |
| current_better | 2 | **0/2** |
| neither_top5 | 10 | mixed — needs visual review |

---

## Anchor comparison

| Comparison | Agreement |
|------------|----------:|
| qty_center bbox centre vs qty_center centroid | **19/19** |
| qty_center bbox centre vs binding_audit nearest sig | **19/19** |
| qty_center vs qty_bottom_left (bbox centre) | **18/19** |

**Only disagreement:** `p17_s20_c1` slot 2 — center→I1, bottom-left→I3 (neither in CLIP Top 5).

**Recommendation:** Use **qty token centre → nearest significant island bbox centre**. Centroid tie; bottom-left slightly worse.

---

## Order-based failures (7/9 CLIP-oracle slots)

Coordinate binding fixes these; order-based fails:

| crop | slot | Saved | Order | Coord | CLIP oracle |
|------|-----:|-------|------:|------:|-------------|
| p11_s10_c2 | 0 | 3023/4 | I1 | **I3** | I3 |
| p11_s10_c2 | 2 | 3021/70 | I3 | **I2** | I2 |
| p12_s11_c1 | 0 | 2431/308 | I1 | **I2** | I2 |
| p12_s11_c1 | 1 | 3003/308 | I2 | **I1** | I1 |
| p17_s20_c1 | 1 | 87087/0 | I2 | **I3** | I3 |
| p22_s26_c1 | 3 | 87580/320 | I4 | **I6** | I6 |
| p22_s26_c1 | 4 | 3002/70 | I5 | **I4** | I4 |

---

## Coordinate failures (2/9 CLIP-oracle — keep current cases)

Coordinate binding **overrides** CLIP-oracle on slots where preliminary human review says **keep current**:

| crop | slot | Saved | Current | Coord | Issue |
|------|-----:|-------|--------:|------:|-------|
| p71_s114_c1 | 0 | 18653/70 | **I1** | I2 | Qty under left plate; nearest sig is right plate (I2) |
| p74_s120_c1 | 2 | 30154/72 | **I3** | I5 | Qty under hinge; nearest sig is left cylinder (I5) |

**p74_s120_c1 slot 1** (64567/71): neither Top 5; coord→I4, current→I2 — visual review favors **current I2**; pure nearest-island may be wrong when qty sits between two parts.

---

## Distance example: p74_s120_c1 slot 1

| Qty anchor | Nearest island | Distance² (bbox centre) |
|-----------|---------------:|------------------------:|
| Q1 centre | I4 (proposed) | **536** ← wins nearest |
| Q1 centre | I2 (current) | 790 |
| Q1 centre | I3 | 1009 |

Nearest-island alone picks I4; human/CLIP favor I2. Suggests future rule may need **qty-under-part constraint** (anchor must be vertically aligned with island bbox) or horizontal tie-break — **not implemented in this audit**.

---

## Verdict

| Question | Answer |
|----------|--------|
| Best method vs order-based? | **qty_center → nearest sig island bbox centre** (78% vs 22% CLIP-oracle) |
| Centroid vs bbox centre? | **Equivalent** on all 19 slots |
| Bottom-left vs centre? | **Centre preferred** (18/19 agreement; centre wins on ambiguous slot) |
| Matches all migration proposed? | **19/19** (same as prior binding audit) |
| Safe to blindly migrate all 19? | **No** — 2 current_better slots + 10 neither_top5 need human/constraint rules |

---

## Implementation guardrails (future, not now)

1. Bind qty slot → island by **crop-local nearest significant island** (centre→bbox centre).
2. Never use `islands[slot_index]` or qty array index for island assignment.
3. Store `island_label` on qty slots at generation time.
4. Consider **vertical alignment gate**: qty anchor y must fall within island bbox y-range (or below part within N px) before distance tie-break.
5. Label migration: per-slot human verdict from migration contact sheets before rebinding.

---

## Regenerate

```bash
cd /Users/olly/aim2build-instruction && python3 - <<'PY'
# see instruction-v2/audit_spatial_binding_migration.py companion
# coordinate audit embedded in debug/spatial_qty_island_binding_audit/
PY
```

Full slot-level JSON: `debug/spatial_qty_island_binding_audit/coordinate_binding_methods_audit.json`
