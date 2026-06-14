# Spatial Qty ↔ Island Binding Audit

**Set:** 70618 Bags 1–4  
**Date:** 2026-06-12  
**Status:** Audit only — no data, label, or code changes

---

## Question

How many existing **reviewed** slots would bind to a different island if **nearest significant island** matching were used instead of `islands[slot_index]`?

---

## Method

| | Current (production) | Proposed (audit) |
|---|---------------------|------------------|
| Binding | `islands[slot_index]` on full sorted CC list | Nearest **significant** island to `qty_token_box` center |
| Scope | Qty-driven crops with `qty_token_boxes` + mask | Same |
| Reviewed slot | Saved part, unknown, or ignored at slot index | Same |

Output: `debug/spatial_qty_island_binding_audit/binding_audit.json`

Bag 3: no qty crops with token boxes in cache (no masks in pipeline).

---

## Totals

| Metric | Count |
|--------|------:|
| Qty crops audited | 50 |
| Crops with ≥1 slot rebinding | 25 |
| Slots that would rebind | 68 |
| **Reviewed slots affected** | **19** |

### By bag

| Bag | Qty crops | Crops w/ change | Slots change | Reviewed affected |
|-----|----------:|----------------:|-------------:|------------------:|
| 1 | 22 | 11 | 29 | **13** |
| 2 | 15 | 12 | 34 | **3** |
| 3 | 0 | 0 | 0 | 0 |
| 4 | 13 | 2 | 5 | **3** |

---

## Crops with reviewed labels affected

### Bag 1 (13 reviewed slots)

| crop_id | Reviewed affected | Total reviewed | Changes (reviewed slots only) |
|---------|------------------:|---------------:|-------------------------------|
| p11_s10_c2 | 3 | 3 | s0: 1→3, s1: 2→1, s2: 3→2 |
| p12_s11_c1 | 2 | 3 | s0: 1→2, s1: 2→1 |
| p17_s20_c1 | 3 | 3 | s0: 1→2, s1: 2→3, s2: 3→1 |
| p7_s1_c1 | 2 | 3 | s1: 2→1, s2: 3→1 |
| p8_s3_c1 | 1 | 2 | s1: 2→1 |
| p9_s5_c1 | 1 | 2 | s1: 2→1 |
| p9_s6_c2 | 1 | 2 | s1: 2→1 |

### Bag 2 (3 reviewed slots)

| crop_id | Reviewed affected | Total reviewed | Changes (reviewed slots only) |
|---------|------------------:|---------------:|-------------------------------|
| p22_s26_c1 | 3 | 5 | s2: 3→5, s3: 4→6, s4: 5→4 |

### Bag 4 (3 reviewed slots)

| crop_id | Reviewed affected | Total reviewed | Changes (reviewed slots only) |
|---------|------------------:|---------------:|-------------------------------|
| p71_s114_c1 | 1 | 1 | s0: 1→2 |
| p74_s120_c1 | 2 | 3 | s1: 2→4, s2: 3→5 |

*(p74 slots 3–4 would also rebind but are unreviewed.)*

---

## Example: p74_s120_c1

| Slot | Current island | Proposed island | Reviewed |
|------|---------------:|----------------:|:--------:|
| 0 | 1 | 1 | 98313/72 |
| 1 | 2 | 4 | 64567/71 ✗ |
| 2 | 3 | 5 | 30154/72 ✗ |
| 3 | 4 | 3 | — |
| 4 | 5 | 2 | — |

Count correct (5=5=5); mapping scrambled under current binding.

---

## Implications

- **19 saved review decisions** may reference wrong island geometry today.
- Rebinding fix requires **label migration plan** before implementation (see `SPATIAL_QTY_ISLAND_BINDING_TASK.md`).
- Bag 4 exposure audit unchanged: **0 exceptions**.

---

## Next step

Design binding fix + saved-label migration; no implementation until approved.
