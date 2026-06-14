# Follow-up Task: Spatial qty-to-island binding audit

**Status:** OPEN — design/audit only; no implementation  
**Created:** 2026-06-12  
**Trigger crop:** `p74_s120_c1` (70618 Bag 4, step 120)  
**Related audit:** `debug/bag4_audit_p74_s120_c1/`

---

## Context

Bag 4 exception review and island slot supplementation are **signed off**:

- Refined exposure audit: **0 exceptions** (`debug/bag4_slot_exposure_audit.json`)
- Genuine under-exposure fixes applied for `p71_s114_c1`, `p77_s123_c1`, `p70_s112_c1`

`p74_s120_c1` is **not** an exposure-count issue. Do not treat it as an exception or re-open Bag 4 exception work.

---

## Confirmed findings (p74_s120_c1)

| Metric | Value |
|--------|------:|
| Visible parts | 5 |
| Significant islands | 5 |
| qty tokens | 5 |
| Review slots | 5 |

**Count is correct.**

### Problem

1. `qty_token_boxes` are stored in **OCR discovery order** (order tokens were found), not spatial order (e.g. left-to-right, top-to-bottom).
2. Qty-driven slot cutouts bind via `islands[slot_index]` when `island_label` is unset — i.e. sorted island list index, **not** nearest qty box.
3. Slot index follows qty array order → **slot ↔ island mapping scrambles**.

### Impact

- Slot thumbnails show wrong part for qty anchor
- Saved labels attach to wrong visual slot
- CLIP / candidate ranking queries wrong cutout geometry

### Example (p74_s120_c1)

| Slot | Qty token x | Part under token | Cutout island (by slot_index) | Aligned? |
|------|------------:|------------------|-------------------------------|----------|
| 0 | 194 | Island 1 | Island 1 | ✓ |
| 1 | 260 | Island 4 | Island 2 | ✗ |
| 2 | 26 | Island 5 | Island 3 | ✗ |
| 3 | 76 | Island 3 | Island 4 | ✗ |
| 4 | 148 | Island 2 | Island 5 | ✗ |

Contact sheet: `debug/bag4_audit_p74_s120_c1/contact_sheet_p74_s120_c1.png`

---

## Goal

Bind qty slots to the **correct** significant island geometry:

- **Option A:** Spatially sort `qty_token_boxes` before `_build_qty_sequence` / slot generation, **or**
- **Option B:** For each qty slot, assign `island_label` via nearest significant island (centroid / bbox distance to qty box)

Either approach must preserve:

- Existing qty slot count (do not change exposure-count rules)
- Saved-label slot indices where already reviewed
- `training_labels` schema
- Island supplement logic (signed-off under-exposure fix)

---

## Scope

### In scope (audit phase)

1. Inventory qty-driven crops across Bags 1–4 where `len(qty_token_boxes) >= 2`
2. Measure slot ↔ island alignment today (nearest-qty vs `islands[slot_index]`)
3. Count crops with scrambled mapping (≥1 mismatched slot)
4. Compare Option A vs Option B for p74 and representative multi-qty crops
5. Document risk to existing saved labels on mis-mapped crops (e.g. p74 slots 0–2)

### Out of scope (until explicit approval)

- Implementation in `bag_review_service.py` or crop_cache pipeline
- OCR / segmentation / matching / CLIP changes
- Bag 4 manual review session
- Re-opening Bag 4 exception audit (remains **0 exceptions**)

---

## Key code paths (reference)

| Location | Role |
|----------|------|
| `clean/services/bag_review_service.py` → `_build_qty_sequence()` | 1 slot per qty token (array order) |
| `clean/services/bag_review_service.py` → `build_review_model()` | qty path; `island_label=None` on qty slots |
| `clean/services/full_crop_mask_paths.py` → `ensure_island_slot_cutout_path()` | `islands[slot_index]` when `island_label` unset |
| `debug/crop_cache/*.json` → `qty_token_boxes` | OCR discovery order persisted |

---

## Acceptance criteria (audit deliverable)

- [ ] List of affected crops with mismatch count per bag
- [ ] Contact sheet(s) for worst cases (include p74)
- [ ] Recommended binding strategy (A, B, or hybrid) with trade-offs
- [ ] Saved-label migration note for crops already reviewed with scrambled mapping
- [ ] Explicit sign-off gate before any code change

---

## Current Bag 4 status (unchanged)

| Metric | Value |
|--------|------:|
| Exposure exceptions | **0** |
| Total review slots | 87 |
| Human review | **Paused** — resume after this task is scoped, not before |

---

## Next step

Run read-only **Spatial qty-to-island binding audit** across the dataset. No implementation until audit deliverable is reviewed and approved.
