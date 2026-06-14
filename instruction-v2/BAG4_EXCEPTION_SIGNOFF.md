# Bag 4 Exception Signoff

**Set:** 70618 Bag 4  
**Review date:** 2026-06-12  
**Status:** COMPLETE

Human review of naive island-vs-slot exceptions. Audit rules updated per verdicts.

---

## Verdicts

### Genuine exceptions — UNDER-EXPOSED

Geometry correct; review slots under-expose visible parts.

| crop_id | Verdict | Notes |
|---------|---------|-------|
| p71_s114_c1 | GEOMETRY_CORRECT / REVIEW_SLOTS_WRONG | 2 sig islands; qty OCR only `1x`; 1 slot |
| p77_s123_c1 | GEOMETRY_CORRECT / REVIEW_SLOTS_WRONG | 2 sig islands; single qty anchor; 1 slot |
| p70_s112_c1 | GEOMETRY_CORRECT / REVIEW_SLOTS_WRONG | 2 sig islands; 1 saved label; 1 slot |

### False positives — naive OVER-EXPOSED

Both geometry and review slots are correct; naive audit over-reported.

| crop_id | Verdict | Suppression reason |
|---------|---------|-------------------|
| p60_s80_c2 | BOTH_CORRECT | `qty_authoritative` — 3 qty tokens = 3 exposure slots; raw CC = 3 |
| p61_s84_c1 | BOTH_CORRECT | `ignored_slots` — 2 total slots, 1 ignored → 1 exposure slot = 1 sig island |
| p62_s86_c1 | BOTH_CORRECT | `ignored_slots` — 4 total slots, 2 ignored → 2 exposure slots = 2 sig islands |

---

## Refined audit rules (reporting only)

Implemented in `instruction-v2/audit_bag_slot_exposure.py`.

| Metric | Definition |
|--------|------------|
| `exposure_slots` | Review slots where `ignored=false` |
| UNDER-EXPOSED | `significant_islands > exposure_slots` |
| OVER-EXPOSED | `exposure_slots > significant_islands`, unless suppressed |

OVER suppressions (false-positive guards):

1. **ignored_slots** — historical slot span includes ignored indices; compare active exposure only
2. **qty_authoritative** — `slot_source=qty` and `len(qty_text) == exposure_slots`
3. **raw_islands_cover_exposure** — all CC blobs accounted for; significant filter is conservative

---

## Post-signoff exception count

| Audit | Exceptions |
|-------|----------:|
| Naive (total slots vs sig islands) | 6 |
| Refined (exposure-aware) | **3** (all UNDER-EXPOSED) |

Output: `debug/bag4_slot_exposure_audit.json`

---

## Next step

Fix review-slot exposure for the 3 genuine UNDER cases (qty/island reconciliation).  
No OCR, segmentation, or matching changes until that design is agreed.
