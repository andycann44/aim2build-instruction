# Bag 4 Exception Review

**Set:** 70618 Bag 4  
**Purpose:** Human review of crops where mask geometry and review-slot exposure disagree.  
**Initial naive scope:** 6 exceptions (2026-06-12)  
**Refined scope after signoff:** **3 genuine exceptions** — see [BAG4_EXCEPTION_SIGNOFF.md](./BAG4_EXCEPTION_SIGNOFF.md)

**No OCR, segmentation, matching, or review-slot generation changes in this step.**

---

## Summary

| Audit | UNDER | OVER | Total |
|-------|------:|-----:|------:|
| Naive (`total_review_slots` vs sig islands) | 3 | 3 | 6 |
| **Refined (`exposure_slots` + suppressions)** | **3** | **0** | **3** |

Refined audit: `instruction-v2/audit_bag_slot_exposure.py`  
Output JSON: `debug/bag4_slot_exposure_audit.json`

Contact-sheet assets (all 6 naive candidates): `debug/bag4_exception_review/<crop_id>/`

---

## Genuine exceptions — UNDER-EXPOSED

Mask shows **more significant parts** than active review exposure.

**Human verdict:** GEOMETRY_CORRECT / REVIEW_SLOTS_WRONG (all 3)

### p71_s114_c1

| Field | Value |
|-------|-------|
| Significant islands | **2** (raw CC: 4) |
| Exposure slots | **1** |
| slot_source | `qty` |
| qty_text | `1x` |

Qty OCR missed `2x`; qty path suppresses island fallback.

![p71_s114_c1 contact sheet](../debug/bag4_exception_review/p71_s114_c1/contact_sheet_p71_s114_c1.png)

### p77_s123_c1

| Field | Value |
|-------|-------|
| Significant islands | **2** (raw CC: 4) |
| Exposure slots | **1** |
| slot_source | `qty` |
| qty_text | `2x` |

Single qty anchor; second part island has no slot.

![p77_s123_c1 contact sheet](../debug/bag4_exception_review/p77_s123_c1/contact_sheet_p77_s123_c1.png)

### p70_s112_c1

| Field | Value |
|-------|-------|
| Significant islands | **2** (raw CC: 4) |
| Exposure slots | **1** |
| slot_source | `saved_label` |
| Saved labels | 1 (`3039 / 70`) |

Partial human review — second island unlabeled.

![p70_s112_c1 contact sheet](../debug/bag4_exception_review/p70_s112_c1/contact_sheet_p70_s112_c1.png)

---

## False positives — naive OVER-EXPOSED (suppressed)

These appeared in the naive audit but are **not** genuine exposure mismatches.

**Human verdict:** BOTH_CORRECT (all 3)

| crop_id | Naive gap | Why suppressed |
|---------|----------:|----------------|
| p60_s80_c2 | −1 | `qty_authoritative` — 3 qty tokens, 3 exposure slots, 3 raw CC |
| p61_s84_c1 | −1 | `ignored_slots` — 1 active exposure slot = 1 sig island |
| p62_s86_c1 | −2 | `ignored_slots` — 2 active exposure slots = 2 sig islands |

Contact sheets retained for reference under `debug/bag4_exception_review/`.

---

## Refined audit rules

| Term | Definition |
|------|------------|
| `total_review_slots` | All slot indices in review model (includes ignored) |
| `exposure_slots` | Slots where `ignored=false` |
| UNDER-EXPOSED | `significant_islands > exposure_slots` |
| OVER-EXPOSED | `exposure_slots > significant_islands`, unless qty/ignored/raw CC explains the gap |

Run audit:

```bash
cd instruction-v2 && python3 audit_bag_slot_exposure.py
```

---

## Next step

Address the 3 genuine UNDER-EXPOSED crops (qty/island reconciliation for review-slot exposure).  
Review-slot generation rules unchanged until that fix is designed.
