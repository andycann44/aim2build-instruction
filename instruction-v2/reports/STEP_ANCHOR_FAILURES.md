# Step Anchor Failures

Evidence catalogue of OCR-anchor mistakes causing remaining sequence-parity failures.

**Date:** 2026-06-16  
**Source:** `indexes/05_step_map.json` (commit `39d84c1`) — no pipeline rerun, no patches.  
**Visual crops:** `reports/step_parity_review_pack/`  
**Rules:** `docs/RULEBOOK.md` § Sequence Plausibility Rules (SP1–SP6)

---

## Executive summary

Four pages contain the **minimum root OCR-anchor mistakes** that poison global step sequence coherence:

| Page | Bag | Printed | Detected | OCR raw | Failure class | Downstream impact |
|---:|---:|---:|---:|---|---|---|
| **186** | 10 | 325 | **345** | `345` | Digit swap (2↔4) | False main-chain anchor; blocks genuine 325–344 |
| **189** | 10 | 331 | **351** | `351` | Digit swap (3↔5) | Second false high anchor; cascades to step 352 |
| **237** | 13 | 424 | **474** | `474` | Digit swap (2↔7) | +50 jump at bag start; blocks genuine 425–467 |
| **271** | 14 | 472 | **44** | `44` | Fragment / slash interference | Backwards step; off main chain |

Fixing these four anchors (or rejecting them under SP2/SP4) would restore sequence coherence for Bags 10, 13, and 14 without touching unrelated pages.

**Cascade failures (not root causes):**

- p186 step 346: printed **326**, OCR raw `376` → same-page correction forced **346** because prior anchor was wrong **345**
- p189 step 352: printed **332**, OCR raw `332` (correct) → same-page correction forced **352** because prior anchor was wrong **351**

---

## Page 186 — Bag 10

**Context:** Previous page (p185) ends at step **324** (correct). Expected next: **325**.

### Anchor 1 — step index 1

| Field | Value |
|---|---|
| **Printed step** | **325** |
| **Detected step** | **345** |
| **OCR output** | `345` (`tesseract_step_box_tight_threshold_6x`) |
| **OCR confidence** | 0.75 |
| **Step box** | x=30 y=149 w=83 h=36 |
| **Components** | 3 |
| **Main chain** | ON (false positive) |
| **Badge crop** | `step_parity_review_pack/bag10_p186_s345_i1_badge.png` |

![Badge p186 — printed 325](step_parity_review_pack/bag10_p186_s345_i1_badge.png)

**Why the wrong anchor won**

1. **OCR digit swap:** Badge shows **325**; Tesseract read **345** (middle digit 2→4). No alternate read or sequence check challenged it.
2. **Range gate only:** `filterInvalidStepNumber()` accepts any `0 < n < 1000` — 345 passes.
3. **Gap too large for recovery, none for rejection:** Jump from p185 max 324 → 345 is gap **20**; `promoteSequenceGapCandidates()` skips gaps **> 6** but does not reject the high anchor.
4. **Outlier check passed:** `rejectSequenceOutliers()` uses forward context including later false highs (p189 **351**), so 345 appears validated (`forwardMax` ≈ 351 ≥ 340).

**Sequence impact:** Establishes false main-chain floor at 345; genuine steps 325–344 on pp 187–193 fall off chain.

---

### Anchor 2 — step index 2

| Field | Value |
|---|---|
| **Printed step** | **326** |
| **Detected step** | **346** |
| **OCR output** | Raw `376` → **sequence-corrected** to `346` |
| **Correction signal** | `step_number_sequence_corrected: true`, `step_number_before_sequence_correction: 376`, `step_number_expected_from_previous: 346` |
| **Step box** | x=30 y=709 w=84 h=36 |
| **Main chain** | ON (cascade) |
| **Badge crop** | `step_parity_review_pack/bag10_p186_s346_i2_badge.png` |

![Badge p186 — printed 326](step_parity_review_pack/bag10_p186_s346_i2_badge.png)

**Why the wrong anchor won**

1. **Cascade from Anchor 1:** `correctSamePageSequentialOcr()` expects previous+1 = 345+1 = **346**.
2. Raw OCR read `376` (single-digit diff from 346) was **overwritten** to 346 to satisfy same-page monotonic expectation — but expectation was built on wrong 345.
3. Badge confirms printed **326**; neither 376 nor 346 matches print.

**Classification:** Cascade failure — root cause is Anchor 1.

---

## Page 189 — Bag 10

**Context:** Genuine sequence at this point should be ~329–332. False main chain already at 345+ from p186.

### Anchor 1 — step index 1

| Field | Value |
|---|---|
| **Printed step** | **331** |
| **Detected step** | **351** |
| **OCR output** | `351` (`tesseract_step_box_tight_threshold_6x`) |
| **OCR confidence** | 0.75 |
| **Step box** | x=115 y=287 w=55 h=36 |
| **Components** | 2 |
| **Main chain** | ON (false positive) |
| **Badge crop** | `step_parity_review_pack/bag10_p189_s351_i1_badge.png` |

![Badge p189 — printed 331](step_parity_review_pack/bag10_p189_s351_i1_badge.png)

**Why the wrong anchor won**

1. **OCR digit swap:** Badge shows **331**; Tesseract read **351** (middle digit 3→5).
2. **No cross-page plausibility check:** 351 > 345 (false p186 anchor) so it extends main chain rather than being questioned.
3. **Full-page audit did not correct:** Right-column audit candidate on same page read **333** (also wrong) — primary box OCR won without sequence validation.

**Sequence impact:** Second false high anchor; validates p186 errors in `rejectSequenceOutliers()` forward context.

---

### Anchor 2 — step index 2

| Field | Value |
|---|---|
| **Printed step** | **332** |
| **Detected step** | **352** |
| **OCR output** | Raw `332` (correct print match) → **sequence-corrected** to `352` |
| **Rejected reads** | `3532` rejected (`invalid_step_number_1000_or_more`) |
| **Correction signal** | `step_number_before_sequence_correction: 332`, `step_number_expected_from_previous: 352` |
| **Step box** | x=115 y=737 w=55 h=37 |
| **Main chain** | ON (cascade) |
| **Badge crop** | `step_parity_review_pack/bag10_p189_s352_i2_badge.png` |

![Badge p189 — printed 332](step_parity_review_pack/bag10_p189_s352_i2_badge.png)

**Why the wrong anchor won**

1. **OCR actually read the printed number** (`332`) on one threshold — but `correctSamePageSequentialOcr()` replaced it with **352** because prior anchor was wrong **351**.
2. Same-page correction treats OCR as subordinate to local +1 expectation (SP6 violation: sequence inference overrode correct OCR).

**Classification:** Cascade failure — root cause is Anchor 1 (351).

---

## Page 237 — Bag 13

**Context:** Bag 12 ends at step **423** (p236). Bag 13 opens; expected first global step **424**.

### Anchor 1 — step index 1

| Field | Value |
|---|---|
| **Printed step** | **424** |
| **Detected step** | **474** |
| **OCR output** | `474` (`tesseract_step_box_tight_threshold_6x`) |
| **OCR confidence** | 0.75 |
| **Step box** | x=115 y=845 w=82 h=36 |
| **Components** | 3 |
| **Main chain** | ON (false positive) |
| **Badge crop** | `step_parity_review_pack/bag13_p237_s474_i1_badge.png` |
| **Page role** | Bag 13 start page (bag-label graphic + first build step) |

![Badge p237 — printed 424](step_parity_review_pack/bag13_p237_s474_i1_badge.png)

**Why the wrong anchor won**

1. **OCR digit swap:** Badge shows **424**; Tesseract read **474** (middle digit 2→7). Classic 4↔7 confusion on bold sans-serif digits.
2. **No cross-bag continuity check (SP5):** Bag 12 max **423** → detected **474** is gap **50**. `promoteSequenceGapCandidates()` only operates within a bag; cross-bag jump unchecked.
3. **Gap rejection missing (SP3):** Gap > 6 with zero recovered intermediates — should be suspicious; no rejection fired.
4. **Outlier check bypassed:** On bag-first page (idx=0), `rejectSequenceOutliers()` forward context temporarily included page 240 false highs **478/479** (later removed), which satisfied `noForwardAnchor = false` for step 474 at evaluation time.

**Sequence impact:** Single anchor blocks **44/45** genuine steps (425–467) from main chain; Bag 14 globals also off-chain (chain stuck at 474).

---

## Page 271 — Bag 14

**Context:** Previous page (p270) has spurious step **5** (panel digit). Expected global step ~**471–472**.

### Anchor 1 — step index 3

| Field | Value |
|---|---|
| **Printed step** | **472** |
| **Detected step** | **44** |
| **OCR output** | `44` (`tesseract_step_box_gray_5x`) |
| **OCR confidence** | 0.65 |
| **Rejected reads** | `4` ×2 (`ocr_digit_count_less_than_visual_component_count`, visual_component_count=2) |
| **Step box** | x=115 y=142 w=51 h=36 |
| **Components** | 2 (printed number has 3 digits) |
| **Main chain** | OFF |
| **Badge crop** | `step_parity_review_pack/bag14_p271_s44_i3_badge.png` |

![Badge p271 — printed 472](step_parity_review_pack/bag14_p271_s44_i3_badge.png)

**Why the wrong anchor won**

1. **Fragment OCR:** Badge shows **472** with diagonal strike-through lines across the **7**. Tesseract returned **44** — third digit lost; alternative reads were single `4` (rejected for component count mismatch).
2. **Component/detection mismatch:** Box has **2** visual components but printed step has **3** digits — merge or crop too narrow; OCR never produced `472`.
3. **No backwards rejection (SP4):** Detected **44** ≪ previous globals (~468–470); no rule rejects backwards transition.
4. **Low confidence accepted:** `step_number_confidence: 0.65` — below typical 0.75 on other pages — still accepted because `rejection_reason` is null.

**Sequence impact:** Contributes to Bag 14 incoherence; backwards jump p269 step 470 → p271 step 44.

---

## Minimum failure set

```text
ROOT OCR-ANCHOR MISTAKES (4)
────────────────────────────
p186  325 → 345   OCR 2↔4
p189  331 → 351   OCR 3↔5
p237  424 → 474   OCR 2↔7   (+ cross-bag gap 50)
p271  472 → 44    OCR fragment / slash interference

CASCADE (2 — fix roots above)
─────────────────────────────
p186  326 → 346   same-page correction from wrong 345
p189  332 → 352   same-page correction from wrong 351
```

## Common acceptance gaps

| Gap | Rule | Effect on these pages |
|---|---|---|
| OCR accepted without sequence check | SP6 | All four pages |
| Large jump not rejected | SP2, SP3 | p186, p237 |
| Cross-bag continuity ignored | SP5 | p237 |
| Backwards step not rejected | SP4 | p271 |
| Same-page correction trusts wrong prior | SP1 | p186 s346, p189 s352 |
| Forward context validates correlated errors | — | p186, p237 |

## Method

- Manifest fields: `step_number`, `signals.step_number_raw_text`, `signals.step_number_source`, `signals.step_number_sequence_corrected`, `signals.step_number_rejected_reads`
- Printed step: visual verification from badge crops (`step_box` + 12 px pad)
- Main-chain status: greedy page-order filter per `STEP_SEQUENCE_AUDIT.md`
- No runtime code modified
