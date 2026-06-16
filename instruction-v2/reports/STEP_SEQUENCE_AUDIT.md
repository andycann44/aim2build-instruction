# Step Sequence Audit

Read-only audit of global step sequence coherence in `instruction-v2/indexes/05_step_map.json`.

**Date:** 2026-06-16  
**Scope:** Accepted steps only (`step_number` set, `rejection_reason` null). No pipeline rerun, no manifest edits.

## Manifest summary

| Metric | Value |
|---|---:|
| `steps[]` total entries | 561 |
| Accepted steps | 521 |
| Rejected / null `step_number` | 40 |
| Manifest `step_count` | 561 |
| Global step range (accepted) | 1 – 527 |

## Per-bag summary

| Bag | Pages | Count | Min step | Max step | Span | Main-chain | Coherence |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 6–21 | 25 | 1 | 25 | 25 | 25/25 (100%) | PASS |
| 2 | 22–38 | 15 | 26 | 40 | 15 | 15/15 (100%) | PASS |
| 3 | 39–57 | 35 | 16 | 75 | 60 | 32/35 (91%) | WARN |
| 4 | 58–80 | 52 | 1 | 131 | 131 | 49/52 (94%) | WARN |
| 5 | 81–103 | 30 | 16 | 162 | 147 | 29/30 (97%) | WARN |
| 6 | 104–130 | 43 | 20 | 206 | 187 | 41/43 (95%) | WARN |
| 7 | 131–147 | 38 | 2 | 241 | 240 | 33/38 (87%) | WARN |
| 8 | 148–163 | 35 | 2 | 278 | 277 | 30/35 (86%) | WARN |
| 9 | 164–178 | 30 | 279 | 309 | 31 | 30/30 (100%) | PASS |
| 10 | 179–193 | 32 | 310 | 352 | 43 | 19/32 (59%) | FAIL |
| 11 | 194–212 | 19 | 11 | 364 | 354 | 12/19 (63%) | FAIL |
| 12 | 213–236 | 58 | 2 | 423 | 422 | 47/58 (81%) | WARN |
| 13 | 237–265 | 45 | 1 | 474 | 474 | 1/45 (2%) | FAIL |
| 14 | 266–273 | 9 | 5 | 474 | 470 | 0/9 (0%) | FAIL |
| 15 | 274–312 | 55 | 1 | 527 | 527 | 48/55 (87%) | WARN |

**Coherence** = share of accepted steps that survive a greedy page-order monotonic filter (each step must exceed all prior kept steps). This separates the dominant global sequence from spurious low-number OCR hits.

## Verdict

### Is the global step sequence coherent Bag 1 → Bag 15?

**Partially — not as a flat accepted set; largely yes as a dominant main chain.**

| Layer | Result |
|---|---|
| **Raw accepted set (521 steps)** | **NOT coherent.** 31 backwards jumps, 28 duplicate step numbers, 12/14 bag-boundary transitions go backwards when comparing bag max → next bag min. |
| **Main chain (411 steps, greedy increasing)** | **Mostly coherent.** Strictly increasing 1 → 527 across page order. 27 single-step gaps (expected: not every integer is a build step). 3 large gaps (>5 missing). |
| **Bags 1–2, 8–9** | **Coherent.** 100% main-chain; clean cross-bag handoffs (25→26, 40→44, 278→279). |
| **Bags 13–14** | **Incoherent.** Bag 13 opens with step 474 on p237 then regresses to 425+; 44/45 and 9/9 accepted steps fall off main chain. Bag 14 contributes 0 main-chain steps. |

The +305 step recovery (OCR cap removal) successfully surfaced genuine printed global steps ≥201, but also retained **~110 spurious low-number anchors** (panel digits, qty-box locals, substep fragments) that pollute per-bag min/max and cause backwards transitions in the raw set.

## 1. Backwards jumps (page order within bag)

A backwards jump is any accepted step whose `step_number` is less than the previous step on a later (or same) page.

**Total: 29** across bags 3–15 (bags 1–2 clean).

| Bag | Count | Worst jump |
|---:|---:|---|
| 3 | 1 | p55 step 75 → p56 step 16 (Δ-59) |
| 4 | 3 | p79 step 131 → p80 step 1 (Δ-130) |
| 5 | 1 | p101 step 160 → p102 step 16 (Δ-144) |
| 6 | 2 | p124 step 200 → p125 step 20 (Δ-180) |
| 7 | 4 | p146 step 241 → p147 step 3 (Δ-238) |
| 8 | 2 | p156 step 260 → p157 step 2 (Δ-258) |
| 10 | 2 | p186 step 346 → p187 step 327 (Δ-19) |
| 11 | 1 | p194 step 347 → p194 step 11 (Δ-336) |
| 12 | 4 | p232 step 415 → p232 step 7 (Δ-408) |
| 13 | 4 | p260 step 464 → p260 step 1 (Δ-463) |
| 14 | 2 | p269 step 470 → p270 step 5 (Δ-465) |
| 15 | 3 | p299 step 527 → p304 step 2 (Δ-525) |

### Notable backwards jumps

- **Bag 3:** p55 step 75 → p56 steps 16–17. Pages 56–57 are bag-3 tail; steps 16–18 are spurious low-number reads, not global sequence.
- **Bag 4:** p79 step 131 → p80 step 1. Step 1 on p80 is a false anchor (qty/panel digit).
- **Bag 6:** p124 step 200 → p125 step 20. Step 20 is spurious.
- **Bag 7:** p140 step 224 → p140 step 7 (same page). Two detections; step 7 is noise on a 200+ page.
- **Bag 10:** p186 step 346 → p187 step 327. Page 186–190 cluster has out-of-order 345/346 before 327–352.
- **Bag 12:** p218 steps 314–317 amid 366+ sequence. Likely OCR bleed from adjacent layout or mis-ordered page reads.
- **Bag 13:** p237 step 474 → p238 step 425. Bag opens with high anchor then regresses; 474 may be 424 OCR (4↔7 swap) or end-of-previous-bag bleed.
- **Bag 14:** All 9 accepted steps are off main chain; mix of globals (468–474) and locals (5, 7, 14, 44).

## 2. Duplicate step numbers

**Global duplicates: 28** distinct step numbers appearing more than once (across any bags/pages).

| Step | Occurrences | Bags | Notes |
|---:|---:|---|---|
| 1 | 5 | [1, 4, 13, 15] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 2 | 7 | [1, 7, 8, 12, 13, 15] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 3 | 3 | [1, 7, 8] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 4 | 5 | [1, 4, 7, 8, 15] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 5 | 3 | [1, 14, 15] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 7 | 6 | [1, 4, 7, 12, 14, 15] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 8 | 4 | [1, 8, 12, 15] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 9 | 2 | [1, 15] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 11 | 2 | [1, 11] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 14 | 2 | [1, 14] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 16 | 3 | [1, 3, 5] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 17 | 2 | [1, 3] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 18 | 2 | [1, 3] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 20 | 2 | [1, 6] | Early-step locals recurring in later bags — classic panel/qty OCR noise |
| 44 | 2 | [3, 14] |  |
| 145 | 2 | [5, 7] |  |
| 146 | 2 | [5, 6] |  |
| 314 | 2 | [10, 12] |  |
| 315 | 2 | [10, 12] |  |
| 316 | 2 | [10, 12] |  |
| 317 | 2 | [10, 12] |  |
| 345 | 2 | [10] | Bag 10 pages 186/192 and 186/193 — same-page layout duplicates |
| 346 | 2 | [10] | Bag 10 pages 186/192 and 186/193 — same-page layout duplicates |
| 351 | 2 | [10, 11] |  |
| 352 | 2 | [10, 11] |  |
| 397 | 2 | [12] | Bag 12 p226/p227 |
| 398 | 2 | [12] |  |
| 474 | 2 | [13, 14] | Bag 13 p237 + Bag 14 p273 |

**Within-bag duplicates:**

- Bag 8: step **2** ×2 on pages [154, 157]
- Bag 10: step **345** ×2 on pages [186, 192]
- Bag 10: step **346** ×2 on pages [186, 193]
- Bag 12: step **397** ×2 on pages [226, 227]
- Bag 12: step **398** ×2 on pages [226, 227]
- Bag 13: step **1** ×2 on pages [258, 260]

## 3. Impossible transitions

Flagged when adjacent pages (≤2 page gap) show a step jump >8 or a backwards cross-page transition.

**Total flags: 37**

| Bag | Type | Detail |
|---:|---|---|
| 3 | adjacent_page_backwards | p55 max 75 → p56 min 16 (Δ-59) |
| 4 | adjacent_page_backwards | p71 max 115 → p72 min 7 (Δ-108) |
| 4 | adjacent_page_backwards | p74 max 120 → p76 min 4 (Δ-116) |
| 4 | adjacent_page_backwards | p79 max 131 → p80 min 1 (Δ-130) |
| 5 | adjacent_page_backwards | p101 max 160 → p102 min 16 (Δ-144) |
| 5 | adjacent_page_jump | p102 max 16 → p103 min 162 (jump +146) |
| 6 | adjacent_page_backwards | p111 max 175 → p112 min 146 (Δ-29) |
| 6 | adjacent_page_jump | p112 max 146 → p113 min 177 (jump +31) |
| 6 | adjacent_page_backwards | p124 max 200 → p125 min 20 (Δ-180) |
| 6 | adjacent_page_jump | p125 max 20 → p126 min 202 (jump +182) |
| 7 | adjacent_page_jump | p132 max 4 → p133 min 207 (jump +203) |
| 7 | adjacent_page_backwards | p134 max 214 → p135 min 145 (Δ-69) |
| 7 | adjacent_page_jump | p135 max 145 → p136 min 216 (jump +71) |
| 7 | adjacent_page_backwards | p139 max 223 → p140 min 7 (Δ-216) |
| 7 | adjacent_page_backwards | p146 max 241 → p147 min 2 (Δ-239) |
| 8 | adjacent_page_jump | p148 max 8 → p150 min 247 (jump +239) |
| 8 | adjacent_page_backwards | p153 max 255 → p154 min 2 (Δ-253) |
| 8 | adjacent_page_backwards | p156 max 260 → p157 min 2 (Δ-258) |
| 8 | adjacent_page_jump | p157 max 4 → p158 min 262 (jump +258) |
| 10 | adjacent_page_jump | p185 max 324 → p186 min 345 (jump +21) |
| 10 | adjacent_page_backwards | p186 max 346 → p187 min 327 (Δ-19) |
| 10 | adjacent_page_jump | p187 max 328 → p189 min 351 (jump +23) |
| 10 | adjacent_page_backwards | p189 max 352 → p190 min 336 (Δ-16) |
| 12 | adjacent_page_backwards | p217 max 373 → p218 min 314 (Δ-59) |
| 12 | adjacent_page_jump | p219 max 317 → p220 min 378 (jump +61) |
| 12 | adjacent_page_backwards | p226 max 398 → p227 min 395 (Δ-3) |
| 12 | adjacent_page_backwards | p227 max 398 → p228 min 2 (Δ-396) |
| 12 | adjacent_page_backwards | p231 max 413 → p232 min 7 (Δ-406) |
| 13 | adjacent_page_backwards | p237 max 474 → p238 min 425 (Δ-49) |
| 13 | adjacent_page_backwards | p257 max 462 → p258 min 1 (Δ-461) |
| 13 | adjacent_page_backwards | p258 max 463 → p260 min 1 (Δ-462) |
| 13 | adjacent_page_backwards | p260 max 464 → p262 min 2 (Δ-462) |
| 14 | adjacent_page_jump | p266 max 14 → p267 min 468 (jump +454) |
| 14 | adjacent_page_backwards | p267 max 468 → p268 min 7 (Δ-461) |
| 14 | adjacent_page_backwards | p269 max 470 → p270 min 5 (Δ-465) |
| 14 | adjacent_page_jump | p270 max 5 → p271 min 44 (jump +39) |
| 14 | adjacent_page_jump | p271 max 44 → p272 min 473 (jump +429) |

### Cross-bag transitions (raw accepted min/max)

| Transition | Prev bag max | Next bag min | Gap | Status |
|---|---:|---:|---:|---|
| 1→2 | 25 | 26 | 1 | OK |
| 2→3 | 40 | 16 | -24 | BACKWARDS |
| 3→4 | 75 | 1 | -74 | BACKWARDS |
| 4→5 | 131 | 16 | -115 | BACKWARDS |
| 5→6 | 162 | 20 | -142 | BACKWARDS |
| 6→7 | 206 | 2 | -204 | BACKWARDS |
| 7→8 | 241 | 2 | -239 | BACKWARDS |
| 8→9 | 278 | 279 | 1 | OK |
| 9→10 | 309 | 310 | 1 | OK |
| 10→11 | 352 | 11 | -341 | BACKWARDS |
| 11→12 | 364 | 2 | -362 | BACKWARDS |
| 12→13 | 423 | 1 | -422 | BACKWARDS |
| 13→14 | 474 | 5 | -469 | BACKWARDS |
| 14→15 | 474 | 1 | -473 | BACKWARDS |

Raw cross-bag min/max is misleading when next-bag mins include spurious low steps. **Main-chain cross-bag transitions:**

| Transition | Prev max | Next min | Gap | Status |
|---|---:|---:|---:|---|
| 1→2 | 25 | 26 | 1 | OK |
| 2→3 | 40 | 44 | 4 | OK |
| 3→4 | 75 | 79 | 4 | OK |
| 4→5 | 131 | 133 | 2 | OK |
| 5→6 | 162 | 163 | 1 | OK |
| 6→7 | 206 | 207 | 1 | OK |
| 7→8 | 241 | 247 | 6 | LARGE GAP |
| 8→9 | 278 | 279 | 1 | OK |
| 9→10 | 309 | 310 | 1 | OK |
| 10→11 | 352 | 353 | 1 | OK |
| 11→12 | 364 | 366 | 2 | OK |
| 12→13 | 423 | 474 | 51 | LARGE GAP |
| 13→14 | 474 | — | — | no_main_chain_in_next |
| 14→15 | — | — | — | no_main_chain_in_prev |
| 15→16 | 527 | — | — | no_main_chain_in_next |

## 4. Large gaps

**Threshold:** >5 missing integers between consecutive accepted steps in page order.

### Raw accepted set: 30 large gaps

| From | To | Missing |
|---|---|---:|
| B3 p57 step 18 | B4 p59 step 79 | 60 |
| B4 p72 step 7 | B4 p72 step 117 | 109 |
| B4 p76 step 4 | B4 p78 step 124 | 119 |
| B4 p80 step 1 | B5 p82 step 133 | 131 |
| B5 p102 step 16 | B5 p103 step 162 | 145 |
| B6 p112 step 146 | B6 p113 step 177 | 30 |
| B6 p125 step 20 | B6 p126 step 202 | 181 |
| B7 p132 step 4 | B7 p133 step 207 | 202 |
| B7 p135 step 145 | B7 p136 step 216 | 70 |
| B7 p140 step 7 | B7 p141 step 225 | 217 |
| B7 p147 step 2 | B8 p148 step 8 | 5 |
| B8 p148 step 8 | B8 p150 step 247 | 238 |
| B8 p154 step 2 | B8 p155 step 258 | 255 |
| B8 p157 step 4 | B8 p158 step 262 | 257 |
| B10 p185 step 324 | B10 p186 step 345 | 20 |
| … | … | (15 more) |

### Main chain: 3 large gaps

| From | To | Missing steps |
|---|---|---:|
| B10 p185 step 324 | B10 p186 step 345 | 20 |
| B12 p225 step 390 | B12 p226 step 397 | 6 |
| B12 p236 step 423 | B13 p237 step 474 | 50 |

The dominant gap is **423 → 474** (50 missing: 424–473). Bag 13 p237 reports step 474 first; steps 425–467 follow on later pages — suggesting either (a) steps 424–473 exist but were not detected, or (b) p237 step 474 is a misread and the true sequence continues from ~424.

## 5. Main chain profile

Greedy page-order filter keeping only strictly increasing `step_number` values:

- **Length:** 411 / 521 accepted (79%)
- **Range:** step 1 (bag 1 p7) → step 527 (bag 15 p299)
- **Monotonic:** yes (by construction)
- **Single-step gaps:** 24 (normal for multi-step-per-page layouts)

### Excluded steps (110 off main chain)

Typical causes:
- Single-digit / low-teens OCR on pages with 200+ global steps (bags 7, 8, 12, 13, 14, 15)
- Bag 3 tail pages 56–57 (steps 16–18)
- Bag 10 pages 186–193 out-of-order cluster (steps 327–352 detected after 345–346)
- Bag 12 page 218 regression (steps 314–317)
- Bag 13–14 tail pages with steps 1, 2, 5, 7, 14, 44

## 6. Per-bag assessment

| Bag | Status | Notes |
|---:|---|---|
| 1 | PASS | Steps 1–25, strictly increasing, 100% main chain. |
| 2 | PASS | Steps 26–40, strictly increasing, 100% main chain. Clean handoff from bag 1. |
| 3 | WARN | Steps 44–75 coherent on pages 40–55. Tail pages 56–57 have spurious steps 16–18 (3 off chain). |
| 4 | WARN | Steps 79–131 dominate; 3 spurious lows (p72/7, p76/4, p80/1) cause backwards jumps. |
| 5 | WARN | Steps 133–162 mostly coherent; p102 step 16 is spurious. |
| 6 | WARN | Steps 163–206 mostly coherent; p125 step 20 spurious; p112 step 146 regression. |
| 7 | WARN | Steps 207–241 main chain solid; 5 spurious lows on pp 132, 140, 147. |
| 8 | WARN | Steps 247–278 main chain; 5 spurious lows on pp 148, 154, 157. Gap 241→247 (6 missing) at bag boundary. |
| 9 | PASS | Steps 279–309, 100% main chain, contiguous. Clean handoff from bag 8. |
| 10 | FAIL | Steps 310–352 present but page order scrambled on pp 186–193; duplicates 345/346; 13/32 off main chain. |
| 11 | WARN | Steps 347–364 main chain; p194 step 11 is spurious. |
| 12 | WARN | Steps 366–423 mostly coherent; p218 regression (314–317), p228/232 spurious lows, p226 gap. |
| 13 | FAIL | Opens p237 step 474 then 425–467; 44/45 off main chain. Sequence inversion at bag start. |
| 14 | FAIL | 9 accepted steps, 0 on main chain. Mix of valid high steps and local-digit noise. |
| 15 | WARN | Steps 475–527 main chain strong (48/55); tail pages 304–311 have spurious lows 1–9. |

## 7. Conclusions

1. **The recovered 200+ steps are real** (per `STEP_RECOVERY_SAMPLE.md`), but the accepted set still contains ~110 false low-number anchors that break raw sequence coherence.
2. **A dominant global sequence 1→527 exists** and is largely continuous when spurious lows are filtered (411-step main chain).
3. **Cleanest bags:** 1, 2, 9 (100% coherent). **Bags 8→9→10→11** handoffs are clean on the main chain.
4. **Highest-risk bags for sequence signoff:** 10 (page-order scramble), 13 (inverted bag start / 424–473 gap), 14 (no main-chain contribution).
5. **Recommended next checks (read-only):** visual audit of bag 13 p237 (step 474 vs expected ~424), bag 10 pp 186–193 ordering, bag 12 p218 (steps 314–317), bag 3 pp 56–57 (steps 16–18).

## Method

- **Source:** `instruction-v2/indexes/05_step_map.json` (commit `39d84c1` stage4 output)
- **Bag boundaries:** `instruction-v2/indexes/04_bag_map.json`
- **Accepted step:** `step_number != null` and `rejection_reason == null`
- **Ordering:** page ascending, then `step_index`, then `step_number`
- **Large gap threshold:** >5 missing integers
- **Adjacent-page jump threshold:** >8 (from `audit_bag_step_sequence.py`)

