# Sequence Validator Implementation Plan

Implementation plan for a **Sequence Plausibility Validator** in Stage 4 (`step_map_scan.mjs`).

**Date:** 2026-06-16  
**Status:** IMPLEMENTATION PLAN ONLY — no runtime patches, no pipeline rerun  
**Rule source:** `docs/RULEBOOK.md` § Sequence Plausibility Rules (SP1–SP6, `PARITY_RULE`)  
**Design input:** `reports/ANCHOR_CORRECTION_PLAN.md`  
**Pipeline context:** `docs/SETUP_V1_TRUSTED_BASELINE.md` — Step Map **UNDER INVESTIGATION**; Stage 4 via `stage4_step_map.py`

---

## 1. Validator inputs

Single function (design name: `validateSequencePlausibility`). Three explicit inputs plus derived context passed from hook points.

### `anchor`

One candidate step anchor **after OCR, before acceptance** into `steps[]`.

| Field | Source | Use |
|---|---|---|
| `bag` | Page/bag loop | Bag boundary detection (SP5) |
| `page` | `pageEntry.page` | Ordering, first-page-of-bag flag |
| `step_index` | Primary box index on page | Same-page ordering |
| `step_number` | Post-`filterInvalidStepNumber()` OCR value | Transition arithmetic |
| `step_box` | `{ x, y, w, h }` | Geometry signals (SP6) |
| `signals.step_number_raw_text` | Tesseract | OCR evidence (SP6) |
| `signals.step_number_confidence` | OCR pipeline | Weak-evidence gate (SP6) |
| `signals.step_number_rejected_reads` | OCR pipeline | Fragment / component mismatch (SP6) |
| `signals.components` | Visual grouping | Digit-count vs component count (SP6) |
| `source` | e.g. primary vs `sequence_gap_full_page_audit` | Gap-recovery evidence (SP3) |

**Not an input:** printed ground truth. Validator has no oracle; it uses sequence context and OCR quality signals only.

### `prevTrustedMax`

Last **accepted** global step number on the trusted chain, in strict page order across the current bag.

| Property | Definition |
|---|---|
| **Initial value (Bag 1)** | `0` or `null` (no prior) |
| **Update rule** | Increment only when validator returns **ACCEPT** for an anchor |
| **Excludes** | Rejected anchors, SUSPICIOUS anchors (unless later promoted), spurious low digits on same page that were rejected |
| **Scope** | Within current bag during `analyzePage()` sweep; carried page-to-page inside bag loop |

**Purpose:** SP1, SP2, SP3, SP4 — all forward/backward transition checks use this, not raw manifest max or unvalidated page locals.

**Example (p186):** After p185 accepts step **324**, `prevTrustedMax = 324` when validating p186 anchor **345**.

### `priorBagMax`

Last **accepted** global step number at end of Bag **N−1**.

| Property | Definition |
|---|---|
| **Initial value (Bag 1)** | `0` or `null` |
| **Update rule** | Set to final `prevTrustedMax` when bag loop completes; passed into first page(s) of next bag |
| **Scope** | Cross-bag only (SP5) |

**Purpose:** First anchor on Bag N start page (e.g. p237, Bag 13) compares against Bag 12 exit (step **423**), not an empty in-bag context.

**Trusted baseline alignment:** Bag boundaries are **TRUSTED** per `SETUP_V1_TRUSTED_BASELINE.md` (Bag 13 starts page **237**). Validator uses boundary metadata only for *when* to apply SP5, not to reset step numbering.

### Derived context (computed at hook, not caller-supplied)

| Derived | Rule |
|---|---|
| `effectivePrior` | `prevTrustedMax`, or `priorBagMax` when `page === bag.start_page` and no in-bag accepts yet |
| `gap` | `anchor.step_number - effectivePrior` |
| `isFirstPageOfBag` | `page === bag.start_page` |
| `prevOnPage` | Last **ACCEPT** on same page (for same-page +1, SP1) |

---

## 2. Validator outputs

Three outcomes. Exactly one per anchor evaluation.

### `ACCEPT`

| Meaning | Effect |
|---|---|
| Anchor is sequence-plausible | Write to `steps[]`; set `rejection_reason: null` |
| Update chain | `prevTrustedMax ← anchor.step_number`; update `prevOnPage` |
| Signals | `signals.sequence_plausibility: "accept"` |

**When:** `gap === 1`, or `1 < gap ≤ 6` with gap-recovery evidence pending (see SP3), or no prior (first step of set).

### `REJECT`

| Meaning | Effect |
|---|---|
| Anchor fails plausibility | Do **not** write to `steps[]`, or write with `rejection_reason` set |
| Update chain | No change to `prevTrustedMax` |
| Signals | `signals.sequence_plausibility: "reject"`, `signals.sequence_plausibility_rules: ["SP2", …]` |

**Suggested `rejection_reason` values:**

- `implausible_backwards_transition` (SP4)
- `implausible_forward_gap` (SP2, SP3)
- `implausible_cross_bag_transition` (SP5)
- `insufficient_ocr_evidence` (SP6)

### `SUSPICIOUS`

| Meaning | Effect |
|---|---|
| Borderline — sequence or OCR weak but not clearly impossible | Do **not** update `prevTrustedMax` |
| Default handling | Omit from `steps[]` **or** accept with `rejection_reason: null` and `signals.sequence_plausibility: "suspicious"` for downstream human/diagnostic review |

**When (design):**

- `gap` in `{2…6}` with no gap-recovery candidate yet (defer until post-`promoteSequenceGapCandidates()` re-check)
- OCR confidence below threshold (e.g. `< 0.70`) but sequence otherwise OK
- Component count vs digit count mismatch without backwards/large-gap violation (p271-like partial reads before full SP4 trigger)

**Proof-page policy:** p186, p189, p237, p271 root anchors resolve to **REJECT**, not SUSPICIOUS.

---

## 3. Rulebook rules checked (SP1–SP6)

| Rule | Check | ACCEPT | REJECT | SUSPICIOUS |
|---|---|---|---|---|
| **SP1** — Global steps sequential (+1 expected) | `gap === 1` vs prior | Primary path | If `gap > 6` (via SP2/SP3) | If `2 ≤ gap ≤ 6` pending gap audit |
| **SP2** — Large jumps suspicious | `gap > 6` without evidence | — | Yes | — |
| **SP3** — Gap rejection separate from recovery | Large gap + no `recoveredSteps` / audit evidence | If gap ≤ 6 and recovery will run | Yes if gap > 6 and no evidence | If gap ≤ 6, evidence not yet known |
| **SP4** — Backwards suspicious | `anchor.step_number < effectivePrior` | — | Yes | — |
| **SP5** — Cross-bag continuity | First page of bag: compare to `priorBagMax` | If `gap === 1` from bag exit | Yes if large gap or backwards vs `priorBagMax` | Rare |
| **SP6** — OCR not sole truth | Confidence, rejected_reads, components | Supports ACCEPT when sequence OK | Contributes to REJECT when combined with SP4 | Weak OCR alone |

**Evaluation order (design):**

1. SP4 — backwards → **REJECT**
2. SP5 — cross-bag large gap (first bag page) → **REJECT** if SP2/SP3 fire
3. SP2 + SP3 — forward gap > 6 → **REJECT**
4. SP1 — gap === 1 → **ACCEPT**
5. SP1 + SP3 — gap 2–6 → **SUSPICIOUS** until gap recovery pass; **ACCEPT** if recovery confirms intermediates
6. SP6 — modulates SUSPICIOUS vs REJECT on borderline backwards/low reads

**Not checked by validator:** V1 crop-cache truth (separate `applyV1StepAnchorGuardrails()`), bag-label graphics, substep panel digits (partially SP6 geometry heuristics only).

---

## 4. Exact runtime hook points

Target file: `instruction-v2/step_map_scan.mjs`  
Entry: `instruction-v2/stage4_step_map.py` (unchanged wrapper)

### Hook A — `analyzePage()`

**Location:** After primary-box OCR assignment (~L611–L631), **before** `steps[]` assembly (~L644).

```text
for each primary group box:
  readPrintedStepNumber → filterInvalidStepNumber
  validateSequencePlausibility(anchor, { prevTrustedMax, priorBagMax, … })
  if ACCEPT → include in steps draft
  if REJECT → skip or record rejected candidate
  if SUSPICIOUS → skip or flag
```

**Responsibilities:**

- Per-anchor validation with live `prevTrustedMax`
- Pass `priorBagMax` when `page === bag.start_page` and in-bag `prevTrustedMax` not yet established
- Update `prevTrustedMax` only on **ACCEPT**, in page order (sort by `step_box.y` before validating same-page anchors)

**Parameters:** Extend `analyzePage(..., sequenceContext)` where `sequenceContext = { prevTrustedMax, priorBagMax, bag, recoveredSteps }`.

### Hook B — `correctSamePageSequentialOcr()`

**Location:** ~L475–L502; currently called at ~L662 **after** draft `steps[]` built.

**Design change:**

| Current | Planned |
|---|---|
| Always runs; overwrites OCR to `previous + 1` on single-digit diff | **Gated:** run only when both anchors already **ACCEPT** by validator |
| Sequence inference overrides OCR (SP6 violation) | **Disabled** when validator **REJECT** on prior anchor; never correct toward implausible prior |

**Hook behaviour:**

```text
if validator enabled:
  correctSamePageSequentialOcr(steps.filter(acceptedByValidator))
else:
  [legacy path — not used after rollout]
```

**Alternative:** Remove call entirely; rely on OCR + validator + gap recovery. Minimum plan: **gate** — do not invoke if any same-page anchor was **REJECT** or **SUSPICIOUS**.

### Hook C — Bag processing loop

**Location:** ~L927–L945 (`for (const bag of bagMap.bags)`).

```text
priorBagMax = 0   // module-level or outer loop state

for each bag:
  prevTrustedMax = priorBagMax   // seed in-bag chain from previous bag exit
  pageRecords = []

  for each page in bag:
    pageAnalysis = analyzePage(..., { prevTrustedMax, priorBagMax, bag })
    prevTrustedMax = pageAnalysis.prevTrustedMax   // returned updated
    pageRecords.push(...)

  promoteSequenceGapCandidates(pageRecords)      // unchanged — SP3 recovery
  promoteRightColumnExtensions(pageRecords)

  optional: re-validate gap-promoted anchors (SUSPICIOUS → ACCEPT if recovery fills gap)

  priorBagMax = prevTrustedMax   // end-of-bag handoff for SP5

  rejectSequenceOutliers(pageRecords)   // demote or remove post-rollout; forward-context unreliable
```

**Responsibilities:**

- Maintain `priorBagMax` across bags (SP5)
- Seed each bag with previous bag's trusted exit step
- Optional second validator pass after gap promotion for **SUSPICIOUS** → **ACCEPT**

**Baseline alignment:** Bag start pages are trusted (e.g. Bag 13 → p237, Bag 14 → p266). SP5 applies at p237 and p271 within their bags; p271 uses in-bag `prevTrustedMax` from trusted prior pages, not panel noise on p270.

---

## 5. Expected effect on proof pages

Root anchors only. Cascade steps (p186 s346, p189 s352) addressed via Hook B gating.

### p186 — printed 325, detected 345

| Input | Value |
|---|---|
| `prevTrustedMax` | 324 (p185) |
| `priorBagMax` | n/a (mid-bag) |
| `gap` | +21 |

| Output | **REJECT** |
|---|---|
| Rules | SP1, SP2, SP3, SP6 |
| Reason | `implausible_forward_gap` — gap > 6, no recovery evidence |
| Chain | `prevTrustedMax` stays **324** |
| Cascade | Step 346 not corrected from false 345 base (Hook B) |

**Post-validator:** Page may have zero accepted primary steps until OCR retry or gap audit finds **325**. Genuine steps 327+ on later pages can **ACCEPT** against corrected chain once **325–326** exist.

### p189 — printed 331, detected 351

| Input | Value |
|---|---|
| `prevTrustedMax` | ~330 (if p186 rejected and p187–188 genuine steps accepted) |
| `gap` | +21 (true path) or +6 (if chain still corrupted) |

| Output | **REJECT** |
|---|---|
| Rules | SP1, SP2, SP6 (SP3 if gap ≤ 6 on corrupted prior) |
| Reason | `implausible_forward_gap` or fails +1 expectation |
| Chain | Unchanged trusted max |

**Dependency:** Rejecting p186 prevents false prior **345** that would let 351 appear as +6 “plausible.”

### p237 — printed 424, detected 474

| Input | Value |
|---|---|
| `priorBagMax` | **423** (Bag 12 exit, p236) |
| `prevTrustedMax` | 423 (seeded at bag start) |
| `effectivePrior` | 423 |
| `gap` | +50 |

| Output | **REJECT** |
|---|---|
| Rules | SP1, SP2, SP3, SP5, SP6 |
| Reason | `implausible_cross_bag_transition` / `implausible_forward_gap` |
| Chain | Bag 13 trusted max remains **423** until valid **424** accepted |

**Post-validator:** Steps 425–467 on p238+ can **ACCEPT** sequentially once **424** enters chain via OCR retry. No +50 poison on main chain.

### p271 — printed 472, detected 44

| Input | Value |
|---|---|
| `prevTrustedMax` | ~**470** (p269; p270 step 5 rejected by SP4) |
| `gap` | −426 |

| Output | **REJECT** |
|---|---|
| Rules | SP1, SP4, SP6 |
| Reason | `implausible_backwards_transition`; weak OCR (confidence 0.65, component mismatch) |
| Chain | Stays at ~470 |

**Post-validator:** Wrong **44** not in `steps[]`. **472** requires OCR retry (slash interference); validator does not auto-accept printed value.

---

## Summary

| Page | Current | After validator |
|---:|---|---|
| p186 | ACCEPT 345 | **REJECT** 345 |
| p189 | ACCEPT 351 | **REJECT** 351 |
| p237 | ACCEPT 474 | **REJECT** 474 |
| p271 | ACCEPT 44 | **REJECT** 44 |

**Step Map signoff impact:** Addresses primary sequence-parity blockers for Bags 10, 13, 14 per `SETUP_V1_TRUSTED_BASELINE.md` investigation scope. Validator implements missing SP3 gap **rejection**; does not alone complete Step Map signoff (OCR recovery for printed values still required).

---

## References

- `docs/RULEBOOK.md` — SP1–SP6 (`PARITY_RULE`)
- `reports/ANCHOR_CORRECTION_PLAN.md` — hook design and single-validator analysis
- `reports/STEP_ANCHOR_FAILURES.md` — proof-page evidence
- `docs/SETUP_V1_TRUSTED_BASELINE.md` — Stage 4 scope, trust matrix, bag boundaries
- `step_map_scan.mjs` — `analyzePage()`, `correctSamePageSequentialOcr()`, bag loop L927–945
