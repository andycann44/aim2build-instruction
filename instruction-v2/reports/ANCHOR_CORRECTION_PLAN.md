# Anchor Correction Plan

Design-only plan for correcting the four root OCR-anchor failures documented in `STEP_ANCHOR_FAILURES.md`, aligned with `RULEBOOK.md` § Sequence Plausibility Rules (SP1–SP6).

**Date:** 2026-06-16  
**Status:** DESIGN ONLY — no runtime patches, no pipeline rerun  
**Sources:** `reports/STEP_ANCHOR_FAILURES.md`, `docs/RULEBOOK.md`, `step_map_scan.mjs`

---

## Scope

Four **root** anchors. Cascade failures on p186 (step 346) and p189 (step 352) are not separate roots — they are prevented if roots are blocked and `correctSamePageSequentialOcr()` is gated.

| Page | Bag | Printed | Detected | OCR raw |
|---:|---:|---:|---:|---|
| **186** | 10 | 325 | 345 | `345` |
| **189** | 10 | 331 | 351 | `351` |
| **237** | 13 | 424 | 474 | `474` |
| **271** | 14 | 472 | 44 | `44` |

---

## Per-anchor analysis

### p186 — Bag 10, step index 1

| Field | Value |
|---|---|
| **Printed value** | **325** |
| **Detected value** | **345** |
| **Prior trusted step** | 324 (p185, correct) |
| **Transition** | 324 → 345 (gap **+20**) |
| **OCR raw** | `345` (`tesseract_step_box_tight_threshold_6x`) |

**Violated sequence rules**

| Rule | Violation |
|---|---|
| **SP1** | Expected +1 → **325**; accepted **345** breaks sequential expectation |
| **SP2** | Jump +20 > gap-recovery threshold (6); accepted only because OCR value is larger |
| **SP3** | Gap recovery skipped (gap > 6); gap **rejection** never applied |
| **SP6** | OCR `345` accepted without sequence or visual corroboration |
| SP4 | — (forward jump, not backwards) |
| SP5 | — (same bag) |

**Minimum runtime location for correction**

| Location | File / function | Why |
|---|---|---|
| **Primary** | `step_map_scan.mjs` → `analyzePage()` between OCR assign (~L625) and `steps[]` assembly (L644) | Reject anchor before it enters accepted set |
| **Context required** | Bag loop (~L927) must pass `prevTrustedMax = 324` from prior pageRecords | SP1/SP2 need cross-page prior |
| **Secondary (cascade)** | `correctSamePageSequentialOcr()` (L662) — gate or run **after** validator | Prevents p186 step 346 cascade |

**Intended correction:** Reject **345** (`rejection_reason: implausible_page_transition`) or flag for human review. Do **not** promote to `steps[]`. Genuine **325** may require alternate OCR threshold or gap audit — out of scope for sequence-only rejection.

---

### p189 — Bag 10, step index 1

| Field | Value |
|---|---|
| **Printed value** | **331** |
| **Detected value** | **351** |
| **Prior trusted step** | Should be ~328–330 (genuine chain); currently corrupted to **345** if p186 accepted |
| **Transition (if p186 fixed)** | ~330 → 331 (+1 expected) |
| **Transition (current manifest)** | 345 → 351 (+6, at recovery threshold edge) |
| **OCR raw** | `351` |

**Violated sequence rules**

| Rule | Violation |
|---|---|
| **SP1** | Printed sequence expects **331**; detected **351** is +20 from true prior 324→325 path |
| **SP2** | Digit swap inflates step; not validated against expected +1 |
| **SP6** | OCR `351` accepted without independent sequence check |
| SP3 | Partial — gap from false 345→351 is +5 (within recovery window) but **built on false prior** |
| SP4 | — |
| SP5 | — |

**Minimum runtime location for correction**

| Location | File / function | Why |
|---|---|---|
| **Primary** | Same as p186: `analyzePage()` pre-acceptance with `prevTrustedMax` | Must use **trusted** running max, not raw corrupted page max |
| **Dependency** | p186 must be rejected first, or validator uses trusted chain excluding implausible anchors | Otherwise 351 appears plausible relative to false 345 |

**Intended correction:** Reject **351**. With p186 fixed, validator sees prior ~330 and gap +21 → SP2/SP3 fire. Fixing p186 alone may be sufficient; p189 still fails SP6 in isolation (OCR ≠ print) but sequence gate catches the overshoot.

---

### p237 — Bag 13, step index 1

| Field | Value |
|---|---|
| **Printed value** | **424** |
| **Detected value** | **474** |
| **Prior trusted step** | **423** (Bag 12, p236) |
| **Transition** | 423 → 474 (gap **+50**, cross-bag) |
| **OCR raw** | `474` |

**Violated sequence rules**

| Rule | Violation |
|---|---|
| **SP1** | Expected **424** (+1 from 423); accepted **474** |
| **SP2** | Jump +50 far exceeds threshold 6 |
| **SP3** | No recovered intermediates 424–473; rejection missing |
| **SP5** | Bag 13 opens without plausibility check against Bag 12 final step |
| **SP6** | OCR `474` accepted; badge shows **424** |
| SP4 | — |

**Minimum runtime location for correction**

| Location | File / function | Why |
|---|---|---|
| **Primary** | Bag loop (~L927) → before `promoteSequenceGapCandidates()` (L943), with **`priorBagMax = 423`** from previous bag's trusted max | SP5 requires cross-bag context; per-page `analyzePage()` alone lacks Bag 12 handoff |
| **Alternative minimum** | Same validator invoked on first page of bag with `context.priorBagMax` injected at `analyzePage()` call | Equivalent if bag loop maintains cross-bag state |

**Intended correction:** Reject **474**. Accept **424** only after OCR re-read or digit-swap correction — sequence validator alone **rejects** the failure; **correction to printed value** may need OCR retry (see § Limits).

---

### p271 — Bag 14, step index 3

| Field | Value |
|---|---|
| **Printed value** | **472** |
| **Detected value** | **44** |
| **Prior trusted step** | ~**470** (p269); p270 step **5** is spurious panel digit |
| **Transition** | 470 → 44 (Δ **−426**) |
| **OCR raw** | `44` (alt reads `4`, rejected for component mismatch) |

**Violated sequence rules**

| Rule | Violation |
|---|---|
| **SP4** | Large backwards transition; must be noise unless verified |
| **SP6** | Fragment OCR accepted; 2 components vs 3 printed digits; confidence 0.65 |
| **SP1** | Breaks +1 expectation (470 → 471/472) |
| SP2 | — (backwards, not overshoot) |
| SP3 | — |
| SP5 | — (same bag, but needs trusted prior not p270 noise) |

**Minimum runtime location for correction**

| Location | File / function | Why |
|---|---|---|
| **Primary** | `analyzePage()` pre-acceptance (L644) with `prevTrustedMax ≈ 470` | SP4 backwards check at accept time |
| **Signal helper** | Same validator: `components` (2) vs OCR digit count / `step_number_rejected_reads` | SP6 — weak OCR evidence |
| **Prior selection** | Validator must use **trusted chain max**, not p270 min step **5** | Raw page max would wrong-foot context |

**Intended correction:** Reject **44**. Does not auto-produce **472** — requires OCR retry or human review (§ Limits).

---

## Violated rules summary

| Page | SP1 | SP2 | SP3 | SP4 | SP5 | SP6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| p186 | ✓ | ✓ | ✓ | | | ✓ |
| p189 | ✓ | ✓ | partial | | | ✓ |
| p237 | ✓ | ✓ | ✓ | | ✓ | ✓ |
| p271 | ✓ | | | ✓ | | ✓ |

---

## Proposed component: Sequence Plausibility Validator

Single function, multiple call sites. Design sketch only.

```text
validateSequencePlausibility(anchor, context) → { accept | reject | flag, reason, signals }

context:
  prevTrustedMax      // last accepted global step (page order, validated only)
  priorBagMax         // last accepted step in previous bag (for SP5)
  gapRecoveryThreshold // 6 (existing promoteSequenceGapCandidates constant)
  recoveredSteps      // set of steps promoted via sequence_gap_full_page_audit
  pageStepsSoFar      // same-page prior accepted steps (for +1 check)
```

**Checks (maps to SP1–SP6)**

| Check | Rules | Logic |
|---|---|---|
| Backwards | SP4 | `anchor.step_number < prevTrustedMax` → reject |
| Large forward gap | SP1, SP2, SP3 | `gap = anchor.step_number - prevTrustedMax`; if `gap > 6` and no recovered intermediates → reject |
| Cross-bag gap | SP5 | On first page of bag: compare to `priorBagMax` same as above |
| Same-page +1 | SP1 | Optional soft check: second anchor on page should be `priorOnPage + 1` unless gap-audit evidence |
| OCR evidence weak | SP6 | Low confidence, component/digit mismatch, rejected_reads → reject or flag |
| OCR vs sequence | SP6 | Do **not** auto-correct OCR to satisfy sequence (invert `correctSamePageSequentialOcr` priority) |

---

## Minimum insertion points in current pipeline

Current Stage 4 order (`step_map_scan.mjs`):

```text
analyzePage()           ← OCR per box
  → filterInvalidStepNumber()
  → steps[] assembly
  → correctSamePageSequentialOcr()   ← CASCADE RISK
applyV1StepAnchorGuardrails()
promoteSequenceGapCandidates()       ← gap recovery only
promoteRightColumnExtensions()
rejectSequenceOutliers()             ← too late; forward-context pollution
steps.push → manifest
```

**Recommended insertion (design)**

```text
analyzePage()
  → OCR
  → validateSequencePlausibility(anchor, context)   ← NEW (per anchor)
  → steps[] assembly (only if accept)
  → [DISABLE or gate] correctSamePageSequentialOcr()

Bag loop:
  → maintain prevTrustedMax / priorBagMax across pages and bags
  → after all pages: promoteSequenceGapCandidates()  (unchanged)
  → optional second pass: validateSequencePlausibility on promoted gap fills
```

| Call site | Prevents |
|---|---|
| Per-anchor in `analyzePage()` before L644 | p186, p189, p271 acceptance |
| Cross-bag `priorBagMax` in bag loop | p237 |
| Gate `correctSamePageSequentialOcr()` | p186 s346, p189 s352 cascades |
| Retire or demote `rejectSequenceOutliers()` forward-context logic | p186, p237 correlated false validation |

---

## Question: Can one validator after OCR and before acceptance prevent all four?

### Short answer

**Yes — for prevention of wrong acceptance.**  
**No — for automatic correction to printed values without additional OCR.**

All four root failures are **acceptance failures**: wrong OCR values entered `steps[]` with `rejection_reason: null`. A single **Sequence Plausibility Validator** invoked **after OCR and before step acceptance**, with the right **context**, would reject all four.

### Conditions required

The validator must be **one function** but **not one call in isolation**. It needs:

1. **Rolling `prevTrustedMax`** — only validated anchors update it (SP1, SP2, SP3, SP4)
2. **`priorBagMax` on bag boundaries** (SP5) — p237 cannot be caught with per-page context alone
3. **Trusted prior, not raw page max** — p271 requires ignoring p270 spurious step **5** (SP4, SP6)
4. **`correctSamePageSequentialOcr()` gated or removed** — otherwise cascades occur after validator (p186 s346, p189 s352)
5. **No forward-context validation** — current `rejectSequenceOutliers()` pattern must not be reused; prior-only or trusted-chain-only

### Per-failure verdict

| Page | Prevented by single validator? | Mechanism |
|---|---|---|
| **p186** | **Yes** | `345 - 324 = 21 > 6`, no gap recovery → reject (SP2, SP3) |
| **p189** | **Yes** | With trusted prior: `351` overshoots expected ~331 path (SP2, SP6); even if prior corrupted, fixing p186 unblocks this |
| **p237** | **Yes** | `474 - 423 = 50 > 6`, cross-bag (SP5, SP3) |
| **p271** | **Yes** | `44 < 470` backwards (SP4); weak OCR signals (SP6) |

### What the validator does **not** do alone

| Gap | Limit |
|---|---|
| **Correction to printed value** | Validator rejects **345**; does not produce **325** without OCR re-read, alternate threshold, or V1 truth |
| **p237 424** | Rejecting **474** leaves a missing step until OCR succeeds on retry |
| **p271 472** | Rejecting **44** is correct; recovering **472** needs better OCR on slash-interference badge |
| **Gap fill** | Legitimate gaps 1–6 still need existing `promoteSequenceGapCandidates()` **after** validator accepts boundary anchors |

### Design conclusion

| | |
|---|---|
| **Single validator component?** | Yes — one `validateSequencePlausibility()` implementing SP1–SP6 |
| **Single call site?** | No — per-anchor call in `analyzePage()` **plus** cross-bag state in bag loop |
| **After OCR, before acceptance?** | Yes — that placement prevents all four root failures |
| **Sufficient for parity signoff?** | Rejection yes; full parity also needs OCR retry path or V1 truth for rejected anchors (BAG3 parity: OCR correction against truth) |

---

## Implementation order (when patching — not now)

1. Add validator + `prevTrustedMax` / `priorBagMax` state in bag loop
2. Gate `correctSamePageSequentialOcr()` — never override OCR that passes plausibility; never correct toward wrong prior
3. Wire rejection into `rejection_reason` + `signals.sequence_plausibility_rejected`
4. Proof pages: **186, 189, 237, 271** must reject detected values; **325, 331, 424, 472** remain acceptance targets via OCR improvement or truth import
5. Regression: Bags 1–3 gap recovery (gaps ≤6) must still promote via `promoteSequenceGapCandidates()`

---

## References

- `reports/STEP_ANCHOR_FAILURES.md` — root failure evidence
- `docs/RULEBOOK.md` § Sequence Plausibility Rules (SP1–SP6)
- `step_map_scan.mjs` — `analyzePage()`, `correctSamePageSequentialOcr()`, `promoteSequenceGapCandidates()`, `rejectSequenceOutliers()`
- `reports/STEP_SEQUENCE_AUDIT.md` — main-chain impact analysis
- `reports/STEP_PARITY_REVIEW_PACK.md` — badge crops and visual verification
