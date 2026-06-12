# ADR-001: V2 Parity Gate Before Expansion

**Status:** Accepted  
**Date:** 2026-06-03  
**Deciders:** Andy Cannell (aim2build)

---

## Context

`instruction-v2` is a local-manifest-first rewrite of the V1 LEGO instruction processing pipeline. V1 is a proven, human-validated Python/FastAPI system (`clean/`) that processes a PDF instruction booklet through bag detection, step detection, callout crop extraction, quantity OCR, part segmentation, and CLIP-based matching, with human review at each stage. V1 has accumulated a validated corpus of training labels, crop cache, bag truth, and confirmed matches for set `70618`.

V2 replicates each stage as an immutable JSON manifest written by standalone Python/Node scripts. It does not import V1 modules; it copies or adapts equivalent logic. The goal is a reproducible, auditable pipeline that can be run offline and extended without touching the live V1 server.

At the point this decision was recorded, all 11 pipeline stages (Stage 0 through Stage 11) were implemented and the pipeline validator reported PASS for Stages 0–8. Stages 9–11 (matching, audit, manual config) were structurally present but acknowledged as weak.

### The Bag System

The instruction set is divided into numbered bags. Each bag covers a contiguous page range in the PDF, contains a sequence of printed build steps, and has a set of associated callout crops. The bag structure is the primary unit of all downstream processing. Getting bag boundaries, step numbers, crop boxes, quantity OCR, and segmentation right for one bag before moving to the next is the only way to validate that V2 is faithful to V1 truth.

Bags 1 and 2 were chosen as the initial parity gates because:

- They are the first bags in the set (pages 6–21 and 22–38 respectively).
- V1 has stored, reviewed crop cache and training labels for both (`debug/crop_cache/70618_bag1.json`, `debug/crop_cache/70618_bag2.json`, `debug/training_labels/70618_bag1.json`, `debug/training_labels/70618_bag2.json`).
- Bag 1 covers a known difficult case: Step 8 on page 10, which has a callout panel that sits close to the printed step number anchor and is not in the V1 crop cache.
- Bag 2 adds a second data point and validates that step number continuity (steps 26–40) and crop IDs transfer correctly across a bag boundary.

---

## Problem

V2 cannot be trusted to process Bag 3 and beyond if:

1. Bag boundaries are wrong. A single missing or misplaced bag start changes every downstream page range, step assignment, crop ID, and segmentation entry for all subsequent bags.
2. Step numbers are synthetic or wrong. V1 uses actual printed step numbers as stable identifiers. V2 was initially producing local visual order (`step_index`) and `null` for `step_number` in many cases. This breaks crop ID traceability and review matching against V1 ground truth.
3. Callout crop boxes miss valid crops. A crop missed in Stage 6 produces no OCR entry, no segmentation entry, and no match candidate. The bag parity checks are the only structured gate that would catch a systematically missed crop before it propagates to all remaining bags.
4. Parity checking against V1 labels becomes harder as bag count grows. Bag 1 and 2 together cover 37 pages, 40 steps, and 37 callout crops — a tractable verification surface. Bags 3–11 add hundreds more. Validating parity retrospectively across all bags at once is impractical.

---

## Options Considered

### Option A: Full pipeline run first, then parity audit all bags at the end

Run V2 end-to-end for all bags, then compare V2 outputs against V1 training labels and crop cache in a single audit pass.

| Dimension | Assessment |
|---|---|
| Defect detection | Late — errors in early bags pollute all later bags |
| Review cost | High — a missed crop or wrong step number in bag 1 requires re-running all downstream stages |
| Alignment with V1 review discipline | Low — V1 required human sign-off per bag before proceeding |
| Risk | High — matching work could be started on corrupted foundations |

**Pros:** Faster to reach an end-to-end run result.  
**Cons:** Errors compound silently. Matching quality work would begin before crop/segmentation correctness is confirmed.

### Option B: Sequential bag parity gates before expansion (chosen)

For each bag, run a structured parity check comparing V2 manifests against V1 crop cache, training labels, step numbers, and bag truth before proceeding to the next bag. Gate all matching improvement work behind Bag 1 and Bag 2 parity sign-off. Gate Bag 3+ work behind the same sequential gate.

| Dimension | Assessment |
|---|---|
| Defect detection | Early — errors are caught and fixed within the bag they originate in |
| Review cost | Low per bug — each fix is isolated to one bag's manifests |
| Alignment with V1 review discipline | High — mirrors V1's bag-by-bag human review model |
| Risk | Low — matching and segmentation work begin only on verified foundations |

**Pros:** Errors are isolated. Parity sign-off documents are permanent evidence. V1 ground truth is consumed bag-by-bag in the same order it was produced.  
**Cons:** Expansion is gated; Bag 3+ cannot begin until Bag 1 and 2 pass.

---

## Decision

**Option B was adopted.** Bag 1 and Bag 2 were used as sequential parity gates. All matching improvement work and all Bag 3+ processing are gated behind both parity sign-offs.

The parity gate enforces the following invariants before any bag is marked approved:

1. Bag start and end pages match `bag_truth.db` source.
2. All V1 crop-cache crop IDs are present in V2 with no missing entries.
3. Step number sequence matches V1 expected range exactly (no extra, no missing steps).
4. All reviewed training-label crop records resolve to a V2 crop entry.
5. No false-positive steps exist within the bag's page range.
6. No missing callout crops for steps that have a visible callout panel.
7. No quantity mismatches for crops where V1 qty is known.

---

## Failures Found and Fixed

### Bag 1 Parity Failures

**Failure 1 — Step 8 missing callout crop (Stage 6 regression)**

Step 8 on page 10 was present in `05_step_map.json` but absent from `06_callout_crop_box_map.json`. It was not in the V1 crop cache, so the Stage 6 V1-import path did not cover it. The fresh V2 callout detector (`callout_crop_box_scan.mjs::detectCalloutRectByEdges()`) rejected the only candidate component because of the rule:

```js
if (pageBottom > stepY - 5) continue;
```

For Step 8, `pageBottom = 795` and `stepY = 795`, so `795 > 790` was true and the crop was dropped. The callout panel was visually valid with a `4x` quantity label.

**Fix:** A `clipped_left_callout_panel_slack` geometry rule was introduced to handle callout panels whose bottom edge is close to or coincides with the step anchor y-coordinate. After the fix, the Bag 1 parity check for Step 8 passed:

```
Step 8 qty entries: [{'qty_text': ['4x'], 'qty_numbers': [4], 'crop_id': 'bag_01_page_010_step_8_015'}]
```

**Failure 2 — Printed step numbers missing (Stage 5 regression)**

`step_number` was `null` for most steps early in development. V2 was tracking local visual detection order (`step_index`) but not reading the printed yellow step numbers from the PDF pages. This was addressed by adapting V1 `step_detector_service.py` OCR logic into `step_map_scan.mjs` so that `step_number` reflects the actual printed value.

### Bag 2 Parity

Bag 2 passed all checks after the Stage 5/6 fixes applied during Bag 1 remediation. All 15 V1 crop-cache entries were present, step numbers 26–40 matched exactly, and no missing callouts or quantity mismatches were found.

### Bag 3 Parity Failure

**Failure — OCR misread step 18 as 650 / step 78 as 18 (Stage 5 OCR error)**

On page 50, the printed step number `60` was misread as `650` by the initial OCR implementation. On page 57, the printed step number `78` was misread as `18`. These are Bag 3 steps that would have entered downstream stages with wrong identifiers, breaking crop IDs and traceability.

**Fix:** V1 training label ground truth was used to apply step-number corrections as `parity_override_source: v1_training_labels_step_correction` entries in the Stage 6 manifest. The corrected crop IDs (`p50_s60_c4`, `p57_s78_c1`) were written with explicit `corrected_from_step` fields. The false step 18 was removed from Bag 3 step/callout/qty manifests; the false step 650 was also removed.

---

## Why Parity Is Required Before Bag 3+

The V1 Parity Plan (`V1_PARITY_PLAN.md`) established a mandatory ordering constraint:

> Matching must not be tuned before crop, OCR, and segmentation parity are proven.

The rationale is:

1. **Cascading dependency.** Each stage reads only manifests written by earlier stages. A corrupt bag boundary in `04_bag_map.json` silently propagates wrong page ranges into `05_step_map.json`, then wrong step anchors into `06_callout_crop_box_map.json`, then wrong crops into `07_qty_ocr_map.json` and `08_part_segmentation_map.json`, and finally wrong segment cutouts into `09_match_manifest.json`. The error is invisible unless a per-bag parity gate is enforced.
2. **Matching signal is meaningless on bad crops.** The provisional RGB-only matching in Stage 9 is already weak. Running it on miscropped or misidentified callouts produces noise that would mislead any future matching quality work.
3. **V1 training labels are bag-scoped.** The V1 ground truth (`debug/training_labels/70618_bag*.json`) is structured per bag. Consuming it sequentially is the natural way to surface differences between V1 and V2 logic one bag at a time.
4. **Human review remains authoritative.** The parity gate requires that all human-reviewed V1 crops resolve to V2 entries. This preserves the authority of V1 human decisions within V2 without importing V1 modules or overwriting V2 manifests with V1 data silently.

---

## Consequences

**What becomes easier:**
- Matching improvement work begins on verified crop/segmentation foundations.
- V2 bugs discovered in later bags are traceable to a specific stage and bag range.
- New bags can be processed incrementally with a documented pass/fail history per bag.
- Any regression introduced by a code change is immediately detectable by re-running the parity check for the affected bag.

**What becomes harder:**
- Bag 3+ cannot begin until Bag 1 and Bag 2 sign-offs are complete. This is intentional.
- A new developer must understand the V1 crop cache and training label schemas before diagnosing a parity failure.
- The sign-off process requires reading V1 debug files that are not part of the V2 manifest tree.

**What we will need to revisit:**
- Bags 4–11 do not yet have sign-offs. The same process must be applied sequentially.
- The pending gap window between pages 105–163 is unresolved and will require human review before the bags in that range can be signed off.
- Stage 7 (segmentation) parity is not yet proven against V1 slot-aware logic (`ai_snap_crop_service.py`, `part_crop_normalize_service.py`). Segmentation parity should be a gate within each bag sign-off before matching quality work proceeds.

---

## Verification / Regression Tests

The following checks constitute the regression suite for any future re-run of Bags 1 and 2. They are machine-verifiable from the current manifests.

### Bag 1 Regression Checks

| Check | Source Manifests | Expected Result |
|---|---|---|
| Bag 1 start page = 6 | `04_bag_map.json`, `bag_truth.db` | `start_page: 6, source: v1_bag_truth_db` |
| Pages 6–21 all present | `02_page_index.json` | `missing_pages: []` |
| All 22 V1 crop-cache IDs present in Stage 6 | `06_callout_crop_box_map.json`, `debug/crop_cache/70618_bag1.json` | `missing: []` |
| Step numbers = [1..25] | `05_step_map.json` | `extra: [], missing_expected: []` |
| Step 65 absent | `05_step_map.json` | `step65_absent: true` |
| Step 8 present with crop | `06_callout_crop_box_map.json` | `crop_id: bag_01_page_010_step_8_015` |
| Step 8 qty = 4x | `07_qty_ocr_map.json` | `qty_numbers: [4], confidence ≥ 0.96` |
| All 22 reviewed training-label crop records resolve | `debug/training_labels/70618_bag1.json`, `06_callout_crop_box_map.json` | `resolved: 22, unresolved: 0` |

### Bag 2 Regression Checks

| Check | Source Manifests | Expected Result |
|---|---|---|
| Bag 2 start page = 22, end page = 38 | `04_bag_map.json`, `bag_truth.db` | `start_page: 22, end_page: 38, source: v1_bag_truth_db` |
| Pages 22–38 all present | `02_page_index.json` | `missing_pages: []` |
| All 15 V1 crop-cache IDs present in Stage 6 | `06_callout_crop_box_map.json`, `debug/crop_cache/70618_bag2.json` | `missing: []` |
| Step numbers = [26..40] | `05_step_map.json` | `unexpected: []` |
| All 13 reviewed training-label crop records resolve | `debug/training_labels/70618_bag2.json`, `06_callout_crop_box_map.json` | `resolved: 13, unresolved: 0` |
| No false-positive steps | `05_step_map.json` | `false_positive_count: 0` |
| No missing callouts | `06_callout_crop_box_map.json` | `missing_callouts: []` |
| No qty mismatches | `07_qty_ocr_map.json` | `qty_mismatches: []` |

### Bag 3 Regression Checks (partial — Blocker 3 only)

| Check | Source Manifests | Expected Result |
|---|---|---|
| Step 18 absent from Bag 3 | `05_step_map.json` | `bag3_step_numbers_contain_18: false` |
| Step 650 absent from Bag 3 | `05_step_map.json` | `bag3_step_numbers_contain_650: false` |
| Step 78 present on page 57 | `05_step_map.json`, `06_callout_crop_box_map.json` | `page57_step_map: [78]` |
| Step 60 present on page 50 | `05_step_map.json`, `06_callout_crop_box_map.json` | `crop_id: p50_s60_c4` |
| p57_s78_c1 qty = 8x | `07_qty_ocr_map.json` | `qty_numbers: [8]` |

---

## Evidence Artifacts

| Artifact | Location |
|---|---|
| Bag 1 parity sign-off | `BAG1_PARITY_SIGNOFF.md` |
| Bag 1 visual review | `BAG1_REVIEW.md` |
| Bag 2 parity review | `BAG2_PARITY_REVIEW.md` |
| Bag 2 parity sign-off | `BAG2_PARITY_SIGNOFF.md` |
| Bag 3 blocker 3 sign-off | `BAG3_PARITY_SIGNOFF.md` |
| Step 8 missing callout audit | `STEP_8_MISSING_CALLOUT_AUDIT.md` |
| Step 8 Stage 6 failure root cause | `STEP_8_STAGE6_FAILURE_AUDIT.md` |
| V1 parity plan | `V1_PARITY_PLAN.md` |
| Pipeline checkpoint 2026-06-02 | `CHECKPOINT_2026-06-02.md` |
| Behaviour audit | `BEHAVIOUR_AUDIT.md` |
| Pipeline validator | `instruction-v2/validate_pipeline.py` |

---

## Proposed Next ADRs

The following five ADRs are recommended as the next records for this project, in priority order:

---

### ADR-002: Manifest-First, Immutable-Stage Pipeline Architecture

**Why it should exist:** Every V2 stage writes exactly one JSON manifest and does not modify manifests owned by other stages. This is a foundational architectural choice that is currently implicit across `SOURCE_OF_TRUTH.md` and the stage scripts, but is not documented as a decision. The ADR should record why immutability was chosen over a stateful database or V1's mutable server model, what the authority rules are (`11_manual_match_config.json` as the only human-writable manifest), and what the consequences are for stage re-runs and rollback.

---

### ADR-003: V1 Crop Cache and Training Labels as Ground Truth for Parity Verification

**Why it should exist:** V2 uses `debug/crop_cache/70618_bag*.json` and `debug/training_labels/70618_bag*.json` as its ground truth reference for parity checks. These files are V1 artefacts, not V2 manifests. The ADR should record why they are authoritative (human-reviewed, accumulated over V1 production), what their schema means (crop IDs, page/step references, elem_* exclusions), when they can be used to override V2-detected values (step correction, missing crop recovery), and when they cannot (autonomous label writes, auto-acceptance of matches).

---

### ADR-004: Bag Gap Review as a Human-in-the-Loop Gate

**Why it should exist:** Stage 3b (`03b_bag_gap_review.json`) was introduced because the automated bag candidate scanner (`bag_candidate_scan.mjs`) missed page 81 as the start of Bag 5. The human gap review stage is the only place in the pipeline where a pending human decision can block downstream manifest generation. The ADR should record why this gate exists, how the gap window is defined (page range between last known bag end and next candidate), what evidence is presented (scored candidate pages, contact sheet), and why the human decision must not be overwritten by re-running the scanner.

---

### ADR-005: No Autonomous Label Writes — AI Recommendations Are Derived-Only

**Why it should exist:** `SOURCE_OF_TRUTH.md` and `V1_PARITY_PLAN.md` both state that AI recommendations must never auto-save labels. The planned `12_ai_override_recommendations.json` is explicitly derived-only. The V1 system (`clean/routers/instruction_debug.py::save_label`) required explicit human action to write training labels. The ADR should record this as a firm constraint: no pipeline stage, AI service, or automated process may write to `debug/training_labels/` or promote a provisional match to a confirmed label without explicit human review and config entry in `11_manual_match_config.json`.

---

### ADR-006: Slot-Aware Segmentation Required Before Matching Quality Work

**Why it should exist:** V2's current Stage 8 (`stage7_part_segmentation.py`) uses whole-callout connected-component segmentation — it extracts the largest foreground regions from the callout crop image. V1 uses qty-anchor-driven, slot-aware segmentation (`ai_snap_crop_service.py`, `part_crop_normalize_service.py`) where each slot is owned by a specific qty anchor row, with fallback strategies for repeated-qty rows and ambiguous components. The ADR should record that matching quality work (Stage 9) must not be advanced until Stage 8 segmentation produces slot-aware outputs, and document the V1 logic components that must be adapted (row grouping, qty-anchor ownership, master-island fallback, `needs_review` slot status).
