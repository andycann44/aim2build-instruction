# Rulebook — Master Inventory from V1 Parity Work

Documentation only. Rules listed here are **extracted from existing parity documents**; nothing in this file changes runtime behaviour.

## Scope and sources

| Source | Role |
|---|---|
| `V1_BEHAVIOUR_CONTRACTS.md` | Stage contracts (bag, step, crop, qty, segmentation) |
| `V1_CURRENT_VS_OLD_GUARDRAIL_AUDIT.md` | Named V1 guardrail functions to port |
| `V1_STEP_SOURCE_AUDIT.md` | V1 step/crop truth chain, anchor filter, fallback |
| `V1_PARITY_PLAN.md` | Stage parity requirements and proof checklist |
| `V1_RECOVERY_PORT_PLAN.md` | Port rules, recovery matrix, do-not-approximate constraints |
| `BAG1_PARITY_SIGNOFF.md` | Bag 1 approved checks |
| `BAG2_PARITY_SIGNOFF.md` | Bag 2 approved checks |
| `BAG2_PARITY_REVIEW.md` | Bag 2 extended review checks |
| `BAG3_PARITY_SIGNOFF.md` | Bag 3 approved checks + sequence-gap audit rule |
| `BAG3_SEQUENCE_GAP_PARITY_AUDIT.md` | Gap-row classifications |
| `BAG3_STEP_NUMBER_STATISTICS_AUDIT.md` | Bag 3 visual size cluster audit |
| `BAG4_PARITY_SIGNOFF.md` | Bag 4 in-progress checks, human-review exceptions |
| `BEHAVIOUR_AUDIT.md` | V2 stage weaknesses and review authority notes |

**Not used:** `docs/parity/BAG4_PARITY_SIGNOFF.md` (empty file).

**V2 active status legend:** `pass` = parity signoff passed; `partial` = contract says incomplete; `audit-only` = evidence rule, not detection redesign; `policy` = process gate; `fail` = signoff not met.

---

## STEP RULES

### Visual cluster — same font size

| Rule | Purpose | Source | V2 status | Evidence |
|---|---|---|---|---|
| **Accepted step height cluster** | Valid Bag 3 global steps share a tight height band | `BAG3_STEP_NUMBER_STATISTICS_AUDIT.md` §Median dimensions | Audit observation (Bag 3) | Median height **36.0 px**; low/high band **25.2–46.8 px** (median ±30%). All 35 accepted Bag 3 steps fell inside band. |
| **Accepted step width cluster** | Valid multi-digit global steps share a tight width band | `BAG3_STEP_NUMBER_STATISTICS_AUDIT.md` §Median dimensions | Audit observation (Bag 3) | Median width **54.0 px**; band **37.8–70.2 px**. Accepted two-digit steps cluster at ~51–55 px wide. |
| **Pre-OCR size gate (proposed)** | Exclude OCR-noise regions before accepting anchors | `BAG3_STEP_NUMBER_STATISTICS_AUDIT.md` §Interpretation | Not implemented as formal rule | "A pre-OCR size gate using median ±30% would retain most accepted steps while excluding typical OCR-noise regions." |

**NOT DOCUMENTED** as an explicit V1 contract rule for "same font size" outside the Bag 3 statistics audit.

---

### Visual cluster — same baseline

| Rule | Purpose | Source | V2 status | Evidence |
|---|---|---|---|---|
| — | — | — | — | **NOT DOCUMENTED** in listed parity sources as an explicit baseline-alignment rule. |

---

### Visual cluster — same spacing (digit gap)

| Rule | Purpose | Source | V2 status | Evidence |
|---|---|---|---|---|
| **Multi-digit component count** | Global LEGO step numbers are usually 2+ connected digit components | `BAG3_STEP_NUMBER_STATISTICS_AUDIT.md` §Cluster summary | Audit observation (Bag 3) | Median **component count = 2.0**; band **1.4–2.6**. All 35 accepted Bag 3 steps: components = 2 (except none below band). |
| **Single-component noise contrast** | Rejected audit candidates are typically single narrow digits | `BAG3_STEP_NUMBER_STATISTICS_AUDIT.md` §Rejected audit candidates, §Interpretation | Audit observation (Bag 3) | Rejected candidates often **14–28 px wide, 1 component** vs accepted cluster **~51–55 px, 2 components**. |

**NOT DOCUMENTED** as an explicit inter-digit gap threshold rule in parity contracts.

---

### Visual cluster — same colour/style

| Rule | Purpose | Source | V2 status | Evidence |
|---|---|---|---|---|
| — | — | — | — | **NOT DOCUMENTED** in listed parity sources as a colour/style matching rule for step anchors. |

---

### Dominant page step

| Rule | Purpose | Source | V2 status | Evidence |
|---|---|---|---|---|
| **V1 `main_steps` vs `sub_steps` distinction** | V1 detector separates primary page steps from subordinate digit tokens | `V1_STEP_SOURCE_AUDIT.md` §Step detector | Partial — V2 uses separate `step_map_scan.mjs` | `detect_steps` output includes `main_steps`, `sub_steps`, `step_candidates`, `classified_step_boxes`. |
| **Printed step is authoritative, not local order** | `step_number` must be the printed LEGO step, not detection order | `V1_BEHAVIOUR_CONTRACTS.md` §2; `V1_PARITY_PLAN.md` §Stage 4 | Partial | "`step_number` is the actual printed LEGO instruction step number or `null`. No synthetic numbering." |
| **Separate `page`, `step_index`, `step_number`** | Preserve traceability; `step_index` is local order only | `V1_BEHAVIOUR_CONTRACTS.md` §2; `V1_RECOVERY_PORT_PLAN.md` §Step Detection And Anchors | Yes (approved bags) | "V2 must not use `step_index` as a fallback `step_number`." |

---

### Sequence plausibility

| Rule | Purpose | Source | V2 status | Evidence |
|---|---|---|---|---|
| **No synthetic numbering** | Never invent step numbers from index or gap fill | `V1_BEHAVIOUR_CONTRACTS.md` §2; `V1_RECOVERY_PORT_PLAN.md` | Partial | "No synthetic numbering is allowed." |
| **Reject impossible anchors** | Filter OCR garbage (e.g. 4-digit numbers) | `V1_BEHAVIOUR_CONTRACTS.md` §2; `V1_STEP_SOURCE_AUDIT.md` §Step anchor filter | Partial | Keeps `0 < step_number < 1000`; rejects anchors like `1583`. Applied via `_filter_invalid_step_anchor_boxes`. |
| **Bag expected step sequence** | Accepted steps must match V1 crop-cache / truth sequence for the bag | `BAG1_PARITY_SIGNOFF.md` check 4; `BAG2_PARITY_SIGNOFF.md` check 5; `BAG2_PARITY_REVIEW.md` check 4 | Pass Bags 1–2 | Bag 1: steps 1–25, no extra/missing. Bag 2: steps 26–40, `unexpected=[]`. |
| **No unexpected step explosions** | Catch runaway OCR / false high step numbers | `BAG2_PARITY_SIGNOFF.md` check 5 | Pass Bag 2 | `false_positive_steps=[]`. |
| **Step 65 absent (Bag 1)** | Bag-specific negative check | `BAG1_PARITY_SIGNOFF.md` check 5 | Pass Bag 1 | `step65_absent=True`. |
| **Out-of-sequence = likely false positive** | Extra semantic callouts outside bag truth are OCR errors | `BAG4_PARITY_SIGNOFF.md` §Extra Semantic Callouts | Signoff classification (Bag 4) | e.g. page 70 step 11: "Out of Bag 4 sequence; likely OCR/false positive." |
| **Sequence-Gap Full-Page Audit** | On step-number jump, audit entire page for missing opposite-column steps | `BAG3_PARITY_SIGNOFF.md` §Bag 3 Parity Rule; `BAG4_PARITY_SIGNOFF.md` §Next Work | Audit-only | "If V2 detects a step-number jump, the audit/fallback must inspect the full page, not only the area near detected anchors." |
| **Sequence-gap audit scope** | Define what full-page audit must check | `BAG3_PARITY_SIGNOFF.md` | Audit-only | Check left and right columns; top and bottom callout zones; compare against V1 truth boxes. |
| **Restore only confirmed V1 truth on gap audit** | Do not invent crops during gap repair | `BAG3_PARITY_SIGNOFF.md`; `BAG4_PARITY_SIGNOFF.md` | Audit-only | "restore exact V1 crops only when visually/evidentially confirmed"; preserve `crop_id`, `page`, `step`, `crop_box`, `qty`. |
| **Gap row classifications** | Distinguish shared vs V2-only sequence failures | `BAG3_SEQUENCE_GAP_PARITY_AUDIT.md` | Diagnostic | `EMPTY_STEP`: both pipelines detect step, no callout. `V2_FAILURE`: V2 gap audit found anchor V1 missed, still no callout. |
| **OCR step correction against V1 truth** | Fix misread printed numbers to match V1 | `BAG3_PARITY_SIGNOFF.md` (page 57); `BAG4_PARITY_SIGNOFF.md` (page 65) | Applied where documented | Page 57: printed `78`, OCR `18`. Page 65: OCR `6` → V1 step `96`. |
| **Do not approximate step detection** | V2 must port or copy exact V1 `detect_steps` | `V1_RECOVERY_PORT_PLAN.md` §Port Rules | Policy | "Do not approximate step detection." |

---

### Substep rejection

| Rule | Purpose | Source | V2 status | Evidence |
|---|---|---|---|---|
| **V1 `sub_steps` output channel** | V1 detector emits subordinate step tokens separately from `main_steps` | `V1_STEP_SOURCE_AUDIT.md` §Step detector | Partial | Output fields include `sub_steps` alongside `main_steps`. |
| **Reject impossible anchor boxes** | `_filter_invalid_step_anchor_boxes` removes invalid OCR anchors | `V1_CURRENT_VS_OLD_GUARDRAIL_AUDIT.md` §Step Guardrails; `V1_STEP_SOURCE_AUDIT.md` | Partial | "port now" recommendation for `_filter_invalid_step_anchor_boxes`. |
| **Invalid anchors → null or rejected** | Record why a candidate was rejected | `V1_BEHAVIOUR_CONTRACTS.md` §2; `V1_RECOVERY_PORT_PLAN.md` | Partial | "Invalid anchors must become `null` or be rejected with a reason." Manifest must carry `rejection_reason`. |
| **False-positive step list empty** | No spurious steps in approved bag range | `BAG2_PARITY_SIGNOFF.md` check 8; `BAG2_PARITY_REVIEW.md` check 5 | Pass Bag 2 | `false_positive_steps=[]`, `false_positive_count=0`. |
| **Single-component narrow digits = noise (audit contrast)** | Rejected full-page audit candidates resemble substeps/noise, not global steps | `BAG3_STEP_NUMBER_STATISTICS_AUDIT.md` §Rejected audit candidates | Audit observation | Rejected candidates: values like `7`, `2`, `3`, `4` at **10–28 px wide, 1 component**; not promoted to `steps[]`. |

**NOT DOCUMENTED** as an explicit rule named "reject local callout numbers 1/2/3 as substeps" in listed parity contracts. Closest documented evidence is V1 `sub_steps` channel and Bag 3 rejected-candidate size/component contrast.

---

### Other documented step rules

| Rule | Purpose | Source | V2 status | Evidence |
|---|---|---|---|---|
| **Printed step detection contract** | Stage 4 must read actual printed numbers | `V1_BEHAVIOUR_CONTRACTS.md` §2; `V1_PARITY_PLAN.md` §Stage 4 | Partial | Proof pages: 12 (11,12), 23 (27), impossible anchors rejected. |
| **Page-level fallback step recovery** | Recover steps when raw detector misses (e.g. page 7) | `V1_STEP_SOURCE_AUDIT.md` §Page-level fallback | Partial | `_detect_page_step_number_boxes`, `_detect_step_number_below_panel`, `_page_level_callout_candidates_for_fallback`. Page 7 crops from `page_level_callout_assignment`. |
| **Right-half gap recovery OCR** | Recover missing right-column steps during crop build | `V1_STEP_SOURCE_AUDIT.md` §Crop builder | Partial | "Optionally recovers missing right-half steps with local OCR." |
| **`crop_id` format** | Stable identifiers: `p{page}_s{step}_c{idx}` | `V1_STEP_SOURCE_AUDIT.md` §Crop builder | Pass (approved bags) | `crop_id = f"p{page}_s{step_number}_c{idx}"`. |
| **V1 crop cache import before regenerate** | Persisted V1 crop truth is authoritative for proven pages | `V1_STEP_SOURCE_AUDIT.md` §Recommendation; `V1_RECOVERY_PORT_PLAN.md` | Yes Bags 1–3 | "Do not port raw step detection as the next recovery item"; import `debug/crop_cache/{set_num}_bag{bag}.json` first. |
| **Page 81 is bag start, not step proof** | Bag-start pages are out of step-detection scope | `V1_STEP_SOURCE_AUDIT.md` §Page 81 | N/A | "Page 81 is a bag start, not a step/crop proof page." |
| **V1 step guardrails to port** | Named functions for printed-number recovery | `V1_CURRENT_VS_OLD_GUARDRAIL_AUDIT.md` §Step Guardrails | Not exact parity | `_detect_page_step_number_boxes`, `_detect_step_number_below_panel` — "port now". |

---

## CROP RULES

| Rule | Purpose | Source | V2 status | Evidence |
|---|---|---|---|---|
| **V1 repair/fallback callout logic** | Callout boxes must use V1 repair + page-level fallback chain | `V1_BEHAVIOUR_CONTRACTS.md` §3; `V1_CURRENT_VS_OLD_GUARDRAIL_AUDIT.md` §Callout Guardrails | Partial | Functions: `_repair_callout_box_candidate_crop`, `_detect_page_level_callout_panels`, `_page_level_callout_candidates_for_fallback`, `_refine_page_level_panel_with_step_geometry`. |
| **Fresh/final crop beats stale saved crop** | Prefer newly detected crop over stale training-label box | `V1_BEHAVIOUR_CONTRACTS.md` §3; `V1_RECOVERY_PORT_PLAN.md` §Callout repair/fallback | Partial | Page 23 step 27: fresh `113,28,313,161` must beat stale `113,3,313,193`. |
| **Record crop selection provenance** | Manifest must explain which crop won and why | `V1_BEHAVIOUR_CONTRACTS.md` §3 | Partial | "selection source, repair/fallback source, and stale-vs-fresh decision." |
| **Block segmentation on ambiguous crop source** | Downstream gate when crop origin unclear | `V1_BEHAVIOUR_CONTRACTS.md` §3 | Planned gate | "Do not proceed to segmentation if crop source is ambiguous." |
| **Edge-detect callout baseline** | Primary crop detector path | `V1_CURRENT_VS_OLD_GUARDRAIL_AUDIT.md`; `V1_PARITY_PLAN.md` §Stage 5 | Partial | `_detect_callout_rect_by_edges`; V2 has simplified adaptation. |
| **Known crop parity proof pages** | Acceptance tests for crop geometry | `V1_BEHAVIOUR_CONTRACTS.md` §3; `V1_PARITY_PLAN.md` §Stage 5 | Partial | Page 12 steps 11/12; page 23 step 27. |
| **`clipped_left_callout_panel_slack` geometry rule** | Handle panels extending below step anchor | `BAG1_PARITY_SIGNOFF.md` §Step 8 Evidence | Pass Bag 1 | `geometry_rule=clipped_left_callout_panel_slack` on page 10 step 8. |
| **All V1 crop-cache IDs exist in V2** | Crop coverage parity per bag | `BAG1_PARITY_SIGNOFF.md` check 3; `BAG2_*`; `BAG3_PARITY_SIGNOFF.md` | Pass Bags 1–3; **fail Bag 4** | Bag 1: 22/22. Bag 2: 15/15. Bag 3: 30/30. Bag 4: 20/29. |
| **Reviewed training-label crops must resolve** | Human-reviewed crops link to V2 callouts | `BAG1_PARITY_SIGNOFF.md` check 10; `BAG2_PARITY_SIGNOFF.md` check 4 | Pass Bags 1–2 | Bag 1: 22/22. Bag 2: 13/13. |
| **Restore missing right-column crops from V1 truth** | Fix opposite-column misses | `BAG3_PARITY_SIGNOFF.md`; `BAG4_PARITY_SIGNOFF.md` §Applied Parity Fixes | Applied where documented | Bag 3 pages 49/50/52; Bag 4 pages 59–66. |
| **Remap duplicate physical crops to correct V1 step** | Avoid duplicate step IDs on same page | `BAG3_PARITY_SIGNOFF.md` §Page 40 | Applied Bag 3 | Page 40: duplicate V2 step 44 remapped to step 45 / `p40_s45_c3`. |
| **No missing callouts / qty in approved bags** | Crop+qty completeness | `BAG2_PARITY_SIGNOFF.md` checks 6–7 | Pass Bag 2 | `missing_callouts=[]`; `missing_qty=[]`. |
| **Page 81 = bag-start/non-callout classification** | Prevent silent drop of bag-start pages | `V1_BEHAVIOUR_CONTRACTS.md` §3 | Partial | "explicitly classified as a bag-start/add-bag page or non-callout page, not silently disappear." |
| **Do not approximate callout detection** | Port exact V1 lab flow | `V1_RECOVERY_PORT_PLAN.md` §Port Rules; §Callout Crop And Repair | Policy | Exact chain: `detect_steps` → `_contact_sheet_step_boxes_from_detected` → `_build_step_region` → `_detect_callout_rect_by_edges` → repair/fallback. |
| **Extra semantic callouts = out-of-sequence OCR** | V2 callouts without V1 truth are flagged, not silently accepted | `BAG4_PARITY_SIGNOFF.md` §Extra Semantic Callouts | Signoff (Bag 4) | 9 extra semantic callouts documented with sequence/OCR notes. |
| **Page 69 step 110 human-review exception** | Real callout missing from V1 truth — separate accounting | `BAG4_PARITY_SIGNOFF.md` §Human Review Required | Active classification | `TRUE_CALLOUT_MISSING_FROM_V1_TRUTH`; "Do not count as V2_FALSE_POSITIVE"; "should not be silently added to V1 parity." |
| **Page 163 step 278 not a real part crop** | Known note from parity visual audit work | Referenced in step-regression Azure audit prompt (not in core V1 contracts) | — | **NOT in listed V1 parity contract files.** Included only in downstream audit tooling notes. |

---

## SEGMENTATION RULES

| Rule | Purpose | Source | V2 status | Evidence |
|---|---|---|---|---|
| **Qty anchors own slots** | Segmentation is slot-aware, not whole-crop CC extraction | `V1_BEHAVIOUR_CONTRACTS.md` §5; `V1_PARITY_PLAN.md` §Stage 7 | **Missing** | "Qty anchors own slots." V2 is "whole-callout connected-component segmentation." |
| **Repeated-qty de-duplication / ownership** | Avoid duplicate slots from repeated qty rows | `V1_BEHAVIOUR_CONTRACTS.md` §5; `V1_PARITY_PLAN.md` §Stage 7 | **Missing** | Page 23 `1x` rows must not duplicate into extra slots. |
| **Master-island / window fallback** | Recover ambiguous slots | `V1_BEHAVIOUR_CONTRACTS.md` §5; `V1_RECOVERY_PORT_PLAN.md` | **Missing** | V2 must support master-island/window fallback. |
| **`needs_review` for uncertain slots** | Do not emit confident bad cutouts | `V1_BEHAVIOUR_CONTRACTS.md` §5; `V1_RECOVERY_PORT_PLAN.md` §Needs-review semantics | **Missing** | Statuses: `needs_review`, `needs_review_low_alpha`. |
| **Slot ownership proof fields** | Manifest must prove qty→slot assignment | `V1_BEHAVIOUR_CONTRACTS.md` §5; `V1_PARITY_PLAN.md` | **Missing** | `qty_anchor`, `qty_row`, `slot_window`, `component_source`, review status. |
| **Segmentation proof pages** | Acceptance tests | `V1_BEHAVIOUR_CONTRACTS.md` §5 Minimum Proof | **Missing** | Page 12 step 11 → 3 slot-owned outputs; step 12 → `6x` slot; page 23 step 27 repeated `1x`. |
| **Do not approximate slot segmentation** | Port exact V1 slot flow | `V1_RECOVERY_PORT_PLAN.md` §Port Rules | Policy | "Do not approximate slot segmentation." |
| **Matching paused until segmentation parity** | Stage gate | `V1_PARITY_PLAN.md` §Priority; `V1_BEHAVIOUR_CONTRACTS.md` §Contract Gate | Active policy | "Matching must not be tuned before crop, OCR, and segmentation parity are proven." |
| **Segmentation missing check (Bag 2)** | Bag-level completeness | `BAG2_PARITY_REVIEW.md` check 8 | Pass Bag 2 | `segmentation_missing=[]`. |
| **V1 slot flow functions to port** | Source reference | `V1_RECOVERY_PORT_PLAN.md` §Recovery Matrix | Not ported | `auto_mask_slots`, `_segment_step_callout_slot`, `create_shape_masks_for_callout_slots`, `normalize_slot_crop_from_qty`. |

---

## REVIEW RULES

| Rule | Purpose | Source | V2 status | Evidence |
|---|---|---|---|---|
| **Contract gate before matching** | Pause matching until contracts pass or human-excepted | `V1_BEHAVIOUR_CONTRACTS.md` §Contract Gate | Active policy | "Part matching should remain paused until these contracts are either `pass` … or explicitly accepted as a human-review exception." |
| **Human decisions authoritative** | Review overrides are final | `V1_PARITY_PLAN.md` §Parity Principles; `BEHAVIOUR_AUDIT.md` §Stage 11 | Yes | "Human decisions remain authoritative." |
| **No auto-save labels** | Manifests must not write training labels autonomously | `V1_BEHAVIOUR_CONTRACTS.md` §1; `V1_PARITY_PLAN.md` §Stages 8–10 | Yes | Bag-number manifest "must not auto-save"; matching "must not auto-save labels." |
| **Match audit read-only until parity proven** | Audit stage discipline | `V1_PARITY_PLAN.md` §Stage 9; `BEHAVIOUR_AUDIT.md` §Stage 10 | Yes | "audit should remain read-only until matching and segmentation parity are proven." |
| **Preserve human gap accept/reject decisions** | Bag gap review authority | `V1_PARITY_PLAN.md` §Stage 3b | Yes | "preserve human accepted/rejected decisions without overwrite." |
| **Exclude non-crop `elem_*` records from crop parity** | Label record filtering | `BAG1_PARITY_SIGNOFF.md` §Excluded Non-Crop Label Records | Pass Bag 1 | Records with `page=0`, `step=0`, no crop path excluded from check 10. |
| **Bag parity signoff gates** | Bag-by-bag approval | `BAG1/2/3_PARITY_SIGNOFF.md`; `BAG2_PARITY_REVIEW.md` | Bags 1–3 approved; Bag 4 in progress | `BAG1_PARITY_APPROVED`, `BAG2_PARITY_APPROVED/PASS`, `BAG3_PARITY_APPROVED`, `BAG4_PARITY_IN_PROGRESS`. |
| **Gap audit is evidence-only** | No silent promotion from audit | `BAG3_PARITY_SIGNOFF.md`; `BAG3_SEQUENCE_GAP_PARITY_AUDIT.md` header | Yes | "parity-audit rule only. It is not a detection redesign." Audit doc: "Read-only. No promotion." |
| **Qty OCR review prerequisites** | Stage ordering before slots | `V1_BEHAVIOUR_CONTRACTS.md` §4; `V1_RECOVERY_PORT_PLAN.md` | Partial | Qty validated, row-grouped, de-duplicated; step-number contamination filtered before slot creation. |
| **Matching skip/flag `needs_review` slots** | Review before match | `V1_BEHAVIOUR_CONTRACTS.md` §5; `V1_RECOVERY_PORT_PLAN.md` | Not implemented | Segmentation contract still `missing`. |
| **V1 truth incomplete exception (page 69 step 110)** | Diagnostic only, not pass/fail | `BAG4_PARITY_SIGNOFF.md` §Bag 4 parity audit update | Active | `V1_TRUTH_INCOMPLETE`; "Do not count as V2_FAILURE"; "Do not count as V2_FALSE_POSITIVE." |
| **AI recommendations derived-only** | No auto-accept from AI | `V1_PARITY_PLAN.md` §Stage 10 | Policy | "`12_ai_override_recommendations.json` never auto-accepts labels." |

---

## Bag approval summary

| Bag | Status | Step rules evidenced | Crop rules evidenced |
|---|---|---|---|
| 1 | **APPROVED** | Sequence 1–25; step 65 absent; step 8 present | 22/22 V1 crop IDs; `clipped_left_callout_panel_slack` on step 8 |
| 2 | **APPROVED** | Sequence 26–40; no false positives | 15/15 crop IDs; no missing callouts/qty |
| 3 | **APPROVED** | Sequence-gap full-page audit rule; OCR 78 fix | 30/30 crop IDs; right-column restores |
| 4 | **IN PROGRESS** | OCR corrections; out-of-sequence extras flagged | 20/29 crop IDs; 5 confirmed detector misses remain |

---

## Gaps explicitly NOT documented in parity sources

The following were requested for inventory but **do not appear** as named rules in the listed parity documents:

- Same baseline alignment threshold for multi-digit merge
- Same colour/style matching for step anchors
- "Dominant page step" as a formal rejection rule (only V1 `main_steps`/`sub_steps` channel)
- Explicit "reject substeps 1/2/3 in parts panels" contract (only audit contrast evidence in Bag 3 statistics)

These may exist in runtime code (`step_map_scan.mjs`, `step_detector_service.py`) but are **outside the scope of this rulebook**, which documents only parity-work artifacts.
