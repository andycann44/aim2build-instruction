# V1 Parity Plan

Date: 2026-06-02

Goal: make `instruction-v2` as good as or better than the proven V1 workflow by using V1 as the working reference, while keeping V2 local-manifest-first.

This is a plan only. No code or manifests were changed.

## Parity Principles

- V1 is the behavioural reference until V2 proves a stage is equal or better.
- V2 should keep one manifest per stage and no duplicate label source of truth.
- Human decisions remain authoritative.
- Matching must not be tuned before crop, OCR, and segmentation parity are proven.
- Each parity step needs a proof artifact: counts, sample overlays, contact sheets, and known-case comparisons.

## Priority Summary

| Priority | Area | Reason |
|---|---|---|
| P0 | Bag detection / gap repair | A missing bag window changes every downstream range. |
| P0 | Step detection / printed step numbers | Bad or synthetic step metadata breaks crop IDs and review traceability. |
| P0 | Callout crop boxes | Segmentation cannot recover from bad callout crops. |
| P0 | Qty OCR | V1 segmentation is slot-aware; slots depend on qty anchors. |
| P0 | Slot-aware segmentation | Current V2 whole-callout component extraction is not V1 parity. |
| P1 | Matching | Matching quality should improve only after segmentation parity. |

## Stage 0 - Set Context

- Current V2 implementation: `stage0_set_context.py` writes `indexes/00_set_context.json`.
- V1 working source/function: `clean/services/instruction_buildability_source.py::load_instruction_set_parts`.
- Missing V1 behaviour: none critical for current parity target.
- Parity requirement: V2 set context contains the same expected part/color universe used by V1 matching/review for the set.
- How to prove parity: compare V2 `00_set_context.json` row count and `(part_num, color_id, qty)` totals against V1 catalog loader output for `70618`.
- Priority: P2.

## Stage 1 - PDF Manifest

- Current V2 implementation: `phase1_pdf_pages.py`, `render_pdf_pages.mjs`; outputs `01_pdf_manifest.json` and `02_page_index.json`.
- V1 working source/function: `clean/services/debug_service.py::resolve_page_image_path` and existing rendered page folders under V1 debug paths.
- Missing V1 behaviour: V1 page resolver can use existing rendered debug folders; V2 intentionally owns its local PDF/page copies.
- Parity requirement: V2 renders the same PDF pages with stable page numbers and usable image dimensions.
- How to prove parity: checksum PDF; verify page count; compare selected page images/dimensions against V1 rendered references for pages 12, 23, 81.
- Priority: P2.

## Stage 2 - Bag Candidates

- Current V2 implementation: `stage2_bag_candidates.py`, `bag_candidate_scan.mjs`; outputs `03_bag_candidates.json`.
- V1 working source/function: `clean/routers/debug.py::_resolve_bag_page_range`, `clean/services/step_sequence_bag_service.py::build_step_sequence_prepass_for_pages`, `clean/services/step_sequence_bag_service.py::scan_step_bag_sequence`, plus V1 set/gap scan services.
- Missing V1 behaviour: V2 candidate detection is simpler and can miss bag-start pages that do not look like obvious bag panels.
- Parity requirement: V2 must detect enough candidate/evidence pages that all real bag starts either appear directly in `03_bag_candidates.json` or are flagged by Stage 3b.
- How to prove parity: compare detected starts/ranges against known accepted corrections, especially page 81 as observed bag 5; verify no large unexplained range gap enters `04_bag_map.json` without Stage 3b review.
- Priority: P0.

## Stage 3b - Bag Gap Review / Missing Bag Window Review

- Current V2 implementation: `stage3b_bag_gap_review.py`, `gap_window_scan.mjs`, `create_bag_gap_review_contact_sheet.py`; outputs `03b_bag_gap_review.json`.
- V1 working source/function: `clean/services/gap_scan_service.py`, V1 routes `clean/routers/gap_scan.py` and `clean/routers/gap_review.py`.
- Missing V1 behaviour: V2 now scores pages inside gap windows, but parity is not proven across all known missing/ambiguous windows.
- Parity requirement: every suspicious gap must include scored candidate pages, confidence/evidence, and preserve human accepted/rejected decisions without overwrite.
- How to prove parity: for `gap_after_bag_03_before_bag_04`, confirm page 81 remains accepted by human and ranks as a top candidate; for the pending 105-163 window, confirm the contact sheet presents enough evidence for human review.
- Priority: P0.

## Stage 3 - Bag Map

- Current V2 implementation: `stage3_bag_map.py`; outputs `04_bag_map.json`.
- V1 working source/function: range resolution concepts from `clean/routers/debug.py::_resolve_bag_page_range` and `clean/services/step_sequence_bag_service.py`.
- Missing V1 behaviour: V2 final map does not yet have a full truth/reconciliation layer for all bags.
- Parity requirement: accepted human gap corrections must be applied exactly, with trace/evidence, and no downstream stage should infer hidden bag starts outside reviewed evidence.
- How to prove parity: `04_bag_map.json` includes page 81 as observed bag 5 with `human_gap_review` trace; bag ranges are monotonic and cover the intended instruction windows.
- Priority: P0.

## Stage 4 - Step Detection / Printed Step Numbers

- Current V2 implementation: `stage4_step_map.py`, `step_map_scan.mjs`; outputs `05_step_map.json`.
- V1 working source/function: `clean/services/step_detector_service.py::detect_steps`; supporting V1 debug/contact-sheet helpers.
- Missing V1 behaviour: V2 was previously using local order or nulls; current improvement detects printed numbers but parity needs broader validation.
- Parity requirement: `page` is the PDF/page number, `step_index` is local detected order, and `step_number` is the actual printed LEGO instruction step number or `null`. No synthetic numbering.
- How to prove parity: sample pages with known printed numbers show exact values, e.g. page 12 has step 11 and step 12; overlays say `page 12 · step 11`, not `step? 1`.
- Priority: P0.

## Stage 5 - Callout Crop Boxes

- Current V2 implementation: `stage5_callout_crop_boxes.py`, `callout_crop_box_scan.mjs`; outputs `06_callout_crop_box_map.json`.
- V1 working source/function: `clean/routers/callout_crop_lab.py`; exact reference flow `step_detector_service.detect_steps`, `_contact_sheet_step_boxes_from_detected`, `_build_step_region`, `_detect_callout_rect_by_edges`; related V1 debug functions in `clean/routers/debug.py`.
- Missing V1 behaviour: V2 crop boxes can differ from V1 proven crops. Known examples: page 12 step 11 V2 crop width `282` vs V1 saved `313`; page 23 step 27 V2 crop `y=28,h=161` vs V1 saved `y=3,h=193`.
- Parity requirement: for known V1 working crops, V2 crop boxes must either match V1 or explain a deliberate improved box with overlay evidence.
- How to prove parity: compare crop boxes and image sizes for page 12 step 11, page 12 step 12, page 23 step 27; review crop overlays before OCR/segmentation.
- Priority: P0.

## Stage 6 - Qty OCR

- Current V2 implementation: `stage6_qty_ocr.py`, `qty_ocr_scan.mjs`; outputs `07_qty_ocr_map.json`.
- V1 working source/function: `clean/routers/instruction_debug.py::_extract_detected_qty_details_from_crop`, `_auto_qty_payload_for_crop`, `_qty_payload_for_page_level_callout_crop`, and imported `clean/routers/debug.py::_extract_detected_qty_details_from_crop`.
- Missing V1 behaviour: V2 misses or duplicates qty tokens that V1 handled. Known examples: page 12 step 12 misses V1's `6x`; page 23 step 27 has duplicate `1x` boxes at the same locations.
- Parity requirement: V2 must recover V1-level qty text, token boxes, row grouping, de-duplication, and confidence/source metadata needed by slot segmentation.
- How to prove parity: compare OCR tokens against V1 analysis bundles for page 12 step 11, page 12 step 12, and page 23 step 27; require no duplicate boxes within near-identical geometry unless intentionally retained with evidence.
- Priority: P0.

## Stage 7 - Slot-Aware Segmentation

- Current V2 implementation: `stage7_part_segmentation.py`, `part_segmentation_scan.mjs`; outputs `08_part_segmentation_map.json` and assets under `debug/part_segmentation/`.
- V1 working source/function: `clean/routers/instruction_debug.py::auto_mask_slots`, `clean/services/ai_snap_crop_service.py::create_shape_masks_for_callout_slots`, `clean/services/part_crop_normalize_service.py::normalize_slot_crop_from_qty`, and `clean/routers/instruction_debug.py::_segment_step_callout_slot`.
- Missing V1 behaviour: V2 is whole-callout connected-component segmentation, not qty-anchor/slot-aware segmentation. It lacks V1 row grouping, qty-anchor ownership, repeated-qty handling, master-island fallback, window fallback, and `needs_review` slot statuses.
- Parity requirement: V2 segmentation must decide slot ownership before normalization. It should emit the same manifest schema, but internally use V1-style slot-aware logic and carry proof fields in metrics: qty anchor, slot window, component source, rejection/review reason, and selected component box.
- How to prove parity: for page 12 step 11, V2 produces three slot-aware outputs corresponding to the V1 qty anchors; for page 12 step 12, V2 recovers the `6x` slot; for page 23 step 27, V2 handles repeated `1x` rows without duplicate/ambiguous slots. Compare V2 cutouts against V1 saved `analysis_bundles` and review contact sheets.
- Priority: P0.

## Stage 8 - Matching

- Current V2 implementation: `stage8_match.py`, `match_segments_scan.mjs`; outputs `09_match_manifest.json`.
- V1 working source/function: `clean/routers/instruction_debug.py::buildability_clip_suggest`, CLIP helpers in `clean/routers/instruction_debug.py`, catalog image helper `tools/a2b_clip_match_probe.py::_ensure_catalog_image_for_pair`, and later references `clean/services/part_candidate_service.py`, `clean/services/confirmed_memory_service.py`.
- Missing V1 behaviour: V2 currently uses provisional RGB-only matching. It lacks V1 Top 5 CLIP candidate flow, catalog image parity, score components, and memory-derived boosts.
- Parity requirement: after segmentation parity, V2 matching must return Top 5 candidates in the same expected catalog universe, with candidate image, `part_num`, `color_id`, score, source, and score components when available. It must not auto-save labels.
- How to prove parity: run a fixed sample set and compare V2 Top 5 against V1 `/debug/buildability-clip-suggest`; verify Bag 1 known baseline remains human-reviewable, not autonomous.
- Priority: P1.

## Stage 9 - Match Audit

- Current V2 implementation: `audit_matches.py`, `create_match_audit_contact_sheet.py`; outputs `10_match_audit.json` and `debug/match_audit/contact_sheet.html`.
- V1 working source/function: V1 manual review UI in `clean/routers/instruction_debug.py::manual_match_review`.
- Missing V1 behaviour: V2 audit is static and does not yet support full manual correction/save workflow.
- Parity requirement: audit should remain read-only until matching and segmentation parity are proven; then it can become a review surface that writes only to the approved V2 human decision layer.
- How to prove parity: contact sheet shows highest/lowest/random examples with enough evidence for human review; no labels are written.
- Priority: P2.

## Stage 10 - Manual Config

- Current V2 implementation: `indexes/11_manual_match_config.json`, displayed by audit generation.
- V1 working source/function: `clean/services/bag_review_service.py`, V1 save/unknown/ignored routes in `clean/routers/instruction_debug.py`.
- Missing V1 behaviour: V2 manual config is not a full review UI and does not write final labels.
- Parity requirement: keep manual decisions authoritative inside V2 config until an explicit promotion path exists. AI recommendations must remain derived and non-authoritative.
- How to prove parity: manual override rows affect audit display but do not write `training_labels`; future `12_ai_override_recommendations.json` never auto-accepts labels.
- Priority: P2.

## Proof Checklist Before Matching Work

1. Bag gap page 81 remains accepted human evidence for observed bag 5.
2. Step overlays use real printed numbers where detected.
3. Known crop boxes match or intentionally improve over V1 for page 12 step 11, page 12 step 12, page 23 step 27.
4. Qty OCR recovers V1 tokens for those samples, including page 12 step 12 `6x`.
5. Segmentation is slot-aware and can mark unresolved slots as needs-review instead of emitting questionable components.
6. Only after the above: rerun matching and compare Top 5 against V1.
