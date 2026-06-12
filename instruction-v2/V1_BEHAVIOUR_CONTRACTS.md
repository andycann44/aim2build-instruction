# V1 Behaviour Contracts

Purpose: define the V1 behaviours V2 must match before part matching continues.

This is documentation only. No V2 logic, manifests, clean code, or matching outputs were changed.

## Contract Gate

Part matching should remain paused until these contracts are either:

- `pass`, proven on the listed pages, or
- explicitly accepted as a human-review exception.

## 1. Bag Number Recognition

Contract:

- V2 must read the visible bag number.
- V2 must not only score a page as “bag-like”.
- V2 must expose exact/conflicting bag number evidence before final bag-map acceptance.

V1 source functions:

- `clean/services/page_analyzer.py::analyze_page`
- `clean/services/analyzer_scan_service.py::_build_sequence_scan_row`
- `clean/services/gap_scan_service.py::_normalize_analysis_row`
- `clean/services/gap_scan_service.py::_score_window_candidate`
- `clean/services/gap_scan_service.py::_build_candidate`
- Routes: `clean/routers/sequence.py::api_analyze_gap_page`, `clean/routers/gap_scan.py::api_gap_scan`, `clean/routers/gap_review.py::gap_review`

Current V2 status: `missing`

Proof test pages:

- Page 81: human accepted visible Bag 5.
- Page 100: currently inferred by cluster, needs visible-number check.
- Page 131: human observation says visible Bag 7.
- Page 148: human observation says visible Bag 8.

Required fix before matching:

- Add a V2 bag-number recognition manifest/checkpoint before final bag-map acceptance.
- It must output `bag_number`, `ocr_raw`, `number_box`, `number_box_found`, `panel_found`, `panel_source`, `confidence`, and exact/conflict reasons.
- It must not auto-save; human review remains final authority.

## 2. Step Number Recognition

Contract:

- V2 must detect the printed instruction step number.
- V2 must reject impossible anchors.
- V2 must not use `step_index` as a fallback `step_number`.
- Invalid anchors must become `null` or be rejected with a reason.

V1 source functions:

- `clean/services/step_detector_service.py::detect_steps`
- `clean/routers/instruction_debug.py::_filter_invalid_step_anchor_boxes`
- Step detector flow consumed by `clean/routers/callout_crop_lab.py`

Current V2 status: `partial`

Proof test pages:

- Page 7: printed steps 1 and 2; V2 Bag 1 bootstrap identified page 7, but current `05_step_map.json` is stale and was not rerun from page 6.
- Page 12: printed steps 11 and 12.
- Page 23 step 27: must remain actual printed step 27, not local index.
- Any page with impossible OCR such as `1583`: must be rejected.

Required fix before matching:

- Rerun/verify Stage 5 after accepted bag-map changes.
- Preserve `page`, `step_index`, and printed `step_number` as separate fields.
- Keep `rejection_reason` for invalid anchors.
- Prove no synthetic numbering appears in `05_step_map.json`.

## 3. Callout Crop Detection

Contract:

- V2 must use V1 repair/fallback logic for callout crop boxes.
- V2 must prefer a fresh/final detected crop box over a stale saved crop box.
- V2 must preserve enough evidence to explain crop-box selection.

V1 source functions:

- `clean/routers/instruction_debug.py::_detect_callout_rect_by_edges`
- `clean/routers/instruction_debug.py::_repair_callout_box_candidate_crop`
- `clean/routers/instruction_debug.py::_detect_page_level_callout_panels`
- `clean/routers/instruction_debug.py::_page_level_callout_candidates_for_fallback`
- `clean/routers/instruction_debug.py::_refine_page_level_panel_with_step_geometry`
- Proven lab flow: `clean/routers/callout_crop_lab.py`

Current V2 status: `partial`

Proof test pages:

- Page 12 step 11: crop must match V1-proven callout region.
- Page 12 step 12: crop must support V1-proven `6x` qty OCR.
- Page 23 step 27: expected fresh crop box `113,28,313,161`; stale crop `113,3,313,193` must not win.
- Page 81: should be explicitly classified as a bag-start/add-bag page or non-callout page, not silently disappear.

Required fix before matching:

- Stage 6 must prove crop-box parity on these pages.
- Manifest must record selection source, repair/fallback source, and stale-vs-fresh decision.
- Do not proceed to segmentation if crop source is ambiguous.

## 4. Qty OCR

Contract:

- V2 must validate quantity tokens.
- V2 must avoid step-number contamination.
- V2 must support region retry.
- V2 must expose row grouping, duplicate handling, token boxes, and confidence/source.

V1 source functions:

- `clean/routers/instruction_debug.py::_extract_detected_qty_details_from_crop`
- `clean/routers/instruction_debug.py::_final_crop_qty_token_is_valid`
- `clean/routers/instruction_debug.py::_qty_payload_for_page_level_callout_crop`
- `clean/routers/instruction_debug.py::_auto_qty_payload_for_crop`
- Related qty extraction helpers in `clean/routers/debug.py`

Current V2 status: `partial`

Proof test pages:

- Page 12 step 11: expected qty anchors `[8, 4, 4]`.
- Page 12 step 12: expected qty anchor `6x`; V2 must keep the improved `["6x"]`.
- Page 23 step 27: repeated `1x` rows must not duplicate into extra slots.
- Page 81: no ordinary qty callout should be recorded only if page classification explains why.

Required fix before matching:

- Stage 7 must expose validated qty tokens, row groups, duplicate-collapse decisions, and retry source.
- Qty tokens must be filtered against step-number contamination.
- Slot creation must consume validated qty anchors, not raw OCR blobs.

## 5. Slot-Aware Segmentation

Contract:

- Qty anchors own slots.
- Repeated qty handling must avoid duplicate/ambiguous ownership.
- V2 must support master-island/window fallback.
- If uncertain, the slot must be `needs_review`, not silently matched.

V1 source functions:

- `clean/routers/instruction_debug.py::auto_mask_slots`
- `clean/routers/instruction_debug.py::_segment_step_callout_slot`
- `clean/services/ai_snap_crop_service.py::create_shape_masks_for_callout_slots`
- `clean/services/part_crop_normalize_service.py::normalize_slot_crop_from_qty`

Current V2 status: `missing`

Proof test pages:

- Page 12 step 11: three qty anchors must produce three slot-owned outputs.
- Page 12 step 12: `6x` qty anchor must own the visible slot.
- Page 23 step 27: repeated `1x` rows must produce the correct slots without duplicate false slots.
- Any bad segmentation-audit row: uncertain output must become `needs_review`.

Required fix before matching:

- Stage 8 must be slot-aware before matching resumes.
- Manifest can keep current schema, but entries must carry slot ownership proof: `qty_anchor`, `qty_row`, `slot_window`, `component_source`, fallback/rejection reason, and review status.
- Matching must skip or flag `needs_review` slots rather than treating them as confident cutouts.

## Minimum Proof Before Matching Continues

Required passing checks:

- Bag-number recognition reads visible Bag 7 on page 131 and visible Bag 8 on page 148.
- Step map detects printed steps without impossible anchors or synthetic numbering.
- Page 23 step 27 uses fresh crop box `113,28,313,161`.
- Page 12 step 12 keeps `6x`.
- Page 23 repeated `1x` tokens are row-grouped and de-duplicated.
- Page 12 step 11 produces three slot-owned segment outputs.
- Any uncertain segmentation is marked `needs_review`.

Until these pass, V2 part matching remains provisional and should not be treated as label-ready.
