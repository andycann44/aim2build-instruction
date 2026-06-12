# Instruction V2 Behaviour Audit

This audit walks the current `instruction-v2` program from start to the latest implemented review layer and identifies where unmapped V1 services should be introduced.

Documentation only. No code, manifests, or V1 files were changed by this audit.

## Stage 0 - Set Context

- Current V2 file/script: `stage0_set_context.py`
- Input manifest: none
- Output manifest: `indexes/00_set_context.json`
- Current behaviour: Reads catalog set parts and writes the set context used by later matching.
- Weakness observed: Depends on current catalog loader shape; no independent V2 schema adapter yet.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/instruction_buildability_source.py`
- Should introduce now? no
- Reason: Stage 0 is accepted and already uses the existing catalog DB/loader path as read-only input. Keep it stable until matching quality work needs richer catalog metadata.

## Stage 1 - PDF Manifest

- Current V2 file/script: `phase1_pdf_pages.py`, `render_pdf_pages.mjs`
- Input manifest: none
- Output manifest: `indexes/01_pdf_manifest.json`
- Current behaviour: Copies/saves one instruction PDF under V2 and records PDF identity/checksum.
- Weakness observed: No V1 load-set UX or download orchestration inside V2.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/dataset_service.py`, `clean/routers/load_set.py`
- Should introduce now? no
- Reason: Local-manifest-first PDF input is working. V1 load/download UX can remain separate until V2 needs a user-facing import flow.

## Stage 2 - Page Index

- Current V2 file/script: `phase1_pdf_pages.py`, `render_pdf_pages.mjs`
- Input manifest: `indexes/01_pdf_manifest.json`
- Output manifest: `indexes/02_page_index.json`
- Current behaviour: Renders PDF pages into `instruction-v2/pages/` and records page image paths/checksums.
- Weakness observed: No page precheck, text-heavy filtering, or V1 page directory resolver abstraction.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/debug_service.py`, `clean/services/precheck_service.py`, `clean/services/analyzer_scan_service.py`
- Should introduce now? no
- Reason: Rendering and indexing are stable. Precheck/analyzer services should stay V1-side until V2 needs page quality gates.

## Stage 3 - Bag Candidates

- Current V2 file/script: `stage2_bag_candidates.py`, `bag_candidate_scan.mjs`
- Input manifest: `indexes/02_page_index.json`
- Output manifest: `indexes/03_bag_candidates.json`
- Current behaviour: Performs local visual scan over rendered pages and records candidate bag/evidence pages.
- Weakness observed: Missed page -99 bag-start window that was later found by human review at page 81.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/gap_scan_service.py`, `clean/services/set_scan_service.py`, `clean/services/precheck_service.py`
- Should introduce now? later
- Reason: The new Stage 3b catches missing windows, so Stage 3 can remain simple. V1 gap/set scan ideas should be introduced after the current human gap review loop proves enough examples.

## Stage 3b - Gap Review

- Current V2 file/script: `stage3b_bag_gap_review.py`, `create_bag_gap_review_contact_sheet.py`
- Input manifest: `indexes/03_bag_candidates.json`
- Output manifest: `indexes/03b_bag_gap_review.json`
- Current behaviour: Detects suspicious page gaps between bag candidate clusters, writes pending gap windows, and lets human correction be recorded manually.
- Weakness observed: Candidate pages inside accepted gap were empty; human found page 81 manually via screenshot/page review. The stage flags gaps but does not actively scan pages inside them for bag panels.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/gap_scan_service.py`, `clean/routers/gap_scan.py`, `clean/routers/gap_review.py`, `clean/services/step_detector_service.py`
- Should introduce now? yes
- Reason: This is the strongest immediate candidate. V2 should adapt the useful parts of `gap_scan_service.py` to score pages inside gap windows before final bag-map regeneration, while keeping human approval authoritative.

## Stage 4 - Bag Map

- Current V2 file/script: `stage3_bag_map.py`
- Input manifest: `indexes/03_bag_candidates.json`, `indexes/03b_bag_gap_review.json`
- Output manifest: `indexes/04_bag_map.json`
- Current behaviour: Builds bag page ranges from candidate clusters plus accepted Stage 3b human corrections. Page 81 is now included as human-observed bag 5 evidence.
- Weakness observed: Bag numbering after observed human corrections is inferred and may need more human/sequence validation. A pending gap between pages 105- 163 remains unresolved.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/gap_scan_service.py`, `clean/services/truth_service.py`, `clean/services/bag_truth_store.py`
- Should introduce now? later
- Reason: Do not complicate Stage 4 until Stage 3b has a better gap scanner and accepted/rejected decisions. Truth storage should not be introduced until V2 has a deliberate authority model.

## Stage 5 - Step Box Map

- Current V2 file/script: `stage4_step_map.py`, `step_map_scan.mjs`
- Input manifest: `indexes/04_bag_map.json`, `indexes/02_page_index.json`
- Output manifest: `indexes/05_step_map.json`
- Current behaviour: Detects visual step-box candidates and records `page`, `step_index`, `step_number`, `step_box`, confidence, and overlays.
- Weakness observed: Printed yellow step numbers are mostly not OCR-detected; `step_number` is often `null`. Detection is visual-only and simpler than V1.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/step_detector_service.py`, `clean/routers/step_debug.py`
- Should introduce now? yes
- Reason: This is a correctness issue. V1 `step_detector_service.py` has OCR/visual step-number logic that should be adapted into V2 so `step_number` is actual printed instruction metadata, not local order.

## Stage 6 - Callout Crop Boxes

- Current V2 file/script: `stage5_callout_crop_boxes.py`, `callout_crop_box_scan.mjs`
- Input manifest: `indexes/05_step_map.json`, `indexes/02_page_index.json`
- Output manifest: `indexes/06_callout_crop_box_map.json`
- Current behaviour: Uses a V2-local adaptation of the proven callout crop lab flow to detect callout/crop boxes and save crops/overlays.
- Weakness observed: Depends on step-box quality; crop boxes can be wrong if step boxes or bag ranges are wrong.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/step_detector_service.py`; mapped reference file `clean/routers/callout_crop_lab.py`
- Should introduce now? later
- Reason: Improve Stage 5 step metadata first. The crop flow is currently good enough to continue human review/audit work.

## Stage 7 - Qty OCR

- Current V2 file/script: `stage6_qty_ocr.py`, `qty_ocr_scan.mjs`
- Input manifest: `indexes/06_callout_crop_box_map.json`
- Output manifest: `indexes/07_qty_ocr_map.json`
- Current behaviour: Runs local Tesseract variants over callout crops and records quantity text/numbers/token boxes.
- Weakness observed: OCR confidence can be low or noisy; no V1 token-order heuristics are fully ported.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/inventory_scan_service.py`, plus mapped V1 OCR helper in `clean/routers/debug.py`
- Should introduce now? later
- Reason: Quantity OCR is currently usable as downstream segmentation support. Revisit after segmentation/matching quality is audited with humans.

## Stage 8 - Part Segmentation

- Current V2 file/script: `stage7_part_segmentation.py`, `part_segmentation_scan.mjs`
- Input manifest: `indexes/06_callout_crop_box_map.json`, `indexes/07_qty_ocr_map.json`
- Output manifest: `indexes/08_part_segmentation_map.json`
- Current behaviour: Uses foreground connected components, with quantity token boxes erased, to produce masks/cutouts/overlays.
- Weakness observed: Component segmentation can split/merge parts, include non-part foreground, or miss low-contrast shapes.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/ai_snap_crop_service.py`, `clean/services/part_crop_normalize_service.py`
- Should introduce now? yes
- Reason: Segmentation quality directly controls matching quality. V1 `ai_snap_crop_service.py` and `part_crop_normalize_service.py` should be adapted into V2 as a stronger segmentation/normalization layer, still writing only V2 manifests/assets.

## Stage 9 - Matching

- Current V2 file/script: `stage8_match.py`, `match_segments_scan.mjs`
- Input manifest: `indexes/00_set_context.json`, `indexes/08_part_segmentation_map.json`
- Output manifest: `indexes/09_match_manifest.json`
- Current behaviour: Provisional RGB foreground matching against expected set part/color rows.
- Weakness observed: Matching quality is weak; RGB-only matching can pick wrong shapes, and some rows look like correct part/wrong color.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/part_candidate_service.py`, `clean/services/part_crop_normalize_service.py`, `clean/services/confirmed_memory_service.py`, `tools/a2b_clip_match_probe.py`, `clean/services/azure_openai_service.py`
- Should introduce now? later
- Reason: Do not add AI or memory before segmentation and manual config flow are stable. First introduce better candidate generation/normalization, then confirmed memory, then optional AI recommendation for uncertain rows only.

## Stage 10 - Match Audit

- Current V2 file/script: `audit_matches.py`, `create_match_audit_contact_sheet.py`
- Input manifest: `indexes/09_match_manifest.json`, `indexes/11_manual_match_config.json`
- Output manifest: `indexes/10_match_audit.json`; debug output `debug/match_audit/contact_sheet.html`
- Current behaviour: Selects highest/lowest/random provisional matches and displays current top candidate plus manual config overlay.
- Weakness observed: Audit quality is limited by provisional matching; HTML has no save/edit loop and does not yet drive structured manual decisions.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/bag_review_service.py`, `clean/routers/mask_review.py`, mapped `clean/routers/instruction_debug.py`
- Should introduce now? later
- Reason: Keep audit static for now. Add editable review UI only after authority boundaries for `11_manual_match_config.json` are finalized.

## Stage 11 - Manual Config

- Current V2 file/script: manual JSON file only; displayed by `audit_matches.py` and contact sheet generator.
- Input manifest: human-edited config plus `indexes/09_match_manifest.json` context.
- Output manifest: `indexes/11_manual_match_config.json`
- Current behaviour: Provides a human override layer with statuses `pending`, `accepted`, `rejected`, and `needs_ai_check`.
- Weakness observed: No UI/editor, no schema validator, and no AI override recommendation implementation. It is not connected to final review labels.
- Related V1 service/router from `UNMAPPED_V1_AUDIT.md`: `clean/services/bag_review_service.py`, `clean/services/azure_openai_service.py`, `clean/services/confirmed_memory_service.py`, mapped `clean/routers/instruction_debug.py`
- Should introduce now? later
- Reason: Preserve this as the human decision layer. Next work should add validators and AI recommendations as derived files only; never write `training_labels` or auto-accept.

## Priority Recommendations

1. Introduce/adapt `clean/services/gap_scan_service.py` into Stage 3b page-window scoring.
2. Introduce/adapt `clean/services/step_detector_service.py` into Stage 5 so printed `step_number` improves.
3. Introduce/adapt `clean/services/ai_snap_crop_service.py` and `clean/services/part_crop_normalize_service.py` into Stage 8 segmentation/normalization.
4. Introduce/adapt `clean/services/part_candidate_service.py` only after segmentation quality improves.
5. Introduce `clean/services/confirmed_memory_service.py` later, after human decisions in `11_manual_match_config.json` are stable and explicitly promoted.
6. Introduce `clean/services/azure_openai_service.py` only for `needs_ai_check` rows, producing recommendations in future `indexes/12_ai_override_recommendations.json`; never auto-save labels.
