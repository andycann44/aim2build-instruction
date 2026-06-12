# V1 Recovery Port Plan

Purpose: recover the proven V1 behaviour stage-for-stage inside `instruction-v2`.

Principle: V1 is the behaviour source of truth. V2 is only the clean manifest/output wrapper.

This is documentation only. No code was edited, no manifests were regenerated, and no commits were made.

## Port Rules

- Do not approximate bag detection.
- Do not approximate step detection.
- Do not approximate callout crop detection.
- Do not approximate slot segmentation.
- Use current V1 functions where possible.
- If direct import is unsafe or creates debug-system coupling, copy the exact V1 logic into V2 and document the copied source.
- Any deviation from V1 must be documented and approved before coding.
- Existing V2 heuristic/invented logic is not parity unless it calls or exactly copies V1 behaviour.

## Recovery Matrix

| Stage | Exact V1 file/function | Exact V1 inputs | Exact V1 outputs | V2 wrapper/output manifest | V2 currently matches exactly? | Action |
| --- | --- | --- | --- | --- | --- | --- |
| PDF/page loading | `clean/services/debug_service.py::_find_latest_pages_dir_for_set`; `clean/services/debug_service.py::resolve_page_image_path` where used by routes/services | `set_num`; existing rendered V1 debug page folder; requested page number | Stable `page_###.png` path for downstream V1 services | `01_pdf_manifest.json`, `02_page_index.json`, page PNGs under `instruction-v2/pages/` | No. V2 renders its own pages with `phase1_pdf_pages.py`/`render_pdf_pages.mjs`. Behaviour may be compatible but is not exact V1 page loading. | Leave V2 wrapper for local manifests only if page image dimensions/checksums are proven equivalent; otherwise import/copy V1 page resolver semantics into a V2 adapter. |
| Page analyzer | `clean/services/page_analyzer.py::configure_pages_dir`; `clean/services/page_analyzer.py::analyze_page` | Rendered pages dir; page number; `include_image` flag | Page analysis dict: `page_kind`, `panel_found`, `panel_box`, `panel_source`, `shell_found`, `grey_bag_found`, `number_box_found`, `bag_number`, `ocr_raw`, `confidence`, `overview_page`, `build_step_page`, `bag_start_card_*`, `white_box_transition_*` | Future V2 page-analysis manifest, likely `03a_page_analyzer.json` or embedded read-only evidence in Stage 3b | No. V2 does not currently run exact V1 `page_analyzer`. Recent V2 `bag_number_recognizer.mjs` is heuristic and not V1 parity. | Import exact function if dependency/runtime is available; otherwise copy exact V1 logic and keep outputs field-compatible. Replace/bypass V2 heuristic recognizer unless approved as deviation. |
| Bag number recognition | `clean/services/page_analyzer.py::analyze_page`; `clean/services/analyzer_scan_service.py::_build_sequence_scan_row`; `clean/services/gap_scan_service.py::_normalize_analysis_row`; `clean/services/gap_scan_service.py::_score_window_candidate`; `clean/services/gap_scan_service.py::_build_candidate` | `set_num`; page number; pages dir; missing bag number/window bounds; precheck row | Candidate row with `bag_number`, `ocr_raw`, `number_box_found`, `panel_found`, `panel_source`, `shell_found`, `grey_bag_found`, `strong_structure`, `page_kind`, `confidence`, exact/conflict reasons, score | `03_bag_candidates.json`, `03b_bag_gap_review.json`, future dedicated bag-number evidence manifest if needed | No. V2 now has detected numbers, but via approximate JS geometry/OCR, not exact V1 pipeline. | Import/copy exact V1 page analyzer plus analyzer/gap scoring. V2 wrapper should only serialize V1 rows. |
| Bag sequence/gap repair | `clean/services/sequence_service.py::run_sequence_scan`; `clean/services/sequence_service.py::get_sequence_from_truth`; `clean/services/sequence_service.py::get_missing_window`; `clean/services/gap_scan_service.py::scan_gaps`; `clean/services/gap_scan_service.py::scan_gap_for_bag`; routes `clean/routers/gap_scan.py::api_gap_scan`, `clean/routers/gap_review.py::gap_review` | Confirmed bag truth rows from V1 truth store; missing bag number; bounded previous/next confirmed pages; analyzer rows | Missing bag windows; gap candidates ranked by exact/conflicting detected bag number, structure, precheck, green-box exclusions, early bias; human-review UI rows | `03b_bag_gap_review.json`, `04_bag_map.json` after accepted review | No. V2 has candidate-cluster gaps and a custom bootstrap rule; not exact V1 truth/gap workflow. | Import exact V1 sequence/gap services or copy them with a V2-local truth adapter. Preserve human corrections as V2 manifest state, but scoring must come from exact V1 logic. |
| Step detection | `clean/services/step_detector_service.py::detect_steps` | `set_num`; page number; rendered page image resolved through V1 debug service | Dict with detected step boxes, `main_steps`, `sub_steps`, `step_candidates`, classified boxes, confidence/source fields | `05_step_map.json` | No. V2 uses `step_map_scan.mjs`, an approximate separate detector. | Import exact V1 `detect_steps` through a V2 adapter or copy exact logic. V2 should only map V1 output into manifest fields. |
| Step anchor filtering | `clean/routers/instruction_debug.py::_filter_invalid_step_anchor_boxes` plus V1 step detector flow | V1 step boxes/candidates | Filtered/rejected anchors; impossible OCR anchors removed or marked invalid | `05_step_map.json` rejection fields | Partial. V2 added similar sanity filtering, but not exact V1 behaviour. | Copy exact `_filter_invalid_step_anchor_boxes` logic or import it if safe. Record V1 rejection reason in manifest. |
| Callout crop detection | Proven V1 lab flow: `clean/services/step_detector_service.py::detect_steps`; `clean/routers/debug.py::_contact_sheet_step_boxes_from_detected`; `clean/routers/callout_crop_lab.py::_build_step_region`; `clean/routers/instruction_debug.py::_detect_callout_rect_by_edges` | Page image; V1 detected step boxes; built step region; optional debug context | Callout rectangle/crop box around the instruction callout for a step | `06_callout_crop_box_map.json`; crops under `instruction-v2/debug/callout_crop_boxes/` | No. V2 has separate `callout_crop_box_scan.mjs` with partial parity and known manual overrides. | Import/copy exact V1 lab flow into V2. The wrapper writes the manifest and images only. |
| Callout repair/fallback | `clean/routers/instruction_debug.py::_repair_callout_box_candidate_crop`; `_detect_page_level_callout_panels`; `_page_level_callout_candidates_for_fallback`; `_refine_page_level_panel_with_step_geometry`; `_qty_payload_for_page_level_callout_crop` where page-level assignment applies | Page image; candidate crop box; step geometry; page-level panel candidates; qty payload context | Repaired/fallback crop boxes with source labels and confidence/evidence | `06_callout_crop_box_map.json` selection-source fields | No. V2 has some crop priority rules but not exact V1 repair/fallback chain. | Copy exact repair/fallback functions and preserve source labels in V2 manifest. Fresh/final detected crop must beat stale saved crop per V1 behaviour. |
| Qty OCR validation | `clean/routers/instruction_debug.py::_extract_detected_qty_details_from_crop`; `_final_crop_qty_token_is_valid`; `_auto_qty_payload_for_crop`; `_qty_payload_for_page_level_callout_crop`; related imported helpers in `clean/routers/debug.py` | Final callout crop image; step number/context; optional page-level callout source label | Qty details: `qty_text`, `qty_numbers`, token boxes, validation result, source/region, de-duplicated ordered tokens | `07_qty_ocr_map.json` | Partial. V2 has improved sample handling but does not exactly run V1 validation/retry/duplicate logic. | Copy/import exact V1 OCR functions. V2 should not invent token validation; it should serialize V1 output. |
| Qty-anchor slot ownership | V1 slot flow in `clean/routers/instruction_debug.py::auto_mask_slots`; JS/UI helper logic around qty slots; service calls to `create_shape_masks_for_callout_slots`; `normalize_slot_crop_from_qty` | Crop records with V1 qty payload/token boxes; selected/manual qty slot state; crop image | Qty-slot sequence; `next_qty_index`; slot ownership state; selected qty labels; per-slot status | `08_part_segmentation_map.json` plus review model fields if needed | No. V2 segmentation is global component extraction, not qty-anchor slot ownership. | Copy/import exact V1 slot ownership flow. Manifest entries must represent V1-owned slots, not arbitrary components. |
| Slot-aware crop extraction | `clean/routers/instruction_debug.py::_segment_step_callout_slot`; `clean/services/ai_snap_crop_service.py::create_shape_masks_for_callout_slots`; `clean/services/part_crop_normalize_service.py::normalize_slot_crop_from_qty` | Callout crop image; qty token/box; slot index; mask/crop output paths; optional master mask/context | Step-masked cutouts, part cutouts, shape masks, overlays, per-slot source/status/metrics | `08_part_segmentation_map.json`; debug masks/cutouts/overlays | No. V2 emits component segments and cutouts but not exact V1 slot extraction. | Import exact services where possible. If path dependencies make import unsafe, copy exact service logic into V2 with only path/output adapters changed. |
| Needs-review semantics | V1 mask/slot statuses in `auto_mask_slots`, `_segment_step_callout_slot`, `create_shape_masks_for_callout_slots`, `normalize_slot_crop_from_qty`; UI handling in `manual_match_review` | Per-slot segmentation result, mask quality, alpha/foreground metrics, qty slot state, manual review state | Statuses such as `needs_review`, `needs_review_low_alpha`, ignored/unknown/manual slot states; UI-visible review warnings | `08_part_segmentation_map.json`, `10_match_audit.json`, `11_manual_match_config.json` | No. V2 has provisional confidence but not exact V1 needs-review semantics. | Copy exact V1 status semantics. Matching must skip/flag needs-review slots instead of treating them as confident candidates. |

## Stage Notes

### PDF/Page Loading

V2 may keep local PDF and page manifests because they are wrapper infrastructure, but page images must be proven equivalent to V1 rendered pages before downstream V1 functions are expected to behave identically.

Action before coding:

- Compare selected V1/V2 page images for page number, dimensions, and visual content.
- If V1 functions depend on `debug_service` page resolution, create a V2 adapter that points V1 code at `instruction-v2/pages/...`.

### Bag Number Recognition

The recent V2 bag-number detector is not a V1 recovery port. It is useful as a diagnostic experiment, but it is not accepted as parity because it does not run:

- `page_analyzer.analyze_page`
- `analyzer_scan_service._build_sequence_scan_row`
- `gap_scan_service._score_window_candidate`

Recovery action:

- Replace or bypass the V2 detector with exact V1 analyzer/gap row output.
- Keep `detected_bag_number` only as a serialized V1 field.

### Bag Sequence/Gap Repair

V1 gap repair is number-aware and truth-bounded. It does not merely scan for bag-like pages.

Recovery action:

- Build a V2-local truth adapter from V2 accepted review manifests.
- Feed that into exact V1 `sequence_service`/`gap_scan_service` behaviour, or copy the exact logic with the truth adapter substituted.

### Step Detection And Anchors

V2 must stop using an independent step detector once the V1 recovery port begins. The manifest should map V1 step detector output into:

- `page`
- `step_index`
- `step_number`
- `step_box`
- `confidence`
- `rejection_reason`
- `debug_overlay_path`

No synthetic numbering is allowed.

### Callout Crop And Repair

The V1 callout crop flow is already identified:

`step_detector_service.detect_steps`
→ `_contact_sheet_step_boxes_from_detected`
→ `_build_step_region`
→ `_detect_callout_rect_by_edges`
→ repair/fallback functions

Recovery action:

- Port this exact chain.
- Do not keep V2 crop-box heuristics unless documented and approved.

### Qty OCR And Slot Ownership

Qty OCR is not an isolated OCR stage in V1. It feeds slot ownership. Therefore V2 must recover:

- validation
- region retry
- duplicate collapse
- row grouping
- qty token boxes
- qty-slot ownership

Only after that should segmentation run.

### Slot-Aware Segmentation

V1 segmentation is slot-aware. V2's current component segmentation is not behaviour parity.

Recovery action:

- Port `auto_mask_slots`, `_segment_step_callout_slot`, `create_shape_masks_for_callout_slots`, and `normalize_slot_crop_from_qty`.
- V2 manifest paths can differ, but status/ownership behaviour must match V1.

## Required Approval Points

Before coding any recovery port, confirm:

1. Whether V2 may import V1 modules directly even if they reference `clean.services.debug_service`, or whether V2 must copy exact logic and replace only filesystem adapters.
2. Whether V2 should create a page-dir adapter so V1 functions read `instruction-v2/pages/...`.
3. Whether V2 accepted/human review manifests should act as the truth adapter for V1 `sequence_service`/`gap_scan_service`.
4. Whether previous V2 heuristic files should remain as deprecated diagnostics or be ignored by the recovery pipeline.

## Do Not Continue Matching Until

- V1 page analyzer output is available in V2 manifests.
- V1 gap scan behaviour drives bag-number/gap candidates.
- V1 step detector output drives step map.
- V1 callout crop and repair/fallback output drives callout crops.
- V1 qty OCR validation output drives qty anchors.
- V1 slot-aware segmentation output drives cutouts/masks.
- `needs_review` semantics are preserved.

Only then should V2 matching resume.
