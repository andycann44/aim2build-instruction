# Instruction V2 Migration Ledger

This ledger marks V1 files/functions that have been used as references or adapted into `instruction-v2`.

Do not delete V1 files from this ledger yet. V2 is local-manifest-first and currently copies/adapts equivalent logic rather than importing these V1 modules directly.

## Marker Key

- `@@@ USED_IN_V2`: V2 has used this file/function as a reference or has adapted equivalent logic.
- `@@@ STILL_REFERENCED_BY_V1`: V1 routes/services still import or call this file/function.
- `@@@ READY_TO_ARCHIVE_LATER`: Candidate for later archive only after V1 references are retired and V2 replacement is validated.
- `@@@ DO_NOT_DELETE_YET`: Not safe to delete now.

## Entries

### `clean/services/page_analyzer.py`

Markers:

- `@@@ USED_IN_V2`
- `@@@ STILL_REFERENCED_BY_V1`
- `@@@ DO_NOT_DELETE_YET`

V2 stage using equivalent logic:

- Stage 2 Bag Candidates: page-level visual scan concepts are adapted into `bag_candidate_scan.mjs`.
- Stage 3b Bag Gap Review contact sheet: page thumbnails are read from `02_page_index.json`, not from V1.

V2 import or copied/adapted logic:

- Copied/adapted logic. V2 does not import `clean/services/page_analyzer.py`.

Grep evidence of remaining V1 references:

- `clean/services/analyzer_scan_service.py:1:from clean.services import page_analyzer, debug_service, precheck_service`
- `clean/routers/workflow.py:20:from clean.services.page_analyzer import analyze_page, configure_pages_dir`
- `clean/routers/debug.py:29:    page_analyzer,`
- `clean/routers/sequence.py:3:from clean.services import sequence_service, page_analyzer, debug_service`

safe_to_delete: no

Reason:

- Still used by V1 analyzer, sequence, workflow, and debug routes.

### `clean/services/sequence_service.py`

Markers:

- `@@@ USED_IN_V2`
- `@@@ STILL_REFERENCED_BY_V1`
- `@@@ DO_NOT_DELETE_YET`

V2 stage using equivalent logic:

- Stage 2 Bag Candidates and Stage 3b Bag Gap Review: sequence/gap-review concepts are represented as manifests rather than mutable V1 truth.

V2 import or copied/adapted logic:

- Copied/adapted logic. V2 does not import `clean/services/sequence_service.py`.

Grep evidence of remaining V1 references:

- `clean/services/workflow_service.py:1:from clean.services import analyzer_scan_service, sequence_service, truth_service`
- `clean/services/gap_scan_service.py:5:from clean.services import debug_service, precheck_service, sequence_service, truth_service`
- `clean/routers/gap_review.py:6:from clean.services import gap_scan_service, sequence_service`
- `clean/routers/sequence.py:3:from clean.services import sequence_service, page_analyzer, debug_service`

safe_to_delete: no

Reason:

- Still used by V1 workflow, gap scan/review, and sequence routes.

### `clean/services/step_sequence_bag_service.py`

Markers:

- `@@@ USED_IN_V2`
- `@@@ STILL_REFERENCED_BY_V1`
- `@@@ DO_NOT_DELETE_YET`

V2 stage using equivalent logic:

- Stage 2 Bag Candidates: candidate bag-start detection and sequence prepass ideas.
- Stage 3 Bag Map: candidate clusters become page ranges.
- Stage 3b Bag Gap Review: missing bag-start windows are now represented in `03b_bag_gap_review.json`.

V2 import or copied/adapted logic:

- Copied/adapted logic. V2 does not import `clean/services/step_sequence_bag_service.py`.

Grep evidence of remaining V1 references:

- `clean/services/step_sequence_openai_bag_service.py:6:from clean.services import debug_service, step_sequence_bag_service`
- `clean/routers/step_bag_scan.py:3:from clean.services import step_sequence_bag_service`
- `clean/routers/workflow.py:18:    step_sequence_bag_service,`
- `clean/routers/debug.py:25:    step_sequence_bag_service,`

safe_to_delete: no

Reason:

- Still used by V1 step bag scan, OpenAI verifier, workflow, and debug routes.

### `clean/services/step_sequence_openai_bag_service.py`

Markers:

- `@@@ USED_IN_V2`
- `@@@ STILL_REFERENCED_BY_V1`
- `@@@ DO_NOT_DELETE_YET`

V2 stage using equivalent logic:

- Stage 2 Bag Candidates: documented as an optional V1 verifier reference only.
- No V2 AI verifier implementation exists yet.

V2 import or copied/adapted logic:

- Reference only. V2 does not import `clean/services/step_sequence_openai_bag_service.py`.

Grep evidence of remaining V1 references:

- `clean/routers/step_bag_openai_scan.py:3:from clean.services import step_sequence_openai_bag_service`
- `clean/routers/step_bag_openai_scan.py:13:    model: str = Query(step_sequence_openai_bag_service.DEFAULT_MODEL),`
- `clean/routers/step_bag_openai_scan.py:16:        return step_sequence_openai_bag_service.scan_step_bag_sequence_openai_verify(`

safe_to_delete: no

Reason:

- Still used by V1 OpenAI bag scan route. V2 has not replaced AI verification.

### `clean/routers/callout_crop_lab.py`

Markers:

- `@@@ USED_IN_V2`
- `@@@ STILL_REFERENCED_BY_V1`
- `@@@ DO_NOT_DELETE_YET`

V2 stage using equivalent logic:

- Stage 5 Callout Crop Box Map: `callout_crop_box_scan.mjs` adapts the proven callout crop-box flow.

V2 import or copied/adapted logic:

- Copied/adapted logic. V2 does not import `clean/routers/callout_crop_lab.py`.

Grep evidence of remaining V1 references:

- `clean/main.py:24:# app.include_router(callout_crop_lab.router)`
- `clean/routers/callout_crop_lab.py:101:def _build_step_region(`
- `clean/routers/callout_crop_lab.py:250:def callout_crop_lab(`
- `clean/routers/callout_crop_lab.py:316:            edge_box = _detect_callout_rect_by_edges(`
- `clean/routers/callout_crop_lab.py:384:                qty_result = _extract_detected_qty_details_from_crop(final_crop)`

safe_to_delete: no

Reason:

- Even though the router include is currently commented, this file is still the documented/proven lab reference for V2 Stage 5 and imports V1 debug/instruction helpers.

### `clean/routers/debug.py`

Markers:

- `@@@ USED_IN_V2`
- `@@@ STILL_REFERENCED_BY_V1`
- `@@@ DO_NOT_DELETE_YET`

V2 stage using equivalent logic:

- Stage 2 Bag Candidates: `_resolve_bag_page_range` and bag/page analyzer flows were used as reference.
- Stage 4 Step Box Map: `_contact_sheet_step_boxes_from_detected` was used as reference.
- Stage 6 Qty OCR: `_extract_detected_qty_details_from_crop` was used as reference.

V2 import or copied/adapted logic:

- Copied/adapted logic. V2 does not import `clean/routers/debug.py`.

Grep evidence of remaining V1 references:

- `clean/main.py:3:from clean.routers import home, analyzer_scan, instruction_debug, sequence, debug, load_set, set_scan, workflow, debug_truth, step_debug, step_bag_scan, step_bag_openai_scan, gap_scan`
- `clean/routers/callout_crop_lab.py:23:from clean.routers.debug import (`
- `clean/routers/instruction_debug.py:40:from clean.routers.debug import (`
- `clean/routers/debug.py:387:def _contact_sheet_step_boxes_from_detected(`
- `clean/routers/debug.py:1712:def _extract_detected_qty_details_from_crop(crop_img) -> Dict[str, Any]:`
- `clean/routers/debug.py:2331:def _resolve_bag_page_range(set_num: str, bag: int) -> Tuple[List[int], int, int]:`

safe_to_delete: no

Reason:

- Still mounted by V1 and imported by V1 callout/instruction routes. Also provides V1 debug page-image and bag truth routes.

### `clean/routers/instruction_debug.py`

Markers:

- `@@@ USED_IN_V2`
- `@@@ STILL_REFERENCED_BY_V1`
- `@@@ DO_NOT_DELETE_YET`

V2 stage using equivalent logic:

- Stage 5 Callout Crop Box Map: `_detect_callout_rect_by_edges` reference.
- Stage 6 Qty OCR: `_extract_detected_qty_details_from_crop` reference.
- Stage 7 Part Segmentation: `auto_mask_slots` / slot segmentation flow reference.
- Stage 8 Matching: `buildability_clip_suggest` reference, though V2 currently uses provisional RGB-only matching.
- Stage 9/11 Human Review: `manual_match_review`, `save_label`, and manual config/review concepts.

V2 import or copied/adapted logic:

- Copied/adapted logic. V2 does not import `clean/routers/instruction_debug.py`.

Grep evidence of remaining V1 references:

- `clean/main.py:22:app.include_router(instruction_debug.router)`
- `clean/routers/callout_crop_lab.py:31:from clean.routers.instruction_debug import (`
- `clean/routers/instruction_debug.py:442:def _detect_callout_rect_by_edges(`
- `clean/routers/instruction_debug.py:1917:def _extract_detected_qty_details_from_crop(crop_img) -> Dict[str, Any]:`
- `clean/routers/instruction_debug.py:4554:async def save_label(req: Request):`
- `clean/routers/instruction_debug.py:13722:async def auto_mask_slots(req: Request):`
- `clean/routers/instruction_debug.py:14073:async def buildability_clip_suggest(req: Request):`
- `clean/routers/instruction_debug.py:20438:def manual_match_review(`

safe_to_delete: no

Reason:

- Still mounted by V1 and contains active review, save-label, segmentation, and matching routes. V2 has not replaced the human review workflow with final training-label writes.
