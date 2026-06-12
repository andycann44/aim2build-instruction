# V1 vs V2 Qty OCR and Callout Comparison

Date: 2026-06-02

Goal: identify why V2 callout crops and qty OCR differ from proven V1 outputs, and which differences break slot ownership.

Documentation only. No code or manifests were changed.

## Sources Compared

V1 references:
- `clean/routers/instruction_debug.py::_detect_callout_rect_by_edges`
- `clean/routers/instruction_debug.py::_extract_detected_qty_details_from_crop`
- saved working examples under `debug/ai_training/analysis_bundles/`

V2 manifests:
- `instruction-v2/indexes/06_callout_crop_box_map.json`
- `instruction-v2/indexes/07_qty_ocr_map.json`

Samples:
- page 12 step 11
- page 12 step 12
- page 23 step 27
- page 81

## Summary Finding

V2 is not yet callout/OCR parity with V1.

The strongest divergences are:
- V2 crop boxes sometimes differ from V1 saved working crop boxes.
- V2 qty OCR can miss a V1-proven token.
- V2 qty OCR can emit duplicate boxes for the same printed quantity.
- Page 81 is accepted in the bag map as human-observed bag 5, but current downstream step/callout/OCR manifests have no page 81 callout entries.

These differences break slot ownership because V1's slot segmentation depends on clean qty anchors. If the crop is shifted/truncated, if a qty token is missing, or if duplicate qty tokens are retained, V2 cannot reliably map a visual part component to a single intended slot.

## V1 Behaviour Reference

### Callout Crop Detection

Reference function:
- `clean/routers/instruction_debug.py::_detect_callout_rect_by_edges`

V1 working callout crop flow detects a rectangular instruction callout region around step material lists, using edge/rectangle evidence and crop validation from the debug/callout lab path. Saved examples show crop boxes attached to OCR metadata in `analysis_bundles`.

### Qty OCR

Reference function:
- `clean/routers/instruction_debug.py::_extract_detected_qty_details_from_crop`

V1 saved examples include:
- local crop-coordinate qty token boxes
- `source_region`
- confidence values
- crop box
- crop image size

Those fields matter because V1 segmentation later groups qty tokens into rows and assigns components to qty anchors.

## V2 Behaviour Reference

### Callout Crop Boxes

V2 reads/writes:
- `06_callout_crop_box_map.json`

The recorded source function string is:
- `clean/routers/callout_crop_lab.py flow: step_detector_service.detect_steps -> _contact_sheet_step_boxes_from_detected -> _build_step_region -> _detect_callout_rect_by_edges`

### Qty OCR

V2 reads/writes:
- `07_qty_ocr_map.json`

The recorded source string is:
- `tesseract_cli_variants_reference:_extract_qty_tokens_from_image->_extract_detected_qty_details_from_crop`

V2 records qty text, numbers, token boxes, confidence, and source, but currently does not expose row grouping or duplicate-collapse decisions in the manifest.

## Sample 1 - Page 12 Step 11

### Callout Crop Box

V1 saved working example:
- analysis bundle: `debug/ai_training/analysis_bundles/70618_bag1_p12_s11_c1/metadata.json`
- crop box: `[29, 29, 313, 97]`
- crop dimensions: `313 x 97`

V2:
- crop path: `instruction-v2/debug/callout_crop_boxes/bag_01_page_012_step_11_001_crop.png`
- crop box: `[28, 28, 282, 97]`
- crop dimensions: `282 x 97`
- confidence: `0.9047`

### Qty OCR

V1 qty text and boxes:
- `8x` at `[30, 69, 16, 11]`, source `final_crop_full`, confidence `null`
- `4x` at `[143, 69, 17, 11]`, source `final_crop_full:adaptive_s4_psm11`, confidence `95.0`
- `4x` at `[234, 69, 17, 11]`, source `final_crop_full:adaptive_s4_psm11`, confidence `96.0`
- row grouping: one row by shared y/cy around `69/74`
- duplicate handling: no duplicates in saved V1 metadata

V2 qty text and boxes:
- `8x` at `[31, 71, 17, 12]`, confidence `0.9469`
- `4x` at `[144, 71, 17, 12]`, confidence `0.5581`
- `4x` at `[236, 71, 17, 12]`, confidence `0.0451`
- qty numbers: `[8, 4, 4]`
- row grouping: not recorded; boxes imply one row around y `71`
- duplicate handling: no duplicates, but the third token is extremely low confidence
- manifest confidence: `0.5167`

### Divergence

- V2 crop is 31 px narrower than V1.
- V2 qty boxes are shifted by about 1-2 px relative to V1, likely due to crop origin and OCR variant differences.
- V2 third `4x` confidence is `0.0451`, while V1 confidence is `96.0`.

### Slot Ownership Impact

The slot count is still correct, but V2 weakens the third qty anchor. Slot-aware segmentation can still attempt three slots, but the third slot should be flagged lower confidence unless V2 recovers the V1 OCR confidence/source behaviour.

## Sample 2 - Page 12 Step 12

### Callout Crop Box

V1 saved working example:
- analysis bundle: `debug/ai_training/analysis_bundles/70618_bag1_p12_s12_c2/metadata.json`
- crop box: `[29, 596, 313, 97]`
- crop dimensions: `313 x 97`

V2:
- crop path: `instruction-v2/debug/callout_crop_boxes/bag_01_page_012_step_12_002_crop.png`
- crop box: `[28, 595, 313, 97]`
- crop dimensions: `313 x 97`
- confidence: `0.9042`

### Qty OCR

V1 qty text and boxes:
- `6x` at `[132, 57, 16, 11]`, source `final_crop_full:adaptive_s5_psm11`, confidence `96.0`
- qty numbers: `[6]`
- row grouping: one row, one token
- duplicate handling: no duplicates

V2 qty text and boxes:
- qty text: `[]`
- qty numbers: `[]`
- qty token boxes: `[]`
- confidence: `0`

### Divergence

- Crop box is nearly identical, only shifted `-1 px` in x/y compared with V1.
- Despite near-identical crop geometry, V2 misses the V1-proven `6x` token entirely.

### Slot Ownership Impact

This is a hard slot ownership failure. V1 has one high-confidence qty anchor; V2 has no anchor, so Stage 8 cannot know that the visible part belongs to a `6x` slot. Whole-callout component segmentation may still emit a component, but it is no longer linked to a quantity slot.

## Sample 3 - Page 23 Step 27

### Callout Crop Box

V1 saved working example:
- analysis bundle: `debug/ai_training/analysis_bundles/70618_bag2_p23_s27_c1/metadata.json`
- crop box: `[113, 3, 313, 193]`
- crop dimensions: `313 x 193`

V2:
- crop path: `instruction-v2/debug/callout_crop_boxes/bag_01_page_023_step_27_023_crop.png`
- crop box: `[113, 28, 313, 161]`
- crop dimensions: `313 x 161`
- confidence: `0.9984`

### Qty OCR

V1 qty text and boxes:
- `1x` at `[11, 54, 14, 11]`, confidence `94.0`, source `final_crop_full:adaptive_s4_psm6`
- `1x` at `[118, 50, 14, 16]`, confidence `93.0`, source `final_crop_up_right:adaptive_s4_psm6`
- `1x` at `[211, 54, 14, 11]`, confidence `94.0`, source `final_crop_full:adaptive_s4_psm6`
- `1x` at `[11, 135, 14, 11]`, confidence `94.0`, source `final_crop_full:adaptive_s4_psm6`
- `1x` at `[68, 129, 28, 17]`, confidence `61.0`, source `final_crop_right_expanded:adaptive_s5_psm6`
- `1x` at `[139, 135, 14, 11]`, confidence `94.0`, source `final_crop_full:adaptive_s4_psm6`
- `1x` at `[221, 135, 14, 11]`, confidence `93.0`, source `final_crop_full:adaptive_s4_psm6`
- qty numbers: seven `1` values
- row grouping: two rows, approximately y `50-54` and y `129-135`
- duplicate handling: no near-identical duplicate boxes retained in saved V1 metadata

V2 qty text and boxes:
- `1x` at `[12, 56, 14, 12]`, confidence `0.9544`
- `1x` at `[13, 56, 13, 11]`, confidence `0.9212`
- `1x` at `[120, 56, 13, 11]`, confidence `0.9574`
- `1x` at `[120, 56, 14, 12]`, confidence `0.9339`
- `1x` at `[213, 56, 13, 11]`, confidence `0.9578`
- `1x` at `[213, 56, 14, 12]`, confidence `0.9184`
- `1x` at `[12, 137, 14, 12]`, confidence `0.9544`
- `1x` at `[13, 137, 13, 11]`, confidence `0.9212`
- `1x` at `[69, 137, 14, 12]`, confidence `0.9556`
- `1x` at `[141, 137, 13, 11]`, confidence `0.9124`
- `1x` at `[141, 137, 14, 12]`, confidence `0.9051`
- `1x` at `[223, 137, 14, 12]`, confidence `0.95`
- qty numbers: twelve `1` values
- row grouping: not recorded; boxes imply two rows around y `56` and y `137`
- duplicate handling: near-identical duplicate boxes are retained
- manifest confidence: `0.9368`

### Divergence

- V2 crop starts 25 px lower and is 32 px shorter than V1.
- V2 records 12 qty boxes where V1 records 7.
- V2 retains duplicate boxes at essentially the same locations:
  - `[12,56]` and `[13,56]`
  - `[120,56]` repeated
  - `[213,56]` repeated
  - `[12,137]` and `[13,137]`
  - `[141,137]` repeated
- V2 has high token confidence despite duplicate retention.

### Slot Ownership Impact

This is a severe slot ownership failure. V1 has seven slot anchors; V2 has twelve anchors for the same visual material list. A slot-aware segmentation stage would over-create slots unless it de-duplicates geometrically overlapping OCR tokens before component assignment.

## Sample 4 - Page 81

### Bag Map Context

V2 bag map:
- page 81 is accepted as observed bag 5.
- source: `human_gap_review`
- evidence window: `gap_after_bag_03_before_bag_04`
- notes: `Screenshot/page review shows page 81 clearly starts bag 5.`

### V2 Step/Callout/OCR State

Current V2 `05_step_map.json` page summary:
- page: `81`
- bag: `5`
- step candidate count: `0`
- debug overlay path: `null`

Current V2 callout/OCR manifests:
- `06_callout_crop_box_map.json`: no entries with `page == 81`
- `07_qty_ocr_map.json`: no entries with `page == 81`

Saved V1 working example:
- no matching V1 `analysis_bundles/*p81*` metadata file was found in this pass.

### Divergence

Page 81 is known to be important because human review accepted it as bag 5, but current downstream V2 has no step/callout/qty outputs for it. The screenshot shows page 81 as a bag-start/add-bag instruction page rather than a normal material callout step, so zero callout crops may be acceptable only if V2 records it intentionally as a non-step/bag-start page.

### Slot Ownership Impact

This does not directly break a slot, because page 81 may not contain ordinary qty callouts. It does break traceability: a human-accepted bag-start page currently disappears from the callout/OCR chain except as a bag-map range boundary. V2 needs an explicit page classification so this is not confused with a failed step/callout detector.

## Divergence Matrix

| Sample | Crop divergence | Qty divergence | Duplicate issue | Slot ownership impact |
|---|---|---|---|---|
| page 12 step 11 | V2 width `282` vs V1 `313` | Same texts, weaker third confidence | No | Medium: third anchor is weak. |
| page 12 step 12 | Near-identical crop | V2 misses V1 `6x` | No | High: no qty anchor for the slot. |
| page 23 step 27 | V2 lower/shorter crop | V2 has 12 tokens vs V1 7 | Yes | High: over-created slot anchors. |
| page 81 | Accepted bag page has no downstream callout/OCR entry | Not applicable | Not applicable | Traceability issue unless explicitly classified as non-callout page. |

## What Breaks Slot Ownership

1. Missing qty token:
   - Page 12 step 12 is the clearest example.
   - A visible part without a qty anchor cannot become a V1-style slot.

2. Duplicate qty tokens:
   - Page 23 is the clearest example.
   - Duplicate anchors inflate expected slot count and make component assignment ambiguous.

3. Crop geometry drift:
   - Page 12 step 11 and page 23 show V2 boxes that differ from V1.
   - Even small crop shifts can move OCR boxes and component boxes; large height/width differences can remove useful row context.

4. Missing downstream classification:
   - Page 81 is accepted by human gap review but has no callout/OCR record.
   - V2 should record whether this page is intentionally skipped as a bag-start/non-step page.

## Recommended Parity Requirements

Before changing segmentation or matching again:

1. Callout crops:
   - Compare V2 crop boxes against V1 saved crop boxes for known samples.
   - Require exact match or documented improved crop evidence.

2. Qty OCR:
   - Restore V1-style adaptive OCR variants and source metadata for hard crops.
   - Record row grouping in `07_qty_ocr_map.json`.
   - Add geometric duplicate suppression before Stage 8.

3. Page classification:
   - For accepted bag-start pages like page 81, record intentional non-callout/bag-start status so zero step/callout entries are not silent failures.

4. Slot ownership:
   - Stage 8 should consume de-duplicated, row-grouped qty anchors.
   - If a V1-proven anchor is missing, the affected crop should be marked needs-review instead of treated as ordinary component segmentation.
