# V1 vs V2 Segmentation Comparison

Date: 2026-06-02

Purpose: explain why current V2 part segmentation can be worse than the old working V1 output, without changing code or regenerating manifests.

## Scope

V2 inputs inspected:
- `instruction-v2/indexes/06_callout_crop_box_map.json`
- `instruction-v2/indexes/07_qty_ocr_map.json`
- `instruction-v2/indexes/08_part_segmentation_map.json`
- `instruction-v2/debug/part_segmentation/`

V1 references inspected read-only:
- `clean/services/ai_snap_crop_service.py`
- `clean/services/part_crop_normalize_service.py`
- `clean/routers/instruction_debug.py`
- saved V1 outputs under `debug/ai_training/analysis_bundles/`, `debug/ai_training/part_cutouts/`, `debug/ai_training/full_crop_masks/`, `debug/ai_training/bg_masks/`, and related overlay folders

No code or manifest regeneration was performed for this report.

## Short Finding

V2 is not actually equivalent to the old working V1 slot segmentation flow.

V1's good path is qty-anchor/slot-aware:
- Build a full callout foreground mask.
- Scrub qty text globally.
- Group qty tokens into rows.
- Assign foreground components to qty anchors/columns.
- Use fallbacks such as master-island and slot-window extraction.
- Save tight per-slot cutouts.
- Optionally run a step-callout slot segmentation path after `auto-mask-slots`.

Current V2 is whole-callout component extraction:
- Build one foreground mask over the callout crop.
- Erase OCR qty boxes.
- Run component extraction and simple filtering/merging.
- Emit every surviving component as a segment.
- Normalize each component to a square transparent cutout.

That means V2 can produce nice-looking individual blobs, but it does not know which component belongs to which qty slot. It also cannot apply V1's slot ownership, master-island fallback, duplicate-qty handling, or per-slot rejection logic. Matching quality then suffers because V2 may output partial shapes, merged shapes, text remnants, or components from an input crop that differs from V1's proven crop.

## Code Path Comparison

### V1 foreground mask method

Primary reference: `clean/services/ai_snap_crop_service.py`

Relevant functions:
- `_estimate_background_bgr`
- `_lab_dist`
- `_foreground_mask_for_image`
- `_recover_grey_foreground`
- `_recover_light_foreground`
- `_rebuild_sharp_mask`
- `_select_component_near_slot_anchor`
- `create_shape_masks_for_callout_slots`

V1 estimates background from weighted border pixels plus smooth low-gradient interior pixels, then thresholds in LAB space. It has special recovery passes for grey/light/transparent foreground and contour rebuilding. The important point is that V1 uses this mask as a master callout mask, then assigns components to slots using qty-token anchors.

Older V1 fallback in `clean/routers/instruction_debug.py` also exists:
- `_sam2_segment_crop`
- `_cv_fallback_mask`
- `_hybrid_segment_crop`

That route attempts SAM2 on individual slot windows, rejects bad masks by coverage/largest-component guards, then falls back to colour distance.

### V2 foreground mask method

Current V2 reference: `instruction-v2/part_segmentation_scan.mjs`

V2 estimates a dominant background from border and smooth pixels, thresholds by RGB distance/saturation/luminance, erases qty boxes, removes text-like leftovers, runs open/close/fill, extracts connected components, and merges nearby components.

This is useful, but it is still global component extraction. It lacks V1's qty-anchor assignment and its per-slot fallback chain.

### V1 component filtering rules

Primary reference: `clean/services/ai_snap_crop_service.py:create_shape_masks_for_callout_slots`

Observed V1 logic:
- Normalize qty token boxes and group them into rows.
- Scrub all qty token glyphs before component analysis.
- Build column bounds from adjacent qty-token centers.
- Extract global components from the cleaned master mask.
- Reject tiny/thin components.
- Reject assignment frame/ring components.
- Reject text strips, frame outlines, large blocks, and non-part components.
- Assign components using qty-anchor proximity.
- For repeated qty labels, avoid unsafe global assignment and use fallback behavior.
- If global assignment fails, try master-island fallback or window fallback.
- Return a slot as `needs_review` instead of pretending a poor component is valid.

V1 output carries slot-specific fields such as:
- `slot_index`
- `status`
- `slot_window`
- `component_box`
- `function_path_used`
- `part_cutout_path`
- `shape_mask_path`
- `alpha_pixel_count`
- `actual_saved_cutout_size`

### V2 component filtering rules

Current V2 output fields inspected:
- `segment_box`
- `mask_path`
- `cutout_path`
- `overlay_path`
- `confidence`
- `metrics.foreground_pixels`
- `metrics.density`
- `metrics.merged_component_count`
- `metrics.normalized_box`

V2 keeps plausible connected components after generic filters and assigns each surviving component a `segment_index`. It does not score components against qty anchors and does not know whether a segment is the part for slot 1, slot 2, etc. It also has no `needs_review_no_component` equivalent at the slot level.

## Sample Comparisons

### Sample 1: page 12 step 11

V2 input:
- crop path: `instruction-v2/debug/callout_crop_boxes/bag_01_page_012_step_11_001_crop.png`
- crop box: `[x=28, y=28, w=282, h=97]`
- image size: `282 x 97`
- qty OCR boxes:
  - `8x` at `[31, 71, 17, 12]`, confidence `0.9469`
  - `4x` at `[144, 71, 17, 12]`, confidence `0.5581`
  - `4x` at `[236, 71, 17, 12]`, confidence `0.0451`
- V2 segments:
  - seg 1 box `[145,15,58,53]`, normalized box `[138,8,72,67]`, cutout `96 x 96`
  - seg 2 box `[30,24,81,42]`, normalized box `[20,14,101,62]`, cutout `101 x 101`
  - seg 3 box `[235,35,44,31]`, normalized box `[229,29,53,43]`, cutout `96 x 96`

V1 saved reference:
- analysis bundle: `debug/ai_training/analysis_bundles/70618_bag1_p12_s11_c1/`
- original crop: `debug/ai_training/analysis_bundles/70618_bag1_p12_s11_c1/original_crop.png`
- crop box from metadata: `[29, 29, 313, 97]`
- image size: `313 x 97`
- qty OCR boxes:
  - `8x` at `[30,69,16,11]`
  - `4x` at `[143,69,17,11]`
  - additional `4x` token present in bundle metadata
- V1 output example:
  - `slot_0_cutout.png`, size `24 x 21`
  - baseline slot candidate masks and qty-scrubbed masks are saved in the same bundle

Where V2 diverges:
- V2 crop is 31 px narrower than V1 (`282` vs `313`), so component geometry is not the same starting point.
- V2 keeps three global components, while V1 treats these as qty-anchored slots.
- V2 emits normalized square-ish cutouts for each component; V1 saved tight slot cutouts.
- V2's third qty token has very low confidence (`0.0451`), but V2 still treats the component set globally rather than deferring that slot or using V1's fallback review state.

### Sample 2: page 12 step 12

V2 input:
- crop path: `instruction-v2/debug/callout_crop_boxes/bag_01_page_012_step_12_002_crop.png`
- crop box: `[x=28, y=595, w=313, h=97]`
- image size: `313 x 97`
- qty OCR boxes: none detected
- V2 segment:
  - seg 1 box `[133,25,47,31]`
  - normalized box `[127,19,59,43]`
  - cutout size `96 x 96`
  - confidence `0.7837`

V1 saved reference:
- analysis bundle: `debug/ai_training/analysis_bundles/70618_bag1_p12_s12_c2/`
- original crop size: `313 x 97`
- metadata crop box: `[29, 596, 313, 97]`
- qty OCR box:
  - `6x` at `[132,57,16,11]`, confidence `96.0`
- V1 output example:
  - `slot_0_cutout.png`, size `25 x 21`

Where V2 diverges:
- The crop size is aligned, but V2 missed the qty token entirely.
- Without the qty token, V2 cannot build a slot anchor.
- V1 has a high-confidence `6x` qty anchor and can associate the nearby component with slot 0.
- V2 emits a component but loses the quantity-slot relationship that the review/matching flow needs.

### Sample 3: page 23 step 27

V2 input:
- crop path: `instruction-v2/debug/callout_crop_boxes/bag_01_page_023_step_27_023_crop.png`
- crop box: `[x=113, y=28, w=313, h=161]`
- image size: `313 x 161`
- qty OCR boxes: 12 tokens detected, but several are duplicates at the same coordinates:
  - duplicate `1x` around `[12,56]`
  - duplicate `1x` around `[120,56]`
  - duplicate `1x` around `[213,56]`
  - duplicate `1x` around `[12,137]`
  - duplicate `1x` around `[141,137]`
  - additional tokens at `[69,137]` and `[223,137]`
- V2 example segments:
  - seg 1 box `[119,13,30,38]`, cutout `96 x 96`
  - seg 2 box `[211,17,23,34]`
  - seg 3 box `[11,22,48,29]`

V1 saved reference:
- analysis bundle: `debug/ai_training/analysis_bundles/70618_bag2_p23_s27_c1/`
- original crop size: `313 x 193`
- metadata crop box: `[113, 3, 313, 193]`
- V1 output examples:
  - `slot_0_cutout.png`, size `40 x 51`
  - additional saved slot cutouts: `slot_2_cutout.png`, `slot_3_cutout.png`, `slot_4_cutout.png`, `slot_5_cutout.png`, `slot_6_cutout.png`

Where V2 diverges:
- V2 crop starts 25 px lower and is 32 px shorter than the V1 reference (`y=28,h=161` vs `y=3,h=193`).
- V2 duplicate qty OCR boxes inflate slot evidence and can create duplicate/ambiguous slot expectations.
- V1 explicitly had repeated-qty handling and slot assignment logic for this exact pattern.
- V2 simply emits foreground components, so it can miss slot ownership even when the visual component itself is reasonable.

### Bad audit example: lowest-confidence V2 segment

V2 lowest-confidence segment observed:
- crop id: `bag_10_page_260_step_464_454`
- segment: `7`
- confidence: `0.5926`
- segment box: `[13,105,71,34]`
- normalized box: `[4,96,89,52]`
- density: `0.2672`
- cutout: `debug/part_segmentation/bag_10_page_260_step_464_454_seg_07_cutout.png`
- mask: `debug/part_segmentation/bag_10_page_260_step_464_454_seg_07_mask.png`
- overlay: `debug/part_segmentation/bag_10_page_260_step_464_454_seg_07_overlay.png`

Why this is risky:
- V2 confidence is still above 0.59, so the current confidence scale is not a strong bad-segment detector.
- The density is low and the normalized box is pushed near the lower crop boundary (`y=96`), suggesting partial/edge-adjacent extraction risk.
- V1's slot-aware flow would have more chances to reject this as `needs_review`, attach it to a specific qty anchor, or avoid assigning it if the component does not satisfy anchor proximity.

## Output Shape Comparison

V1 examples:
- page 12 step 11 `slot_0_cutout.png`: `24 x 21`
- page 12 step 12 `slot_0_cutout.png`: `25 x 21`
- page 23 step 27 `slot_0_cutout.png`: `40 x 51`

V2 examples:
- page 12 step 11 seg 1 cutout: `96 x 96`
- page 12 step 12 seg 1 cutout: `96 x 96`
- page 23 step 27 seg 1 cutout: `96 x 96`

Interpretation:
- V2 normalization may be helpful for embedding/matching later, but it has changed the visual distribution from V1's tight cutout outputs.
- The immediate quality problem is not square normalization alone; it is that V2 reaches normalization through a less slot-aware extraction path.

## Likely Root Causes

1. V2 Stage 5 callout crop boxes are not always identical to V1 proven crops.
   - Page 12 step 11 lost width.
   - Page 23 lost top/bottom context.

2. V2 Stage 6 qty OCR is weaker or differently de-duplicated for key crops.
   - Page 12 step 12 missed the `6x` token that V1 found.
   - Page 23 duplicated many `1x` tokens at the same locations.

3. V2 Stage 8 lacks V1's qty-anchor slot assignment.
   - V1 assigns components to slot windows and qty anchors.
   - V2 assigns `segment_index` to surviving connected components.

4. V2 lacks V1's slot-level fallback/review states.
   - V1 can return `needs_review_no_component`, `needs_review_low_alpha`, and other slot statuses.
   - V2 currently emits components and confidence metrics, but not slot-level failure semantics.

5. V2's confidence is not calibrated as a review gate.
   - Lowest observed V2 confidence is still about `0.59`.
   - Bad or partial shapes can look numerically acceptable.

## Recommendation

Do not tune matching yet.

Next useful implementation should be a Stage 8 parity pass, not a ranking pass:
- Keep the V2 manifest schema.
- Reintroduce V1's qty-token row grouping.
- De-duplicate qty boxes before segmentation.
- Use qty anchors/column bounds to assign components.
- Add a V2 equivalent of V1 `needs_review` slot semantics, without writing labels.
- Compare V2 crop boxes against V1 callout crop evidence for known samples before segmentation.
- Only normalize after slot ownership has been decided.

This would make V2 segmentation closer to the old working V1 flow while preserving V2's local-manifest-first shape.
