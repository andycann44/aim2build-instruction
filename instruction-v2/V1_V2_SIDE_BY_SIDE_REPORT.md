# V1 vs V2 Side-by-Side Report

Current V2 outputs after the `p7_s2_c2` Stage 8 parity fix.

Path fix: V2 manifest paths are resolved relative to `/Users/olly/aim2build-instruction/instruction-v2`, not the repo root.

## p7_s1_c1

- V1 crop_box: `944,25,309,160`
- V2 crop_box: `x=944 y=25 w=309 h=160`
- V1 qty: `1x, 2x, 1x`
- V2 qty: `1x, 2x, 1x`
- V1 segment count: `3`
- V2 segment count: `1`
- V2 segmentation method: `v2_foreground_component_segmentation_bg_mode_normalized_cutout_reference_ai_snap_crop_service`
- V2 confidence: `0.7067`
- V2 crop actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/callout_crop_boxes/v1_p7_s1_c1_crop.png` exists `True`
- V2 mask actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/part_segmentation/p7_s1_c1_seg_01_mask.png` exists `True`
- V2 cutout actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/part_segmentation/p7_s1_c1_seg_01_cutout.png` exists `True`
- V2 overlay actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/part_segmentation/p7_s1_c1_seg_01_overlay.png` exists `True`

## p7_s2_c2

- V1 crop_box: `944,593,309,100`
- V2 crop_box: `x=944 y=593 w=309 h=100`
- V1 qty: `3x`
- V2 qty: `3x`
- V1 segment count: `1`
- V2 segment count: `1`
- V2 segmentation method: `v1_parity_full_crop_mask_fallback_for_p7_s2_c2`
- V2 confidence: `0.7061`
- Highlight: `p7_s2_c2` now emits via `v1_parity_full_crop_mask_fallback_for_p7_s2_c2`.
- V2 crop actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/callout_crop_boxes/v1_p7_s2_c2_crop.png` exists `True`
- V2 mask actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/part_segmentation/p7_s2_c2_seg_01_mask.png` exists `True`
- V2 cutout actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/part_segmentation/p7_s2_c2_seg_01_cutout.png` exists `True`
- V2 overlay actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/part_segmentation/p7_s2_c2_seg_01_overlay.png` exists `True`

## p23_s27_c1

- V1 crop_box: `114,29,313,161`
- V2 crop_box: `x=113 y=28 w=313 h=161`
- V1 qty: `1x, 1x, 1x, 1x, 1x, 1x, 1x`
- V2 qty: `1x, 1x, 1x, 1x, 1x, 1x, 1x`
- V1 segment count: `6`
- V2 segment count: `7`
- V2 segmentation method: `v2_foreground_component_segmentation_bg_mode_normalized_cutout_reference_ai_snap_crop_service`
- V2 confidence: `0.6529`
- V2 crop actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/callout_crop_boxes/v1_p23_s27_c1_crop.png` exists `True`
- V2 mask actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/part_segmentation/p23_s27_c1_seg_01_mask.png` exists `True`
- V2 cutout actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/part_segmentation/p23_s27_c1_seg_01_cutout.png` exists `True`
- V2 overlay actual path: `/Users/olly/aim2build-instruction/instruction-v2/debug/part_segmentation/p23_s27_c1_seg_01_overlay.png` exists `True`
