# Instruction V2 Checkpoint - 2026-06-02

## Pipeline Stages Implemented

- Stage 0: Set Context -> `indexes/00_set_context.json`
- Stage 1: PDF Manifest -> `indexes/01_pdf_manifest.json`
- Stage 2: Page Index -> `indexes/02_page_index.json`
- Stage 3: Bag Candidates -> `indexes/03_bag_candidates.json`
- Stage 3b: Bag Gap Review -> `indexes/03b_bag_gap_review.json`
- Stage 4: Bag Map -> `indexes/04_bag_map.json`
- Stage 5: Step Box Map -> `indexes/05_step_map.json`
- Stage 6: Callout Crop Box Map -> `indexes/06_callout_crop_box_map.json`
- Stage 7: Quantity OCR Map -> `indexes/07_qty_ocr_map.json`
- Stage 8: Part Segmentation Map -> `indexes/08_part_segmentation_map.json`
- Stage 9: Match Manifest -> `indexes/09_match_manifest.json`
- Stage 10: Match Audit -> `indexes/10_match_audit.json`
- Stage 11: Manual Match Config -> `indexes/11_manual_match_config.json`

## Current Counts

- Set context parts: 439 candidate part/color rows
- PDF pages rendered: 312
- Bag candidates: 33 candidate pages
- Bag gap windows: 2
- Accepted gap corrections: 1
- Bag map ranges: 11
- Step boxes: 540
- Callout crop boxes: 339
- Quantity OCR entries: 339
- Part segmentation entries: 545
- Match entries: 545
- Match candidate pool: 439
- Match audit entries: 30

## Accepted Human Corrections

- `gap_after_bag_03_before_bag_04`
  - Accepted page: 81
  - Observed bag number: 5
  - Review source: human
  - Note: Screenshot/page review shows page 81 clearly starts bag 5.

This correction is applied in `indexes/04_bag_map.json` as a `human_gap_review` source entry with `observed_bag_number: 5`.

## Validator Status

Latest `python3 instruction-v2/validate_pipeline.py` result:

```text
PASS stage 0 set context
PASS stage 1 pdf manifest
PASS stage 2 page index
PASS stage 3 bag candidates
PASS stage 4 bag map
PASS stage 5 step box map
PASS stage 6 callout crop box map
PASS stage 7 qty ocr
PASS stage 8 part segmentation
```

## Known Issues

- Matching quality still weak.
- Matching is RGB-only.
- AI override is not implemented.
- Manual review is still required.
