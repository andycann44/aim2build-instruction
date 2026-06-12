# Instruction V2 Source Of Truth

## Authority Rules

Authoritative human decisions:

- `11_manual_match_config.json`

Authoritative AI recommendations:

- future `12_ai_override_recommendations.json`

AI recommendations are never authoritative.

Human decisions always win.

## Stage 0

Set Context

Purpose: Define the expected catalog parts and colors for the instruction set.

Input manifests: none

Output manifests:

- `00_set_context.json`

Can modify? no

Human review required? no

## Stage 1

PDF Manifest

Purpose: Record the selected instruction PDF and its local V2 copy.

Input manifests: none

Output manifests:

- `01_pdf_manifest.json`

Can modify? no

Human review required? no

## Stage 2

Page Index

Purpose: Record rendered instruction pages and page image paths.

Input manifests:

- `01_pdf_manifest.json`

Output manifests:

- `02_page_index.json`

Can modify? no

Human review required? no

## Stage 3

Bag Candidates

Purpose: Detect candidate pages that may indicate bag starts or bag-related page boundaries.

Input manifests:

- `02_page_index.json`

Output manifests:

- `03_bag_candidates.json`

Can modify? no

Human review required? no

## Stage 3b

Bag Gap Review / Missing Bag Window Review

Purpose: Detect missing bag start windows, show candidate pages, and allow human correction before final `04_bag_map.json`.

Input manifests:

- `03_bag_candidates.json`
- `02_page_index.json`

Output manifests:

- `03b_bag_gap_review.json`

Can modify? yes

Human review required? yes

## Stage 4

Bag Map

Purpose: Convert bag candidate pages and human-reviewed gap corrections into final bag page ranges.

Input manifests:

- `03_bag_candidates.json`
- `03b_bag_gap_review.json`

Output manifests:

- `04_bag_map.json`

Can modify? no

Human review required? no

## Stage 5

Step Box Map

Purpose: Detect step boxes on pages within bag ranges.

Input manifests:

- `02_page_index.json`
- `04_bag_map.json`

Output manifests:

- `05_step_box_map.json` or `05_step_map.json`

Can modify? no

Human review required? no

## Stage 6

Callout Crop Box Map

Purpose: Detect callout crop boxes associated with step regions.

Input manifests:

- `02_page_index.json`
- `04_bag_map.json`
- `05_step_box_map.json` or `05_step_map.json`

Output manifests:

- `06_callout_crop_box_map.json`

Can modify? no

Human review required? no

## Stage 7

Quantity OCR Map

Purpose: Extract visible quantity text and quantity token boxes from callout crops.

Input manifests:

- `06_callout_crop_box_map.json`

Output manifests:

- `07_qty_ocr_map.json`

Can modify? no

Human review required? no

## Stage 8

Part Segmentation Map

Purpose: Segment visible part regions inside each callout crop and record masks, cutouts, and overlays.

Input manifests:

- `06_callout_crop_box_map.json`
- `07_qty_ocr_map.json`

Output manifests:

- `08_part_segmentation_map.json`

Can modify? no

Human review required? no

## Stage 9

Match Manifest

Purpose: Produce provisional candidate matches for each segmented cutout.

Input manifests:

- `00_set_context.json`
- `08_part_segmentation_map.json`

Output manifests:

- `09_match_manifest.json`

Can modify? no

Human review required? yes

## Stage 10

Match Audit

Purpose: Select representative provisional matches for visual inspection.

Input manifests:

- `09_match_manifest.json`

Output manifests:

- `10_match_audit.json`

Can modify? no

Human review required? yes

## Stage 11

Manual Match Config

Purpose: Store human decisions and overrides for provisional matches before any automated recommendation is trusted.

Input manifests:

- `09_match_manifest.json`
- `10_match_audit.json`

Output manifests:

- `11_manual_match_config.json`

Can modify? yes

Human review required? yes
