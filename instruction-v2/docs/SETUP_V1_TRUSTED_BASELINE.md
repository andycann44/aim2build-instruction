# Instruction Reader Trusted Baseline

Status: ACTIVE  
Date: 2026-06-15

## Sole Entrypoint

`instruction-v2/a2b_instruction_run.py`

## Sole Setup Documentation

`instruction-v2/docs/CLEAN_SETUP.md`

## Rule Source

`instruction-v2/docs/RULEBOOK.md`

## Approved Runtime Stages

- `stage0_set_context.py`
- `phase1_pdf_pages.py`
- `stage2_bag_candidates.py`
- `stage3b_bag_gap_review.py`
- `stage3_bag_map.py`
- `stage4_step_map.py`
- `stage5_callout_crop_boxes.py`
- `stage5d_sequence_completeness_diagnostics.py`
- `stage5e_export_crop_cache.py`
- `stage6_qty_ocr.py`
- `stage7_part_segmentation.py`
- `stage8_match.py`

## Excluded From Clean Pipeline

- `stage5_orchestrator.py`
- `stage5f_callout_quality.py`

## Verified Bag Boundaries

15/15 visually verified from `04_bag_map.json`.

| Bag | Start Page |
|---:|---:|
| 1 | 6 |
| 2 | 22 |
| 3 | 39 |
| 4 | 58 |
| 5 | 81 |
| 6 | 104 |
| 7 | 131 |
| 8 | 148 |
| 9 | 164 |
| 10 | 179 |
| 11 | 194 |
| 12 | 213 |
| 13 | 237 |
| 14 | 266 |
| 15 | 274 |

**Source:** `instruction-v2/reports/ALL_BAG_BOUNDARY_VERIFICATION.md`

**Marker crops:** `instruction-v2/reports/all_bag_boundary_crops/`

**Status:** TRUSTED

## Review Flow

```
debug/crop_cache
  -> V1 Manual Match Review UI
  -> debug/training_labels
```

## Current Trust Matrix

| Area | Status |
|---|---|
| PDF Render | TRUSTED |
| Bag Detection | TRUSTED |
| Bag Mapping | TRUSTED |
| Bag Boundaries | TRUSTED |
| Step Map | UNDER INVESTIGATION |
| Crop Map | UNDER INVESTIGATION |
| Segmentation | UNDER INVESTIGATION |
| Match Suggestions | BLOCKED UNTIL PRIOR STAGES PASS |

## Current Focus

Step Map parity.

**Known investigation pages:**

- 59
- 146
- 163
- 214
- 238
- 254
- 277

**Known multi-digit examples:**

- 79
- 239
- 278
- 368
- 369
- 426
- 459
- 483

## Important Rule

A valid build step does not require a parts crop.

Step detection and crop detection are separate systems.

## Future Work Gate

Do not move to:

- matching improvements
- AI ranking
- Azure automation
- R2 page/crop automation
- Bag 6 review

until:

1. Step Map parity is signed off.
2. Crop parity is signed off.
3. Segmentation parity is signed off.
