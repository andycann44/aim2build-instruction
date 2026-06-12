# Bag 2 Parity Sign-Off

STATUS = BAG2_PARITY_APPROVED

## Sources Read

- `debug/bag_truth.db`
- `debug/crop_cache/70618_bag2.json`
- `debug/training_labels/70618_bag2.json`
- `instruction-v2/indexes/04_bag_map.json`
- `instruction-v2/indexes/05_step_map.json`
- `instruction-v2/indexes/06_callout_crop_box_map.json`
- `instruction-v2/indexes/07_qty_ocr_map.json`
- `instruction-v2/indexes/08_part_segmentation_map.json`

## Checks

| Check | Result | Evidence |
|---|---|---|
| 1. Bag 2 start page matches bag_truth.db | PASS | bag_truth_start=22; v2_start=22; v2_source=v1_bag_truth_db |
| 2. Bag 2 page range exists | PASS | v2_range=22-38; truth_expected_end=38; missing_pages=[] |
| 3. Every V1 crop-cache crop id exists in V2 | PASS | v1_crop_cache_count=15; missing=[] |
| 4. Every reviewed training_labels crop resolves to a V2 crop | PASS | actual_training_label_crops=13; resolved=13; unresolved=[] |
| 5. No unexpected step-number explosions | PASS | v2_steps=[26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]; expected_range=[26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]; unexpected=[] |
| 6. Missing callout list | PASS | missing_callouts=[] |
| 7. Missing qty list | PASS | missing_qty=[]; qty_mismatches=[] |
| 8. False-positive step list | PASS | false_positive_steps=[] |
| 9. Crop count comparison | PASS | v1_crop_cache=15; training_labels_actual_crops=13; v2_callouts=15 |
| 10. PASS / FAIL summary | PASS | Summary generated below |

## Exact Reasons

- All sign-off checks passed.

## Crop Count Comparison

```text
V1 crop cache: 15
training_labels actual crops: 13
V2 callouts: 15
```

## Missing Callouts

```json
[]
```

## Missing Qty

```json
[]
```

## Qty Mismatches

```json
[]
```

## False-Positive Steps

```json
[]
```

## Non-Crop Records

```text
(none)
```
