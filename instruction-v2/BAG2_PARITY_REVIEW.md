# Bag 2 Parity Review

STATUS = BAG2_PARITY_PASS

## Sources Read

- `debug/bag_truth.db`
- `debug/crop_cache/70618_bag2.json`
- `debug/training_labels/70618_bag2.json`
- `instruction-v2/indexes/04_bag_map.json`
- `instruction-v2/indexes/05_step_map.json`
- `instruction-v2/indexes/06_callout_crop_box_map.json`
- `instruction-v2/indexes/07_qty_ocr_map.json`
- `instruction-v2/indexes/08_part_segmentation_map.json`
- `instruction-v2/indexes/02_page_index.json`

## Checks

| Check | Result | Evidence |
|---|---|---|
| 1. Bag 2 start/end pages from bag truth | PASS | V1 truth start=22; next bag start=39; expected_end=38; V2 start=22; V2 end=38; source=v1_bag_truth_db |
| 2. All Bag 2 pages exist | PASS | page_range=22-38; missing_pages=[] |
| 3. All V1 crop-cache crop ids exist in V2 | PASS | v1_crop_cache_count=15; missing=[] |
| 4. Step numbers are sensible | PASS | V2 step_numbers=[26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]; expected=[26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]; extra=[]; missing=[] |
| 5. False-positive steps | PASS | false_positive_count=0; entries=[] |
| 6. Missing callout crops | PASS | missing_callouts=[] |
| 7. Qty mismatches | PASS | qty_mismatches=[] |
| 8. Segmentation missing for known crops | PASS | segmentation_missing=[] |
| 9. Reviewed labels that resolve | PASS | actual_label_crop_records=13; resolved=13; unresolved=[] |
| 10. Non-crop elem_* records separated | PASS | non_crop_records=0; ids=[] |

## Bag 2 Truth

```text
bag_truth.db: bag=2 start_page=22 source=detector confidence=0.65
next bag truth: bag=3 start_page=39
V2 bag_map: start_page=22 end_page=38 source=v1_bag_truth_db confidence=0.65
```

## V1 Crop Cache Coverage

```text
V1 crop-cache crops: 15
V2 Bag 2 callout crops: 15
Missing V1 ids in V2: []
```

## Step Coverage

```text
V2 Bag 2 step numbers: [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
Expected from V1 crop cache range: [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
Extra: []
Missing: []
```

## Missing Callouts

```text
[]
```

## Qty Mismatches

```text
[]
```

## Segmentation Missing

```text
[]
```

## Reviewed Label Resolution

```text
actual crop label records: 13
resolved: 13
unresolved: 0
non-crop records separated: 0
```

## Non-Crop Label Records

```text
(none)
```
