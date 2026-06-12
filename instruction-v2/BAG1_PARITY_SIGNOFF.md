# Bag 1 Parity Signoff

STATUS = BAG1_PARITY_APPROVED

## Sources Read

- `debug/bag_truth.db`
- `debug/crop_cache/70618_bag1.json`
- `debug/training_labels/70618_bag1.json`
- `instruction-v2/indexes/04_bag_map.json`
- `instruction-v2/indexes/05_step_map.json`
- `instruction-v2/indexes/06_callout_crop_box_map.json`
- `instruction-v2/indexes/07_qty_ocr_map.json`
- `instruction-v2/indexes/02_page_index.json`

## Checks

| Check | Result | Evidence |
|---|---|---|
| 1. Bag start page matches V1 truth | PASS | V1 bag_truth.db start_page=6; V2 start_page=6; source=v1_bag_truth_db |
| 2. All Bag 1 pages exist | PASS | Bag 1 page range 6-21; missing_pages=[] |
| 3. All V1 crop ids exist in V2 | PASS | V1 crop ids=22; missing=[] |
| 4. No extra step numbers exist | PASS | V2 Bag 1 step_numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]; expected=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]; extra=[]; missing_expected=[] |
| 5. Step 65 absent | PASS | step65_absent=True |
| 6. Step 8 present | PASS | Step 8 callout entries=1 |
| 7. Step 8 qty = 4 | PASS | Step 8 qty entries=[{'qty_text': ['4x'], 'qty_numbers': [4], 'crop_id': 'bag_01_page_010_step_8_015'}] |
| 8. p7_s1_c1 present | PASS | crop_path=debug/callout_crop_boxes/v1_p7_s1_c1_crop.png |
| 9. p7_s2_c2 present | PASS | crop_path=debug/callout_crop_boxes/v1_p7_s2_c2_crop.png |
| 10. All known reviewed Bag 1 crops resolve | PASS | actual_training_label_crop_records=22; resolved=22; unresolved=[]; non_crop_element_records_excluded=19 |

## Bag 1 Truth

```text
V1 bag_truth.db: bag=1 start_page=6 source=detector confidence=0.65
V2 bag_map: bag=1 start_page=6 end_page=21 source=v1_bag_truth_db confidence=0.65
```

## V1 Crop Cache Coverage

```text
V1 crop_cache crop ids: 22
Missing in V2: []
```

## Step Number Coverage

```text
V2 Bag 1 step numbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
Expected Bag 1 printed step sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
Extra: []
Missing expected: []
```

## Step 8 Evidence

```text
callout_crop_box={'x': 10, 'y': 610, 'w': 265, 'h': 185} crop_image_path=debug/callout_crop_boxes/bag_01_page_010_step_8_015_crop.png geometry_rule=clipped_left_callout_panel_slack
qty_text=['4x'] qty_numbers=[4] confidence=0.9657
```

## Reviewed Crop Resolution

```text
training label actual crop records: 22
resolved actual crop records: 22
unresolved actual crop records: 0
non-crop element records excluded: 19
```

## Excluded Non-Crop Label Records

These `debug/training_labels` records have `page=0`, `step=0`, and no crop image path, so they are catalog/element label records rather than Bag 1 instruction crop records. They were not counted in check 10.

```text
elem_4530589_p310, elem_4529247_p310, elem_4529242_p311, elem_4521914_p310, elem_4521886_p311, elem_4520638_p311, elem_4519742_p310, elem_4517986_p310, elem_4516546_p311, elem_4515373_p309, elem_4515365_p309, elem_4514553_p309, elem_4507045_p310, elem_4495724_p311, elem_4495704_p311, elem_4289538_p309, elem_4288212_p309, elem_6117400_p310, elem_6126075_p310
```
