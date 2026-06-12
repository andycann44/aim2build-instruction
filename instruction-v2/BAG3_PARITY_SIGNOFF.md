# Bag 3 Parity Signoff

STATUS = BAG3_PARITY_APPROVED

Page 57 is parity-approved and closed.

The page 57 investigation proved:

- printed step number is `78`
- old OCR read was `18`
- V1 training-label truth maps the crop to `p57_s78_c1`
- V2 now maps that crop to `p57_s78_c1`
- qty is `8x`
- no further page 57 work is required

This audit now tracks only:

- missing callouts
- extra callouts
- qty mismatches
- page mismatches

## Bag 3 Parity Rule: Sequence-Gap Full-Page Audit

If V2 detects a step-number jump, the audit/fallback must inspect the full page, not only the area near detected anchors.

Example:

```text
V2 detects: 59, 60
Expected next sequence: 61, 62
Observed issue: 61/62 missing
```

Required audit behaviour:

- check both left and right columns
- check top and bottom callout zones
- compare against V1 truth boxes
- restore exact V1 crops only when visually/evidentially confirmed
- preserve V1 `crop_id`, `page`, `step`, `crop_box`, and `qty`

Reason:

Bag 3 has proven that V2 often detects left-column steps but misses right-column/opposite-side steps.

Confirmed examples:

- page 49 detected 55/56 but missed right-column 57/58
- page 50 detected 59/60 but missed right-column 61/62
- page 40 missed right-side 47

This is a parity-audit rule only. It is not a detection redesign.

## Sources

- Bag truth: `debug/bag_truth.db`
- V1 crop truth: `debug/training_labels/70618_bag3.json`
- V2 step map: `instruction-v2/indexes/05_step_map.json`
- V2 callouts: `instruction-v2/indexes/06_callout_crop_box_map.json`
- V2 qty OCR: `instruction-v2/indexes/07_qty_ocr_map.json`

## Bag 3 Range

| Source | Start | End |
|---|---:|---:|
| `bag_truth.db` | 39 | 57 |

Bag 4 starts at page 58, so Bag 3 parity scope is pages 39-57.

## Summary

| Check | Result | Evidence |
|---|---|---|
| Page 57 approved | PASS | `p57_s78_c1` present; page 57 step is `78`; qty is `[8]`; old OCR `18` is only a corrected raw read |
| V1 crop ids resolved in V2 | PASS | V1 Bag 3 crop ids: 30; resolved V2 crop ids: 30; missing: 0 |
| Missing semantic page/step callouts | PASS | 0 page/step callouts missing by semantic count |
| Extra semantic callouts | PASS | No extra semantic callouts remain |
| Qty mismatches on resolved crop ids | PASS | No mismatches among currently resolved V1 crop ids |
| Page/step mismatches on resolved crop ids | PASS | No page/step mismatches among currently resolved V1 crop ids |

## Approved / Closed

| Crop id | Page | Step | Qty | Status |
|---|---:|---:|---|---|
| `p39_s41_c3` | 39 | 41 | `[1, 2]` | resolved |
| `p39_s42_c1` | 39 | 42 | `[1]` | resolved |
| `p39_s43_c2` | 39 | 43 | `[3, 1, 1]` | resolved |
| `p40_s44_c1` | 40 | 44 | `[1, 1, 1]` | resolved |
| `p40_s45_c3` | 40 | 45 | `[1]` | resolved; remapped from duplicate V2 step 44 |
| `p40_s47_c2` | 40 | 47 | `[1, 1]` | resolved; restored from V1 truth box |
| `p49_s55_c2` | 49 | 55 | `[1, 1]` | resolved |
| `p49_s56_c4` | 49 | 56 | `[1, 1]` | resolved |
| `p49_s57_c1` | 49 | 57 | `[2]` | resolved; restored from V1 truth box |
| `p49_s58_c3` | 49 | 58 | `[1, 1]` | resolved; restored from V1 truth box |
| `p50_s59_c2` | 50 | 59 | `[1, 1]` | resolved |
| `p50_s60_c4` | 50 | 60 | `[2]` | resolved |
| `p50_s61_c1` | 50 | 61 | `[2]` | resolved; restored from V1 truth box |
| `p50_s62_c3` | 50 | 62 | `[2]` | resolved; restored from V1 truth box |
| `p52_s64_c2` | 52 | 64 | `[1, 1]` | resolved |
| `p52_s65_c4` | 52 | 65 | `[1, 1]` | resolved |
| `p52_s66_c1` | 52 | 66 | `[1, 1]` | resolved; restored from V1 truth box |
| `p52_s67_c3` | 52 | 67 | `[1, 2]` | resolved; restored from V1 truth box |
| `p57_s78_c1` | 57 | 78 | `[8]` | parity-approved; page 57 closed |

## Page 40 Result

Page 40 is now parity-aligned at the callout/step level.

| Source | Semantic callouts |
|---|---|
| V1 truth | `p40_s44_c1` -> step 44; `p40_s45_c3` -> step 45; `p40_s47_c2` -> step 47 |
| V2 current | `p40_s44_c1` -> step 44; `p40_s45_c3` -> step 45; `p40_s47_c2` -> step 47 |

The previous lower-left duplicate V2 step `44` was remapped to step `45` / `p40_s45_c3`.
The missing right-side callout was restored from V1 truth as step `47` / `p40_s47_c2`.

## Page 49 Result

Page 49 is now parity-aligned at the callout/step level.

| Source | Semantic callouts |
|---|---|
| V1 truth | `p49_s55_c2` -> step 55; `p49_s56_c4` -> step 56; `p49_s57_c1` -> step 57; `p49_s58_c3` -> step 58 |
| V2 current | `p49_s55_c2` -> step 55; `p49_s56_c4` -> step 56; `p49_s57_c1` -> step 57; `p49_s58_c3` -> step 58 |

The existing left-column V2 crops were mapped to V1 crop ids `p49_s55_c2` and `p49_s56_c4`.
The missing right-column callouts were restored from V1 truth as `p49_s57_c1` and `p49_s58_c3`.

## Page 50 Result

Page 50 is now parity-aligned at the callout/step level.

| Source | Semantic callouts |
|---|---|
| V1 truth | `p50_s59_c2` -> step 59; `p50_s60_c4` -> step 60; `p50_s61_c1` -> step 61; `p50_s62_c3` -> step 62 |
| V2 current | `p50_s59_c2` -> step 59; `p50_s60_c4` -> step 60; `p50_s61_c1` -> step 61; `p50_s62_c3` -> step 62 |

The existing left-column V2 step 59 crop was mapped to V1 crop id `p50_s59_c2`.
The existing step 60 correction remains mapped to `p50_s60_c4`.
The missing right-column callouts were restored from V1 truth as `p50_s61_c1` and `p50_s62_c3`.

## Page 52 Result

Page 52 is now parity-aligned at the callout/step level.

| Source | Semantic callouts |
|---|---|
| V1 truth | `p52_s64_c2` -> step 64; `p52_s65_c4` -> step 65; `p52_s66_c1` -> step 66; `p52_s67_c3` -> step 67 |
| V2 current | `p52_s64_c2` -> step 64; `p52_s65_c4` -> step 65; `p52_s66_c1` -> step 66; `p52_s67_c3` -> step 67 |

The existing left-column V2 crops were mapped to V1 crop ids `p52_s64_c2` and `p52_s65_c4`.
The missing right-column callouts were restored from V1 truth as `p52_s66_c1` and `p52_s67_c3`.

## Missing V1 Crop-Id Resolution

No missing V1 crop IDs remain.

```text
V1 crop ids: 30
V2 resolved crop ids: 30
missing: []
```

## Missing Semantic Callouts

No missing semantic callouts remain.

## Extra Semantic Callouts

No extra semantic callouts remain.

## Qty Mismatches

No qty mismatches were found among resolved crop ids:

```text
p39_s41_c3
p39_s42_c1
p39_s43_c2
p40_s44_c1
p40_s45_c3
p40_s47_c2
p49_s55_c2
p49_s56_c4
p49_s57_c1
p49_s58_c3
p50_s59_c2
p50_s60_c4
p50_s61_c1
p50_s62_c3
p52_s64_c2
p52_s65_c4
p52_s66_c1
p52_s67_c3
p57_s78_c1
```

All V1 crop ids resolve; no qty mismatches remain.

## Page Mismatches

No page/step mismatches were found among resolved crop ids.

All V1 crop ids resolve; no page/step mismatches remain.

## Approval Evidence

Bag 3 parity check output:

```text
V1 IDs 30
Resolved IDs 30
Missing IDs []
Extra IDs []
Missing semantic []
Extra semantic []
Qty mismatches []
Page mismatches []
```

Bag 3 is approved for the current parity scope: callout/step mapping, crop IDs, page/step, and qty.
