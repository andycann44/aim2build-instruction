# Bag 4 Parity Signoff

STATUS = BAG4_PARITY_IN_PROGRESS

Pages 59/60, 61/62, 63, 64, 65, and 66 have been parity-fixed using exact V1 truth. Bag 4 is not approved yet.

Approved previous bags:

- Bag 1: `BAG1_PARITY_SIGNOFF.md`
- Bag 2: `BAG2_PARITY_SIGNOFF.md`
- Bag 3: `BAG3_PARITY_SIGNOFF.md`

## Bag 4 Range

| Source | Start | End |
|---|---:|---:|
| `debug/bag_truth.db` | 58 | 80 |

Bag 4 starts at page 58 and Bag 5 starts at page 81, so Bag 4 parity scope is pages 58-80.

## Sources

- V1 truth: `debug/training_labels/70618_bag4.json`
- V2 bag map: `instruction-v2/indexes/04_bag_map.json`
- V2 step map: `instruction-v2/indexes/05_step_map.json`
- V2 callouts: `instruction-v2/indexes/06_callout_crop_box_map.json`
- V2 qty OCR: `instruction-v2/indexes/07_qty_ocr_map.json`

## Applied Parity Fixes

Pages 59/60, 61/62, 63, and 64 were fixed by mapping existing physical V2 crops and restoring missed right-column crops with exact V1 truth.

Page 65 was fixed by mapping existing left-column physical V2 crops, correcting OCR step `6` to V1 step `96`, and restoring missing right-column V1 truth crops:

| Crop ID | Page | Step | V1 crop box | Qty text | Result |
|---|---:|---:|---|---|---|
| `p65_s95_c2` | 65 | 95 | `[114,143,313,107]` | `['1x']` | mapped existing physical V2 crop |
| `p65_s96_c3` | 65 | 96 | `[114,610,313,97]` | `['1x','2x']` | corrected OCR step `6` to `96` and mapped existing physical V2 crop |
| `p65_s97_c1` | 65 | 97 | `[854,29,312,97]` | `['1x']` | restored missing right-column crop |
| `p65_s98_c4` | 65 | 98 | `[854,610,312,110]` | `['1x','2x']` | restored missing right-column crop |

Page 66 was fixed by mapping existing left-column physical V2 crops and restoring the missing right-column V1 truth crop:

| Crop ID | Page | Step | V1 crop box | Qty text | Result |
|---|---:|---:|---|---|---|
| `p66_s99_c2` | 66 | 99 | `[29,29,313,97]` | `['4x']` | mapped existing physical V2 crop |
| `p66_s100_c3` | 66 | 100 | `[29,610,313,97]` | `['1x']` | mapped existing physical V2 crop |
| `p66_s101_c1` | 66 | 101 | `[769,29,312,97]` | `['1x','1x']` | restored missing right-column crop |

## Human Review Required Outside V1 Truth

Page 69 step 110 is visually present and appears to be a real boxed callout, but it is not present in `debug/training_labels/70618_bag4.json`.

Classification: `TRUE_CALLOUT_MISSING_FROM_V1_TRUTH`

Evidence:

| Page | Step | Candidate crop box | Visual qty text | V2 emitted physical crop? | Status |
|---:|---:|---|---|---|---|
| 69 | 110 | `[854,610,312,126]` | `['2x','1x']` | no | human review required; keep separate from parity pass/fail accounting unless a new reviewed crop ID is approved |

Evidence files:

- `instruction-v2/debug/bag4_step110_audit/page_069_step_110_candidate_overlay_full.png`
- `instruction-v2/debug/bag4_step110_audit/page_069_step_110_candidate_crop_full.png`

This should not be deleted as a false positive, and it should not be silently added to V1 parity as if V1 had reviewed it.

## Inventory Summary

| Check | Result | Evidence |
|---|---|---|
| V1 Bag 4 crop count | INFO | 29 V1 crop IDs |
| V2 resolved crop count | FAIL | 20 of 29 V1 crop IDs currently resolve in V2 |
| V2 callout count in Bag 4 range | INFO | 33 callout entries in pages 58-80 |
| Missing crop IDs | FAIL | 9 missing V1 crop IDs |
| Extra crop IDs | PASS | No extra V2 crop IDs in `06_callout_crop_box_map.json` for Bag 4 |
| Missing semantic callouts | FAIL | 5 missing page/step callouts |
| Extra semantic callouts | FAIL | 9 extra page/step callouts |
| Qty mismatches | PASS | None among resolved V1 crop IDs |
| Page mismatches | PASS | None among resolved V1 crop IDs |
| Step mismatches | PASS | None among resolved V1 crop IDs |

## Missing Crop IDs

```text
p68_s104_c2
p68_s105_c1
p68_s106_c3
p69_s107_c2
p69_s108_c4
p69_s109_c1
p78_s127_c3
p79_s128_c2
p79_s130_c1
```

## Extra Crop IDs

No extra V2 crop IDs were found in `06_callout_crop_box_map.json` for Bag 4.

## Missing Semantic Callouts

| Page | Step | V1 count | V2 count |
|---:|---:|---:|---:|
| 68 | 105 | 1 | 0 |
| 68 | 106 | 1 | 0 |
| 69 | 109 | 1 | 0 |
| 78 | 127 | 1 | 0 |
| 79 | 130 | 1 | 0 |

## Extra Semantic Callouts

| Page | Step | V1 count | V2 count | Notes |
|---:|---:|---:|---:|---|
| 70 | 11 | 0 | 1 | Out of Bag 4 sequence; likely OCR/false positive. |
| 71 | 114 | 0 | 1 | Out of expected sequence for current V1 Bag 4 crop truth. |
| 73 | 118 | 0 | 1 | Out of expected sequence for current V1 Bag 4 crop truth. |
| 74 | 120 | 0 | 1 | Out of expected sequence for current V1 Bag 4 crop truth. |
| 76 | 122 | 0 | 1 | Out of expected sequence for current V1 Bag 4 crop truth. |
| 77 | 173 | 0 | 1 | Likely OCR step-number error. |
| 78 | 124 | 0 | 1 | V1 truth expects page 78 step 127. |
| 78 | 125 | 0 | 1 | V1 truth expects page 78 step 127. |
| 79 | 129 | 0 | 1 | V1 truth expects page 79 steps 128 and 130. |

## Qty Mismatches

```text
qty_mismatches: []
```

## Page Mismatches

```text
page_mismatches: []
```

## Step Mismatches

```text
step_mismatches: []
```

## Next Work

Recommended next target:

1. Audit page 68 as the next sequence block.
2. Compare V1 truth crops `p68_s104_c2`, `p68_s105_c1`, and `p68_s106_c3` against V2 physical crops.
3. Apply the full-page sequence-gap audit rule as evidence only; restore or remap only exact V1 truth crops after visual/evidential confirmation.

## Bag 4 parity audit update - pages 68, 69, 78, 79

Reclassified as physical crop exists / mapping-only, not detector failure:
- p68_s104_c2
- p69_s107_c2
- p69_s108_c4
- p79_s128_c2

Confirmed V2 detector failures:
- p68_s105_c1
- p68_s106_c3
- p69_s109_c1
- p78_s127_c3
- p79_s130_c1

Diagnostic sequence completeness evidence:
- page 78 visible steps: 124,125,126,127
- page 78 emitted steps: 124,125
- page 78 missing diagnostic steps: 126,127
- page 79 visible steps: 128,129,130,131
- page 79 emitted steps: 128,129
- page 79 missing diagnostic steps: 130,131

Separate V1 truth exception:
- page 69 step 110 = V1_TRUTH_INCOMPLETE
- Do not count as V2_FAILURE
- Do not count as V2_FALSE_POSITIVE

Current genuine Bag 4 detector misses remaining:
5
