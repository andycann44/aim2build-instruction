# Step 8 Missing Callout Audit

Scope: audit only. No manifests were regenerated and no code was edited.

## Summary

Bag 1 Step 8 disappeared between Stage 5 and Stage 6.

Step 8 is present in `05_step_map.json` on page 10, and it was not rejected by the step-number guardrail. It is not present in the imported V1 crop cache, and it is not present in `06_callout_crop_box_map.json` or `07_qty_ocr_map.json`.

This is a missed callout crop, not a true no-callout step.

## 1. Is Step 8 present in `05_step_map.json`?

Yes.

Relevant Bag 1 entry:

```json
{
  "bag": 1,
  "page": 10,
  "step_index": 2,
  "step_number": 8,
  "crop_id": null,
  "source": null,
  "rejection_reason": null
}
```

Nearby page 10 Stage 5 entries:

```text
page 10 step_index 1 step_number 7 rejection_reason null
page 10 step_index 2 step_number 8 rejection_reason null
```

So Step 8 exists at Stage 5.

## 2. Is Step 8 present in V1 crop cache?

No.

Checked:

- `debug/crop_cache/70618_bag1.json`
- `instruction-v2/indexes/05c_v1_crop_cache_import.json`

Findings:

- `debug/crop_cache/70618_bag1.json` is a list with 22 entries.
- No entry has `step: 8`.
- No entry has crop id like `p10_s8_*`.
- There are no page 10 entries in `debug/crop_cache/70618_bag1.json`.
- `05c_v1_crop_cache_import.json` also has no Step 8/page 10 entry.

Therefore V2 had no persisted V1 crop truth for Step 8.

## 3. Is Step 8 present in `06_callout_crop_box_map.json`?

No, not for Bag 1 page 10.

The only page 10 Bag 1 Stage 6 entry is Step 7:

```json
{
  "bag": 1,
  "page": 10,
  "step": 7,
  "callout_crop_box": {
    "x": 28,
    "y": 28,
    "w": 246,
    "h": 145
  },
  "crop_image_path": "debug/callout_crop_boxes/bag_01_page_010_step_7_014_crop.png"
}
```

There is no Bag 1 page 10 Step 8 entry in Stage 6.

## 4. If missing from Stage 6, why?

Because Stage 6 did not have V1 crop-cache truth for Step 8 and the fresh V2 callout detector failed to emit a crop for the raw Step 8 anchor.

Current Stage 6 flow:

1. Import V1 crop-cache entries first.
2. Skip raw step anchors when a V1 page/step crop already exists.
3. For remaining raw steps, run fresh callout detection.
4. Only append an entry when `analyzeStep()` returns a callout box.

For Step 8:

- It is a raw Stage 5 step.
- It has no V1 cache entry.
- It is not rejected.
- It enters fresh Stage 6 detection.
- No Stage 6 entry is emitted.

Likely detector failure:

- Page 10 Step 8 step anchor is near the left side below the lower callout.
- The visible Step 8 callout panel extends down around/under the printed step number area.
- `callout_crop_box_scan.mjs::detectCalloutRectByEdges()` rejects components whose `pageBottom > stepY - 5`.
- For this page layout, that rule can reject the valid Step 8 panel because the panel extends below the step anchor's y threshold.

So the crop is lost inside fresh callout detection, not during review rendering.

## 5. Did the guardrail accidentally reject its crop?

No.

The Stage 5 Step 8 entry has:

```text
step_number: 8
rejection_reason: null
```

The guardrail did not reject Step 8. It only rejected page 9 raw visual anchors after trusted V1 crop-cache steps owned that page.

## 6. Is there a crop id for it in V1?

No.

Expected V1-style id would likely be:

```text
p10_s8_c?
```

No such id exists in:

- `debug/crop_cache/70618_bag1.json`
- `instruction-v2/indexes/05c_v1_crop_cache_import.json`

Stage 6 also does not create a V2 id for it because no crop is emitted.

## 7. Is this a true no-callout step or a missed crop?

Missed crop.

Visual inspection of:

```text
instruction-v2/pages/70618_01/page_010.png
```

shows:

- Step 7 has a visible callout panel at the top-left with `1x`.
- Step 8 has a visible callout panel on the left with `4x`.

Therefore Bag 1 Step 8 should have a callout crop.

## Stage Where Step 8 Disappeared

```text
05_step_map.json
  page 10 step 8 present, not rejected

05c_v1_crop_cache_import.json
  page 10 step 8 absent

06_callout_crop_box_map.json
  page 10 step 8 absent

07_qty_ocr_map.json
  page 10 step 8 absent

Bag 1 review renderer
  correctly reports "No callout crop for this step"
```

Conclusion: Step 8 disappeared at Stage 6 fresh callout crop detection.

## Recommended Next Fix

Do not change Step 5.

The next fix should be a Stage 6 callout-crop parity fix for this page layout, using V1 callout repair/fallback behaviour as reference. In particular, inspect/port the V1 flow around:

- `_detect_callout_rect_by_edges`
- `_repair_callout_box_candidate_crop`
- `_detect_page_level_callout_panels`
- `_page_level_callout_candidates_for_fallback`
- `_refine_page_level_panel_with_step_geometry`

The immediate proof target should be:

```text
Bag 1 page 10 step 8 -> emits a callout crop with qty 4x
```

