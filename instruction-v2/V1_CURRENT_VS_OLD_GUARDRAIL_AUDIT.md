# V1 Current vs Old Guardrail Audit

Date: 2026-06-03

Goal: compare the uploaded/pasted old V1 reference against the current VS Code repo V1 file at `clean/routers/instruction_debug.py`, and identify whether the old bag/step/callout/qty guardrails still exist.

Documentation only. No code or manifests were changed.

## Source Files

- Old V1 reference: `/Users/andy/.codex/attachments/955d1fe7-1c71-4c1d-8dc6-215d350e28ea/pasted-text.txt`
- Current V1 file: `clean/routers/instruction_debug.py`

## Diff Summary

The audited old V1 reference and current repo V1 file are materially identical for this audit.

`diff -u` shows only an end-of-file newline difference:

```text
-        return blank, "failed", info
\ No newline at end of file
+        return blank, "failed", info
```

All requested guardrails/fallbacks exist in both old V1 and current VS Code V1.

## Guardrail Matrix

| Item | Exists in old V1? | Exists in current VS Code V1? | Changed materially? | Current line | Recommendation for V2 |
|---|---:|---:|---:|---:|---|
| `_filter_invalid_step_anchor_boxes` | yes | yes | no | 1 | port now |
| `_PAGE_CALLOUT_DETECTION_CACHE` | yes | yes | no | 97 | port later |
| `_page_callout_cache_entry` | yes | yes | no | 163 | port later |
| `_crop_cache_path` | yes | yes | no | 214 | port later |
| `_load_crop_detection_cache` | yes | yes | no | 220 | port later |
| `_write_crop_detection_cache` | yes | yes | no | 237 | port later |
| `_detect_callout_rect_by_edges` | yes | yes | no | 442 | port now |
| `_repair_callout_box_candidate_crop` | yes | yes | no | 815 | port now |
| `_detect_page_level_callout_panels` | yes | yes | no | 1351 | port now |
| `_page_level_callout_candidates_for_fallback` | yes | yes | no | 1435 | port now |
| `_refine_page_level_panel_with_step_geometry` | yes | yes | no | 1250 | port now |
| `_detect_page_step_number_boxes` | yes | yes | no | 953 | port now |
| `_detect_step_number_below_panel` | yes | yes | no | 1043 | port now |
| `_final_crop_qty_token_is_valid` | yes | yes | no | 1735 | port now |
| `_qty_payload_for_page_level_callout_crop` | yes | yes | no | 1776 | port now |
| `_extract_detected_qty_details_from_crop` | yes | yes | no | 1917 | port now |
| `_auto_qty_payload_for_crop` | yes | yes | no | 2005 | port now |

## Notes By Area

### Bag / Page State

The current V1 still has both in-memory page callout cache and persistent crop-detection disk cache:

- `_PAGE_CALLOUT_DETECTION_CACHE`
- `_page_callout_cache_entry`
- `_crop_cache_path`
- `_load_crop_detection_cache`
- `_write_crop_detection_cache`

These are useful, but V2 is manifest-first. Porting the cache model directly is lower priority than porting the detection behaviour. Recommendation: port later, likely as manifest metadata or explicit reuse policy rather than hidden state.

### Step Guardrails

The current V1 still has:

- `_filter_invalid_step_anchor_boxes`
- `_detect_page_step_number_boxes`
- `_detect_step_number_below_panel`

These are directly relevant to V2 Stage 4 because V2 needs printed step numbers, not synthetic step indexes. Recommendation: port now.

### Callout Guardrails

The current V1 still has:

- `_detect_callout_rect_by_edges`
- `_repair_callout_box_candidate_crop`
- `_detect_page_level_callout_panels`
- `_page_level_callout_candidates_for_fallback`
- `_refine_page_level_panel_with_step_geometry`

These are the highest-value callout additions for V2 Stage 6. V2 already has a simplified `_detect_callout_rect_by_edges` adaptation, but it does not yet carry the full repair and page-level fallback behaviour. Recommendation: port now.

### Qty OCR Guardrails

The current V1 still has:

- `_final_crop_qty_token_is_valid`
- `_qty_payload_for_page_level_callout_crop`
- `_extract_detected_qty_details_from_crop`
- `_auto_qty_payload_for_crop`

These directly address known V2 problems such as missing `6x`, duplicate `1x` tokens, and step-number contamination. Recommendation: port now.

## Recommendation

Use the current repo V1 file as the reference. The uploaded old V1 does not contain missing guardrails that were removed from the current repo; they are still present.

For V2, the next parity work should port behaviour in this order:

1. Step anchor sanity and printed-number recovery.
2. Callout repair and page-level fallback.
3. Qty token validity, row grouping, duplicate handling, and page-level qty payload logic.
4. Cache/reuse policy later, expressed as V2 manifest metadata rather than hidden mutable cache.
