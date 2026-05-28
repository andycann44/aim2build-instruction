# Obsolete / Dead-Code Inventory

Tracks items that are no longer actively used in the training/debug pipeline.
**Nothing is deleted automatically.** This file drives manual cleanup decisions.

Updated: 2026-05-28

---

## Obsolete Endpoints

| endpoint | file | replacement | safe_to_delete | reason |
|---|---|---|---|---|
| `POST /debug/training-store/analyze-bundle` | `clean/routers/instruction_debug.py` | `analyse-bundle` (same file) | yes | US-spelling alias that delegates 1:1. Only kept to avoid breaking old bookmarks. |
| `POST /debug/training-store/approve-bundle` | `clean/routers/instruction_debug.py` | `accept-split-candidate` | no | Operates on the bundle-level `review_status` field (approve/reject). Still used by quick-review flow. Do not delete until quick-review is fully migrated to per-candidate status. |
| `POST /debug/training-store/reject-bundle` | `clean/routers/instruction_debug.py` | `reject-split-candidate` | no | Same as above — bundle-level rejection is still used by quick-review. |
| `POST /debug/training-store/upload-r2` | `clean/routers/instruction_debug.py` | `confirm-candidate-part` (triggers upload internally) | no | Still callable standalone for manual re-uploads; not yet confirmed dead. |

---

## Deprecated Candidate Fields

Fields present in stored JSON (`split_candidate_paths.candidates[]`) that are no longer read by display or business logic. Safe to stop writing; existing values can be ignored.

| field | file | replacement | safe_to_delete | reason |
|---|---|---|---|---|
| `original_candidate_path` | `_write_manual_refinement_outputs` in `instruction_debug.py` | `base_candidate_path` | yes | Redundant `setdefault`; introduced before `base_candidate_path` was named. Never read back in UI or logic. |
| `original_mask_path` | `_write_manual_refinement_outputs` | `mask_path` (unchanged field) | yes | Written as a `setdefault` snapshot but never consulted. |
| `original_thumbnail_path` | `_write_manual_refinement_outputs` | `thumbnail_path` (unchanged chain) | yes | Same — written but never read. |
| `raw_mask_path` | `generate_split_candidates` in `training_ai_review_service.py` | — | no | Points to the raw tight mask before alpha expansion. Only value is offline debug; not consumed by any live display code. Keep writing for now, do not read. |
| `alpha_expansion` | `generate_split_candidates` | — | yes | Dict of `{mode, dilate_px, close_kernel, raw_alpha_area, expanded_alpha_area}`. Written at generation time, never consumed by any downstream logic or UI. |
| `v2_mask_path` | `update_split_candidate_status` in `training_bundle_index_service.py` | `mask_path` | no | Copy of `mask_path` taken at accept time; created but never displayed or used by upload/confirm flow. Safe to stop copying; keep existing values. |
| `source_box` | `generate_split_candidates` | `box` | no | Original source box before tight cropping. Used only in `_box_iou` dedup during generation; not needed after. Could be dropped from stored dict once generation is stable. |
| `component_label` | `generate_split_candidates` | — | no | Label from connected-components pass. Used transiently during generation, stored for debug. Not read by any UI display. |
| `component_area` | `generate_split_candidates` | — | no | Pixel area of the mask component. Displayed in `bundle-debug` response only. Not part of normal review flow. |
| `label` | `generate_split_candidates` | — | no | AI-returned label string (e.g. "Mask Component Candidate 1"). Stored but not rendered in main review UI. |
| `reason` | `generate_split_candidates` | — | no | AI-returned reason string. Same — stored, not rendered. |
| `confidence` | `generate_split_candidates` | — | no | AI-suggested region confidence. Stored, not shown in main UI. |
| `display_index` | `generate_split_candidates` | computed by `_candidate_display_index()` | yes | `_candidate_display_index` always falls back to `index+1` anyway; storing it separately adds no value. |
| `qty_scrub_status` | everywhere | `qty_text_state` | no | Old string field for scrub state. `qty_text_state` is the authoritative new field. `qty_scrub_status` is still written for visual display in candidate meta. Keep until UI is updated to read `qty_text_state` instead. |

---

## Dead Helper Functions / Constants

| symbol | file | replacement | safe_to_delete | reason |
|---|---|---|---|---|
| `_TRAINING_REVIEW_COMPLETE_STATES` | `instruction_debug.py` line ~5079 | inline check in `_next_pending_bundle_payload` | no | Used in one place (`_next_pending_bundle_payload`). Consider inlining or renaming to `_REVIEW_BLOCKING_STATES` since the name is misleading — these states mean the bundle is *blocked*, not complete. |
| `_blue_callout_background_mask` | `training_ai_review_service.py` | `_remove_background_fringe_from_manual_mask` (LAB-based) | no | Returns a hard-coded HSV blue mask. Still referenced in `generate_split_candidates` alpha expansion. Cannot delete until that call site is updated to the generic LAB fringe removal. |

---

## Unused Debug Overlay Files (on disk)

These files are written to the analysis bundle directory but are not displayed anywhere in the current review UI. They are safe to ignore (not delete yet — may be useful for offline analysis).

| file pattern | written by | displayed_where | safe_to_delete | reason |
|---|---|---|---|---|
| `candidate_N_background_likeness_mask.png` | `_remove_background_fringe_from_manual_mask` | `candidate-alpha-debug` JSON only | no | Not rendered in any HTML UI. Path stored in `manual_bg_likeness_mask_path`. |
| `candidate_N_refined_minus_background_mask.png` | `_remove_background_fringe_from_manual_mask` | `candidate-alpha-debug` JSON only | no | Debug overlay showing pass1 (red) and pass2 (magenta) removed pixels. Path stored in `manual_bg_minus_mask_path`. |
| `candidate_N_manual_trimap.png` | `_write_manual_refinement_outputs` | `candidate-alpha-debug` JSON only | no | Matting trimap; kept for offline matting analysis. |
| `candidate_N_manual_refined_overlay.png` | `_write_manual_refinement_outputs` | nowhere | yes | Green overlay of refined mask on original. Not linked from any UI. |
| `candidate_N_colour_cleanup_overlay.png` | `save-picked-colour-cleanup` | nowhere | yes | Magenta overlay of removed pixels. Not linked from any UI. |
| `candidate_N_colour_cleanup_mask.png` | `save-picked-colour-cleanup` | nowhere | no | Binary mask of *removed* pixels (not remaining alpha). Stored as `colour_cleanup_mask_path`. Not displayed but may be useful for undo. Do not confuse with `colour_cleanup_alpha_path` (remaining alpha). |

---

## Replaced Mask Flows

| flow | file | replacement | safe_to_delete | reason |
|---|---|---|---|---|
| Blue-background alpha expansion (HSV hard-coded) | `_blue_callout_background_mask` in `training_ai_review_service.py` | `_remove_background_fringe_from_manual_mask` (generic LAB) | no | Still used in `generate_split_candidates` initial alpha expansion. Replace with generic flow and then delete. |
| `candidate_path` overwritten on manual amend | old `_write_manual_refinement_outputs` | `current_candidate_path` + `base_candidate_path` (refactored) | — | Was writing the amended path back to `candidate_path`. Now `candidate_path` = immutable base; `current_candidate_path` advances. Any bundle processed before 2026-05-28 may still have the old layout in stored JSON. |
| `qty_scrubbed_path` gate for Accept Clean | `mark_split_candidate` and confirm guard | `_candidate_qty_is_clean()` checking `qty_text_state` | — | Old gate blocked accept if `qty_scrubbed_path` was empty. New gate accepts `manual_mark_clean` and `not_needed` states too. Legacy `qty_scrubbed_path` still accepted as fallback. |

---

## Stale Cache Formats

| item | file | replacement | safe_to_delete | reason |
|---|---|---|---|---|
| Quick review cache JSON (`{set}_bag{n}_quick_review.json`) | `debug/reports/` | regenerated by `build-quick-review-cache` | no | Cache is invalidated via `stale=true` flag when any candidate mutates. Stale caches are harmless but take disk space. Old caches from before the `base_candidate_path` refactor may contain `candidate_path` pointing to amended images. Re-run `build-quick-review-cache` to refresh. |
| Bundle `split_candidate_paths` JSON with no `current_candidate_path` | PostgreSQL `training_bundle_index` | generated fresh by `generate-split-candidates` | no | Bundles created before 2026-05-28 lack `current_candidate_path`, `base_candidate_path`, `qty_text_state`, `cleanup_history`. `_candidate_display_path()` falls back gracefully. Re-running generation rebuilds these fields. |
