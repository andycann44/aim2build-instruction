# Live Pipeline Map — Training / Debug Tooling

Documents the **current active pipeline only**. See `obsolete_inventory.md` for deprecated flows.

Updated: 2026-05-28

---

## Overview

```
crop image
    │
    ▼
Bundle Registration ──────────────────────────────── postgres: training_bundle_index
    │
    ▼
OCR + Mask Analysis (generate-split-candidates)
    │  ┌─ baseline_slot candidates  (connected-components from master mask)
    │  └─ ai_suggested candidates   (OpenAI region suggestions)
    │
    ▼
Candidate State (per-candidate JSON in split_candidate_paths)
    │
    ├── Qty Scrub flow  ──────────────────────── scrub-candidate-qty | mark-qty-clean
    ├── Manual Amend flow  ───────────────────── save-manual-mask-amendment → rerun-manual-mask-refinement
    │       └── Background Fringe Removal  (automatic, on every refinement)
    ├── Colour Cleanup flow  ─────────────────── save-picked-colour-cleanup
    │
    ▼
Accept + Confirm  ────────────────────────────── accept-split-candidate → confirm-candidate-part
    │
    ▼
Bag Audit  ───────────────────────────────────── bag-completion-audit
    │
    ▼
Quick Review  ────────────────────────────────── build-quick-review-cache → quick-review
```

---

## Candidate State Fields

Every candidate is a JSON dict inside `split_candidate_paths.candidates[]` in `training_bundle_index`.

### Core identity

| field | set by | meaning |
|---|---|---|
| `index` | `generate_split_candidates` | Stable integer index within the bag's candidates. Used as the primary key in all mutations. |
| `display_index` | `generate_split_candidates` | 1-based display number. Prefer computing via `_candidate_display_index()`. |
| `group` | `generate_split_candidates` | `"baseline_slot"` or `"ai_suggested"`. |
| `status` | `mark_split_candidate` | `"pending"`, `"accepted"`, or `"rejected"`. |
| `review_state` | `set_split_candidate_review_state` / `update_split_candidate_status` | `""` (clean), `"mask_amended"`, `"needs_mask_expand"`, `"needs_ocr_review"`, `"needs_manual_crop"`. Cleared to `""` on accept. |
| `box` | `generate_split_candidates` / `_write_manual_refinement_outputs` | `[x, y, w, h]` in original image pixels. Updated when manual amend changes the tight bounding box. |

### Image path chain (priority order)

All display and tool input uses `_candidate_display_path()`, which resolves in this order:

```
current_candidate_path
  └─ colour_cleanup_path         (after save-picked-colour-cleanup)
  └─ manual_amended_candidate_path (after save-manual-mask-amendment + rerun)
  └─ qty_scrubbed_path           (after scrub-candidate-qty, if not subsequently amended)
  └─ candidate_path              (fallback: original crop, immutable)
```

| field | set by | notes |
|---|---|---|
| `base_candidate_path` | `generate_split_candidates` | Original crop from OCR mask. **Never overwritten.** |
| `candidate_path` | `generate_split_candidates` | Alias for `base_candidate_path` in new data. In pre-2026-05-28 data this may point to amended image (old code overwrote it). |
| `current_candidate_path` | every mutation tool | Always points to the latest output image. Updated by qty scrub, manual amend, colour cleanup. |
| `current_alpha_path` | every mutation tool | Always points to the latest remaining-alpha mask (grayscale, 0–255). NOT the removed-pixels mask. |
| `manual_amended_candidate_path` | `_write_manual_refinement_outputs` | BGRA PNG of the manually refined candidate. |
| `qty_scrubbed_path` | `scrub_candidate_qty` | BGRA PNG with qty text zeroed out. Cleared on manual amend. |
| `colour_cleanup_path` | `save-picked-colour-cleanup` | BGRA PNG after picked-colour removal. |
| `colour_cleanup_alpha_path` | `save-picked-colour-cleanup` | Remaining alpha mask after colour cleanup. Used by match scoring. |
| `thumbnail_path` | every mutation tool | Always matches `current_candidate_path`; kept for legacy display code. |

### Qty text state

| field | values | meaning |
|---|---|---|
| `qty_detected` | bool | True if OCR found qty tokens overlapping this candidate. |
| `qty_values` | `[int, ...]` | Detected qty values (e.g. `[2]`). |
| `qty_token_boxes` | `[{x,y,w,h,text}, ...]` | OCR box positions in original image coords. |
| `qty_text_state` | see below | Authoritative clean/dirty state for accept gate. |
| `qty_scrub_status` | string | Legacy display string. Prefer `qty_text_state`. |

`qty_text_state` values:

| value | meaning |
|---|---|
| `"not_needed"` | qty not detected; no scrub needed. |
| `"auto_scrubbed"` | qty tokens zeroed out automatically at generation time. |
| `"scrubbed"` | qty re-scrubbed via `scrub-candidate-qty` after mutation. |
| `"manual_mark_clean"` | user confirmed qty is visually absent; no image change needed. |
| `"manual_amended_needs_qty_scrub"` | manual amend cleared the scrub; re-scrub or mark clean required before accept. |
| `"not_detected"` | explicitly recorded as no qty. |

`_candidate_qty_is_clean()` returns `True` for: `not_needed`, `auto_scrubbed`, `scrubbed`, `manual_mark_clean`, `not_detected`, or legacy `qty_scrubbed_path` non-empty.

### Mask review state

| field | set by | meaning |
|---|---|---|
| `mask_review_state` | `_write_manual_refinement_outputs` | `""` or `"mask_amended"`. Informational; not used in accept gate. |
| `manual_mask_alpha_path` | `_write_manual_refinement_outputs` | Fine-grained alpha from matting step. |
| `manual_refined_mask_path` | `_write_manual_refinement_outputs` | Binary refined mask. |
| `manual_amended_mask_path` | `_write_manual_refinement_outputs` | Binary mask from amend step. |
| `manual_bg_removed_pixels` | `_write_manual_refinement_outputs` | Total pixels removed by fringe cleanup. |
| `manual_bg_pass1_removed` | `_write_manual_refinement_outputs` | Pixels removed by LAB colour pass. |
| `manual_bg_pass2_removed` | `_write_manual_refinement_outputs` | Pixels removed by connected-boundary pass. |

### Audit trail

| field | set by | meaning |
|---|---|---|
| `cleanup_history` | every mutation tool | Append-only list of `{op, path, at}` dicts. |

---

## Flows

### 1. Bundle Registration

```
POST /debug/training-store/register-bundle?bundle_id=…
POST /debug/training-store/export-batch          (batch registration)
```

Writes a row to `training_bundle_index`. Sets `review_status = "needs_review"`.

### 2. OCR + Mask Analysis (generate-split-candidates)

```
POST /debug/training-store/generate-split-candidates?bundle_id=…
```

Entry point: `generate_split_candidates()` in `training_ai_review_service.py`.

Actions:
1. Reads `original_crop` + `raw_mask` from bundle metadata
2. Runs connected-components on the mask → `baseline_slot` candidates
3. Calls OpenAI vision API for additional region suggestions → `ai_suggested` candidates
4. For each candidate: tight-crops the BGRA image, runs OCR qty detection, optionally auto-scrubs qty
5. Writes candidate PNGs to `debug/ai_training/analysis_bundles/{bundle_id}/`
6. Stores full candidate list as JSON in `split_candidate_paths` column

Output fields set: `base_candidate_path`, `current_candidate_path`, `current_alpha_path`, `qty_detected`, `qty_values`, `qty_text_state`, `cleanup_history = []`.

### 3. Qty Scrub Flow

Two options (both set `qty_text_state` to a clean state):

**Auto-scrub** (happens at generation time if qty detected):
- `current_candidate_path` = `qty_scrubbed_path` (zeroed-out BGRA)
- `qty_text_state = "auto_scrubbed"`

**Manual re-scrub** (after mask amend clears scrub):
```
POST /debug/training-store/scrub-candidate-qty?bundle_id=…&candidate_index=…
```
- Reads `current_candidate_path`, applies `_scrub_qty_boxes_from_bgra()`
- Writes new `_qty_scrubbed.png` alongside
- Sets `current_candidate_path`, `current_alpha_path`, `qty_text_state = "auto_scrubbed"`

**Mark Qty Clean** (when qty is visually absent after amend — no image change):
```
POST /debug/training-store/mark-qty-clean?bundle_id=…&candidate_index=…
```
- Metadata only; sets `qty_text_state = "manual_mark_clean"`

### 4. Manual Amend Flow

```
POST /debug/training-store/save-manual-mask-amendment?bundle_id=…&candidate_index=…
  body: { mask_png_base64 }

POST /debug/training-store/rerun-manual-mask-refinement?bundle_id=…&candidate_index=…
```

Actions (in `_write_manual_refinement_outputs`):
1. Applies foreground matting (`_manual_mask_foreground_refinement`)
2. Runs `_remove_background_fringe_from_manual_mask` (LAB-based, two-pass)
3. Tight-crops to non-zero region → updates `box`
4. Saves `_manual_amended.png` (BGRA), `_manual_amended_mask.png` (binary)
5. Sets `current_candidate_path = amended_candidate_path`
6. Clears `qty_scrubbed_path`, sets `qty_text_state = "manual_amended_needs_qty_scrub"` (if qty detected)
7. Appends `{op: "manual_amend"}` to `cleanup_history`

After amend: user must run Qty Scrub or Mark Qty Clean before accepting.

#### Background Fringe Removal (automatic on every refinement)

`_remove_background_fringe_from_manual_mask()` runs two passes:

- **Pass 1**: LAB colour distance from border-pixel background model. Removes pixels within `max(18.0, bg_spread × 1.5)` of background colour, protected by Sobel gradient > 25.0.
- **Pass 2**: Connected-components boundary fringe. Removes small components touching the alpha boundary that are within `max(28.0, bg_spread × 2.5)` of background colour.

### 5. Colour Cleanup Flow

```
POST /debug/training-store/save-picked-colour-cleanup
  body: { bundle_id, candidate_index, picked_rgb: [r,g,b], tolerance: float }
```

Actions:
1. Reads `current_candidate_path` via `_candidate_display_path()`
2. Computes per-pixel LAB distance from `picked_rgb`
3. Sobel edge protection (> 25.0)
4. Saves `_colour_cleanup.png` (cleaned BGRA), `_colour_cleanup_alpha.png` (remaining alpha), `_colour_cleanup_mask.png` (removed-pixels debug)
5. Sets `current_candidate_path = colour_cleanup_path`, `current_alpha_path = colour_cleanup_alpha_path`
6. Appends `{op: "colour_cleanup"}` to `cleanup_history`

### 6. Accept + Confirm Flow

**Accept:**
```
POST /debug/training-store/accept-split-candidate?bundle_id=…&candidate_index=…
```
Guard: `_candidate_qty_is_clean(candidate)` must return True.
Clears `review_state = ""`, sets `status = "accepted"`.

**Confirm part:**
```
POST /debug/training-store/confirm-candidate-part
  body: { bundle_id, candidate_index, part_num, color_id, element_id, qty }
```
Guard: `status == "accepted"` AND `_candidate_qty_is_clean(candidate)`.
Writes a row to `candidate_training_examples`. Triggers R2 upload.
Uses `_candidate_display_path()` as the thumbnail/training image path.

### 7. Bag Audit Flow

```
GET /debug/training-store/bag-completion-audit?set_num=…&bag_num=…
```

Aggregates across all bundles for the bag:
- OCR qty total vs confirmed qty total
- Unmapped accepted candidates
- Qty mismatches
- Per-part set_required vs confirmed_qty vs remaining_qty
- Review state counts

Returns standalone HTML with embedded JSON.

Surfaced from the Training Review UI bag-complete banner and `showBagComplete()` JS.

### 8. Quick Review Flow

```
POST /debug/training-store/build-quick-review-cache?set_num=…&bag_num=…
GET  /debug/training-store/quick-review?set_num=…&bag_num=…
```

Cache: `debug/reports/{set}_bag{n}_quick_review.json`

The cache is invalidated (marked `stale=true`) by `_mark_quick_review_cache_stale_for_bundle()` whenever any mutation writes to a candidate. Rebuilt on demand by `build-quick-review-cache`.

Quick review shows problem bundles (missing mask, OCR errors, review_state flags) and links to next bag.

---

## Key Helper Functions

| function | file | purpose |
|---|---|---|
| `_candidate_display_path(candidate)` | `instruction_debug.py` | Priority-ordered path resolution for display and tool input |
| `_candidate_qty_is_clean(candidate)` | `instruction_debug.py` | Accept gate check — returns True if qty is in a confirmed-clean state |
| `_training_review_candidate_part_matches(...)` | `instruction_debug.py` | Builds scored+ranked part suggestions for a candidate using CLIP-like colour/shape scoring |
| `_slot_mask_query_profile(thumbnail, mask)` | `instruction_debug.py` | Reads a candidate image + alpha mask → colour/shape profile for scoring |
| `_update_candidate_in_split_paths(paths, idx, updated)` | `instruction_debug.py` | Merges updated candidate into all three candidate lists (candidates, baseline_slot, ai_suggested) |
| `_mark_quick_review_cache_stale_for_bundle(bundle_id, reason)` | `instruction_debug.py` | Marks quick review cache stale after any mutation |
| `_remove_background_fringe_from_manual_mask(...)` | `instruction_debug.py` | Two-pass LAB fringe removal on manual amend output |
| `generate_split_candidates(bundle_id)` | `training_ai_review_service.py` | Full candidate generation: mask → crop → OCR → qty scrub |
| `scrub_candidate_qty(bundle_id, candidate_index)` | `training_ai_review_service.py` | Re-scrub qty from current image; reads `current_candidate_path` |
| `mark_split_candidate(bundle_id, candidate_index, status)` | `training_ai_review_service.py` | Accept/reject with qty-clean gate |

---

## Database Schema (relevant columns)

`training_bundle_index`:
- `bundle_id` — primary key (e.g. `70618_bag1_p7_s1_c1`)
- `set_num`, `bag_num`, `page_num`, `step_num`, `crop_num`
- `review_status` — bundle-level (`needs_review`, `approved`, `rejected`)
- `split_candidate_paths` — JSON blob containing all candidate dicts

`candidate_training_examples`:
- `bundle_id`, `candidate_index` — foreign key pair
- `part_num`, `color_id`, `element_id`, `qty`
- `thumbnail_path`, `r2_path`
- `confirmed_by`, `confirmed_at`
