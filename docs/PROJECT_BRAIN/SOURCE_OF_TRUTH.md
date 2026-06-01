# Source of Truth

## Guiding Principle

Every concern in this project has exactly one authoritative state file. Derived
files, caches, and embeddings are downstream outputs. They must never be treated
as authoritative and must never be used to reconstruct state that belongs in the
authoritative file.

---

## Authoritative State Files

### Bag Review State

```
debug/training_labels/{set_num}_bag{bag}.json
```

**This is the one and only Bag Review source of truth.**

- Contains the per-slot labels (part_num, color_id, qty) for every crop in a bag
- Contains `unknown_slots` and `ignored_slots` per crop
- Contains `schema_version`, `set_num`, `bag`, `created_at`, `source`
- Written by `bag_review_service.save_review_state()` using `path.write_text()` — **not atomic**
- Legacy debug routes write via `_write_labels()` in `instruction_debug.py`, which **is atomic** (writes to a `.tmp` file then calls `temp_path.replace(path)`)
- Atomic write unification is a future cleanup task — see DECISIONS.md D14
- Verified SHA-256 for Bag 1: `fdf5ca0a5cc04787e7d5d7ef699ed3c928d4a4236521b44e80fc17c55879522b`

#### Why this file is the source of truth

1. It is the only file that `bag_review_service` writes label state to.
2. All review actions (`save-label`, `mark-slot-unknown`, `mark-slot-ignored`) converge here.
3. The review model is built by reading this file plus the crop cache — the crop cache is read-only input; the training labels file holds all human decisions.
4. Training exports, CLIP memory, and suggestion priors are all derived from it.

#### What may write to it

| Writer | Permitted | Notes |
|---|---|---|
| `bag_review_service.save_review_state()` | ✅ Yes | Primary writer for Bag Review slot operations; uses `path.write_text()` (not atomic) |
| `POST /debug/save-label` (slot path) | ✅ Yes | When `selected_slot_index >= 0`, delegates to `bag_review_service.save_slot_label()` |
| `POST /debug/save-label` (legacy path) | ⚠️ Exists | When no `selected_slot_index`, writes via `_write_labels()` directly in the router |
| `POST /debug/mark-slot-unknown` | ✅ Yes | Delegates to `bag_review_service.mark_slot_unknown()` |
| `POST /debug/mark-slot-ignored` | ✅ Yes | Delegates to `bag_review_service.mark_slot_ignored()` |
| `POST /debug/remove-label` | ⚠️ Exists | Legacy/debug route; writes via `_write_labels()` directly |
| `POST /debug/set-crop-status` | ⚠️ Exists | Legacy/debug route; writes via `_write_labels()` directly |
| `POST /debug/delete-crop` | ⚠️ Exists | Legacy/debug route; writes via `_write_labels()` directly |
| `POST /debug/save-manual-crop` | ⚠️ Exists | Legacy/debug route; writes via `_write_labels()` directly |
| `POST /debug/update-crop-qty` | ⚠️ Exists | Legacy/debug route; writes via `_write_labels()` directly |
| `POST /debug/buildability-clip-suggest` | ❌ No | Must not write review state |
| Any suggestion route | ❌ No | Suggestions are read-only to this file |
| Any training-store route | ❌ No | Training pipeline reads labels; does not write review state |
| New code not listed above | ❌ No | Do not add new write paths; consolidate existing ones instead |

#### What must NOT write to it

- CLIP suggestion routes
- Training bundle export routes
- Crop generation pipeline
- Catalog embedding generation
- Any new code not listed in the table above

> **Note on write path duplication:** Two write mechanisms currently coexist —
> `bag_review_service.save_review_state()` and `_write_labels()` in
> `instruction_debug.py`. Both write to the same training labels JSON. This is
> the current truth. Unifying all writes through a single path is a future
> cleanup task (see DECISIONS.md D14). Do not add a third mechanism.

---

### Other Authoritative State Files

| File | Concern | Writer |
|---|---|---|
| `debug/training_labels/{set_num}_manual_color_calibration.json` | Manual colour calibrations for a set | `POST /debug/save-manual-color-calibration` |
| `debug/ai_training/ai_training.sqlite` | Training bundle index, candidate examples, confirmations | Training store service |
| `debug/ai_training/training_store/index.json` | Training pack bundle registry | `training_store_service` |

---

## Derived Files

Derived files are computed from authoritative state. They can be regenerated.
They must never be edited by hand or treated as a source of truth.

| File | Derived from | How regenerated |
|---|---|---|
| `debug/training_labels/{set_num}_bag{bag}_clip_memory.json` | Training labels + preview imagery | `POST /debug/buildability-clip-suggest` (updates incrementally) |
| `debug/confirmed_memory/confirmed_memory_embeddings.npy` | `ai_training.sqlite` confirmed examples + training CLIP embeddings | `confirmed_memory_service.build_confirmed_memory_index()` |
| `debug/confirmed_memory/confirmed_memory_items.json` | Same | Same |
| `debug/catalog_clip_embeddings/embeddings.npy` | Catalog part images | `POST /debug/generate-catalog-clip-embeddings` |
| `debug/catalog_clip_embeddings/items.json` | Catalog part images | Same |
| `debug/catalog_clip_embeddings/manifest.json` | Same | Same |
| `debug/ai_training/analysis_bundles/` | Crop analysis + masks | Training store pipeline |
| `debug/ai_training/step_segmented_cutouts/` | Crop images + segmentation | Segmentation pipeline |
| `debug/ai_training/part_cutouts/` | Crop images + segmentation | Segmentation pipeline |
| `debug/catalog_maps/{set_num}_element_map.json` | Catalog DB query for set | Catalog pipeline |

---

## Caches

Caches are inputs to the review pipeline. They are written by upstream processes
(crop detection) and are **read-only** from the perspective of the Bag Review
workflow. They must not be modified by the review service.

| File | Purpose | Written by |
|---|---|---|
| `debug/crop_cache/{set_num}_bag{bag}.json` | Detected crop regions with page/step/bbox/qty metadata | Crop generation pipeline |
| `debug/server_catalog/lego_catalog.db` | LEGO part catalog (parts, colors, elements) | External catalog tool |
| `debug/part_image_cache/` | Cached part images from Rebrickable | Catalog image caching tool |

---

## CLIP Memory: Authoritative vs Derived

`debug/training_labels/{set_num}_bag{bag}_clip_memory.json` lives next to the
training labels file but is **not** review state. It caches CLIP embeddings
computed from confirmed label+image pairs to speed up suggestions. It can be
deleted and regenerated without any loss of authoritative data.

---

## Schema Version

The current schema version for training labels is `"1.1"`.

A label file has this top-level structure:

```json
{
  "schema_version": "1.1",
  "set_num": "70618",
  "bag": 1,
  "created_at": "<ISO8601Z>",
  "source": {
    "route": "/debug/manual-match-review",
    "type": "bag_review",
    "crop_image_path_kind": "page_image_with_crop_box"
  },
  "crops": {
    "<crop_id>": {
      "page": 7,
      "step": 1,
      "qty": [1],
      "qty_text": ["1x"],
      "status": "good",
      "review_status": "reviewed",
      "crop_box": [...],
      "crop_box_format": "xywh",
      "crop_image_path": "...",
      "parts": [...],
      "unknown_slots": [],
      "ignored_slots": [],
      "annotated_at": "<ISO8601Z>"
    }
  }
}
```
