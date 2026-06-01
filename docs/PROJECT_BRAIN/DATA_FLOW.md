# Data Flow

End-to-end flow showing exactly which files are read and written at each stage.

---

## Stage 1 — PDF Ingestion

```
Input:  LEGO instruction PDF (loaded via /debug/load-set)
Output: Page images stored in the loaded set's page directory

Route:  GET /debug/load-set  →  POST /debug/load-set
```

The PDF is rasterised into per-page images. The path to the current pages
directory is tracked in:

```
debug/.current_pages_dir
```

---

## Stage 2 — Crop Detection (Crop Generation Pipeline)

```
Input:  Page images from the loaded set
Output: debug/crop_cache/{set_num}_bag{bag}.json

Routes: /api/full-bag-scan, /api/bag-find-chunked, /debug/full-bag-scan
```

The crop detection pipeline scans instruction pages, identifies part callout
regions, and writes a crop cache file containing:

- `crop_id` — unique identifier (`p{page}_s{step}_c{callout}`)
- `page`, `step`
- `crop_box` — bounding box in `xywh` format
- `crop_image_path` — path to the source page image
- `qty_numbers`, `qty_text` — detected quantities
- `data_uri` — optional inline image preview

**The crop cache is read-only from this point forward.** The Bag Review service
reads it but never writes to it.

---

## Stage 3 — Bag Review

```
Input:  debug/crop_cache/{set_num}_bag{bag}.json       (read-only)
        debug/training_labels/{set_num}_bag{bag}.json  (read + write)

UI:     GET /debug/manual-match-review?set_num=...&bag=...
Writes: debug/training_labels/{set_num}_bag{bag}.json
```

The human reviewer:

1. Views each crop and its detected quantity slots
2. For each slot, assigns a part label (part_num + color_id + qty), OR marks
   it unknown, OR marks it ignored
3. Each action writes to the training labels JSON via `bag_review_service`

### File reads per review page load

```
manual_match_review route handler
  → reads: debug/server_catalog/lego_catalog.db  (parts list and colour data for the UI)
  → calls: bag_review_service.build_review_model(set_num, bag)

bag_review_service.build_review_model(set_num, bag)
  → reads: debug/crop_cache/{set_num}_bag{bag}.json
  → reads: debug/training_labels/{set_num}_bag{bag}.json
  → reads: debug/ai_training/step_segmented_cutouts/ (checks for slot asset files)
  → reads: debug/ai_training/part_cutouts/ (checks for slot asset files)
```

### File writes per review action

```
POST /debug/save-label
  → writes: debug/training_labels/{set_num}_bag{bag}.json
  (via bag_review_service.save_slot_label())

POST /debug/mark-slot-unknown
  → writes: debug/training_labels/{set_num}_bag{bag}.json
  (via bag_review_service.mark_slot_unknown())

POST /debug/mark-slot-ignored
  → writes: debug/training_labels/{set_num}_bag{bag}.json
  (via bag_review_service.mark_slot_ignored())
```

---

## Stage 4 — CLIP Suggestion (Assists Review; Does Not Replace It)

```
Input:  debug/training_labels/{set_num}_bag{bag}.json    (read-only for review state)
        debug/training_labels/{set_num}_bag{bag}_clip_memory.json  (CLIP cache)
        debug/catalog_clip_embeddings/embeddings.npy
        debug/catalog_clip_embeddings/items.json
        debug/ai_training/step_segmented_cutouts/         (slot imagery)
        debug/ai_training/part_cutouts/                   (part imagery)

Route:  POST /debug/buildability-clip-suggest

Output: JSON suggestions returned to UI (top-N candidates)
        debug/training_labels/{set_num}_bag{bag}_clip_memory.json  (CLIP cache updated)
```

The suggestion pipeline:

1. Reads confirmed labels from the training labels file
2. Loads or builds CLIP embeddings for confirmed label+image pairs into CLIP memory
3. Queries catalog CLIP embeddings for similarity
4. Returns top-N part suggestions to the UI

**The suggestion route reads review state but MUST NOT write to it.**  
**CLIP memory is derived and not authoritative.**

---

## Stage 5 — Training Label Export

```
Input:  debug/training_labels/{set_num}_bag{bag}.json
        debug/ai_training/step_segmented_cutouts/
        debug/ai_training/part_cutouts/

Routes: POST /debug/training-store/register-bundle
        POST /debug/training-store/export-batch
        POST /debug/export-training-packs

Output: debug/ai_training/training_store/
        debug/ai_training/analysis_bundles/
        debug/ai_training/ai_training.sqlite  (bundle index)
```

Reviewed labels are bundled into training packs with associated imagery
artefacts. The training store tracks bundle state in `ai_training.sqlite`.

---

## Stage 6 — Cloud Upload

```
Input:  debug/ai_training/training_store/ (approved bundles)

Routes: POST /debug/training-store/upload-r2
        POST /debug/training-store/upload-r2-batch
        GET  /debug/training-store/prepare-azure-index

Output: Cloudflare R2 / Azure AI storage
```

Approved bundles are uploaded to cloud storage for model training.

---

## Stage 7 — CLIP Catalog Generation (Periodic / On-Demand)

```
Input:  debug/server_catalog/lego_catalog.db
        debug/part_image_cache/

Route:  POST /debug/generate-catalog-clip-embeddings

Output: debug/catalog_clip_embeddings/embeddings.npy
        debug/catalog_clip_embeddings/items.json
        debug/catalog_clip_embeddings/manifest.json
```

Generates CLIP embeddings for the entire part catalog. Run when the catalog
changes or embeddings need refreshing. Output is used by the suggestion pipeline.

---

## Full File Flow Summary

```
PDF
 │
 ▼
Page images (loaded set dir)
 │
 ▼  [Crop Generation Pipeline]
debug/crop_cache/{set_num}_bag{bag}.json          ← WRITTEN by crop pipeline
 │                                                  READ-ONLY for review
 ▼  [bag_review_service.build_review_model()]
Review UI (/debug/manual-match-review)
 │
 ├── save-label ──────────────────────────────────►  debug/training_labels/{set_num}_bag{bag}.json
 ├── mark-slot-unknown ───────────────────────────►  (SOURCE OF TRUTH)
 └── mark-slot-ignored ───────────────────────────►
                                                           │
                    [buildability-clip-suggest]            │
  debug/catalog_clip_embeddings/ ──────────────────────────┤
  step_segmented_cutouts/ ─────────────────────────────────┤
  part_cutouts/ ───────────────────────────────────────────┘
                                                           │
                                                           ▼ (derived)
                               debug/training_labels/{set_num}_bag{bag}_clip_memory.json
                                                           │
                                                           ▼ [Training Store Pipeline]
                                     debug/ai_training/training_store/
                                     debug/ai_training/ai_training.sqlite
                                                           │
                                                           ▼ [Cloud Upload]
                                               Cloudflare R2 / Azure
```
