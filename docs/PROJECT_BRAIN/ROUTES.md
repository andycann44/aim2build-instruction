# Routes

All routes live in `clean/routers/`. The main router for Bag Review and
training is `clean/routers/instruction_debug.py`.

---

## Review Routes

These routes directly read or write Bag Review state.

---

### `GET /debug/manual-match-review`

**Purpose:** Render the Bag Review UI for a given set and bag.

**Reads:**
- `debug/training_labels/{set_num}_bag{bag}.json` via `bag_review_service.load_review_state()`
- `debug/crop_cache/{set_num}_bag{bag}.json` via `bag_review_service._load_crop_cache()`
- `debug/server_catalog/lego_catalog.db` (part/color catalog)

**Writes:** Nothing.

**Source-of-truth impact:** Read-only. Returns the full review model including
progress, slot status, and display text.

---

### `POST /debug/save-label`

**Purpose:** Save a part label for a specific slot in a crop.

**Reads:**
- `debug/training_labels/{set_num}_bag{bag}.json`
- `debug/crop_cache/{set_num}_bag{bag}.json`

**Writes:**
- `debug/training_labels/{set_num}_bag{bag}.json`

**Source-of-truth impact:** **Authoritative write.** When `selected_slot_index`
is provided, delegates to `bag_review_service.save_slot_label()`. This is the
canonical path for labelling a slot. Saving a label for a slot automatically
clears that slot from `unknown_slots` and `ignored_slots`.

---

### `POST /debug/mark-slot-unknown`

**Purpose:** Mark a specific slot as unknown (part cannot be identified).

**Reads:**
- `debug/training_labels/{set_num}_bag{bag}.json`
- `debug/crop_cache/{set_num}_bag{bag}.json`

**Writes:**
- `debug/training_labels/{set_num}_bag{bag}.json`

**Source-of-truth impact:** **Authoritative write.** Delegates to
`bag_review_service.mark_slot_unknown()`. Adds slot to `unknown_slots` and
removes it from `ignored_slots`.

---

### `POST /debug/mark-slot-ignored`

**Purpose:** Mark a specific slot as ignored (e.g. a detected extra slot that
is not a real distinct part).

**Reads:**
- `debug/training_labels/{set_num}_bag{bag}.json`
- `debug/crop_cache/{set_num}_bag{bag}.json`

**Writes:**
- `debug/training_labels/{set_num}_bag{bag}.json`

**Source-of-truth impact:** **Authoritative write.** Delegates to
`bag_review_service.mark_slot_ignored()`. Adds slot to `ignored_slots` and
removes it from `unknown_slots`.

---

### `POST /debug/remove-label`

**Purpose:** Remove a saved label from a crop slot.

**Reads:** `debug/training_labels/{set_num}_bag{bag}.json`

**Writes:** `debug/training_labels/{set_num}_bag{bag}.json`

**Source-of-truth impact:** **Authoritative write.** Removes a part entry from
a crop's `parts` list and resets `review_status` if no parts remain.

---

### `POST /debug/set-crop-status`

**Purpose:** Manually set the `status` field of a crop record.

**Reads:** `debug/training_labels/{set_num}_bag{bag}.json`

**Writes:** `debug/training_labels/{set_num}_bag{bag}.json`

**Source-of-truth impact:** **Authoritative write.** Changes crop-level status
(e.g. `"good"`, `"hidden"`).

---

### `POST /debug/delete-crop`

**Purpose:** Delete a crop record entirely from review state.

**Reads:** `debug/training_labels/{set_num}_bag{bag}.json`

**Writes:** `debug/training_labels/{set_num}_bag{bag}.json`

**Source-of-truth impact:** **Authoritative write.** Removes the crop key from
the `crops` dict.

---

### `POST /debug/save-manual-crop`

**Purpose:** Manually create a new crop record in review state.

**Reads:** `debug/training_labels/{set_num}_bag{bag}.json`

**Writes:** `debug/training_labels/{set_num}_bag{bag}.json`

**Source-of-truth impact:** **Authoritative write.**

---

### `POST /debug/update-crop-qty`

**Purpose:** Update the qty/qty_text for an existing crop record.

**Reads:** `debug/training_labels/{set_num}_bag{bag}.json`

**Writes:** `debug/training_labels/{set_num}_bag{bag}.json`

**Source-of-truth impact:** **Authoritative write.**

---

### `POST /debug/save-manual-color-calibration`

**Purpose:** Save a manual colour calibration sample for a specific LEGO colour
against a set's page imagery.

**Reads:** Existing calibration file (if any) at
`debug/training_labels/{set_num}_manual_color_calibration.json`

**Writes:**
- `debug/training_labels/{set_num}_manual_color_calibration.json`

**Source-of-truth impact:** **Authoritative write.** This file is a separate
authoritative state file for colour calibration; it is distinct from the Bag
Review training labels JSON.

---

### `POST /debug/next-unfilled-crop`

**Purpose:** Return the next crop/slot that has not been reviewed.

**Reads:** `debug/training_labels/{set_num}_bag{bag}.json`, crop cache

**Writes:** Nothing.

**Source-of-truth impact:** Read-only.

---

### `GET /debug/export-training-data`

**Purpose:** Export the training label JSON for a set/bag as a download.

**Reads:** `debug/training_labels/{set_num}_bag{bag}.json`

**Writes:** Nothing.

**Source-of-truth impact:** Read-only.

---

## Suggestion Routes

These routes assist the reviewer with part suggestions. They must not write
authoritative review state.

---

### `POST /debug/buildability-clip-suggest`

**Purpose:** Return top-5 CLIP-based part suggestions for a given crop/slot.

**Reads:**
- `debug/training_labels/{set_num}_bag{bag}.json` (reads confirmed labels as
  context; does NOT write review state)
- `debug/training_labels/{set_num}_bag{bag}_clip_memory.json` (CLIP memory cache)
- `debug/catalog_clip_embeddings/embeddings.npy`
- `debug/catalog_clip_embeddings/items.json`
- Crop preview imagery from `debug/ai_training/step_segmented_cutouts/` and
  `debug/ai_training/part_cutouts/`

**Writes:**
- `debug/training_labels/{set_num}_bag{bag}_clip_memory.json` (derived; may
  update incrementally as new confirmed labels are processed)

**Source-of-truth impact:** Must NOT write to
`debug/training_labels/{set_num}_bag{bag}.json`. CLIP memory is derived.

---

### `POST /debug/manual-match-clip-suggest`

**Purpose:** Placeholder CLIP suggest (currently returns empty suggestions).

**Reads:** Nothing meaningful.

**Writes:** Nothing.

**Source-of-truth impact:** None.

---

### `POST /debug/ai-rank-slot`

**Purpose:** AI-assisted slot ranking/matching.

**Reads:** Crop and catalog data.

**Writes:** Nothing authoritative.

**Source-of-truth impact:** Read-only for review state.

---

### `GET /debug/catalog-match-test`

**Purpose:** Test catalog CLIP match for a given crop.

**Reads:** `debug/catalog_clip_embeddings/`, catalog DB, crop data.

**Writes:** Nothing.

**Source-of-truth impact:** None.

---

### `POST /debug/catalog-match-feedback`

**Purpose:** Record feedback on a catalog match result.

**Reads/Writes:** `debug/catalog_match_feedback/` (feedback files).

**Source-of-truth impact:** Feedback only; does not affect review state.

---

## Training Routes

These routes manage training bundle export, review, and cloud upload.

---

### `POST /debug/training-store/register-bundle`

**Purpose:** Register a new training bundle in the training store index.

**Reads:** `debug/ai_training/analysis_bundles/`

**Writes:** `debug/ai_training/training_store/index.json`, bundle metadata

---

### `POST /debug/training-store/export-batch`

**Purpose:** Export a batch of training bundles.

**Reads:** Training store index, analysis bundles.

**Writes:** `debug/ai_training/training_store/`

---

### `GET /debug/training-store/control-panel`

**Purpose:** HTML control panel for training store management.

**Reads:** Training store index, DB.

**Writes:** Nothing.

---

### `POST /debug/training-store/approve-bundle` / `reject-bundle`

**Purpose:** Mark a bundle approved or rejected for training.

**Writes:** Training store index / DB.

---

### `POST /debug/training-store/upload-r2` / `upload-r2-batch`

**Purpose:** Upload training bundles to Cloudflare R2.

**Reads:** Training store bundles.

**Writes:** Remote R2 storage.

---

### `GET /debug/training-store/prepare-azure-index`

**Purpose:** Prepare bundle index for Azure AI deployment.

**Reads:** Training store.

**Writes:** Local index artefacts.

---

### `POST /debug/generate-training-clip-embeddings`

**Purpose:** Generate CLIP embeddings for training pack crops.

**Reads:** Training pack imagery.

**Writes:** `debug/clip_training_embeddings/`

---

### `POST /debug/generate-catalog-clip-embeddings`

**Purpose:** Generate CLIP embeddings for the full part catalog.

**Reads:** `debug/server_catalog/lego_catalog.db`, part images.

**Writes:** `debug/catalog_clip_embeddings/embeddings.npy`, `items.json`, `manifest.json`

---

### `POST /debug/export-training-packs`

**Purpose:** Bundle and export training data packs.

**Reads:** Training labels, crop imagery.

**Writes:** Training pack archives.

---

## Debug-Only Routes

These routes are diagnostic/visual tools. They do not write authoritative state.

| Route | Purpose |
|---|---|
| `GET /api/step-bag-scan-openai-verify` | OpenAI-assisted step-bag scan verification |
| `GET /debug/manual-page-image` | Serve a page image |
| `GET /debug/ai-snap-artifact` | Serve an AI snap artefact file |
| `GET /debug/export-crop-analysis-bundle` | Export analysis bundle for a crop |
| `GET /debug/training-store/bag-completion-audit` | Audit bag completion in training store |
| `GET /debug/training-store/review-queue` | Show pending review queue |
| `GET /debug/training-store/review-stats` | Training store review statistics |
| `GET /debug/training-store/bundle-debug` | Debug info for a bundle |
| `GET /debug/training-store/candidate-alpha-debug` | Alpha debug for a candidate |
| `GET /debug/dev/find-obsolete-candidates` | Find obsolete candidates |
| `GET /debug/training-store/review-ui` | Training store review UI |
| `GET /debug/normalize-part-crop` | Part crop normalisation preview |
| `GET /debug/normalize-slot-crop` | Slot crop normalisation preview |
| `GET /debug/instruction-buildability` | Instruction buildability analysis view |
| `GET /debug/shape-overlay-diagnostic` | Shape overlay diagnostic |
| `GET /debug/clip-match-file` | CLIP match file viewer |
| `GET /debug/page-image` | Serve page image |
| `GET /debug/bag-parts-lab` | Bag parts visual lab |
| `GET /debug/step-part-lab` | Step/part visual lab |
| `GET /debug/bag-truth-visual` | Bag truth visual |
| `GET /debug/bag-step-contact-sheet` | Bag step contact sheet |
| `GET /debug/inventory-overlay` | Inventory overlay |
| `GET /debug/gap-table` | Gap analysis table |
| `POST /debug/auto-mask-slots` | Auto-mask slot regions |
| `POST /debug/slot-mask-candidates` | Slot mask candidates |
| `GET /debug/mask-review` | Mask review UI |
| `GET /debug/gap-review` | Gap review UI |
| `GET /health` | Health check |
