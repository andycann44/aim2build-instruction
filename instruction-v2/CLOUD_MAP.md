# instruction-v2 Cloud Map

This document maps the current cloud/storage usage that touches the instruction analyzer and training/review flow. It is documentation only; no V2 cloud migration is implemented here.

Important naming note: the repo uses the variable name `AZURE_DATABASE_URL`, but the current database client is `psycopg`, not SQL Server. I found no active `pyodbc`, TDS, or SQL Server/Azure SQL Database client path. The current "Azure SQL" path is therefore an Azure-hosted Postgres-style bundle index unless the connection string points somewhere unusual.

## Current Azure-Named Database Usage

| Area | Current code location | Tables | Reads | Writes | Notes |
|---|---|---|---|---|---|
| Training bundle cloud index | `clean/services/training_bundle_index_service.py` | `training_bundle_index`, `candidate_training_examples` | Bundle rows, review queue, review stats, candidate confirmations, confirmed part totals/usage | Creates schema, upserts bundle rows, updates review status, stores AI analysis JSON, split candidate paths, candidate confirmations | Uses `AZURE_DATABASE_URL`; connects with `psycopg.connect(..., row_factory=dict_row)`. |
| Local bundle registration mirroring | `clean/services/training_store_service.py::register_analysis_bundle` | `training_bundle_index` through `register_bundle()` | Local `debug/ai_training/training_store/index.json`, bundle metadata | Registers/updates Postgres bundle row after local bundle registration | Local JSON remains the immediate bundle index; cloud row is a mirror/lookup. |
| R2 upload post-registration | `clean/services/training_cloud_sync_service.py::upload_bundle_to_r2` | `training_bundle_index` through `register_bundle()` | Approved local bundle entry and metadata | Registers bundle row with `r2_prefix` after successful upload | This couples R2 upload completion to cloud bundle-index registration. |
| Candidate confirmation memory/usage | `clean/services/training_bundle_index_service.py::confirm_candidate_part`, `list_confirmed_part_usage`, `list_confirmed_part_totals_for_set`, `unconfirm_candidate_part` | `candidate_training_examples`, joined with `training_bundle_index` | Confirmed part usage by set/part/color | Inserts/updates/deletes confirmed candidate rows | Used for confirmed part accounting and candidate review flow. |
| Split candidate state | `clean/services/training_bundle_index_service.py::update_split_candidates`, `update_split_candidate_status`, `auto_promote_bundle_review_status` | `training_bundle_index` | Existing row `split_candidate_paths` | Updates JSONB split candidates and review promotion | Cloud stores mutable review metadata for training candidates. |
| Routes exposing DB rows/actions | `clean/routers/instruction_debug.py` | Same tables via services above | Bundle index row, confirmed usage, confirmed totals | Candidate confirmation/unconfirmation, review state updates indirectly | Key routes include `/debug/training-store/bundle-index-row`, `/debug/training-store/confirm-candidate-part`, `/debug/training-store/unconfirm-candidate-part`. |

### Current Table Shapes

`training_bundle_index` is created/altered in `training_bundle_index_service.ensure_schema()` with these fields:

- Identity and location: `id`, `bundle_id`, `set_num`, `bag_num`, `page_num`, `step_num`, `crop_num`
- Bundle summary: `slot_count`, `manifest_path`, `r2_prefix`
- Review state: `approved`, `review_status`, `review_notes`, `mask_quality`, `split_quality`, `qty_text_present`, `multi_part_merge`, `reviewed_at`, `reviewed_by`
- AI/split state: `ai_analysis_json`, `ai_reviewed_at`, `ai_model`, `split_candidate_count`, `split_candidate_paths`
- Timestamps: `created_at`, `updated_at`

`candidate_training_examples` is created in the same service with:

- Identity: `id`, `bundle_id`, `candidate_index`
- Confirmed part label: `part_num`, `color_id`, `element_id`, `qty`
- Assets/status: `thumbnail_path`, `r2_path`
- Audit: `confirmed_by`, `confirmed_at`

## Current R2 Usage

| Area | Current code location | Routes | Reads | Writes | Remote keys |
|---|---|---|---|---|---|
| Prepare approved training bundle upload | `clean/services/training_cloud_sync_service.py::prepare_bundle_for_r2` | `GET /debug/training-store/prepare-r2-upload` | Local training store entry and artifact paths | None; dry-run manifest only | `training-bundles/{bundle_id}/{filename}` |
| Upload approved training bundle | `clean/services/training_cloud_sync_service.py::upload_bundle_to_r2` | `POST /debug/training-store/upload-r2`, `POST /debug/training-store/upload-r2-batch` | Approved local training bundle artifacts | Cloudflare R2 via `boto3.client("s3", endpoint_url=...)`; local training store `r2_status`, `r2_paths`; cloud bundle index registration | `training-bundles/{bundle_id}/...` |
| Upload confirmed candidate assets | `clean/services/training_cloud_sync_service.py::upload_confirmed_candidate_assets` | Triggered by `POST /debug/training-store/confirm-candidate-part` | Accepted/qty-clean candidate image, optional mask, generated metadata | Cloudflare R2; Postgres `split_candidate_paths`; local bundle metadata upload status | `confirmed-candidates/{bundle_id}/candidate_{index}_{part}_{color}/...` |
| R2 config loading | `clean/services/training_cloud_sync_service.py::_load_r2_config` | Used by upload functions | Environment or `.env` keys only | None | Required keys: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`; optional `R2_PUBLIC_BASE_URL` |

R2 uploads use `boto3` with a Cloudflare endpoint:

- Endpoint pattern: `https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com`
- Bucket: `R2_BUCKET`
- Public path resolution: `R2_PUBLIC_BASE_URL/{r2_key}` when configured, otherwise the raw R2 key is stored

## Other Cloud/Azure Usage

| Area | Current code location | Purpose | Cloud state? |
|---|---|---|---|
| Azure/OpenAI ranking/review | `clean/services/azure_openai_service.py`, `clean/services/training_ai_review_service.py` | Calls Azure OpenAI/OpenAI for ranking or AI review when env vars are present | No durable cloud storage in these services, aside from results written back through local/DB paths elsewhere. |
| MCP/tool reference | `tools/a2b_mcp_server.py` | Mentions `clean/services/azure_openai_service.py` as an available source | No storage. |

## Local State That Should Stay Local For Now

These are high-churn pipeline/debug artifacts and should remain local during V2 pipeline stabilization:

- `instruction-v2/indexes/*.json`: V2 stage manifests while the pipeline is still being proven.
- `instruction-v2/pages/`: rendered page PNGs for local deterministic testing.
- `instruction-v2/debug/`: candidate, bag map, and step overlay debug images.
- Existing Bag Review source of truth: `debug/training_labels/{set_num}_bag{bag}.json` until V2 has an explicit replacement.
- Existing crop/segmentation intermediate outputs: crop caches, cutouts, masks, overlays, CLIP test outputs.
- Local training bundle source artifacts until a bundle is approved for upload.
- Secret/config files such as `.env`; document key names only, never values.

## What Should Eventually Move To Azure-Backed Database

For V2, the database should own durable review/training metadata, not rendered image blobs.

Good database candidates:

- Set/run identity: `set_num`, source PDF hash, pipeline version, run id.
- Stage manifest registry: stage name, manifest path or R2 key, sha256, created_at, status.
- Bag map decisions after human/system validation: bag number, start/end pages, confidence, evidence pages.
- Step map decisions after validation: bag, page, step number, step box, confidence, source.
- Human review labels after V2 replaces the current JSON source of truth: slot id, part/color/qty, status, reviewer, timestamps.
- Training candidate confirmations: equivalent of `candidate_training_examples`.
- Review state and audit trail: confirmed/unknown/ignored/rejected/needs-fix.
- Derived memory/index metadata: hashes and pointers to generated memory indexes, not raw embeddings unless there is a clear retrieval plan.

Existing table direction to reuse or evolve:

- `training_bundle_index` can become or inspire a V2 `pipeline_bundle_index` / `review_bundle_index`.
- `candidate_training_examples` can become or inspire a V2 `confirmed_slot_labels` table.

## What Should Eventually Move To R2

R2 should hold large binary assets and stable inspectable artifacts:

- Source PDFs after ingestion.
- Rendered page PNGs if shared review/replay is needed.
- Bag candidate/debug crops only when needed for audit or review.
- Step overlay images for approved/validated runs.
- Callout crops, slot crops, part cutouts, shape masks, and segmentation overlays.
- Candidate thumbnails and confirmed training examples.
- Manifest snapshots when a stage is accepted, using content-addressed or run-addressed keys.
- CLIP/catalog embedding files only if they need to be shared across machines or workers.

Suggested V2 R2 key shape:

```text
instruction-v2/{set_num}/{run_id}/pdf/source.pdf
instruction-v2/{set_num}/{run_id}/pages/page_001.png
instruction-v2/{set_num}/{run_id}/manifests/05_step_map.json
instruction-v2/{set_num}/{run_id}/debug/step_map/bag_01_page_012.png
instruction-v2/{set_num}/{run_id}/review/slots/{slot_id}/part_cutout.png
instruction-v2/{set_num}/{run_id}/training/{label_id}/thumbnail.png
```

## What Should Not Move To Cloud Yet

- Unverified Stage 2/3 candidate noise. Keep local until bag detection is validated.
- Raw temporary render directories and transient failed-run outputs.
- Experimental ranking weights, penalties, or model-debug traces.
- Local-only UI state while Bag Review is still being shaped.
- Any duplicate source of truth for labels before the review-state migration is explicitly designed.

## V2 Boundary Recommendation

For the next V2 stages, keep cloud as a documented boundary only:

1. Continue writing local `instruction-v2/indexes/*.json` manifests.
2. Add no cloud writes until the local stage sequence is stable through human review.
3. When ready, add a single cloud export stage that uploads accepted manifests/assets to R2 and registers pointers/checksums in the Azure-backed database.
4. Keep label source of truth singular: either current JSON or future DB row, not both as writable authorities.
