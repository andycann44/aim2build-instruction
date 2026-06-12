# Unmapped V1 Audit

This audit lists V1 files in `clean/services/*.py`, `clean/routers/*.py`, `clean/*.py`, and `tools/*.py` that do not yet have their own entry in `instruction-v2/MIGRATION_LEDGER.md`.

Documentation only. Do not delete or archive from this list without a separate review.

## Summary

- Existing migration ledger mapped entries: 7
- Unmapped files found here: 42
- None of these files are marked safe to delete by this audit.

## Unmapped Files

| File path | Short purpose guess | Imported anywhere? | Grep evidence of imports/routes | Suggested category |
|---|---|---:|---|---|
| `clean/services/ai_snap_crop_service.py` | SAM/foreground mask and crop/cutout service for slot segmentation. | yes | `clean/routers/instruction_debug.py` references auto mask slots; `clean/services/part_crop_normalize_service.py` imports `_foreground_mask_for_image`. | candidate for V2 later |
| `clean/services/analyzer_scan_service.py` | Page analyzer scan orchestration over rendered pages. | yes | `clean/routers/analyzer_scan.py` imports `analyzer_scan_service`. | still-needed V1 route |
| `clean/services/auto_confirm_service.py` | Auto-confirmation helper for high-confidence candidates/truth rows. | unknown | No route decorator in file; imports `analyzer_scan_service` and `truth_service`. | unknown needs review |
| `clean/services/azure_openai_service.py` | Azure OpenAI image/crop ranking helpers. | yes | Imports V1 debug OpenAI helpers; likely called from instruction/debug AI paths. | candidate for V2 later |
| `clean/services/bag_review_service.py` | Bag review state service around training label JSON. | yes | Used by `clean/routers/instruction_debug.py` Bag Review work. | still-needed V1 route |
| `clean/services/bag_truth_store.py` | SQLite store for confirmed bag/page truth. | yes | Imported by truth/debug workflow services. | still-needed V1 route |
| `clean/services/confirmed_memory_service.py` | Confirmed-label memory scoring/index service. | yes | Imports `training_bundle_index_service`; used by training/memory paths. | candidate for V2 later |
| `clean/services/dataset_service.py` | Load/download instruction PDFs and switch rendered page directories. | yes | `clean/routers/load_set.py` imports `dataset_service`. | still-needed V1 route |
| `clean/services/debug_service.py` | Resolve rendered page image paths and page dirs. | yes | Imported by many V1 services including analyzer, gap scan, inventory, part candidate, precheck, set scan. | still-needed V1 route |
| `clean/services/gap_scan_service.py` | V1 missing bag gap scanner. | yes | `clean/routers/gap_scan.py` and `clean/routers/gap_review.py` import it. | candidate for V2 later |
| `clean/services/instruction_buildability_source.py` | Catalog DB set-parts loader. | yes | Used by V2 Stage 0 as read-only catalog source; likely imported by instruction debug/buildability paths. | candidate for V2 later |
| `clean/services/inventory_scan_service.py` | OCR/catalog matching for inventory pages. | unknown | Imports `debug_service`, OpenCV, Tesseract; no direct route evidence in this scan. | unknown needs review |
| `clean/services/part_candidate_service.py` | Candidate part/color generation from crops/catalog DB. | unknown | Imports `debug_service`; no direct route evidence in this scan. | candidate for V2 later |
| `clean/services/part_crop_matcher.py` | Simple RGB/color-distance matcher. | unknown | Standalone functions `match_by_colour`, `_dist`; no route evidence in this scan. | likely archive later |
| `clean/services/part_crop_normalize_service.py` | Normalize/crop foreground parts from instruction crops. | yes | Imports `_foreground_mask_for_image` from `ai_snap_crop_service`; used around crop normalization flows. | candidate for V2 later |
| `clean/services/precheck_service.py` | OCR/layout precheck for text-heavy or unsuitable pages. | yes | Imported by analyzer/gap/set scan services. | still-needed V1 route |
| `clean/services/set_scan_service.py` | V1 set candidate page scanner. | yes | `clean/routers/set_scan.py` imports `set_scan_service`. | still-needed V1 route |
| `clean/services/step_detector_service.py` | OCR/visual step number detection. | yes | Imported by `clean/routers/step_debug.py`, `clean/routers/debug.py`, `clean/routers/instruction_debug.py`, and workflow routes. | candidate for V2 later |
| `clean/services/training_ai_review_service.py` | AI review over training bundle/split candidates. | yes | Imports training bundle index update helpers. | still-needed V1 route |
| `clean/services/training_bundle_index_service.py` | Azure/Postgres bundle index service. | yes | Imported by confirmed memory, training AI review, cloud sync. | still-needed V1 route |
| `clean/services/training_cloud_sync_service.py` | R2/cloud sync for training artifacts. | yes | Imports training store and bundle index services. | still-needed V1 route |
| `clean/services/training_store_service.py` | Local training store/index helpers. | yes | Imported by training cloud sync; routes in `instruction_debug.py` reference training-store equivalents. | still-needed V1 route |
| `clean/services/truth_service.py` | Confirmed bag truth persistence/query helpers. | yes | Imported by `debug_truth.py`, analyzer/auto-confirm/workflow services. | still-needed V1 route |
| `clean/services/workflow_service.py` | Workflow-level orchestration over analyzer/sequence/truth. | unknown | Imports analyzer scan, sequence, truth services; no direct route evidence in this scan. | unknown needs review |
| `clean/routers/__init__.py` | Router package marker. | yes | Package import target for `clean.routers`. | still-needed V1 route |
| `clean/routers/analyzer_scan.py` | V1 analyzer scan API/debug pages. | yes | Route decorators: `/api/scan-set-with-analyzer`, `/debug/scan-set-with-analyzer`; included by `clean/main.py`. | still-needed V1 route |
| `clean/routers/debug_truth.py` | V1 debug truth API route. | yes | Route decorator: `/api/debug/truth`; included by `clean/main.py`. | still-needed V1 route |
| `clean/routers/gap_review.py` | V1 HTML gap review page. | unknown | Route decorators: `/gap-review`, `/debug/gap-review`; imports `gap_scan_service`, `sequence_service`. Not currently listed in `clean/main.py` imports seen in scan. | candidate for V2 later |
| `clean/routers/gap_scan.py` | V1 gap scan API route. | yes | Route decorator: `/api/gap-scan`; included by `clean/main.py`. | still-needed V1 route |
| `clean/routers/home.py` | V1 home/debug navigation page. | yes | Route decorator: `/`; included by `clean/main.py`. | still-needed V1 route |
| `clean/routers/load_set.py` | V1 load-set debug form/API. | yes | Route decorators: `/debug/load-set`, `/api/load-set-status`; included by `clean/main.py`. | still-needed V1 route |
| `clean/routers/mask_review.py` | V1 mask review UI/API. | unknown | Route decorators: `/debug/mask-review`, `/api/debug/mask-review/verdict`; not seen in `clean/main.py` include scan. | old experiment |
| `clean/routers/sequence.py` | V1 sequence scan API/debug route. | yes | Route decorators use `sequence_service` and `page_analyzer`; included by `clean/main.py`. | still-needed V1 route |
| `clean/routers/set_scan.py` | V1 set candidate scan API/debug pages. | yes | Route decorators: `/api/scan-set-candidates`, `/debug/scan-set-candidates`; included by `clean/main.py`. | still-needed V1 route |
| `clean/routers/step_bag_openai_scan.py` | V1 OpenAI bag sequence verifier route. | yes | Imports `step_sequence_openai_bag_service`; included by `clean/main.py`. | still-needed V1 route |
| `clean/routers/step_bag_scan.py` | V1 local step/bag sequence scan route. | yes | Imports `step_sequence_bag_service`; included by `clean/main.py`. | still-needed V1 route |
| `clean/routers/step_debug.py` | V1 step detector debug/overlay routes. | yes | Route decorators: `/debug/step-detect`, `/debug/step-overlay`; included by `clean/main.py`. | still-needed V1 route |
| `clean/routers/workflow.py` | V1 workflow/debug UI and orchestration routes. | yes | Included by `clean/main.py`; imports `step_sequence_bag_service`, `page_analyzer`, workflow helpers. | still-needed V1 route |
| `clean/main.py` | FastAPI app composition and router mounting. | yes | Imports and includes V1 routers. | still-needed V1 route |
| `tools/a2b_clip_match_probe.py` | Local CLIP/catalog image probe utility. | no | No import evidence in scanned files; referenced in `PIPELINE_MAP.md` as catalog image helper. | candidate for V2 later |
| `tools/a2b_mcp_server.py` | MCP helper exposing git/compile tools. | no | No import evidence in scanned files. | unknown needs review |
| `tools/openai_page_structure.py` | CLI/tool for OpenAI page structure analysis. | no | Imports `clean.services.debug_service.resolve_page_image_path`; no route evidence. | old experiment |

## Notes

- “Imported anywhere?” is based on repository grep evidence in the scanned surfaces, not a full runtime import graph.
- Files marked “likely archive later” or “old experiment” are not approved for deletion. They need a separate owner review.
- Several services are already conceptually represented in V2, but V2 does not yet replace V1 routes, cloud sync, review-label writes, or AI override flows.
