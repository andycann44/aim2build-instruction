# instruction-v2 Pipeline Map

This map records the proven current Bag 1 code path so `instruction-v2` can be built from one known-good reference instead of rediscovering the pipeline.

Bag 1 baseline reference:

- Source of truth: `debug/training_labels/70618_bag1.json`
- sha256: `fdf5ca0a5cc04787e7d5d7ef699ed3c928d4a4236521b44e80fc17c55879522b`
- Result: 48 slots total, 47 confirmed, 1 ignored, 100% complete

Rules for V2:

- Do not duplicate review state. `debug/training_labels/{set_num}_bag{bag}.json` remains the authoritative human review state until V2 deliberately replaces it.
- V2 manifests should point to source assets, derived assets, and checksums. They should not create a second label truth.
- Current `clean/` and `debug/` paths below are source material only.

| Stage | Existing source file/function | Input | Output | V2 manifest name | Proven by Bag 1 |
|---|---|---|---|---|---|
| 0. Catalog set parts | `clean/services/instruction_buildability_source.py::load_instruction_set_parts` | `set_num`; `debug/server_catalog/lego_catalog.db` tables/views including set requirements, elements, images, and colors | Set part rows with `part_num`, `color_id`, color metadata, required qty, element/image info, and shape/color keys | `instruction-v2/indexes/catalog_parts_manifest.json` | Yes. Bag 1 review and candidate matching use this catalog part universe. |
| 1. PDF render | Current page lookup: `clean/services/debug_service.py::resolve_page_image_path`; V2 scaffold: `instruction-v2/phase1_pdf_pages.py::run_phase1` | Instruction PDF or existing rendered page directory | Saved PDF, rendered page PNGs, and page index | `instruction-v2/indexes/01_pdf_manifest.json`; `instruction-v2/indexes/02_page_index.json` | Yes for the current rendered Bag 1 pages under `debug/70618/.../pages`; V2 render scaffold exists. |
| 2. Bag candidate detection | `clean/routers/debug.py::_resolve_bag_page_range`; `clean/services/step_sequence_bag_service.py::build_step_sequence_prepass_for_pages`; `clean/services/step_sequence_bag_service.py::scan_step_bag_sequence`; optional verifier `clean/services/step_sequence_openai_bag_service.py::scan_step_bag_sequence_openai_verify` | Page index/rendered pages, detected step sequence, bag-start evidence | Candidate bag-number/evidence pages | `instruction-v2/indexes/03_bag_candidates.json` | Yes. Bag 1 crop cache was built from a resolved Bag 1 range; V2 keeps candidates separate first. |
| 3b. Bag gap review / missing bag window review | Uploaded references `gap_scan.py` and `gap_review.py` | Candidate bag pages, rendered pages, and detected evidence gaps | Missing bag start windows, candidate pages for review, and human corrections before final bag map | `instruction-v2/indexes/03b_bag_gap_review.json` | Newly identified as required. Bag 1 proves final ranges matter, but V2 needs this correction checkpoint before `04_bag_map.json`. |
| 3. Bag range map | `clean/routers/debug.py::_resolve_bag_page_range`; sequence/range logic from `clean/services/step_sequence_bag_service.py`; consumes gap review corrections when present | Candidate bag pages, page index, and gap review corrections | Final/provisional bag page ranges with confidence and evidence pages | `instruction-v2/indexes/04_bag_map.json` | Yes. Bag 1 review depended on a concrete bag range; V2 records the map as its own manifest. |
| 4. Step box map | `clean/services/step_detector_service.py::detect_steps`; `clean/routers/debug.py::_contact_sheet_step_boxes_from_detected`; normalization/filtering reference in `clean/routers/instruction_debug.py::_filter_invalid_step_anchor_boxes` | Bag map and rendered page PNGs | Step boxes/anchors per page, with bag/page association and debug overlays | `instruction-v2/indexes/05_step_box_map.json` | Yes. Bag 1 callout crop generation consumed detected step anchors. |
| 5. Callout crop box map | Proven source/reference route: `clean/routers/callout_crop_lab.py`; exact source functions: `clean/services/step_detector_service.py::detect_steps`, `clean/routers/debug.py::_contact_sheet_step_boxes_from_detected`, `clean/routers/callout_crop_lab.py::_build_step_region`, `clean/routers/instruction_debug.py::_detect_callout_rect_by_edges` | Step box map and rendered page PNGs | Callout/crop rectangle boxes per step, with confidence/evidence and debug overlays; no crop image extraction yet | `instruction-v2/indexes/06_callout_crop_box_map.json` | Yes. `callout_crop_lab.py` is the proven lab/reference for the callout crop-box flow used before crop extraction. |
| 6. Qty OCR | `clean/routers/instruction_debug.py::_extract_detected_qty_details_from_crop`; `_auto_qty_payload_for_crop`; `_qty_payload_for_page_level_callout_crop`; underlying import `clean/routers/debug.py::_extract_detected_qty_details_from_crop` | Callout crop image and step context | Quantity text, quantity numbers, token boxes, source regions, and ordered qty list attached to crop records | `instruction-v2/indexes/07_qty_slots_manifest.json` | Yes. Bag 1 yielded 48 review slots, with one verified ignored extra slot. |
| 7. Part segmentation | Route `clean/routers/instruction_debug.py::auto_mask_slots`; service `clean/services/ai_snap_crop_service.py::create_shape_masks_for_callout_slots`; per-slot helper `clean/routers/instruction_debug.py::_segment_step_callout_slot` | Crop records, qty token boxes, crop images | Per-slot segmentation assets including step-masked cutouts, part cutouts, shape masks, and overlay/debug images | `instruction-v2/indexes/08_part_segmentation_manifest.json` | Yes. Bag 1 review used generated `step_segmented_cutouts`, `part_cutouts`, and shape masks. |
| 8. Candidate matching | Route `clean/routers/instruction_debug.py::buildability_clip_suggest`; catalog image helper `tools/a2b_clip_match_probe.py::_ensure_catalog_image_for_pair`; CLIP helpers in `clean/routers/instruction_debug.py` including `_clip_load` and `_clip_embed_image`; catalog source `load_instruction_set_parts` | Slot image assets, catalog set parts, catalog CLIP embeddings, and derived clip memory | Top candidate list with `part_num`, `color_id`, score, candidate image, and score components when available | `instruction-v2/indexes/09_candidate_matches_manifest.json` | Yes. Bag 1 Top 5 suggestions were good enough for human-in-the-loop review. |
| 9. Human review/save label | UI route `clean/routers/instruction_debug.py::manual_match_review`; save route `clean/routers/instruction_debug.py::save_label`; unknown/ignored routes `mark_slot_unknown` and `mark_slot_ignored`; state logic `clean/services/bag_review_service.py::build_review_model`, `save_slot_label`, `mark_slot_unknown`, `mark_slot_ignored` | Crop cache, preview assets, Top 5 candidate response, and human choice | Authoritative review label JSON at `debug/training_labels/{set_num}_bag{bag}.json`; confirmed/unknown/ignored status per slot | `instruction-v2/indexes/10_review_state_manifest.json` as pointer/checksum only, not copied labels | Yes. Bag 1 baseline is verified at 48/48 complete. |

## Derived Assets

These are read-only or derived for review. They are not review source of truth.

| Asset/system | Current path | Role | Source of truth? |
|---|---|---|---|
| Crop cache | `debug/crop_cache/{set_num}_bag{bag}.json` | Derived crop/slot input for review | No |
| Step segmented cutouts | `debug/ai_training/step_segmented_cutouts/` | Slot preview and CLIP input | No |
| Part cutouts | `debug/ai_training/part_cutouts/` | Slot preview and optional CLIP input | No |
| Shape masks | `debug/ai_training/shape_masks/` or mask paths recorded by segmentation | Segmentation/debug evidence | No |
| Catalog embeddings | `debug/catalog_clip_embeddings/items.json`, `debug/catalog_clip_embeddings/embeddings.npy` | Derived candidate matching index | No |
| Clip memory | `debug/training_labels/{set_num}_bag{bag}_clip_memory.json` | Derived from confirmed labels for candidate boosts | No |
| Human labels | `debug/training_labels/{set_num}_bag{bag}.json` | Confirmed, unknown, and ignored review state | Yes |

## End-to-End Flow

```text
0 catalog set parts
-> 1 PDF render
-> 2 bag candidate detection
-> 3b bag gap review / missing bag window review
-> 3 bag range map
-> 4 step box map
-> 5 callout crop box map
-> 6 qty OCR
-> 7 part segmentation
-> 8 candidate matching
-> 9 human review/save label
-> debug/training_labels/{set_num}_bag{bag}.json
-> memory/training derived later
```
