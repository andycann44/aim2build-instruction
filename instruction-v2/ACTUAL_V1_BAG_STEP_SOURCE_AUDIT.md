# Actual V1 Bag / Step / Crop Source Audit

Created: 2026-06-03

## Executive Finding

The remembered working V1 flow did **not** come from raw `page_analyzer.analyze_page` alone.

The working V1 bag/step/crop flow is a composed workflow:

1. Bag starts are persisted in V1 truth stores.
2. Step ranges/candidates are found from printed step-number resets.
3. `/debug/instruction-buildability` reads saved bag truth, detects steps, builds callout crops, and emits crop IDs like `p23_s27_c1`.
4. Review/save routes then persist labels under `debug/training_labels/`.

The most important discovery: `clean/routers/instruction_debug.py::_resolve_bag_page_range()` reads `debug/bag_truth.db` via `clean/services/bag_truth_store.py`, and that DB already contains Bag 7 page 131 and Bag 8 page 148.

Raw current V1 `page_analyzer/analyzer_scan/gap_scan` was proven separately to miss or misread the remembered proof pages. Therefore V2 should not port that raw analyzer as the source of truth for bag starts.

## Files / Stores Found

### Bag truth stores

#### `bag_inspector.db`

Used by:

- `clean/services/truth_service.py`
- `clean/services/sequence_service.py`
- `clean/services/gap_scan_service.py`
- `clean/routers/gap_review.py`
- `clean/routers/sequence.py`
- `clean/routers/debug.py::debug_save_bag_truth`

Observed rows for set `70618`:

| Bag | Start Page | Source | Confidence | Confirmed |
| --- | ---: | --- | ---: | ---: |
| 1 | 6 | manual | 1.0 | 1 |
| 3 | 39 | manual | 1.0 | 1 |
| 4 | 58 | manual | 1.0 | 1 |
| 5 | 81 | manual | 1.0 | 1 |
| 6 | 104 | manual | 1.0 | 1 |

This is a human-confirmed/manual truth subset used by the sequence/gap review flow.

#### `debug/bag_truth.db`

Used by:

- `clean/services/bag_truth_store.py`
- `clean/routers/debug.py::debug_bag_truth_visual`
- `clean/routers/instruction_debug.py::_resolve_bag_page_range`
- `clean/routers/instruction_debug.py::_build_instruction_callout_crops`

Observed rows for set `70618`:

| Bag | Start Page | Source | Confidence |
| --- | ---: | --- | ---: |
| 1 | 6 | detector | 0.65 |
| 2 | 22 | detector | 0.65 |
| 3 | 39 | detector | 0.65 |
| 4 | 58 | detector | 0.65 |
| 5 | 81 | detector | 0.65 |
| 6 | 104 | detector | 0.65 |
| 7 | 131 | detector | -0.15 |
| 8 | 148 | detector | -0.15 |
| 9 | 164 | detector | -0.15 |
| 10 | 179 | detector | -0.15 |
| 11 | 194 | detector | 0.65 |
| 12 | 213 | detector | -0.15 |
| 13 | 237 | manual | 0.998 |
| 14 | 266 | manual | 0.998 |
| 15 | 274 | manual | 0.998 |

This is the store that matters for the old instruction-buildability crop flow.

### Crop caches

#### `debug/crop_cache/70618_bag1.json`

Observed:

- 22 crops
- pages 7-21
- steps 1-25
- first IDs: `p7_s1_c1`, `p7_s2_c2`, `p8_s3_c1`, `p8_s4_c2`, `p9_s5_c1`

This proves V1 Bag 1 was not beginning at page 12 in the working crop flow.

#### `debug/crop_cache/70618_bag2.json`

Observed:

- 15 crops
- pages 22-38
- steps 26-40
- first IDs: `p22_s26_c1`, `p23_s27_c1`, `p24_s28_c1`, `p25_s29_c1`, `p26_s30_c1`

This aligns with `debug/bag_truth.db` Bag 2 start page 22.

### Training labels

Found:

- `debug/training_labels/70618_bag1.json`
- `debug/training_labels/70618_bag2.json`
- `debug/training_labels/70618_bag3.json`
- `debug/training_labels/70618_bag4.json`

These are review/label state, not the original bag-start detector. They contain crop IDs, page, step, qty, crop box, status, review status, and parts.

Important note: `debug/training_labels/70618_bag2.json` contains stale or manually adjusted crop boxes. Example:

- `p23_s27_c1` has `crop_box: [113, 3, 313, 193]`
- fresh `debug/crop_cache/70618_bag2.json` has `crop_box: [114, 29, 313, 161]`

This matches the known rule: fresh/final detected crop box wins over stale saved crop box.

## Endpoint / UI Map

### Step bag scan

Endpoint:

- `GET /api/step-bag-scan`

Source:

- `clean/routers/step_bag_scan.py::api_step_bag_scan`
- `clean/services/step_sequence_bag_service.py::scan_step_bag_sequence`

Behaviour:

- Reads rendered pages.
- Calls `step_detector_service.detect_steps`.
- Tracks visible printed main step numbers.
- Finds candidate bag starts from step sequence reset, for example high previous step -> step 1.

This is the code path that matches the human reasoning: “page 7 is step 1 and 2, so the bag page is likely page 6.”

### OpenAI bag verification

Endpoint:

- `GET /api/step-bag-scan-openai-verify`

Source:

- `clean/routers/step_bag_openai_scan.py::api_step_bag_scan_openai_verify`
- `clean/services/step_sequence_openai_bag_service.py::scan_step_bag_sequence_openai_verify`

Cache found:

- `debug/openai_bag_verify/70618/page_106.json`

Observed page 106 cache:

- `is_real_bag_start: false`
- `main_steps: [163]`
- `previous_page_main_steps: [1,2,3,4,5]`
- confidence `0.98`

This is verification evidence, not the primary bag truth.

### Gap review / save truth

Endpoints:

- `GET /debug/gap-review`
- `GET /gap-review`
- `GET /api/gap-scan`
- `POST /api/debug/save-bag-truth`
- `GET /debug/bag-truth-visual`

Sources:

- `clean/routers/gap_review.py::gap_review`
- `clean/routers/gap_scan.py::api_gap_scan`
- `clean/routers/debug.py::debug_save_bag_truth`
- `clean/routers/debug.py::debug_bag_truth_visual`
- `clean/services/truth_service.py`
- `clean/services/bag_truth_store.py`

Important split:

- `gap_review.py` reads `sequence_service` / `gap_scan_service`, which rely on `truth_service` and `bag_inspector.db`.
- `instruction_debug.py` crop-building reads `bag_truth_store` and `debug/bag_truth.db`.

V2 must account for this split before claiming V1 parity.

### Instruction buildability / crop generation

Endpoint:

- `GET /debug/instruction-buildability`

Source:

- `clean/routers/instruction_debug.py::instruction_buildability`
- `clean/routers/instruction_debug.py::_build_instruction_callout_crops`
- `clean/routers/instruction_debug.py::_resolve_bag_page_range`

Behaviour:

- Resolves bag page range from `debug/bag_truth.db`.
- Detects printed step boxes through `step_detector_service.detect_steps`.
- Filters impossible step anchors with `_filter_invalid_step_anchor_boxes`.
- Detects callout crops with `_detect_callout_rect_by_edges`.
- Falls back to page-level callout candidates when edge detection fails.
- Extracts qty with `_qty_payload_for_page_level_callout_crop` / `_auto_qty_payload_for_crop`.
- Generates crop IDs using:

```text
crop_id = f"p{int(page)}_s{max(step_number, 0)}_c{idx}"
```

This is where IDs like `p23_s27_c1` are generated.

### Manual match / label review

Endpoint:

- `GET /debug/manual-match-review`

Source:

- `clean/routers/instruction_debug.py::manual_match_review`
- `clean/services/bag_review_service.py`
- `POST /debug/save-label`
- `POST /debug/mark-slot-unknown`
- `POST /debug/mark-slot-ignored`

Behaviour:

- Reads `debug/training_labels/{set_num}_bag{bag}.json`.
- Shows crops/slots/candidates.
- Saves human review decisions back to training labels.

This is label state, not bag-map discovery.

## Answered Questions

### 1. Which V1 endpoint/UI produced the working bag/step/crop flow?

The working crop flow came from:

- `GET /debug/instruction-buildability`
- backed by `clean/routers/instruction_debug.py::_build_instruction_callout_crops`

The bag-start support came from:

- `debug/bag_truth.db` via `clean/services/bag_truth_store.py`
- plus diagnostic/review UIs such as `/debug/bag-truth-visual`, `/debug/gap-review`, and `/api/step-bag-scan`.

### 2. Was the correct bag map raw OCR, cached output, or human-reviewed/corrected data?

It was persisted truth/cached state, not raw OCR alone.

There are two truth layers:

- `bag_inspector.db`: human-confirmed/manual subset used by sequence/gap services.
- `debug/bag_truth.db`: fuller detector/manual bag truth used by instruction-buildability crop generation.

The remembered pages 131 and 148 are present in `debug/bag_truth.db`, not reproduced by current raw `page_analyzer`.

### 3. Where are bag starts stored?

Bag starts are stored in:

- `bag_inspector.db`, table `bag_truth`, accessed by `clean/services/truth_service.py`
- `debug/bag_truth.db`, table `bag_truth`, accessed by `clean/services/bag_truth_store.py`

For the actual buildability crop flow, `debug/bag_truth.db` is the key store.

### 4. Where are step/crop IDs like `p23_s27_c1` generated?

They are generated in:

- `clean/routers/instruction_debug.py::_build_instruction_callout_crops`

The literal format is:

```text
p{page}_s{step_number}_c{idx}
```

Example:

- page 23
- step 27
- first callout candidate
- `p23_s27_c1`

### 5. Which exact V1 source should V2 port next?

V2 should port the actual buildability bag/step/crop source, not raw bag OCR:

1. `clean/services/bag_truth_store.py`
   - Read `debug/bag_truth.db` bag starts as the V1 buildability truth source.
   - This should drive V2 bag map parity before further matching.

2. `clean/routers/instruction_debug.py::_resolve_bag_page_range`
   - Recreate exact page range resolution from saved bag truth.

3. `clean/services/step_sequence_bag_service.py::scan_step_bag_sequence`
   - Use as diagnostic/recovery evidence for missing starts, especially Bag 1 page 6 / step 1 on page 7.

4. `clean/routers/instruction_debug.py::_build_instruction_callout_crops`
   - Port once bag ranges are sourced from the same truth store.
   - This is the actual source of `p*_s*_c*` crop IDs and V1 callout crops.

## V2 Recommendation

Stop treating visible bag number recognition as the next source of truth.

Next V2 recovery item should be:

```text
Read V1 buildability bag truth from debug/bag_truth.db
→ write/repair V2 04_bag_map.json from that persisted V1 truth
→ preserve a clear source field: v1_debug_bag_truth_db
```

Only after that should V2 port `_build_instruction_callout_crops`, because that function assumes bag ranges are already available from V1 truth.
