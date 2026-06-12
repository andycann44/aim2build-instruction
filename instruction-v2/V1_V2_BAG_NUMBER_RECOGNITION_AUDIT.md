# V1 vs V2 Bag Number Recognition Audit

Audit date: 2026-06-03

Scope: audit/report only. No manifests were regenerated, no code was changed outside this document, and no V2 bag-map changes were made.

## Goal

Find which V1 function/route recognised visible bag numbers on pages such as 131 and 148, then compare that behavior with current V2.

Known human observations:

- Page 131 shows Bag 7.
- Page 148 shows Bag 8.

## V1 Code Path

The V1 path that recognises visible bag numbers is:

1. `clean/routers/sequence.py`
   - `/api/analyze-gap-page`
   - Configures the rendered page directory and calls `page_analyzer.analyze_page(page, include_image=False)`.
   - Evidence: `clean/routers/sequence.py:13-39`.

2. `clean/services/page_analyzer.py`
   - `analyze_page(page, include_image=False)`
   - Detects a bag intro panel, shell, grey bag blob, number crop, and OCR value.
   - Writes result fields including `bag_number`, `ocr_raw`, `number_box_found`, `panel_found`, `panel_source`, `shell_found`, `grey_bag_found`, `bag_start_card_found`, `white_box_transition_found`, and `confidence`.
   - Evidence: `clean/services/page_analyzer.py:2503-3035`.

3. `clean/services/analyzer_scan_service.py`
   - `_build_sequence_scan_row(page)`
   - Converts the `page_analyzer` result into a gap-scan row and preserves `bag_number`.
   - Evidence: `clean/services/analyzer_scan_service.py:57-101`.

4. `clean/services/gap_scan_service.py`
   - `_normalize_analysis_row(...)`
   - `_score_window_candidate(...)`
   - `_build_candidate(...)`
   - Scores exact bag-number matches strongly, flags conflicting bag numbers, and emits candidate rows with `bag_number`.
   - Evidence: `clean/services/gap_scan_service.py:66-260`.

5. `clean/routers/gap_review.py`
   - `/gap-review` and `/debug/gap-review`
   - Displays candidate `bag_number`, `ocr_raw`, panel signals, number box, confidence, and conflict labels.
   - Evidence: `clean/routers/gap_review.py:64-120`.

Related but not the visible bag-number recognizer:

- `clean/services/sequence_service.py` builds missing windows from confirmed truth rows. It does not OCR visible bag numbers.
- `clean/services/step_sequence_bag_service.py` detects step sequence resets, e.g. a reset from high step numbers to step 1. It helps find possible starts but does not read the visible grey bag number.

## V2 Code Path

Current V2 path is:

1. `instruction-v2/gap_window_scan.mjs`
   - Finds orange arrow components, dark number-like groups, central number-like groups, and scores structure.
   - It does not OCR/read the actual bag number.
   - Output has `score`, `confidence`, `reasons`, `signals`, and `number_like_groups`, but no `bag_number`.
   - Evidence: `instruction-v2/gap_window_scan.mjs:116-221`.

2. `instruction-v2/stage3b_bag_gap_review.py`
   - Builds pending gap windows and calls `gap_window_scan.mjs`.
   - Preserves human accepted corrections.
   - Has Bag 1 bootstrap evidence, but no visible bag-number OCR path.
   - Evidence: `instruction-v2/stage3b_bag_gap_review.py:134-166`, `169-353`.

3. `instruction-v2/stage3_bag_map.py`
   - Builds bag ranges from candidate clusters plus accepted human gap corrections.
   - Assigns bag numbers by sequence unless an accepted human correction has `observed_bag_number`.
   - It does not read visible bag numbers from page images.
   - Evidence: `instruction-v2/stage3_bag_map.py:47-70`, `100-197`.

## Runtime Availability Note

I attempted to run `clean.services.page_analyzer.analyze_page(...)` directly against the six V2 rendered page images as a read-only check. Both the system Python and bundled Python failed to import `cv2`, so concrete live V1 analyzer outputs were not available in this shell.

Therefore the V1 detected values below are either:

- from existing persisted V1 debug/cache where available, or
- marked unavailable when no persisted result exists.

Existing persisted V1 debug file `debug/70618/full_bag_scan.json` only covers an earlier partial scan and does not include pages 81, 100, 131, or 148.

## Page Comparison

| Page | Human Observation | V1 Detected Bag Number If Available | V2 Detected Bag Number If Available | Current V2 Assignment | V1 Method/Function | V2 Method/Function | Difference | V1 Logic To Port |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | Bag 1 material/start page | Not available from persisted V1 cache | None; V2 bootstrap records material/start evidence only | Bag 1, `bag_1_bootstrap_rule`, pages 6-38 | `page_analyzer.analyze_page`; gap path would expose `bag_number` if OCR succeeds | `stage3b_bag_gap_review._build_bag_1_bootstrap_review`; `gap_window_scan.mjs` structure score | V2 accepted page 6 by structure/step bootstrap, not by visible bag-number OCR | Port V1 `page_analyzer` number OCR fields into Stage 3b evidence |
| 7 | Printed steps 1/2 page, not material page | Not available from persisted V1 cache | None; V2 bootstrap inferred earliest step-1 page from sequence gap | Bag 1, range page | `step_sequence_bag_service.scan_step_bag_sequence` can detect step resets; `page_analyzer` is not the primary tool for step pages | `stage3b_bag_gap_review._scan_step_window` and `_infer_early_step_1_page` | V2 uses step inference here, which is appropriate, but does not OCR visible bag numbers | Keep step reset logic separate; do not use page 7 as bag-number evidence |
| 81 | Human accepted Bag 5 | Not available from persisted V1 cache in this audit | None in automated V2; `observed_bag_number=5` exists only from human accepted correction | Bag 5, `human_gap_review`, pages 81-99 | V1 route `/api/analyze-gap-page` would expose page analyzer `bag_number`; gap review displays it | V2 `gap_window_scan.mjs` found structure; human correction added `observed_bag_number=5` | V2 cannot independently confirm the printed 5; it relies on human correction | Port V1 visible bag-number OCR so accepted corrections can be cross-checked |
| 100 | Candidate cluster start currently after Bag 5 | Not available from persisted V1 cache | None; V2 has structure candidate only, no number value | Bag 6, `candidate_cluster`, pages 100-163 | `page_analyzer.analyze_page` plus `gap_scan_service._score_window_candidate` would distinguish exact/conflicting number | `03_bag_candidates.json` and `stage3_bag_map` infer sequence from cluster | V2 assigns Bag 6 by sequence, not by visible number | Port V1 exact/conflicting bag-number scoring before final bag map |
| 131 | Shows Bag 7 | Not available from persisted V1 cache, but this is exactly the V1 method's intended output field | None; V2 candidate has structure score 92.121 and no `bag_number` | Inside current Bag 6 range, pending gap `gap_after_bag_04_before_bag_05` top candidate | `page_analyzer.analyze_page` -> `bag_number`; `gap_scan_service._score_window_candidate` exact/conflict scoring | `gap_window_scan.mjs` sees orange arrow + number-like groups only | V2 sees a strong bag-start-like page but cannot tell it says 7, so it cannot resolve the pending gap correctly | Port V1 `page_analyzer` bag-number OCR and `gap_scan_service` scoring into Stage 3b |
| 148 | Shows Bag 8 | Not available from persisted V1 cache, but this is exactly the V1 method's intended output field | None; V2 candidate has structure score 88.603 and no `bag_number` | Inside current Bag 6 range, same pending gap window | `page_analyzer.analyze_page` -> `bag_number`; `gap_scan_service._score_window_candidate` exact/conflict scoring | `gap_window_scan.mjs` sees orange arrow + number-like groups only | V2 sees another strong bag-start-like page but cannot tell it says 8 | Port V1 visible bag-number OCR and conflict-aware ranking |

## Existing V2 Evidence For Sample Pages

From current manifests:

- Page 6:
  - `03b_bag_gap_review.json` Bag 1 bootstrap evidence: `score=97.5`, `confidence=0.775`.
  - No V2 visible `bag_number` field.

- Page 7:
  - No current `05_step_map.json` entry because Stage 5 has not been rerun from Bag 1 page 6.
  - Bag 1 bootstrap records `earliest_step_1_page=7` by `sequence_gap_inference`.

- Page 81:
  - `03b_bag_gap_review.json` accepted correction: `accepted_page=81`, `observed_bag_number=5`, `review_source=human`.
  - V2 structure score: `91.038`, `confidence=0.775`.
  - No automated visible `bag_number`.

- Page 100:
  - `03_bag_candidates.json` candidate score: `0.5006`.
  - Current `04_bag_map.json`: page 100 lies in Bag 6.
  - No automated visible `bag_number`.

- Page 131:
  - `03b_bag_gap_review.json` pending gap candidate score: `92.121`, `confidence=0.775`.
  - Human observation says visible Bag 7.
  - Current `04_bag_map.json`: page 131 lies inside Bag 6 range.
  - No automated visible `bag_number`.

- Page 148:
  - `03b_bag_gap_review.json` pending gap candidate score: `88.603`, `confidence=0.775`.
  - Human observation says visible Bag 8.
  - Current `04_bag_map.json`: page 148 lies inside Bag 6 range.
  - Current `05_step_map.json` has a step entry on page 148 with printed step number 8, which is a step signal, not a bag-number signal.
  - No automated visible `bag_number`.

## Main Finding

V1 had a number-aware bag start recognizer. V2 currently only has a structure-aware bag start recognizer.

That is why pages 131 and 148 can be strong V2 candidates but still cannot repair the bag sequence. V2 knows they look like bag-start pages, but it does not know they visibly say Bag 7 and Bag 8.

## V1 Logic To Port First

Port the visible bag-number recognition path, not the whole V1 workflow:

1. From `clean/services/page_analyzer.py`
   - `analyze_page`
   - panel detection
   - intro-region fallback
   - panel bag-number OCR fallback
   - no-shell panel rescue
   - full-page fallback
   - bag-start card refinement
   - output fields: `bag_number`, `ocr_raw`, `number_box`, `number_box_found`, `panel_found`, `panel_source`, `shell_found`, `grey_bag_found`, `bag_start_card_found`, `confidence`.

2. From `clean/services/analyzer_scan_service.py`
   - `_build_sequence_scan_row`
   - Preserve analyzer output into a simple row format.

3. From `clean/services/gap_scan_service.py`
   - `_normalize_analysis_row`
   - `_score_window_candidate`
   - `_build_candidate`
   - Especially exact match/conflicting number scoring:
     - exact match adds strong positive evidence
     - conflicting number subtracts evidence
     - number-only/no-structure cases are penalized
     - multi-step green boxes are excluded/penalized

4. From `clean/routers/gap_review.py`
   - Display candidate `bag_number`, `ocr_raw`, number box presence, confidence, and conflict label for human review.

## Recommendation

Next V2 parity step should be a Stage 3b bag-number recognizer/audit, still local-manifest-first:

- Read `02_page_index.json`.
- Read `03b_bag_gap_review.json`.
- For each gap candidate page, run a V1-style visible bag-number recognizer.
- Write a separate derived manifest such as `03c_bag_number_recognition.json`.
- Do not auto-accept.
- Feed `bag_number`, `ocr_raw`, `number_box`, `confidence`, and conflict/exact-match reasons into the gap review contact sheet.
- Human review remains final authority.

Do not use this to silently rewrite `04_bag_map.json`.
