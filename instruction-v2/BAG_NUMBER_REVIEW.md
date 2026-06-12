# Bag Number Review

Review-only output from detected visible bag-number evidence. This does not modify `04_bag_map.json` and does not auto-accept noisy pages.

## Inputs
- `instruction-v2/indexes/03_bag_candidates.json`
- `instruction-v2/indexes/03b_bag_gap_review.json`
- `instruction-v2/indexes/04_bag_map.json`
- `instruction-v2/indexes/02_page_index.json`

## Summary
- Pages with detected bag numbers: 37
- Confirmed candidate flags: 1
- Conflict flags: 31
- Noisy front-matter/overview flags: 5

## Required Human Review
- Page 81: detected Bag 5, current assignment 5 (81-99); page 81 Bag 5 confirmed candidate
- Page 131: detected Bag 7, current assignment 6 (100-163); page 131 Bag 7 conflicts with current gap expectation; detected Bag 7 conflicts with current bag_map assignment Bag 6; inside pending gap window gap_after_bag_04_before_bag_05
- Page 148: detected Bag 8, current assignment 6 (100-163); page 148 Bag 8 conflicts with current gap expectation; detected Bag 8 conflicts with current bag_map assignment Bag 6; inside pending gap window gap_after_bag_04_before_bag_05

## All Detected Bag Numbers
| Page | Detected Bag | Confidence | Current Assignment | Evidence Sources | Flags | Link |
| --- | ---: | ---: | --- | --- | --- | --- |
| 1 | 7 | 0.92 | none (none) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | noisy front-matter/overview page | [page 1](pages/70618_01/page_001.png) |
| 2 | 1 | 0.92 | none (none) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | noisy front-matter/overview page | [page 2](pages/70618_01/page_002.png) |
| 3 | 4 | 0.92 | none (none) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | noisy front-matter/overview page | [page 3](pages/70618_01/page_003.png) |
| 4 | 7 | 0.92 | none (none) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | noisy front-matter/overview page | [page 4](pages/70618_01/page_004.png) |
| 5 | 5 | 0.76 | none (none) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | noisy front-matter/overview page | [page 5](pages/70618_01/page_005.png) |
| 14 | 7 | 0.78 | 1 (6-38) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 7 conflicts with current bag_map assignment Bag 1 | [page 14](pages/70618_01/page_014.png) |
| 20 | 5 | 0.76 | 1 (6-38) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 1 | [page 20](pages/70618_01/page_020.png) |
| 22 | 5 | 0.76 | 1 (6-38) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 1 | [page 22](pages/70618_01/page_022.png) |
| 39 | 5 | 0.76 | 2 (39-52) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 2 | [page 39](pages/70618_01/page_039.png) |
| 44 | 5 | 0.76 | 2 (39-52) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 2 | [page 44](pages/70618_01/page_044.png) |
| 53 | 8 | 0.88 | 3 (53-80) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 8 conflicts with current bag_map assignment Bag 3 | [page 53](pages/70618_01/page_053.png) |
| 58 | 4 | 0.92 | 3 (53-80) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 4 conflicts with current bag_map assignment Bag 3 | [page 58](pages/70618_01/page_058.png) |
| 60 | 8 | 0.88 | 3 (53-80) | 03b_bag_gap_review.gap_windows.candidate_pages | detected Bag 8 conflicts with current bag_map assignment Bag 3 | [page 60](pages/70618_01/page_060.png) |
| 62 | 4 | 0.92 | 3 (53-80) | 03b_bag_gap_review.gap_windows.candidate_pages | detected Bag 4 conflicts with current bag_map assignment Bag 3 | [page 62](pages/70618_01/page_062.png) |
| 68 | 4 | 0.92 | 3 (53-80) | 03b_bag_gap_review.gap_windows.candidate_pages | detected Bag 4 conflicts with current bag_map assignment Bag 3 | [page 68](pages/70618_01/page_068.png) |
| 70 | 4 | 0.92 | 3 (53-80) | 03b_bag_gap_review.gap_windows.candidate_pages | detected Bag 4 conflicts with current bag_map assignment Bag 3 | [page 70](pages/70618_01/page_070.png) |
| 81 | 5 | 0.76 | 5 (81-99) | 03b_bag_gap_review.gap_windows.candidate_pages | page 81 Bag 5 confirmed candidate | [page 81](pages/70618_01/page_081.png) |
| 104 | 5 | 0.76 | 6 (100-163) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 6 | [page 104](pages/70618_01/page_104.png) |
| 106 | 5 | 0.76 | 6 (100-163) | 03b_bag_gap_review.gap_windows.candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 6; inside pending gap window gap_after_bag_04_before_bag_05 | [page 106](pages/70618_01/page_106.png) |
| 108 | 8 | 0.88 | 6 (100-163) | 03b_bag_gap_review.gap_windows.candidate_pages | detected Bag 8 conflicts with current bag_map assignment Bag 6; inside pending gap window gap_after_bag_04_before_bag_05 | [page 108](pages/70618_01/page_108.png) |
| 109 | 8 | 0.88 | 6 (100-163) | 03b_bag_gap_review.gap_windows.candidate_pages | detected Bag 8 conflicts with current bag_map assignment Bag 6; inside pending gap window gap_after_bag_04_before_bag_05 | [page 109](pages/70618_01/page_109.png) |
| 112 | 2 | 0.92 | 6 (100-163) | 03b_bag_gap_review.gap_windows.candidate_pages | detected Bag 2 conflicts with current bag_map assignment Bag 6; inside pending gap window gap_after_bag_04_before_bag_05 | [page 112](pages/70618_01/page_112.png) |
| 114 | 5 | 0.76 | 6 (100-163) | 03b_bag_gap_review.gap_windows.candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 6; inside pending gap window gap_after_bag_04_before_bag_05 | [page 114](pages/70618_01/page_114.png) |
| 115 | 5 | 0.76 | 6 (100-163) | 03b_bag_gap_review.gap_windows.candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 6; inside pending gap window gap_after_bag_04_before_bag_05 | [page 115](pages/70618_01/page_115.png) |
| 131 | 7 | 0.92 | 6 (100-163) | 03b_bag_gap_review.gap_windows.candidate_pages | page 131 Bag 7 conflicts with current gap expectation; detected Bag 7 conflicts with current bag_map assignment Bag 6; inside pending gap window gap_after_bag_04_before_bag_05 | [page 131](pages/70618_01/page_131.png) |
| 148 | 8 | 0.88 | 6 (100-163) | 03b_bag_gap_review.gap_windows.candidate_pages | page 148 Bag 8 conflicts with current gap expectation; detected Bag 8 conflicts with current bag_map assignment Bag 6; inside pending gap window gap_after_bag_04_before_bag_05 | [page 148](pages/70618_01/page_148.png) |
| 164 | 5 | 0.76 | 7 (164-193) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 7 | [page 164](pages/70618_01/page_164.png) |
| 213 | 5 | 0.76 | 9 (213-227) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 9 | [page 213](pages/70618_01/page_213.png) |
| 232 | 5 | 0.76 | 10 (228-264) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 10 | [page 232](pages/70618_01/page_232.png) |
| 237 | 5 | 0.76 | 10 (228-264) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 10 | [page 237](pages/70618_01/page_237.png) |
| 238 | 5 | 0.76 | 10 (228-264) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 10 | [page 238](pages/70618_01/page_238.png) |
| 265 | 5 | 0.76 | 11 (265-295) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 5 conflicts with current bag_map assignment Bag 11 | [page 265](pages/70618_01/page_265.png) |
| 266 | 4 | 0.92 | 11 (265-295) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 4 conflicts with current bag_map assignment Bag 11 | [page 266](pages/70618_01/page_266.png) |
| 300 | 7 | 0.78 | 12 (296-312) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 7 conflicts with current bag_map assignment Bag 12 | [page 300](pages/70618_01/page_300.png) |
| 302 | 2 | 0.92 | 12 (296-312) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 2 conflicts with current bag_map assignment Bag 12 | [page 302](pages/70618_01/page_302.png) |
| 303 | 11 | 0.92 | 12 (296-312) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 11 conflicts with current bag_map assignment Bag 12 | [page 303](pages/70618_01/page_303.png) |
| 305 | 1 | 0.92 | 12 (296-312) | 03_bag_candidates.json, 03b_bag_gap_review.detected_bag_candidate_pages | detected Bag 1 conflicts with current bag_map assignment Bag 12 | [page 305](pages/70618_01/page_305.png) |

## Notes
- Detected bag numbers are evidence, not source of truth.
- Front-matter/overview pages are expected to be noisy because they contain many visible bag numbers.
- Page 131 and page 148 are especially important because they show Bag 7 and Bag 8 while the current map still places them inside Bag 6.
- Human review should decide whether and how to correct the bag map.
