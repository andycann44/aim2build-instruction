# Bag Location Verification

Audit-only verification of current V2 bag locations. No bag map, step map, segmentation, or matching fixes are applied by this document.

## Inputs
- `instruction-v2/indexes/03_bag_candidates.json`
- `instruction-v2/indexes/03b_bag_gap_review.json`
- `instruction-v2/indexes/04_bag_map.json`
- `instruction-v2/indexes/05_step_map.json`
- `instruction-v2/indexes/02_page_index.json`

## Summary
- Bags in `04_bag_map.json`: 11
- Definite: 0
- NOT DEFINITE: 11
- Pending gap windows: 1
- Accepted gap windows: 1
- Bag 1 bootstrap: status `accepted`, proposed page `6`, earliest step-1 page `7`

## Sequence Flags
- NOT DEFINITE: bag sequence skips 4 before 5
- NOT DEFINITE: missing bag number(s): 4

## Bag Table
| Bag | Pages | Source | Confidence | Evidence Pages | First Step Page | First Printed Step | Start | First Step | Status | Verdict |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6-38 | `bag_1_bootstrap_rule` | 0.775 | 6, 12, 14, 20, 22 | 12 | 11 | [page 6](pages/70618_01/page_006.png) | [page 12](pages/70618_01/page_012.png) | inferred | NOT DEFINITE: bag sequence skips 4 before 5; missing bag number(s): 4; step map first detects this bag at page 12; downstream step map still needs rerun from bootstrap start |
| 2 | 39-52 | `candidate_cluster` | 0.891 | 39, 44, 46 | 39 | 44 | [page 39](pages/70618_01/page_039.png) | [page 39](pages/70618_01/page_039.png) | inferred | NOT DEFINITE: bag sequence skips 4 before 5; missing bag number(s): 4 |
| 3 | 53-80 | `candidate_cluster` | 0.781 | 53, 58, 59 | 53 | 68 | [page 53](pages/70618_01/page_053.png) | [page 53](pages/70618_01/page_053.png) | inferred | NOT DEFINITE: bag sequence skips 4 before 5; missing bag number(s): 4 |
| 5 | 81-99 | `human_gap_review` | 0.990 | 81 | 82 | 133 | [page 81](pages/70618_01/page_081.png) | [page 82](pages/70618_01/page_082.png) | confirmed | NOT DEFINITE: pending gap review gap_after_bag_04_before_bag_05 adjacent to bag 5; bag sequence skips 4 before 5; missing bag number(s): 4 |
| 6 | 100-163 | `candidate_cluster` | 0.606 | 100, 104 | 100 | 159 | [page 100](pages/70618_01/page_100.png) | [page 100](pages/70618_01/page_100.png) | inferred | NOT DEFINITE: low confidence 0.606; pending gap review gap_after_bag_04_before_bag_05 overlaps pages 105-163; bag sequence skips 4 before 5; missing bag number(s): 4 |
| 7 | 164-193 | `candidate_cluster` | 0.733 | 164, 169 | 164 | 29 | [page 164](pages/70618_01/page_164.png) | [page 164](pages/70618_01/page_164.png) | inferred | NOT DEFINITE: bag sequence skips 4 before 5; missing bag number(s): 4 |
| 8 | 194-212 | `candidate_cluster` | 0.602 | 194 | 194 | 34 | [page 194](pages/70618_01/page_194.png) | [page 194](pages/70618_01/page_194.png) | inferred | NOT DEFINITE: low confidence 0.602; bag sequence skips 4 before 5; missing bag number(s): 4 |
| 9 | 213-227 | `candidate_cluster` | 0.751 | 213 | 213 | 36 | [page 213](pages/70618_01/page_213.png) | [page 213](pages/70618_01/page_213.png) | inferred | NOT DEFINITE: bag sequence skips 4 before 5; missing bag number(s): 4 |
| 10 | 228-264 | `candidate_cluster` | 0.761 | 228, 232, 237, 238 | 228 | 399 | [page 228](pages/70618_01/page_228.png) | [page 228](pages/70618_01/page_228.png) | inferred | NOT DEFINITE: bag sequence skips 4 before 5; missing bag number(s): 4 |
| 11 | 265-295 | `candidate_cluster` | 0.807 | 265, 266 | 265 | 7 | [page 265](pages/70618_01/page_265.png) | [page 265](pages/70618_01/page_265.png) | inferred | NOT DEFINITE: bag sequence skips 4 before 5; missing bag number(s): 4 |
| 12 | 296-312 | `candidate_cluster` | 0.815 | 296, 300, 302, 303, 305, 306 | 296 | 54 | [page 296](pages/70618_01/page_296.png) | [page 296](pages/70618_01/page_296.png) | inferred | NOT DEFINITE: bag sequence skips 4 before 5; missing bag number(s): 4 |

## Human Review Needed
- Bag 1, pages 6-38: bag sequence skips 4 before 5; missing bag number(s): 4; step map first detects this bag at page 12; downstream step map still needs rerun from bootstrap start
- Bag 2, pages 39-52: bag sequence skips 4 before 5; missing bag number(s): 4
- Bag 3, pages 53-80: bag sequence skips 4 before 5; missing bag number(s): 4
- Bag 5, pages 81-99: pending gap review gap_after_bag_04_before_bag_05 adjacent to bag 5; bag sequence skips 4 before 5; missing bag number(s): 4
- Bag 6, pages 100-163: low confidence 0.606; pending gap review gap_after_bag_04_before_bag_05 overlaps pages 105-163; bag sequence skips 4 before 5; missing bag number(s): 4
- Bag 7, pages 164-193: bag sequence skips 4 before 5; missing bag number(s): 4
- Bag 8, pages 194-212: low confidence 0.602; bag sequence skips 4 before 5; missing bag number(s): 4
- Bag 9, pages 213-227: bag sequence skips 4 before 5; missing bag number(s): 4
- Bag 10, pages 228-264: bag sequence skips 4 before 5; missing bag number(s): 4
- Bag 11, pages 265-295: bag sequence skips 4 before 5; missing bag number(s): 4
- Bag 12, pages 296-312: bag sequence skips 4 before 5; missing bag number(s): 4

## Pending Gap Windows
- `gap_after_bag_04_before_bag_05` pages 105-163; top candidate page 131; status `pending`.

