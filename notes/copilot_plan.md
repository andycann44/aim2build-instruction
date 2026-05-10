## 2026-05-10

### Findings

Page 7 evidence:
- C1: full_callout_box x=822 y=0 w=277 h=194 / inner_detected_region x=822 y=0 w=154 h=120
- C2: full_callout_box x=941 y=593 w=257 h=101 / inner_detected_region x=941 y=593 w=171 h=73

Changes already applied:
- Removed near-white pale_blue_ratio branch (V>=198, S<=45)
- Raised expansion Canny from 28/96 to 40/120
- Added candidate_score to tile figcaption
- Applied expansion-ratio guard in _build_material_crop_candidates: if full_w > inner_w * 1.6 OR full_h > inner_h * 1.6, display uses inner_detected_region instead (full_callout_box preserved in metadata)

Result: visually almost unchanged — large pale-blue strips still selected as winners.

### Diagnosis

Root cause: scoring rewards large pale-blue area and penalises nothing for emptiness.

Base score formula in _validate_local_candidate (lines ~930–937):
```
pale_blue_ratio * 2.0 + bright_ratio * 1.5 + max(0, mean_value - border_mean) / 32.0 - edge_density
```

An empty pale-blue strip scores near-perfectly: pale_blue_ratio≈0.90, bright_ratio≈0.90, edge_density≈0.01 → score≈3.1

A tight part+qty callout is more textured: pale_blue_ratio≈0.55, bright_ratio≈0.60, edge_density≈0.10 → score≈2.0

The expansion bonuses compound this:
- Contour path: `score += min(1.2, box_area / seed_area * 2)` — bigger box = bigger bonus
- Merge path: `score += min(1.5, merged_area / seed_area * 2)` — same

No term penalises a box for being large relative to the search zone, and no term rewards content density.

The `area_ratio` variable (already computed as `box_area / search_area`, bounded 0.02–0.35 by guards) is the correct proxy for size relative to scene:
- Tight callout: area_ratio ≈ 0.05–0.12
- Large empty strip: area_ratio ≈ 0.25–0.35

The `edge_density` subtraction already penalises noisy boxes but accidentally rewards zero edges (empty strips). A low-edge-density penalty corrects this.

### Proposed patch

Single expression change in `_validate_local_candidate`, return dict score field only. No structural changes. No detector, no expansion, no merge, no cluster, no services.

File: clean/routers/debug.py
Location: `_validate_local_candidate` return dict, score expression (~line 932)

Before:
```python
            return {
                "local_box": [x, y, w, h],
                "abs_box": abs_box,
                "score": (
                    pale_blue_ratio * 2.0
                    + bright_ratio * 1.5
                    + max(0.0, mean_value - border_mean) / 32.0
                    - edge_density
                ),
            }
```

After:
```python
            return {
                "local_box": [x, y, w, h],
                "abs_box": abs_box,
                "score": (
                    pale_blue_ratio * 2.0
                    + bright_ratio * 1.5
                    + max(0.0, mean_value - border_mean) / 32.0
                    - edge_density
                    - max(0.0, area_ratio - 0.12) * 4.0
                    - max(0.0, 0.10 - edge_density) * 3.0
                ),
            }
```

Term 1: `- max(0.0, area_ratio - 0.12) * 4.0`
- No penalty for boxes up to 12% of search zone (tight callout)
- Strip at area_ratio=0.30: penalty = (0.30 − 0.12) × 4 = 0.72
- Strip at area_ratio=0.35: penalty = (0.35 − 0.12) × 4 = 0.92

Term 2: `- max(0.0, 0.10 - edge_density) * 3.0`
- No penalty when edge_density >= 0.10 (content present)
- Empty strip at edge_density=0.01: penalty = (0.10 − 0.01) × 3 = 0.27
- Corrects existing formula which accidentally rewarded zero edges

Combined effect on page 7 candidates:
- Empty strip: 3.1 − 0.72 − 0.27 = ~2.1 (or lower for larger strips)
- Tight callout: 2.0 − 0 − 0 = 2.0
- Winner now depends on exact measurements; at area_ratio=0.35 strip drops to ~1.38

If strip still wins, increase area_ratio coefficient from 4.0 to 6.0.
If valid callouts disappear, reduce from 4.0 to 2.5.
Breakeven threshold 0.12 can be raised to 0.15 if tight callouts are being caught.

### Verification

After applying:
1. Reload /debug/step-part-lab?step=7
2. Check score: on C1 and C2 tiles
   - Tight callout should score ~1.8–2.2
   - Empty strip should score ~0.9–1.5
3. If C1/C2 now show tight boxes: patch is working
4. If valid callout disappears entirely: reduce area_ratio coefficient 4.0 → 2.5
5. If strips still win: check full_callout_box area vs page size to confirm area_ratio; increase coefficient 4.0 → 6.0 or lower breakeven from 0.12 → 0.10

---

## 2026-05-10 - Set 70618 page 23 step 27 callout detection notes

### Goal

Find a reliable deterministic computer-vision method to detect step-part callout boxes on LEGO instruction pages, with step 27 on page 23 of set 70618 as the concrete reference case.

### Page 23 evidence

Detected step anchor on page 23:
- step 27 box from existing detector: `x=116 y=207 w=53 h=36`

Existing padded search region that works reasonably:
- from current `_build_instruction_callout_crops` / lab geometry:
- `search_box = [74, 27, 593, 207]`
- size: `519 x 180`

Current edge detector result for step 27:
- `_detect_callout_rect_by_edges(...)` returns:
- `rect = [109, 27, 319, 169]`

Hough border evidence in the step-27 search area:
- top horizontal border fragments:
  - `x=125..413 y=7`
  - `x=117..420 y=10`
- bottom horizontal border fragments:
  - `x=118..419 y=167`
  - `x=123..414 y=169`
- left vertical border fragments:
  - `x=112 y=20..156`
  - `x=115 y=12..164`
- right vertical border fragments:
  - `x=423 y=13..164`
  - `x=425 y=18..159`

Separate unrelated right-side image evidence:
- second vertical pair:
  - `x=537 y=20..172`
  - `x=540 y=12..180`
- separate lower horizontal pair:
  - `x=543..649 y=183`
  - `x=551..649 y=186`

Background-difference connected components inside the same search region:
- local background estimate from search-box border:
  - `BGR ≈ [252, 236, 211]`
- two large components only:
  - main component: local `[40, 0, 313, 163]` => page `[114, 27, 313, 163]`
  - right-side component: local `[465, 0, 54, 180]` => page `[539, 27, 54, 180]`
- sustained low-density vertical gap between them:
  - local x gap: `352..463`

Interpretation:
- the true callout is the left large component directly above/right of the step number
- the build/model image on the right is a separate component with its own border support
- there is a strong whitespace valley between them, so the right edge can be stopped deterministically without using blue alone

### Proposed deterministic method

#### 1. Locate the step-number anchor

Primary anchor:
- keep using the existing `step_detector_service.detect_steps(...)` + `_contact_sheet_step_boxes_from_detected(...)`

Fallback anchor:
- if no visual step box exists, use the existing OCR-style large-number fallback approach already present in `clean/routers/debug.py`
- restrict to 1–2 digit numbers with a plausible size box

Why this is reliable:
- for page 23, step 27 is already correctly anchored at the bottom-left of the target callout
- the step number anchor gives a stable geometric reference even when page colours vary

#### 2. Define the search region above/right of the anchor

Use the existing padded region logic already used in the lab:
- `pad_left = max(18, int(step_w * 0.8))`
- `pad_above = max(75, int(step_h * 5.0))`
- `pad_right = max(220, int(step_w * 8.0))`
- `x1 = step_x - pad_left`
- `y1 = step_y - pad_above`
- `x2 = step_x + step_w + pad_right`
- `y2 = step_y`

Why this works on page 23:
- it contains the whole callout `[109,27,319,169]`
- it excludes most irrelevant content below the step number
- it still allows some rightward search so the box can be completed if the callout extends to the right

#### 3. Identify the true qty+part callout group

Recommended primary detector:
- do not rely on absolute blue hue
- instead estimate local background from the border bands of the search ROI
- build an object mask from colour-distance to that local background:
  - `diff = norm(pixel - local_bg)`
  - threshold around `> 30` worked on page 23
- morph:
  - open small noise
  - close to unify the callout content/body

Then find connected components / contours and score candidates.

Candidate filters:
- candidate must be above the step anchor:
  - `candidate_bottom <= step_y - small_margin`
- width > height
- aspect roughly callout-like:
  - around `1.2 .. 4.8`
- candidate area should be a moderate fraction of the search area
- candidate center should stay near the step anchor horizontally
- reject obviously tiny or huge components

Candidate scoring:
- prefer the component whose bottom is closest above the step number
- prefer the component with the strongest dark border-line support
- prefer the component that contains multiple internal foreground blobs consistent with parts/qty labels
- use qty OCR tokens only as a bonus / confirmation, not as the only seed

Why this is robust:
- on page 23 the background-difference mask already separates the true callout from the right-side build image with no blue-specific rule required
- this generalises better to non-blue instruction styles because it is based on contrast to local page background, not fixed hue alone

#### 4. Stop the right edge before the separate build/model image

Use a two-stage right-edge stop:

Stage A: border support
- search for paired dark horizontal/vertical border fragments around the candidate
- on page 23 the true callout has a consistent box:
  - left border near `x=112..115`
  - right border near `x=423..425`
  - top border near `y=7..10`
  - bottom border near `y=167..169`
- this gives a box ending around page x `423..425`, well before the unrelated right-side image

Stage B: whitespace valley / separate component stop
- after the candidate body is found, inspect vertical foreground-density profile
- if there is a sustained low-density gap after the candidate and before another component, stop at the left side of that gap
- on page 23 there is a clear valley from local x `352..463`, between:
  - true callout component `[40,0,313,163]`
  - right-side component `[465,0,54,180]`

Recommended rule:
- once a candidate is chosen, do not let the right edge expand across a blank vertical run wider than about `12-20 px`
- if a second disconnected component begins after that gap, reject merging it into the callout

This is the key rule that prevents cropping the separate build/model image on the right.

### Debug overlays and metrics to draw

For each candidate page / step:

1. Step anchor overlay
- draw the detected step-number box in green
- label with `step_number`

2. Search region overlay
- draw the padded search ROI in cyan
- show absolute coords

3. Background estimate
- log sampled local background `BGR` and/or `HSV`

4. Object mask diagnostics
- show / log connected components from the background-difference mask
- for each component:
  - absolute box
  - area
  - aspect ratio
  - bottom y
  - center x distance from step anchor

5. Border-line diagnostics
- overlay accepted horizontal and vertical line groups
- show:
  - top line candidates
  - bottom line candidates
  - left/right verticals

6. Gap diagnostics
- show vertical foreground-density profile summary
- mark any long low-density run used as a stop boundary

7. Final decision diagnostics
- selected candidate box
- rejected component boxes with reasons such as:
  - `below_step`
  - `too_tall`
  - `second_component_after_gap`
  - `no_border_support`
  - `too_far_right`

8. Optional qty token debug
- if qty token boxes exist, draw them
- but do not make detection depend on them for this page/style

### Smallest safe patch location in `clean/routers/instruction_debug.py`

Smallest safe place to patch:
- `clean/routers/instruction_debug.py`
- function: `_detect_callout_rect_by_edges(...)`

Why here:
- `_build_instruction_callout_crops(...)` already calls this function per step box
- the route, crop-save flow, AI ranking, schema, and UI can remain untouched
- the function already owns:
  - step-anchor-relative search box input
  - local ROI extraction
  - final returned `[x, y, w, h]`

Recommended patch shape:
- keep the current Hough-line logic as fallback
- before the existing Hough grouping, add a new primary stage:
  1. estimate local background from ROI border
  2. build foreground/background-difference mask
  3. get connected components
  4. score components relative to the step anchor
  5. validate with border-line support and/or whitespace-gap stop
  6. return the chosen box if valid
- if that stage fails, continue into the current Hough-line path unchanged

This is the safest minimal patch because it changes only the local callout detector, not the surrounding pipeline.

### Practical recommendation

For step 27 specifically, the best deterministic detector is:
- step-number anchor first
- padded search region above/right of the anchor
- local-background-difference component detection as the primary grouping method
- dark border-line support to confirm the true box
- whitespace-gap stop to avoid merging the separate right-side build image

This should be more reliable than:
- pure colour-fill detection
- pure OCR qty token seeding
- pure Hough-only border detection

because page 23 already shows:
- good anchor
- separable connected components
- strong border evidence
- a clear gap before the unrelated right-side image

---

## 2026-05-10 - Set 70618 bag 3 page 39 missing steps 42 and 43

### What instruction-buildability uses

`/debug/instruction-buildability` gets step boxes here:
- `clean/routers/instruction_debug.py`
- `_build_instruction_callout_crops(...)`
- lines around:
  - `detected = step_detector_service.detect_steps(str(set_num), int(page))`
  - `step_boxes = _contact_sheet_step_boxes_from_detected(detected)`

`_contact_sheet_step_boxes_from_detected(...)` lives in:
- `clean/routers/debug.py`

Important behavior:
- it returns only `classified_step_boxes` whose `step_group == "main_steps"`
- so anything classified as `sub_steps` or filtered out before classification never reaches instruction-buildability

### Does the detector scan the full page?

Partly, but not consistently.

Full-page scan:
- `_extract_numeric_tokens(...)` in `clean/services/step_detector_service.py`
- runs `pytesseract.image_to_data(img, config="--psm 6")` on the full page
- so OCR numeric tokens are collected from the whole page

Restricted scans / restricted acceptance:
- `_visual_candidates_from_image(...)`
  - only scans `roi = img[0:top_limit, 0:left_limit]`
  - `left_limit = int(page_width * 0.22)`
  - so visual number detection is left-side only
- `_is_valid_step_region(...)`
  - rejects any candidate with `x >= page_width * 0.45`
  - so right-half OCR hits are discarded even if OCR found them
- `_classify_main_and_sub_steps(...)`
  - builds `left_band` from candidates near the leftmost x only
  - sub-step collection also ignores candidates with `x > page_width * 0.35`

Net result:
- the detector is effectively biased to left-side step numbers
- full-page OCR sees some right-side numbers, but the validity/classification logic throws them away

### What actually happens on page 39

Observed detection for page 39:
- `detect_steps('70618', 39)` returns only one main step:
  - `41`

Observed page sequence around it:
- page 38 main steps: `[40]`
- page 39 main steps: `[41]`
- page 40 main steps: `[44, 45]`

So the visible sequence jumps:
- `41 -> 44`
- missing on page 39: `42`, `43`

### Exact evidence

From `detect_steps('70618', 39)`:

`numeric_tokens`:
- `41` at `x=114 y=644 w=42 h=36`
- `43` at `x=913 y=634 w=56 h=37`

`visual_candidates`:
- only `41`

`step_candidates`:
- only `41`

`classified_step_boxes`:
- only `41` as `main_steps`

So:
- step `43` is actually found by full-page OCR
- but it is dropped before `step_candidates`

The reason is explicit in `_is_valid_step_region(...)`:
```python
    if x >= page_width * 0.45:
        return False
```

For page 39:
- page width is `1565`
- `page_width * 0.45 = 704.25`
- step `43` is at `x=913`
- therefore it is rejected even though OCR found it with confidence `96`

### Why step 42 is missed

Step `42` is not present in the full-page `numeric_tokens` output.

However, it is OCR-readable in a local crop around the visible label:
- crop around the 42 area returns:
  - text: `42`
  - confidence: about `95-96`

This means:
- the full-page OCR pass misses `42`
- but the number itself is visually clear and recoverable if we run a focused local search

Likely reasons the full-page pass misses it:
- page-level `--psm 6` OCR is sensitive to large structured layouts
- the right-half step area contains nearby callout boxes and part images
- `42` is not rescued by visual detection because `_visual_candidates_from_image(...)` only scans the left-side ROI

### Root cause summary

There are two separate failure modes:

1. Step `43` is detected by OCR but rejected by left-side validity rules
2. Step `42` is missed by the coarse full-page OCR pass and is never recovered because:
   - visual scanning is left-side only
   - there is no sequence-gap recovery pass

### Smallest safe place to add a sequence-gap recovery pass

Best place for a tiny recovery pass:
- `clean/services/step_detector_service.py`
- inside `detect_steps(...)`
- after:
  - `step_candidates = _dedupe_candidates(candidates)`
- before:
  - `_classify_main_and_sub_steps(...)`

Why this is the smallest safe place:
- it keeps all downstream consumers unchanged:
  - instruction-buildability
  - callout crop lab
  - workflow
  - step sequence services
- it can repair the candidate list once, centrally
- classification can then treat recovered steps normally

### Proposed tiny recovery patch

Add a focused recovery pass inside `detect_steps(...)`:

1. Build initial `step_candidates` as today
2. Look at detected numeric values in ascending order
3. If there is a gap where values jump by 3 or more:
   - for example `41 -> 44`
   - missing values are `42`, `43`
4. For each missing value:
   - search the current page for that exact number in a targeted region, not full-page OCR only

Recommended targeted search rule:
- use the page area around existing and neighboring detected steps
- for page 39 specifically:
  - search to the right of step `41`
  - within the upper/middle band where step numbers normally sit

Minimal deterministic search box heuristic:
- if we have a lower detected step `N` and an upper detected step `N+3` on the next page or in nearby sequence context:
  - search current page in the band:
    - `x` from `last_detected_step_x - 60` to `page_width - margin`
    - `y` from `last_detected_step_y - 220` to `last_detected_step_y + 80`
- for each candidate crop window:
  - run OCR restricted to digits
  - accept only exact missing values

Even smaller/safer variant:
- do not use next-page context inside `detect_steps(...)`
- instead recover any OCR numeric token that was filtered only by the left-side rule if:
  - its value is within a small forward range of the current detected page sequence
- plus add a local right-half OCR probe only when detected main steps are sparse

But for the requested explicit gap rule, the best consumer-level place is:
- `clean/routers/instruction_debug.py`
- in `_build_instruction_callout_crops(...)`
- after collecting `step_boxes` for all pages in the bag/page range
- compare page-to-page sequence:
  - page 38 = 40
  - page 39 = 41
  - page 40 = 44,45
- then run a current-page recovery scan on page 39 for 42 and 43

### Best tiny-patch recommendation

If the goal is to fix instruction-buildability only with minimal blast radius:
- add the gap-recovery pass in:
  - `clean/routers/instruction_debug.py`
  - inside `_build_instruction_callout_crops(...)`

Reason:
- the user asked specifically about instruction-buildability
- this avoids changing global step-detection behavior for all routes
- it can use cross-page context naturally because `_build_instruction_callout_crops(...)` iterates the rendered page range

Recovery flow there:
1. collect normal `step_boxes` per page as today
2. track detected step values across consecutive pages
3. if sequence jumps from `N` to `N+3`:
   - run a focused OCR recovery search on the current page for `N+1` and `N+2`
4. append recovered boxes to that page’s `step_boxes`
5. continue unchanged

### Best long-term fix

The more correct root fix is still in `step_detector_service.py`:
- stop assuming main steps live only on the left side
- remove or relax:
  - `x >= page_width * 0.45` hard reject
  - left-only visual ROI
  - left-band-only main-step classification

But that is broader.

For a tiny safe patch, use a sequence-gap recovery pass first.
