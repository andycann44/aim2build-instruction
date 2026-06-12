# Step 8 Stage 6 Failure Audit

Scope: audit only. No code edits, no manifest regeneration, no commit.

## Target

Bag 1 page 10 Step 8 visibly contains a valid callout:

- boxed region present
- part image visible
- qty label is `4x`

For Aim2Build Instruction V2 this must generate a callout crop.

## Inputs Checked

- `instruction-v2/pages/70618_01/page_010.png`
- `instruction-v2/indexes/05_step_map.json`
- `instruction-v2/indexes/06_callout_crop_box_map.json`
- `instruction-v2/callout_crop_box_scan.mjs`

## Stage 5 Step 8 Entry

Step 8 is present and not rejected:

```json
{
  "bag": 1,
  "page": 10,
  "step_index": 2,
  "step_number": 8,
  "rejection_reason": null,
  "step_box": {
    "x": 30,
    "y": 795,
    "w": 25,
    "h": 37
  },
  "confidence": 1
}
```

So this is not a Stage 5 disappearance.

## Stage 6 Search Region

Stage 6 uses `buildStepRegion()` from `callout_crop_box_scan.mjs`.

For Step 8:

```text
page size: 1565 x 1191
step_box: x=30 y=795 w=25 h=37
region: x1=10 y1=610 x2=275 y2=795
roi: width=265 height=185
stepY: 795
```

The visible Step 8 callout box extends farther right than this ROI, and the ROI starts inside/near the callout panel rather than cleanly around it.

## 1. Was the boxed region detected?

Not as a clean callout box.

The detector produced one connected component that filled the entire search ROI:

```json
{
  "x": 0,
  "y": 0,
  "w": 265,
  "h": 185,
  "page_x": 10,
  "page_y": 610,
  "pageBottom": 795,
  "aspect": 1.432,
  "areaRatio": 1,
  "yellowRatio": 0,
  "reason": "pageBottom > stepY - 5"
}
```

So the detector did see foreground/change pixels in the Step 8 region, but it did not isolate the actual boxed callout. It collapsed the ROI into one full-region component.

## 2. Was it rejected?

Yes.

The only candidate component was rejected.

No accepted candidates remained, so `detectCalloutRectByEdges()` returned `null`, and Stage 6 emitted no crop for Step 8.

## 3. Which filter rejected it?

This Stage 6 rule rejected it:

```js
if (pageBottom > stepY - 5) continue;
```

For Step 8:

```text
pageBottom = 795
stepY - 5 = 790
795 > 790 => rejected
```

This is the exact filter that prevented crop creation.

## 4. Is the box below minimum size?

No.

The rejected component was:

```text
w = 265
h = 185
```

That is above the Stage 6 minimum size rule:

```js
if (box.w < 70 || box.h < 28) continue;
```

It was rejected before size/aspect/area/yellow checks were reached.

## 5. Is it being mistaken for part of the main build image?

Not exactly.

It is being treated as one full-ROI foreground component because the Stage 6 search region is poorly shaped for this layout:

- `x2=275` cuts off the right side of the actual callout panel.
- `y1=610` starts inside/near the panel rather than above its full boundary.
- The component touches the ROI bottom at `y2=795`.

That makes the detector see the whole ROI as a foreground blob instead of isolating the rounded rectangular callout panel.

The root issue is not that Step 8 is confused with the large model image. The root issue is that the current Step 8 search window and bottom-boundary rule are incompatible with a valid callout panel that sits close to the printed step number.

## 6. What exact Stage 6 rule prevented crop creation?

The crop was prevented by this rule in `callout_crop_box_scan.mjs::detectCalloutRectByEdges()`:

```js
if (pageBottom > stepY - 5) continue;
```

Because the only detected component had:

```text
pageBottom = 795
stepY = 795
```

It failed:

```text
795 > 790
```

No other candidate survived, so Stage 6 returned no callout crop for Step 8.

## Conclusion

Step 8 is a real missed crop.

Failure point:

```text
Stage 6 fresh callout crop detection
```

Exact failing rule:

```text
reject candidate when pageBottom > stepY - 5
```

Immediate next fix should be a Stage 6 V1 parity repair/fallback for callout panels whose valid box is close to, or extends slightly below/around, the printed step number anchor.

