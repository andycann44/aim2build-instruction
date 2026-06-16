# Step Map Multi-Digit Parity

Status: IN PROGRESS  
Date: 2026-06-16  
Scope: Step Map only (Stage 4). No crop / segmentation / matching work.

## Targets

| Step | Page | Bag | Status | Notes |
|---:|---:|---:|---|---|
| 79 | 59 | 4 | **PASS** | 2-digit merge; baseline-aware grouping |
| 239 | 146 | 7 | **PASS** | 3-component merge; OCR `239` |
| 278 | 163 | 8 | **PASS** | `mergeNeighboringGroups()` + primary OCR |
| 368 | 214 | 12 | **PASS** | 3-component merge; OCR `368` |
| 369 | 214 | 12 | **PASS** | 3-component merge; OCR `369` |
| 426 | 238 | 13 | **PASS** | sequence correction `476`→`426` |
| 459 | 254 | 13 | **PASS** | 3-component merge; OCR `459` |
| 483 | 277 | 15 | **PASS** | overlap match + primary OCR fallback |

**Result: 8 / 8 PASS** (2026-06-16 stage4 rerun)

## Root cause: V2 `> 200` OCR cap (fixed)

`step_map_scan.mjs` rejected all OCR reads `> 200` with `implausible_step_value_exceeds_200`.

**RULEBOOK / V1 contract:** `0 < step_number < 1000` (`instruction_debug.py::_filter_invalid_step_anchor_boxes`).

**Effect:** Boxes were detected (yellow overlays showed `step ?`) but never entered `steps[]` because OCR values 239–483 were rejected.

**Fix:** Align cap to V1: reject only `>= 1000`.

**Result after fix + stage4 rerun:** **8/8 targets pass.**

## Remaining failures

None for the eight known multi-digit targets.

## Important rule (from trusted baseline)

A valid build step does not require a parts crop. Step detection and crop detection are separate systems.

Page 163 step 278 has no callout crop — it is still a valid global step anchor.

## Stage 4 changes (this session)

| Change | File | Rationale |
|---|---|---|
| OCR cap `200` → `1000` | `step_map_scan.mjs` | V1 parity |
| `mergeNeighboringGroups()` | `step_map_scan.mjs` | 3-digit trailing fragment merge |
| `boxesVisuallyMatch()` | `step_map_scan.mjs` | Primary/full-page read alignment |
| Direct primary OCR fallback | `step_map_scan.mjs` | When full-page match fails |
| `correctSamePageSequentialOcr()` | `step_map_scan.mjs` | 476→426 style single-digit OCR fix |

## Verification command

```bash
cd instruction-v2 && python3 stage4_step_map.py
python3 - <<'PY'
import json
from pathlib import Path
TARGETS = {79:59, 239:146, 278:163, 368:214, 369:214, 426:238, 459:254, 483:277}
m = json.loads(Path("indexes/05_step_map.json").read_text())
for step, page in TARGETS.items():
    ok = any(s.get("step_number")==step and s["page"]==page and not s.get("rejection_reason") for s in m["steps"])
    print(("PASS" if ok else "FAIL"), step, page)
PY
```

## Out of scope

- Crop map parity
- Segmentation parity
- Bag 6 review
- Matching / Azure / R2 automation
