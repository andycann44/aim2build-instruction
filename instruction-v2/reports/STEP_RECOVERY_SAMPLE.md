# Step Recovery Visual Sample

Read-only visual verification that recovered steps with `step_number > 200` are genuine printed global step anchors.

## Method

- **Current manifest:** `indexes/05_step_map.json` (commit `39d84c1`)
- **Parent manifest:** commit `0e85cc8`
- **Recovered step:** accepted in current, absent in parent (exact page+step+box key), `step_number > 200`
- **Sample:** `random.seed(70618)`, 5 per bag from bags **7, 9, 12, 15**
- **Page images:** `pages/70618_01/page_XXX.png`
- **Badge crops:** `reports/step_recovery_samples/*_badge.png` (step_box + 12 px pad)

## Pass criteria

1. Printed number on page matches detected `step_number`
2. Standard LEGO global-step layout (large black digits, left/column gutter)
3. Not a page number (footer), substep panel digit, or qty-box local number

## Summary

| Bag | Sampled | Visual PASS |
|---:|---:|---:|
| 7 | 5 | 5 |
| 9 | 5 | 5 |
| 12 | 5 | 5 |
| 15 | 5 | 5 |
| **Total** | **20** | **20** |

**Conclusion:** All 20 recovered samples are genuine printed global step numbers.

---

## Bag 7 — Page 134 — Step 213

| Field | Value |
|---|---|
| Detected step | **213** |
| OCR raw | `213` |
| Components | 3 |
| Step box | x=770 y=142 w=73 h=36 |
| Page image | `pages/70618_01/page_134.png` |
| Badge crop | `reports/step_recovery_samples/bag07_p134_s213_badge.png` |
| Context crop | `reports/step_recovery_samples/bag07_p134_s213_context.png` |

**Visual verification:** **PASS**

Printed **213** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 7 — Page 134 — Step 214

| Field | Value |
|---|---|
| Detected step | **214** |
| OCR raw | `214` |
| Components | 3 |
| Step box | x=770 y=723 w=73 h=36 |
| Page image | `pages/70618_01/page_134.png` |
| Badge crop | `reports/step_recovery_samples/bag07_p134_s214_badge.png` |
| Context crop | `reports/step_recovery_samples/bag07_p134_s214_context.png` |

**Visual verification:** **PASS**

Printed **214** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 7 — Page 140 — Step 224

| Field | Value |
|---|---|
| Detected step | **224** |
| OCR raw | `224` |
| Components | 3 |
| Step box | x=30 y=142 w=83 h=36 |
| Page image | `pages/70618_01/page_140.png` |
| Badge crop | `reports/step_recovery_samples/bag07_p140_s224_badge.png` |
| Context crop | `reports/step_recovery_samples/bag07_p140_s224_context.png` |

**Visual verification:** **PASS**

Printed **224** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 7 — Page 143 — Step 228

| Field | Value |
|---|---|
| Detected step | **228** |
| OCR raw | `228` |
| Components | 2 |
| Step box | x=115 y=791 w=54 h=36 |
| Page image | `pages/70618_01/page_143.png` |
| Badge crop | `reports/step_recovery_samples/bag07_p143_s228_badge.png` |
| Context crop | `reports/step_recovery_samples/bag07_p143_s228_context.png` |

**Visual verification:** **PASS**

Printed **228** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 7 — Page 145 — Step 237

| Field | Value |
|---|---|
| Detected step | **237** |
| OCR raw | `237` |
| Components | 3 |
| Step box | x=855 y=142 w=81 h=36 |
| Page image | `pages/70618_01/page_145.png` |
| Badge crop | `reports/step_recovery_samples/bag07_p145_s237_badge.png` |
| Context crop | `reports/step_recovery_samples/bag07_p145_s237_context.png` |

**Visual verification:** **PASS**

Printed **237** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 9 — Page 166 — Step 286

| Field | Value |
|---|---|
| Detected step | **286** |
| OCR raw | `286` |
| Components | 3 |
| Step box | x=770 y=723 w=83 h=36 |
| Page image | `pages/70618_01/page_166.png` |
| Badge crop | `reports/step_recovery_samples/bag09_p166_s286_badge.png` |
| Context crop | `reports/step_recovery_samples/bag09_p166_s286_context.png` |

**Visual verification:** **PASS**

Printed **286** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 9 — Page 167 — Step 288

| Field | Value |
|---|---|
| Detected step | **288** |
| OCR raw | `288` |
| Components | 2 |
| Step box | x=115 y=728 w=54 h=37 |
| Page image | `pages/70618_01/page_167.png` |
| Badge crop | `reports/step_recovery_samples/bag09_p167_s288_badge.png` |
| Context crop | `reports/step_recovery_samples/bag09_p167_s288_context.png` |

**Visual verification:** **PASS**

Printed **288** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 9 — Page 168 — Step 291

| Field | Value |
|---|---|
| Detected step | **291** |
| OCR raw | `291` |
| Components | 3 |
| Step box | x=30 y=168 w=70 h=36 |
| Page image | `pages/70618_01/page_168.png` |
| Badge crop | `reports/step_recovery_samples/bag09_p168_s291_badge.png` |
| Context crop | `reports/step_recovery_samples/bag09_p168_s291_context.png` |

**Visual verification:** **PASS**

Printed **291** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 9 — Page 168 — Step 293

| Field | Value |
|---|---|
| Detected step | **293** |
| OCR raw | `293` |
| Components | 3 |
| Step box | x=770 y=168 w=83 h=36 |
| Page image | `pages/70618_01/page_168.png` |
| Badge crop | `reports/step_recovery_samples/bag09_p168_s293_badge.png` |
| Context crop | `reports/step_recovery_samples/bag09_p168_s293_context.png` |

**Visual verification:** **PASS**

Printed **293** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 9 — Page 174 — Step 302

| Field | Value |
|---|---|
| Detected step | **302** |
| OCR raw | `302` |
| Components | 3 |
| Step box | x=770 y=142 w=84 h=36 |
| Page image | `pages/70618_01/page_174.png` |
| Badge crop | `reports/step_recovery_samples/bag09_p174_s302_badge.png` |
| Context crop | `reports/step_recovery_samples/bag09_p174_s302_context.png` |

**Visual verification:** **PASS**

Printed **302** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 12 — Page 213 — Step 367

| Field | Value |
|---|---|
| Detected step | **367** |
| OCR raw | `367` |
| Components | 3 |
| Step box | x=855 y=489 w=82 h=36 |
| Page image | `pages/70618_01/page_213.png` |
| Badge crop | `reports/step_recovery_samples/bag12_p213_s367_badge.png` |
| Context crop | `reports/step_recovery_samples/bag12_p213_s367_context.png` |

**Visual verification:** **PASS**

Printed **367** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 12 — Page 217 — Step 373

| Field | Value |
|---|---|
| Detected step | **373** |
| OCR raw | `373` |
| Components | 3 |
| Step box | x=115 y=210 w=80 h=36 |
| Page image | `pages/70618_01/page_217.png` |
| Badge crop | `reports/step_recovery_samples/bag12_p217_s373_badge.png` |
| Context crop | `reports/step_recovery_samples/bag12_p217_s373_context.png` |

**Visual verification:** **PASS**

Printed **373** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 12 — Page 225 — Step 389

| Field | Value |
|---|---|
| Detected step | **389** |
| OCR raw | `389` |
| Components | 3 |
| Step box | x=855 y=163 w=83 h=37 |
| Page image | `pages/70618_01/page_225.png` |
| Badge crop | `reports/step_recovery_samples/bag12_p225_s389_badge.png` |
| Context crop | `reports/step_recovery_samples/bag12_p225_s389_context.png` |

**Visual verification:** **PASS**

Printed **389** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 12 — Page 229 — Step 405

| Field | Value |
|---|---|
| Detected step | **405** |
| OCR raw | `405` |
| Components | 3 |
| Step box | x=855 y=723 w=84 h=36 |
| Page image | `pages/70618_01/page_229.png` |
| Badge crop | `reports/step_recovery_samples/bag12_p229_s405_badge.png` |
| Context crop | `reports/step_recovery_samples/bag12_p229_s405_context.png` |

**Visual verification:** **PASS**

Printed **405** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 12 — Page 232 — Step 416

| Field | Value |
|---|---|
| Detected step | **416** |
| OCR raw | `416` |
| Components | 3 |
| Step box | x=625 y=142 w=73 h=37 |
| Page image | `pages/70618_01/page_232.png` |
| Badge crop | `reports/step_recovery_samples/bag12_p232_s416_badge.png` |
| Context crop | `reports/step_recovery_samples/bag12_p232_s416_context.png` |

**Visual verification:** **PASS**

Printed **416** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 15 — Page 281 — Step 491

| Field | Value |
|---|---|
| Detected step | **491** |
| OCR raw | `491` |
| Components | 3 |
| Step box | x=1101 y=142 w=70 h=36 |
| Page image | `pages/70618_01/page_281.png` |
| Badge crop | `reports/step_recovery_samples/bag15_p281_s491_badge.png` |
| Context crop | `reports/step_recovery_samples/bag15_p281_s491_context.png` |

**Visual verification:** **PASS**

Printed **491** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 15 — Page 282 — Step 492

| Field | Value |
|---|---|
| Detected step | **492** |
| OCR raw | `492` |
| Components | 3 |
| Step box | x=30 y=142 w=82 h=36 |
| Page image | `pages/70618_01/page_282.png` |
| Badge crop | `reports/step_recovery_samples/bag15_p282_s492_badge.png` |
| Context crop | `reports/step_recovery_samples/bag15_p282_s492_context.png` |

**Visual verification:** **PASS**

Printed **492** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 15 — Page 282 — Step 494

| Field | Value |
|---|---|
| Detected step | **494** |
| OCR raw | `494` |
| Components | 3 |
| Step box | x=1016 y=142 w=83 h=36 |
| Page image | `pages/70618_01/page_282.png` |
| Badge crop | `reports/step_recovery_samples/bag15_p282_s494_badge.png` |
| Context crop | `reports/step_recovery_samples/bag15_p282_s494_context.png` |

**Visual verification:** **PASS**

Printed **494** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 15 — Page 292 — Step 517

| Field | Value |
|---|---|
| Detected step | **517** |
| OCR raw | `517` |
| Components | 3 |
| Step box | x=1017 y=142 w=70 h=36 |
| Page image | `pages/70618_01/page_292.png` |
| Badge crop | `reports/step_recovery_samples/bag15_p292_s517_badge.png` |
| Context crop | `reports/step_recovery_samples/bag15_p292_s517_context.png` |

**Visual verification:** **PASS**

Printed **517** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

## Bag 15 — Page 293 — Step 518

| Field | Value |
|---|---|
| Detected step | **518** |
| OCR raw | `518` |
| Components | 3 |
| Step box | x=115 y=142 w=72 h=36 |
| Page image | `pages/70618_01/page_293.png` |
| Badge crop | `reports/step_recovery_samples/bag15_p293_s518_badge.png` |
| Context crop | `reports/step_recovery_samples/bag15_p293_s518_context.png` |

**Visual verification:** **PASS**

Printed **518** visible in badge crop; matches detected step_number and OCR raw text. Large bold black global-step typography on light-blue instruction background. Context crop shows standard step layout (parts box above, build illustration beside). Not a page footer number or substep digit.

