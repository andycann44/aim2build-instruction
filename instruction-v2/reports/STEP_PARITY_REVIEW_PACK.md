# Step Parity Review Pack

Focused visual review explaining why **Bags 10, 13, and 14** fail sequence coherence.

**Date:** 2026-06-16  
**Source:** `indexes/05_step_map.json` only — no pipeline rerun, no patches.

**Main chain** = greedy page-order filter from `STEP_SEQUENCE_AUDIT.md` (keep step only if `step_number` exceeds all prior kept steps).

**Assets:** `reports/step_parity_review_pack/`

---

## Executive summary

| Bag | Pages | Steps | On chain | Off chain | Root cause |
|---:|---|---:|---:|---:|---|
| **10** | 186–193 | 17 | 4 | 13 | OCR digit swaps on pp 186 & 189 (325→345, 331→351) create false high anchors on main chain; genuine steps 327–344 fall off chain; page-order scramble pp 190–193 |
| **13** | 237–265 | 45 | 1 | 44 | **p237 step 474 is printed 424** (4↔7 OCR). One false anchor poisons chain; all genuine 425–467 steps become off-chain; tail pages have footer/substep noise (1, 2) |
| **14** | 266–273 | 9 | 0 | 9 | p266 bag-label **14** accepted as step; sub-assembly digits 5/7; p271 misread 472→44; all globals off-chain because chain stuck at 474 from bag-13 p237 |

### Why these bags fail

```
Bag 12 ends main chain at step 423 (p236)
         │
         ▼
Bag 13 p237: detected 474 ──printed 424──► false +50 jump on main chain
         │
         ├── steps 425–467: genuine, but ALL off chain (474 already consumed)
         └── tail noise: steps 1, 2 on pp 258/260/262

Bag 14: chain still at 474
         ├── p266: bag-label 14 accepted as global step
         ├── p268/270: sub-assembly panel digits 7, 5
         └── p267–273 globals 468–474: genuine but off chain (≤474)

Bag 10 pp 186–193:
         p185 main chain ends at 324
         ├── p186: printed 325/326, detected 345/346 → ON chain (wrong)
         ├── p187–188: printed 327+ correct but OFF chain (<345)
         ├── p189: printed 331/332, detected 351/352 → ON chain (wrong)
         └── p190–193: genuine 336–346 OFF chain; duplicates 345/346
```

---
## Bag 10 — pages 186–193

**Main chain entering range:** step **324** (bag 10 p185)

**Steps in range:** 17 | **On chain:** 4 | **Off chain:** 13

| Page | Step | Chain | Printed | Issue |
|---:|---:|---|---|---|
| 186 | 345 | **ON** | 325 | OCR 2↔4 swap: printed 325, detected 345 |
| 186 | 346 | **ON** | 326 | OCR 2↔4 swap: printed 326, detected 346 |
| 187 | 327 | off | 327 | Genuine global step |
| 187 | 328 | off | 328 | Genuine global step |
| 189 | 351 | **ON** | 331 | OCR 3↔5 swap: printed 331, detected 351 |
| 189 | 352 | **ON** | 332 | OCR 3↔5 swap: printed 332, detected 352 |
| 190 | 336 | off | 336 | Genuine global step |
| 190 | 337 | off | 337 | Genuine global step |
| 190 | 338 | off | 338 | Genuine global step |
| 190 | 339 | off | 339 | Genuine global step |
| 191 | 340 | off | 340 | Genuine global step |
| 191 | 341 | off | 341 | Genuine global step |
| 191 | 342 | off | 342 | Genuine global step |
| 191 | 343 | off | 343 | Genuine global step |
| 192 | 344 | off | 344 | Genuine global step |
| 192 | 345 | off | 345 | Genuine global step |
| 193 | 346 | off | 346 | Genuine global step |

### Page 186

![Page 186 thumbnail](step_parity_review_pack/page_186_thumb.png)

#### Step 345 (index 1) — ✅ ON MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **345** |
| Printed (visual) | **325** |
| Main chain | ✅ ON MAIN CHAIN |
| Step box | x=30 y=149 w=83 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p186_s345_i1_badge.png` |

![Badge p186 step 345](step_parity_review_pack/bag10_p186_s345_i1_badge.png)

**Assessment:** OCR 2↔4 swap: printed 325, detected 345

#### Step 346 (index 2) — ✅ ON MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **346** |
| Printed (visual) | **326** |
| Main chain | ✅ ON MAIN CHAIN |
| Step box | x=30 y=709 w=84 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p186_s346_i2_badge.png` |

![Badge p186 step 346](step_parity_review_pack/bag10_p186_s346_i2_badge.png)

**Assessment:** OCR 2↔4 swap: printed 326, detected 346

### Page 187

![Page 187 thumbnail](step_parity_review_pack/page_187_thumb.png)

#### Step 327 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **327** |
| Printed (visual) | **327** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=142 w=82 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p187_s327_i1_badge.png` |

![Badge p187 step 327](step_parity_review_pack/bag10_p187_s327_i1_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

#### Step 328 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **328** |
| Printed (visual) | **328** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=709 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p187_s328_i2_badge.png` |

![Badge p187 step 328](step_parity_review_pack/bag10_p187_s328_i2_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

### Page 189

![Page 189 thumbnail](step_parity_review_pack/page_189_thumb.png)

#### Step 351 (index 1) — ✅ ON MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **351** |
| Printed (visual) | **331** |
| Main chain | ✅ ON MAIN CHAIN |
| Step box | x=115 y=287 w=55 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p189_s351_i1_badge.png` |

![Badge p189 step 351](step_parity_review_pack/bag10_p189_s351_i1_badge.png)

**Assessment:** OCR 3↔5 swap: printed 331, detected 351

#### Step 352 (index 2) — ✅ ON MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **352** |
| Printed (visual) | **332** |
| Main chain | ✅ ON MAIN CHAIN |
| Step box | x=115 y=737 w=55 h=37 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p189_s352_i2_badge.png` |

![Badge p189 step 352](step_parity_review_pack/bag10_p189_s352_i2_badge.png)

**Assessment:** OCR 3↔5 swap: printed 332, detected 352

### Page 190

![Page 190 thumbnail](step_parity_review_pack/page_190_thumb.png)

#### Step 336 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **336** |
| Printed (visual) | **336** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=142 w=84 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p190_s336_i1_badge.png` |

![Badge p190 step 336](step_parity_review_pack/bag10_p190_s336_i1_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

#### Step 337 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **337** |
| Printed (visual) | **337** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=723 w=81 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p190_s337_i2_badge.png` |

![Badge p190 step 337](step_parity_review_pack/bag10_p190_s337_i2_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

#### Step 338 (index 3) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **338** |
| Printed (visual) | **338** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=770 y=142 w=83 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag10_p190_s338_i3_badge.png` |

![Badge p190 step 338](step_parity_review_pack/bag10_p190_s338_i3_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

#### Step 339 (index 4) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **339** |
| Printed (visual) | **339** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=770 y=723 w=83 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag10_p190_s339_i4_badge.png` |

![Badge p190 step 339](step_parity_review_pack/bag10_p190_s339_i4_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

### Page 191

![Page 191 thumbnail](step_parity_review_pack/page_191_thumb.png)

#### Step 340 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **340** |
| Printed (visual) | **340** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=142 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p191_s340_i1_badge.png` |

![Badge p191 step 340](step_parity_review_pack/bag10_p191_s340_i1_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

#### Step 341 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **341** |
| Printed (visual) | **341** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=723 w=69 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p191_s341_i2_badge.png` |

![Badge p191 step 341](step_parity_review_pack/bag10_p191_s341_i2_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

#### Step 342 (index 3) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **342** |
| Printed (visual) | **342** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=855 y=142 w=82 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag10_p191_s342_i3_badge.png` |

![Badge p191 step 342](step_parity_review_pack/bag10_p191_s342_i3_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

#### Step 343 (index 4) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **343** |
| Printed (visual) | **343** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=855 y=723 w=83 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag10_p191_s343_i4_badge.png` |

![Badge p191 step 343](step_parity_review_pack/bag10_p191_s343_i4_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

### Page 192

![Page 192 thumbnail](step_parity_review_pack/page_192_thumb.png)

#### Step 344 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **344** |
| Printed (visual) | **344** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=206 w=83 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p192_s344_i1_badge.png` |

![Badge p192 step 344](step_parity_review_pack/bag10_p192_s344_i1_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

#### Step 345 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **345** |
| Printed (visual) | **345** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=641 w=83 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p192_s345_i2_badge.png` |

![Badge p192 step 345](step_parity_review_pack/bag10_p192_s345_i2_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

### Page 193

![Page 193 thumbnail](step_parity_review_pack/page_193_thumb.png)

#### Step 346 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **346** |
| Printed (visual) | **346** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=142 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag10_p193_s346_i1_badge.png` |

![Badge p193 step 346](step_parity_review_pack/bag10_p193_s346_i1_badge.png)

**Assessment:** Genuine global step; off chain because earlier p186/p189 OCR overshoot (345/351) blocked 325–344 range

---

## Bag 13 — pages 237–265

**Main chain entering range:** step **423** (bag 12 p236)

**Steps in range:** 45 | **On chain:** 1 | **Off chain:** 44

| Page | Step | Chain | Printed | Issue |
|---:|---:|---|---|---|
| 237 | 474 | **ON** | 424 | OCR 4↔7 swap: printed 424, detected 474  |
| 238 | 425 | off | 425 | Genuine global step |
| 238 | 426 | off | 426 | Genuine global step |
| 239 | 427 | off | 427 | Genuine global step |
| 241 | 430 | off | 430 | Genuine global step |
| 242 | 431 | off | 431 | Genuine global step |
| 243 | 432 | off | 432 | Genuine global step |
| 244 | 433 | off | 433 | Genuine global step |
| 244 | 434 | off | 434 | Genuine global step |
| 245 | 435 | off | 435 | Genuine global step |
| 245 | 436 | off | 436 | Genuine global step |
| 246 | 437 | off | 437 | Genuine global step |
| 246 | 438 | off | 438 | Genuine global step |
| 247 | 439 | off | 439 | Genuine global step |
| 247 | 440 | off | 440 | Genuine global step |
| 248 | 441 | off | 441 | Genuine global step |
| 249 | 442 | off | 442 | Genuine global step |
| 249 | 443 | off | 443 | Genuine global step |
| 249 | 444 | off | 444 | Genuine global step |
| 249 | 445 | off | 445 | Genuine global step |
| 250 | 446 | off | 446 | Genuine global step |
| 250 | 447 | off | 447 | Genuine global step |
| 250 | 448 | off | 448 | Genuine global step |
| 250 | 449 | off | 449 | Genuine global step |
| 251 | 450 | off | 450 | Genuine global step |
| 251 | 451 | off | 451 | Genuine global step |
| 251 | 452 | off | 452 | Genuine global step |
| 251 | 453 | off | 453 | Genuine global step |
| 252 | 454 | off | 454 | Genuine global step |
| 252 | 455 | off | 455 | Genuine global step |
| 253 | 456 | off | 456 | Genuine global step |
| 253 | 457 | off | 457 | Genuine global step |
| 254 | 458 | off | 458 | Genuine global step |
| 254 | 459 | off | 459 | Genuine global step |
| 255 | 460 | off | 460 | Genuine global step |
| 256 | 461 | off | 461 | Genuine global step |
| 257 | 462 | off | 462 | Genuine global step |
| 258 | 463 | off | 463 | Genuine global step |
| 258 | 1 | off | 1 | Spurious low-number anchor (small box y=886)  |
| 260 | 464 | off | 464 | Genuine global step |
| 260 | 1 | off | 1 | Spurious low-number anchor (small box y=849)  |
| 262 | 465 | off | 465 | Genuine global step |
| 262 | 2 | off | 2 | Spurious low-number anchor (small box y=886)  |
| 264 | 466 | off | 466 | Genuine global step |
| 265 | 467 | off | 467 | Genuine global step |

### Page 237

![Page 237 thumbnail](step_parity_review_pack/page_237_thumb.png)

#### Step 474 (index 1) — ✅ ON MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **474** |
| Printed (visual) | **424** |
| Main chain | ✅ ON MAIN CHAIN |
| Step box | x=115 y=845 w=82 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p237_s474_i1_badge.png` |

![Badge p237 step 474](step_parity_review_pack/bag13_p237_s474_i1_badge.png)

**Assessment:** OCR 4↔7 swap: printed 424, detected 474 — root cause of bag-13 gap

### Page 238

![Page 238 thumbnail](step_parity_review_pack/page_238_thumb.png)

#### Step 425 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **425** |
| Printed (visual) | **425** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=142 w=82 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p238_s425_i1_badge.png` |

![Badge p238 step 425](step_parity_review_pack/bag13_p238_s425_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 426 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **426** |
| Printed (visual) | **426** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=722 w=83 h=37 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p238_s426_i2_badge.png` |

![Badge p238 step 426](step_parity_review_pack/bag13_p238_s426_i2_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 239

![Page 239 thumbnail](step_parity_review_pack/page_239_thumb.png)

#### Step 427 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **427** |
| Printed (visual) | **427** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=237 w=81 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p239_s427_i1_badge.png` |

![Badge p239 step 427](step_parity_review_pack/bag13_p239_s427_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 241

![Page 241 thumbnail](step_parity_review_pack/page_241_thumb.png)

#### Step 430 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **430** |
| Printed (visual) | **430** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=142 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p241_s430_i1_badge.png` |

![Badge p241 step 430](step_parity_review_pack/bag13_p241_s430_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 242

![Page 242 thumbnail](step_parity_review_pack/page_242_thumb.png)

#### Step 431 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **431** |
| Printed (visual) | **431** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=142 w=69 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p242_s431_i1_badge.png` |

![Badge p242 step 431](step_parity_review_pack/bag13_p242_s431_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 243

![Page 243 thumbnail](step_parity_review_pack/page_243_thumb.png)

#### Step 432 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **432** |
| Printed (visual) | **432** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=142 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p243_s432_i1_badge.png` |

![Badge p243 step 432](step_parity_review_pack/bag13_p243_s432_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 244

![Page 244 thumbnail](step_parity_review_pack/page_244_thumb.png)

#### Step 433 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **433** |
| Printed (visual) | **433** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=203 w=83 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p244_s433_i1_badge.png` |

![Badge p244 step 433](step_parity_review_pack/bag13_p244_s433_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 434 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **434** |
| Printed (visual) | **434** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=712 w=83 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p244_s434_i2_badge.png` |

![Badge p244 step 434](step_parity_review_pack/bag13_p244_s434_i2_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 245

![Page 245 thumbnail](step_parity_review_pack/page_245_thumb.png)

#### Step 435 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **435** |
| Printed (visual) | **435** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=203 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p245_s435_i1_badge.png` |

![Badge p245 step 435](step_parity_review_pack/bag13_p245_s435_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 436 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **436** |
| Printed (visual) | **436** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=712 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p245_s436_i2_badge.png` |

![Badge p245 step 436](step_parity_review_pack/bag13_p245_s436_i2_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 246

![Page 246 thumbnail](step_parity_review_pack/page_246_thumb.png)

#### Step 437 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **437** |
| Printed (visual) | **437** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=142 w=81 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p246_s437_i1_badge.png` |

![Badge p246 step 437](step_parity_review_pack/bag13_p246_s437_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 438 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **438** |
| Printed (visual) | **438** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=712 w=83 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p246_s438_i2_badge.png` |

![Badge p246 step 438](step_parity_review_pack/bag13_p246_s438_i2_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 247

![Page 247 thumbnail](step_parity_review_pack/page_247_thumb.png)

#### Step 439 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **439** |
| Printed (visual) | **439** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=142 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p247_s439_i1_badge.png` |

![Badge p247 step 439](step_parity_review_pack/bag13_p247_s439_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 440 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **440** |
| Printed (visual) | **440** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=716 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p247_s440_i2_badge.png` |

![Badge p247 step 440](step_parity_review_pack/bag13_p247_s440_i2_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 248

![Page 248 thumbnail](step_parity_review_pack/page_248_thumb.png)

#### Step 441 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **441** |
| Printed (visual) | **441** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=169 w=69 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p248_s441_i1_badge.png` |

![Badge p248 step 441](step_parity_review_pack/bag13_p248_s441_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 249

![Page 249 thumbnail](step_parity_review_pack/page_249_thumb.png)

#### Step 442 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **442** |
| Printed (visual) | **442** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=339 w=54 h=35 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p249_s442_i1_badge.png` |

![Badge p249 step 442](step_parity_review_pack/bag13_p249_s442_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 443 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **443** |
| Printed (visual) | **443** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=723 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p249_s443_i2_badge.png` |

![Badge p249 step 443](step_parity_review_pack/bag13_p249_s443_i2_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 444 (index 3) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **444** |
| Printed (visual) | **444** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=855 y=142 w=83 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag13_p249_s444_i3_badge.png` |

![Badge p249 step 444](step_parity_review_pack/bag13_p249_s444_i3_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 445 (index 4) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **445** |
| Printed (visual) | **445** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=855 y=723 w=82 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag13_p249_s445_i4_badge.png` |

![Badge p249 step 445](step_parity_review_pack/bag13_p249_s445_i4_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 250

![Page 250 thumbnail](step_parity_review_pack/page_250_thumb.png)

#### Step 446 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **446** |
| Printed (visual) | **446** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=142 w=83 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p250_s446_i1_badge.png` |

![Badge p250 step 446](step_parity_review_pack/bag13_p250_s446_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 447 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **447** |
| Printed (visual) | **447** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=723 w=80 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p250_s447_i2_badge.png` |

![Badge p250 step 447](step_parity_review_pack/bag13_p250_s447_i2_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 448 (index 3) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **448** |
| Printed (visual) | **448** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=770 y=150 w=83 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag13_p250_s448_i3_badge.png` |

![Badge p250 step 448](step_parity_review_pack/bag13_p250_s448_i3_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 449 (index 4) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **449** |
| Printed (visual) | **449** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=770 y=723 w=82 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag13_p250_s449_i4_badge.png` |

![Badge p250 step 449](step_parity_review_pack/bag13_p250_s449_i4_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 251

![Page 251 thumbnail](step_parity_review_pack/page_251_thumb.png)

#### Step 450 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **450** |
| Printed (visual) | **450** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=142 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p251_s450_i1_badge.png` |

![Badge p251 step 450](step_parity_review_pack/bag13_p251_s450_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 451 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **451** |
| Printed (visual) | **451** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=723 w=68 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p251_s451_i2_badge.png` |

![Badge p251 step 451](step_parity_review_pack/bag13_p251_s451_i2_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 452 (index 3) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **452** |
| Printed (visual) | **452** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=855 y=142 w=82 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag13_p251_s452_i3_badge.png` |

![Badge p251 step 452](step_parity_review_pack/bag13_p251_s452_i3_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 453 (index 4) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **453** |
| Printed (visual) | **453** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=855 y=723 w=83 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag13_p251_s453_i4_badge.png` |

![Badge p251 step 453](step_parity_review_pack/bag13_p251_s453_i4_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 252

![Page 252 thumbnail](step_parity_review_pack/page_252_thumb.png)

#### Step 454 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **454** |
| Printed (visual) | **454** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=142 w=83 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p252_s454_i1_badge.png` |

![Badge p252 step 454](step_parity_review_pack/bag13_p252_s454_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 455 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **455** |
| Printed (visual) | **455** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=741 y=143 w=83 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag13_p252_s455_i2_badge.png` |

![Badge p252 step 455](step_parity_review_pack/bag13_p252_s455_i2_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 253

![Page 253 thumbnail](step_parity_review_pack/page_253_thumb.png)

#### Step 456 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **456** |
| Printed (visual) | **456** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=156 w=54 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p253_s456_i1_badge.png` |

![Badge p253 step 456](step_parity_review_pack/bag13_p253_s456_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 457 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **457** |
| Printed (visual) | **457** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=855 y=142 w=80 h=36 |
| Source | sequence_gap_full_page_audit |
| Badge crop | `step_parity_review_pack/bag13_p253_s457_i2_badge.png` |

![Badge p253 step 457](step_parity_review_pack/bag13_p253_s457_i2_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 254

![Page 254 thumbnail](step_parity_review_pack/page_254_thumb.png)

#### Step 458 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **458** |
| Printed (visual) | **458** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=142 w=83 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p254_s458_i1_badge.png` |

![Badge p254 step 458](step_parity_review_pack/bag13_p254_s458_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 459 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **459** |
| Printed (visual) | **459** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=530 w=82 h=37 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p254_s459_i2_badge.png` |

![Badge p254 step 459](step_parity_review_pack/bag13_p254_s459_i2_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 255

![Page 255 thumbnail](step_parity_review_pack/page_255_thumb.png)

#### Step 460 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **460** |
| Printed (visual) | **460** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=196 w=55 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p255_s460_i1_badge.png` |

![Badge p255 step 460](step_parity_review_pack/bag13_p255_s460_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 256

![Page 256 thumbnail](step_parity_review_pack/page_256_thumb.png)

#### Step 461 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **461** |
| Printed (visual) | **461** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=142 w=70 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p256_s461_i1_badge.png` |

![Badge p256 step 461](step_parity_review_pack/bag13_p256_s461_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 257

![Page 257 thumbnail](step_parity_review_pack/page_257_thumb.png)

#### Step 462 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **462** |
| Printed (visual) | **462** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=142 w=55 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p257_s462_i1_badge.png` |

![Badge p257 step 462](step_parity_review_pack/bag13_p257_s462_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 258

![Page 258 thumbnail](step_parity_review_pack/page_258_thumb.png)

#### Step 463 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **463** |
| Printed (visual) | **463** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=190 w=83 h=37 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p258_s463_i1_badge.png` |

![Badge p258 step 463](step_parity_review_pack/bag13_p258_s463_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 1 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **1** |
| Printed (visual) | **1** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=79 y=886 w=10 h=30 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p258_s1_i2_badge.png` |

![Badge p258 step 1](step_parity_review_pack/bag13_p258_s1_i2_badge.png)

**Assessment:** Spurious low-number anchor (small box y=886) — panel/substep/page-footer digit

### Page 260

![Page 260 thumbnail](step_parity_review_pack/page_260_thumb.png)

#### Step 464 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **464** |
| Printed (visual) | **464** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=371 w=83 h=37 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p260_s464_i1_badge.png` |

![Badge p260 step 464](step_parity_review_pack/bag13_p260_s464_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 1 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **1** |
| Printed (visual) | **1** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=31 y=849 w=10 h=30 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p260_s1_i2_badge.png` |

![Badge p260 step 1](step_parity_review_pack/bag13_p260_s1_i2_badge.png)

**Assessment:** Spurious low-number anchor (small box y=849) — panel/substep/page-footer digit

### Page 262

![Page 262 thumbnail](step_parity_review_pack/page_262_thumb.png)

#### Step 465 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **465** |
| Printed (visual) | **465** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=374 w=83 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p262_s465_i1_badge.png` |

![Badge p262 step 465](step_parity_review_pack/bag13_p262_s465_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

#### Step 2 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **2** |
| Printed (visual) | **2** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=156 y=886 w=21 h=30 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p262_s2_i2_badge.png` |

![Badge p262 step 2](step_parity_review_pack/bag13_p262_s2_i2_badge.png)

**Assessment:** Spurious low-number anchor (small box y=886) — panel/substep/page-footer digit

### Page 264

![Page 264 thumbnail](step_parity_review_pack/page_264_thumb.png)

#### Step 466 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **466** |
| Printed (visual) | **466** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=308 w=84 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p264_s466_i1_badge.png` |

![Badge p264 step 466](step_parity_review_pack/bag13_p264_s466_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

### Page 265

![Page 265 thumbnail](step_parity_review_pack/page_265_thumb.png)

#### Step 467 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **467** |
| Printed (visual) | **467** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=239 w=56 h=37 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag13_p265_s467_i1_badge.png` |

![Badge p265 step 467](step_parity_review_pack/bag13_p265_s467_i1_badge.png)

**Assessment:** Genuine global step; off chain because p237 false anchor 474 (printed 424) consumed main chain

---

## Bag 14 — pages 266–273

**Main chain entering range:** step **474** (bag 13 p237)

**Steps in range:** 9 | **On chain:** 0 | **Off chain:** 9

| Page | Step | Chain | Printed | Issue |
|---:|---:|---|---|---|
| 266 | 14 | off | 14 (bag label) | Bag-14 indicator graphic, not a global build step |
| 267 | 468 | off | 468 | Genuine global step |
| 268 | 469 | off | 469 | Genuine global step |
| 268 | 7 | off | 7 | Spurious low-number anchor (small box y=679)  |
| 269 | 470 | off | 470 | Genuine global step |
| 270 | 5 | off | 5 | Spurious low-number anchor (small box y=820)  |
| 271 | 44 | off | 472 | OCR fragment: printed 472, detected 44 |
| 272 | 473 | off | 473 | Genuine global step |
| 273 | 474 | off | 474 | Genuine global step |

### Page 266

![Page 266 thumbnail](step_parity_review_pack/page_266_thumb.png)

#### Step 14 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **14** |
| Printed (visual) | **14 (bag label)** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=109 y=327 w=61 h=49 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag14_p266_s14_i1_badge.png` |

![Badge p266 step 14](step_parity_review_pack/bag14_p266_s14_i1_badge.png)

**Assessment:** Bag-14 indicator graphic, not a global build step

### Page 267

![Page 267 thumbnail](step_parity_review_pack/page_267_thumb.png)

#### Step 468 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **468** |
| Printed (visual) | **468** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=165 w=55 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag14_p267_s468_i1_badge.png` |

![Badge p267 step 468](step_parity_review_pack/bag14_p267_s468_i1_badge.png)

**Assessment:** Genuine global step; off chain because main chain already at 474 from bag-13 p237 misread

### Page 268

![Page 268 thumbnail](step_parity_review_pack/page_268_thumb.png)

#### Step 469 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **469** |
| Printed (visual) | **469** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=142 w=83 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag14_p268_s469_i1_badge.png` |

![Badge p268 step 469](step_parity_review_pack/bag14_p268_s469_i1_badge.png)

**Assessment:** Genuine global step; off chain because main chain already at 474 from bag-13 p237 misread

#### Step 7 (index 4) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **7** |
| Printed (visual) | **7** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=0 y=679 w=24 h=43 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag14_p268_s7_i4_badge.png` |

![Badge p268 step 7](step_parity_review_pack/bag14_p268_s7_i4_badge.png)

**Assessment:** Spurious low-number anchor (small box y=679) — panel/substep/page-footer digit

### Page 269

![Page 269 thumbnail](step_parity_review_pack/page_269_thumb.png)

#### Step 470 (index 2) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **470** |
| Printed (visual) | **470** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=142 w=80 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag14_p269_s470_i2_badge.png` |

![Badge p269 step 470](step_parity_review_pack/bag14_p269_s470_i2_badge.png)

**Assessment:** Genuine global step; off chain because main chain already at 474 from bag-13 p237 misread

### Page 270

![Page 270 thumbnail](step_parity_review_pack/page_270_thumb.png)

#### Step 5 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **5** |
| Printed (visual) | **5** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=1 y=820 w=17 h=48 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag14_p270_s5_i1_badge.png` |

![Badge p270 step 5](step_parity_review_pack/bag14_p270_s5_i1_badge.png)

**Assessment:** Spurious low-number anchor (small box y=820) — panel/substep/page-footer digit

### Page 271

![Page 271 thumbnail](step_parity_review_pack/page_271_thumb.png)

#### Step 44 (index 3) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **44** |
| Printed (visual) | **472** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=142 w=51 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag14_p271_s44_i3_badge.png` |

![Badge p271 step 44](step_parity_review_pack/bag14_p271_s44_i3_badge.png)

**Assessment:** OCR fragment: printed 472, detected 44

### Page 272

![Page 272 thumbnail](step_parity_review_pack/page_272_thumb.png)

#### Step 473 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **473** |
| Printed (visual) | **473** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=30 y=142 w=79 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag14_p272_s473_i1_badge.png` |

![Badge p272 step 473](step_parity_review_pack/bag14_p272_s473_i1_badge.png)

**Assessment:** Genuine global step; off chain because main chain already at 474 from bag-13 p237 misread

### Page 273

![Page 273 thumbnail](step_parity_review_pack/page_273_thumb.png)

#### Step 474 (index 1) — ❌ OFF MAIN CHAIN

| Field | Value |
|---|---|
| Detected step | **474** |
| Printed (visual) | **474** |
| Main chain | ❌ OFF MAIN CHAIN |
| Step box | x=115 y=142 w=76 h=36 |
| Source | visual detection |
| Badge crop | `step_parity_review_pack/bag14_p273_s474_i1_badge.png` |

![Badge p273 step 474](step_parity_review_pack/bag14_p273_s474_i1_badge.png)

**Assessment:** Genuine global step; off chain because main chain already at 474 from bag-13 p237 misread

---

## Recommendations (read-only — no action taken)

1. **Bag 13 p237:** Re-OCR or correct 474 → 424. This single fix would restore 44 off-chain steps (425–467) to main chain and unblock bag 14.
2. **Bag 10 pp 186/189:** Correct 345→325, 346→326, 351→331, 352→332. Would restore pp 187–193 steps to main chain.
3. **Bag 14 p266:** Reject bag-label box (printed **14** on bag graphic, not global step).
4. **Bags 13–14 tail:** Reject footer/substep anchors (steps 1, 2, 5, 7) via position/size rules (y>650, w<25).
5. **Bag 14 p271:** Correct 44 → 472.

## Method

- Manifest: `instruction-v2/indexes/05_step_map.json`
- Page images: `pages/70618_01/page_XXX.png`
- Badge crops: `step_box` + 12 px pad
- Page thumbnails: 360 px wide
- Main-chain algorithm: identical to `STEP_SEQUENCE_AUDIT.md`

