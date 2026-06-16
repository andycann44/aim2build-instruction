# Step Count Delta Analysis

Read-only comparison of `instruction-v2/indexes/05_step_map.json`.

| | Commit | Description |
|---|---|---|
| **Parent (before)** | `0e85cc8` | Stage4 baseline-aware 2-digit merge (page 59 → step 79) |
| **Current (after)** | `39d84c1` | OCR cap aligned to V1 + 3-digit merge + visual match + sequence correction |

## Executive summary

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Manifest `step_count` | 250 | 561 | +311 |
| `steps[]` entries (all) | 250 | 561 | +311 |
| **Accepted** (`rejection_reason` null) | 216 | 521 | +305 |
| Rejected | 34 | 40 | +6 |

### Why `step_count` rose 250 → 561 (+311)

The manifest `step_count` equals `steps[]` length (accepted + rejected). The increase is almost entirely **newly accepted global step anchors** in bags where printed step numbers exceed 200.

| Mechanism | Effect |
|---|---|
| **OCR cap removal** (`> 200` → V1 `< 1000`) | Previously detected boxes showed `step ?` in overlays; OCR reads 201–499 were discarded. Unlocks ~entire bags 7–15. |
| **Neighboring merge** | 3-digit printed steps (e.g. 239, 368) now merge 3 digit components before OCR. |
| **Visual match + primary OCR fallback** | Primary boxes that diverged from full-page reads now OCR directly. |
| **Sequence correction** | Same-page OCR fixes (e.g. 476→426). |
| **Removals** | −9 fragment anchors consolidated (e.g. page 59 `7`+`7` → `79`). |

Net accepted delta: **+305** (216 → 521). Remaining +6 in total `steps[]` is additional rejected rows (34 → 40).

## 1. Newly accepted steps by bag

| Bag | Before | After | Delta |
|---:|---:|---:|---:|
| 1 | 24 | 25 | +1 |
| 2 | 15 | 15 | +0 |
| 3 | 35 | 35 | +0 |
| 4 | 51 | 52 | +1 |
| 5 | 30 | 30 | +0 |
| 6 | 37 | 43 | +6 |
| 7 | 4 | 38 | +34 |
| 8 | 5 | 35 | +30 |
| 9 | 0 | 30 | +30 |
| 10 | 0 | 32 | +32 |
| 11 | 1 | 19 | +18 |
| 12 | 3 | 58 | +55 |
| 13 | 2 | 45 | +43 |
| 14 | 2 | 9 | +7 |
| 15 | 7 | 55 | +48 |

Bags **1–5** (steps mostly ≤ 200): **+1** net (bag 1 +1, bag 4 +1).
Bags **6–15**: **+304** net — these page ranges contain printed steps 201–499 blocked by the old cap.

## 2. Newly accepted steps by page

- Pages with increases: **161**
- Pages with decreases: **0**
- Sum of positive deltas: **+305**

Full per-page delta table (pages with any change):

| Page | Before | After | Delta |
|---:|---:|---:|---:|
| 10 | 1 | 2 | +1 |
| 78 | 2 | 3 | +1 |
| 121 | 2 | 3 | +1 |
| 126 | 0 | 1 | +1 |
| 127 | 0 | 1 | +1 |
| 128 | 0 | 1 | +1 |
| 129 | 0 | 1 | +1 |
| 130 | 0 | 1 | +1 |
| 133 | 0 | 4 | +4 |
| 134 | 0 | 4 | +4 |
| 135 | 0 | 1 | +1 |
| 136 | 0 | 2 | +2 |
| 137 | 0 | 1 | +1 |
| 138 | 0 | 2 | +2 |
| 139 | 0 | 2 | +2 |
| 140 | 1 | 2 | +1 |
| 141 | 0 | 1 | +1 |
| 142 | 0 | 1 | +1 |
| 143 | 0 | 4 | +4 |
| 144 | 0 | 4 | +4 |
| 145 | 0 | 4 | +4 |
| 146 | 0 | 3 | +3 |
| 150 | 0 | 4 | +4 |
| 151 | 0 | 1 | +1 |
| 152 | 0 | 2 | +2 |
| 153 | 0 | 2 | +2 |
| 154 | 1 | 2 | +1 |
| 155 | 0 | 1 | +1 |
| 156 | 0 | 2 | +2 |
| 158 | 0 | 1 | +1 |
| 159 | 0 | 4 | +4 |
| 160 | 0 | 4 | +4 |
| 161 | 0 | 4 | +4 |
| 162 | 0 | 3 | +3 |
| 163 | 0 | 1 | +1 |
| 165 | 0 | 3 | +3 |
| 166 | 0 | 4 | +4 |
| 167 | 0 | 4 | +4 |
| 168 | 0 | 3 | +3 |
| 169 | 0 | 1 | +1 |
| 170 | 0 | 2 | +2 |
| 171 | 0 | 2 | +2 |
| 172 | 0 | 1 | +1 |
| 173 | 0 | 1 | +1 |
| 174 | 0 | 2 | +2 |
| 175 | 0 | 2 | +2 |
| 176 | 0 | 2 | +2 |
| 177 | 0 | 1 | +1 |
| 178 | 0 | 2 | +2 |
| 179 | 0 | 1 | +1 |
| 180 | 0 | 3 | +3 |
| 181 | 0 | 3 | +3 |
| 182 | 0 | 2 | +2 |
| 183 | 0 | 2 | +2 |
| 184 | 0 | 2 | +2 |
| 185 | 0 | 2 | +2 |
| 186 | 0 | 2 | +2 |
| 187 | 0 | 2 | +2 |
| 189 | 0 | 2 | +2 |
| 190 | 0 | 4 | +4 |
| 191 | 0 | 4 | +4 |
| 192 | 0 | 2 | +2 |
| 193 | 0 | 1 | +1 |
| 194 | 1 | 2 | +1 |
| 195 | 0 | 1 | +1 |
| 196 | 0 | 1 | +1 |
| 197 | 0 | 1 | +1 |
| 198 | 0 | 1 | +1 |
| 199 | 0 | 1 | +1 |
| 200 | 0 | 1 | +1 |
| 201 | 0 | 1 | +1 |
| 202 | 0 | 1 | +1 |
| 203 | 0 | 1 | +1 |
| 204 | 0 | 1 | +1 |
| 205 | 0 | 1 | +1 |
| 206 | 0 | 1 | +1 |
| 207 | 0 | 1 | +1 |
| 208 | 0 | 1 | +1 |
| 209 | 0 | 1 | +1 |
| 210 | 0 | 1 | +1 |
| 211 | 0 | 1 | +1 |
| 213 | 0 | 2 | +2 |
| 214 | 0 | 2 | +2 |
| 215 | 0 | 2 | +2 |
| 216 | 0 | 1 | +1 |
| 217 | 0 | 1 | +1 |
| 218 | 0 | 2 | +2 |
| 219 | 0 | 2 | +2 |
| 220 | 0 | 2 | +2 |
| 221 | 0 | 2 | +2 |
| 222 | 0 | 2 | +2 |
| 223 | 0 | 2 | +2 |
| 224 | 0 | 1 | +1 |
| 225 | 0 | 4 | +4 |
| 226 | 0 | 2 | +2 |
| 227 | 0 | 4 | +4 |
| 228 | 1 | 4 | +3 |
| 229 | 0 | 4 | +4 |
| 230 | 0 | 4 | +4 |
| 231 | 0 | 4 | +4 |
| 232 | 2 | 5 | +3 |
| 233 | 0 | 1 | +1 |
| 234 | 0 | 2 | +2 |
| 235 | 0 | 2 | +2 |
| 236 | 0 | 1 | +1 |
| 237 | 0 | 1 | +1 |
| 238 | 0 | 2 | +2 |
| 239 | 0 | 1 | +1 |
| 241 | 0 | 1 | +1 |
| 242 | 0 | 1 | +1 |
| 243 | 0 | 1 | +1 |
| 244 | 0 | 2 | +2 |
| 245 | 0 | 2 | +2 |
| 246 | 0 | 2 | +2 |
| 247 | 0 | 2 | +2 |
| 248 | 0 | 1 | +1 |
| 249 | 0 | 4 | +4 |
| 250 | 0 | 4 | +4 |
| 251 | 0 | 4 | +4 |
| 252 | 0 | 2 | +2 |
| 253 | 0 | 2 | +2 |
| 254 | 0 | 2 | +2 |
| 255 | 0 | 1 | +1 |
| 256 | 0 | 1 | +1 |
| 257 | 0 | 1 | +1 |
| 258 | 1 | 2 | +1 |
| 260 | 1 | 2 | +1 |
| 262 | 0 | 2 | +2 |
| 264 | 0 | 1 | +1 |
| 265 | 0 | 1 | +1 |
| 267 | 0 | 1 | +1 |
| 268 | 1 | 2 | +1 |
| 269 | 0 | 1 | +1 |
| 270 | 0 | 1 | +1 |
| 271 | 0 | 1 | +1 |
| 272 | 0 | 1 | +1 |
| 273 | 0 | 1 | +1 |
| 275 | 0 | 5 | +5 |
| 276 | 0 | 3 | +3 |
| 277 | 0 | 1 | +1 |
| 278 | 0 | 1 | +1 |
| 279 | 0 | 2 | +2 |
| 280 | 0 | 1 | +1 |
| 281 | 0 | 4 | +4 |
| 282 | 0 | 3 | +3 |
| 283 | 0 | 3 | +3 |
| 284 | 0 | 2 | +2 |
| 285 | 0 | 3 | +3 |
| 286 | 0 | 1 | +1 |
| 287 | 0 | 2 | +2 |
| 288 | 0 | 1 | +1 |
| 289 | 0 | 1 | +1 |
| 290 | 0 | 4 | +4 |
| 291 | 0 | 2 | +2 |
| 292 | 0 | 3 | +3 |
| 293 | 0 | 1 | +1 |
| 295 | 0 | 1 | +1 |
| 296 | 0 | 1 | +1 |
| 297 | 0 | 1 | +1 |
| 298 | 0 | 1 | +1 |
| 299 | 0 | 1 | +1 |

## 3. Top 25 pages with largest increases

| Rank | Page | Before | After | Delta | Bag |
|---:|---:|---:|---:|---:|---:|
| 1 | 275 | 0 | 5 | +5 | 15 |
| 2 | 133 | 0 | 4 | +4 | 7 |
| 3 | 134 | 0 | 4 | +4 | 7 |
| 4 | 143 | 0 | 4 | +4 | 7 |
| 5 | 144 | 0 | 4 | +4 | 7 |
| 6 | 145 | 0 | 4 | +4 | 7 |
| 7 | 150 | 0 | 4 | +4 | 8 |
| 8 | 159 | 0 | 4 | +4 | 8 |
| 9 | 160 | 0 | 4 | +4 | 8 |
| 10 | 161 | 0 | 4 | +4 | 8 |
| 11 | 166 | 0 | 4 | +4 | 9 |
| 12 | 167 | 0 | 4 | +4 | 9 |
| 13 | 190 | 0 | 4 | +4 | 10 |
| 14 | 191 | 0 | 4 | +4 | 10 |
| 15 | 225 | 0 | 4 | +4 | 12 |
| 16 | 227 | 0 | 4 | +4 | 12 |
| 17 | 229 | 0 | 4 | +4 | 12 |
| 18 | 230 | 0 | 4 | +4 | 12 |
| 19 | 231 | 0 | 4 | +4 | 12 |
| 20 | 249 | 0 | 4 | +4 | 13 |
| 21 | 250 | 0 | 4 | +4 | 13 |
| 22 | 251 | 0 | 4 | +4 | 13 |
| 23 | 281 | 0 | 4 | +4 | 15 |
| 24 | 290 | 0 | 4 | +4 | 15 |
| 25 | 146 | 0 | 3 | +3 | 7 |

**Pattern:** Most top pages went from **0 → 4 or 0 → 5** accepted steps. These are normal multi-callout instruction pages in bags 7–15 whose step numbers (207–479) were entirely suppressed by the `> 200` OCR cap in the parent commit.

Example after-state page 133: steps **207, 208, 209, 210**. Example page 275: **475–479**.

## 4. Each new accepted step — source rule

New entries (exact `page+step_number+box` key, in after not before): **314**.
Removed entries (in before not after): **9**.

### Rule attribution summary

| Source rule | New steps |
|---|---:|
| neighboring merge | 225 |
| OCR cap removal | 64 |
| sequence correction | 20 |
| primary detection | 4 |
| visual match | 1 |

Priority: `sequence correction` → `neighboring merge` (components ≥ 3) → `OCR cap removal` (step > 200) → `visual match` (2-component parity targets) → `primary detection`.

### All new steps

| Page | Step | Components | OCR raw | Source rule |
|---:|---:|---:|---|---|
| 10 | 8 | 1 | 8 | primary detection |
| 40 | 45 | 2 | 44 | sequence correction |
| 56 | 17 | 2 | 77 | sequence correction |
| 78 | 125 | 3 | 125 | neighboring merge |
| 121 | 191 | 3 | 197 | sequence correction |
| 121 | 192 | 3 | 192 | neighboring merge |
| 126 | 202 | 3 | 202 | neighboring merge |
| 127 | 203 | 2 | 203 | OCR cap removal |
| 128 | 204 | 3 | 204 | neighboring merge |
| 129 | 205 | 2 | 205 | OCR cap removal |
| 130 | 206 | 3 | 206 | neighboring merge |
| 133 | 207 | 2 | 207 | OCR cap removal |
| 133 | 208 | 2 | 208 | OCR cap removal |
| 133 | 209 | 3 | 209 | neighboring merge |
| 133 | 210 | 3 | 210 | neighboring merge |
| 134 | 211 | 3 | 211 | neighboring merge |
| 134 | 212 | 3 | 212 | neighboring merge |
| 134 | 213 | 3 | 213 | neighboring merge |
| 134 | 214 | 3 | 214 | neighboring merge |
| 135 | 145 | 3 | 145 | neighboring merge |
| 136 | 216 | 3 | 216 | neighboring merge |
| 136 | 217 | 3 | 217 | neighboring merge |
| 137 | 219 | 3 | 219 | neighboring merge |
| 138 | 220 | 3 | 220 | neighboring merge |
| 138 | 221 | 3 | 221 | neighboring merge |
| 139 | 222 | 2 | 222 | OCR cap removal |
| 139 | 223 | 2 | 223 | OCR cap removal |
| 140 | 224 | 3 | 224 | neighboring merge |
| 141 | 225 | 2 | 225 | OCR cap removal |
| 142 | 226 | 3 | 226 | neighboring merge |
| 143 | 227 | 3 | 227 | neighboring merge |
| 143 | 228 | 2 | 228 | OCR cap removal |
| 143 | 229 | 3 | 229 | neighboring merge |
| 143 | 230 | 3 | 230 | neighboring merge |
| 144 | 231 | 3 | 231 | neighboring merge |
| 144 | 232 | 3 | 232 | neighboring merge |
| 144 | 233 | 3 | 233 | neighboring merge |
| 144 | 234 | 3 | 234 | neighboring merge |
| 145 | 235 | 2 | 235 | OCR cap removal |
| 145 | 236 | 2 | 236 | OCR cap removal |
| 145 | 237 | 3 | 237 | neighboring merge |
| 145 | 238 | 3 | 238 | neighboring merge |
| 146 | 239 | 3 | 239 | neighboring merge |
| 146 | 240 | 3 | 240 | neighboring merge |
| 146 | 241 | 3 | 241 | neighboring merge |
| 147 | 3 | 1 | 7 | sequence correction |
| 150 | 247 | 3 | 247 | neighboring merge |
| 150 | 248 | 3 | 248 | neighboring merge |
| 150 | 249 | 3 | 249 | neighboring merge |
| 150 | 250 | 3 | 250 | neighboring merge |
| 151 | 251 | 3 | 251 | neighboring merge |
| 152 | 252 | 3 | 252 | neighboring merge |
| 152 | 253 | 3 | 253 | neighboring merge |
| 153 | 254 | 3 | 254 | neighboring merge |
| 153 | 255 | 2 | 255 | OCR cap removal |
| 154 | 256 | 3 | 256 | neighboring merge |
| 155 | 258 | 2 | 258 | OCR cap removal |
| 156 | 259 | 3 | 259 | neighboring merge |
| 156 | 260 | 3 | 260 | neighboring merge |
| 158 | 262 | 3 | 262 | neighboring merge |
| 159 | 263 | 2 | 263 | OCR cap removal |
| 159 | 264 | 2 | 264 | OCR cap removal |
| 159 | 265 | 3 | 265 | neighboring merge |
| 159 | 266 | 3 | 266 | neighboring merge |
| 160 | 267 | 3 | 267 | neighboring merge |
| 160 | 268 | 3 | 268 | neighboring merge |
| 160 | 269 | 3 | 269 | neighboring merge |
| 160 | 270 | 3 | 270 | neighboring merge |
| 161 | 271 | 3 | 271 | neighboring merge |
| 161 | 272 | 3 | 272 | neighboring merge |
| 161 | 273 | 3 | 273 | neighboring merge |
| 161 | 274 | 3 | 274 | neighboring merge |
| 162 | 275 | 3 | 275 | neighboring merge |
| 162 | 276 | 3 | 276 | neighboring merge |
| 162 | 277 | 3 | 277 | neighboring merge |
| 163 | 278 | 3 | 278 | neighboring merge |
| 165 | 279 | 3 | 279 | neighboring merge |
| 165 | 280 | 2 | 280 | OCR cap removal |
| 165 | 281 | 3 | 281 | neighboring merge |
| 166 | 283 | 3 | 283 | neighboring merge |
| 166 | 284 | 3 | 284 | neighboring merge |
| 166 | 285 | 3 | 285 | neighboring merge |
| 166 | 286 | 3 | 286 | neighboring merge |
| 167 | 287 | 3 | 287 | neighboring merge |
| 167 | 288 | 2 | 288 | OCR cap removal |
| 167 | 289 | 3 | 289 | neighboring merge |
| 167 | 290 | 3 | 290 | neighboring merge |
| 168 | 291 | 3 | 291 | neighboring merge |
| 168 | 292 | 3 | 292 | neighboring merge |
| 168 | 293 | 3 | 293 | neighboring merge |
| 169 | 294 | 2 | 294 | OCR cap removal |
| 170 | 295 | 3 | 295 | neighboring merge |
| 170 | 296 | 3 | 296 | neighboring merge |
| 171 | 297 | 3 | 297 | neighboring merge |
| 171 | 298 | 3 | 298 | neighboring merge |
| 172 | 299 | 3 | 299 | neighboring merge |
| 173 | 300 | 2 | 300 | OCR cap removal |
| 174 | 301 | 3 | 301 | neighboring merge |
| 174 | 302 | 3 | 302 | neighboring merge |
| 175 | 303 | 2 | 303 | OCR cap removal |
| 175 | 304 | 3 | 304 | neighboring merge |
| 176 | 305 | 3 | 305 | neighboring merge |
| 176 | 306 | 3 | 306 | neighboring merge |
| 177 | 307 | 2 | 307 | OCR cap removal |
| 178 | 308 | 3 | 308 | neighboring merge |
| 178 | 309 | 3 | 309 | neighboring merge |
| 179 | 310 | 3 | 310 | neighboring merge |
| 180 | 311 | 3 | 311 | neighboring merge |
| 180 | 312 | 3 | 312 | neighboring merge |
| 180 | 313 | 3 | 313 | neighboring merge |
| 181 | 314 | 3 | 314 | neighboring merge |
| 181 | 315 | 3 | 315 | neighboring merge |
| 181 | 316 | 3 | 316 | neighboring merge |
| 182 | 317 | 3 | 317 | neighboring merge |
| 182 | 318 | 3 | 318 | neighboring merge |
| 183 | 319 | 3 | 319 | neighboring merge |
| 183 | 320 | 3 | 320 | neighboring merge |
| 184 | 321 | 3 | 321 | neighboring merge |
| 184 | 322 | 3 | 372 | sequence correction |
| 185 | 323 | 2 | 323 | OCR cap removal |
| 185 | 324 | 3 | 394 | sequence correction |
| 186 | 345 | 3 | 345 | neighboring merge |
| 186 | 346 | 3 | 376 | sequence correction |
| 187 | 327 | 3 | 327 | neighboring merge |
| 187 | 328 | 2 | 378 | sequence correction |
| 189 | 351 | 2 | 351 | OCR cap removal |
| 189 | 352 | 2 | 332 | sequence correction |
| 190 | 336 | 3 | 336 | neighboring merge |
| 190 | 337 | 3 | 337 | neighboring merge |
| 190 | 338 | 3 | 338 | neighboring merge |
| 190 | 339 | 3 | 339 | neighboring merge |
| 191 | 340 | 2 | 340 | OCR cap removal |
| 191 | 341 | 3 | 341 | neighboring merge |
| 191 | 342 | 3 | 342 | neighboring merge |
| 191 | 343 | 3 | 343 | neighboring merge |
| 192 | 344 | 3 | 344 | neighboring merge |
| 192 | 345 | 3 | 345 | neighboring merge |
| 193 | 346 | 2 | 346 | OCR cap removal |
| 194 | 347 | 3 | 347 | neighboring merge |
| 195 | 348 | 2 | 348 | OCR cap removal |
| 196 | 349 | 3 | 349 | neighboring merge |
| 197 | 350 | 2 | 350 | OCR cap removal |
| 198 | 351 | 3 | 351 | neighboring merge |
| 199 | 352 | 2 | 352 | OCR cap removal |
| 200 | 353 | 3 | 353 | neighboring merge |
| 201 | 354 | 2 | 354 | OCR cap removal |
| 202 | 355 | 3 | 355 | neighboring merge |
| 203 | 356 | 2 | 356 | OCR cap removal |
| 204 | 357 | 3 | 357 | neighboring merge |
| 205 | 358 | 2 | 358 | OCR cap removal |
| 206 | 359 | 3 | 359 | neighboring merge |
| 207 | 360 | 2 | 360 | OCR cap removal |
| 208 | 361 | 3 | 361 | neighboring merge |
| 209 | 362 | 2 | 362 | OCR cap removal |
| 210 | 363 | 3 | 363 | neighboring merge |
| 211 | 364 | 2 | 364 | OCR cap removal |
| 213 | 366 | 2 | 366 | OCR cap removal |
| 213 | 367 | 3 | 367 | neighboring merge |
| 214 | 368 | 3 | 368 | neighboring merge |
| 214 | 369 | 3 | 369 | neighboring merge |
| 215 | 370 | 3 | 370 | neighboring merge |
| 215 | 371 | 3 | 371 | neighboring merge |
| 216 | 372 | 3 | 372 | neighboring merge |
| 217 | 373 | 3 | 373 | neighboring merge |
| 218 | 314 | 3 | 314 | neighboring merge |
| 218 | 315 | 3 | 345 | sequence correction |
| 219 | 316 | 3 | 316 | neighboring merge |
| 219 | 317 | 3 | 377 | sequence correction |
| 220 | 378 | 3 | 378 | neighboring merge |
| 220 | 379 | 3 | 379 | neighboring merge |
| 221 | 380 | 2 | 380 | OCR cap removal |
| 221 | 381 | 2 | 351 | sequence correction |
| 222 | 382 | 3 | 382 | neighboring merge |
| 222 | 383 | 3 | 383 | neighboring merge |
| 223 | 384 | 2 | 384 | OCR cap removal |
| 223 | 385 | 2 | 385 | OCR cap removal |
| 224 | 386 | 3 | 386 | neighboring merge |
| 225 | 387 | 3 | 387 | neighboring merge |
| 225 | 388 | 2 | 388 | OCR cap removal |
| 225 | 389 | 3 | 389 | neighboring merge |
| 225 | 390 | 3 | 390 | neighboring merge |
| 226 | 397 | 3 | 397 | neighboring merge |
| 226 | 398 | 3 | 392 | sequence correction |
| 227 | 395 | 2 | 395 | OCR cap removal |
| 227 | 396 | 2 | 396 | OCR cap removal |
| 227 | 397 | 3 | 397 | neighboring merge |
| 227 | 398 | 3 | 398 | neighboring merge |
| 228 | 399 | 3 | 399 | neighboring merge |
| 228 | 400 | 3 | 400 | neighboring merge |
| 228 | 401 | 3 | 401 | neighboring merge |
| 229 | 402 | 2 | 402 | OCR cap removal |
| 229 | 403 | 2 | 403 | OCR cap removal |
| 229 | 404 | 3 | 404 | neighboring merge |
| 229 | 405 | 3 | 405 | neighboring merge |
| 230 | 406 | 3 | 406 | neighboring merge |
| 230 | 407 | 3 | 407 | neighboring merge |
| 230 | 408 | 3 | 408 | neighboring merge |
| 230 | 409 | 3 | 409 | neighboring merge |
| 231 | 410 | 3 | 410 | neighboring merge |
| 231 | 411 | 3 | 411 | neighboring merge |
| 231 | 412 | 3 | 412 | neighboring merge |
| 231 | 413 | 3 | 413 | neighboring merge |
| 232 | 8 | 1 | 7 | sequence correction |
| 232 | 414 | 3 | 414 | neighboring merge |
| 232 | 415 | 3 | 415 | neighboring merge |
| 232 | 416 | 3 | 416 | neighboring merge |
| 233 | 418 | 3 | 418 | neighboring merge |
| 234 | 419 | 3 | 419 | neighboring merge |
| 234 | 420 | 3 | 420 | neighboring merge |
| 235 | 421 | 2 | 421 | OCR cap removal |
| 235 | 422 | 2 | 422 | OCR cap removal |
| 236 | 423 | 3 | 423 | neighboring merge |
| 237 | 474 | 3 | 474 | neighboring merge |
| 238 | 425 | 3 | 425 | neighboring merge |
| 238 | 426 | 3 | 476 | sequence correction |
| 239 | 427 | 3 | 427 | neighboring merge |
| 241 | 430 | 2 | 430 | OCR cap removal |
| 242 | 431 | 3 | 431 | neighboring merge |
| 243 | 432 | 2 | 432 | OCR cap removal |
| 244 | 433 | 3 | 433 | neighboring merge |
| 244 | 434 | 3 | 434 | neighboring merge |
| 245 | 435 | 2 | 435 | OCR cap removal |
| 245 | 436 | 2 | 436 | OCR cap removal |
| 246 | 437 | 3 | 437 | neighboring merge |
| 246 | 438 | 3 | 438 | neighboring merge |
| 247 | 439 | 2 | 439 | OCR cap removal |
| 247 | 440 | 2 | 440 | OCR cap removal |
| 248 | 441 | 3 | 441 | neighboring merge |
| 249 | 442 | 2 | 442 | OCR cap removal |
| 249 | 443 | 2 | 443 | OCR cap removal |
| 249 | 444 | 3 | 444 | neighboring merge |
| 249 | 445 | 3 | 445 | neighboring merge |
| 250 | 446 | 3 | 446 | neighboring merge |
| 250 | 447 | 3 | 447 | neighboring merge |
| 250 | 448 | 3 | 448 | neighboring merge |
| 250 | 449 | 3 | 449 | neighboring merge |
| 251 | 450 | 2 | 450 | OCR cap removal |
| 251 | 451 | 3 | 451 | neighboring merge |
| 251 | 452 | 3 | 452 | neighboring merge |
| 251 | 453 | 3 | 453 | neighboring merge |
| 252 | 454 | 3 | 454 | neighboring merge |
| 252 | 455 | 3 | 455 | neighboring merge |
| 253 | 456 | 2 | 456 | OCR cap removal |
| 253 | 457 | 3 | 457 | neighboring merge |
| 254 | 458 | 3 | 458 | neighboring merge |
| 254 | 459 | 3 | 459 | neighboring merge |
| 255 | 460 | 2 | 460 | OCR cap removal |
| 256 | 461 | 3 | 461 | neighboring merge |
| 257 | 462 | 2 | 462 | OCR cap removal |
| 258 | 463 | 3 | 463 | neighboring merge |
| 260 | 464 | 3 | 464 | neighboring merge |
| 262 | 2 | 1 | 2 | primary detection |
| 262 | 465 | 3 | 465 | neighboring merge |
| 264 | 466 | 3 | 466 | neighboring merge |
| 265 | 467 | 2 | 467 | OCR cap removal |
| 267 | 468 | 2 | 468 | OCR cap removal |
| 268 | 469 | 3 | 469 | neighboring merge |
| 269 | 470 | 3 | 470 | neighboring merge |
| 270 | 5 | 1 | 5 | primary detection |
| 271 | 44 | 2 | 44 | primary detection |
| 272 | 473 | 3 | 473 | neighboring merge |
| 273 | 474 | 3 | 474 | neighboring merge |
| 275 | 475 | 3 | 475 | neighboring merge |
| 275 | 476 | 3 | 416 | sequence correction |
| 275 | 477 | 3 | 477 | neighboring merge |
| 275 | 478 | 3 | 478 | neighboring merge |
| 275 | 479 | 3 | 479 | neighboring merge |
| 276 | 480 | 3 | 480 | neighboring merge |
| 276 | 481 | 3 | 481 | neighboring merge |
| 276 | 482 | 3 | 482 | neighboring merge |
| 277 | 483 | 2 | 483 | visual match |
| 278 | 484 | 3 | 484 | neighboring merge |
| 279 | 485 | 2 | 485 | OCR cap removal |
| 279 | 486 | 3 | 486 | neighboring merge |
| 280 | 487 | 3 | 487 | neighboring merge |
| 281 | 488 | 2 | 488 | OCR cap removal |
| 281 | 489 | 3 | 489 | neighboring merge |
| 281 | 490 | 3 | 490 | neighboring merge |
| 281 | 491 | 3 | 491 | neighboring merge |
| 282 | 492 | 3 | 492 | neighboring merge |
| 282 | 493 | 3 | 493 | neighboring merge |
| 282 | 494 | 3 | 494 | neighboring merge |
| 283 | 495 | 2 | 495 | OCR cap removal |
| 283 | 496 | 3 | 496 | neighboring merge |
| 283 | 497 | 3 | 497 | neighboring merge |
| 284 | 498 | 3 | 498 | neighboring merge |
| 284 | 499 | 3 | 499 | neighboring merge |
| 285 | 500 | 2 | 500 | OCR cap removal |
| 285 | 501 | 3 | 501 | neighboring merge |
| 285 | 502 | 3 | 502 | neighboring merge |
| 286 | 503 | 3 | 503 | neighboring merge |
| 287 | 504 | 2 | 504 | OCR cap removal |
| 287 | 505 | 3 | 505 | neighboring merge |
| 288 | 506 | 3 | 506 | neighboring merge |
| 289 | 507 | 3 | 507 | neighboring merge |
| 290 | 508 | 3 | 508 | neighboring merge |
| 290 | 509 | 3 | 509 | neighboring merge |
| 290 | 510 | 3 | 510 | neighboring merge |
| 290 | 511 | 3 | 511 | neighboring merge |
| 291 | 512 | 3 | 512 | neighboring merge |
| 291 | 514 | 3 | 514 | neighboring merge |
| 292 | 515 | 3 | 515 | neighboring merge |
| 292 | 516 | 3 | 516 | neighboring merge |
| 292 | 517 | 3 | 517 | neighboring merge |
| 293 | 518 | 3 | 518 | neighboring merge |
| 295 | 522 | 2 | 522 | OCR cap removal |
| 296 | 524 | 3 | 524 | neighboring merge |
| 297 | 525 | 3 | 525 | neighboring merge |
| 298 | 526 | 3 | 526 | neighboring merge |
| 299 | 527 | 3 | 527 | neighboring merge |
| 304 | 2 | 1 | 7 | sequence correction |
| 305 | 8 | 1 | 7 | sequence correction |
| 305 | 9 | 1 | 7 | sequence correction |
| 311 | 5 | 1 | 7 | sequence correction |

### Removed steps (parent only)

| Page | Step | Components | Likely reason |
|---:|---:|---:|---|
| 40 | 44 | 2 | superseded |
| 56 | 77 | 2 | superseded |
| 121 | 197 | 3 | superseded |
| 147 | 7 | 1 | single-digit fragment consolidated |
| 232 | 7 | 1 | single-digit fragment consolidated |
| 304 | 7 | 1 | single-digit fragment consolidated |
| 305 | 7 | 1 | single-digit fragment consolidated |
| 305 | 7 | 1 | single-digit fragment consolidated |
| 311 | 7 | 1 | single-digit fragment consolidated |

## 5. Noise check — substeps / callout-local / page numbers

| Category | Count | Finding |
|---|---:|---|
| Page numbers (footer zone or step==page) | 0 | **None** among new steps |
| Likely substeps / panel digits (1–12, 1 component, w≤30) | 9 | **8** marginal; not driver of +311 |
| Global steps > 200 | 298 | **298** — expected late-manual anchors |

### Likely substep / callout-local candidates (8)

| Page | Step | Box (x,y,w,h) | Components |
|---:|---:|---|---:|
| 10 | 8 | 30,795,25,37 | 1 |
| 147 | 3 | 3,493,15,31 | 1 |
| 232 | 8 | 103,410,13,32 | 1 |
| 262 | 2 | 156,886,21,30 | 1 |
| 270 | 5 | 1,820,17,48 | 1 |
| 304 | 2 | 85,461,29,66 | 1 |
| 305 | 8 | 0,268,18,33 | 1 |
| 305 | 9 | 51,327,14,68 | 1 |
| 311 | 5 | 115,872,27,62 | 1 |

**Assessment:**

- **Substeps:** 8 low-number single-component anchors may be panel/sub-assembly digits; combined they account for **< 3%** of new entries.
- **Callout-local numbers:** No qty-box digit pattern detected; new high-number anchors use standard left-gutter global step geometry.
- **Page numbers:** 0 matches.

## Known parity targets (8 multi-digit)

| Step | Page | Before | After | Rule |
|---:|---:|---|---|---|
| 79 | 59 | yes | yes | primary detection |
| 239 | 146 | no | yes | neighboring merge |
| 278 | 163 | no | yes | neighboring merge |
| 368 | 214 | no | yes | neighboring merge |
| 369 | 214 | no | yes | neighboring merge |
| 426 | 238 | no | yes | sequence correction |
| 459 | 254 | no | yes | neighboring merge |
| 483 | 277 | no | yes | visual match |
