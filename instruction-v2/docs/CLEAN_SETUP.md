# Instruction Reader — Clean Setup

Single entrypoint for the known-good V2 pipeline plus V1 review handoff.

**Runner:** `instruction-v2/a2b_instruction_run.py`

V2 writes `instruction-v2/indexes/`. Stage 5e exports to `debug/crop_cache/`. The V1 review UI reads `debug/crop_cache/` and writes human labels to `debug/training_labels/`. Do not touch Azure/R2 or rewrite review verdicts from this flow.

---

## 1. Pipeline order

Baseline stages only (in execution order):

| # | Script | Writes |
|---|--------|--------|
| 0 | `stage0_set_context.py` | `indexes/00_set_context.json` |
| 1 | `phase1_pdf_pages.py` | `indexes/01_pdf_manifest.json`, `indexes/02_page_index.json`, `pages/<run_id>/` |
| 2 | `stage2_bag_candidates.py` | `indexes/03_bag_candidates.json` |
| 3 | `stage3b_bag_gap_review.py` | `indexes/03b_bag_gap_review.json` |
| 4 | `stage3_bag_map.py` | `indexes/04_bag_map.json` |
| 5 | `stage4_step_map.py` | `indexes/05_step_map.json` |
| 6 | `stage5_callout_crop_boxes.py` | `indexes/06_callout_crop_box_map.json` |
| 7 | `stage5d_sequence_completeness_diagnostics.py` | `indexes/06a_sequence_completeness_diagnostics.json` |
| 8 | `stage5e_export_crop_cache.py` | `debug/crop_cache/{set}_bag{N}.json` |
| 9 | `stage6_qty_ocr.py` | `indexes/07_qty_ocr_map.json` |
| 10 | `stage7_part_segmentation.py` | `indexes/08_part_segmentation_map.json` |
| 11 | `stage8_match.py` | `indexes/09_match_manifest.json` |

**Stoppoints** (`--to`):

| Value | Runs through |
|-------|----------------|
| `step-map` | Stages 0–5 |
| `crop-cache` | Stages 0–8 (includes crop-cache export) |
| `qty` | Stages 0–9 |
| `segmentation` | Stages 0–10 |
| `match` | Stages 0–11 (full baseline) |

Stages 0–7 and 9–11 are **set-wide** (rebuild indexes for the whole PDF). Only stage 5e is **bag-scoped** (`--bag`).

---

## 2. Source files (edit / version control)

### Runner

- `instruction-v2/a2b_instruction_run.py` — unified CLI entrypoint

### Baseline stage scripts (do not replace with orchestrators)

- `stage0_set_context.py`
- `phase1_pdf_pages.py`
- `stage2_bag_candidates.py`
- `stage3b_bag_gap_review.py`
- `stage3_bag_map.py`
- `stage4_step_map.py`
- `stage5_callout_crop_boxes.py`
- `stage5d_sequence_completeness_diagnostics.py`
- `stage5e_export_crop_cache.py`
- `stage6_qty_ocr.py`
- `stage7_part_segmentation.py`
- `stage8_match.py`

### Supporting source (invoked by stages, not by runner directly)

- `paths.py` — V2 directory constants
- `validate_pipeline.py` — preflight before stage 8
- `render_pdf_pages.mjs`, `bag_candidate_scan.mjs`, `gap_window_scan.mjs`, `step_map_scan.mjs`, `callout_crop_box_scan.mjs`, `qty_ocr_scan.mjs`, `part_segmentation_scan.mjs`, `match_segments_scan.mjs`

### V1 review app (read crop cache, write labels)

- `clean/main.py`
- `clean/routers/instruction_debug.py` — `GET /debug/manual-match-review`
- `clean/services/bag_review_service.py`

### Input PDF

- `instruction-v2/pdfs/70618_01.pdf` (default for set 70618)

---

## 3. Runtime folders (generated / working state)

| Path | Role |
|------|------|
| `instruction-v2/indexes/` | V2 manifest chain (`00`–`09`, etc.) |
| `instruction-v2/pages/` | Rendered page PNGs per run id |
| `instruction-v2/debug/` | V2 stage debug overlays (step map, callouts, segmentation) |
| `debug/crop_cache/` | **V1 review crop list** — written by stage 5e |
| `debug/training_labels/` | **Human labels** — source of truth; never overwritten by V2 |
| `debug/ai_training/` | Masks, cutouts (V1 + stage 7 artifacts) |
| `debug/{set}/{run_id}/pages/` | V1-rendered pages (legacy; V2 uses `instruction-v2/pages/`) |

---

## 4. Banned / experimental (do not use in clean runs)

| Item | Reason |
|------|--------|
| `stage5_orchestrator.py` | Non-baseline orchestration; not in repo baseline |
| `stage5f_callout_quality.py` | Experimental quality gate; not baseline |
| `recover_bag*_v1_callouts.py` | Recovery / parity experiments |
| `promote_bag4_recovered_crops.py` | Manual promotion path |
| `apply_bag4_sequence_corrections.py` | Audit-driven corrections |
| `agents/pdf_ingest_agent.py` | R2 ingest; out of scope |
| Azure/R2 upload flags on export routes | Cloud sync disabled for clean setup |

Audit scripts (`audit_*.py`, `create_*_contact_sheet.py`, `create_bag*_page_review_report.py`) are diagnostic only — run manually, not via `a2b_instruction_run.py`.

---

## 5. Exact commands — set 70618

From repo root. Defaults: `--set-num 70618`, `--pdf instruction-v2/pdfs/70618_01.pdf`, `--bag 4`.

### Full index rebuild to step map

```bash
python instruction-v2/a2b_instruction_run.py \
  --set-num 70618 \
  --pdf instruction-v2/pdfs/70618_01.pdf \
  --to step-map
```

### Bag 4 — rebuild indexes + export crop cache + open review

```bash
python instruction-v2/a2b_instruction_run.py \
  --set-num 70618 \
  --pdf instruction-v2/pdfs/70618_01.pdf \
  --bag 4 \
  --to crop-cache \
  --open-review
```

Crop cache output: `debug/crop_cache/70618_bag4.json`

Review UI (V1 server must be running):

```text
http://127.0.0.1:8000/debug/manual-match-review?set_num=70618&bag=4
```

Start server:

```bash
uvicorn clean.main:app --reload --host 127.0.0.1 --port 8000
```

### Bag 5 — crop cache + review

Full rebuild through crop-cache export:

```bash
python instruction-v2/a2b_instruction_run.py \
  --set-num 70618 \
  --pdf instruction-v2/pdfs/70618_01.pdf \
  --bag 5 \
  --to crop-cache \
  --open-review
```

If indexes are already current, export bag 5 only (no full rebuild):

```bash
python instruction-v2/stage5e_export_crop_cache.py --set-num 70618 --bag 5
python instruction-v2/a2b_instruction_run.py --set-num 70618 --bag 5 --open-review
```

Output: `debug/crop_cache/70618_bag5.json`  
Review: `?set_num=70618&bag=5`

### Bag 6 — crop cache + review

Full rebuild through crop-cache export:

```bash
python instruction-v2/a2b_instruction_run.py \
  --set-num 70618 \
  --pdf instruction-v2/pdfs/70618_01.pdf \
  --bag 6 \
  --to crop-cache \
  --open-review
```

If indexes are already current, export bag 6 only:

```bash
python instruction-v2/stage5e_export_crop_cache.py --set-num 70618 --bag 6
python instruction-v2/a2b_instruction_run.py --set-num 70618 --bag 6 --open-review
```

Output: `debug/crop_cache/70618_bag6.json`  
Review: `?set_num=70618&bag=6`

### Continue pipeline after crop cache (qty → match)

Bag 4 example through qty OCR:

```bash
python instruction-v2/a2b_instruction_run.py \
  --set-num 70618 \
  --pdf instruction-v2/pdfs/70618_01.pdf \
  --bag 4 \
  --to qty
```

Through segmentation:

```bash
python instruction-v2/a2b_instruction_run.py \
  --set-num 70618 \
  --pdf instruction-v2/pdfs/70618_01.pdf \
  --bag 4 \
  --to segmentation
```

Full baseline through match manifest:

```bash
python instruction-v2/a2b_instruction_run.py \
  --set-num 70618 \
  --pdf instruction-v2/pdfs/70618_01.pdf \
  --bag 4 \
  --to match
```

### Open review only (no pipeline)

```bash
python instruction-v2/a2b_instruction_run.py \
  --set-num 70618 \
  --bag 4 \
  --open-review
```

---

## Rules recap

1. **V2 writes indexes** under `instruction-v2/indexes/`.
2. **V2 exports crop cache** via stage 5e to `debug/crop_cache/`.
3. **V1 review UI** reads crop cache; labels stay in `debug/training_labels/`.
4. **No Azure/R2** from this runner.
5. **No review verdict changes** — runner does not touch `training_labels/`.
6. **No stage script edits** — runner only subprocess-invokes baseline scripts.
