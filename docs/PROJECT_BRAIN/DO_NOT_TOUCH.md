# Do Not Touch

This document lists systems, files, and behaviours that must not be modified
without an explicit review and deliberate decision. These are the load-bearing
walls of the project.

Future AI agents: **read this file before making any changes.** If a change you
are about to make would affect anything listed here, stop and ask for explicit
authorisation.

---

## 1 — Source-of-Truth Rules

**Do not change which file is the authoritative source of truth for Bag Review.**

The file `debug/training_labels/{set_num}_bag{bag}.json` is the one and only
authoritative store for bag review state. Any change that:

- Moves review state to a database
- Duplicates review state in a second file
- Causes a non-`bag_review_service` code path to write to this file

...requires explicit authorisation and a decision update in `DECISIONS.md`.

---

## 2 — Bag Review Workflow

**Do not change the Bag Review workflow without explicit review.**

The workflow is:

```
crop cache (read-only) → bag_review_service → training_labels JSON
                       ↗ (also: legacy _write_labels() in instruction_debug.py)
```

Specifically:

- Bag Review slot-state actions use `bag_review_service`:
  - `save_slot_label` (via `save-label` with `selected_slot_index >= 0`)
  - `mark_slot_unknown`
  - `mark_slot_ignored`
- Legacy/debug mutation routes write directly via `_write_labels()` in `instruction_debug.py`:
  - `save-label` (non-slot path, when no `selected_slot_index`)
  - `remove-label`
  - `set-crop-status`
  - `delete-crop`
  - `save-manual-crop`
  - `update-crop-qty`
- The `load_review_state` / `save_review_state` pair is the canonical
  Bag Review read/write interface

**Do not add more write paths.** The legacy routes already exist and are
tolerated. Do not extend this pattern. Future work should consolidate, not
expand (see DECISIONS.md D14).

---

## 3 — `training_labels` Schema

**Do not change the schema of `debug/training_labels/{set_num}_bag{bag}.json`
without bumping `schema_version` and updating `DECISIONS.md`.**

Current schema version: `"1.1"`

Protected fields:

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | Must be `"1.1"` currently |
| `set_num` | string | Must match the filename |
| `bag` | int | Must match the filename |
| `crops` | dict | Keys are `crop_id` strings |
| `crops[id].parts` | list | Each part has `part_num`, `color_id`, `qty`, `selected_slot_index` |
| `crops[id].unknown_slots` | list[int] | Sorted list of unknown slot indexes |
| `crops[id].ignored_slots` | list[int] | Sorted list of ignored slot indexes |
| `crops[id].review_status` | string | `"reviewed"` or `"unreviewed"` |
| `crops[id].annotated_at` | string | ISO8601Z timestamp of last write |

---

## 4 — `save-label` Behaviour

**Do not change the semantics of `POST /debug/save-label`.**

Current behaviour that must be preserved:

- When `selected_slot_index` is provided and `>= 0`, delegates to
  `bag_review_service.save_slot_label()` — this is the correct path for Bag
  Review
- Saving a label for a slot removes that slot from `unknown_slots` and
  `ignored_slots` (a label overrides an unknown/ignored mark)
- The label is stored under `parts[].selected_slot_index` so it can be
  round-tripped correctly

---

## 5 — Unknown / Ignored Slot Semantics

**Do not change the semantics of unknown and ignored slots.**

- `unknown`: The part is present but cannot be identified. The slot is
  considered reviewed (counts toward completion).
- `ignored`: The slot is a false detection or an artefact (e.g. a qty=2 crop
  where the second detected slot has no separate part to label). The slot is
  considered reviewed and excluded from training.
- A slot can only be in one state: `saved_label` OR `unknown` OR `ignored`.
  Setting any one of these clears the others for that slot.
- All three states count as "complete" for progress calculation:
  `complete = confirmed + unknown + ignored`

---

## 6 — Crop Cache Contracts

**Do not let the Bag Review service write to the crop cache.**

`debug/crop_cache/{set_num}_bag{bag}.json` is populated by the crop detection
pipeline. The Bag Review service reads it as an immutable input. It provides:

- `crop_id` (primary key for the review model)
- `page`, `step`, `crop_box`, `crop_box_format`
- `crop_image_path`
- `qty_numbers`, `qty_text` (or `candidate_detected_*` variants)

If the crop cache is updated (e.g. crop detection is re-run), the review state
in `training_labels` is not automatically invalidated — the reviewer must check
that any changed crops are still correctly labelled.

---

## 7 — Catalog Embedding Format

**Do not change the format of `debug/catalog_clip_embeddings/` without
updating the suggestion pipeline.**

Current format:

- `embeddings.npy` — NumPy float32 array, shape `(N, D)` where D is the CLIP
  embedding dimension
- `items.json` — list of N dicts, each with at least `part_num`, `color_id`,
  `element_id`
- `manifest.json` — metadata about the embedding run (date, model, count)

The `buildability-clip-suggest` route depends on this format. Any change to
the array layout or item schema requires updating the suggestion route and
regenerating the embeddings.

---

## 8 — `bag_review_service.py` Is Not a General-Purpose Label Store

**Do not add new concerns to `bag_review_service.py` that are not directly
related to the Bag Review workflow.**

The service is responsible for:
- Loading and saving review state
- Building the review model (crops + slots + progress)
- Mutating individual slots (save, unknown, ignored)

It is not responsible for:
- Training bundle export
- CLIP embedding generation
- Catalog queries
- Suggestion ranking

---

## Current Status

### Set 70618 — Bag 1

Review complete at baseline (2026-06-01):

```
total_slots:        48
confirmed_slots:    47
unknown_slots:       0
ignored_slots:       1
needs_review_slots:  0
percent_complete:   100
```

**Ignored slot detail:**

| crop_id | slot_index | display | reason |
|---|---|---|---|
| `p20_s23_c1` | 2 | Slot 3: IGNORED | detected extra slot; the crop has two real parts |

**Verified SHA-256:**

```
fdf5ca0a5cc04787e7d5d7ef699ed3c928d4a4236521b44e80fc17c55879522b
```

File: `debug/training_labels/70618_bag1.json`

---

### Set 70618 — Bags 2, 3, 4

Review files exist at:

```
debug/training_labels/70618_bag2.json
debug/training_labels/70618_bag3.json
debug/training_labels/70618_bag4.json
```

These bags have not been verified at a baseline equivalent to Bag 1. Do not
assume they are complete.

---

## Verification Commands

Run these to verify the system is healthy and Bag 1 remains at baseline.

### 1 — Compile check

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/a2b_pycache \
  .venv/bin/python -m py_compile \
  clean/services/bag_review_service.py \
  clean/routers/instruction_debug.py
```

### 2 — Import check

```bash
.venv/bin/python -c "import clean.main; print('ok')"
```

### 3 — HTTP status check

```bash
.venv/bin/python - <<'PY'
from fastapi.testclient import TestClient
from clean.main import app
client = TestClient(app)
res = client.get('/debug/manual-match-review', params={'set_num': '70618', 'bag': 1})
print(res.status_code)
PY
```

Expected: `200`

### 4 — Bag 1 review model check

```bash
.venv/bin/python - <<'PY'
from clean.services.bag_review_service import build_review_model
m = build_review_model('70618', 1)
p = m['progress']
assert p['total_slots'] == 48, p
assert p['confirmed_slots'] == 47, p
assert p['ignored_slots'] == 1, p
assert p['unknown_slots'] == 0, p
assert p['needs_review_slots'] == 0, p
assert p['percent_complete'] == 100, p
assert p['next_unreviewed'] is None, p
print('Bag 1 baseline OK:', p)
PY
```

Expected output:

```
Bag 1 baseline OK: {'total_slots': 48, 'confirmed_slots': 47, 'unknown_slots': 0, 'ignored_slots': 1, 'needs_review_slots': 0, 'percent_complete': 100, 'next_unreviewed': None}
```

---

| Area | Risk if changed without review |
|---|---|
| `bag_review_service.save_review_state()` | Could corrupt or lose authoritative label data |
| `bag_review_service.save_slot_label()` | Could break slot-level label storage |
| `bag_review_service.mark_slot_unknown()` | Could corrupt unknown/ignored tracking |
| `bag_review_service.mark_slot_ignored()` | Could corrupt unknown/ignored tracking |
| `POST /debug/save-label` slot routing | Could route labels to wrong slot or wrong file |
| Schema of `training_labels` JSON | Could break read/write compatibility |
| `_label_store_path()` logic | Could write to wrong file location |
| `_crop_cache_path()` logic | Could break crop loading |
| Slot completion counting logic in `build_review_model()` | Could misreport progress |
| CLIP memory write in `buildability-clip-suggest` | Could accidentally write to training labels |
