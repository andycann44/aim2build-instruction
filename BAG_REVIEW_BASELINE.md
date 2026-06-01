# Bag Review Baseline

Date: 2026-06-01

## Scope

This baseline captures the current Bag Review workflow for set `70618`, bag `1`.

The review workflow stays inside the current repo. It does not create a new repo
or introduce a new review database.

## Source Of Truth

Authoritative review state:

```text
debug/training_labels/70618_bag1.json
```

SHA-256 at baseline:

```text
fdf5ca0a5cc04787e7d5d7ef699ed3c928d4a4236521b44e80fc17c55879522b
```

This file is the only Bag Review write source of truth. Crop cache, preview
assets, catalog data, CLIP memory, and UI state are read-only inputs or derived
outputs for review purposes.

## Bag 1 Completion

Baseline progress from `bag_review_service.build_review_model("70618", 1)`:

```text
total_slots: 48
confirmed_slots: 47
unknown_slots: 0
ignored_slots: 1
needs_review_slots: 0
percent_complete: 100
next_unreviewed: None
```

Ignored extra slot:

```text
crop_id: p20_s23_c1
slot_index: 2
display: Slot 3: IGNORED
reason: detected extra slot; the crop has two real parts
```

## Clean Review Module

Primary module:

```text
clean/services/bag_review_service.py
```

Public functions:

```text
load_review_state(set_num, bag)
save_review_state(set_num, bag, state)
build_review_model(set_num, bag)
mark_slot_unknown(set_num, bag, crop_id, slot_index)
mark_slot_ignored(set_num, bag, crop_id, slot_index)
save_slot_label(set_num, bag, crop_id, slot_index, part_num, color_id, qty=None)
```

## Route Links

Review UI:

```text
GET /debug/manual-match-review?set_num=70618&bag=1
```

Uses:

```text
bag_review_service.build_review_model()
```

Writes:

```text
none directly
```

Slot label save:

```text
POST /debug/save-label
```

For explicit `selected_slot_index`, delegates to:

```text
bag_review_service.save_slot_label()
```

Unknown slot:

```text
POST /debug/mark-slot-unknown
```

Delegates to:

```text
bag_review_service.mark_slot_unknown()
```

Ignored extra slot:

```text
POST /debug/mark-slot-ignored
```

Delegates to:

```text
bag_review_service.mark_slot_ignored()
```

Top 5 suggestions:

```text
POST /debug/buildability-clip-suggest
```

This is a suggestion service for Bag Review. It must not write authoritative
review state. It may update derived CLIP memory.

## Read-Only Inputs

Crop cache:

```text
debug/crop_cache/70618_bag1.json
```

Preview assets:

```text
debug/ai_training/step_segmented_cutouts/
debug/ai_training/part_cutouts/
```

Catalog/embedding inputs:

```text
debug/catalog_clip_embeddings/items.json
debug/catalog_clip_embeddings/embeddings.npy
debug/server_catalog/
debug/part_image_cache/
```

## Derived Outputs

CLIP memory:

```text
debug/training_labels/70618_bag1_clip_memory.json
```

This is derived from confirmed labels and preview imagery. It is not review
state.

## Verification Commands

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/a2b_pycache .venv/bin/python -m py_compile clean/services/bag_review_service.py clean/routers/instruction_debug.py
.venv/bin/python -c "import clean.main; print('ok')"
.venv/bin/python - <<'PY'
from fastapi.testclient import TestClient
from clean.main import app
client = TestClient(app)
res = client.get('/debug/manual-match-review', params={'set_num': '70618', 'bag': 1})
print(res.status_code)
PY
```

Expected review model check:

```bash
.venv/bin/python - <<'PY'
from clean.services.bag_review_service import build_review_model
m = build_review_model('70618', 1)
print(m['progress'])
PY
```

Expected output:

```text
{'total_slots': 48, 'confirmed_slots': 47, 'unknown_slots': 0, 'ignored_slots': 1, 'needs_review_slots': 0, 'percent_complete': 100, 'next_unreviewed': None}
```

## Flow

```text
PDF / crop detection / crop cache
        ↓
debug/crop_cache/{set_num}_bag{bag}.json
        ↓ read-only
bag_review_service
        ↓ derived review model
/debug/manual-match-review UI
        ↓ user action
/debug/save-label
/debug/mark-slot-unknown
/debug/mark-slot-ignored
        ↓ writes only review state
debug/training_labels/{set_num}_bag{bag}.json
        ↓ derived later
clip memory / training exports / suggestion priors
```
