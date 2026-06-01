# Decisions

This document records the current project decisions. Future AI agents and
developers must follow these decisions unless they are explicitly updated here.

---

## D1 — Bag Review First

Human bag review is the first mandatory step before any training data is
exported or used for model training.

No training bundle should be created from unreviewed bag data.

---

## D2 — Human-Reviewed Labels Are Authoritative

AI-generated suggestions, CLIP matches, and auto-detect results are inputs to
the reviewer's decision. They do not become authoritative until a human confirms
them via the Bag Review UI.

The human confirmation action (save-label, mark-slot-unknown, mark-slot-ignored)
is what makes a label authoritative.

---

## D3 — Review State Stored Only in `training_labels` JSON

The Bag Review source of truth is:

```
debug/training_labels/{set_num}_bag{bag}.json
```

Review state is not stored in:
- SQLite databases
- Memory caches
- The crop cache
- Any derived embedding file
- Any other JSON file

There is no duplicate store of review state.

---

## D4 — CLIP Memory Is Derived

`debug/training_labels/{set_num}_bag{bag}_clip_memory.json` is a derived
cache. It is built from confirmed labels and preview imagery. It can be deleted
and regenerated at any time without loss of authoritative data.

It must not be edited by hand. It must not be used to recover review state.

---

## D5 — Catalog Embeddings Are Derived

`debug/catalog_clip_embeddings/` contains pre-computed CLIP embeddings for the
LEGO part catalog. These are derived from the catalog DB and part images. They
can be regenerated via `POST /debug/generate-catalog-clip-embeddings`.

---

## D6 — Crop Cache Is Input-Only

`debug/crop_cache/{set_num}_bag{bag}.json` is written by the crop detection
pipeline and is read-only from the perspective of the Bag Review service.

The Bag Review service must never write to the crop cache.

---

## D7 — No Duplicate Review State

Review state exists in exactly one place per bag:

```
debug/training_labels/{set_num}_bag{bag}.json
```

If a future feature requires a second representation of review state (e.g. a
database), it must be derived from this file, not used as an alternative source
of truth.

---

## D8 — One Source of Truth Per Concern

Each concern in the system has exactly one authoritative file or store. See
`SOURCE_OF_TRUTH.md` for the full registry. No two stores may hold conflicting
authoritative state for the same concern.

---

## D9 — Bag 1 Baseline Is Complete and Verified

Set `70618`, Bag 1 has been reviewed and verified at baseline (2026-06-01).

Progress at baseline:

```
total_slots:       48
confirmed_slots:   47
unknown_slots:      0
ignored_slots:      1
needs_review_slots: 0
percent_complete:  100
```

Verified SHA-256 of `debug/training_labels/70618_bag1.json`:

```
fdf5ca0a5cc04787e7d5d7ef699ed3c928d4a4236521b44e80fc17c55879522b
```

The ignored slot (`p20_s23_c1`, slot_index 2) was explicitly reviewed and
marked ignored: detected extra slot; the crop has two real parts.

---

## D10 — Future Bags Follow the Same Workflow

Bags 2, 3, 4, and all future bags for any set follow the identical workflow:

1. Run crop generation to produce `debug/crop_cache/{set_num}_bag{bag}.json`
2. Review all slots via `GET /debug/manual-match-review?set_num=...&bag=...`
3. Save labels via `POST /debug/save-label`
4. Mark unknowns and ignored slots as appropriate
5. Verify 100% completion via `build_review_model()` progress output
6. Update `BAG_REVIEW_BASELINE.md` with new SHA-256 and progress stats
7. Proceed to training export

---

## D11 — Bag Review Slot Operations Use `bag_review_service`

Bag Review slot-state actions write via `bag_review_service.save_review_state()`:

- `save-label` when `selected_slot_index >= 0` → `bag_review_service.save_slot_label()`
- `mark-slot-unknown` → `bag_review_service.mark_slot_unknown()`
- `mark-slot-ignored` → `bag_review_service.mark_slot_ignored()`

Legacy and debug mutation routes still write directly via `_write_labels()` in
`instruction_debug.py` (atomic: temp file + replace):

- `save-label` (non-slot path, when `selected_slot_index` is absent)
- `remove-label`
- `set-crop-status`
- `delete-crop`
- `save-manual-crop`
- `update-crop-qty`

Both paths write to the same training labels JSON. This is the current truth.
It is not ideal — see D14.

---

## D12 — Suggestions Must Not Pollute Review State

The suggestion pipeline (`buildability-clip-suggest`) reads review state and
may update the CLIP memory cache. It must never write to the training labels
JSON. If a suggestion route is ever modified, this constraint must be preserved.

---

## D13 — Schema Version

The current schema version for training label files is `"1.1"`. Any change to
the schema must bump the version and be documented here.

---

## D14 — Future: Unify All Training Labels Writes

All writes to `debug/training_labels/{set_num}_bag{bag}.json` should eventually
route through `bag_review_service` or a single unified label store service.

Currently `_write_labels()` in `instruction_debug.py` is a second write path.
It must not grow further. No new write paths may be added outside of
`bag_review_service`.

This unification is a future cleanup task. It is not current state.
