from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from clean.services.training_bundle_index_service import (
    _connect,
    ensure_schema,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_EMB_DIR = _REPO_ROOT / "debug" / "clip_training_embeddings"
_MEMORY_DIR = _REPO_ROOT / "debug" / "confirmed_memory"

_MEMORY_EMBEDDINGS = _MEMORY_DIR / "confirmed_memory_embeddings.npy"
_MEMORY_ITEMS = _MEMORY_DIR / "confirmed_memory_items.json"
_MEMORY_MANIFEST = _MEMORY_DIR / "manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_confirmed_rows() -> List[Dict[str, Any]]:
    ensure_schema()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.element_id,
                    e.part_num,
                    e.color_id,
                    e.bundle_id,
                    e.candidate_index,
                    e.qty,
                    e.confirmed_by,
                    e.confirmed_at,
                    b.set_num
                FROM candidate_training_examples e
                JOIN training_bundle_index b ON b.bundle_id = e.bundle_id
                WHERE COALESCE(e.element_id, '') != ''
                ORDER BY e.confirmed_at ASC;
                """
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def build_confirmed_memory_index() -> Dict[str, Any]:
    """Build the confirmed-part memory index using Design A (element_id join).

    Joins Postgres confirmed examples against the existing training-pack
    CLIP embeddings via element_id.  No re-embedding is performed.

    Writes three files to debug/confirmed_memory/:
        confirmed_memory_embeddings.npy  — float32 (M, 512) L2-normalised
        confirmed_memory_items.json      — index-aligned metadata
        manifest.json                    — build summary
    """
    items_path = _TRAINING_EMB_DIR / "items.json"
    emb_path = _TRAINING_EMB_DIR / "embeddings.npy"

    if not items_path.exists():
        raise FileNotFoundError(f"items.json not found: {items_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.npy not found: {emb_path}")

    training_items: List[Dict[str, Any]] = json.loads(items_path.read_text(encoding="utf-8"))
    training_embeddings: np.ndarray = np.load(str(emb_path))

    # element_id → list of row indices in the training embedding matrix
    # An element_id may appear on multiple pages; keep all occurrences.
    element_id_to_indices: Dict[str, List[int]] = {}
    for item in training_items:
        eid = str(item.get("element_id") or "").strip()
        if not eid:
            continue
        idx = int(item.get("index", 0))
        element_id_to_indices.setdefault(eid, []).append(idx)

    confirmed_rows = _load_confirmed_rows()

    # Deduplicate confirmed label by element_id, accumulating part/color variants
    # and counting how many human confirmations back each element_id.
    seen_element_ids: Dict[str, Dict[str, Any]] = {}
    for row in confirmed_rows:
        eid = str(row.get("element_id") or "").strip()
        if not eid:
            continue
        if eid not in seen_element_ids:
            seen_element_ids[eid] = {
                "element_id": eid,
                "part_num": str(row.get("part_num") or ""),
                "color_id": row.get("color_id"),
                "set_num": str(row.get("set_num") or ""),
                "confirmed_count": 0,
                "confirmed_by": [],
                "bundle_ids": [],
            }
        entry = seen_element_ids[eid]
        entry["confirmed_count"] += 1
        confirmed_by = str(row.get("confirmed_by") or "")
        if confirmed_by and confirmed_by not in entry["confirmed_by"]:
            entry["confirmed_by"].append(confirmed_by)
        bundle_id = str(row.get("bundle_id") or "")
        if bundle_id and bundle_id not in entry["bundle_ids"]:
            entry["bundle_ids"].append(bundle_id)

    matched_element_ids: List[str] = []
    unmatched_element_ids: List[str] = []
    memory_embedding_rows: List[np.ndarray] = []
    memory_items: List[Dict[str, Any]] = []

    for eid, label in seen_element_ids.items():
        training_indices = element_id_to_indices.get(eid)
        if not training_indices:
            unmatched_element_ids.append(eid)
            continue
        matched_element_ids.append(eid)
        for origin_index in training_indices:
            if origin_index >= len(training_embeddings):
                continue
            vec = training_embeddings[origin_index]
            memory_embedding_rows.append(vec)
            memory_items.append(
                {
                    "element_id": eid,
                    "part_num": label["part_num"],
                    "color_id": label["color_id"],
                    "set_num": label["set_num"],
                    "confirmed_count": label["confirmed_count"],
                    "confirmed_by": label["confirmed_by"],
                    "bundle_ids": label["bundle_ids"],
                    "origin_embedding_index": origin_index,
                }
            )

    if not memory_embedding_rows:
        memory_matrix = np.zeros((0, training_embeddings.shape[1]), dtype=np.float32)
    else:
        raw = np.stack(memory_embedding_rows, axis=0).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        memory_matrix = raw / norms

    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    np.save(str(_MEMORY_EMBEDDINGS), memory_matrix)
    _MEMORY_ITEMS.write_text(
        json.dumps(memory_items, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    unique_confirmed = len(seen_element_ids)
    duplicate_rows = len(confirmed_rows) - unique_confirmed
    matched_count = len(matched_element_ids)
    coverage_ratio = round(matched_count / unique_confirmed, 4) if unique_confirmed else 0.0

    manifest = {
        "generated_at": _utc_now(),
        "confirmed_rows": len(confirmed_rows),
        "unique_confirmed_element_ids": unique_confirmed,
        "duplicate_confirmed_rows": duplicate_rows,
        "matched_element_ids": sorted(matched_element_ids),
        "unmatched_element_ids": sorted(unmatched_element_ids),
        "coverage_ratio": coverage_ratio,
        "memory_embedding_shape": list(memory_matrix.shape),
        "training_embeddings_source": str(_TRAINING_EMB_DIR),
        "output_dir": str(_MEMORY_DIR),
    }
    _MEMORY_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "confirmed_rows": len(confirmed_rows),
        "unique_confirmed_element_ids": unique_confirmed,
        "duplicate_confirmed_rows": duplicate_rows,
        "matched_element_ids": sorted(matched_element_ids),
        "unmatched_element_ids": sorted(unmatched_element_ids),
        "coverage_ratio": coverage_ratio,
        "memory_embedding_shape": list(memory_matrix.shape),
        "manifest_path": str(_MEMORY_MANIFEST),
    }


def load_confirmed_memory_index() -> Dict[str, Any]:
    """Load the confirmed-part memory index from disk.

    Returns a dict with:
        embeddings  — np.ndarray (M, 512) float32, or empty array if not built
        items       — list of dicts, index-aligned with embeddings
        manifest    — the build manifest dict
        ok          — bool
    """
    if not _MEMORY_EMBEDDINGS.exists():
        return {
            "ok": False,
            "reason": "index_not_built",
            "embeddings": np.zeros((0, 512), dtype=np.float32),
            "items": [],
            "manifest": {},
        }
    embeddings = np.load(str(_MEMORY_EMBEDDINGS))
    items: List[Dict[str, Any]] = []
    if _MEMORY_ITEMS.exists():
        raw = json.loads(_MEMORY_ITEMS.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            items = raw
    manifest: Dict[str, Any] = {}
    if _MEMORY_MANIFEST.exists():
        raw_m = json.loads(_MEMORY_MANIFEST.read_text(encoding="utf-8"))
        if isinstance(raw_m, dict):
            manifest = raw_m
    return {
        "ok": True,
        "embeddings": embeddings,
        "items": items,
        "manifest": manifest,
    }


def score_confirmed_memory(
    query_vec: np.ndarray,
    index: Dict[str, Any],
    *,
    threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    """Score a query embedding against the confirmed-part memory index.

    Args:
        query_vec:  float32 array of shape (512,), should be L2-normalised.
        index:      result of load_confirmed_memory_index().
        threshold:  minimum cosine similarity to include in results.

    Returns:
        List of dicts sorted by score descending, each containing:
            element_id, part_num, color_id, score, confirmed_count, confirmed_by
        Only entries with score >= threshold are returned.
        Empty list if the index is empty or not built.
    """
    embeddings: np.ndarray = index.get("embeddings", np.zeros((0, 512), dtype=np.float32))
    items: List[Dict[str, Any]] = index.get("items", [])

    if embeddings.shape[0] == 0 or not items:
        return []

    q = query_vec.astype(np.float32).flatten()
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return []
    q = q / norm

    # Cosine similarity: both query and memory embeddings are L2-normalised
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scores: np.ndarray = embeddings @ q  # shape (M,)

    # Aggregate to element_id level: take the max score across multiple
    # embedding rows for the same element_id (different pages).
    best_by_element: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(items):
        if i >= len(scores):
            break
        s = float(scores[i])
        eid = str(item.get("element_id") or "")
        if not eid:
            continue
        if eid not in best_by_element or s > best_by_element[eid]["score"]:
            best_by_element[eid] = {
                "element_id": eid,
                "part_num": str(item.get("part_num") or ""),
                "color_id": item.get("color_id"),
                "set_num": str(item.get("set_num") or ""),
                "score": s,
                "confirmed_count": int(item.get("confirmed_count") or 0),
                "confirmed_by": list(item.get("confirmed_by") or []),
            }

    results = [
        entry
        for entry in best_by_element.values()
        if entry["score"] >= threshold
    ]
    results.sort(key=lambda e: e["score"], reverse=True)
    return results
