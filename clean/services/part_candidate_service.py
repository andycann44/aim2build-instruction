import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from clean.services import debug_service


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_set_num(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return text
    if "-" not in text:
        return f"{text}-1"
    return text


def _resolve_catalog_db_path() -> Optional[Path]:
    env_path = str(os.getenv("A2B_CATALOG_DB") or "").strip()
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    candidates = [
        _repo_root() / "backend" / "app" / "data" / "lego_catalog.db",
        _repo_root() / "debug" / "server_catalog" / "lego_catalog.db",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _connect_catalog() -> sqlite3.Connection:
    db_path = _resolve_catalog_db_path()
    if db_path is None:
        raise FileNotFoundError("Could not locate lego_catalog.db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _catalog_map_path(set_num: str) -> Path:
    return debug_service.DEBUG_ROOT / "catalog_maps" / f"{str(set_num or '').strip()}_element_map.json"


def _overview_manifest_path(set_num: str) -> Path:
    return debug_service.DEBUG_ROOT / "part_image_cache" / str(set_num or "").strip() / "manifest.json"


def _hex_to_bgr(rgb_hex: str) -> Tuple[int, int, int]:
    value = str(rgb_hex or "").strip().lstrip("#")
    if len(value) != 6:
        return (0, 0, 0)
    try:
        return (
            int(value[4:6], 16),
            int(value[2:4], 16),
            int(value[0:2], 16),
        )
    except Exception:
        return (0, 0, 0)


def _estimate_crop_bgr(crop_img: Any) -> Tuple[int, int, int]:
    if crop_img is None or getattr(crop_img, "size", 0) == 0:
        return (0, 0, 0)
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # Prefer likely part pixels over pale callout background.
    keep = ((sat >= 42) & (val >= 52)) | ((val >= 68) & (val <= 188))
    pale_blue = (hsv[:, :, 0] >= 82) & (hsv[:, :, 0] <= 132) & (sat <= 140) & (val >= 150)
    near_white = (sat <= 36) & (val >= 205)
    dark_text = (val <= 58) & (sat <= 95)
    keep = keep & ~pale_blue & ~near_white & ~dark_text
    pixels = crop_img[keep]
    if pixels.size == 0:
        pixels = crop_img.reshape(-1, 3)
    med = np.median(pixels.reshape(-1, 3), axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _resolve_local_image_path(img_url: str) -> Optional[Path]:
    text = str(img_url or "").strip()
    if not text:
        return None
    if text.startswith(("http://", "https://")):
        return None
    if text.startswith("file://"):
        text = text[7:]
    path = Path(text)
    if path.exists():
        return path
    repo_path = _repo_root() / text
    if repo_path.exists():
        return repo_path
    return None


def _load_local_candidate_image(img_url: str) -> Optional[Any]:
    path = _resolve_local_image_path(img_url)
    if path is None:
        return None
    img = cv2.imread(str(path))
    return img if img is not None and getattr(img, "size", 0) != 0 else None


def _load_overview_items(set_num: str) -> List[Dict[str, Any]]:
    set_key = str(set_num or "").strip()
    if not set_key:
        return []

    element_map: Dict[str, Dict[str, Any]] = {}
    map_path = _catalog_map_path(set_key)
    if map_path.exists():
        try:
            loaded = json.loads(map_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for element_id, row in loaded.items():
                    if not isinstance(row, dict):
                        continue
                    element_map[str(element_id)] = {
                        "part_num": str(row.get("part_num") or "").strip(),
                        "color_id": int(row.get("color_id", 0) or 0),
                        "qty": int(row.get("qty", 0) or 0),
                    }
        except Exception:
            element_map = {}

    items: List[Dict[str, Any]] = []
    manifest_path = _overview_manifest_path(set_key)
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            loaded = []
        if isinstance(loaded, list):
            for row in loaded:
                if not isinstance(row, dict):
                    continue
                element_id = str(row.get("element_id") or "").strip()
                mapped = element_map.get(element_id, {}) if element_id else {}
                part_num = str(row.get("part_num") or mapped.get("part_num") or "").strip()
                color_id = int(row.get("color_id", mapped.get("color_id", 0)) or 0)
                qty = int(row.get("qty", mapped.get("qty", 0)) or 0)
                local_path = str(row.get("local_path") or "").strip()
                img_url = str(row.get("img_url") or "").strip()
                image_ref = local_path or img_url
                items.append(
                    {
                        "element_id": element_id or None,
                        "part_num": part_num,
                        "color_id": color_id,
                        "qty": qty if qty > 0 else None,
                        "image_ref": image_ref,
                    }
                )

    if items:
        return items

    for element_id, row in element_map.items():
        items.append(
            {
                "element_id": str(element_id or "").strip() or None,
                "part_num": str(row.get("part_num") or "").strip(),
                "color_id": int(row.get("color_id", 0) or 0),
                "qty": int(row.get("qty", 0) or 0) or None,
                "image_ref": "",
            }
        )
    return items


def _color_distance(left_bgr: Tuple[int, int, int], right_bgr: Tuple[int, int, int]) -> float:
    return math.sqrt(sum((float(left_bgr[i]) - float(right_bgr[i])) ** 2 for i in range(3)))


def _compute_visual_similarity(crop_img: Any, candidate_img: Any) -> Tuple[Optional[float], str, Dict[str, float]]:
    if crop_img is None or candidate_img is None:
        return None, "", {}
    try:
        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        cand_gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
        crop_gray = cv2.resize(crop_gray, (96, 96), interpolation=cv2.INTER_AREA)
        cand_gray = cv2.resize(cand_gray, (96, 96), interpolation=cv2.INTER_AREA)
        crop_edges = cv2.Canny(crop_gray, 60, 160)
        cand_edges = cv2.Canny(cand_gray, 60, 160)

        edge_density_crop = float(crop_edges.mean()) / 255.0
        edge_density_cand = float(cand_edges.mean()) / 255.0
        edge_density_score = abs(edge_density_crop - edge_density_cand)

        def _quadrant_hist(edge_img: Any) -> np.ndarray:
            h, w = edge_img.shape[:2]
            quads = [
                edge_img[: h // 2, : w // 2],
                edge_img[: h // 2, w // 2 :],
                edge_img[h // 2 :, : w // 2],
                edge_img[h // 2 :, w // 2 :],
            ]
            return np.array([(float(q.mean()) / 255.0) if q.size else 0.0 for q in quads], dtype=np.float32)

        edge_hist_score = float(np.mean(np.abs(_quadrant_hist(crop_edges) - _quadrant_hist(cand_edges))))

        crop_contours, _ = cv2.findContours(crop_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cand_contours, _ = cv2.findContours(cand_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_density_crop = float(len(crop_contours)) / float(max(1, crop_edges.shape[0] * crop_edges.shape[1]))
        contour_density_cand = float(len(cand_contours)) / float(max(1, cand_edges.shape[0] * cand_edges.shape[1]))
        contour_density_score = abs(contour_density_crop - contour_density_cand) * 250.0

        scores: List[float] = [edge_density_score, edge_hist_score, contour_density_score]
        methods: List[str] = ["edge_density", "edge_histogram", "contour_density"]
        breakdown: Dict[str, float] = {
            "edge_density": round(edge_density_score, 4),
            "edge_histogram": round(edge_hist_score, 4),
            "contour_density": round(contour_density_score, 4),
        }

        orb_factory = getattr(cv2, "ORB_create", None)
        if callable(orb_factory):
            orb = orb_factory(nfeatures=128)
            kp1, des1 = orb.detectAndCompute(crop_gray, None)
            kp2, des2 = orb.detectAndCompute(cand_gray, None)
            if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des1, des2)
                if matches:
                    matches = sorted(matches, key=lambda m: m.distance)[:24]
                    orb_score = float(sum(float(m.distance) for m in matches) / max(1, len(matches))) / 100.0
                    scores.append(orb_score)
                    methods.append("orb")
                    breakdown["orb"] = round(orb_score, 4)

        return float(sum(scores) / max(1, len(scores))), ",".join(methods), breakdown
    except Exception:
        return None, "", {}


def _nearest_color_ids(conn: sqlite3.Connection, crop_bgr: Tuple[int, int, int], limit: int = 6) -> List[int]:
    rows = conn.execute(
        """
        SELECT color_id, name, rgb
        FROM colors
        """
    ).fetchall()
    scored: List[Tuple[float, int]] = []
    for row in rows:
        scored.append((_color_distance(crop_bgr, _hex_to_bgr(str(row["rgb"] or ""))), int(row["color_id"])))
    scored.sort(key=lambda item: item[0])
    return [color_id for _, color_id in scored[: max(1, int(limit))]]


def _metallic_rank(color_name: str) -> int:
    name = str(color_name or "").strip().lower()
    if any(token in name for token in ("chrome", "pearl", "metal", "silver", "gold", "drum")):
        return 0
    return 1


def _load_set_part_pairs(conn: sqlite3.Connection, set_num: str) -> List[Tuple[str, int]]:
    normalized = _normalize_set_num(set_num)
    rows = conn.execute(
        """
        SELECT part_num, color_id
        FROM set_parts
        WHERE set_num = ?
        """,
        (normalized,),
    ).fetchall()
    if rows:
        return [(str(row["part_num"] or ""), int(row["color_id"] or 0)) for row in rows]

    inv = conn.execute(
        """
        SELECT inventory_id
        FROM inventories
        WHERE set_num = ?
        ORDER BY version DESC, inventory_id DESC
        LIMIT 1
        """,
        (normalized,),
    ).fetchone()
    if not inv:
        return []
    inv_rows = conn.execute(
        """
        SELECT part_num, color_id
        FROM inventory_parts
        WHERE inventory_id = ?
          AND COALESCE(is_spare, 0) = 0
        GROUP BY part_num, color_id
        """,
        (int(inv["inventory_id"]),),
    ).fetchall()
    return [(str(row["part_num"] or ""), int(row["color_id"] or 0)) for row in inv_rows]


def _lookup_color_name(conn: sqlite3.Connection, color_id: int) -> str:
    row = conn.execute(
        """
        SELECT name
        FROM colors
        WHERE color_id = ?
        LIMIT 1
        """,
        (int(color_id or 0),),
    ).fetchone()
    return str((row["name"] if row else "") or "")


def _lookup_part_name(conn: sqlite3.Connection, part_num: str) -> str:
    row = conn.execute(
        """
        SELECT name
        FROM parts
        WHERE part_num = ?
        LIMIT 1
        """,
        (str(part_num or "").strip(),),
    ).fetchone()
    return str((row["name"] if row else "") or "")


def _remaining_key(part_num: str, color_id: int) -> str:
    return f"{str(part_num or '').strip()}::{int(color_id or 0)}"


def _apply_remaining_parts_adjustment(
    part_num: str,
    color_id: int,
    score: float,
    score_breakdown: Dict[str, float],
    reasons: List[str],
    remaining_parts: Optional[Dict[str, Dict[str, Any]]],
    hide_depleted: bool = False,
) -> Tuple[float, Optional[int], bool]:
    if not remaining_parts:
        return score, None, False
    row = remaining_parts.get(_remaining_key(part_num, color_id))
    if not isinstance(row, dict):
        score_breakdown["remaining_adjustment"] = 0.0
        return score, None, False
    remaining_qty = int(row.get("remaining_qty", 0) or 0)
    if remaining_qty > 0:
        score -= 30.0
        reasons.append("remaining_stock")
        score_breakdown["remaining_adjustment"] = -30.0
        return score, remaining_qty, False
    if hide_depleted:
        return score, remaining_qty, True
    score += 90.0
    reasons.append("depleted_stock")
    score_breakdown["remaining_adjustment"] = 90.0
    return score, remaining_qty, False


def get_part_candidates_for_crop(
    crop_image_path: str,
    max_candidates: int = 20,
    color_ids: Optional[List[int]] = None,
    metallic_mode: bool = False,
    set_num: Optional[str] = None,
    remaining_parts: Optional[Dict[str, Dict[str, Any]]] = None,
    hide_depleted: bool = False,
) -> List[Dict[str, Any]]:
    image_path = Path(str(crop_image_path or "").strip())
    if not image_path.exists():
        return []

    crop_img = cv2.imread(str(image_path))
    if crop_img is None:
        return []

    conn = _connect_catalog()
    try:
        resolved_color_ids = [int(color_id) for color_id in list(color_ids or []) if color_id is not None]
        crop_bgr = _estimate_crop_bgr(crop_img)
        if not resolved_color_ids:
            resolved_color_ids = _nearest_color_ids(conn, crop_bgr, limit=6)
        set_pairs = set(_load_set_part_pairs(conn, str(set_num or "").strip())) if set_num else set()
        overview_items = _load_overview_items(str(set_num or "").strip()) if set_num else []

        overview_results: List[Dict[str, Any]] = []
        for item in overview_items:
            part_num = str(item.get("part_num") or "").strip()
            color_id = int(item.get("color_id", 0) or 0)
            if not part_num:
                continue
            if color_ids and color_id not in resolved_color_ids:
                continue
            candidate_img = _load_local_candidate_image(str(item.get("image_ref") or ""))
            if candidate_img is None:
                continue
            visual_score, visual_method, visual_breakdown = _compute_visual_similarity(crop_img, candidate_img)
            if visual_score is None:
                continue
            score = float(visual_score * 12.0)
            score_breakdown: Dict[str, float] = {
                "overview_visual_weighted": round(score, 3),
            }
            for key, value in visual_breakdown.items():
                score_breakdown[f"visual_{key}"] = value
            if color_ids and color_id in resolved_color_ids:
                score -= 20.0
                score_breakdown["color_bonus"] = -20.0
            if set_pairs and (part_num, color_id) in set_pairs:
                score -= 15.0
                score_breakdown["set_bonus"] = -15.0
            overview_reasons: List[str] = ["overview_match", "visual_local"]
            score, remaining_qty, skip_candidate = _apply_remaining_parts_adjustment(
                part_num,
                color_id,
                score,
                score_breakdown,
                overview_reasons,
                remaining_parts,
                hide_depleted=hide_depleted,
            )
            if skip_candidate:
                continue
            score_breakdown["final"] = round(score, 3)
            overview_results.append(
                {
                    "part_num": part_num,
                    "color_id": color_id,
                    "part_name": _lookup_part_name(conn, part_num),
                    "color_name": _lookup_color_name(conn, color_id),
                    "img_url": str(item.get("image_ref") or ""),
                    "score": float(round(score, 3)),
                    "visual_score": float(round(visual_score, 4)),
                    "visual_method": visual_method,
                    "score_breakdown": score_breakdown,
                    "reason": ",".join(overview_reasons),
                    "candidate_source": "overview",
                    "element_id": item.get("element_id"),
                    "overview_score": float(round(visual_score, 4)),
                    "remaining_qty": remaining_qty,
                }
            )

        placeholders = ",".join("?" for _ in resolved_color_ids) if resolved_color_ids else ""
        params: List[Any] = list(resolved_color_ids)
        sql = """
            SELECT ei.part_num, ei.color_id, ei.img_url, p.name AS part_name, c.name AS color_name, c.rgb
            FROM element_images ei
            JOIN parts p ON p.part_num = ei.part_num
            JOIN colors c ON c.color_id = ei.color_id
        """
        if resolved_color_ids:
            sql += f" WHERE ei.color_id IN ({placeholders})"
        sql += " ORDER BY ei.part_num, ei.color_id LIMIT 600"
        rows = conn.execute(sql, params).fetchall()

        all_results: List[Dict[str, Any]] = list(overview_results)
        seen_keys = {
            (str(item.get("part_num") or ""), int(item.get("color_id", 0) or 0))
            for item in overview_results
        }
        for row in rows:
            candidate_bgr = _hex_to_bgr(str(row["rgb"] or ""))
            distance = _color_distance(crop_bgr, candidate_bgr)
            score = distance
            reasons: List[str] = []
            score_breakdown: Dict[str, float] = {"color_distance": round(distance, 3)}
            pair = (str(row["part_num"] or ""), int(row["color_id"] or 0))
            if set_pairs:
                if pair in set_pairs:
                    score -= 60.0
                    reasons.append("set_match")
                    score_breakdown["set_bonus"] = -60.0
                    candidate_source = "set_parts"
                else:
                    reasons.append("catalog_fallback")
                    score_breakdown["set_bonus"] = 0.0
                    candidate_source = "catalog"
            else:
                candidate_source = "catalog"
            if color_ids:
                if int(row["color_id"]) in resolved_color_ids:
                    score -= 40.0
                    reasons.append("color_match")
                    score_breakdown["color_bonus"] = -40.0
                else:
                    reasons.append("color_mismatch")
                    score_breakdown["color_bonus"] = 0.0
            else:
                reasons.append("nearest_color")
                score_breakdown["color_bonus"] = 0.0
            img_url = str(row["img_url"] or "")
            if img_url:
                score -= 5.0
                reasons.append("image_available")
                score_breakdown["image_bonus"] = -5.0
            else:
                reasons.append("no_image")
                score_breakdown["image_bonus"] = 0.0
            if metallic_mode:
                if _metallic_rank(str(row["color_name"] or "")) == 0:
                    score -= 8.0
                    reasons.append("metallic_pref")
                    score_breakdown["metallic_bonus"] = -8.0
                else:
                    score += 4.0
                    reasons.append("non_metallic_penalty")
                    score_breakdown["metallic_bonus"] = 4.0
            visual_score = None
            visual_method = ""
            candidate_img = _load_local_candidate_image(img_url)
            if candidate_img is not None:
                visual_score, visual_method, visual_breakdown = _compute_visual_similarity(crop_img, candidate_img)
                if visual_score is not None:
                    score += visual_score * 25.0
                    reasons.append("visual_local")
                    score_breakdown["visual_weighted"] = round(visual_score * 25.0, 3)
                    for key, value in visual_breakdown.items():
                        score_breakdown[f"visual_{key}"] = value
            score, remaining_qty, skip_candidate = _apply_remaining_parts_adjustment(
                str(row["part_num"] or ""),
                int(row["color_id"] or 0),
                score,
                score_breakdown,
                reasons,
                remaining_parts,
                hide_depleted=hide_depleted,
            )
            if skip_candidate:
                continue
            score_breakdown["final"] = round(score, 3)
            if pair in seen_keys:
                continue
            all_results.append(
                {
                    "part_num": str(row["part_num"] or ""),
                    "color_id": int(row["color_id"] or 0),
                    "part_name": str(row["part_name"] or ""),
                    "color_name": str(row["color_name"] or ""),
                    "img_url": img_url,
                    "score": float(round(score, 3)),
                    "visual_score": None if visual_score is None else float(round(visual_score, 4)),
                    "visual_method": visual_method,
                    "score_breakdown": score_breakdown,
                    "reason": ",".join(reasons),
                    "candidate_source": candidate_source,
                    "element_id": None,
                    "overview_score": None,
                    "remaining_qty": remaining_qty,
                }
            )

        all_results.sort(key=lambda item: (float(item["score"]), str(item["part_num"]), int(item["color_id"])))
        if set_pairs:
            set_results = [item for item in all_results if (str(item["part_num"]), int(item["color_id"])) in set_pairs]
            min_set_results = min(max(3, int(max_candidates) // 2), int(max_candidates))
            if len(set_results) >= min_set_results:
                return set_results[: max(1, int(max_candidates))]
        return all_results[: max(1, int(max_candidates))]
    finally:
        conn.close()


def _main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python3 clean/services/part_candidate_service.py <crop_image_path>")
        return 2
    results = get_part_candidates_for_crop(argv[1])
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
