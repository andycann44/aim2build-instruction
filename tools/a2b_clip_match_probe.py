#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional, TypeVar
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    torch = None
    CLIPModel = None
    CLIPProcessor = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]

# Change these paths in-place for whichever local sample you want to probe first.
CLIP_MODEL_NAME_OR_PATH = "openai/clip-vit-base-patch32"
CLIP_LOCAL_FILES_ONLY = True
MAX_CROPS = 15
MAX_CATALOG_IMAGES = 120
TOP_K_MATCHES = 10
BATCH_SIZE = 8
EVAL_SET_NUM = "70618"
EVAL_BAGS = (1, 2)

PRENORMALIZED_CROP_GLOBS = [
    str(REPO_ROOT / "training_data_crops" / "**" / "*normalized*.png"),
    str(REPO_ROOT / "debug" / "**" / "*slot_normalized*.png"),
    str(REPO_ROOT / "debug" / "**" / "*normalized*.png"),
]

TRAINING_LABEL_JSON_GLOBS = [
    str(REPO_ROOT / "debug" / "training_labels" / "*.json"),
]

CATALOG_MANIFEST_GLOBS = [
    str(REPO_ROOT / "debug" / "part_image_cache" / "*" / "manifest.json"),
]

APP_CATALOG_DB_PATH = Path("~/aim2build-app-v2/backend/app/data/lego_catalog.db").expanduser()
CLIP_CATALOG_CACHE_DIR = REPO_ROOT / "debug" / "clip_catalog_cache"
CLIP_PROBE_CROP_DIR = REPO_ROOT / "debug" / "clip_probe_crops"
CLIP_PROBE_REPORT_DIR = REPO_ROOT / "debug" / "clip_probe_reports"
MAX_CATALOG_DB_IMAGES = 200

CATALOG_IMAGE_DIR_CANDIDATES = [
    CLIP_CATALOG_CACHE_DIR,
    REPO_ROOT / "debug" / "part_image_cache",
    REPO_ROOT / "debug" / "catalog_part_images",
    REPO_ROOT / "debug" / "element_images",
    REPO_ROOT / "debug" / "part_image_cache" / "70618",  # Bag 1 images
]
# TODO: Confirm the real local mirror path for catalog part images.
# In this checkout, `debug/part_image_cache/<set>/manifest.json` points at remote
# URLs and `local_path` is null, so CLIP matching only works if matching images
# already exist under one of `CATALOG_IMAGE_DIR_CANDIDATES`.

CATALOG_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
T = TypeVar("T")


@dataclass(frozen=True)
class ProbeImage:
    item_id: str
    image_path: Optional[Path]
    pil_image: Image.Image
    summary: str
    sort_key: tuple[Any, ...]


@dataclass(frozen=True)
class CatalogImage:
    item_id: str
    image_path: Path
    pil_image: Image.Image
    part_num: str
    color_id: Optional[int]
    qty: Optional[int]
    summary: str
    sort_key: tuple[Any, ...]


def _expand_globs(patterns: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(REPO_ROOT.glob(str(Path(pattern).relative_to(REPO_ROOT)))):
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            found.append(path)
    return found


def _resolve_local_path(raw_path: Any) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    if text.startswith("file://"):
        text = text[7:]
    direct = Path(text).expanduser()
    if direct.exists():
        return direct
    relative = REPO_ROOT / text.lstrip("/")
    if relative.exists():
        return relative
    return None


def _load_rgb_image(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception:
        return None


def _safe_cache_token(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return "unknown"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return safe or "unknown"


def _catalog_db_image_rows(limit: int) -> list[dict[str, Any]]:
    db_path = APP_CATALOG_DB_PATH
    if not db_path.exists():
        return []

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT part_num, color_id, img_url
            FROM element_images
            WHERE TRIM(COALESCE(img_url, '')) <> ''
            ORDER BY part_num, color_id
            LIMIT ?
            """,
            [max(1, min(int(limit), MAX_CATALOG_DB_IMAGES))],
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _download_catalog_image_to_cache(
    part_num: str,
    color_id: Optional[int],
    img_url: str,
) -> Optional[Path]:
    normalized_part_num = str(part_num or "").strip()
    normalized_url = str(img_url or "").strip()
    if not normalized_part_num or not normalized_url:
        return None

    color_label = "none" if color_id is None else str(int(color_id))
    target_path = CLIP_CATALOG_CACHE_DIR / f"{_safe_cache_token(normalized_part_num)}_{_safe_cache_token(color_label)}.png"
    if target_path.exists() and target_path.is_file():
        return target_path

    CLIP_CATALOG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        request = Request(normalized_url, headers={"User-Agent": "a2b-clip-match-probe/1.0"})
        with urlopen(request, timeout=20) as response:
            raw = response.read()
        with Image.open(BytesIO(raw)) as image:
            if image.mode in {"RGBA", "LA"} or ("transparency" in image.info):
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                composited = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                composited = image.convert("RGB")
            composited.save(target_path, format="PNG")
        return target_path
    except Exception:
        return None


def _load_catalog_db_cache_images(limit: int) -> list[CatalogImage]:
    rows = _catalog_db_image_rows(limit)
    loaded: list[CatalogImage] = []
    seen_paths: set[Path] = set()
    for row in rows:
        part_num = str(row.get("part_num") or "").strip()
        raw_color_id = row.get("color_id")
        img_url = str(row.get("img_url") or "").strip()
        if not part_num or not img_url:
            continue
        try:
            color_id = int(raw_color_id) if raw_color_id is not None else None
        except (TypeError, ValueError):
            color_id = None
        local_path = _download_catalog_image_to_cache(part_num, color_id, img_url)
        if local_path is None or not local_path.exists():
            continue
        resolved = local_path.resolve()
        if resolved in seen_paths:
            continue
        image = _load_rgb_image(local_path)
        if image is None:
            continue
        seen_paths.add(resolved)
        loaded.append(
            CatalogImage(
                item_id=local_path.stem,
                image_path=local_path,
                pil_image=image,
                part_num=part_num,
                color_id=color_id,
                qty=None,
                summary=f"catalog db image: part={part_num} color={color_id} path={local_path}",
                sort_key=(-1, part_num, color_id if color_id is not None else -1, str(local_path)),
            )
        )
        if len(loaded) >= limit:
            break
    return loaded


def _catalog_db_image_url_for_pair(part_num: str, color_id: int) -> Optional[str]:
    db_path = APP_CATALOG_DB_PATH
    if not db_path.exists():
        return None

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT img_url
            FROM element_images
            WHERE part_num = ? AND color_id = ? AND TRIM(COALESCE(img_url, '')) <> ''
            ORDER BY img_url
            LIMIT 1
            """,
            [str(part_num or "").strip(), int(color_id)],
        ).fetchone()
        if row is None:
            return None
        return str(row["img_url"] or "").strip() or None
    finally:
        conn.close()


def _ensure_catalog_image_for_pair(part_num: str, color_id: int) -> Optional[Path]:
    cached_path = CLIP_CATALOG_CACHE_DIR / f"{_safe_cache_token(part_num)}_{_safe_cache_token(color_id)}.png"
    if cached_path.exists() and cached_path.is_file():
        return cached_path

    for base_dir in CATALOG_IMAGE_DIR_CANDIDATES:
        for candidate in (
            base_dir / f"{part_num}_{color_id}.png",
            base_dir / f"{part_num}_{color_id}.jpg",
            base_dir / f"{part_num}_{color_id}.jpeg",
            base_dir / f"{part_num}_{color_id}.webp",
        ):
            if candidate.exists() and candidate.is_file():
                return candidate

    img_url = _catalog_db_image_url_for_pair(part_num, color_id)
    if not img_url:
        return None
    return _download_catalog_image_to_cache(part_num, int(color_id), img_url)


def _foreground_mask_from_instruction_crop(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bgr = img_bgr

    pale_blue_mask = (
        (hsv[:, :, 0] >= 80)
        & (hsv[:, :, 0] <= 130)
        & (hsv[:, :, 1] <= 150)
        & (hsv[:, :, 2] >= 135)
    )
    extra_pale_blue_mask = (
        (bgr[:, :, 0] >= 150)
        & (bgr[:, :, 1] >= 150)
        & (bgr[:, :, 2] >= 130)
        & ((bgr[:, :, 0].astype(np.int16) - bgr[:, :, 2].astype(np.int16)) >= 8)
    )
    fg_mask = (~(pale_blue_mask | extra_pale_blue_mask)).astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return fg_mask


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    image_area = int(mask.shape[0] * mask.shape[1])
    min_area = max(40, int(image_area * 0.0025))
    component_mask = np.zeros_like(mask)
    for label in range(1, int(num_labels)):
        area = int(stats[label, cv2.CC_STAT_AREA] or 0)
        if area < min_area:
            continue
        component_mask[labels == label] = 255
    if int(component_mask.max()) == 0:
        return mask
    return component_mask


def _tight_box_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0, 0, int(mask.shape[1]), int(mask.shape[0])]
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _padded_box(box: list[int], width: int, height: int, pad_ratio: float = 0.12, min_pad: int = 8) -> list[int]:
    x, y, w, h = [int(value) for value in box]
    pad = max(int(min_pad), int(round(max(w, h) * float(pad_ratio))))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def _normalize_crop_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    mask = _largest_component_mask(_foreground_mask_from_instruction_crop(crop_bgr))
    x, y, w, h = _padded_box(_tight_box_from_mask(mask), crop_bgr.shape[1], crop_bgr.shape[0])
    crop = crop_bgr[y : y + h, x : x + w]
    crop_mask = mask[y : y + h, x : x + w]

    masked_crop = np.full_like(crop, 255)
    masked_crop[crop_mask > 0] = crop[crop_mask > 0]

    side = max(masked_crop.shape[0], masked_crop.shape[1], 96)
    square = np.full((side, side, 3), 255, dtype=np.uint8)
    offset_y = max(0, (side - masked_crop.shape[0]) // 2)
    offset_x = max(0, (side - masked_crop.shape[1]) // 2)
    square[offset_y : offset_y + masked_crop.shape[0], offset_x : offset_x + masked_crop.shape[1]] = masked_crop
    return square


def _crop_box_from_page(page_path: Path, crop_box: list[Any]) -> Optional[Image.Image]:
    if len(crop_box) != 4:
        return None
    page_bgr = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if page_bgr is None or getattr(page_bgr, "size", 0) == 0:
        return None

    try:
        x, y, w, h = [int(value) for value in crop_box]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(int(page_bgr.shape[1]), x1 + w)
    y2 = min(int(page_bgr.shape[0]), y1 + h)
    if x2 <= x1 or y2 <= y1:
        return None

    crop_bgr = page_bgr[y1:y2, x1:x2]
    normalized_bgr = _normalize_crop_bgr(crop_bgr)
    rgb = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _extract_crop_region_to_file(page_path: Path, crop_box: list[Any], crop_id: str) -> Optional[Path]:
    if len(crop_box) != 4:
        return None
    page_bgr = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if page_bgr is None or getattr(page_bgr, "size", 0) == 0:
        return None

    try:
        x, y, w, h = [int(value) for value in crop_box]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(int(page_bgr.shape[1]), x1 + w)
    y2 = min(int(page_bgr.shape[0]), y1 + h)
    if x2 <= x1 or y2 <= y1:
        return None

    crop_bgr = page_bgr[y1:y2, x1:x2]
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None

    CLIP_PROBE_CROP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLIP_PROBE_CROP_DIR / f"{str(crop_id or 'crop').strip() or 'crop'}.png"
    if not cv2.imwrite(str(out_path), crop_bgr):
        return None
    return out_path


def _load_pre_normalized_crops(limit: int) -> list[ProbeImage]:
    loaded: list[ProbeImage] = []
    for path in _expand_globs(PRENORMALIZED_CROP_GLOBS):
        image = _load_rgb_image(path)
        if image is None:
            continue
        loaded.append(
            ProbeImage(
                item_id=path.stem,
                image_path=path,
                pil_image=image,
                summary=f"pre-normalized file: {path}",
                sort_key=(0, str(path)),
            )
        )
        if len(loaded) >= limit:
            break
    return loaded


def _training_status_rank(status: str) -> int:
    normalized = status.strip().lower()
    if normalized == "good":
        return 0
    if normalized == "reviewed":
        return 1
    if normalized == "hidden":
        return 2
    return 3


def _load_training_label_crops(limit: int) -> list[ProbeImage]:
    candidates: list[ProbeImage] = []
    for path in _expand_globs(TRAINING_LABEL_JSON_GLOBS):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        set_num = str(payload.get("set_num") or "").strip() or "unknown"
        bag = payload.get("bag")
        crops = payload.get("crops") or {}
        if not isinstance(crops, dict):
            continue

        for crop_id, row in sorted(crops.items()):
            if not isinstance(row, dict):
                continue
            crop_box = row.get("crop_box")
            if not isinstance(crop_box, list):
                continue
            page_path = _resolve_local_path(row.get("crop_image_path"))
            if page_path is None:
                continue
            image = _crop_box_from_page(page_path, crop_box)
            if image is None:
                continue
            page = int(row.get("page", 0) or 0)
            step = int(row.get("step", 0) or 0)
            status = str(row.get("status") or "").strip()
            bag_label = f"bag{bag}" if bag not in (None, "") else "bag?"
            summary = (
                f"training label crop: set={set_num} {bag_label} crop_id={crop_id} "
                f"page={page} step={step} status={status or 'unknown'} page_image={page_path}"
            )
            candidates.append(
                ProbeImage(
                    item_id=f"{set_num}_{bag_label}_{crop_id}",
                    image_path=page_path,
                    pil_image=image,
                    summary=summary,
                    sort_key=(
                        1,
                        _training_status_rank(status),
                        set_num,
                        int(bag or 0) if str(bag or "").isdigit() else 0,
                        page,
                        step,
                        str(crop_id),
                    ),
                )
            )

    candidates.sort(key=lambda item: item.sort_key)
    return candidates[:limit]


def _scan_catalog_image_pool() -> tuple[dict[str, Path], list[Path]]:
    stem_index: dict[str, Path] = {}
    ordered_paths: list[Path] = []
    seen_paths: set[Path] = set()

    for base_dir in CATALOG_IMAGE_DIR_CANDIDATES:
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in CATALOG_IMAGE_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            ordered_paths.append(path)
            stem_index.setdefault(path.stem.lower(), path)
            stem_index.setdefault(path.name.lower(), path)
    return stem_index, ordered_paths


def _manifest_lookup_keys(part_num: str, color_id: Optional[int]) -> list[str]:
    keys = [part_num.lower()]
    if color_id is not None:
        keys.extend(
            [
                f"{part_num}__{color_id}".lower(),
                f"{part_num}_{color_id}".lower(),
                f"{part_num}-{color_id}".lower(),
            ]
        )
    return keys


def _load_catalog_images(limit: int) -> list[CatalogImage]:
    loaded: list[CatalogImage] = _load_catalog_db_cache_images(min(limit, MAX_CATALOG_DB_IMAGES))
    seen_paths: set[Path] = {item.image_path.resolve() for item in loaded}
    stem_index, ordered_paths = _scan_catalog_image_pool()

    for manifest_path in _expand_globs(CATALOG_MANIFEST_GLOBS):
        try:
            rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            part_num = str(row.get("part_num") or "").strip()
            if not part_num:
                continue
            raw_color = row.get("color_id")
            try:
                color_id = int(raw_color) if raw_color is not None else None
            except (TypeError, ValueError):
                color_id = None
            try:
                qty = int(row.get("qty")) if row.get("qty") is not None else None
            except (TypeError, ValueError):
                qty = None

            local_path = _resolve_local_path(row.get("local_path"))
            if local_path is None:
                for key in _manifest_lookup_keys(part_num, color_id):
                    local_path = stem_index.get(key)
                    if local_path is not None:
                        break
            if local_path is None or not local_path.exists():
                continue
            resolved = local_path.resolve()
            if resolved in seen_paths:
                continue

            image = _load_rgb_image(local_path)
            if image is None:
                continue
            seen_paths.add(resolved)
            loaded.append(
                CatalogImage(
                    item_id=local_path.stem,
                    image_path=local_path,
                    pil_image=image,
                    part_num=part_num,
                    color_id=color_id,
                    qty=qty,
                    summary=(
                        f"manifest catalog image: part={part_num} color={color_id} qty={qty} path={local_path}"
                    ),
                    sort_key=(0, -(qty or 0), part_num, color_id if color_id is not None else -1, str(local_path)),
                )
            )
            if len(loaded) >= limit:
                break
        if len(loaded) >= limit:
            break

    if len(loaded) < limit:
        for path in ordered_paths:
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            image = _load_rgb_image(path)
            if image is None:
                continue
            seen_paths.add(resolved)
            stem = path.stem
            loaded.append(
                CatalogImage(
                    item_id=stem,
                    image_path=path,
                    pil_image=image,
                    part_num=stem,
                    color_id=None,
                    qty=None,
                    summary=f"scanned catalog image: {path}",
                    sort_key=(1, str(path)),
                )
            )
            if len(loaded) >= limit:
                break

    loaded.sort(key=lambda item: item.sort_key)
    return loaded[:limit]


def _load_clip_model() -> tuple[Any, Any, str]:
    if IMPORT_ERROR is not None or torch is None or CLIPModel is None or CLIPProcessor is None:
        raise RuntimeError(
            "Missing local CLIP dependencies. Install `torch` and `transformers` in the active environment."
        ) from IMPORT_ERROR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME_OR_PATH,
            local_files_only=CLIP_LOCAL_FILES_ONLY,
        )
        processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_NAME_OR_PATH,
            local_files_only=CLIP_LOCAL_FILES_ONLY,
        )
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "Could not load the CLIP model locally. "
            "Set `CLIP_MODEL_NAME_OR_PATH` to a cached Hugging Face model id or a local model directory."
        ) from exc

    model.eval()
    model.to(device)
    return model, processor, device


def _embed_images(items: list[Image.Image], model: Any, processor: Any, device: str) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is not available")

    batches: list[np.ndarray] = []
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        encoded = processor(images=batch, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device)
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixel_values)
            features = features / features.norm(dim=-1, keepdim=True)
        batches.append(features.detach().cpu().numpy())
    return np.concatenate(batches, axis=0)


def _sanitize_embeddings(items: list[T], vectors: np.ndarray) -> tuple[list[T], np.ndarray]:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not len(items):
        return [], np.zeros((0, 0), dtype=np.float32)
    if array.shape[0] != len(items):
        limit = min(len(items), int(array.shape[0]))
        items = list(items[:limit])
        array = array[:limit]
    finite_mask = np.isfinite(array).all(axis=1)
    kept_items: list[T] = []
    kept_rows: list[np.ndarray] = []
    for item, row, is_finite in zip(items, array, finite_mask):
        if not bool(is_finite):
            continue
        norm = float(np.linalg.norm(row))
        if not np.isfinite(norm) or norm <= 0.0:
            continue
        kept_items.append(item)
        kept_rows.append((row / norm).astype(np.float32, copy=False))
    if not kept_rows:
        return [], np.zeros((0, array.shape[1]), dtype=np.float32)
    return kept_items, np.stack(kept_rows).astype(np.float32, copy=False)


def _fit_preview(image: Image.Image, size: int = 220) -> Image.Image:
    rgb = image.convert("RGB")
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    scale = min(size / max(1, rgb.width), size / max(1, rgb.height))
    resized = rgb.resize(
        (max(1, int(round(rgb.width * scale))), max(1, int(round(rgb.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    offset = ((size - resized.width) // 2, (size - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def _save_miss_contact_sheet(
    crop_id: str,
    query_image: Image.Image,
    expected_label: str,
    expected_image: Optional[Image.Image],
    predicted: list[CatalogImage],
) -> Optional[Path]:
    CLIP_PROBE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tile_size = 220
    label_h = 54
    gap = 16
    cards: list[tuple[str, Optional[Image.Image]]] = [
        ("Query", query_image),
        (f"Expected\n{expected_label}", expected_image),
    ]
    for index, candidate in enumerate(predicted[:5], start=1):
        cards.append((f"Top {index}\n{candidate.part_num}/{candidate.color_id}", candidate.pil_image))

    width = (len(cards) * tile_size) + ((len(cards) + 1) * gap)
    height = tile_size + label_h + (gap * 2)
    sheet = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)

    for idx, (label, image) in enumerate(cards):
        x = gap + idx * (tile_size + gap)
        y = gap
        draw.rounded_rectangle((x - 2, y - 2, x + tile_size + 2, y + tile_size + label_h + 2), radius=14, fill=(255, 255, 255), outline=(210, 218, 228), width=2)
        if image is not None:
            sheet.paste(_fit_preview(image, tile_size), (x, y))
        else:
            draw.rectangle((x, y, x + tile_size, y + tile_size), fill=(250, 250, 250), outline=(220, 220, 220), width=1)
            draw.text((x + 12, y + (tile_size // 2) - 8), "Image missing", fill=(120, 120, 120))
        label_y = y + tile_size + 8
        for line_index, line in enumerate(str(label).splitlines()):
            draw.text((x + 8, label_y + line_index * 18), line, fill=(25, 35, 45))

    out_path = CLIP_PROBE_REPORT_DIR / f"{str(crop_id or 'miss').strip() or 'miss'}.png"
    sheet.save(out_path, format="PNG")
    return out_path


def _print_config_summary(crops: list[ProbeImage], catalog_images: list[CatalogImage]) -> None:
    print(f"CLIP model: {CLIP_MODEL_NAME_OR_PATH} (local_files_only={CLIP_LOCAL_FILES_ONLY})")
    print(f"Loaded {len(crops)} crop images and {len(catalog_images)} catalog images")
    print(f"Top-K per crop: {min(TOP_K_MATCHES, len(catalog_images))}")
    print()


def _load_saved_good_label_queries(set_num: str, bag: int) -> list[dict[str, Any]]:
    path = REPO_ROOT / "debug" / "training_labels" / f"{str(set_num).strip()}_bag{int(bag)}.json"
    if not path.exists():
        print(f"set={set_num} bag={bag} good_labels=0 file_missing={path}")
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"set={set_num} bag={bag} good_labels=0 parse_error={exc}")
        return []

    crops = payload.get("crops") or {}
    if not isinstance(crops, dict):
        print(f"set={set_num} bag={bag} good_labels=0 invalid_crops_payload")
        return []

    rows: list[dict[str, Any]] = []
    for crop_id, row in sorted(crops.items()):
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").strip().lower() != "good":
            continue
        parts = [dict(part) for part in list(row.get("parts", []) or []) if isinstance(part, dict)]
        if not parts:
            continue
        first_part = parts[0]
        crop_box = row.get("crop_box")
        page_path = _resolve_local_path(row.get("crop_image_path"))
        extracted_crop_path = ""
        if isinstance(crop_box, list) and page_path is not None:
            extracted = _extract_crop_region_to_file(page_path, crop_box, str(crop_id))
            if extracted is not None:
                extracted_crop_path = str(extracted)
        query_image = _load_rgb_image(Path(extracted_crop_path)) if extracted_crop_path else None
        if query_image is None:
            continue
        rows.append(
            {
                "crop_id": str(crop_id),
                "expected_part_num": str(first_part.get("part_num") or "").strip(),
                "expected_color_id": first_part.get("color_id"),
                "extracted_crop_path": extracted_crop_path,
                "query_image": query_image,
            }
        )

    print(f"set={set_num} bag={bag} good_labels={len(rows)}")
    for row in rows:
        print(
            f"  crop_id={row['crop_id']} expected_part_num={row['expected_part_num'] or 'unknown'} "
            f"color_id={row['expected_color_id']} extracted_crop_path={row['extracted_crop_path']}"
        )
    print()
    return rows


def main() -> int:
    queries: list[dict[str, Any]] = []
    for bag in EVAL_BAGS:
        queries.extend(_load_saved_good_label_queries(EVAL_SET_NUM, int(bag)))
    if not queries:
        print("No saved good-label query crops available.")
        return 1

    catalog_images = _load_catalog_images(MAX_CATALOG_IMAGES)
    if not catalog_images:
        print("No local catalog part images were found.")
        print("Check `CATALOG_IMAGE_DIR_CANDIDATES` and the TODO near that setting in this file.")
        return 1

    expected_pairs = sorted(
        {
            (
                str(query["expected_part_num"] or "").strip(),
                int(query["expected_color_id"]),
            )
            for query in queries
            if str(query["expected_part_num"] or "").strip() and query["expected_color_id"] is not None
        }
    )
    candidate_keys = {
        (str(candidate.part_num or "").strip(), int(candidate.color_id))
        for candidate in catalog_images
        if candidate.color_id is not None
    }
    missing_expected = [pair for pair in expected_pairs if pair not in candidate_keys]
    if missing_expected:
        for part_num, color_id in missing_expected:
            image_path = _ensure_catalog_image_for_pair(part_num, int(color_id))
            if image_path is None or not image_path.exists():
                continue
            image = _load_rgb_image(image_path)
            if image is None:
                continue
            key = (str(part_num).strip(), int(color_id))
            if key in candidate_keys:
                continue
            catalog_images.append(
                CatalogImage(
                    item_id=image_path.stem,
                    image_path=image_path,
                    pil_image=image,
                    part_num=str(part_num).strip(),
                    color_id=int(color_id),
                    qty=None,
                    summary=f"expected catalog image: part={part_num} color={color_id} path={image_path}",
                    sort_key=(-2, str(part_num), int(color_id), str(image_path)),
                )
            )
            candidate_keys.add(key)

    catalog_images.sort(key=lambda item: item.sort_key)
    expected_present = [pair for pair in expected_pairs if pair in candidate_keys]
    missing_expected = [pair for pair in expected_pairs if pair not in candidate_keys]
    catalog_by_pair = {
        (
            str(candidate.part_num or "").strip(),
            int(candidate.color_id) if candidate.color_id is not None else None,
        ): candidate
        for candidate in catalog_images
    }

    print(f"expected_total={len(expected_pairs)}")
    print(f"expected_present_count={len(expected_present)}")
    print("missing_expected list=" + json.dumps(
        [
            {"part_num": part_num, "color_id": color_id}
            for part_num, color_id in missing_expected
        ]
    ))

    model, processor, device = _load_clip_model()
    query_vectors = _embed_images([item["query_image"] for item in queries], model, processor, device)
    catalog_vectors = _embed_images([item.pil_image for item in catalog_images], model, processor, device)
    queries, query_vectors = _sanitize_embeddings(queries, query_vectors)
    catalog_images, catalog_vectors = _sanitize_embeddings(catalog_images, catalog_vectors)
    if not queries or query_vectors.size == 0:
        print("No finite query embeddings available.")
        return 1
    if not catalog_images or catalog_vectors.size == 0:
        print("No finite catalog embeddings available.")
        return 1
    similarity = query_vectors @ catalog_vectors.T

    total_evaluated = len(queries)
    top1_count = 0
    top5_count = 0
    top10_count = 0
    top_k = min(TOP_K_MATCHES, len(catalog_images))

    print(f"catalog_candidates={len(catalog_images)}")
    for query_index, query in enumerate(queries):
        expected_color_id = int(query["expected_color_id"]) if query["expected_color_id"] is not None else None
        same_color_indices = [
            idx
            for idx, candidate in enumerate(catalog_images)
            if expected_color_id is not None and candidate.color_id is not None and int(candidate.color_id) == expected_color_id
        ]
        if same_color_indices:
            ranked_pool = np.array(same_color_indices, dtype=np.int64)
            ranked_scores = similarity[query_index, ranked_pool]
            best_indices = ranked_pool[np.argsort(-ranked_scores)[:top_k]]
        else:
            best_indices = np.argsort(-similarity[query_index])[:top_k]
        ranked = [catalog_images[int(idx)] for idx in best_indices]
        expected_pair = (
            str(query["expected_part_num"] or "").strip(),
            expected_color_id,
        )
        hit_rank: Optional[int] = None
        for rank, candidate in enumerate(ranked, start=1):
            pair = (str(candidate.part_num), int(candidate.color_id) if candidate.color_id is not None else None)
            if pair == expected_pair:
                hit_rank = rank
                break
        in_top1 = hit_rank == 1
        in_top5 = hit_rank is not None and hit_rank <= 5
        in_top10 = hit_rank is not None and hit_rank <= 10
        if in_top1:
            top1_count += 1
        if in_top5:
            top5_count += 1
        if in_top10:
            top10_count += 1

        predicted_top1 = ranked[0] if ranked else None
        predicted_label = (
            f"{predicted_top1.part_num}/{predicted_top1.color_id}"
            if predicted_top1 is not None
            else "none"
        )
        print(
            f"crop_id={query['crop_id']} expected={query['expected_part_num']}/{query['expected_color_id']} "
            f"top1={predicted_label} in_top1={str(in_top1).lower()} "
            f"in_top5={str(in_top5).lower()} in_top10={str(in_top10).lower()}"
        )
        if not in_top1:
            expected_candidate = catalog_by_pair.get(expected_pair)
            report_path = _save_miss_contact_sheet(
                str(query["crop_id"]),
                query["query_image"],
                f"{query['expected_part_num']}/{query['expected_color_id']}",
                expected_candidate.pil_image if expected_candidate is not None else None,
                ranked[:5],
            )
            if report_path is not None:
                print(f"  miss_report={report_path}")

    print()
    print(f"total evaluated: {total_evaluated}")
    print(f"top1 count: {top1_count}")
    print(f"top5 count: {top5_count}")
    print(f"top10 count: {top10_count}")
    return 0

    crops = _load_pre_normalized_crops(MAX_CROPS)
    if len(crops) < MAX_CROPS:
        seen_ids = {item.item_id for item in crops}
        for item in _load_training_label_crops(MAX_CROPS):
            if item.item_id in seen_ids:
                continue
            crops.append(item)
            seen_ids.add(item.item_id)
            if len(crops) >= MAX_CROPS:
                break

    if not crops:
        print("No local instruction crop images found.")
        print("Check `PRENORMALIZED_CROP_GLOBS` and `TRAINING_LABEL_JSON_GLOBS` at the top of this file.")
        return 1

    catalog_images = _load_catalog_images(MAX_CATALOG_IMAGES)
    if not catalog_images:
        print("No local catalog part images were found.")
        print("Check `CATALOG_IMAGE_DIR_CANDIDATES` and the TODO near that setting in this file.")
        print(f"Loaded {len(crops)} crop images, but there is nothing local to match them against yet.")
        return 1

    model, processor, device = _load_clip_model()
    crop_vectors = _embed_images([item.pil_image for item in crops], model, processor, device)
    catalog_vectors = _embed_images([item.pil_image for item in catalog_images], model, processor, device)
    similarity = crop_vectors @ catalog_vectors.T

    _print_config_summary(crops, catalog_images)
    top_k = min(TOP_K_MATCHES, len(catalog_images))
    for crop_index, crop in enumerate(crops):
        print(f"=== {crop.item_id} ===")
        print(crop.summary)
        best_indices = np.argsort(-similarity[crop_index])[:top_k]
        for rank, catalog_index in enumerate(best_indices, start=1):
            candidate = catalog_images[int(catalog_index)]
            score = float(similarity[crop_index, int(catalog_index)])
            print(
                f"{rank:02d}. score={score:.4f} "
                f"part={candidate.part_num} color={candidate.color_id} qty={candidate.qty} "
                f"path={candidate.image_path}"
            )
        print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
