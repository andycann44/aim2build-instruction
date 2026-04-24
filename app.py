from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import html
import os
from pathlib import Path
import threading
from urllib.parse import parse_qs
import cv2
import numpy as np
import pytesseract

from lego_reader.downloader import DownloadError, SetNotFoundError, download_set_pdfs
from lego_reader.pdf_reader import read_pdf_pages
from lego_reader.utils import ensure_dir, normalize_set_num

app = FastAPI()

BASE = "/Users/olly/aim2build-instruction/debug/21330/21330_01/pages"
REFERENCE_PAGE = 28
PROJECT_ROOT = Path("/Users/olly/aim2build-instruction")
INSTRUCTIONS_ROOT = PROJECT_ROOT / "instructions"
DEBUG_ROOT = PROJECT_ROOT / "debug"

FAST_PANEL_PRECHECK_CACHE = {}
FAST_PANEL_PRECHECK_CACHE_LOCK = threading.Lock()
ANALYZE_PAGE_CACHE = {}
ANALYZE_PAGE_CACHE_LOCK = threading.Lock()
SEQUENCE_SCAN_ROWS_CACHE = {}
SEQUENCE_SCAN_ROWS_CACHE_LOCK = threading.Lock()
SEQUENCE_SCAN_MAX_WORKERS = max(1, min(4, os.cpu_count() or 1))


def _clone_cached_value(value):
    return copy.deepcopy(value)


def _path_cache_key(path: str):
    try:
        stat = os.stat(path)
        return (path, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (path, None, None)


def _clear_detector_caches():
    with FAST_PANEL_PRECHECK_CACHE_LOCK:
        FAST_PANEL_PRECHECK_CACHE.clear()
    with ANALYZE_PAGE_CACHE_LOCK:
        ANALYZE_PAGE_CACHE.clear()
    with SEQUENCE_SCAN_ROWS_CACHE_LOCK:
        SEQUENCE_SCAN_ROWS_CACHE.clear()



def _switch_detector_pages_dir(pages_dir: Path):
    global BASE, SHELL_TEMPLATE
    BASE = pages_dir.as_posix()
    _clear_detector_caches()
    SHELL_TEMPLATE = make_shell_template()



def _render_load_set_page(error_message=None, set_num_value=""):
    safe_value = html.escape(set_num_value or "")
    safe_base = html.escape(BASE)
    error_html = ""
    if error_message:
        error_html = f'<p class="error">{html.escape(str(error_message))}</p>'

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Load LEGO Set</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 16px;
          background: #f2f2f2;
        }}
        .card {{
          max-width: 560px;
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 16px;
        }}
        .btn {{
          display: inline-block;
          padding: 10px 14px;
          background: #222;
          color: #fff;
          text-decoration: none;
          border-radius: 6px;
          border: 0;
          cursor: pointer;
        }}
        .field {{
          margin-bottom: 12px;
        }}
        label {{
          display: block;
          font-weight: bold;
          margin-bottom: 6px;
        }}
        input[type="text"] {{
          width: 100%;
          box-sizing: border-box;
          padding: 10px 12px;
          border: 1px solid #ccc;
          border-radius: 6px;
          font-size: 16px;
        }}
        .hint {{
          color: #555;
          font-size: 14px;
          margin-top: 8px;
        }}
        .error {{
          color: #b00020;
          font-weight: bold;
        }}
      </style>
    </head>
    <body>
      <div class="card">
        <a class="btn" href="/debug/bag-thumbs">Current Bag Thumbnails</a>
        <h2>Load LEGO Set</h2>
        <p>Enter a LEGO set number. The app will download the first official PDF, render its pages, switch the detector to that set, and redirect to the accepted bag thumbnails.</p>
        {error_html}
        <form method="post" action="/debug/load-set">
          <div class="field">
            <label for="set_num">Set Number</label>
            <input id="set_num" name="set_num" type="text" value="{safe_value}" placeholder="21330" required />
          </div>
          <button class="btn" type="submit">Download And Analyze</button>
        </form>
        <p class="hint">One PDF is analyzed at a time. If LEGO provides multiple PDFs, this uses the first one.</p>
        <p class="hint"><strong>Current pages dir:</strong> {safe_base}</p>
      </div>
    </body>
    </html>
    """


def page_path(page: int) -> str:
    return os.path.join(BASE, f"page_{page:03d}.png")


def list_pages():
    out = []
    for name in sorted(os.listdir(BASE)):
        if name.startswith("page_") and name.endswith(".png"):
            try:
                out.append(int(name.replace("page_", "").replace(".png", "")))
            except ValueError:
                pass
    return out


def draw_box(img, box, color, t=3):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, t)


def crop(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w]


# --------------------------------------------------
# STAGE 1: BAG PANEL
# --------------------------------------------------
def _find_bag_intro_panel_once(
    img,
    *,
    search_w_ratio,
    search_h_ratio,
    min_area_ratio,
    max_area_ratio,
    max_x_ratio,
    max_y_ratio,
    min_roi_mean,
    min_mid_edge_density,
    position_penalty=0.0,
):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if float(np.mean(gray)) < 150:
        return None, None

    sx1, sy1 = 0, 0
    sx2, sy2 = int(w * search_w_ratio), int(h * search_h_ratio)
    search = gray[sy1:sy2, sx1:sx2]

    th = cv2.threshold(search, 228, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1e9
    page_area = float(w * h)

    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        x += sx1
        y += sy1

        area = bw * bh
        area_ratio = area / page_area
        wh_ratio = bw / float(max(bh, 1))

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if wh_ratio < 1.2 or wh_ratio > 5.5:
            continue
        if x > int(w * max_x_ratio):
            continue
        if y > int(h * max_y_ratio):
            continue

        roi = gray[y:y + bh, x:x + bw]
        if roi.size == 0:
            continue

        roi_mean = float(np.mean(roi))
        roi_std = float(np.std(roi))
        if roi_mean < min_roi_mean:
            continue

        left = roi[:, :max(1, int(bw * 0.30))]
        mid = roi[:, int(bw * 0.30):max(int(bw * 0.48), int(bw * 0.30) + 1)]
        right = roi[:, max(int(bw * 0.48), int(bw * 0.30) + 1):]

        if left.size == 0 or mid.size == 0 or right.size == 0:
            continue

        left_mean = float(np.mean(left))
        right_mean = float(np.mean(right))
        left_std = float(np.std(left))
        right_std = float(np.std(right))

        if left_mean > 242:
            continue
        if left_std < 6:
            continue
        if right_std < 6:
            continue

        mid_edges = cv2.Canny(mid, 30, 120)
        mid_edge_density = float(np.mean(mid_edges > 0))
        if mid_edge_density < min_mid_edge_density:
            continue

        contrast_gap = right_mean - left_mean

        score = 0.0
        score += left_std * 1.8
        score += right_std * 0.8
        score += mid_edge_density * 1600.0
        score += contrast_gap * 1.2
        score += roi_std * 0.6
        score -= abs(area_ratio - 0.18) * 300.0
        score -= (x / max(1.0, w)) * position_penalty
        score -= (y / max(1.0, h)) * (position_penalty * 0.5)

        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)

    if best is None:
        return None, None

    return best, float(best_score)


def find_bag_intro_panel_with_debug(img):
    strict_panel, strict_score = _find_bag_intro_panel_once(
        img,
        search_w_ratio=0.75,
        search_h_ratio=0.48,
        min_area_ratio=0.06,
        max_area_ratio=0.36,
        max_x_ratio=0.10,
        max_y_ratio=0.12,
        min_roi_mean=190.0,
        min_mid_edge_density=0.004,
        position_penalty=0.0,
    )
    if strict_panel is not None:
        return strict_panel, "strict_top_left", strict_score

    relaxed_panel, relaxed_score = _find_bag_intro_panel_once(
        img,
        search_w_ratio=0.90,
        search_h_ratio=0.60,
        min_area_ratio=0.05,
        max_area_ratio=0.36,
        max_x_ratio=0.38,
        max_y_ratio=0.34,
        min_roi_mean=185.0,
        min_mid_edge_density=0.0035,
        position_penalty=120.0,
    )
    if relaxed_panel is not None:
        return relaxed_panel, "relaxed_top_band", relaxed_score

    return None, "none", None


def find_bag_intro_panel(img):
    panel, _, _ = find_bag_intro_panel_with_debug(img)
    return panel


def _fast_panel_precheck_once(
    gray,
    *,
    search_w_ratio,
    search_h_ratio,
    min_area_ratio,
    max_area_ratio,
    max_x_ratio,
    max_y_ratio,
    min_roi_mean,
    min_mid_edge_density,
):
    if gray is None or gray.size == 0:
        return False

    h, w = gray.shape[:2]
    if float(np.mean(gray)) < 145:
        return False

    sx1, sy1 = 0, 0
    sx2, sy2 = int(w * search_w_ratio), int(h * search_h_ratio)
    search = gray[sy1:sy2, sx1:sx2]
    if search.size == 0:
        return False

    th = cv2.threshold(search, 226, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(w * h)

    for contour in cnts:
        x, y, bw, bh = cv2.boundingRect(contour)
        area_ratio = (bw * bh) / max(1.0, page_area)
        wh_ratio = bw / float(max(bh, 1))

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if wh_ratio < 1.1 or wh_ratio > 5.8:
            continue
        if x > int(w * max_x_ratio):
            continue
        if y > int(h * max_y_ratio):
            continue

        roi = gray[y:y + bh, x:x + bw]
        if roi.size == 0:
            continue
        if float(np.mean(roi)) < min_roi_mean:
            continue

        left = roi[:, :max(1, int(bw * 0.30))]
        mid = roi[:, int(bw * 0.30):max(int(bw * 0.48), int(bw * 0.30) + 1)]
        right = roi[:, max(int(bw * 0.48), int(bw * 0.30) + 1):]
        if left.size == 0 or mid.size == 0 or right.size == 0:
            continue

        left_std = float(np.std(left))
        right_std = float(np.std(right))
        if left_std < 4 and right_std < 4:
            continue
        if float(np.mean(left)) > 246:
            continue

        mid_edges = cv2.Canny(mid, 25, 100)
        if float(np.mean(mid_edges > 0)) < min_mid_edge_density:
            continue

        return True

    return False


def _passes_fast_panel_precheck_path(path):
    cache_key = _path_cache_key(path)
    with FAST_PANEL_PRECHECK_CACHE_LOCK:
        cached = FAST_PANEL_PRECHECK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    gray = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if gray is None:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return True

    passes = _fast_panel_precheck_once(
        gray,
        search_w_ratio=0.75,
        search_h_ratio=0.48,
        min_area_ratio=0.04,
        max_area_ratio=0.40,
        max_x_ratio=0.12,
        max_y_ratio=0.14,
        min_roi_mean=180.0,
        min_mid_edge_density=0.002,
    )

    if not passes:
        passes = _fast_panel_precheck_once(
            gray,
            search_w_ratio=0.90,
            search_h_ratio=0.60,
            min_area_ratio=0.04,
            max_area_ratio=0.40,
            max_x_ratio=0.42,
            max_y_ratio=0.36,
            min_roi_mean=175.0,
            min_mid_edge_density=0.0015,
        )

    with FAST_PANEL_PRECHECK_CACHE_LOCK:
        FAST_PANEL_PRECHECK_CACHE[cache_key] = passes
    return passes


# --------------------------------------------------
# STAGE 2: SHELL TEMPLATE MATCH
# --------------------------------------------------
def make_shell_template():
    ref_path = page_path(REFERENCE_PAGE)
    if not os.path.exists(ref_path):
        return None

    ref = cv2.imread(ref_path)
    if ref is None:
        return None

    panel = find_bag_intro_panel(ref)
    if panel is None:
        return None

    panel_img = crop(ref, panel)
    ph, pw = panel_img.shape[:2]

    x = int(pw * 0.03)
    y = int(ph * 0.18)
    w = int(pw * 0.24)
    h = int(ph * 0.64)

    shell = panel_img[y:y + h, x:x + w]
    if shell.size == 0:
        return None

    gray = cv2.cvtColor(shell, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 130)

    mh, mw = edges.shape[:2]
    mask = np.ones((mh, mw), dtype=np.uint8) * 255

    nx1 = int(mw * 0.28)
    ny1 = int(mh * 0.25)
    nx2 = int(mw * 0.72)
    ny2 = int(mh * 0.72)
    mask[ny1:ny2, nx1:nx2] = 0

    edges = cv2.bitwise_and(edges, edges, mask=mask)

    return {
        "template_edges": edges,
        "template_w": mw,
        "template_h": mh,
    }


SHELL_TEMPLATE = make_shell_template()


def _box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / float(union_area)


def _border_center_stats(gray_roi):
    if gray_roi is None or gray_roi.size == 0:
        return None

    h, w = gray_roi.shape[:2]
    border = np.concatenate([
        gray_roi[:max(1, h // 10), :].ravel(),
        gray_roi[-max(1, h // 10):, :].ravel(),
        gray_roi[:, :max(1, w // 10)].ravel(),
        gray_roi[:, -max(1, w // 10):].ravel(),
    ])
    center = gray_roi[
        max(1, int(h * 0.18)):max(h - 1, int(h * 0.82)),
        max(1, int(w * 0.18)):max(w - 1, int(w * 0.82)),
    ]
    if border.size == 0 or center.size == 0:
        return None

    return {
        "border_mean": float(np.mean(border)),
        "center_mean": float(np.mean(center)),
        "std": float(np.std(gray_roi)),
    }


def find_rectangle_candidates(panel_img):
    if panel_img is None or panel_img.size == 0:
        return []

    ph, pw = panel_img.shape[:2]
    gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY)

    sx1, sy1 = 0, 0
    sx2, sy2 = int(pw * 0.42), int(ph * 0.95)
    search = gray[sy1:sy2, sx1:sx2]
    if search.size == 0:
        return []

    dark_mask = cv2.threshold(search, 205, 255, cv2.THRESH_BINARY_INV)[1]
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    edge_mask = cv2.Canny(search, 35, 130)
    candidate_mask = cv2.bitwise_or(dark_mask, edge_mask)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_total = float(search.shape[0] * search.shape[1])
    candidates = []

    for contour in cnts:
        x, y, bw, bh = cv2.boundingRect(contour)
        area_ratio = (bw * bh) / max(1.0, area_total)
        wh_ratio = bw / float(max(bh, 1))

        if area_ratio < 0.10 or area_ratio > 0.80:
            continue
        if wh_ratio < 0.75 or wh_ratio > 1.80:
            continue

        roi = search[y:y + bh, x:x + bw]
        if roi.size == 0:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) < 4 or len(approx) > 10:
            continue

        stats = _border_center_stats(roi)
        if stats is None:
            continue

        border_mean = stats["border_mean"]
        center_mean = stats["center_mean"]
        roi_std = stats["std"]
        if border_mean < 175:
            continue
        if center_mean > border_mean - 4:
            continue
        if roi_std < 12:
            continue

        edge_density = float(np.mean(cv2.Canny(roi, 30, 120) > 0))
        if edge_density < 0.01:
            continue

        score = 0.0
        score += max(0.0, border_mean - center_mean) / 45.0
        score += roi_std / 45.0
        score += edge_density * 5.0
        score += max(0.0, 1.2 - abs(wh_ratio - 1.1))
        score -= abs(area_ratio - 0.38) * 2.0
        score = max(0.0, min(1.0, score / 3.2))

        candidates.append({
            "box": (x + sx1, y + sy1, bw, bh),
            "score": float(score),
        })

    deduped = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if any(_box_iou(candidate["box"], existing["box"]) > 0.70 for existing in deduped):
            continue
        deduped.append(candidate)

    return deduped[:8]


def _find_square_box_inside_candidate(candidate_img):
    if candidate_img is None or candidate_img.size == 0:
        return None, None

    ch, cw = candidate_img.shape[:2]
    gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)

    search_x1, search_y1 = 0, 0
    search_x2, search_y2 = int(cw * 0.88), ch
    search = gray[search_y1:search_y2, search_x1:search_x2]
    if search.size == 0:
        return None, None

    mask = cv2.threshold(search, 205, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_area = float(ch * cw)
    best = None
    best_score = None

    for contour in cnts:
        x, y, bw, bh = cv2.boundingRect(contour)
        area_ratio = (bw * bh) / max(1.0, candidate_area)
        wh_ratio = bw / float(max(bh, 1))

        if area_ratio < 0.18 or area_ratio > 0.90:
            continue
        if wh_ratio < 0.75 or wh_ratio > 1.35:
            continue
        if bh < ch * 0.45:
            continue
        if x > int(cw * 0.32):
            continue

        roi = search[y:y + bh, x:x + bw]
        if roi.size == 0:
            continue

        stats = _border_center_stats(roi)
        if stats is None:
            continue

        border_mean = stats["border_mean"]
        center_mean = stats["center_mean"]
        roi_std = stats["std"]
        if border_mean < 170:
            continue
        if center_mean > border_mean - 4:
            continue
        if roi_std < 12:
            continue

        edge_density = float(np.mean(cv2.Canny(roi, 30, 120) > 0))
        if edge_density < 0.01:
            continue

        score = 0.0
        score += max(0.0, border_mean - center_mean) / 45.0
        score += roi_std / 45.0
        score += edge_density * 5.0
        score += max(0.0, 1.2 - abs(wh_ratio - 1.0))
        score -= abs(area_ratio - 0.45) * 2.0
        score = max(0.0, min(1.0, score / 3.0))

        if best_score is None or score > best_score:
            best_score = score
            best = (x + search_x1, y + search_y1, bw, bh)

    return best, (float(best_score) if best_score is not None else None)


def _score_side_tail(candidate_img, square_box, side):
    if candidate_img is None or candidate_img.size == 0 or square_box is None:
        return 0.0

    gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape[:2]
    sx, sy, sw, sh = square_box

    y1 = max(0, sy + int(sh * 0.15))
    y2 = min(ch, sy + int(sh * 0.85))
    if y2 <= y1:
        return 0.0

    if side == "right":
        x1 = max(0, sx + sw - int(sw * 0.08))
        x2 = cw
        band_x1 = max(0, sx + sw - int(sw * 0.05))
        band_x2 = min(cw, sx + sw + max(2, int(sw * 0.18)))
    else:
        x1 = 0
        x2 = min(cw, sx + int(sw * 0.08))
        band_x1 = max(0, sx - max(2, int(sw * 0.18)))
        band_x2 = min(cw, sx + int(sw * 0.05))

    if x2 - x1 < max(8, int(cw * 0.12)):
        return 0.0

    tail_roi = gray[y1:y2, x1:x2]
    if tail_roi.size == 0:
        return 0.0

    tail_mask = cv2.threshold(tail_roi, 210, 255, cv2.THRESH_BINARY_INV)[1]
    tail_mask = cv2.morphologyEx(tail_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    tail_mask = cv2.morphologyEx(tail_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    edge_density = float(np.mean(cv2.Canny(tail_roi, 30, 120) > 0))
    dark_density = float(np.mean(tail_mask > 0))

    cnts, _ = cv2.findContours(tail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area_ratio = 0.0
    tail_area = float(tail_roi.shape[0] * tail_roi.shape[1])
    for contour in cnts:
        _, _, bw, bh = cv2.boundingRect(contour)
        largest_area_ratio = max(largest_area_ratio, (bw * bh) / max(1.0, tail_area))

    connector_band = gray[y1:y2, band_x1:band_x2]
    connector_dark_density = 0.0
    if connector_band.size != 0:
        connector_mask = cv2.threshold(connector_band, 210, 255, cv2.THRESH_BINARY_INV)[1]
        connector_dark_density = float(np.mean(connector_mask > 0))

    score = 0.0
    score += edge_density * 6.0
    score += dark_density * 1.5
    score += largest_area_ratio * 2.0
    score += connector_dark_density * 2.5
    return max(0.0, min(1.0, score / 2.2))


def _find_side_tail(candidate_img, square_box):
    right_score = _score_side_tail(candidate_img, square_box, "right")
    left_score = _score_side_tail(candidate_img, square_box, "left")

    if right_score >= left_score and right_score >= 0.22:
        return "right", right_score
    if left_score > right_score and left_score >= 0.22:
        return "left", left_score
    return None, max(right_score, left_score)


def _find_structured_shell_candidate_with_debug(panel_img):
    rectangle_candidates = find_rectangle_candidates(panel_img)
    if not rectangle_candidates:
        return None, None, "shape_first"

    best_partial_score = rectangle_candidates[0]["score"]
    best_candidate = None

    for rectangle_candidate in rectangle_candidates:
        candidate_box = rectangle_candidate["box"]
        candidate_img = crop(panel_img, candidate_box)
        if candidate_img is None or candidate_img.size == 0:
            continue

        square_box, square_score = _find_square_box_inside_candidate(candidate_img)
        if square_box is None or square_score is None:
            continue

        tail_side, tail_score = _find_side_tail(candidate_img, square_box)
        if tail_side is None:
            continue

        square_img = crop(candidate_img, square_box)
        bag_rel, bag_score = find_grey_bag_blob(square_img)
        if bag_rel is None:
            bag_rel, bag_score = find_grey_bag_blob(candidate_img)
        if bag_rel is None:
            continue

        bag_quality = max(0.0, min(1.0, (bag_score + 40.0) / 80.0))
        combined_score = (
            (rectangle_candidate["score"] * 0.35)
            + (square_score * 0.25)
            + (tail_score * 0.20)
            + (bag_quality * 0.20)
        )
        combined_score = max(0.0, min(0.99, combined_score))

        candidate = {
            "box": candidate_box,
            "score": float(combined_score),
            "method": f"shape_first_{tail_side}",
        }
        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate

    if best_candidate is None:
        return None, float(best_partial_score), "shape_first"

    return best_candidate["box"], best_candidate["score"], best_candidate["method"]


def _find_square_bag_box_outline_fallback(panel_img):
    if panel_img is None or panel_img.size == 0:
        return None, None

    ph, pw = panel_img.shape[:2]
    gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY)

    sx1, sy1 = 0, 0
    sx2, sy2 = int(pw * 0.42), int(ph * 0.95)
    search = gray[sy1:sy2, sx1:sx2]
    if search.size == 0:
        return None, None

    mask = cv2.threshold(search, 205, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1e9
    area_total = float(search.shape[0] * search.shape[1])

    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area_ratio = (bw * bh) / max(1.0, area_total)
        wh_ratio = bw / float(max(bh, 1))

        if area_ratio < 0.14 or area_ratio > 0.70:
            continue
        if wh_ratio < 0.75 or wh_ratio > 1.80:
            continue

        roi = search[y:y + bh, x:x + bw]
        if roi.size == 0:
            continue

        border = np.concatenate([
            roi[:max(1, bh // 10), :].ravel(),
            roi[-max(1, bh // 10):, :].ravel(),
            roi[:, :max(1, bw // 10)].ravel(),
            roi[:, -max(1, bw // 10):].ravel(),
        ])
        center = roi[
            max(1, int(bh * 0.20)):max(bh - 1, int(bh * 0.80)),
            max(1, int(bw * 0.20)):max(bw - 1, int(bw * 0.80)),
        ]

        if border.size == 0 or center.size == 0:
            continue

        border_mean = float(np.mean(border))
        center_mean = float(np.mean(center))
        std = float(np.std(roi))

        if border_mean < 190:
            continue
        if center_mean > border_mean - 6:
            continue
        if std < 18:
            continue

        score = 0.0
        score += (border_mean - center_mean) * 3.0
        score += std * 1.5
        score -= abs(area_ratio - 0.38) * 140.0
        score -= abs(wh_ratio - 1.15) * 40.0
        score -= x * 0.15

        if score > best_score:
            best_score = score
            best = (x + sx1, y + sy1, bw, bh)

    if best is None:
        return None, None

    normalized_score = max(0.26, min(0.95, best_score / 100.0))
    return best, float(normalized_score)


def find_square_bag_box_with_debug(panel_img):
    if panel_img is None or panel_img.size == 0:
        return None, None, "shape_first"

    return _find_structured_shell_candidate_with_debug(panel_img)


def find_square_bag_box(panel_img):
    box, score, _ = find_square_bag_box_with_debug(panel_img)
    return box, score


# --------------------------------------------------
# STAGE 3: GREY BAG
# --------------------------------------------------
def find_grey_bag_blob(square_img):
    if square_img is None or square_img.size == 0:
        return None, 0.0

    h, w = square_img.shape[:2]
    gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)

    pad = max(2, int(min(w, h) * 0.06))
    inner = gray[pad:h - pad, pad:w - pad]
    if inner.size == 0:
        return None, 0.0

    mask = cv2.inRange(inner, 150, 235)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1e9
    area_total = float(inner.shape[0] * inner.shape[1])

    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area_ratio = (bw * bh) / max(1.0, area_total)
        wh_ratio = bw / float(max(bh, 1))

        if area_ratio < 0.18 or area_ratio > 0.80:
            continue
        if wh_ratio < 0.50 or wh_ratio > 1.30:
            continue

        roi = inner[y:y + bh, x:x + bw]
        if roi.size == 0:
            continue

        mean = float(np.mean(roi))
        std = float(np.std(roi))

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        dx = abs(cx - inner.shape[1] / 2.0) / max(1.0, inner.shape[1] / 2.0)
        dy = abs(cy - inner.shape[0] / 2.0) / max(1.0, inner.shape[0] / 2.0)
        center_penalty = (dx + dy) * 40.0

        score = 0.0
        score += std * 1.6
        score -= abs(mean - 195.0) * 0.8
        score -= abs(area_ratio - 0.42) * 180.0
        score -= abs(wh_ratio - 0.85) * 80.0
        score -= center_penalty

        if score > best_score:
            best_score = score
            best = (x + pad, y + pad, bw, bh)

    if best is None:
        return None, 0.0

    return best, float(best_score)


# --------------------------------------------------
# STAGE 4: NUMBER CROP + OCR
# --------------------------------------------------
def _append_number_candidate(candidates, seen, box, img_w, img_h):
    x, y, w, h = [int(round(v)) for v in box]
    x = max(0, x)
    y = max(0, y)
    w = min(max(0, w), img_w - x)
    h = min(max(0, h), img_h - y)

    if w < 8 or h < 12:
        return

    key = (x, y, w, h)
    if key in seen:
        return

    seen.add(key)
    candidates.append(key)



def _extract_digit_like_boxes(gray_img, *, min_area_ratio=0.01, max_area_ratio=0.35, min_height_ratio=0.28, min_width_ratio=0.05):
    if gray_img is None or gray_img.size == 0:
        return []

    h, w = gray_img.shape[:2]
    roi_area = float(h * w)
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    masks = [
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY_INV)[1],
    ]

    boxes = []
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            area_ratio = (bw * bh) / max(1.0, roi_area)

            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            if bh < h * min_height_ratio:
                continue
            if bw < w * min_width_ratio:
                continue

            boxes.append((x, y, bw, bh))

    deduped = []
    for box in sorted(boxes, key=lambda item: (item[0], item[1], -(item[2] * item[3]))):
        if any(
            abs(box[0] - other[0]) <= 4 and
            abs(box[1] - other[1]) <= 4 and
            abs(box[2] - other[2]) <= 4 and
            abs(box[3] - other[3]) <= 4
            for other in deduped
        ):
            continue
        deduped.append(box)

    return deduped[:8]



def _group_adjacent_digit_boxes(boxes, max_gap):
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda item: (item[0], item[1]))
    groups = []
    current_group = [ordered[0]]

    for box in ordered[1:]:
        prev = current_group[-1]
        gap = box[0] - (prev[0] + prev[2])
        overlap = min(prev[1] + prev[3], box[1] + box[3]) - max(prev[1], box[1])
        overlap_needed = min(prev[3], box[3]) * 0.30

        if gap <= max_gap and overlap > overlap_needed:
            current_group.append(box)
        else:
            groups.append(current_group)
            current_group = [box]

    groups.append(current_group)
    return groups



def _group_union_box(group):
    gx1 = min(item[0] for item in group)
    gy1 = min(item[1] for item in group)
    gx2 = max(item[0] + item[2] for item in group)
    gy2 = max(item[1] + item[3] for item in group)
    return gx1, gy1, gx2 - gx1, gy2 - gy1



def find_number_crop(bag_img):
    if bag_img is None or bag_img.size == 0:
        return []

    h, w = bag_img.shape[:2]
    candidates = []
    seen = set()

    gray = cv2.cvtColor(bag_img, cv2.COLOR_BGR2GRAY)

    roi_x1 = int(w * 0.15)
    roi_x2 = int(w * 0.85)
    roi_y1 = int(h * 0.05)
    roi_y2 = int(h * 0.72)
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

    contour_boxes = []
    has_multi_digit_group = False
    if roi.size != 0:
        roi_boxes = _extract_digit_like_boxes(
            roi,
            min_area_ratio=0.01,
            max_area_ratio=0.35,
            min_height_ratio=0.28,
            min_width_ratio=0.05,
        )
        contour_boxes = [(roi_x1 + x, roi_y1 + y, bw, bh) for x, y, bw, bh in roi_boxes]

        groups = _group_adjacent_digit_boxes(
            contour_boxes,
            max_gap=max(10, int(roi.shape[1] * 0.10)),
        )

        for group in groups:
            if len(group) < 2:
                continue

            has_multi_digit_group = True
            gx1, gy1, gw, gh = _group_union_box(group)

            tight_pad_x = max(8, int(gw * 0.18))
            tight_pad_y = max(6, int(gh * 0.16))
            _append_number_candidate(
                candidates,
                seen,
                (gx1 - tight_pad_x, gy1 - tight_pad_y, gw + (2 * tight_pad_x), gh + (2 * tight_pad_y)),
                w,
                h,
            )

            generous_pad_x = max(10, int(gw * 0.32))
            generous_pad_y = max(8, int(gh * 0.24))
            _append_number_candidate(
                candidates,
                seen,
                (gx1 - generous_pad_x, gy1 - generous_pad_y, gw + (2 * generous_pad_x), gh + (2 * generous_pad_y)),
                w,
                h,
            )

        if not has_multi_digit_group:
            for x, y, bw, bh in contour_boxes:
                pad_x = max(10, int(bw * 1.00))
                pad_y = max(8, int(bh * 0.35))
                _append_number_candidate(
                    candidates,
                    seen,
                    (x - pad_x, y - pad_y, bw + (2 * pad_x), bh + (2 * pad_y)),
                    w,
                    h,
                )

                generous_pad_x = max(12, int(bw * 1.20))
                generous_pad_y = max(10, int(bh * 0.45))
                _append_number_candidate(
                    candidates,
                    seen,
                    (x - generous_pad_x, y - generous_pad_y, bw + (2 * generous_pad_x), bh + (2 * generous_pad_y)),
                    w,
                    h,
                )

    if has_multi_digit_group:
        fallback_boxes = [
            (int(w * 0.24), int(h * 0.16), int(w * 0.54), int(h * 0.56)),
            (int(w * 0.20), int(h * 0.12), int(w * 0.60), int(h * 0.62)),
        ]
    else:
        fallback_boxes = [
            (int(w * 0.28), int(h * 0.16), int(w * 0.40), int(h * 0.54)),
            (int(w * 0.24), int(h * 0.12), int(w * 0.46), int(h * 0.60)),
        ]

    for box in fallback_boxes:
        _append_number_candidate(candidates, seen, box, w, h)

    return candidates


def _ocr_hits_from_images(images, configs):
    hits = []
    for img_try in images:
        for cfg in configs:
            raw = pytesseract.image_to_string(img_try, config=cfg).strip()
            digits = "".join(ch for ch in raw if ch.isdigit())
            if len(digits) == 0 or len(digits) > 2:
                continue
            val = int(digits)
            if 1 <= val <= 99:
                hits.append((val, raw))
    return hits


def read_bag_number(number_img):
    if number_img is None or number_img.size == 0:
        return None, ""

    gray = cv2.cvtColor(number_img, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (3, 3), 0)
    
    h, w = up.shape
    cx1 = int(w * 0.25)
    cx2 = int(w * 0.75)
    cy1 = int(h * 0.20)
    cy2 = int(h * 0.80)
    up = up[cy1:cy2, cx1:cx2]

    up = cv2.GaussianBlur(up, (3, 3), 0)

    full_variants = []

    v1 = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    full_variants.append(v1)

    v2 = cv2.threshold(up, 170, 255, cv2.THRESH_BINARY_INV)[1]
    full_variants.append(v2)

    v3 = cv2.GaussianBlur(up, (3, 3), 0)
    v3 = cv2.threshold(v3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    full_variants.append(v3)

    cleaned = []
    for th in full_variants:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0
        H, W = th.shape[:2]

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < (W * H) * 0.008:
                continue
            if h < H * 0.35:
                continue
            if w < W * 0.10:
                continue
            if area > best_area:
                best_area = area
                best = (x, y, w, h)

        if best is None:
            cleaned.append(th)
            continue

        x, y, w, h = best
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        roi = th[y1:y2, x1:x2]
        canvas = np.zeros((max(roi.shape[0] + 20, 120), max(roi.shape[1] + 20, 120)), dtype=np.uint8)
        oy = (canvas.shape[0] - roi.shape[0]) // 2
        ox = (canvas.shape[1] - roi.shape[1]) // 2
        canvas[oy:oy + roi.shape[0], ox:ox + roi.shape[1]] = roi
        cleaned.append(canvas)

    configs = [
        "--psm 10 -c tessedit_char_whitelist=0123456789",
        "--psm 8 -c tessedit_char_whitelist=0123456789",
        "--psm 13 -c tessedit_char_whitelist=0123456789",
    ]

    hits = _ocr_hits_from_images(cleaned, configs)
    if not hits:
        hits = _ocr_hits_from_images(full_variants, configs)

    if not hits:
        # fallback: try OCR on raw upscaled image directly
        raw = pytesseract.image_to_string(
            up,
            config="--psm 8 -c tessedit_char_whitelist=0123456789"
        ).strip()

        digits = "".join(ch for ch in raw if ch.isdigit())

        if digits:
            return int(digits), raw

        return None, ""

    counts = {}
    raw_map = {}
    for val, raw in hits:
        counts[val] = counts.get(val, 0) + 1
        raw_map[val] = raw

    best_val = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    # --- sanity correction for 2 vs 9 ---
    h, w = gray.shape[:2]
    bottom_half = gray[int(h * 0.55):, :]

    bottom_dark_ratio = float((bottom_half < 120).sum()) / max(1, bottom_half.size)

    # If OCR says 2 but bottom has strong dark pixels, it's likely a 9.
    if best_val == 2 and bottom_dark_ratio > 0.12:
        return 9, raw_map[best_val]

    return best_val, raw_map[best_val]



def _estimate_number_structure(gray_img):
    if gray_img is None or gray_img.size == 0:
        return {
            "has_adjacent_pair": False,
            "pair_aspect_ratio": 0.0,
            "crop_aspect_ratio": 0.0,
        }

    h, w = gray_img.shape[:2]
    roi_x1 = int(w * 0.10)
    roi_x2 = int(w * 0.90)
    roi_y1 = int(h * 0.08)
    roi_y2 = int(h * 0.92)
    roi = gray_img[roi_y1:roi_y2, roi_x1:roi_x2]

    boxes = _extract_digit_like_boxes(
        roi,
        min_area_ratio=0.006,
        max_area_ratio=0.45,
        min_height_ratio=0.24,
        min_width_ratio=0.03,
    )
    groups = _group_adjacent_digit_boxes(boxes, max_gap=max(6, int(max(1, roi.shape[1]) * 0.12))) if boxes else []
    pair_groups = [group for group in groups if len(group) >= 2]

    pair_aspect_ratio = 0.0
    if pair_groups:
        best_group = max(pair_groups, key=lambda group: (_group_union_box(group)[2], len(group)))
        _, _, gw, gh = _group_union_box(best_group)
        pair_aspect_ratio = gw / float(max(gh, 1))

    return {
        "has_adjacent_pair": bool(pair_groups),
        "pair_aspect_ratio": pair_aspect_ratio,
        "crop_aspect_ratio": w / float(max(h, 1)),
    }



def read_bag_number_with_score(number_img):
    if number_img is None or number_img.size == 0:
        return None, "", -1.0

    votes = {}

    def add_vote(val, raw, conf_score):
        if val is None or val < 1 or val > 24:
            return

        entry = votes.setdefault(val, {
            "count": 0,
            "best_conf": -1.0,
            "raw": raw or str(val),
        })
        entry["count"] += 1
        if conf_score > entry["best_conf"]:
            entry["best_conf"] = conf_score
            entry["raw"] = raw or str(val)

    base_val, base_raw = read_bag_number(number_img)
    if base_val is not None and 1 <= base_val <= 24:
        add_vote(base_val, base_raw, 25.0)

    gray = cv2.cvtColor(number_img, cv2.COLOR_BGR2GRAY)
    structure = _estimate_number_structure(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    variants = [
        blurred,
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]
    configs = [
        "--psm 13 -c tessedit_char_whitelist=0123456789",
        "--psm 10 -c tessedit_char_whitelist=0123456789",
        "--psm 8 -c tessedit_char_whitelist=0123456789",
        "--psm 7 -c tessedit_char_whitelist=0123456789",
    ]

    for variant in variants:
        up = cv2.resize(variant, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        for cfg in configs:
            data = pytesseract.image_to_data(
                up,
                config=cfg,
                output_type=pytesseract.Output.DICT,
            )

            for raw_text, conf_text in zip(data.get("text", []), data.get("conf", [])):
                digits = "".join(ch for ch in raw_text if ch.isdigit())
                if len(digits) == 0 or len(digits) > 2:
                    continue

                try:
                    val = int(digits)
                except ValueError:
                    continue

                try:
                    conf_score = float(conf_text)
                except (TypeError, ValueError):
                    conf_score = -1.0

                if conf_score < 0.0:
                    continue

                add_vote(val, digits, conf_score)

    if not votes:
        return None, "", -1.0

    def total_vote_score(val, meta):
        digit_len = len(meta["raw"])
        total = meta["best_conf"] + ((meta["count"] - 1) * 4.0) + ((digit_len - 1) * 40.0)

        if structure["has_adjacent_pair"]:
            if digit_len >= 2:
                total += 48.0
                if structure["pair_aspect_ratio"] >= 0.85:
                    total += 18.0
            else:
                total -= 34.0
                if structure["crop_aspect_ratio"] >= 0.65:
                    total -= 8.0
        elif digit_len >= 2 and structure["crop_aspect_ratio"] < 0.58:
            total -= 12.0

        return total

    best_val, best_meta = max(
        votes.items(),
        key=lambda item: (
            total_vote_score(item[0], item[1]),
            item[1]["best_conf"],
            len(item[1]["raw"]),
            item[1]["count"],
        ),
    )
    best_score = total_vote_score(best_val, best_meta)
    return best_val, best_meta["raw"], float(best_score)

# --------------------------------------------------
# SCORING
# --------------------------------------------------
def compute_confidence(panel_found, shell_found, grey_bag_found,
                       number_box_found, bag_number,
                       shell_score, grey_bag_score):

    score = 0.0

    # Stage 1: panel
    if panel_found:
        score += 0.15

    # Stage 2: shell
    if shell_found:
        score += 0.20

        if shell_score is not None:
            score += min(shell_score, 1.0) * 0.15

    # Stage 3: grey bag
    if grey_bag_found:
        score += 0.25

        if grey_bag_score is not None:
            norm = max(0.0, min(1.0, (grey_bag_score + 40) / 80))
            score += norm * 0.15

    # Stage 4: number crop
    if number_box_found:
        score += 0.10

    # Stage 5: OCR validity
    if bag_number is not None:
        score += 0.40
    else:
        score -= 0.30

    return round(min(score, 1.0), 3)


# --------------------------------------------------
# PAGE ANALYSIS
# --------------------------------------------------
def analyze_page(page: int, include_image: bool = True):
    path = page_path(page)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Page not found")

    cache_key = None
    if not include_image:
        cache_key = _path_cache_key(path)
        with ANALYZE_PAGE_CACHE_LOCK:
            cached = ANALYZE_PAGE_CACHE.get(cache_key)
        if cached is not None:
            return _clone_cached_value(cached)

        if not _passes_fast_panel_precheck_path(path):
            fast_reject_result = {
                "page": page,
                "panel_found": False,
                "panel_box": None,
                "panel_source": "fast_reject",
                "panel_score": None,
                "shell_found": False,
                "shell_box": None,
                "shell_score": None,
                "shell_method": "none",
                "grey_bag_found": False,
                "grey_bag_box": None,
                "grey_bag_score": None,
                "number_box_found": False,
                "number_box": None,
                "bag_number": None,
                "ocr_raw": "",
                "confidence": compute_confidence(False, False, False, False, None, None, None),
            }
            with ANALYZE_PAGE_CACHE_LOCK:
                ANALYZE_PAGE_CACHE[cache_key] = _clone_cached_value(fast_reject_result)
            return fast_reject_result

    img = cv2.imread(path)
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read image")

    vis = img.copy() if include_image else None

    panel, panel_source, panel_score = find_bag_intro_panel_with_debug(img)

    square_abs = None
    shell_score = None
    shell_method = "none"
    bag_abs = None
    bag_score = None
    number_abs = None
    bag_number = None
    ocr_raw = ""

    if panel is not None and include_image:
        draw_box(vis, panel, (0, 255, 0), 4)

    # --------------------------------------------------
    # PANEL → SHELL → BAG → NUMBER
    # --------------------------------------------------
    if panel is not None:
        panel_img = crop(img, panel)

        square_rel, shell_score, shell_method = find_square_bag_box_with_debug(panel_img)

        if square_rel is not None:
            px, py, _, _ = panel
            sx, sy, sw, sh = square_rel

            square_abs = (px + sx, py + sy, sw, sh)

            if include_image:
                draw_box(vis, square_abs, (255, 0, 0), 3)

            square_img = crop(img, square_abs)

            bag_rel, bag_score = find_grey_bag_blob(square_img)

            if bag_rel is not None:
                bx, by, _, _ = square_abs
                gx, gy, gw, gh = bag_rel
                bag_abs = (bx + gx, by + gy, gw, gh)

                if include_image:
                    draw_box(vis, bag_abs, (0, 200, 255), 2)

                bag_img = crop(img, bag_abs)

                best_candidate = None
                best_multi_digit_candidate = None

                for number_rel in find_number_crop(bag_img):
                    gx2, gy2, _, _ = bag_abs
                    nx, ny, nw, nh = number_rel

                    number_abs_try = (gx2 + nx, gy2 + ny, nw, nh)
                    number_img = crop(img, number_abs_try)

                    val, raw, ocr_score = read_bag_number_with_score(number_img)
                    if val is None:
                        continue

                    candidate = {
                        "score": ocr_score,
                        "number": val,
                        "raw": raw,
                        "box": number_abs_try,
                    }

                    if best_candidate is None or candidate["score"] > best_candidate["score"]:
                        best_candidate = candidate

                    if len(str(val)) >= 2:
                        if best_multi_digit_candidate is None or candidate["score"] > best_multi_digit_candidate["score"]:
                            best_multi_digit_candidate = candidate

                chosen_candidate = best_candidate
                if (
                    best_candidate is not None and
                    best_multi_digit_candidate is not None and
                    best_multi_digit_candidate["score"] >= 90.0 and
                    best_multi_digit_candidate["score"] + 40.0 >= best_candidate["score"]
                ):
                    chosen_candidate = best_multi_digit_candidate

                if chosen_candidate is not None:
                    bag_number = chosen_candidate["number"]
                    ocr_raw = chosen_candidate["raw"]
                    number_abs = chosen_candidate["box"]

                    if include_image and number_abs is not None:
                        draw_box(vis, number_abs, (0, 0, 255), 2)

    # --------------------------------------------------
    # FLAGS
    # --------------------------------------------------
    panel_found = panel is not None
    shell_found = square_abs is not None
    grey_bag_found = bag_abs is not None
    number_box_found = number_abs is not None

    confidence = compute_confidence(
        panel_found,
        shell_found,
        grey_bag_found,
        number_box_found,
        bag_number,
        shell_score,
        bag_score,
    )

    # --------------------------------------------------
    # DRAW LABELS
    # --------------------------------------------------
    if include_image:
        if panel_found:
            panel_label = "BAG PANEL" if panel_source == "strict_top_left" else f"BAG PANEL {panel_source}"
            cv2.putText(
                vis,
                panel_label,
                (panel[0], max(30, panel[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                vis,
                "NO PANEL",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        if shell_found:
            shell_label = f"SHELL {shell_method} {shell_score:.2f}" if shell_score is not None else f"SHELL {shell_method}"
            cv2.putText(
                vis,
                shell_label,
                (square_abs[0], square_abs[1] + square_abs[3] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
        elif panel_found:
            txt = f"NO SHELL ({shell_score:.2f})" if shell_score is not None else "NO SHELL"
            cv2.putText(
                vis,
                txt,
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        if grey_bag_found:
            cv2.putText(
                vis,
                f"GREY BAG {bag_score:.1f}",
                (bag_abs[0], max(30, bag_abs[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
        elif shell_found:
            cv2.putText(
                vis,
                "NO GREY BAG",
                (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 200, 255),
                2,
            )

        label = f"BAG {bag_number}" if bag_number is not None else "BAG ?"
        if panel_found:
            cv2.putText(
                vis,
                f"{label}  CONF {confidence:.2f}",
                (30, img.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

    result = {
        "page": page,
        "panel_found": panel_found,
        "panel_box": list(panel) if panel_found else None,
        "panel_source": panel_source,
        "panel_score": panel_score,
        "shell_found": shell_found,
        "shell_box": list(square_abs) if shell_found else None,
        "shell_score": shell_score,
        "shell_method": shell_method,
        "grey_bag_found": grey_bag_found,
        "grey_bag_box": list(bag_abs) if grey_bag_found else None,
        "grey_bag_score": bag_score,
        "number_box_found": number_box_found,
        "number_box": list(number_abs) if number_box_found else None,
        "bag_number": bag_number,
        "ocr_raw": ocr_raw,
        "confidence": confidence,
    }

    if include_image:
        ok, buf = cv2.imencode(".png", vis)
        if not ok:
            raise HTTPException(status_code=500, detail="Encode failed")
        result["image_bytes"] = buf.tobytes()
    elif cache_key is not None:
        with ANALYZE_PAGE_CACHE_LOCK:
            ANALYZE_PAGE_CACHE[cache_key] = _clone_cached_value(result)

    return result


# --------------------------------------------------
# SEQUENTIAL SCAN LAYER
# --------------------------------------------------
def _row_has_candidate_signal(r):
    return bool(
        r.get("panel_found") or
        r.get("shell_found") or
        r.get("grey_bag_found") or
        r.get("number_box_found") or
        r.get("bag_number") is not None or
        float(r.get("confidence", 0.0) or 0.0) >= 0.15
    )


def _summarize_gap_candidate(r):
    return {
        "page": r["page"],
        "confidence": r["confidence"],
        "panel_found": r["panel_found"],
        "panel_source": r.get("panel_source"),
        "shell_found": r["shell_found"],
        "shell_method": r.get("shell_method"),
        "grey_bag_found": r["grey_bag_found"],
        "number_box_found": r["number_box_found"],
        "bag_number": r["bag_number"],
        "ocr_raw": r["ocr_raw"],
    }


def build_missing_bag_debug(rows, accepted_sequence, deferred_candidates, bag_number):
    if not rows:
        return None

    pages = [r["page"] for r in rows]
    target_accepted = next((item for item in accepted_sequence if item["number"] == bag_number), None)

    lower_accepted = [item for item in accepted_sequence if item["number"] < bag_number]
    higher_accepted = [item for item in accepted_sequence if item["number"] > bag_number]

    previous_accepted = max(lower_accepted, key=lambda item: item["number"], default=None)
    next_accepted = min(higher_accepted, key=lambda item: item["number"], default=None)

    higher_detected_rows = [r for r in rows if r.get("bag_number") is not None and r["bag_number"] > bag_number]
    next_detected = next_accepted
    if next_detected is None and higher_detected_rows:
        next_row = min(higher_detected_rows, key=lambda item: item["page"])
        next_detected = {
            "page": next_row["page"],
            "number": next_row["bag_number"],
            "confidence": next_row["confidence"],
            "reason": "next_detected_higher_number",
        }

    range_start = previous_accepted["page"] + 1 if previous_accepted is not None else pages[0]
    if next_detected is not None:
        range_end = next_detected["page"] - 1
    else:
        range_end = pages[-1]

    if target_accepted is not None:
        range_start = previous_accepted["page"] + 1 if previous_accepted is not None else pages[0]
        range_end = next_accepted["page"] - 1 if next_accepted is not None else pages[-1]

    if range_end < range_start:
        range_end = range_start

    candidate_pages = [
        _summarize_gap_candidate(r)
        for r in rows
        if range_start <= r["page"] <= range_end and _row_has_candidate_signal(r)
    ]

    return {
        "bag_number": bag_number,
        "status": "accepted" if target_accepted is not None else "missing",
        "accepted_page": target_accepted["page"] if target_accepted is not None else None,
        "previous_accepted": previous_accepted,
        "next_detected_higher": next_detected,
        "likely_range": {
            "start_page": range_start,
            "end_page": range_end,
        },
        "candidate_pages": candidate_pages,
        "candidate_count": len(candidate_pages),
    }


def analyze_missing_bag_debug(
    bag_number,
    *,
    start_number=1,
    end_number=24,
    min_confidence=0.70,
    allow_structure_only_start=True,
):
    data = run_sequential_scan(
        start_number=start_number,
        end_number=end_number,
        min_confidence=min_confidence,
        allow_structure_only_start=allow_structure_only_start,
    )
    gap = build_missing_bag_debug(
        data["rows"],
        data["accepted_sequence"],
        data["deferred_candidates"],
        bag_number,
    )
    return data, gap


def _build_sequence_scan_row(page: int):
    row = analyze_page(page, include_image=False)
    row["strong_structure"] = bool(
        row["panel_found"] and
        row["shell_found"] and
        row["grey_bag_found"]
    )
    return row


def build_sequence_scan_rows():
    pages = sorted(list_pages())
    cache_key = (BASE, tuple(_path_cache_key(page_path(page)) for page in pages))

    with SEQUENCE_SCAN_ROWS_CACHE_LOCK:
        cached_rows = SEQUENCE_SCAN_ROWS_CACHE.get(cache_key)
    if cached_rows is not None:
        return _clone_cached_value(cached_rows)

    max_workers = min(SEQUENCE_SCAN_MAX_WORKERS, max(1, len(pages)))
    if max_workers <= 1:
        rows = [_build_sequence_scan_row(page) for page in pages]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            rows = list(executor.map(_build_sequence_scan_row, pages))

    with SEQUENCE_SCAN_ROWS_CACHE_LOCK:
        SEQUENCE_SCAN_ROWS_CACHE[cache_key] = _clone_cached_value(rows)
    return rows


def run_generic_sequence_scan(
    rows,
    *,
    start_number=1,
    end_number=24,
    min_confidence=0.70,
    allow_structure_only_start=True,
    number_key="bag_number",
    confidence_key="confidence",
    structure_key="strong_structure",
):
    ordered_rows = sorted(
        rows,
        key=lambda item: (
            int(item.get("page", 0) or 0),
            int(item.get(number_key, -1) or -1),
        ),
    )

    expected_next = start_number
    accepted = []
    deferred = []
    structure_only_start_candidates = []
    page_debug_log = []

    for r in ordered_rows:
        page = int(r.get("page", 0) or 0)
        detected_number = r.get(number_key)
        confidence = float(r.get(confidence_key, 0.0) or 0.0)
        strong_structure = bool(r.get(structure_key))

        debug_entry = {
            "page": page,
            "detected_number": detected_number,
            "confidence": confidence,
            "expected_next_before": expected_next,
            "expected_next_after": expected_next,
            "action": "no_number_detected" if detected_number is None else "unhandled",
        }

        if detected_number == expected_next and confidence >= min_confidence:
            accepted.append({
                "page": page,
                "number": detected_number,
                "bag_number": detected_number,
                "confidence": confidence,
                "reason": "accepted_expected_number",
            })
            expected_next += 1
            debug_entry["action"] = "accepted_expected_number"
            debug_entry["expected_next_after"] = expected_next
            page_debug_log.append(debug_entry)
            continue

        if detected_number is not None and detected_number == expected_next:
            deferred.append({
                "page": page,
                "number": detected_number,
                "bag_number": detected_number,
                "confidence": confidence,
                "reason": f"deferred_low_confidence_expected_{expected_next}",
            })
            debug_entry["action"] = "deferred_low_confidence_expected_number"
            page_debug_log.append(debug_entry)
            continue


        if detected_number is not None and detected_number > expected_next:
            deferred.append({
                "page": page,
                "number": detected_number,
                "bag_number": detected_number,
                "confidence": confidence,
                "reason": f"deferred_waiting_for_{expected_next}",
            })
            debug_entry["action"] = "deferred_higher_number_waiting_for_missing_bag"
            debug_entry["waiting_for"] = expected_next
            page_debug_log.append(debug_entry)
            continue

        if detected_number is not None and detected_number < expected_next:
            debug_entry["action"] = "ignored_already_passed_number"
            page_debug_log.append(debug_entry)
            continue

        if (
            detected_number is None
            and strong_structure
            and allow_structure_only_start
            and expected_next == start_number
        ):
            structure_only_start_candidates.append({
                "page": page,
                "expected_number": start_number,
                "confidence": confidence,
                "reason": "strong_structure_no_number_start_candidate",
            })
            debug_entry["strong_structure"] = True
            debug_entry["structure_only_start_candidate"] = True
            page_debug_log.append(debug_entry)
            continue

        debug_entry["strong_structure"] = strong_structure
        page_debug_log.append(debug_entry)

    accepted_numbers = sorted({r["bag_number"] for r in accepted})
    missing = [n for n in range(start_number, end_number + 1) if n not in accepted_numbers]
    expected_next_debug = None
    if expected_next <= end_number:
        expected_next_debug = build_missing_bag_debug(ordered_rows, accepted, deferred, expected_next)


    return {
        "accepted_sequence": accepted,
        "deferred_candidates": deferred,
        "structure_only_start_candidates": structure_only_start_candidates,
        "possible_bag1_candidates": structure_only_start_candidates,
        "missing_numbers": missing,
        "expected_next": expected_next,
        "expected_next_debug": expected_next_debug,
        "page_debug_log": page_debug_log,
        "scan_config": {
            "start_number": start_number,
            "end_number": end_number,
            "min_confidence": min_confidence,
            "allow_structure_only_start": allow_structure_only_start,
            "number_key": number_key,
            "confidence_key": confidence_key,
            "structure_key": structure_key,
        },
        "total_pages": len(ordered_rows),
        "rows": ordered_rows,
    }


def run_sequential_scan(
    start_number=1,
    end_number=24,
    min_confidence=0.70,
    allow_structure_only_start=True,
):
    rows = build_sequence_scan_rows()
    return run_generic_sequence_scan(
        rows,
        start_number=start_number,
        end_number=end_number,
        min_confidence=min_confidence,
        allow_structure_only_start=allow_structure_only_start,
    )


def compute_bag_ranges(accepted_sequence, last_page):
    starts = []
    for item in accepted_sequence:
        if not isinstance(item, dict):
            continue
        page = item.get("page")
        number = item.get("number", item.get("bag_number"))
        if not isinstance(page, int) or not isinstance(number, int):
            continue
        starts.append({
            "number": number,
            "page": page,
        })

    starts.sort(key=lambda item: (item["page"], item["number"]))

    bag_ranges = []
    for idx, item in enumerate(starts):
        start_page = item["page"]
        if idx + 1 < len(starts):
            end_page = starts[idx + 1]["page"] - 1
        else:
            end_page = last_page

        if end_page < start_page:
            end_page = start_page

        bag_ranges.append({
            "bag_number": item["number"],
            "start_page": start_page,
            "end_page": end_page,
        })

    return bag_ranges



def build_bag_ranges_payload(
    *,
    start_number=1,
    end_number=24,
    min_confidence=0.70,
    allow_structure_only_start=True,
):
    data = run_sequential_scan(
        start_number=start_number,
        end_number=end_number,
        min_confidence=min_confidence,
        allow_structure_only_start=allow_structure_only_start,
    )
    pages = list_pages()
    last_page = pages[-1] if pages else 0
    accepted_sequence = [
        {
            "number": item["number"],
            "page": item["page"],
        }
        for item in data["accepted_sequence"]
        if isinstance(item.get("number"), int) and isinstance(item.get("page"), int)
    ]
    bag_ranges = compute_bag_ranges(data["accepted_sequence"], last_page)
    return {
        "accepted_sequence": accepted_sequence,
        "bag_ranges": bag_ranges,
        "missing_numbers": data["missing_numbers"],
        "last_page": last_page,
        "scan_config": data["scan_config"],
    }


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "load_set_ui": "/debug/load-set",
        "viewer": "/debug/compare?a=28&b=24",
        "scan_all": "/api/scan-all",
        "scan_top": "/debug/top-bags",
        "sequence_scan": "/api/sequence-scan",
        "sequence_debug": "/debug/sequential",
        "bag_ranges": "/api/bag-ranges",
        "bags_debug": "/debug/bags",
        "bag_thumbs": "/debug/bag-thumbs",
        "missing_bag_debug": "/api/missing-bag-debug?bag_number=7",
        "missing_bag_view": "/debug/missing-bag?bag_number=7",
    }


@app.get("/debug/load-set", response_class=HTMLResponse)
def debug_load_set_page():
    return _render_load_set_page()


@app.post("/debug/load-set")
async def debug_load_set_submit(request: Request):
    body = await request.body()
    form_values = parse_qs(body.decode("utf-8"))
    set_num = form_values.get("set_num", [""])[0]

    try:
        normalized_set_num = normalize_set_num(set_num)
    except ValueError as error:
        return HTMLResponse(_render_load_set_page(str(error), set_num), status_code=400)

    instructions_dir = ensure_dir(INSTRUCTIONS_ROOT)
    debug_dir = ensure_dir(DEBUG_ROOT)

    try:
        download_result = download_set_pdfs(
            set_num=normalized_set_num,
            instructions_dir=instructions_dir,
        )
    except SetNotFoundError as error:
        return HTMLResponse(_render_load_set_page(str(error), normalized_set_num), status_code=404)
    except DownloadError as error:
        return HTMLResponse(_render_load_set_page(str(error), normalized_set_num), status_code=502)

    if not download_result.pdfs:
        return HTMLResponse(
            _render_load_set_page("No instruction PDFs were returned by LEGO.", normalized_set_num),
            status_code=404,
        )

    downloaded_pdf = download_result.pdfs[0]
    per_pdf_debug_dir = debug_dir / normalized_set_num / downloaded_pdf.local_path.stem
    pages_dir = per_pdf_debug_dir / "pages"

    if not pages_dir.exists() or not any(pages_dir.glob("page_*.png")):
        read_pdf_pages(
            pdf_path=downloaded_pdf.local_path,
            debug=True,
            debug_dir=per_pdf_debug_dir,
        )

    _switch_detector_pages_dir(pages_dir)
    return RedirectResponse(url="/debug/bag-thumbs", status_code=303)


@app.get("/api/page")
def api_page(page: int = Query(...)):
    path = page_path(page)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(path)


@app.get("/api/annotated")
def api_annotated(page: int = Query(...)):
    result = analyze_page(page, include_image=True)
    return Response(content=result["image_bytes"], media_type="image/png")


@app.get("/api/analyze")
def api_analyze(page: int = Query(...)):
    result = analyze_page(page, include_image=False)
    return result


@app.get("/api/scan-all")
def api_scan_all():
    rows = []
    for page in list_pages():
        rows.append(analyze_page(page, include_image=False))

    rows.sort(key=lambda r: (r["confidence"], r["page"]), reverse=True)

    found_with_number = [r for r in rows if r["bag_number"] is not None]
    unique_numbers = sorted({r["bag_number"] for r in found_with_number})
    missing_from_24 = [n for n in range(1, 25) if n not in unique_numbers]

    return {
        "total_pages": len(rows),
        "pages_with_any_panel": sum(1 for r in rows if r["panel_found"]),
        "pages_with_shell": sum(1 for r in rows if r["shell_found"]),
        "pages_with_grey_bag": sum(1 for r in rows if r["grey_bag_found"]),
        "pages_with_number_crop": sum(1 for r in rows if r["number_box_found"]),
        "pages_with_valid_bag_number": len(found_with_number),
        "detected_bag_numbers": unique_numbers,
        "missing_from_1_to_24": missing_from_24,
        "rows": rows,
    }


@app.get("/api/sequence-scan")
def api_sequence_scan(
    start_number: int = Query(1, ge=1),
    end_number: int = Query(24, ge=1),
    min_confidence: float = Query(0.70, ge=0.0, le=1.0),
    allow_structure_only_start: bool = Query(True),
):
    if end_number < start_number:
        raise HTTPException(status_code=400, detail="end_number must be >= start_number")

    return run_sequential_scan(
        start_number=start_number,
        end_number=end_number,
        min_confidence=min_confidence,
        allow_structure_only_start=allow_structure_only_start,
    )


@app.get("/api/bag-ranges")
def api_bag_ranges(
    start_number: int = Query(1, ge=1),
    end_number: int = Query(24, ge=1),
    min_confidence: float = Query(0.70, ge=0.0, le=1.0),
    allow_structure_only_start: bool = Query(True),
):
    if end_number < start_number:
        raise HTTPException(status_code=400, detail="end_number must be >= start_number")

    return build_bag_ranges_payload(
        start_number=start_number,
        end_number=end_number,
        min_confidence=min_confidence,
        allow_structure_only_start=allow_structure_only_start,
    )


@app.get("/api/missing-bag-debug")
def api_missing_bag_debug(
    bag_number: int = Query(..., ge=1),
    start_number: int = Query(1, ge=1),
    end_number: int = Query(24, ge=1),
    min_confidence: float = Query(0.70, ge=0.0, le=1.0),
    allow_structure_only_start: bool = Query(True),
):
    if end_number < start_number:
        raise HTTPException(status_code=400, detail="end_number must be >= start_number")
    if bag_number < start_number or bag_number > end_number:
        raise HTTPException(status_code=400, detail="bag_number must be inside the scan range")

    data, gap = analyze_missing_bag_debug(
        bag_number,
        start_number=start_number,
        end_number=end_number,
        min_confidence=min_confidence,
        allow_structure_only_start=allow_structure_only_start,
    )
    if gap is None:
        raise HTTPException(status_code=404, detail="Could not build missing-bag debug output")

    return {
        "bag_number": bag_number,
        "accepted_sequence": data["accepted_sequence"],
        "missing_numbers": data["missing_numbers"],
        "expected_next": data["expected_next"],
        "debug": gap,
    }


@app.get("/debug/compare", response_class=HTMLResponse)
def compare(a: int = 28, b: int = 24):
    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Two Page Compare</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 16px;
          background: #f2f2f2;
        }}
        .top {{
          margin-bottom: 16px;
          display: flex;
          gap: 12px;
          align-items: center;
          flex-wrap: wrap;
        }}
        .btn {{
          display: inline-block;
          padding: 8px 12px;
          background: #222;
          color: #fff;
          text-decoration: none;
          border-radius: 6px;
        }}
        .row {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
        }}
        .card {{
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 12px;
        }}
        img {{
          max-width: 100%;
          border: 1px solid #ccc;
          display: block;
        }}
        .small {{
          font-size: 14px;
          color: #444;
        }}
      </style>
    </head>
    <body>
      <div class="top">
        <strong>Compare pages {a} and {b}</strong>
        <a class="btn" href="/api/analyze?page={a}" target="_blank">JSON A</a>
        <a class="btn" href="/api/analyze?page={b}" target="_blank">JSON B</a>
        <a class="btn" href="/debug/top-bags">Whole PDF results</a>
        <a class="btn" href="/debug/sequential">Sequential results</a>
      </div>

      <div class="row">
        <div class="card">
          <h3>Page {a} (expected BAG)</h3>
          <div class="small">Green = panel. Blue = shell. Yellow = grey bag. Red = number crop.</div>
          <img src="/api/annotated?page={a}" />
        </div>

        <div class="card">
          <h3>Page {b} (expected NOT BAG)</h3>
          <div class="small">Should usually say NO PANEL.</div>
          <img src="/api/annotated?page={b}" />
        </div>
      </div>
    </body>
    </html>
    """


@app.get("/debug/top-bags", response_class=HTMLResponse)
def debug_top_bags():
    rows = [analyze_page(page, include_image=False) for page in list_pages()]
    rows.sort(key=lambda r: (r["confidence"], r["page"]), reverse=True)

    top = rows[:40]
    detected = sorted({r["bag_number"] for r in rows if r["bag_number"] is not None})
    missing = [n for n in range(1, 25) if n not in detected]

    cards = []
    for r in top:
        label = f"bag {r['bag_number']}" if r["bag_number"] is not None else "no bag number"
        cards.append(f"""
        <tr>
          <td>{r['page']}</td>
          <td>{r['confidence']:.2f}</td>
          <td>{'Y' if r['panel_found'] else ''}</td>
          <td>{'Y' if r['shell_found'] else ''}</td>
          <td>{'Y' if r['grey_bag_found'] else ''}</td>
          <td>{'Y' if r['number_box_found'] else ''}</td>
          <td>{label}</td>
          <td><a href="/api/analyze?page={r['page']}" target="_blank">json</a></td>
          <td><a href="/api/annotated?page={r['page']}" target="_blank">image</a></td>
        </tr>
        """)

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Whole PDF scoring</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 16px;
          background: #f2f2f2;
        }}
        .card {{
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 12px;
          margin-bottom: 16px;
        }}
        table {{
          border-collapse: collapse;
          width: 100%;
          background: white;
        }}
        th, td {{
          border: 1px solid #ddd;
          padding: 8px;
          text-align: left;
          font-size: 14px;
        }}
        th {{
          background: #eee;
        }}
        .btn {{
          display: inline-block;
          padding: 8px 12px;
          background: #222;
          color: #fff;
          text-decoration: none;
          border-radius: 6px;
          margin-right: 8px;
        }}
      </style>
    </head>
    <body>
      <div class="card">
        <a class="btn" href="/debug/compare?a=28&b=24">Back to compare</a>
        <a class="btn" href="/api/scan-all" target="_blank">Raw JSON scan</a>
        <a class="btn" href="/debug/sequential">Sequential scan</a>
        <h2>Whole PDF scoring</h2>
        <p><strong>Detected bag numbers:</strong> {detected}</p>
        <p><strong>Missing from 1..24:</strong> {missing}</p>
      </div>

      <div class="card">
        <table>
          <thead>
            <tr>
              <th>Page</th>
              <th>Confidence</th>
              <th>Panel</th>
              <th>Shell</th>
              <th>Grey bag</th>
              <th>Number crop</th>
              <th>Bag number</th>
              <th>JSON</th>
              <th>Image</th>
            </tr>
          </thead>
          <tbody>
            {''.join(cards)}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """


@app.get("/debug/sequential", response_class=HTMLResponse)
def debug_sequential(
    start_number: int = Query(1, ge=1),
    end_number: int = Query(24, ge=1),
    min_confidence: float = Query(0.70, ge=0.0, le=1.0),
    allow_structure_only_start: bool = Query(True),
):
    if end_number < start_number:
        raise HTTPException(status_code=400, detail="end_number must be >= start_number")

    data = run_sequential_scan(
        start_number=start_number,
        end_number=end_number,
        min_confidence=min_confidence,
        allow_structure_only_start=allow_structure_only_start,
    )
    query_string = (
        f"start_number={start_number}&"
        f"end_number={end_number}&"
        f"min_confidence={min_confidence:.2f}&"
        f"allow_structure_only_start={str(allow_structure_only_start).lower()}"
    )

    accepted_rows = []
    for r in data["accepted_sequence"]:
        accepted_rows.append(f"""
        <tr>
          <td>{r['page']}</td>
          <td>{r['number']}</td>
          <td>{r['confidence']:.2f}</td>
          <td>{r['reason']}</td>
          <td><a href="/api/analyze?page={r['page']}" target="_blank">json</a></td>
          <td><a href="/api/annotated?page={r['page']}" target="_blank">image</a></td>
        </tr>
        """)

    deferred_rows = []
    for r in data["deferred_candidates"]:
        deferred_rows.append(f"""
        <tr>
          <td>{r['page']}</td>
          <td>{r['number']}</td>
          <td>{r['confidence']:.2f}</td>
          <td>{r['reason']}</td>
          <td><a href="/api/analyze?page={r['page']}" target="_blank">json</a></td>
          <td><a href="/api/annotated?page={r['page']}" target="_blank">image</a></td>
        </tr>
        """)

    start_rows = []
    for r in data["structure_only_start_candidates"]:
        start_rows.append(f"""
        <tr>
          <td>{r['page']}</td>
          <td>{r['expected_number']}</td>
          <td>{r['confidence']:.2f}</td>
          <td>{r['reason']}</td>
          <td><a href="/api/analyze?page={r['page']}" target="_blank">json</a></td>
          <td><a href="/api/annotated?page={r['page']}" target="_blank">image</a></td>
        </tr>
        """)

    missing_links = []
    for n in data["missing_numbers"]:
        missing_links.append(
            f'<a class="btn" href="/debug/missing-bag?bag_number={n}&{query_string}">Inspect bag {n}</a>'
        )

    expected_next_link = ""
    if data["expected_next"] <= end_number:
        expected_next_link = (
            f'<a class="btn" href="/debug/missing-bag?bag_number={data["expected_next"]}&{query_string}">'
            f'Inspect expected next ({data["expected_next"]})</a>'
        )

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Sequential scan</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 16px;
          background: #f2f2f2;
        }}
        .card {{
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 12px;
          margin-bottom: 16px;
        }}
        table {{
          border-collapse: collapse;
          width: 100%;
          background: white;
        }}
        th, td {{
          border: 1px solid #ddd;
          padding: 8px;
          text-align: left;
          font-size: 14px;
        }}
        th {{
          background: #eee;
        }}
        .btn {{
          display: inline-block;
          padding: 8px 12px;
          background: #222;
          color: #fff;
          text-decoration: none;
          border-radius: 6px;
          margin-right: 8px;
          margin-bottom: 8px;
        }}
      </style>
    </head>
    <body>
      <div class="card">
        <a class="btn" href="/debug/top-bags">Back to top bags</a>
        <a class="btn" href="/api/sequence-scan?{query_string}" target="_blank">Raw JSON sequence scan</a>
        {expected_next_link}
        <h2>Sequential scan</h2>
        <p><strong>Accepted sequence:</strong> {[r['number'] for r in data['accepted_sequence']]}</p>
        <p><strong>Scan range:</strong> {start_number}..{end_number}</p>
        <p><strong>Minimum confidence:</strong> {min_confidence:.2f}</p>
        <p><strong>Expected next:</strong> {data['expected_next']}</p>
        <p><strong>Missing numbers {start_number}..{end_number}:</strong> {data['missing_numbers']}</p>
        <p><strong>Inspect missing gaps:</strong> {' '.join(missing_links) if missing_links else 'none'}</p>
      </div>

      <div class="card">
        <h3>Accepted sequence</h3>
        <table>
          <thead>
            <tr>
              <th>Page</th>
              <th>Bag number</th>
              <th>Confidence</th>
              <th>Reason</th>
              <th>JSON</th>
              <th>Image</th>
            </tr>
          </thead>
          <tbody>
            {''.join(accepted_rows)}
          </tbody>
        </table>
      </div>

      <div class="card">
        <h3>Deferred candidates</h3>
        <table>
          <thead>
            <tr>
              <th>Page</th>
              <th>Bag number</th>
              <th>Confidence</th>
              <th>Reason</th>
              <th>JSON</th>
              <th>Image</th>
            </tr>
          </thead>
          <tbody>
            {''.join(deferred_rows)}
          </tbody>
        </table>
      </div>

      <div class="card">
        <h3>Structure-only start candidates</h3>
        <table>
          <thead>
            <tr>
              <th>Page</th>
              <th>Expected number</th>
              <th>Confidence</th>
              <th>Reason</th>
              <th>JSON</th>
              <th>Image</th>
            </tr>
          </thead>
          <tbody>
            {''.join(start_rows)}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """


def _make_annotated_thumbnail_data_url(page: int, max_width: int = 200):
    img = cv2.imread(page_path(page))
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return None

    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/debug/bag-thumbs", response_class=HTMLResponse)
def debug_bag_thumbs(
    start_number: int = Query(1, ge=1),
    end_number: int = Query(24, ge=1),
    min_confidence: float = Query(0.70, ge=0.0, le=1.0),
    allow_structure_only_start: bool = Query(True),
):
    if end_number < start_number:
        raise HTTPException(status_code=400, detail="end_number must be >= start_number")

    data = run_sequential_scan(
        start_number=start_number,
        end_number=end_number,
        min_confidence=min_confidence,
        allow_structure_only_start=allow_structure_only_start,
    )
    query_string = (
        f"start_number={start_number}&"
        f"end_number={end_number}&"
        f"min_confidence={min_confidence:.2f}&"
        f"allow_structure_only_start={str(allow_structure_only_start).lower()}"
    )

    tiles = []
    for item in data["accepted_sequence"]:
        page = item["page"]
        number = item["number"]
        thumb_url = _make_annotated_thumbnail_data_url(page, max_width=200)
        if thumb_url is None:
            continue

        tiles.append(f"""
        <a class="thumb-card" href="/api/annotated?page={page}" target="_blank">
          <img src="{thumb_url}" alt="Bag {number} page {page}" loading="lazy" />
          <div class="thumb-meta">Bag {number}</div>
          <div class="thumb-sub">Page {page}</div>
        </a>
        """)

        tiles.append(f"""
        <a class="thumb-card" href="/api/annotated?page={page}" target="_blank">
          <img src="{thumb_url}" alt="Bag {number} page {page}" loading="lazy" />
          <div class="thumb-meta">Bag {number}</div>
          <div class="thumb-sub">Page {page}</div>
        </a>
        """)

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Bag Thumbnails</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 16px;
          background: #f2f2f2;
        }}
        .card {{
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 12px;
          margin-bottom: 16px;
        }}
        .btn {{
          display: inline-block;
          padding: 8px 12px;
          background: #222;
          color: #fff;
          text-decoration: none;
          border-radius: 6px;
          margin-right: 8px;
          margin-bottom: 8px;
        }}
        .thumb-grid {{
          display: flex;
          flex-wrap: wrap;
          gap: 16px;
        }}
        .thumb-card {{
          width: 200px;
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 10px;
          text-decoration: none;
          color: #111;
          box-sizing: border-box;
        }}
        .thumb-card img {{
          display: block;
          width: 100%;
          max-width: 200px;
          height: auto;
          border: 1px solid #ccc;
          border-radius: 4px;
          margin-bottom: 8px;
          background: #fafafa;
        }}
        .thumb-meta {{
          font-weight: bold;
          margin-bottom: 4px;
        }}
        .thumb-sub {{
          color: #444;
          font-size: 14px;
        }}
      </style>
    </head>
    <body>
      <div class="card">
        <a class="btn" href="/debug/sequential?{query_string}">Back to sequential scan</a>
        <a class="btn" href="/api/sequence-scan?{query_string}" target="_blank">Raw JSON sequence scan</a>
        <h2>Accepted Bag Start Thumbnails</h2>
        <p><strong>Accepted sequence:</strong> {[item['number'] for item in data['accepted_sequence']]}</p>
        <p><strong>Accepted pages:</strong> {len(data['accepted_sequence'])}</p>
      </div>

      <div class="thumb-grid">
        {''.join(tiles) if tiles else '<div class="card">No accepted bag starts found.</div>'}
      </div>
    </body>
    </html>
    """


@app.get("/debug/bags", response_class=HTMLResponse)
def debug_bags(
    start_number: int = Query(1, ge=1),
    end_number: int = Query(24, ge=1),
    min_confidence: float = Query(0.70, ge=0.0, le=1.0),
    allow_structure_only_start: bool = Query(True),
):
    if end_number < start_number:
        raise HTTPException(status_code=400, detail="end_number must be >= start_number")

    payload = build_bag_ranges_payload(
        start_number=start_number,
        end_number=end_number,
        min_confidence=min_confidence,
        allow_structure_only_start=allow_structure_only_start,
    )
    query_string = (
        f"start_number={start_number}&"
        f"end_number={end_number}&"
        f"min_confidence={min_confidence:.2f}&"
        f"allow_structure_only_start={str(allow_structure_only_start).lower()}"
    )

    accepted_rows = []
    for item in payload["accepted_sequence"]:
        accepted_rows.append(f"""
        <tr>
          <td>{item['number']}</td>
          <td>{item['page']}</td>
          <td><a href="/api/analyze?page={item['page']}" target="_blank">json</a></td>
          <td><a href="/api/annotated?page={item['page']}" target="_blank">image</a></td>
        </tr>
        """)

    range_rows = []
    for item in payload["bag_ranges"]:
        range_rows.append(f"""
        <tr>
          <td>{item['bag_number']}</td>
          <td>{item['start_page']}</td>
          <td>{item['end_page']}</td>
          <td><a href="/api/analyze?page={item['start_page']}" target="_blank">json</a></td>
          <td><a href="/api/annotated?page={item['start_page']}" target="_blank">image</a></td>
        </tr>
        """)

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Bag ranges</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 16px;
          background: #f2f2f2;
        }}
        .card {{
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 12px;
          margin-bottom: 16px;
        }}
        table {{
          border-collapse: collapse;
          width: 100%;
          background: white;
        }}
        th, td {{
          border: 1px solid #ddd;
          padding: 8px;
          text-align: left;
          font-size: 14px;
        }}
        th {{
          background: #eee;
        }}
        .btn {{
          display: inline-block;
          padding: 8px 12px;
          background: #222;
          color: #fff;
          text-decoration: none;
          border-radius: 6px;
          margin-right: 8px;
          margin-bottom: 8px;
        }}
      </style>
    </head>
    <body>
      <div class="card">
        <a class="btn" href="/debug/sequential?{query_string}">Back to sequential scan</a>
        <a class="btn" href="/api/bag-ranges?{query_string}" target="_blank">Raw JSON bag ranges</a>
        <h2>Bag ranges</h2>
        <p><strong>Accepted sequence:</strong> {[item['number'] for item in payload['accepted_sequence']]}</p>
        <p><strong>Missing numbers:</strong> {payload['missing_numbers']}</p>
        <p><strong>Last page:</strong> {payload['last_page']}</p>
      </div>

      <div class="card">
        <h3>Accepted sequence</h3>
        <table>
          <thead>
            <tr>
              <th>Bag number</th>
              <th>Start page</th>
              <th>JSON</th>
              <th>Image</th>
            </tr>
          </thead>
          <tbody>
            {''.join(accepted_rows)}
          </tbody>
        </table>
      </div>

      <div class="card">
        <h3>Bag ranges</h3>
        <table>
          <thead>
            <tr>
              <th>Bag number</th>
              <th>Start page</th>
              <th>End page</th>
              <th>JSON</th>
              <th>Image</th>
            </tr>
          </thead>
          <tbody>
            {''.join(range_rows)}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """


@app.get("/debug/missing-bag", response_class=HTMLResponse)
def debug_missing_bag(
    bag_number: int = Query(..., ge=1),
    start_number: int = Query(1, ge=1),
    end_number: int = Query(24, ge=1),
    min_confidence: float = Query(0.70, ge=0.0, le=1.0),
    allow_structure_only_start: bool = Query(True),
):
    if end_number < start_number:
        raise HTTPException(status_code=400, detail="end_number must be >= start_number")
    if bag_number < start_number or bag_number > end_number:
        raise HTTPException(status_code=400, detail="bag_number must be inside the scan range")

    data, gap = analyze_missing_bag_debug(
        bag_number,
        start_number=start_number,
        end_number=end_number,
        min_confidence=min_confidence,
        allow_structure_only_start=allow_structure_only_start,
    )
    if gap is None:
        raise HTTPException(status_code=404, detail="Could not build missing-bag debug output")

    query_string = (
        f"bag_number={bag_number}&"
        f"start_number={start_number}&"
        f"end_number={end_number}&"
        f"min_confidence={min_confidence:.2f}&"
        f"allow_structure_only_start={str(allow_structure_only_start).lower()}"
    )

    candidate_rows = []
    for r in gap["candidate_pages"]:
        label = f"bag {r['bag_number']}" if r["bag_number"] is not None else ""
        candidate_rows.append(f"""
        <tr>
          <td>{r['page']}</td>
          <td>{r['confidence']:.2f}</td>
          <td>{'Y' if r['panel_found'] else ''}</td>
          <td>{r.get('panel_source') or ''}</td>
          <td>{'Y' if r['shell_found'] else ''}</td>
          <td>{r.get('shell_method') or ''}</td>
          <td>{'Y' if r['grey_bag_found'] else ''}</td>
          <td>{'Y' if r['number_box_found'] else ''}</td>
          <td>{label}</td>
          <td>{r['ocr_raw']}</td>
          <td><a href="/api/analyze?page={r['page']}" target="_blank">json</a></td>
          <td><a href="/api/annotated?page={r['page']}" target="_blank">image</a></td>
        </tr>
        """)

    accepted_sequence = [item['number'] for item in data['accepted_sequence']]
    prev_text = "none"
    if gap['previous_accepted'] is not None:
        prev_text = f"bag {gap['previous_accepted']['number']} on page {gap['previous_accepted']['page']}"
    next_text = "none"
    if gap['next_detected_higher'] is not None:
        next_text = f"bag {gap['next_detected_higher']['number']} on page {gap['next_detected_higher']['page']}"

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Bag gap debug</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 16px;
          background: #f2f2f2;
        }}
        .card {{
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 12px;
          margin-bottom: 16px;
        }}
        table {{
          border-collapse: collapse;
          width: 100%;
          background: white;
        }}
        th, td {{
          border: 1px solid #ddd;
          padding: 8px;
          text-align: left;
          font-size: 14px;
        }}
        th {{
          background: #eee;
        }}
        .btn {{
          display: inline-block;
          padding: 8px 12px;
          background: #222;
          color: #fff;
          text-decoration: none;
          border-radius: 6px;
          margin-right: 8px;
          margin-bottom: 8px;
        }}
      </style>
    </head>
    <body>
      <div class="card">
        <a class="btn" href="/debug/sequential?start_number={start_number}&end_number={end_number}&min_confidence={min_confidence:.2f}&allow_structure_only_start={str(allow_structure_only_start).lower()}">Back to sequential scan</a>
        <a class="btn" href="/api/missing-bag-debug?{query_string}" target="_blank">Raw JSON gap debug</a>
        <h2>Bag {bag_number} debug</h2>
        <p><strong>Status:</strong> {gap['status']}</p>
        <p><strong>Accepted sequence:</strong> {accepted_sequence}</p>
        <p><strong>Previous accepted:</strong> {prev_text}</p>
        <p><strong>Next detected higher:</strong> {next_text}</p>
        <p><strong>Likely range:</strong> pages {gap['likely_range']['start_page']}..{gap['likely_range']['end_page']}</p>
        <p><strong>Candidate pages in range:</strong> {gap['candidate_count']}</p>
      </div>

      <div class="card">
        <table>
          <thead>
            <tr>
              <th>Page</th>
              <th>Confidence</th>
              <th>Panel</th>
              <th>Panel source</th>
              <th>Shell</th>
              <th>Shell method</th>
              <th>Grey bag</th>
              <th>Number crop</th>
              <th>Bag number</th>
              <th>OCR raw</th>
              <th>JSON</th>
              <th>Image</th>
            </tr>
          </thead>
          <tbody>
            {''.join(candidate_rows)}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """

@app.get("/debug/all-pages", response_class=HTMLResponse)
def debug_all_pages(start: int = 1, end: int = 150):
    imgs = []
    for p in range(start, end + 1):
        imgs.append(f"""
        <div style="margin-bottom:20px;">
            <h3>Page {p}</h3>
            <img src="/api/annotated?page={p}" style="width:100%; border:1px solid #ccc;" />
        </div>
        """)

    return f"""
    <html>
    <body style="font-family:Arial; background:#111; color:white;">
        <h1>All Pages {start} → {end}</h1>
        {''.join(imgs)}
    </body>
    </html>
    """
    
@app.get("/api/number-crop")
def api_number_crop(page: int = Query(...)):
    result = analyze_page(page, include_image=False)

    if not result["number_box_found"]:
        raise HTTPException(status_code=404, detail="Number crop not found")

    img = cv2.imread(page_path(page))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read image")

    number_img = crop(img, tuple(result["number_box"]))
    ok, buf = cv2.imencode(".png", number_img)
    if not ok:
        raise HTTPException(status_code=500, detail="Encode failed")

    return Response(content=buf.tobytes(), media_type="image/png")    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
