from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
import os
import cv2
import numpy as np
import pytesseract

app = FastAPI()

BASE = "/Users/olly/aim2build-instruction/debug/21330/21330_01/pages"
REFERENCE_PAGE = 28


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
    global SHELL_TEMPLATE

    if panel_img is None or panel_img.size == 0:
        return None, None, "none"

    if SHELL_TEMPLATE is None:
        SHELL_TEMPLATE = make_shell_template()
        if SHELL_TEMPLATE is None:
            fallback_box, fallback_score = _find_square_bag_box_outline_fallback(panel_img)
            if fallback_box is not None:
                return fallback_box, fallback_score, "outline_fallback"
            return None, None, "none"

    ph, pw = panel_img.shape[:2]
    gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY)

    sx1, sy1 = 0, 0
    sx2, sy2 = int(pw * 0.42), int(ph * 0.95)
    search = gray[sy1:sy2, sx1:sx2]

    search_edges = cv2.Canny(search, 35, 130)

    th, tw = SHELL_TEMPLATE["template_h"], SHELL_TEMPLATE["template_w"]
    max_val = None

    if search_edges.shape[0] >= th and search_edges.shape[1] >= tw:
        result = cv2.matchTemplate(
            search_edges,
            SHELL_TEMPLATE["template_edges"],
            cv2.TM_CCOEFF_NORMED,
        )

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= 0.25:
            x, y = max_loc
            return (x + sx1, y + sy1, tw, th), float(max_val), "template"

    fallback_box, fallback_score = _find_square_bag_box_outline_fallback(panel_img)
    if fallback_box is not None:
        return fallback_box, fallback_score, "outline_fallback"

    return None, (float(max_val) if max_val is not None else None), "template"


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
def find_number_crop(bag_img):
    if bag_img is None or bag_img.size == 0:
        return None

    h, w = bag_img.shape[:2]
    x = int(w * 0.20)
    y = int(h * 0.20)
    bw = int(w * 0.60)
    bh = int(h * 0.55)
    return (x, y, bw, bh)


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
            # boost strong matches
            score += min(shell_score, 1.0) * 0.15

    # Stage 3: grey bag (IMPORTANT FILTER)
    if grey_bag_found:
        score += 0.25

        if grey_bag_score is not None:
            # stabilise score (avoid huge values)
            norm = max(0.0, min(1.0, (grey_bag_score + 40) / 80))
            score += norm * 0.15

    # Stage 4: number crop
    if number_box_found:
        score += 0.10

    # Stage 5: OCR validity
    if bag_number is not None:
        score += 0.25

    return round(min(score, 1.0), 3)


# --------------------------------------------------
# PAGE ANALYSIS
# --------------------------------------------------
def analyze_page(page: int, include_image: bool = True):
    path = page_path(page)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Page not found")

    img = cv2.imread(path)
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read image")

    vis = img.copy()

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
                number_rel = find_number_crop(bag_img)

                if number_rel is not None:
                    gx2, gy2, _, _ = bag_abs
                    nx, ny, nw, nh = number_rel
                    number_abs = (gx2 + nx, gy2 + ny, nw, nh)

                    if include_image:
                        draw_box(vis, number_abs, (0, 0, 255), 2)

                    number_img = crop(img, number_abs)
                    bag_number, ocr_raw = read_bag_number(number_img)

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


def build_sequence_scan_rows():
    rows = []
    for page in sorted(list_pages()):
        r = analyze_page(page, include_image=False)

        strong_structure = (
            r["panel_found"] and
            r["shell_found"] and
            r["grey_bag_found"]
        )

        r["strong_structure"] = strong_structure
        rows.append(r)

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
            "strong_structure": strong_structure,
            "action": "no_number_detected" if detected_number is None else "ignored_non_bag_signal",
        }

        if detected_number is not None and (detected_number < start_number or detected_number > end_number):
            debug_entry["action"] = "ignored_out_of_scan_range"
            page_debug_log.append(debug_entry)
            continue

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

        if (
            not accepted
            and detected_number is None
            and strong_structure
            and allow_structure_only_start
            and expected_next == start_number
            and confidence >= min_confidence
        ):
            structure_only_start_candidates.append({
                "page": page,
                "expected_number": start_number,
                "confidence": confidence,
                "reason": "strong_structure_no_number_start_candidate",
            })
            accepted.append({
                "page": page,
                "number": start_number,
                "bag_number": start_number,
                "confidence": confidence,
                "reason": "accepted_from_structure_start_fallback",
            })
            expected_next = start_number + 1
            debug_entry["action"] = "accepted_from_structure_start_fallback"
            debug_entry["expected_next_after"] = expected_next
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

        if detected_number is None:
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
        "viewer": "/debug/compare?a=28&b=24",
        "scan_all": "/api/scan-all",
        "scan_top": "/debug/top-bags",
        "sequence_scan": "/api/sequence-scan",
        "sequence_debug": "/debug/sequential",
        "bag_ranges": "/api/bag-ranges",
        "bags_debug": "/debug/bags",
        "missing_bag_debug": "/api/missing-bag-debug?bag_number=7",
        "missing_bag_view": "/debug/missing-bag?bag_number=7",
    }


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
