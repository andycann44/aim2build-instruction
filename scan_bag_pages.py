#!/usr/bin/env python3
import os
import cv2
import numpy as np

BASE = "/Users/olly/aim2build-instruction/debug/21330/21330_01/pages"
OUT = "/Users/olly/aim2build-instruction/debug/output"

os.makedirs(OUT, exist_ok=True)


def crop(img, box):
    x, y, w, h = box
    return img[y:y+h, x:x+w]


def draw_box(img, box, color, t=3):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x+w, y+h), color, t)


def find_bag_intro_panel(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # reject dark promo/lifestyle pages
    if float(np.mean(gray)) < 165:
        return None

    # only search where the bag intro lives
    sx1, sy1 = 0, 0
    sx2, sy2 = int(w * 0.72), int(h * 0.45)
    search = gray[sy1:sy2, sx1:sx2]

    # very bright areas only
    th = cv2.threshold(search, 236, 255, cv2.THRESH_BINARY)[1]
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

        # big wide intro panel only
        if area_ratio < 0.08 or area_ratio > 0.34:
            continue
        if wh_ratio < 1.4 or wh_ratio > 5.0:
            continue

        # must start near top-left
        if x > int(w * 0.08):
            continue
        if y > int(h * 0.10):
            continue

        roi = gray[y:y+bh, x:x+bw]
        if roi.size == 0:
            continue

        roi_mean = float(np.mean(roi))
        roi_std = float(np.std(roi))

        # should be a bright white panel overall
        if roi_mean < 215:
            continue

        # split into left / middle / right zones
        left = roi[:, :int(bw * 0.28)]
        mid = roi[:, int(bw * 0.28):int(bw * 0.45)]
        right = roi[:, int(bw * 0.45):]

        if left.size == 0 or mid.size == 0 or right.size == 0:
            continue

        left_mean = float(np.mean(left))
        right_mean = float(np.mean(right))
        left_std = float(np.std(left))
        right_std = float(np.std(right))

        # bag area on left should be darker/more textured than panel background
        if left_mean > 235:
            continue
        if left_std < 12:
            continue

        # right side should contain build preview, so not blank
        if right_std < 10:
            continue

        # middle should contain arrow-like structure, so some edges
        mid_edges = cv2.Canny(mid, 40, 140)
        mid_edge_density = float(np.mean(mid_edges > 0))
        if mid_edge_density < 0.015:
            continue

        # normal build pages with a parts box often fail this contrast pattern
        contrast_gap = right_mean - left_mean
        if contrast_gap < 8:
            continue

        score = 0.0
        score += left_std * 1.8
        score += right_std * 0.8
        score += mid_edge_density * 1200.0
        score += contrast_gap * 2.0
        score += roi_std * 0.6
        score -= abs(area_ratio - 0.18) * 400.0

        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)

    return best


for file in sorted(os.listdir(BASE)):
    if not file.endswith(".png"):
        continue

    path = os.path.join(BASE, file)
    img = cv2.imread(path)
    if img is None:
        continue

    vis = img.copy()

    panel = find_bag_intro_panel(img)

    if panel is not None:
        draw_box(vis, panel, (0, 255, 0), 4)
        cv2.putText(
            vis,
            "BAG PANEL",
            (panel[0], max(30, panel[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            vis,
            "NO PANEL",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    cv2.imwrite(os.path.join(OUT, file), vis)

print("\nDONE - strict panel detection only. Check debug/output/\n")