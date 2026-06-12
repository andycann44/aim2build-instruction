import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";


const execFileAsync = promisify(execFile);


function componentBox(mask, width, height, startX, startY, seen) {
  const stack = [[startX, startY]];
  seen[startY * width + startX] = 1;
  let minX = startX;
  let maxX = startX;
  let minY = startY;
  let maxY = startY;
  let count = 0;

  while (stack.length) {
    const [x, y] = stack.pop();
    count += 1;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);

    for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const idx = ny * width + nx;
      if (!mask[idx] || seen[idx]) continue;
      seen[idx] = 1;
      stack.push([nx, ny]);
    }
  }

  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1, pixels: count };
}


function findComponents(mask, width, height, minPixels = 1) {
  const seen = new Uint8Array(mask.length);
  const components = [];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (!mask[idx] || seen[idx]) continue;
      const box = componentBox(mask, width, height, x, y, seen);
      if (box.pixels >= minPixels) components.push(box);
    }
  }
  return components;
}


function findOrangeArrowComponents(rawRgb, width, height) {
  const orange = new Uint8Array(width * height);
  for (let i = 0; i < orange.length; i += 1) {
    const r = rawRgb[i * 3];
    const g = rawRgb[i * 3 + 1];
    const b = rawRgb[i * 3 + 2];
    if (r > 195 && g > 45 && g < 190 && b < 155 && r - g > 28) orange[i] = 1;
  }
  return findComponents(orange, width, height, 35)
    .filter((box) => box.w >= 8 && box.h >= 5 && box.y < height * 0.80);
}


function arrowRelationshipScore(candidate, arrows) {
  const cx = candidate.x + candidate.w / 2;
  const cy = candidate.y + candidate.h / 2;
  let bestScore = 0;
  let bestArrow = null;
  for (const arrow of arrows) {
    const ax = arrow.x + arrow.w / 2;
    const ay = arrow.y + arrow.h / 2;
    const dx = ax - cx;
    const dy = Math.abs(ay - cy);
    if (dx < 10 || dx > 320) continue;
    if (dy > 120) continue;
    const distanceScore = 1 - Math.min(1, Math.abs(dx - 125) / 180);
    const yScore = 1 - Math.min(1, dy / 120);
    const score = (distanceScore * 0.7) + (yScore * 0.3);
    if (score > bestScore) {
      bestScore = score;
      bestArrow = arrow;
    }
  }
  return {
    score: Number(bestScore.toFixed(4)),
    arrow: bestArrow ? { x: bestArrow.x, y: bestArrow.y, w: bestArrow.w, h: bestArrow.h } : null,
  };
}


function holeCountForComponent(mask, width, height, box) {
  const pad = 2;
  const x1 = Math.max(0, box.x - pad);
  const y1 = Math.max(0, box.y - pad);
  const x2 = Math.min(width, box.x + box.w + pad);
  const y2 = Math.min(height, box.y + box.h + pad);
  const localW = x2 - x1;
  const localH = y2 - y1;
  const bg = new Uint8Array(localW * localH);

  for (let y = 0; y < localH; y += 1) {
    for (let x = 0; x < localW; x += 1) {
      const sourceIdx = (y1 + y) * width + (x1 + x);
      bg[y * localW + x] = mask[sourceIdx] ? 0 : 1;
    }
  }

  const seen = new Uint8Array(bg.length);
  const queue = [];
  for (let x = 0; x < localW; x += 1) {
    for (const y of [0, localH - 1]) {
      const idx = y * localW + x;
      if (bg[idx] && !seen[idx]) {
        seen[idx] = 1;
        queue.push([x, y]);
      }
    }
  }
  for (let y = 0; y < localH; y += 1) {
    for (const x of [0, localW - 1]) {
      const idx = y * localW + x;
      if (bg[idx] && !seen[idx]) {
        seen[idx] = 1;
        queue.push([x, y]);
      }
    }
  }
  while (queue.length) {
    const [x, y] = queue.pop();
    for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (nx < 0 || nx >= localW || ny < 0 || ny >= localH) continue;
      const idx = ny * localW + nx;
      if (!bg[idx] || seen[idx]) continue;
      seen[idx] = 1;
      queue.push([nx, ny]);
    }
  }

  let holes = 0;
  for (let y = 1; y < localH - 1; y += 1) {
    for (let x = 1; x < localW - 1; x += 1) {
      const idx = y * localW + x;
      if (!bg[idx] || seen[idx]) continue;
      const hole = componentBox(bg, localW, localH, x, y, seen);
      if (hole.pixels >= 20 && hole.w >= 4 && hole.h >= 4) holes += 1;
    }
  }
  return holes;
}


function rowFill(mask, width, box, relY) {
  const y = Math.max(0, Math.min(box.y + box.h - 1, Math.round(box.y + box.h * relY)));
  let filled = 0;
  for (let x = box.x; x < box.x + box.w; x += 1) {
    if (mask[y * width + x]) filled += 1;
  }
  return filled / Math.max(1, box.w);
}


function quadrantFill(mask, width, box, qx, qy) {
  const x1 = box.x + Math.floor(box.w * qx);
  const x2 = box.x + Math.floor(box.w * (qx + 0.5));
  const y1 = box.y + Math.floor(box.h * qy);
  const y2 = box.y + Math.floor(box.h * (qy + 0.5));
  let filled = 0;
  let total = 0;
  for (let y = y1; y < y2; y += 1) {
    for (let x = x1; x < x2; x += 1) {
      filled += mask[y * width + x] ? 1 : 0;
      total += 1;
    }
  }
  return total ? filled / total : 0;
}


function classifyLargeDigit(mask, width, height, box) {
  const holes = holeCountForComponent(mask, width, height, box);
  const topFill = rowFill(mask, width, box, 0.12);
  const midFill = rowFill(mask, width, box, 0.48);
  const bottomFill = rowFill(mask, width, box, 0.86);
  const upperLeft = quadrantFill(mask, width, box, 0.0, 0.0);
  const lowerLeft = quadrantFill(mask, width, box, 0.0, 0.5);
  const lowerRight = quadrantFill(mask, width, box, 0.5, 0.5);
  const aspect = box.w / Math.max(1, box.h);

  if (holes >= 2) {
    return {
      value: 8,
      confidence: 0.88,
      source: "geometry_two_holes_after_v1_panel_number_ocr_fallback",
      features: { holes, topFill, midFill, bottomFill, upperLeft, lowerLeft, lowerRight, aspect },
    };
  }
  if (topFill >= 0.55 && lowerLeft <= 0.28 && lowerRight >= 0.22) {
    return {
      value: 7,
      confidence: 0.78,
      source: "geometry_top_bar_diagonal_after_v1_panel_number_ocr_fallback",
      features: { holes, topFill, midFill, bottomFill, upperLeft, lowerLeft, lowerRight, aspect },
    };
  }
  if (topFill >= 0.48 && upperLeft >= 0.30 && lowerLeft >= 0.18 && lowerRight >= 0.25) {
    return {
      value: 5,
      confidence: 0.76,
      source: "geometry_five_shape_after_v1_panel_number_ocr_fallback",
      features: { holes, topFill, midFill, bottomFill, upperLeft, lowerLeft, lowerRight, aspect },
    };
  }
  return {
    value: null,
    confidence: 0,
    source: "geometry_unclassified_after_v1_panel_number_ocr_fallback",
    features: { holes, topFill, midFill, bottomFill, upperLeft, lowerLeft, lowerRight, aspect },
  };
}


async function runTesseractDigit(sharp, imagePath, box, tmpDir) {
  const pad = 8;
  const left = Math.max(0, box.x - pad);
  const top = Math.max(0, box.y - pad);
  const width = box.w + pad * 2;
  const height = box.h + pad * 2;
  const outPath = path.join(tmpDir, `bag_number_${box.x}_${box.y}.png`);
  await sharp(imagePath)
    .extract({ left, top, width, height })
    .resize({ width: Math.max(1, width * 8) })
    .grayscale()
    .normalise()
    .threshold(150)
    .png()
    .toFile(outPath);

  for (const psm of ["10", "8", "13", "7"]) {
    try {
      const { stdout } = await execFileAsync("/opt/homebrew/bin/tesseract", [
        outPath,
        "stdout",
        "--psm",
        psm,
        "-c",
        "tessedit_char_whitelist=0123456789",
      ], { timeout: 5000, maxBuffer: 1024 * 32 });
      const digits = String(stdout || "").replace(/\D+/g, "");
      const value = digits ? Number(digits) : NaN;
      if (digits && digits.length <= 2 && Number.isInteger(value) && value >= 1 && value <= 99) {
        return {
          value,
          confidence: 0.92,
          source: `tesseract_panel_digit_psm_${psm}`,
          raw: digits,
        };
      }
    } catch {
      // Fall back to geometry below.
    }
  }
  return null;
}


export async function detectVisibleBagNumber(sharp, imagePath) {
  const metadata = await sharp(imagePath).metadata();
  const width = metadata.width || 0;
  const height = metadata.height || 0;
  if (!width || !height) {
    return { detected_bag_number: null, confidence: 0, source: "missing_image_metadata", candidates: [] };
  }

  const scan = {
    left: 0,
    top: 0,
    width: Math.min(width, Math.round(width * 0.62)),
    height: Math.min(height, Math.round(height * 0.48)),
  };
  const raw = await sharp(imagePath)
    .extract(scan)
    .grayscale()
    .raw()
    .toBuffer();
  const rawRgb = await sharp(imagePath)
    .extract(scan)
    .removeAlpha()
    .raw()
    .toBuffer();

  const mask = new Uint8Array(scan.width * scan.height);
  for (let i = 0; i < mask.length; i += 1) {
    if (raw[i] < 82) mask[i] = 1;
  }
  const orangeArrows = findOrangeArrowComponents(rawRgb, scan.width, scan.height);

  const components = findComponents(mask, scan.width, scan.height, 80)
    .filter((box) => {
      const aspect = box.w / Math.max(1, box.h);
      if (box.h < 38 || box.h > 95) return false;
      if (box.w < 22 || box.w > 75) return false;
      if (aspect < 0.35 || aspect > 1.35) return false;
      if (box.y < 20 || box.y > scan.height * 0.75) return false;
      return true;
    })
    .map((box) => ({
      ...box,
      page_box: { x: scan.left + box.x, y: scan.top + box.y, w: box.w, h: box.h },
    }))
    .sort((a, b) => {
      const centerBiasA = Math.abs((a.x + a.w / 2) - scan.width * 0.18);
      const centerBiasB = Math.abs((b.x + b.w / 2) - scan.width * 0.18);
      return centerBiasA - centerBiasB || b.pixels - a.pixels;
    })
    .slice(0, 6);

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "a2p_v2_bag_number_"));
  try {
    const candidates = [];
    for (const component of components) {
      const geometry = classifyLargeDigit(mask, scan.width, scan.height, component);
      const ocr = geometry.value === null
        ? await runTesseractDigit(sharp, imagePath, component.page_box, tmpDir)
        : null;
      const selected = ocr && Number.isFinite(ocr.value) ? ocr : geometry;
      const arrowRelation = arrowRelationshipScore(component, orangeArrows);
      const selectionScore = (
        (selected.value === null ? 0 : selected.confidence * 100)
        + (arrowRelation.score * 80)
        + Math.min(20, component.pixels / 80)
        - (component.x > scan.width * 0.58 ? 40 : 0)
      );
      candidates.push({
        detected_bag_number: selected.value,
        confidence: selected.confidence,
        source: selected.source,
        ocr_raw: ocr?.raw || "",
        number_box: component.page_box,
        component_pixels: component.pixels,
        arrow_relationship_score: arrowRelation.score,
        arrow_box: arrowRelation.arrow,
        selection_score: Number(selectionScore.toFixed(4)),
        geometry_features: geometry.features,
      });
    }

    const best = candidates
      .filter((candidate) => candidate.detected_bag_number !== null)
      .sort((a, b) => b.selection_score - a.selection_score || b.confidence - a.confidence || b.component_pixels - a.component_pixels)[0];

    return {
      detected_bag_number: best?.detected_bag_number ?? null,
      confidence: best?.confidence ?? 0,
      source: best?.source || "no_visible_bag_number_detected",
      ocr_raw: best?.ocr_raw || "",
      number_box: best?.number_box || null,
      candidates,
    };
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}
