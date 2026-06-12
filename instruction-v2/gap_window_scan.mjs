import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { detectVisibleBagNumber } from "./bag_number_recognizer.mjs";


function argValue(name) {
  const idx = process.argv.indexOf(name);
  if (idx === -1 || idx + 1 >= process.argv.length) return null;
  return process.argv[idx + 1];
}


function componentBox(pixels, width, height, startX, startY, seen) {
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
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;

    for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const idx = ny * width + nx;
      if (!pixels[idx] || seen[idx]) continue;
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


function groupNumberLikeComponents(components) {
  const groups = [];
  for (const component of components.sort((a, b) => a.y - b.y || a.x - b.x)) {
    let matched = null;
    for (const group of groups) {
      const yOverlap = Math.min(component.y + component.h, group.y + group.h) - Math.max(component.y, group.y);
      const minHeight = Math.max(1, Math.min(component.h, group.h));
      const gap = component.x - (group.x + group.w);
      if (yOverlap / minHeight >= 0.42 && gap >= -6 && gap <= Math.max(24, group.h * 0.55)) {
        matched = group;
        break;
      }
    }
    if (matched) {
      const x2 = Math.max(matched.x + matched.w, component.x + component.w);
      const y2 = Math.max(matched.y + matched.h, component.y + component.h);
      matched.x = Math.min(matched.x, component.x);
      matched.y = Math.min(matched.y, component.y);
      matched.w = x2 - matched.x;
      matched.h = y2 - matched.y;
      matched.components += 1;
      matched.pixels += component.pixels;
    } else {
      groups.push({ ...component, components: 1 });
    }
  }
  return groups;
}


function paleBackgroundScore(raw, width, height, group) {
  const padX = Math.max(10, Math.round(group.w * 0.55));
  const padY = Math.max(10, Math.round(group.h * 0.55));
  const x1 = Math.max(0, group.x - padX);
  const y1 = Math.max(0, group.y - padY);
  const x2 = Math.min(width, group.x + group.w + padX);
  const y2 = Math.min(height, group.y + group.h + padY);
  let pale = 0;
  let total = 0;

  for (let y = y1; y < y2; y += 1) {
    for (let x = x1; x < x2; x += 1) {
      if (x >= group.x && x < group.x + group.w && y >= group.y && y < group.y + group.h) continue;
      const idx = (y * width + x) * 3;
      const r = raw[idx];
      const g = raw[idx + 1];
      const b = raw[idx + 2];
      const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
      const spread = Math.max(r, g, b) - Math.min(r, g, b);
      if (luminance > 145 && spread < 90) pale += 1;
      total += 1;
    }
  }

  return total ? pale / total : 0;
}


async function analyzeGapPage(sharp, repoRoot, pageEntry, windowStart, windowEnd) {
  const imagePath = path.join(repoRoot, pageEntry.image_path);
  const metadata = await sharp(imagePath).metadata();
  const sourceWidth = metadata.width || 0;
  const sourceHeight = metadata.height || 0;
  const width = 420;
  const height = Math.max(1, Math.round((sourceHeight / Math.max(1, sourceWidth)) * width));
  const raw = await sharp(imagePath)
    .resize({ width, height, fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer();
  const bagNumber = await detectVisibleBagNumber(sharp, imagePath);

  const dark = new Uint8Array(width * height);
  const orange = new Uint8Array(width * height);
  for (let i = 0; i < dark.length; i += 1) {
    const r = raw[i * 3];
    const g = raw[i * 3 + 1];
    const b = raw[i * 3 + 2];
    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
    const spread = Math.max(r, g, b) - Math.min(r, g, b);
    if (luminance < 70 && spread < 95) dark[i] = 1;
    if (r > 195 && g > 45 && g < 190 && b < 150 && r - g > 30) orange[i] = 1;
  }

  const darkComponents = findComponents(dark, width, height, 20);
  const orangeComponents = findComponents(orange, width, height, 35)
    .filter((box) => box.w >= 8 && box.h >= 5 && box.y < height * 0.86);

  const numberLikeGroups = groupNumberLikeComponents(
    darkComponents.filter((box) => {
      const aspect = box.w / Math.max(1, box.h);
      if (box.y > height * 0.90) return false;
      return box.h >= 14 && box.h <= 90 && box.w >= 5 && box.w <= 115 && aspect >= 0.10 && aspect <= 3.6;
    })
  ).filter((group) => {
    const aspect = group.w / Math.max(1, group.h);
    group.pale_label_background_score = Number(paleBackgroundScore(raw, width, height, group).toFixed(4));
    group.center_score = Number((1 - Math.min(1, Math.abs((group.x + group.w / 2) - width / 2) / (width / 2))).toFixed(4));
    return group.h >= 18 && group.w >= 8 && aspect >= 0.12 && aspect <= 3.8 && group.pale_label_background_score >= 0.22;
  });

  const strongNumberGroups = numberLikeGroups.filter((group) => group.h >= 30 || group.pixels >= 300);
  const centralNumberGroups = strongNumberGroups.filter((group) => group.center_score >= 0.35);
  const orangeScore = Math.min(1, orangeComponents.length / 2);
  const numberScore = Math.min(1, strongNumberGroups.length / 2);
  const centralScore = Math.min(1, centralNumberGroups.length);
  const structureScore = Math.min(1, (orangeScore * 0.45) + (numberScore * 0.30) + (centralScore * 0.25));
  const span = Math.max(1, windowEnd - windowStart);
  const earlyBias = 1 - Math.min(1, Math.max(0, Number(pageEntry.page) - windowStart) / span);

  let score = 0;
  const reasons = [];
  if (orangeComponents.length) {
    score += 35 * orangeScore;
    reasons.push("+ orange_arrow_components");
  } else {
    score -= 10;
    reasons.push("- no_orange_arrow_components");
  }
  if (strongNumberGroups.length) {
    score += 28 * numberScore;
    reasons.push("+ large_number_like_groups");
  } else {
    score -= 8;
    reasons.push("- no_large_number_like_groups");
  }
  if (centralNumberGroups.length) {
    score += 22 * centralScore;
    reasons.push("+ central_number_like_group");
  }
  score += 12 * earlyBias;
  if (earlyBias >= 0.66) reasons.push("+ near_window_start");
  if (earlyBias <= 0.20) {
    score -= 6;
    reasons.push("- late_in_window");
  }
  if (orangeComponents.length && strongNumberGroups.length) {
    score += 18;
    reasons.push("+ combined_arrow_and_number_structure");
  }
  if (!orangeComponents.length && !strongNumberGroups.length) {
    score -= 18;
    reasons.push("- looks_like_normal_instruction_page");
  }

  return {
    page: Number(pageEntry.page),
    image_path: pageEntry.image_path,
    score: Number(score.toFixed(3)),
    confidence: Number(Math.max(0, Math.min(0.99, structureScore)).toFixed(4)),
    reasons,
    signals: {
      orange_arrow_component_count: orangeComponents.length,
      dark_component_count: darkComponents.length,
      large_number_group_count: strongNumberGroups.length,
      central_number_group_count: centralNumberGroups.length,
      early_window_bias: Number(earlyBias.toFixed(4)),
      structure_score: Number(structureScore.toFixed(4)),
      scan_region: { left: 0, top: 0, width: sourceWidth, height: sourceHeight },
    },
    number_like_groups: strongNumberGroups
      .sort((a, b) => b.pixels - a.pixels)
      .slice(0, 8),
    detected_bag_number: bagNumber.detected_bag_number,
    detected_bag_number_confidence: bagNumber.confidence,
    detected_bag_number_source: bagNumber.source,
    detected_bag_number_box: bagNumber.number_box,
    detected_bag_number_ocr_raw: bagNumber.ocr_raw,
    detected_bag_number_candidates: bagNumber.candidates,
    analysis_source: "instruction_v2_gap_window_scan",
  };
}


const pageIndexPath = argValue("--page-index");
const windowsPath = argValue("--windows");
const repoRoot = argValue("--repo-root");
const outPath = argValue("--out");
const nodeModules = argValue("--node-modules");

if (!pageIndexPath || !windowsPath || !repoRoot || !outPath || !nodeModules) {
  console.error("Usage: gap_window_scan.mjs --page-index <path> --windows <path> --repo-root <dir> --out <path> --node-modules <dir>");
  process.exit(2);
}

const requireFromBundle = createRequire(path.join(nodeModules, "package.json"));
const sharp = requireFromBundle("sharp");
const pageIndex = JSON.parse(fs.readFileSync(pageIndexPath, "utf8"));
const windowsPayload = JSON.parse(fs.readFileSync(windowsPath, "utf8"));
const pagesByNumber = new Map((pageIndex.pages || []).map((entry) => [Number(entry.page), entry]));

const scanByWindow = {};
for (const window of windowsPayload.gap_windows || []) {
  const bounds = window.window || {};
  const startPage = Number(bounds.start_page);
  const endPage = Number(bounds.end_page);
  const candidates = [];
  for (let page = startPage; page <= endPage; page += 1) {
    const pageEntry = pagesByNumber.get(page);
    if (!pageEntry) continue;
    candidates.push(await analyzeGapPage(sharp, repoRoot, pageEntry, startPage, endPage));
  }
  candidates.sort((a, b) => b.score - a.score || a.page - b.page);
  scanByWindow[window.window_id] = candidates.slice(0, 12);
}

fs.writeFileSync(outPath, `${JSON.stringify({ scan_by_window: scanByWindow }, null, 2)}\n`, "utf8");
