import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { detectVisibleBagNumber } from "./bag_number_recognizer.mjs";


function argValue(name) {
  const idx = process.argv.indexOf(name);
  if (idx === -1 || idx + 1 >= process.argv.length) {
    return null;
  }
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


function groupDigits(components) {
  const groups = [];
  for (const component of components.sort((a, b) => a.y - b.y || a.x - b.x)) {
    let matched = null;
    for (const group of groups) {
      const yOverlap = Math.min(component.y + component.h, group.y + group.h) - Math.max(component.y, group.y);
      const minHeight = Math.max(1, Math.min(component.h, group.h));
      const gap = component.x - (group.x + group.w);
      if (yOverlap / minHeight >= 0.45 && gap >= -4 && gap <= 18) {
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


function labelBackgroundScore(raw, width, height, group) {
  const padX = Math.max(8, Math.round(group.w * 0.45));
  const padY = Math.max(8, Math.round(group.h * 0.45));
  const x1 = Math.max(0, group.x - padX);
  const y1 = Math.max(0, group.y - padY);
  const x2 = Math.min(width, group.x + group.w + padX);
  const y2 = Math.min(height, group.y + group.h + padY);
  let pale = 0;
  let total = 0;

  for (let y = y1; y < y2; y += 1) {
    for (let x = x1; x < x2; x += 1) {
      if (x >= group.x && x < group.x + group.w && y >= group.y && y < group.y + group.h) {
        continue;
      }
      const idx = (y * width + x) * 3;
      const r = raw[idx];
      const g = raw[idx + 1];
      const b = raw[idx + 2];
      const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
      const spread = Math.max(r, g, b) - Math.min(r, g, b);
      if (luminance > 150 && spread < 80) {
        pale += 1;
      }
      total += 1;
    }
  }

  return total > 0 ? pale / total : 0;
}


function countOrangeArrowComponents(raw, width, height) {
  const orange = new Uint8Array(width * height);
  for (let i = 0; i < orange.length; i += 1) {
    const r = raw[i * 3];
    const g = raw[i * 3 + 1];
    const b = raw[i * 3 + 2];
    if (r > 205 && g > 55 && g < 185 && b < 135 && r - g > 35 && g - b > 5) {
      orange[i] = 1;
    }
  }

  const seen = new Uint8Array(orange.length);
  const components = [];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (!orange[idx] || seen[idx]) continue;
      const box = componentBox(orange, width, height, x, y, seen);
      if (box.pixels < 45 || box.w < 8 || box.h < 5) continue;
      components.push(box);
    }
  }
  return components;
}


async function analyzePage(sharp, repoRoot, pageEntry, debugDir) {
  const imagePath = path.join(repoRoot, pageEntry.image_path);
  const metadata = await sharp(imagePath).metadata();
  const width = metadata.width || 0;
  const height = metadata.height || 0;
  const scanWidth = Math.max(1, Math.floor(width * 0.48));
  const scanHeight = Math.max(1, Math.floor(height * 0.94));
  const resizedWidth = 360;
  const resizedHeight = Math.max(1, Math.round((scanHeight / scanWidth) * resizedWidth));

  const raw = await sharp(imagePath)
    .extract({ left: 0, top: 0, width: scanWidth, height: scanHeight })
    .resize({ width: resizedWidth, height: resizedHeight, fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer();

  const dark = new Uint8Array(resizedWidth * resizedHeight);
  for (let i = 0; i < dark.length; i += 1) {
    const r = raw[i * 3];
    const g = raw[i * 3 + 1];
    const b = raw[i * 3 + 2];
    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
    if (luminance < 62 && Math.max(r, g, b) - Math.min(r, g, b) < 80) {
      dark[i] = 1;
    }
  }

  const seen = new Uint8Array(dark.length);
  const components = [];
  for (let y = 0; y < resizedHeight; y += 1) {
    for (let x = 0; x < resizedWidth; x += 1) {
      const idx = y * resizedWidth + x;
      if (!dark[idx] || seen[idx]) continue;
      const box = componentBox(dark, resizedWidth, resizedHeight, x, y, seen);
      const aspect = box.w / Math.max(1, box.h);
      if (box.pixels < 18) continue;
      if (box.h < 14 || box.h > 70) continue;
      if (box.w < 5 || box.w > 80) continue;
      if (aspect < 0.12 || aspect > 3.4) continue;
      if (box.y > resizedHeight * 0.90) continue;
      components.push(box);
    }
  }

  const groups = groupDigits(components).filter((group) => {
    const aspect = group.w / Math.max(1, group.h);
    if (!(group.h >= 18 && group.h <= 80 && group.w >= 8 && group.w <= 110 && aspect >= 0.18 && aspect <= 3.5)) {
      return false;
    }
    group.pale_label_background_score = Number(labelBackgroundScore(raw, resizedWidth, resizedHeight, group).toFixed(4));
    return group.pale_label_background_score >= 0.42;
  });

  const orangeArrowComponents = countOrangeArrowComponents(raw, resizedWidth, resizedHeight);
  const bagNumber = await detectVisibleBagNumber(sharp, imagePath);
  const repeatedNumberScore = Math.min(1, groups.length / 5);
  const orangeArrowScore = Math.min(1, orangeArrowComponents.length / 3);
  const coverageScore = Math.min(1, groups.reduce((sum, group) => sum + group.pixels, 0) / 2400);
  const score = Number((0.58 * repeatedNumberScore + 0.34 * orangeArrowScore + 0.08 * coverageScore).toFixed(4));
  const isCandidate = groups.length >= 2 && orangeArrowComponents.length >= 1 && score >= 0.42;

  let debugImagePath = null;
  if (isCandidate) {
    const fileName = `page_${String(pageEntry.page).padStart(3, "0")}_left_band.png`;
    const outPath = path.join(debugDir, fileName);
    await sharp(imagePath)
      .extract({ left: 0, top: 0, width: scanWidth, height: scanHeight })
      .png()
      .toFile(outPath);
    debugImagePath = path.relative(repoRoot, outPath);
  }

  return {
    page: pageEntry.page,
    image_path: pageEntry.image_path,
    score,
    is_candidate: isCandidate,
    signals: {
      large_number_group_count: groups.length,
      dark_component_count: components.length,
      orange_arrow_component_count: orangeArrowComponents.length,
      repeated_number_score: Number(repeatedNumberScore.toFixed(4)),
      orange_arrow_score: Number(orangeArrowScore.toFixed(4)),
      dark_coverage_score: Number(coverageScore.toFixed(4)),
      scan_region: {
        left: 0,
        top: 0,
        width: scanWidth,
        height: scanHeight,
      },
    },
    debug_image_path: debugImagePath,
    number_like_groups: groups.slice(0, 12),
    detected_bag_number: bagNumber.detected_bag_number,
    detected_bag_number_confidence: bagNumber.confidence,
    detected_bag_number_source: bagNumber.source,
    detected_bag_number_box: bagNumber.number_box,
    detected_bag_number_ocr_raw: bagNumber.ocr_raw,
    detected_bag_number_candidates: bagNumber.candidates,
  };
}


const pageIndexPath = argValue("--page-index");
const repoRoot = argValue("--repo-root");
const outPath = argValue("--out");
const debugDir = argValue("--debug-dir");
const nodeModules = argValue("--node-modules");

if (!pageIndexPath || !repoRoot || !outPath || !debugDir || !nodeModules) {
  console.error("Usage: bag_candidate_scan.mjs --page-index <path> --repo-root <dir> --out <path> --debug-dir <dir> --node-modules <dir>");
  process.exit(2);
}

const requireFromBundle = createRequire(path.join(nodeModules, "package.json"));
const sharp = requireFromBundle("sharp");
fs.mkdirSync(debugDir, { recursive: true });
for (const entry of fs.readdirSync(debugDir)) {
  if (/^page_\d{3}_left_band\.png$/.test(entry)) {
    fs.unlinkSync(path.join(debugDir, entry));
  }
}

const pageIndex = JSON.parse(fs.readFileSync(pageIndexPath, "utf8"));
const analyses = [];
for (const pageEntry of pageIndex.pages || []) {
  analyses.push(await analyzePage(sharp, repoRoot, pageEntry, debugDir));
}

const candidates = analyses
  .filter((entry) => entry.is_candidate)
  .sort((a, b) => b.score - a.score || a.page - b.page);

const payload = {
  stage: 2,
  name: "bag_candidate_pages",
  input_manifest: "indexes/02_page_index.json",
  page_count: analyses.length,
  candidate_count: candidates.length,
  debug_dir: path.relative(repoRoot, debugDir),
  candidates,
};

fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ candidate_count: candidates.length, out: path.relative(repoRoot, outPath) }));
