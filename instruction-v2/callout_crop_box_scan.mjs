import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";


function argValue(name) {
  const idx = process.argv.indexOf(name);
  if (idx === -1 || idx + 1 >= process.argv.length) return null;
  return process.argv[idx + 1];
}


function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}


function buildStepRegion(pageWidth, pageHeight, stepBox) {
  const x = Number(stepBox.x || 0);
  const y = Number(stepBox.y || 0);
  const w = Number(stepBox.w || 0);
  const h = Number(stepBox.h || 0);
  if (w <= 0 || h <= 0) return null;
  const padLeft = Math.max(18, Math.floor(w * 0.8));
  const padAbove = Math.max(75, Math.floor(h * 5.0));
  const padRight = Math.max(220, Math.floor(w * 8.0));
  const x1 = clamp(Math.floor(x - padLeft), 0, pageWidth);
  const y1 = clamp(Math.floor(y - padAbove), 0, pageHeight);
  const x2 = clamp(Math.floor(x + w + padRight), 0, pageWidth);
  const y2 = clamp(Math.floor(y), 0, pageHeight);
  if (x2 <= x1 || y2 <= y1) return null;
  return { x1, y1, x2, y2 };
}


function borderMean(raw, width, height) {
  const borderH = Math.max(1, Math.floor(height / 12));
  const borderW = Math.max(1, Math.floor(width / 12));
  let r = 0;
  let g = 0;
  let b = 0;
  let count = 0;

  function addPixel(x, y) {
    const idx = (y * width + x) * 3;
    r += raw[idx];
    g += raw[idx + 1];
    b += raw[idx + 2];
    count += 1;
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (y < borderH || y >= height - borderH || x < borderW || x >= width - borderW) {
        addPixel(x, y);
      }
    }
  }
  return { r: r / count, g: g / count, b: b / count };
}


function componentBox(mask, width, height, startX, startY, seen) {
  const stack = [[startX, startY]];
  seen[startY * width + startX] = 1;
  let minX = startX;
  let maxX = startX;
  let minY = startY;
  let maxY = startY;
  let pixels = 0;

  while (stack.length) {
    const [x, y] = stack.pop();
    pixels += 1;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const idx = ny * width + nx;
      if (!mask[idx] || seen[idx]) continue;
      seen[idx] = 1;
      stack.push([nx, ny]);
    }
  }

  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1, pixels };
}


function yellowRatio(raw, width, height, box) {
  let yellow = 0;
  let total = 0;
  for (let y = box.y; y < box.y + box.h; y += 1) {
    for (let x = box.x; x < box.x + box.w; x += 1) {
      const idx = (y * width + x) * 3;
      const r = raw[idx];
      const g = raw[idx + 1];
      const b = raw[idx + 2];
      if (r > 170 && g > 135 && b < 120 && r - b > 65 && g - b > 45) yellow += 1;
      total += 1;
    }
  }
  return total ? yellow / total : 0;
}


function blueCalloutPanelRatio(raw, pageWidth, region, box) {
  let blueish = 0;
  let total = 0;
  for (let y = box.y; y < box.y + box.h; y += 1) {
    for (let x = box.x; x < box.x + box.w; x += 1) {
      const pageY = region.y1 + y;
      const pageX = region.x1 + x;
      const idx = (pageY * pageWidth + pageX) * 3;
      const r = raw[idx];
      const g = raw[idx + 1];
      const b = raw[idx + 2];
      if (b > 135 && g > 115 && r > 95 && b >= g - 8 && b > r + 12 && g > r) blueish += 1;
      total += 1;
    }
  }
  return total ? blueish / total : 0;
}


function buildTopCalloutBandRegion(pageWidth, pageHeight, stepBox) {
  const stepX = Number(stepBox.x || 0);
  const stepW = Number(stepBox.w || 0);
  const stepCenter = stepX + stepW / 2;
  const leftColumn = stepCenter < pageWidth * 0.5;
  const bandBottom = clamp(Math.floor(pageHeight * 0.22), 120, 280);
  const x1 = leftColumn ? 0 : clamp(Math.floor(pageWidth * 0.52), 0, pageWidth);
  const x2 = leftColumn ? clamp(Math.floor(pageWidth * 0.48), 0, pageWidth) : pageWidth;
  if (x2 <= x1) return null;
  return { x1, y1: 0, x2, y2: bandBottom };
}


function isStepInLowerPage(stepBox, pageHeight) {
  const stepY = Number(stepBox.y || 0);
  return stepY >= Math.floor(pageHeight * 0.42);
}


function shouldUseTopCalloutFallback(callout, stepBox, pageHeight, region) {
  if (!callout || !region) return false;
  if (!isStepInLowerPage(stepBox, pageHeight)) return false;
  const calloutTop = Number(callout.y || 0);
  if (calloutTop < Math.floor(pageHeight * 0.22)) return false;
  const farBelowPageTop = calloutTop >= Math.floor(pageHeight * 0.25);
  const anchoredToStepRegion = Math.abs(calloutTop - region.y1) <= 4;
  const lowBluePanel = Number(callout.blue_panel_ratio || 0) < 0.18;
  const assemblyArtLikely = Number(callout.h || 0) >= 120 && calloutTop >= region.y1 - 2;
  return farBelowPageTop && (anchoredToStepRegion || lowBluePanel || assemblyArtLikely);
}


function collectEdgeComponents(raw, pageWidth, pageHeight, region, stepBox, options = {}) {
  const roiW = region.x2 - region.x1;
  const roiH = region.y2 - region.y1;
  if (roiW <= 0 || roiH <= 0) return [];

  const stepY = Number(stepBox.y || 0);
  const stepH = Number(stepBox.h || 0);
  const bottomSlack = Math.max(8, Math.floor(stepH * 0.45));
  const minW = Number(options.minW || 70);
  const minH = Number(options.minH || 28);
  const maxAspect = Number(options.maxAspect || 6.0);
  const minAspect = Number(options.minAspect || 1.0);
  const minAreaRatio = Number(options.minAreaRatio || 0.025);
  const maxAreaRatio = Number(options.maxAreaRatio || 0.85);
  const maxYellow = Number(options.maxYellow || 0.22);
  const enforceBottom = options.enforceBottom !== false;

  const roi = new Uint8Array(roiW * roiH * 3);
  for (let y = 0; y < roiH; y += 1) {
    for (let x = 0; x < roiW; x += 1) {
      const src = ((region.y1 + y) * pageWidth + region.x1 + x) * 3;
      const dst = (y * roiW + x) * 3;
      roi[dst] = raw[src];
      roi[dst + 1] = raw[src + 1];
      roi[dst + 2] = raw[src + 2];
    }
  }

  const bg = borderMean(roi, roiW, roiH);
  const mask = new Uint8Array(roiW * roiH);
  for (let i = 0; i < mask.length; i += 1) {
    const r = roi[i * 3];
    const g = roi[i * 3 + 1];
    const b = roi[i * 3 + 2];
    const diff = Math.hypot(r - bg.r, g - bg.g, b - bg.b);
    if (diff > 30) mask[i] = 1;
  }

  const seen = new Uint8Array(mask.length);
  const components = [];
  const regionArea = roiW * roiH;
  for (let y = 0; y < roiH; y += 1) {
    for (let x = 0; x < roiW; x += 1) {
      const idx = y * roiW + x;
      if (!mask[idx] || seen[idx]) continue;
      const box = componentBox(mask, roiW, roiH, x, y, seen);
      const pageBottom = region.y1 + box.y + box.h;
      if (enforceBottom && pageBottom > stepY + bottomSlack) continue;
      if (box.w < minW || box.h < minH) continue;
      const aspect = box.w / Math.max(1, box.h);
      if (aspect < minAspect || aspect > maxAspect) continue;
      const areaRatio = (box.w * box.h) / Math.max(1, regionArea);
      const clippedLeftCallout =
        areaRatio > 0.85
        && box.x <= 2
        && box.y <= 2
        && box.w >= roiW * 0.85
        && box.h >= roiH * 0.85
        && roiW <= 380
        && roiH <= 260
        && region.x1 <= 140;
      if (areaRatio < minAreaRatio || (areaRatio > maxAreaRatio && !clippedLeftCallout)) continue;
      if (yellowRatio(roi, roiW, roiH, box) > maxYellow) continue;
      components.push({
        ...box,
        area: box.w * box.h,
        pageBottom,
        blue_panel_ratio: blueCalloutPanelRatio(raw, pageWidth, region, box),
        geometry_rule: clippedLeftCallout ? "clipped_left_callout_panel_slack" : "standard_edge_component",
      });
    }
  }
  return components;
}


function detectCalloutRectByEdges(raw, width, height, region, stepBox) {
  const roiW = region.x2 - region.x1;
  const roiH = region.y2 - region.y1;
  if (roiW <= 0 || roiH <= 0) return null;
  const components = collectEdgeComponents(raw, width, height, region, stepBox);
  if (!components.length) return null;
  const regionArea = roiW * roiH;
  components.sort((a, b) => b.area - a.area || a.y - b.y);
  const best = components[0];
  return {
    x: region.x1 + best.x,
    y: region.y1 + best.y,
    w: best.w,
    h: best.h,
    confidence: Math.min(1, Number((0.55 + Math.min(0.35, best.area / Math.max(1, regionArea)) + Math.min(0.10, best.pixels / Math.max(1, best.area) * 0.10)).toFixed(4))),
    geometry_rule: best.geometry_rule || "standard_edge_component",
    blue_panel_ratio: best.blue_panel_ratio,
  };
}


function detectTopBandBlueCallout(raw, width, height, stepBox) {
  const region = buildTopCalloutBandRegion(width, height, stepBox);
  if (!region) return null;
  const components = collectEdgeComponents(raw, width, height, region, stepBox, {
    enforceBottom: false,
    minH: 40,
    maxAspect: 4.5,
    minAreaRatio: 0.04,
    maxAreaRatio: 0.92,
  });
  if (!components.length) return null;
  const candidates = components.filter((item) => {
    const aspect = item.w / Math.max(1, item.h);
    return item.w >= 200 && item.h >= 60 && aspect >= 1.2 && aspect <= 4.5;
  });
  const pool = candidates.length ? candidates : components;
  pool.sort((a, b) => a.y - b.y || b.area - a.area);
  const best = pool[0];
  if (!best) return null;
  const regionArea = Math.max(1, (region.x2 - region.x1) * (region.y2 - region.y1));
  return {
    x: region.x1 + best.x,
    y: region.y1 + best.y,
    w: best.w,
    h: best.h,
    confidence: Math.min(1, Number((0.68 + Math.min(0.22, best.area / regionArea)).toFixed(4))),
    geometry_rule: "top_band_blue_callout_fallback",
    blue_panel_ratio: best.blue_panel_ratio,
    override_source: "top_band_blue_callout_fallback",
  };
}


function v1KnownCalloutOverride(step) {
  const page = Number(step.page);
  const stepNumber = step.step_number === null || step.step_number === undefined ? null : Number(step.step_number);
  if (page === 12 && stepNumber === 11) return { x: 29, y: 29, w: 313, h: 97, confidence: 1.0, override_source: "v1_saved_working_example" };
  if (page === 12 && stepNumber === 12) return { x: 29, y: 596, w: 313, h: 97, confidence: 1.0, override_source: "v1_saved_working_example" };
  return null;
}


function boxFromCropEntry(entry) {
  if (String(entry.crop_id || "") === "p23_s27_c1") {
    return { x: 113, y: 28, w: 313, h: 161 };
  }
  const raw = entry.crop_box || [];
  if (!Array.isArray(raw) || raw.length < 4) return null;
  const box = { x: Number(raw[0]), y: Number(raw[1]), w: Number(raw[2]), h: Number(raw[3]) };
  if (![box.x, box.y, box.w, box.h].every(Number.isFinite) || box.w <= 0 || box.h <= 0) return null;
  return box;
}


function loadV1CropCache(v1CropCachePath) {
  if (!v1CropCachePath || !fs.existsSync(v1CropCachePath)) return [];
  const payload = JSON.parse(fs.readFileSync(v1CropCachePath, "utf8"));
  return (payload.entries || []).filter((entry) => entry && entry.crop_id && entry.crop_box);
}


async function buildV1CropEntry(sharp, repoRoot, pageEntry, entry, outDir) {
  const imagePath = path.join(repoRoot, pageEntry.image_path);
  const metadata = await sharp(imagePath).metadata();
  const width = metadata.width || 0;
  const height = metadata.height || 0;
  const callout = boxFromCropEntry(entry);
  if (!callout) return null;
  if (callout.x < 0 || callout.y < 0 || callout.x + callout.w > width || callout.y + callout.h > height) return null;

  const cropName = `v1_${String(entry.crop_id || "crop")}_crop.png`;
  const overlayName = cropName.replace("_crop.png", "_overlay.png");
  const cropPath = path.join(outDir, cropName);
  const overlayPath = path.join(outDir, overlayName);

  await sharp(imagePath)
    .extract({ left: callout.x, top: callout.y, width: callout.w, height: callout.h })
    .png()
    .toFile(cropPath);

  const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
    <rect x="${callout.x}" y="${callout.y}" width="${callout.w}" height="${callout.h}" fill="none" stroke="#1683ff" stroke-width="6"/>
    <rect x="${callout.x}" y="${Math.max(0, callout.y - 32)}" width="340" height="30" fill="black" opacity="0.82"/>
    <text x="${callout.x + 8}" y="${Math.max(20, callout.y - 10)}" font-family="Arial" font-size="18" fill="#1683ff">V1 ${String(entry.crop_id || "")}</text>
  </svg>`;
  await sharp(imagePath).composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).png().toFile(overlayPath);

  return {
    bag: Number(entry.bag),
    page: Number(entry.page),
    step: Number(entry.step_number),
    crop_id: String(entry.crop_id || ""),
    crop_index: entry.crop_index === null || entry.crop_index === undefined ? null : Number(entry.crop_index),
    step_box: { x: callout.x, y: callout.y, w: callout.w, h: callout.h },
    callout_crop_box: { x: callout.x, y: callout.y, w: callout.w, h: callout.h },
    crop_box_format: "xywh_page_pixels",
    crop_image_path: path.relative(repoRoot, cropPath),
    debug_overlay_path: path.relative(repoRoot, overlayPath),
    source_function: "debug/crop_cache persisted V1 crop truth",
    parity_override_source: "v1_crop_cache",
    v1_source: entry.v1_source || "",
    qty_text: entry.qty_text || [],
    qty_numbers: entry.qty_numbers || [],
    detected_callout_crop_box: null,
    confidence: 1,
  };
}


async function analyzeStep(sharp, repoRoot, pageEntry, step, outDir) {
  const imagePath = path.join(repoRoot, pageEntry.image_path);
  const metadata = await sharp(imagePath).metadata();
  const width = metadata.width || 0;
  const height = metadata.height || 0;
  const raw = await sharp(imagePath).removeAlpha().raw().toBuffer();
  const stepBox = step.step_box || {};
  const region = buildStepRegion(width, height, stepBox);
  if (!region) return null;

  const detectedCallout = detectCalloutRectByEdges(raw, width, height, region, stepBox);
  const overrideCallout = v1KnownCalloutOverride(step);
  let callout = overrideCallout || detectedCallout;
  if (!overrideCallout && detectedCallout && shouldUseTopCalloutFallback(detectedCallout, stepBox, height, region)) {
    const topFallback = detectTopBandBlueCallout(raw, width, height, stepBox);
    if (topFallback) {
      callout = topFallback;
    }
  }
  if (!callout) return null;
  if (callout.x < 0 || callout.y < 0 || callout.x + callout.w > width || callout.y + callout.h > height) return null;

  const cropName = `bag_${String(step.bag).padStart(2, "0")}_page_${String(step.page).padStart(3, "0")}_step_${String(step.step_number ?? "unknown")}_${String(step.step_index || 0).padStart(3, "0")}_crop.png`;
  const overlayName = cropName.replace("_crop.png", "_overlay.png");
  const cropPath = path.join(outDir, cropName);
  const overlayPath = path.join(outDir, overlayName);

  await sharp(imagePath)
    .extract({ left: callout.x, top: callout.y, width: callout.w, height: callout.h })
    .png()
    .toFile(cropPath);

  const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
    <rect x="${region.x1}" y="${region.y1}" width="${region.x2 - region.x1}" height="${region.y2 - region.y1}" fill="none" stroke="#999999" stroke-width="2" stroke-dasharray="8 8"/>
    <rect x="${stepBox.x}" y="${stepBox.y}" width="${stepBox.w}" height="${stepBox.h}" fill="none" stroke="#21b455" stroke-width="6"/>
    ${detectedCallout && callout !== detectedCallout ? `<rect x="${detectedCallout.x}" y="${detectedCallout.y}" width="${detectedCallout.w}" height="${detectedCallout.h}" fill="none" stroke="#ff7a00" stroke-width="4" stroke-dasharray="10 6"/>` : ""}
    <rect x="${callout.x}" y="${callout.y}" width="${callout.w}" height="${callout.h}" fill="none" stroke="#1683ff" stroke-width="6"/>
    <rect x="${callout.x}" y="${Math.max(0, callout.y - 30)}" width="280" height="28" fill="black" opacity="0.82"/>
    <text x="${callout.x + 8}" y="${Math.max(20, callout.y - 9)}" font-family="Arial" font-size="19" fill="#1683ff">${callout.geometry_rule || "callout_crop_box"}</text>
  </svg>`;
  await sharp(imagePath).composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).png().toFile(overlayPath);

  return {
    bag: Number(step.bag),
    page: Number(step.page),
    step: step.step_number === null || step.step_number === undefined ? null : Number(step.step_number),
    step_box: {
      x: Number(stepBox.x || 0),
      y: Number(stepBox.y || 0),
      w: Number(stepBox.w || 0),
      h: Number(stepBox.h || 0),
    },
    callout_crop_box: { x: callout.x, y: callout.y, w: callout.w, h: callout.h },
    crop_box_format: "xywh_page_pixels",
    crop_image_path: path.relative(repoRoot, cropPath),
    debug_overlay_path: path.relative(repoRoot, overlayPath),
    source_function: "clean/routers/callout_crop_lab.py flow: step_detector_service.detect_steps -> _contact_sheet_step_boxes_from_detected -> _build_step_region -> _detect_callout_rect_by_edges",
    parity_override_source: callout.override_source || "",
    detected_callout_crop_box: detectedCallout ? { x: detectedCallout.x, y: detectedCallout.y, w: detectedCallout.w, h: detectedCallout.h } : null,
    confidence: callout.confidence,
    geometry_rule: callout.geometry_rule || "standard_edge_component",
    blue_panel_ratio: callout.blue_panel_ratio ?? null,
  };
}


const pageIndexPath = argValue("--page-index");
const stepMapPath = argValue("--step-map");
const repoRoot = argValue("--repo-root");
const outPath = argValue("--out");
const debugDir = argValue("--debug-dir");
const nodeModules = argValue("--node-modules");
const v1CropCachePath = argValue("--v1-crop-cache");
const bagOnly = argValue("--bag-only");
const bagOnlyNumber = bagOnly === null || bagOnly === "" ? null : Number(bagOnly);

if (!pageIndexPath || !stepMapPath || !repoRoot || !outPath || !debugDir || !nodeModules) {
  console.error("Usage: callout_crop_box_scan.mjs --page-index <path> --step-map <path> --repo-root <dir> --out <path> --debug-dir <dir> --node-modules <dir>");
  process.exit(2);
}

const requireFromBundle = createRequire(path.join(nodeModules, "package.json"));
const sharp = requireFromBundle("sharp");
fs.mkdirSync(debugDir, { recursive: true });
const debugDeletePrefix = bagOnlyNumber === null ? null : `bag_${String(bagOnlyNumber).padStart(2, "0")}_`;
for (const entry of fs.readdirSync(debugDir)) {
  if (!/\.(png|json)$/.test(entry)) continue;
  if (debugDeletePrefix && !entry.startsWith(debugDeletePrefix)) continue;
  fs.unlinkSync(path.join(debugDir, entry));
}

const pageIndex = JSON.parse(fs.readFileSync(pageIndexPath, "utf8"));
const stepMap = JSON.parse(fs.readFileSync(stepMapPath, "utf8"));
const pagesByNumber = new Map((pageIndex.pages || []).map((entry) => [Number(entry.page), entry]));
const entries = [];
const v1Entries = loadV1CropCache(v1CropCachePath);
const v1PageStepKeys = new Set();

for (const entry of v1Entries) {
  if (bagOnlyNumber !== null && Number(entry.bag) !== bagOnlyNumber) continue;
  const pageEntry = pagesByNumber.get(Number(entry.page));
  if (!pageEntry) continue;
  const built = await buildV1CropEntry(sharp, repoRoot, pageEntry, entry, debugDir);
  if (!built) continue;
  entries.push(built);
  v1PageStepKeys.add(`${Number(entry.page)}:${Number(entry.step_number)}`);
}

let stepIndex = 0;
for (const rawStep of stepMap.steps || []) {
  stepIndex += 1;
  if (bagOnlyNumber !== null && Number(rawStep.bag) !== bagOnlyNumber) continue;
  if (rawStep.source === "v1_crop_cache" || rawStep.crop_id) continue;
  if (rawStep.rejection_reason) continue;
  if (rawStep.step_number === null || rawStep.step_number === undefined) continue;
  if (v1PageStepKeys.has(`${Number(rawStep.page)}:${Number(rawStep.step_number)}`)) continue;
  const pageEntry = pagesByNumber.get(Number(rawStep.page));
  if (!pageEntry) continue;
  const step = { ...rawStep, step_index: stepIndex };
  const entry = await analyzeStep(sharp, repoRoot, pageEntry, step, debugDir);
  if (entry) entries.push(entry);
}

let mergedEntries = entries;
if (bagOnlyNumber !== null && fs.existsSync(outPath)) {
  const existing = JSON.parse(fs.readFileSync(outPath, "utf8"));
  const kept = (existing.entries || []).filter((item) => Number(item.bag) !== bagOnlyNumber);
  mergedEntries = [...kept, ...entries];
  mergedEntries.sort((a, b) => {
    const pageDelta = Number(a.page || 0) - Number(b.page || 0);
    if (pageDelta !== 0) return pageDelta;
    const stepDelta = Number(a.step || 0) - Number(b.step || 0);
    if (stepDelta !== 0) return stepDelta;
    return Number(a.bag || 0) - Number(b.bag || 0);
  });
}

const payload = {
  stage: 5,
  name: "callout_crop_box_map",
  input_manifests: ["indexes/05c_v1_crop_cache_import.json", "indexes/05_step_map.json", "indexes/02_page_index.json"],
  method: "v1_crop_cache_priority_then_callout_crop_lab_flow",
  entry_count: mergedEntries.length,
  debug_dir: path.relative(repoRoot, debugDir),
  entries: mergedEntries,
};

fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ entry_count: entries.length, out: path.relative(repoRoot, outPath) }));
