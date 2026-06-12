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
  return { x1, y1, x2, y2, source: "buildStepRegion" };
}


function buildInstructionDebugSearchRegion(pageWidth, pageHeight, stepBox) {
  const x = Number(stepBox.x || 0);
  const y = Number(stepBox.y || 0);
  const w = Number(stepBox.w || 0);
  const h = Number(stepBox.h || 0);
  if (w <= 0 || h <= 0) return null;
  const x1 = clamp(Math.floor(x - 35), 0, pageWidth);
  const y1 = clamp(Math.floor(y - 290), 0, pageHeight);
  const x2 = clamp(Math.floor(x + w + 690), 0, pageWidth);
  const y2 = clamp(Math.floor(y - 5), 0, pageHeight);
  if (x2 <= x1 || y2 <= y1) return null;
  return { x1, y1, x2, y2, source: "instruction_debug_search" };
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

function detectCalloutRectByEdges(raw, width, height, region, stepBox) {
  const roiW = region.x2 - region.x1;
  const roiH = region.y2 - region.y1;
  if (roiW <= 0 || roiH <= 0) return null;
  const stepY = Number(stepBox.y || 0);
  const stepH = Number(stepBox.h || 0);
  const bottomSlack = Math.max(8, Math.floor(stepH * 0.45));

  const roi = new Uint8Array(roiW * roiH * 3);
  for (let y = 0; y < roiH; y += 1) {
    for (let x = 0; x < roiW; x += 1) {
      const src = ((region.y1 + y) * width + region.x1 + x) * 3;
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
      if (pageBottom > stepY + bottomSlack) continue;
      if (box.w < 70 || box.h < 28) continue;
      const aspect = box.w / Math.max(1, box.h);
      if (aspect < 1.0 || aspect > 6.0) continue;
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
      if (areaRatio < 0.025 || (areaRatio > 0.85 && !clippedLeftCallout)) continue;
      if (yellowRatio(roi, roiW, roiH, box) > 0.22) continue;
      components.push({
        ...box,
        area: box.w * box.h,
        pageBottom,
        geometry_rule: clippedLeftCallout ? "clipped_left_callout_panel_slack" : "standard_edge_component",
      });
    }
  }

  if (!components.length) return null;
  components.sort((a, b) => b.area - a.area || a.y - b.y);
  const best = components[0];
  return {
    x: region.x1 + best.x,
    y: region.y1 + best.y,
    w: best.w,
    h: best.h,
    confidence: Math.min(1, Number((0.55 + Math.min(0.35, best.area / Math.max(1, regionArea)) + Math.min(0.10, best.pixels / Math.max(1, best.area) * 0.10)).toFixed(4))),
    geometry_rule: best.geometry_rule || "standard_edge_component",
    components: components.map((c) => ({
      x: region.x1 + c.x,
      y: region.y1 + c.y,
      w: c.w,
      h: c.h,
      area: c.area,
    })),
    region,
    roi_mask: mask,
    roi_w: roiW,
    roi_h: roiH,
  };
}

const repoRoot = argValue("--repo-root");
const pageIndexPath = argValue("--page-index");
const stepMapPath = argValue("--step-map");
const outDir = argValue("--out-dir");
const nodeModules = argValue("--node-modules");
const stepsArg = argValue("--steps");

if (!repoRoot || !pageIndexPath || !stepMapPath || !outDir || !nodeModules || !stepsArg) {
  console.error("Usage: bag4_callout_recovery.mjs --repo-root <dir> --page-index <path> --step-map <path> --out-dir <dir> --node-modules <dir> --steps 110,112,...");
  process.exit(2);
}

const targetSteps = new Set(stepsArg.split(",").map((v) => Number(v.trim())).filter((v) => Number.isFinite(v)));
const requireFromBundle = createRequire(path.join(nodeModules, "package.json"));
const sharp = requireFromBundle("sharp");
fs.mkdirSync(outDir, { recursive: true });

const pageIndex = JSON.parse(fs.readFileSync(pageIndexPath, "utf8"));
const stepMap = JSON.parse(fs.readFileSync(stepMapPath, "utf8"));
const pagesByNumber = new Map((pageIndex.pages || []).map((entry) => [Number(entry.page), entry]));
const results = [];

for (const rawStep of stepMap.steps || []) {
  if (Number(rawStep.bag) !== 4) continue;
  const stepNumber = Number(rawStep.step_number);
  if (!targetSteps.has(stepNumber)) continue;
  const pageEntry = pagesByNumber.get(Number(rawStep.page));
  if (!pageEntry) continue;

  const imagePath = path.join(repoRoot, pageEntry.image_path);
  const metadata = await sharp(imagePath).metadata();
  const width = metadata.width || 0;
  const height = metadata.height || 0;
  const raw = await sharp(imagePath).removeAlpha().raw().toBuffer();
  const stepBox = rawStep.step_box || {};
  let region = buildStepRegion(width, height, stepBox);
  let detected = region ? detectCalloutRectByEdges(raw, width, height, region, stepBox) : null;
  if (!detected) {
    const fallbackRegion = buildInstructionDebugSearchRegion(width, height, stepBox);
    if (fallbackRegion) {
      const fallbackDetected = detectCalloutRectByEdges(raw, width, height, fallbackRegion, stepBox);
      if (fallbackDetected) {
        region = fallbackRegion;
        detected = { ...fallbackDetected, search_fallback: "instruction_debug_search" };
      }
    }
  }

  const stepDir = path.join(outDir, `step_${String(stepNumber).padStart(3, "0")}`);
  fs.mkdirSync(stepDir, { recursive: true });

  const pageViewBox = region
    ? {
        x: Math.max(0, region.x1 - 20),
        y: Math.max(0, region.y1 - 20),
        w: Math.min(width, region.x2 + 20) - Math.max(0, region.x1 - 20),
        h: Math.min(height, region.y2 + 80) - Math.max(0, region.y1 - 20),
      }
    : { x: 0, y: 0, w: width, h: height };

  const pageCropPath = path.join(stepDir, "page_region.png");
  await sharp(imagePath)
    .extract({ left: pageViewBox.x, top: pageViewBox.y, width: pageViewBox.w, height: pageViewBox.h })
    .png()
    .toFile(pageCropPath);

  let contourPath = path.join(stepDir, "contour_overlay.png");
  let cropBoxPath = path.join(stepDir, "crop_box_overlay.png");
  let sideBySidePath = path.join(stepDir, "side_by_side.png");

  if (detected) {
    const rel = (box) => ({
      x: box.x - pageViewBox.x,
      y: box.y - pageViewBox.y,
      w: box.w,
      h: box.h,
    });
    const calloutRel = rel({ x: detected.x, y: detected.y, w: detected.w, h: detected.h });
    const stepRel = rel({
      x: Number(stepBox.x || 0),
      y: Number(stepBox.y || 0),
      w: Number(stepBox.w || 0),
      h: Number(stepBox.h || 0),
    });
    const regionRel = region
      ? rel({ x: region.x1, y: region.y1, w: region.x2 - region.x1, h: region.y2 - region.y1 })
      : null;

    const contourSvg = `<svg width="${pageViewBox.w}" height="${pageViewBox.h}" xmlns="http://www.w3.org/2000/svg">
      ${regionRel ? `<rect x="${regionRel.x}" y="${regionRel.y}" width="${regionRel.w}" height="${regionRel.h}" fill="none" stroke="#999" stroke-width="2" stroke-dasharray="8 8"/>` : ""}
      <rect x="${stepRel.x}" y="${stepRel.y}" width="${stepRel.w}" height="${stepRel.h}" fill="none" stroke="#21b455" stroke-width="4"/>
      <rect x="${calloutRel.x}" y="${calloutRel.y}" width="${calloutRel.w}" height="${calloutRel.h}" fill="none" stroke="#1683ff" stroke-width="4"/>
      <text x="8" y="22" font-family="Arial" font-size="18" fill="#1683ff">detected contour</text>
    </svg>`;
    await sharp(pageCropPath).composite([{ input: Buffer.from(contourSvg), top: 0, left: 0 }]).png().toFile(contourPath);

    const cropSvg = `<svg width="${pageViewBox.w}" height="${pageViewBox.h}" xmlns="http://www.w3.org/2000/svg">
      <rect x="${stepRel.x}" y="${stepRel.y}" width="${stepRel.w}" height="${stepRel.h}" fill="none" stroke="#21b455" stroke-width="4"/>
      <rect x="${calloutRel.x}" y="${calloutRel.y}" width="${calloutRel.w}" height="${calloutRel.h}" fill="none" stroke="#00ff00" stroke-width="5"/>
      <text x="8" y="22" font-family="Arial" font-size="18" fill="#00ff00">crop box</text>
    </svg>`;
    await sharp(pageCropPath).composite([{ input: Buffer.from(cropSvg), top: 0, left: 0 }]).png().toFile(cropBoxPath);

    const panelW = pageViewBox.w;
    const panelH = pageViewBox.h;
    const labelH = 28;
    const totalW = panelW * 3;
    const totalH = panelH + labelH;
    const labels = ["page region", "detected callout contour", "resulting crop box"];
    const panels = [pageCropPath, contourPath, cropBoxPath];
    const composites = [];
    for (let i = 0; i < 3; i += 1) {
      composites.push({ input: panels[i], top: labelH, left: i * panelW });
    }
    const labelSvg = `<svg width="${totalW}" height="${labelH}" xmlns="http://www.w3.org/2000/svg">
      ${labels.map((label, i) => `<text x="${i * panelW + 8}" y="20" font-family="Arial" font-size="16" fill="#222">${label}</text>`).join("")}
    </svg>`;
    composites.push({ input: Buffer.from(labelSvg), top: 0, left: 0 });
    await sharp({ create: { width: totalW, height: totalH, channels: 3, background: { r: 255, g: 255, b: 255 } } })
      .composite(composites)
      .png()
      .toFile(sideBySidePath);

    const extractedCropPath = path.join(stepDir, "extracted_crop.png");
    await sharp(imagePath)
      .extract({ left: detected.x, top: detected.y, width: detected.w, height: detected.h })
      .png()
      .toFile(extractedCropPath);

    results.push({
      step: stepNumber,
      page: Number(rawStep.page),
      candidate_crop_id: `p${Number(rawStep.page)}_s${stepNumber}_c1`,
      crop_box: [detected.x, detected.y, detected.w, detected.h],
      confidence: detected.confidence,
      geometry_rule: detected.geometry_rule,
      method: "v1_detectCalloutRectByEdges",
      search_region: region,
      side_by_side_png: path.relative(repoRoot, sideBySidePath),
      extracted_crop_png: path.relative(repoRoot, extractedCropPath),
      candidate_json: path.relative(repoRoot, path.join(stepDir, "candidate.json")),
    });

    fs.writeFileSync(
      path.join(stepDir, "candidate.json"),
      `${JSON.stringify({
        crop_id: `p${Number(rawStep.page)}_s${stepNumber}_c1`,
        page: Number(rawStep.page),
        step: stepNumber,
        crop_box: [detected.x, detected.y, detected.w, detected.h],
        crop_box_format: "xywh",
        confidence: detected.confidence,
        geometry_rule: detected.geometry_rule,
        method: "v1_detectCalloutRectByEdges",
        source: "bag4_v1_callout_recovery",
        status: "pending_human_review",
        step_box_xywh: [Number(stepBox.x || 0), Number(stepBox.y || 0), Number(stepBox.w || 0), Number(stepBox.h || 0)],
      }, null, 2)}\n`,
    );
  } else {
    results.push({
      step: stepNumber,
      page: Number(rawStep.page),
      candidate_crop_id: null,
      crop_box: null,
      confidence: 0,
      method: "v1_detectCalloutRectByEdges",
      detection_failed: true,
      search_region: region,
    });
  }
}

results.sort((a, b) => a.page - b.page || a.step - b.step);
const manifest = {
  name: "bag4_v1_callout_recovery",
  method: "callout_crop_box_scan.mjs detectCalloutRectByEdges",
  steps: [...targetSteps].sort((a, b) => a - b),
  results,
};
fs.writeFileSync(path.join(outDir, "index.json"), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ result_count: results.length, out_dir: path.relative(repoRoot, outDir) }));
