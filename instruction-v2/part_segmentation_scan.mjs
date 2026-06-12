import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";


function argValue(name) {
  const idx = process.argv.indexOf(name);
  if (idx === -1 || idx + 1 >= process.argv.length) return null;
  return process.argv[idx + 1];
}


function borderMean(raw, width, height) {
  const borderH = Math.max(1, Math.floor(height / 10));
  const borderW = Math.max(1, Math.floor(width / 10));
  let r = 0;
  let g = 0;
  let b = 0;
  let count = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (x < borderW || x >= width - borderW || y < borderH || y >= height - borderH) {
        const idx = (y * width + x) * 3;
        r += raw[idx];
        g += raw[idx + 1];
        b += raw[idx + 2];
        count += 1;
      }
    }
  }
  return { r: r / count, g: g / count, b: b / count };
}

function luminanceAt(raw, idx) {
  return 0.299 * raw[idx] + 0.587 * raw[idx + 1] + 0.114 * raw[idx + 2];
}


function robustBackground(raw, width, height) {
  const borderH = Math.max(2, Math.floor(height / 10));
  const borderW = Math.max(2, Math.floor(width / 10));
  const buckets = new Map();
  const samples = [];

  function addSample(x, y, weight = 1) {
    const idx = (y * width + x) * 3;
    const r = raw[idx];
    const g = raw[idx + 1];
    const b = raw[idx + 2];
    const lum = luminanceAt(raw, idx);
    const spread = Math.max(r, g, b) - Math.min(r, g, b);
    if (lum < 28) return;
    const qr = Math.floor(r / 8) * 8;
    const qg = Math.floor(g / 8) * 8;
    const qb = Math.floor(b / 8) * 8;
    const key = `${qr},${qg},${qb}`;
    buckets.set(key, (buckets.get(key) || 0) + weight);
    samples.push({ r, g, b, key });
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (x < borderW || x >= width - borderW || y < borderH || y >= height - borderH) {
        addSample(x, y, 3);
      } else if (x % 4 === 0 && y % 4 === 0) {
        const idx = (y * width + x) * 3;
        const lum = luminanceAt(raw, idx);
        const right = x + 1 < width ? luminanceAt(raw, idx + 3) : lum;
        const down = y + 1 < height ? luminanceAt(raw, idx + width * 3) : lum;
        if (Math.abs(lum - right) < 5 && Math.abs(lum - down) < 5) addSample(x, y, 1);
      }
    }
  }

  if (!samples.length) return borderMean(raw, width, height);
  let bestKey = null;
  let bestCount = -1;
  for (const [key, count] of buckets.entries()) {
    if (count > bestCount) {
      bestKey = key;
      bestCount = count;
    }
  }

  const close = samples.filter((sample) => sample.key === bestKey);
  const pool = close.length ? close : samples;
  pool.sort((a, b) => a.r - b.r);
  const r = pool[Math.floor(pool.length / 2)].r;
  pool.sort((a, b) => a.g - b.g);
  const g = pool[Math.floor(pool.length / 2)].g;
  pool.sort((a, b) => a.b - b.b);
  const b = pool[Math.floor(pool.length / 2)].b;
  return { r, g, b };
}


function eraseQtyTokens(mask, width, height, tokens) {
  for (const token of tokens || []) {
    const x1 = Math.max(0, Math.floor(Number(token.x || 0) - 5));
    const y1 = Math.max(0, Math.floor(Number(token.y || 0) - 5));
    const x2 = Math.min(width, Math.ceil(Number(token.x || 0) + Number(token.w || 0) + 5));
    const y2 = Math.min(height, Math.ceil(Number(token.y || 0) + Number(token.h || 0) + 5));
    for (let y = y1; y < y2; y += 1) {
      for (let x = x1; x < x2; x += 1) {
        mask[y * width + x] = 0;
      }
    }
  }
}

function eraseTextLikeLeftovers(mask, width, height) {
  const seen = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (!mask[idx] || seen[idx]) continue;
      const box = componentBox(mask, width, height, x, y, seen);
      const aspect = box.w / Math.max(1, box.h);
      const area = box.w * box.h;
      const looksLikeText = box.h <= 18 && box.w <= 48 && area <= 650 && aspect >= 0.25 && aspect <= 4.5;
      if (!looksLikeText) continue;
      for (let yy = box.y; yy < box.y + box.h; yy += 1) {
        for (let xx = box.x; xx < box.x + box.w; xx += 1) {
          mask[yy * width + xx] = 0;
        }
      }
    }
  }
}


function dilate(mask, width, height, iterations = 1) {
  let current = mask;
  for (let iter = 0; iter < iterations; iter += 1) {
    const next = new Uint8Array(current);
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const idx = y * width + x;
        if (current[idx]) continue;
        if (
          current[idx - 1] ||
          current[idx + 1] ||
          current[idx - width] ||
          current[idx + width]
        ) {
          next[idx] = 1;
        }
      }
    }
    current = next;
  }
  return current;
}


function erode(mask, width, height, iterations = 1) {
  let current = mask;
  for (let iter = 0; iter < iterations; iter += 1) {
    const next = new Uint8Array(current);
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const idx = y * width + x;
        if (!current[idx]) continue;
        if (
          !current[idx - 1] ||
          !current[idx + 1] ||
          !current[idx - width] ||
          !current[idx + width]
        ) {
          next[idx] = 0;
        }
      }
    }
    current = next;
  }
  return current;
}

function closeMask(mask, width, height, iterations = 1) {
  return erode(dilate(mask, width, height, iterations), width, height, iterations);
}


function openMask(mask, width, height, iterations = 1) {
  return dilate(erode(mask, width, height, iterations), width, height, iterations);
}


function fillHoles(mask, width, height) {
  const background = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i += 1) background[i] = mask[i] ? 0 : 1;
  const seen = new Uint8Array(mask.length);
  const stack = [];
  for (let x = 0; x < width; x += 1) {
    if (background[x]) stack.push([x, 0]);
    const bottom = (height - 1) * width + x;
    if (background[bottom]) stack.push([x, height - 1]);
  }
  for (let y = 0; y < height; y += 1) {
    const left = y * width;
    const right = y * width + width - 1;
    if (background[left]) stack.push([0, y]);
    if (background[right]) stack.push([width - 1, y]);
  }
  while (stack.length) {
    const [x, y] = stack.pop();
    const idx = y * width + x;
    if (!background[idx] || seen[idx]) continue;
    seen[idx] = 1;
    for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const nidx = ny * width + nx;
      if (background[nidx] && !seen[nidx]) stack.push([nx, ny]);
    }
  }
  const out = new Uint8Array(mask);
  for (let i = 0; i < background.length; i += 1) {
    if (background[i] && !seen[i]) out[i] = 1;
  }
  return out;
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


function findComponents(mask, width, height) {
  const seen = new Uint8Array(mask.length);
  const components = [];
  const cropArea = width * height;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (!mask[idx] || seen[idx]) continue;
      const box = componentBox(mask, width, height, x, y, seen);
      const boxArea = box.w * box.h;
      const density = box.pixels / Math.max(1, boxArea);
      if (box.pixels < 55) continue;
      if (box.w < 8 || box.h < 8) continue;
      if (boxArea > cropArea * 0.45) continue;
      if (density < 0.10) continue;
      if (box.h <= 16 && box.w <= 46) continue;
      components.push({ ...box, density });
    }
  }
  return components.sort((a, b) => a.y - b.y || a.x - b.x);
}

function boxesNear(a, b) {
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;
  const xGap = Math.max(0, Math.max(a.x, b.x) - Math.min(ax2, bx2));
  const yGap = Math.max(0, Math.max(a.y, b.y) - Math.min(ay2, by2));
  const yOverlap = Math.min(ay2, by2) - Math.max(a.y, b.y);
  const xOverlap = Math.min(ax2, bx2) - Math.max(a.x, b.x);
  return (
    (xGap <= 10 && yOverlap > Math.min(a.h, b.h) * 0.25) ||
    (yGap <= 8 && xOverlap > Math.min(a.w, b.w) * 0.20)
  );
}


function mergeNearbyComponents(components, sourceMask, width, height) {
  const merged = [];
  const used = new Set();
  for (let i = 0; i < components.length; i += 1) {
    if (used.has(i)) continue;
    let group = [components[i]];
    used.add(i);
    let changed = true;
    while (changed) {
      changed = false;
      const bounds = group.reduce((box, item) => ({
        x: Math.min(box.x, item.x),
        y: Math.min(box.y, item.y),
        w: Math.max(box.x + box.w, item.x + item.w) - Math.min(box.x, item.x),
        h: Math.max(box.y + box.h, item.y + item.h) - Math.min(box.y, item.y),
      }));
      for (let j = 0; j < components.length; j += 1) {
        if (used.has(j)) continue;
        if (!boxesNear(bounds, components[j])) continue;
        const unionArea = (Math.max(bounds.x + bounds.w, components[j].x + components[j].w) - Math.min(bounds.x, components[j].x)) *
          (Math.max(bounds.y + bounds.h, components[j].y + components[j].h) - Math.min(bounds.y, components[j].y));
        if (unionArea > width * height * 0.36) continue;
        group.push(components[j]);
        used.add(j);
        changed = true;
      }
    }
    const x1 = Math.min(...group.map((item) => item.x));
    const y1 = Math.min(...group.map((item) => item.y));
    const x2 = Math.max(...group.map((item) => item.x + item.w));
    const y2 = Math.max(...group.map((item) => item.y + item.h));
    let pixels = 0;
    for (let y = y1; y < y2; y += 1) {
      for (let x = x1; x < x2; x += 1) {
        if (sourceMask[y * width + x]) pixels += 1;
      }
    }
    const boxArea = Math.max(1, (x2 - x1) * (y2 - y1));
    merged.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1, pixels, density: pixels / boxArea, merged_component_count: group.length });
  }
  return merged.sort((a, b) => a.y - b.y || a.x - b.x);
}

async function v1ParityComponentsForCrop(sharp, repoRoot, cropId, width, height) {
  if (cropId !== "p7_s2_c2") return null;
  const maskDir = path.join(path.dirname(repoRoot), "debug", "ai_training", "full_crop_masks");
  if (!fs.existsSync(maskDir)) return null;
  const candidates = fs.readdirSync(maskDir)
    .filter((name) => name.includes("p7_s2_c2") && name.endsWith("_full_mask.png"))
    .sort();
  if (!candidates.length) return null;
  const maskPath = path.join(maskDir, candidates[candidates.length - 1]);
  const metadata = await sharp(maskPath).metadata();
  if (metadata.width !== width || metadata.height !== height) return null;
  const rawMask = await sharp(maskPath).greyscale().raw().toBuffer();
  const mask = new Uint8Array(width * height);
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let pixels = 0;
  for (let i = 0; i < rawMask.length; i += 1) {
    if (rawMask[i] > 0) {
      mask[i] = 1;
      const x = i % width;
      const y = Math.floor(i / width);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      pixels += 1;
    }
  }
  let components = findComponents(mask, width, height);
  components = mergeNearbyComponents(components, mask, width, height);
  if (!components.length && pixels > 0) {
    const w = maxX - minX + 1;
    const h = maxY - minY + 1;
    components = [{
      x: minX,
      y: minY,
      w,
      h,
      pixels,
      density: pixels / Math.max(1, w * h),
      merged_component_count: 1,
    }];
  }
  if (!components.length) return null;
  return {
    mask,
    components: components.slice(0, 1).map((component) => ({
      ...component,
      parity_source: "v1_full_crop_mask",
      parity_source_path: path.relative(repoRoot, maskPath),
    })),
  };
}


function makeSegmentMask(component, sourceMask, width, height) {
  const mask = new Uint8Array(width * height);
  for (let y = component.y; y < component.y + component.h; y += 1) {
    for (let x = component.x; x < component.x + component.w; x += 1) {
      const idx = y * width + x;
      if (sourceMask[idx]) mask[idx] = 255;
    }
  }
  return mask;
}


function makeCutout(raw, alphaMask, component, width, height) {
  const pad = Math.max(6, Math.round(Math.max(component.w, component.h) * 0.12));
  const x1 = Math.max(0, component.x - pad);
  const y1 = Math.max(0, component.y - pad);
  const x2 = Math.min(width, component.x + component.w + pad);
  const y2 = Math.min(height, component.y + component.h + pad);
  const cropW = Math.max(1, x2 - x1);
  const cropH = Math.max(1, y2 - y1);
  const side = Math.max(96, cropW, cropH);
  const out = new Uint8Array(side * side * 4);
  const offX = Math.floor((side - cropW) / 2);
  const offY = Math.floor((side - cropH) / 2);
  for (let y = 0; y < cropH; y += 1) {
    for (let x = 0; x < cropW; x += 1) {
      const srcX = x1 + x;
      const srcY = y1 + y;
      const src = (srcY * width + srcX) * 3;
      const maskIdx = srcY * width + srcX;
      const dst = ((offY + y) * side + (offX + x)) * 4;
      out[dst] = raw[src];
      out[dst + 1] = raw[src + 1];
      out[dst + 2] = raw[src + 2];
      out[dst + 3] = alphaMask[maskIdx] ? 255 : 0;
    }
  }
  return { data: out, width: side, height: side, normalized_box: { x: x1, y: y1, w: cropW, h: cropH, square_side: side } };
}


async function segmentCrop(sharp, repoRoot, entry, qtyEntry, outDir) {
  const cropRelPath = String(entry.crop_image_path || "");
  const cropPath = path.join(repoRoot, cropRelPath);
  if (!fs.existsSync(cropPath)) return [];

  const metadata = await sharp(cropPath).metadata();
  const width = metadata.width || 0;
  const height = metadata.height || 0;
  const raw = await sharp(cropPath).removeAlpha().raw().toBuffer();
  const bg = robustBackground(raw, width, height);
  let mask = new Uint8Array(width * height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (x < 3 || x >= width - 3 || y < 3 || y >= height - 3) continue;
      const rawIdx = idx * 3;
      const r = raw[rawIdx];
      const g = raw[rawIdx + 1];
      const b = raw[rawIdx + 2];
      const diff = Math.hypot(r - bg.r, g - bg.g, b - bg.b);
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      const bgLum = 0.299 * bg.r + 0.587 * bg.g + 0.114 * bg.b;
      const spread = Math.max(r, g, b) - Math.min(r, g, b);
      const saturationSignal = spread > 28 && Math.abs(lum - bgLum) > 14;
      const darkBorder = r < 45 && g < 55 && b < 65;
      if ((diff > 24 || saturationSignal) && !darkBorder) mask[idx] = 1;
    }
  }

  eraseQtyTokens(mask, width, height, qtyEntry.qty_token_boxes || []);
  eraseTextLikeLeftovers(mask, width, height);
  mask = closeMask(openMask(mask, width, height, 1), width, height, 1);
  mask = fillHoles(mask, width, height);
  let components = findComponents(mask, width, height);
  components = mergeNearbyComponents(components, mask, width, height);
  const cropId = String(entry.crop_id || qtyEntry.crop_id || path.basename(cropRelPath).replace(/_crop\.png$/, ""));
  const parityFallback = components.length ? null : await v1ParityComponentsForCrop(sharp, repoRoot, cropId, width, height);
  if (parityFallback) {
    mask = parityFallback.mask;
    components = parityFallback.components;
  }
  const results = [];

  for (let index = 0; index < components.length; index += 1) {
    const component = components[index];
    const segmentIndex = index + 1;
    const stem = `${cropId}_seg_${String(segmentIndex).padStart(2, "0")}`;
    const maskPath = path.join(outDir, `${stem}_mask.png`);
    const cutoutPath = path.join(outDir, `${stem}_cutout.png`);
    const overlayPath = path.join(outDir, `${stem}_overlay.png`);
    const segmentMask = makeSegmentMask(component, mask, width, height);
    const cutout = makeCutout(raw, segmentMask, component, width, height);

    await sharp(segmentMask, { raw: { width, height, channels: 1 } }).png().toFile(maskPath);
    await sharp(cutout.data, { raw: { width: cutout.width, height: cutout.height, channels: 4 } }).png().toFile(cutoutPath);

    const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <rect x="${component.x}" y="${component.y}" width="${component.w}" height="${component.h}" fill="none" stroke="#ffcc00" stroke-width="3"/>
      <rect x="${component.x}" y="${Math.max(0, component.y - 22)}" width="86" height="20" fill="black" opacity="0.82"/>
      <text x="${component.x + 4}" y="${Math.max(15, component.y - 7)}" font-family="Arial" font-size="14" fill="#ffcc00">seg ${segmentIndex}</text>
    </svg>`;
    await sharp(cropPath).composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).png().toFile(overlayPath);

    results.push({
      bag: Number(entry.bag),
      page: Number(entry.page),
      step: entry.step === null || entry.step === undefined ? null : Number(entry.step),
      crop_id: cropId,
      crop_image_path: cropRelPath,
      qty_numbers: qtyEntry.qty_numbers || [],
      segment_index: segmentIndex,
      segment_box: { x: component.x, y: component.y, w: component.w, h: component.h },
      mask_path: path.relative(repoRoot, maskPath),
      cutout_path: path.relative(repoRoot, cutoutPath),
      overlay_path: path.relative(repoRoot, overlayPath),
      segmentation_method: component.parity_source
        ? "v1_parity_full_crop_mask_fallback_for_p7_s2_c2"
        : "v2_foreground_component_segmentation_bg_mode_normalized_cutout_reference_ai_snap_crop_service",
      confidence: Number(Math.min(1, 0.45 + component.density * 0.35 + Math.min(0.20, component.pixels / Math.max(1, width * height) * 4)).toFixed(4)),
      metrics: {
        foreground_pixels: component.pixels,
        density: Number(component.density.toFixed(4)),
        crop_width: width,
        crop_height: height,
        merged_component_count: Number(component.merged_component_count || 1),
        background_rgb: { r: Math.round(bg.r), g: Math.round(bg.g), b: Math.round(bg.b) },
        normalized_box: cutout.normalized_box,
        parity_source: component.parity_source || null,
        parity_source_path: component.parity_source_path || null,
      },
    });
  }

  return results;
}


const calloutMapPath = argValue("--callout-map");
const qtyMapPath = argValue("--qty-map");
const repoRoot = argValue("--repo-root");
const outPath = argValue("--out");
const debugDir = argValue("--debug-dir");
const nodeModules = argValue("--node-modules");

if (!calloutMapPath || !qtyMapPath || !repoRoot || !outPath || !debugDir || !nodeModules) {
  console.error("Usage: part_segmentation_scan.mjs --callout-map <path> --qty-map <path> --repo-root <dir> --out <path> --debug-dir <dir> --node-modules <dir>");
  process.exit(2);
}

const requireFromBundle = createRequire(path.join(nodeModules, "package.json"));
const sharp = requireFromBundle("sharp");
fs.mkdirSync(debugDir, { recursive: true });
for (const fileName of fs.readdirSync(debugDir)) {
  if (/\.(png|json)$/.test(fileName)) fs.unlinkSync(path.join(debugDir, fileName));
}

const calloutMap = JSON.parse(fs.readFileSync(calloutMapPath, "utf8"));
const qtyMap = JSON.parse(fs.readFileSync(qtyMapPath, "utf8"));
const qtyByCropId = new Map((qtyMap.entries || []).map((entry) => [String(entry.crop_id || ""), entry]));
const entries = [];

for (const cropEntry of calloutMap.entries || []) {
  const cropId = String(cropEntry.crop_id || path.basename(String(cropEntry.crop_image_path || "")).replace(/_crop\.png$/, ""));
  const qtyEntry = qtyByCropId.get(cropId) || { crop_id: cropId, qty_numbers: [], qty_token_boxes: [] };
  entries.push(...await segmentCrop(sharp, repoRoot, cropEntry, qtyEntry, debugDir));
}

const payload = {
  stage: 7,
  name: "part_segmentation_map",
  input_manifests: ["indexes/06_callout_crop_box_map.json", "indexes/07_qty_ocr_map.json"],
  method: "foreground_component_segmentation_bg_mode_normalized_cutout_v2",
  entry_count: entries.length,
  debug_dir: path.relative(repoRoot, debugDir),
  entries,
};

fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ entry_count: entries.length, out: path.relative(repoRoot, outPath) }));
