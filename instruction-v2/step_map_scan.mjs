import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { execFile } from "node:child_process";
import { promisify } from "node:util";


const execFileAsync = promisify(execFile);


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


function componentsOnSameBaseline(a, b) {
  const yOverlap = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  const minHeight = Math.max(1, Math.min(a.h, b.h));
  return yOverlap / minHeight >= 0.80;
}


function sortComponentsForGrouping(components) {
  const sorted = [...components].sort((a, b) => a.y - b.y || a.x - b.x);
  const rows = [];
  for (const component of sorted) {
    let row = null;
    for (const candidate of rows) {
      if (componentsOnSameBaseline(component, candidate[0])) {
        row = candidate;
        break;
      }
    }
    if (row) row.push(component);
    else rows.push([component]);
  }
  return rows
    .sort((a, b) => Math.min(...a.map((c) => c.y)) - Math.min(...b.map((c) => c.y)))
    .flatMap((row) => row.sort((a, b) => a.x - b.x));
}


function groupComponents(components) {
  const groups = [];
  for (const component of sortComponentsForGrouping(components)) {
    let matched = null;
    for (const group of groups) {
      const yOverlap = Math.min(component.y + component.h, group.y + group.h) - Math.max(component.y, group.y);
      const minHeight = Math.max(1, Math.min(component.h, group.h));
      const gap = component.x - (group.x + group.w);
      // V1: height-ratio guard prevents vertically-mismatched tokens from joining
      const maxH = Math.max(component.h, group.h);
      const minH = Math.min(component.h, group.h);
      const heightRatio = minH / Math.max(1, maxH);
      if (
        yOverlap / minHeight >= 0.45
        && gap >= -6
        && gap <= Math.max(18, group.h * 0.45)
        && heightRatio >= 0.65
        && heightRatio <= 1.55
      ) {
        matched = group;
        break;
      }
      // Narrow-gap fallback: adjacent digit characters in the same step number
      // (e.g. "1" beside "3" in "132") may have slightly different rendered heights
      // that fail the strict ratio check.  When the x-gap is ≤ 5 px and y-overlap
      // is very high the components are almost certainly the same printed number.
      if (gap >= -2 && gap <= 5 && yOverlap / minHeight >= 0.80) {
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


function boxScore(box, width, height) {
  const xRatio = box.x / Math.max(1, width);
  const yRatio = box.y / Math.max(1, height);
  const hRatio = box.h / Math.max(1, height);
  let score = 0.25;
  if (xRatio < 0.06) score += 0.42;
  else if (xRatio < 0.10) score += 0.22;
  if (hRatio > 0.025 && hRatio < 0.085) score += 0.20;
  if (yRatio > 0.08 && yRatio < 0.90) score += 0.14;
  if (box.components <= 3) score += 0.10;
  if (box.pixels > 80) score += 0.05;
  return Math.min(1, Number(score.toFixed(4)));
}

function auditBoxScore(box, width, height) {
  const yRatio = box.y / Math.max(1, height);
  const hRatio = box.h / Math.max(1, height);
  let score = 0.30;
  if (hRatio > 0.025 && hRatio < 0.085) score += 0.22;
  if (yRatio > 0.08 && yRatio < 0.90) score += 0.16;
  if (box.components <= 3) score += 0.12;
  if (box.pixels > 80) score += 0.06;
  return Math.min(1, Number(score.toFixed(4)));
}


async function runTesseractDigits(imagePath) {
  try {
    const { stdout } = await execFileAsync("/opt/homebrew/bin/tesseract", [
      imagePath,
      "stdout",
      "--psm",
      "13",
      "-c",
      "tessedit_char_whitelist=0123456789",
    ], { timeout: 5000, maxBuffer: 1024 * 32 });
    const digits = String(stdout || "").replace(/\D+/g, "");
    return digits || "";
  } catch (_error) {
    return "";
  }
}


async function readPrintedStepNumber(sharp, imagePath, box, width, height, tmpDir, pageNumber, stepIndex) {
  const pad = 8;
  const left = Math.max(0, Number(box.x) - pad);
  const top = Math.max(0, Number(box.y) - pad);
  const right = Math.min(width, Number(box.x) + Number(box.w) + pad);
  const bottom = Math.min(height, Number(box.y) + Number(box.h) + pad);
  const cropWidth = Math.max(1, right - left);
  const cropHeight = Math.max(1, bottom - top);
  const tightLeft = Math.max(0, Number(box.x));
  const tightTop = Math.max(0, Number(box.y));
  const tightWidth = Math.max(1, Math.min(width - tightLeft, Number(box.w)));
  const tightHeight = Math.max(1, Math.min(height - tightTop, Number(box.h)));
  const stem = `page_${String(pageNumber).padStart(3, "0")}_step_${String(stepIndex).padStart(2, "0")}`;

  const variants = [
    {
      name: "tight_threshold_6x",
      path: path.join(tmpDir, `${stem}_tight_threshold_6x.png`),
      left: tightLeft,
      top: tightTop,
      width: tightWidth,
      height: tightHeight,
      build: (img) => img.greyscale().threshold(135).resize({ width: tightWidth * 6, height: tightHeight * 6, kernel: "cubic" }),
    },
    {
      name: "tight_threshold_invert_6x",
      path: path.join(tmpDir, `${stem}_tight_threshold_invert_6x.png`),
      left: tightLeft,
      top: tightTop,
      width: tightWidth,
      height: tightHeight,
      build: (img) => img.greyscale().threshold(105).negate().resize({ width: tightWidth * 6, height: tightHeight * 6, kernel: "cubic" }),
    },
    {
      name: "threshold_5x",
      path: path.join(tmpDir, `${stem}_threshold_5x.png`),
      left,
      top,
      width: cropWidth,
      height: cropHeight,
      build: (img) => img.greyscale().threshold(135).resize({ width: cropWidth * 5, height: cropHeight * 5, kernel: "cubic" }),
    },
    {
      name: "threshold_invert_5x",
      path: path.join(tmpDir, `${stem}_threshold_invert_5x.png`),
      left,
      top,
      width: cropWidth,
      height: cropHeight,
      build: (img) => img.greyscale().threshold(105).negate().resize({ width: cropWidth * 5, height: cropHeight * 5, kernel: "cubic" }),
    },
    {
      name: "gray_5x",
      path: path.join(tmpDir, `${stem}_gray_5x.png`),
      left,
      top,
      width: cropWidth,
      height: cropHeight,
      build: (img) => img.greyscale().normalize().resize({ width: cropWidth * 5, height: cropHeight * 5, kernel: "cubic" }),
    },
  ];
  const rejectedReads = [];

  for (const variant of variants) {
    const crop = sharp(imagePath).extract({ left: variant.left, top: variant.top, width: variant.width, height: variant.height });
    await variant.build(crop).png().toFile(variant.path);
    const text = await runTesseractDigits(variant.path);
    if (!text) continue;
    const value = Number.parseInt(text, 10);
    if (!Number.isFinite(value) || value <= 0) {
      rejectedReads.push({ raw_text: text, source: variant.name, reason: "invalid_non_positive_step_number" });
      continue;
    }
    if (value > 999) {
      rejectedReads.push({ raw_text: text, value, source: variant.name, reason: "invalid_step_number_1000_or_more" });
      continue;
    }
    // V1 parity: keep anchors in 0 < step_number < 1000 (instruction_debug anchor filter)
    if (value >= 1000) {
      rejectedReads.push({ raw_text: text, value, source: variant.name, reason: "invalid_step_number_1000_or_more" });
      continue;
    }
    const componentCount = Number(box.components || 0);
    if (componentCount > 0 && text.length > componentCount) {
      rejectedReads.push({
        raw_text: text,
        value,
        source: variant.name,
        reason: "ocr_digit_count_exceeds_visual_component_count",
        visual_component_count: componentCount,
      });
      continue;
    }
    if (componentCount > 1 && text.length < componentCount) {
      rejectedReads.push({
        raw_text: text,
        value,
        source: variant.name,
        reason: "ocr_digit_count_less_than_visual_component_count",
        visual_component_count: componentCount,
      });
      continue;
    }
    if (componentCount > 1 && /^0\d+/.test(text)) {
      rejectedReads.push({
        raw_text: text,
        value,
        source: variant.name,
        reason: "ocr_leading_zero_on_multi_component_step_number",
        visual_component_count: componentCount,
      });
      continue;
    }
    if (text.length === 1 && Number(box.w) >= Math.max(34, Number(box.h) * 0.85)) {
      rejectedReads.push({
        raw_text: text,
        value,
        source: variant.name,
        reason: "single_digit_read_from_multi_digit_sized_step_box",
        visual_component_count: componentCount,
      });
      continue;
    }
    return {
      value,
      confidence: variant.name === "gray_5x" ? 0.65 : 0.75,
      source: `tesseract_step_box_${variant.name}`,
      raw_text: text,
      rejection_reason: null,
      rejected_reads: rejectedReads,
    };
  }

  return {
    value: null,
    confidence: 0,
    source: rejectedReads.length ? "tesseract_step_box_invalid_digits_rejected" : "tesseract_step_box_no_digits",
    raw_text: rejectedReads.map((item) => item.raw_text).filter(Boolean).join(","),
    rejection_reason: rejectedReads.length ? "invalid_step_number_rejected" : null,
    rejected_reads: rejectedReads,
  };
}


function filterInvalidStepNumber(value) {
  if (value === null || value === undefined || value === "") {
    return { step_number: null, rejection_reason: null };
  }
  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return { step_number: null, rejection_reason: "invalid_non_positive_step_number" };
  }
  if (parsed >= 1000) {
    return { step_number: null, rejection_reason: "invalid_step_number_1000_or_more" };
  }
  // V1 parity: 0 < step_number < 1000
  if (parsed >= 1000) {
    return { step_number: null, rejection_reason: "invalid_step_number_1000_or_more" };
  }
  return { step_number: parsed, rejection_reason: null };
}


function trustedStepNumbersByPage(v1Entries) {
  const byPage = new Map();
  for (const entry of v1Entries || []) {
    const page = Number(entry.page);
    const stepNumber = Number(entry.step_number);
    if (!Number.isFinite(page) || !Number.isFinite(stepNumber) || stepNumber <= 0) continue;
    if (!byPage.has(page)) byPage.set(page, new Set());
    byPage.get(page).add(stepNumber);
  }
  return byPage;
}


function applyV1StepAnchorGuardrails(visualSteps, v1Entries) {
  const trustedByPage = trustedStepNumbersByPage(v1Entries);
  return (visualSteps || []).map((step) => {
    const page = Number(step.page);
    const stepNumber = Number(step.step_number);
    const trusted = trustedByPage.get(page);
    if (!trusted || !Number.isFinite(stepNumber) || stepNumber <= 0) return step;

    if (trusted.has(stepNumber)) {
      return {
        ...step,
        step_number: null,
        rejection_reason: "duplicate_visual_anchor_rejected_by_v1_crop_cache_context",
        signals: {
          ...(step.signals || {}),
          rejected_step_number: stepNumber,
          v1_trusted_step_numbers_on_page: Array.from(trusted).sort((a, b) => a - b),
          v1_guardrail_source: "clean/routers/instruction_debug.py::_filter_invalid_step_anchor_boxes plus V1 crop-cache page context",
        },
      };
    }

    return {
      ...step,
      step_number: null,
      rejection_reason: "non_sequential_step_anchor_rejected_by_v1_crop_cache_context",
      signals: {
        ...(step.signals || {}),
        rejected_step_number: stepNumber,
        v1_trusted_step_numbers_on_page: Array.from(trusted).sort((a, b) => a - b),
        v1_guardrail_source: "clean/routers/instruction_debug.py::_filter_invalid_step_anchor_boxes plus V1 crop-cache page context",
      },
    };
  });
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


function componentPassesStepShape(box, width, height, fullPage = false) {
  const aspect = box.w / Math.max(1, box.h);
  if (box.pixels < 35) return false;
  if (box.h < 24 || box.h > 125) return false;
  if (box.w < 8 || box.w > 40) return false;
  if (box.w * box.h > 2000) return false;
  if (aspect < 0.10 || aspect > 4.5) return false;
  if (!fullPage && box.x > width * 0.11) return false;
  if (fullPage && box.x > width * 0.92) return false;
  if (box.y < height * 0.08) return false;
  if (box.y > height * 0.93) return false;
  // V1: bottom 12% combined with left/right 18% margin = page number zone, not a step
  const cx = box.x + box.w / 2;
  if (box.y > height * 0.88 && (cx < width * 0.18 || cx > width * 0.82)) return false;
  return true;
}


function groupPassesStepShape(box, width, height) {
  const aspect = box.w / Math.max(1, box.h);
  return box.h >= 30 && box.h <= 95 && box.w >= 10 && box.w <= 115 && aspect >= 0.12 && aspect <= 3.6;
}


function boxOverlapRatio(a, b) {
  const xOverlap = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const yOverlap = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  if (xOverlap <= 0 || yOverlap <= 0) return 0;
  const intersection = xOverlap * yOverlap;
  const minArea = Math.min(a.w * a.h, b.w * b.h);
  return intersection / Math.max(1, minArea);
}


function boxesVisuallyMatch(a, b) {
  if (boxOverlapRatio(a, b) >= 0.55) return true;
  return (
    Math.abs(Number(a.x) - Number(b.x)) <= 3
    && Math.abs(Number(a.y) - Number(b.y)) <= 3
    && Math.abs(Number(a.w) - Number(b.w)) <= 4
    && Math.abs(Number(a.h) - Number(b.h)) <= 4
  );
}


function mergeNeighboringGroups(groups) {
  let merged = groups.map((group) => ({ ...group }));
  let changed = true;
  while (changed) {
    changed = false;
    const next = [];
    for (const group of sortComponentsForGrouping(merged)) {
      let matched = null;
      for (const existing of next) {
        const yOverlap = Math.min(group.y + group.h, existing.y + existing.h) - Math.max(group.y, existing.y);
        const minHeight = Math.max(1, Math.min(group.h, existing.h));
        const gap = group.x - (existing.x + existing.w);
        if (yOverlap / minHeight >= 0.75 && gap >= -2 && gap <= 8) {
          matched = existing;
          break;
        }
      }
      if (matched) {
        const x2 = Math.max(matched.x + matched.w, group.x + group.w);
        const y2 = Math.max(matched.y + matched.h, group.y + group.h);
        matched.x = Math.min(matched.x, group.x);
        matched.y = Math.min(matched.y, group.y);
        matched.w = x2 - matched.x;
        matched.h = y2 - matched.y;
        matched.components = Number(matched.components || 1) + Number(group.components || 1);
        matched.pixels = Number(matched.pixels || 0) + Number(group.pixels || 0);
        changed = true;
      } else {
        next.push({ ...group });
      }
    }
    merged = next;
  }
  return merged;
}


function correctSamePageSequentialOcr(steps) {
  const valid = steps
    .filter((step) => Number(step.step_number) > 0 && step.step_box)
    .sort((a, b) => Number(a.step_box.y) - Number(b.step_box.y) || Number(a.step_box.x) - Number(b.step_box.x));
  for (let idx = 1; idx < valid.length; idx += 1) {
    const previous = valid[idx - 1];
    const current = valid[idx];
    const expected = Number(previous.step_number) + 1;
    const currentValue = Number(current.step_number);
    if (!Number.isFinite(expected) || !Number.isFinite(currentValue) || currentValue === expected) continue;
    const currentText = String(currentValue);
    const expectedText = String(expected);
    if (currentText.length !== expectedText.length) continue;
    let diffCount = 0;
    for (let pos = 0; pos < currentText.length; pos += 1) {
      if (currentText[pos] !== expectedText[pos]) diffCount += 1;
    }
    if (diffCount === 1) {
      current.step_number = expected;
      current.signals = {
        ...(current.signals || {}),
        step_number_sequence_corrected: true,
        step_number_before_sequence_correction: currentValue,
        step_number_expected_from_previous: expected,
      };
    }
  }
}


function collectStepGroups(dark, width, height, fullPage = false) {
  const seen = new Uint8Array(dark.length);
  const components = [];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (!dark[idx] || seen[idx]) continue;
      const box = componentBox(dark, width, height, x, y, seen);
      if (!componentPassesStepShape(box, width, height, fullPage)) continue;
      components.push(box);
    }
  }

  return mergeNeighboringGroups(
    groupComponents(components)
      .filter((box) => groupPassesStepShape(box, width, height))
  )
    .map((box) => ({ ...box, confidence: fullPage ? auditBoxScore(box, width, height) : boxScore(box, width, height) }))
    .filter((box) => box.confidence >= (fullPage ? 0.72 : 0.78))
    .sort((a, b) => b.confidence - a.confidence || a.y - b.y)
    .slice(0, fullPage ? 12 : 4);
}


function loadV1CropCache(v1CropCachePath) {
  if (!v1CropCachePath || !fs.existsSync(v1CropCachePath)) return [];
  const payload = JSON.parse(fs.readFileSync(v1CropCachePath, "utf8"));
  return (payload.entries || []).filter((entry) => entry && entry.crop_id && entry.crop_box);
}


async function buildV1StepEntry(sharp, repoRoot, pageEntry, entry, debugDir) {
  const imagePath = path.join(repoRoot, pageEntry.image_path);
  const metadata = await sharp(imagePath).metadata();
  const width = metadata.width || 0;
  const height = metadata.height || 0;
  const box = boxFromCropEntry(entry);
  if (!box) return null;
  const overlayPath = path.join(
    debugDir,
    `v1_crop_cache_bag_${String(entry.bag).padStart(2, "0")}_page_${String(entry.page).padStart(3, "0")}_step_${String(entry.step_number).padStart(3, "0")}_c${String(entry.crop_index || 0).padStart(2, "0")}.png`,
  );
  const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
    <rect x="${box.x}" y="${box.y}" width="${box.w}" height="${box.h}" fill="none" stroke="#00aaff" stroke-width="6"/>
    <rect x="${box.x}" y="${Math.max(0, box.y - 34)}" width="320" height="32" fill="black" opacity="0.82"/>
    <text x="${box.x + 8}" y="${Math.max(22, box.y - 11)}" font-family="Arial" font-size="19" fill="#00d0ff">V1 crop ${String(entry.crop_id || "")}</text>
  </svg>`;
  await sharp(imagePath).composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).png().toFile(overlayPath);
  return {
    bag: Number(entry.bag),
    page: Number(entry.page),
    step_index: Number(entry.crop_index || 0),
    step_number: Number(entry.step_number),
    rejection_reason: null,
    step_box: box,
    confidence: 1,
    debug_overlay_path: path.relative(repoRoot, overlayPath),
    crop_id: String(entry.crop_id || ""),
    crop_index: entry.crop_index === null || entry.crop_index === undefined ? null : Number(entry.crop_index),
    source: "v1_crop_cache",
    signals: {
      crop_box_source: "v1_crop_cache",
      crop_box_priority_rule: String(entry.crop_id || "") === "p23_s27_c1" ? "fresh_corrected_crop_box_beats_stale_saved_box" : "v1_crop_cache",
      v1_source: entry.v1_source || "",
      qty_text: entry.qty_text || [],
      qty_numbers: entry.qty_numbers || [],
    },
  };
}


async function analyzePage(sharp, repoRoot, pageEntry, bagNumber, debugDir, tmpDir) {
  const imagePath = path.join(repoRoot, pageEntry.image_path);
  const metadata = await sharp(imagePath).metadata();
  const width = metadata.width || 0;
  const height = metadata.height || 0;
  const raw = await sharp(imagePath).removeAlpha().raw().toBuffer();
  const dark = new Uint8Array(width * height);

  for (let i = 0; i < dark.length; i += 1) {
    const r = raw[i * 3];
    const g = raw[i * 3 + 1];
    const b = raw[i * 3 + 2];
    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
    const spread = Math.max(r, g, b) - Math.min(r, g, b);
    if (luminance < 72 && spread < 95) dark[i] = 1;
  }

  const groups = collectStepGroups(dark, width, height, false);
  const fullPageGroups = collectStepGroups(dark, width, height, true);

  const overlayRects = [];
  for (let idx = 0; idx < fullPageGroups.length; idx += 1) {
    const box = fullPageGroups[idx];
    const stepIndex = idx + 1;
    const read = await readPrintedStepNumber(sharp, imagePath, box, width, height, tmpDir, pageEntry.page, stepIndex);
    const filteredStep = filterInvalidStepNumber(read.value);
    box.step_number = filteredStep.step_number;
    box.step_number_rejection_reason = filteredStep.rejection_reason || read.rejection_reason || null;
    box.step_number_confidence = read.confidence;
    box.step_number_source = read.source;
    box.step_number_raw_text = read.raw_text;
    box.step_number_rejected_reads = read.rejected_reads || [];
    box.step_index = stepIndex;
  }

  for (let idx = 0; idx < groups.length; idx += 1) {
    const box = groups[idx];
    const stepIndex = idx + 1;
    const fullPageMatch = fullPageGroups.find((candidate) => boxesVisuallyMatch(candidate, box));
    if (fullPageMatch) {
      box.step_number = fullPageMatch.step_number;
      box.step_number_rejection_reason = fullPageMatch.step_number_rejection_reason;
      box.step_number_confidence = fullPageMatch.step_number_confidence;
      box.step_number_source = fullPageMatch.step_number_source;
      box.step_number_raw_text = fullPageMatch.step_number_raw_text;
      box.step_number_rejected_reads = fullPageMatch.step_number_rejected_reads;
    } else {
      const read = await readPrintedStepNumber(sharp, imagePath, box, width, height, tmpDir, pageEntry.page, stepIndex);
      const filteredStep = filterInvalidStepNumber(read.value);
      box.step_number = filteredStep.step_number;
      box.step_number_rejection_reason = filteredStep.rejection_reason || read.rejection_reason || null;
      box.step_number_confidence = read.confidence;
      box.step_number_source = read.source;
      box.step_number_raw_text = read.raw_text;
      box.step_number_rejected_reads = read.rejected_reads || [];
    }
    box.step_index = stepIndex;
    const stepLabel = box.step_number === null || box.step_number === undefined ? "?" : String(box.step_number);
    overlayRects.push(`<rect x="${box.x}" y="${box.y}" width="${box.w}" height="${box.h}" fill="none" stroke="#ffcc00" stroke-width="5"/><rect x="${box.x}" y="${Math.max(0, box.y - 32)}" width="210" height="30" fill="black" opacity="0.82"/><text x="${box.x + 6}" y="${Math.max(20, box.y - 10)}" font-family="Arial" font-size="18" fill="#ffcc00">page ${pageEntry.page} · step ${stepLabel}</text>`);
  }

  const overlayPath = path.join(debugDir, `bag_${String(bagNumber).padStart(2, "0")}_page_${String(pageEntry.page).padStart(3, "0")}.png`);
  if (groups.length) {
    const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">${overlayRects.join("")}</svg>`;
    await sharp(imagePath).composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).png().toFile(overlayPath);
  }

  // V1: visual candidate with no readable OCR digit is hard-rejected, never step_number=null
  const steps = groups.filter((box) => box.step_number !== null && box.step_number !== undefined).map((box) => ({
    bag: bagNumber,
    page: pageEntry.page,
    step_index: box.step_index,
    step_number: box.step_number,
    rejection_reason: box.step_number_rejection_reason || null,
    step_box: { x: box.x, y: box.y, w: box.w, h: box.h },
    confidence: box.confidence,
    debug_overlay_path: groups.length ? path.relative(repoRoot, overlayPath) : null,
    signals: {
      pixels: box.pixels,
      components: box.components,
      step_number_confidence: box.step_number_confidence,
      step_number_source: box.step_number_source,
      step_number_raw_text: box.step_number_raw_text,
      step_number_rejected_reads: box.step_number_rejected_reads,
    },
  }));
  correctSamePageSequentialOcr(steps);

  // V1: audit candidates without a readable step number cannot participate in sequence gap recovery
  const audit_candidates = fullPageGroups.filter((box) => box.step_number !== null && box.step_number !== undefined).map((box) => ({
    bag: bagNumber,
    page: pageEntry.page,
    step_index: box.step_index,
    step_number: box.step_number,
    rejection_reason: box.step_number_rejection_reason || null,
    step_box: { x: box.x, y: box.y, w: box.w, h: box.h },
    confidence: box.confidence,
    debug_overlay_path: groups.length ? path.relative(repoRoot, overlayPath) : null,
    source: "full_page_step_audit_candidate",
    signals: {
      pixels: box.pixels,
      components: box.components,
      step_number_confidence: box.step_number_confidence,
      step_number_source: box.step_number_source,
      step_number_raw_text: box.step_number_raw_text,
      step_number_rejected_reads: box.step_number_rejected_reads,
      audit_zone: Number(box.x) > width * 0.45 ? "right_column" : "left_column",
    },
  }));

  return { steps, audit_candidates };
}


function validStepNumbers(pageSteps) {
  return Array.from(new Set((pageSteps || [])
    .map((step) => Number(step.step_number))
    .filter((value) => Number.isFinite(value) && value > 0)))
    .sort((a, b) => a - b);
}


function stepBoxKey(step) {
  const box = step.step_box || {};
  return [step.page, box.x, box.y, box.w, box.h].map((value) => String(value ?? "")).join(":");
}


function promoteSequenceGapCandidates(pageRecords) {
  for (let idx = 0; idx < pageRecords.length - 1; idx += 1) {
    const current = pageRecords[idx];
    const next = pageRecords[idx + 1];
    const currentNumbers = validStepNumbers(current.steps);
    let nextNumbers = validStepNumbers(next.steps);
    if (!currentNumbers.length || !nextNumbers.length) continue;

    const currentMax = Math.max(...currentNumbers);
    nextNumbers = nextNumbers.filter((value) => value > currentMax);
    if (!nextNumbers.length) continue;
    const nextMin = Math.min(...nextNumbers);
    const gapSize = nextMin - currentMax - 1;
    if (gapSize <= 0 || gapSize > 6) continue;

    const existingNumbers = new Set(currentNumbers);
    const existingBoxes = new Set((current.steps || []).map(stepBoxKey));
    const missing = [];
    const recovered = [];
    const unresolved = [];
    for (let value = currentMax + 1; value < nextMin; value += 1) missing.push(value);

    for (const expected of missing) {
      const candidate = (current.audit_candidates || [])
        .filter((item) => Number(item.step_number) === expected)
        .filter((item) => !existingNumbers.has(expected))
        .filter((item) => !existingBoxes.has(stepBoxKey(item)))
        .sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0))[0];
      if (!candidate) {
        unresolved.push(expected);
        continue;
      }
      const promoted = {
        ...candidate,
        step_index: (current.steps || []).length + recovered.length + 1,
        source: "sequence_gap_full_page_audit",
        signals: {
          ...(candidate.signals || {}),
          sequence_gap_guardrail: true,
          expected_step_number: expected,
          previous_detected_step_number: currentMax,
          next_page: next.page,
          next_detected_step_number: nextMin,
          v1_reference: "clean/services/step_detector_service.py visual component sizing plus full-page sequence-gap audit",
        },
      };
      recovered.push(promoted);
      existingNumbers.add(expected);
      existingBoxes.add(stepBoxKey(promoted));
    }

    current.sequence_gap_diagnostics.push({
      rule: "sequence_gap_full_page_audit",
      page: current.page,
      next_page: next.page,
      previous_detected_step_number: currentMax,
      next_detected_step_number: nextMin,
      expected_missing_steps: missing,
      recovered_steps: recovered.map((step) => Number(step.step_number)),
      unresolved_steps: unresolved,
      status: unresolved.length ? "unresolved" : "recovered",
    });
    current.steps.push(...recovered);
  }
}


// ---------------------------------------------------------------------------
// Sequence-quality helpers
// ---------------------------------------------------------------------------

function trimmedContextMax(values, trimFraction) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const trimCount = Math.max(0, Math.floor(sorted.length * trimFraction));
  const trimmed = trimCount > 0 ? sorted.slice(0, sorted.length - trimCount) : sorted;
  return trimmed.length ? trimmed[trimmed.length - 1] : 0;
}


// Promote right-column full-page audit candidates that directly extend the
// left-column primary sequence on the SAME page (handles pages where primary
// scan misses right-column steps, e.g. page 79 steps 130/131).
function promoteRightColumnExtensions(pageRecords) {
  for (const record of pageRecords) {
    const primaryNumbers = validStepNumbers(record.steps);
    if (!primaryNumbers.length) continue;
    const primaryMax = Math.max(...primaryNumbers);
    const existingNumbers = new Set(primaryNumbers);
    const existingBoxes = new Set((record.steps || []).map(stepBoxKey));
    const promoted = [];

    for (let expected = primaryMax + 1; expected <= primaryMax + 6; expected += 1) {
      const candidate = (record.audit_candidates || [])
        .filter((c) => Number(c.step_number) === expected)
        .filter((c) => (c.signals || {}).audit_zone === "right_column")
        .filter((c) => !existingNumbers.has(expected))
        .filter((c) => !existingBoxes.has(stepBoxKey(c)))
        .sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0))[0];
      if (!candidate) break; // stop at first missing step in sequence

      const promotedStep = {
        ...candidate,
        step_index: (record.steps || []).length + promoted.length + 1,
        source: "right_column_same_page_extension",
        signals: {
          ...(candidate.signals || {}),
          promotion_reason: "right_column_audit_candidate_extends_left_column_primary_sequence",
          source_page: record.page,
          audit_bbox: candidate.step_box,
          ocr_text: (candidate.signals || {}).step_number_raw_text || "",
          primary_max: primaryMax,
          expected_step_number: expected,
        },
      };
      promoted.push(promotedStep);
      existingNumbers.add(expected);
      existingBoxes.add(stepBoxKey(promotedStep));
    }

    if (promoted.length > 0) {
      record.sequence_gap_diagnostics.push({
        rule: "right_column_same_page_extension",
        page: record.page,
        primary_max: primaryMax,
        promoted_steps: promoted.map((s) => ({
          step_number: Number(s.step_number),
          source_page: record.page,
          audit_bbox: s.step_box,
          ocr_text: (s.signals || {}).ocr_text || (s.signals || {}).step_number_raw_text || "",
          promotion_reason: "right_column_audit_candidate_extends_left_column_primary_sequence",
          expected_gap: `${primaryMax}→${Number(s.step_number)}`,
        })),
        source: "right_column_full_page_audit_candidates",
      });
      record.steps.push(...promoted);
    }
  }
}


// Reject primary steps that are large sequence outliers: > 30 above the
// trimmed-20 % context max of the surrounding ±4 pages AND unsupported by
// any forward anchor within 5 of that value.  Using a trimmed max prevents
// correlated outliers on adjacent pages from validating each other.
function rejectSequenceOutliers(pageRecords) {
  const windowSize = 4;
  const trimFraction = 0.20;
  const jumpThreshold = 30;

  for (let idx = 0; idx < pageRecords.length; idx += 1) {
    const record = pageRecords[idx];
    if (!record.steps.length) continue;

    const contextBefore = [];
    for (let j = Math.max(0, idx - windowSize); j < idx; j += 1) {
      contextBefore.push(...validStepNumbers(pageRecords[j].steps));
    }
    const contextAfter = [];
    for (let j = idx + 1; j <= Math.min(pageRecords.length - 1, idx + windowSize); j += 1) {
      contextAfter.push(...validStepNumbers(pageRecords[j].steps));
    }
    const contextAll = [...contextBefore, ...contextAfter];
    if (!contextAll.length) continue;

    const localMax = trimmedContextMax(contextAll, trimFraction);
    const forwardMax = trimmedContextMax(contextAfter, trimFraction);

    for (let k = record.steps.length - 1; k >= 0; k -= 1) {
      const step = record.steps[k];
      const stepNum = Number(step.step_number);
      if (!Number.isFinite(stepNum) || stepNum <= 0) continue;

      const farAboveContext = stepNum > localMax + jumpThreshold;
      const noForwardAnchor = forwardMax < stepNum - 5;

      if (farAboveContext && noForwardAnchor) {
        record.steps.splice(k, 1);
        record.sequence_gap_diagnostics.push({
          rule: "sequence_context_outlier_rejected",
          page: record.page,
          rejected_step_number: stepNum,
          local_context_max: localMax,
          forward_context_max: forwardMax,
          context_window_pages: windowSize,
          reason: `step_${stepNum}_exceeds_trimmed_local_max_${localMax}_by_more_than_${jumpThreshold}_with_no_forward_anchor`,
        });
      }
    }
  }
}


const pageIndexPath = argValue("--page-index");
const bagMapPath = argValue("--bag-map");
const repoRoot = argValue("--repo-root");
const outPath = argValue("--out");
const debugDir = argValue("--debug-dir");
const nodeModules = argValue("--node-modules");
const v1CropCachePath = argValue("--v1-crop-cache");

if (!pageIndexPath || !bagMapPath || !repoRoot || !outPath || !debugDir || !nodeModules) {
  console.error("Usage: step_map_scan.mjs --page-index <path> --bag-map <path> --repo-root <dir> --out <path> --debug-dir <dir> --node-modules <dir>");
  process.exit(2);
}

const requireFromBundle = createRequire(path.join(nodeModules, "package.json"));
const sharp = requireFromBundle("sharp");
fs.mkdirSync(debugDir, { recursive: true });
for (const entry of fs.readdirSync(debugDir)) {
  if (/^bag_\d{2}_page_\d{3}\.png$/.test(entry)) fs.unlinkSync(path.join(debugDir, entry));
}
const tmpDir = path.join(debugDir, "_ocr_tmp");
fs.rmSync(tmpDir, { recursive: true, force: true });
fs.mkdirSync(tmpDir, { recursive: true });

const pageIndex = JSON.parse(fs.readFileSync(pageIndexPath, "utf8"));
const bagMap = JSON.parse(fs.readFileSync(bagMapPath, "utf8"));
const pagesByNumber = new Map((pageIndex.pages || []).map((entry) => [Number(entry.page), entry]));
const v1Entries = loadV1CropCache(v1CropCachePath);
const steps = [];
const pages = [];

for (const bag of bagMap.bags || []) {
  const pageRecords = [];
  for (let page = Number(bag.start_page); page <= Number(bag.end_page); page += 1) {
    const pageEntry = pagesByNumber.get(page);
    if (!pageEntry) continue;
    const pageAnalysis = await analyzePage(sharp, repoRoot, pageEntry, Number(bag.bag), debugDir, tmpDir);
    const guardedPageSteps = applyV1StepAnchorGuardrails(pageAnalysis.steps, v1Entries);
    pageRecords.push({
      bag: Number(bag.bag),
      page,
      steps: guardedPageSteps,
      audit_candidates: pageAnalysis.audit_candidates || [],
      sequence_gap_diagnostics: [],
    });
  }

  promoteSequenceGapCandidates(pageRecords);
  promoteRightColumnExtensions(pageRecords);
  rejectSequenceOutliers(pageRecords);

  for (const record of pageRecords) {
    record.steps.sort((a, b) =>
      Number(a.step_number || 0) - Number(b.step_number || 0)
      || Number((a.step_box || {}).y || 0) - Number((b.step_box || {}).y || 0)
      || Number((a.step_box || {}).x || 0) - Number((b.step_box || {}).x || 0)
    );
    pages.push({
      bag: record.bag,
      page: record.page,
      step_candidate_count: record.steps.filter((step) => step.step_number !== null && step.step_number !== undefined).length,
      rejected_step_anchor_count: record.steps.filter((step) => step.rejection_reason).length,
      full_page_audit_candidate_count: (record.audit_candidates || []).filter((step) => step.step_number !== null && step.step_number !== undefined).length,
      full_page_audit_candidates: (record.audit_candidates || []).map((step) => ({
        step_number: step.step_number,
        rejection_reason: step.rejection_reason || null,
        step_box: step.step_box,
        confidence: step.confidence,
        source: step.source || null,
        signals: step.signals || {},
      })),
      sequence_gap_diagnostics: record.sequence_gap_diagnostics,
      debug_overlay_path: record.steps[0]?.debug_overlay_path || null,
    });
    steps.push(...record.steps);
  }
}

for (const entry of v1Entries) {
  const pageEntry = pagesByNumber.get(Number(entry.page));
  if (!pageEntry) continue;
  const v1Step = await buildV1StepEntry(sharp, repoRoot, pageEntry, entry, debugDir);
  if (v1Step) steps.push(v1Step);
}

steps.sort((a, b) =>
  Number(a.bag) - Number(b.bag)
  || Number(a.page) - Number(b.page)
  || Number(a.step_number || 0) - Number(b.step_number || 0)
  || String(a.crop_id || "").localeCompare(String(b.crop_id || ""))
);

const payload = {
  stage: 4,
  name: "step_box_map",
  input_manifests: ["indexes/04_bag_map.json", "indexes/02_page_index.json", "indexes/05c_v1_crop_cache_import.json"],
  method: "v1_crop_cache_priority_then_visual_step_box_candidates",
  step_count: steps.length,
  page_count: pages.length,
  debug_dir: path.relative(repoRoot, debugDir),
  steps,
  pages,
};

fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
fs.rmSync(tmpDir, { recursive: true, force: true });
console.log(JSON.stringify({ step_count: steps.length, page_count: pages.length, out: path.relative(repoRoot, outPath) }));
