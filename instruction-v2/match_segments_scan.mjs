import fs from 'fs/promises';
import path from 'path';
import { createRequire } from 'module';

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const key = arg.slice(2);
    const value = argv[i + 1];
    if (value && !value.startsWith('--')) {
      args[key] = value;
      i += 1;
    } else {
      args[key] = true;
    }
  }
  return args;
}

const args = parseArgs(process.argv);
const nodeModules = args['node-modules'];
if (nodeModules) {
  process.env.NODE_PATH = nodeModules;
}
const require = createRequire(import.meta.url);
if (nodeModules) {
  require('module').Module._initPaths();
}
const sharp = require('sharp');

const repoRoot = args['repo-root'];
const setContextPath = args['set-context'];
const segmentationMapPath = args['segmentation-map'];
const outPath = args.out;

if (!repoRoot || !setContextPath || !segmentationMapPath || !outPath) {
  throw new Error('Usage: match_segments_scan.mjs --set-context PATH --segmentation-map PATH --repo-root DIR --out PATH');
}

function resolveRepoPath(value) {
  if (!value) return null;
  if (path.isAbsolute(value)) return value;
  return path.join(repoRoot, value);
}

function parseRgb(value) {
  if (typeof value !== 'string') return null;
  const cleaned = value.trim().replace(/^#/, '');
  if (!/^[0-9a-fA-F]{6}$/.test(cleaned)) return null;
  return {
    r: Number.parseInt(cleaned.slice(0, 2), 16),
    g: Number.parseInt(cleaned.slice(2, 4), 16),
    b: Number.parseInt(cleaned.slice(4, 6), 16),
  };
}

function colorDistance(a, b) {
  const dr = a.r - b.r;
  const dg = a.g - b.g;
  const db = a.b - b.b;
  return Math.sqrt((dr * dr) + (dg * dg) + (db * db));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function round(value, places = 4) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function buildCandidatePool(parts) {
  const byKey = new Map();
  for (const part of parts) {
    const rgb = parseRgb(part.rgb);
    if (!rgb) continue;
    const key = `${part.part_num}__${part.color_id}`;
    const current = byKey.get(key);
    const qty = Number(part.set_required_qty ?? part.qty ?? 0);
    if (!current || qty > current.qty) {
      byKey.set(key, {
        part_num: String(part.part_num),
        color_id: Number(part.color_id),
        color_name: part.color_name ?? null,
        rgb_hex: part.rgb,
        rgb,
        qty,
        element_id: part.element_id ?? null,
      });
    }
  }
  return [...byKey.values()];
}

async function extractForegroundStats(cutoutPath) {
  const image = sharp(cutoutPath).ensureAlpha();
  const meta = await image.metadata();
  const { data, info } = await image.raw().toBuffer({ resolveWithObject: true });
  let count = 0;
  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let minX = info.width;
  let minY = info.height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < info.height; y += 1) {
    for (let x = 0; x < info.width; x += 1) {
      const offset = ((y * info.width) + x) * info.channels;
      const alpha = data[offset + 3];
      if (alpha <= 16) continue;
      count += 1;
      rSum += data[offset];
      gSum += data[offset + 1];
      bSum += data[offset + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }

  if (!count) {
    return {
      width: meta.width ?? info.width,
      height: meta.height ?? info.height,
      foreground_pixels: 0,
      foreground_rgb: null,
      foreground_box: null,
      density: 0,
      aspect_ratio: null,
    };
  }

  const boxW = (maxX - minX) + 1;
  const boxH = (maxY - minY) + 1;
  return {
    width: meta.width ?? info.width,
    height: meta.height ?? info.height,
    foreground_pixels: count,
    foreground_rgb: {
      r: Math.round(rSum / count),
      g: Math.round(gSum / count),
      b: Math.round(bSum / count),
    },
    foreground_box: { x: minX, y: minY, w: boxW, h: boxH },
    density: round(count / Math.max(1, boxW * boxH), 4),
    aspect_ratio: round(boxW / Math.max(1, boxH), 4),
  };
}

function rankCandidates(stats, candidatePool) {
  if (!stats.foreground_rgb) return [];

  return candidatePool
    .map((candidate) => {
      const distance = colorDistance(stats.foreground_rgb, candidate.rgb);
      const colorScore = clamp(1 - (distance / 441.6729), 0, 1);
      const densityScore = stats.density ? clamp(1 - Math.abs(stats.density - 0.55), 0, 1) : 0.5;
      const score = (colorScore * 0.92) + (densityScore * 0.08);
      return {
        part_num: candidate.part_num,
        color_id: candidate.color_id,
        color_name: candidate.color_name,
        score: round(score),
        source: 'instruction-v2:set_context_rgb_foreground_cutout_match',
        expected_in_set: true,
        components: {
          color_score: round(colorScore),
          color_distance: round(distance, 2),
          density_score: round(densityScore),
          catalog_rgb: candidate.rgb_hex,
          foreground_rgb: stats.foreground_rgb,
          set_required_qty: candidate.qty,
        },
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
}

async function main() {
  const setContext = JSON.parse(await fs.readFile(setContextPath, 'utf8'));
  const segmentationMap = JSON.parse(await fs.readFile(segmentationMapPath, 'utf8'));
  const candidatePool = buildCandidatePool(setContext.parts ?? []);
  if (!candidatePool.length) {
    throw new Error('No expected catalog candidates with RGB values found in 00_set_context.json');
  }

  const entries = [];
  for (const segment of segmentationMap.entries ?? []) {
    const cutoutPath = resolveRepoPath(segment.cutout_path);
    const stats = await extractForegroundStats(cutoutPath);
    entries.push({
      bag: segment.bag,
      page: segment.page,
      step: segment.step ?? null,
      crop_id: segment.crop_id,
      segment_index: segment.segment_index,
      cutout_path: segment.cutout_path,
      qty_numbers: segment.qty_numbers ?? [],
      top_candidates: rankCandidates(stats, candidatePool),
      source: 'instruction-v2:set_context_rgb_foreground_cutout_match',
      expected_in_set: true,
      segment_metrics: {
        segment_box: segment.segment_box,
        foreground_pixels: stats.foreground_pixels,
        foreground_rgb: stats.foreground_rgb,
        foreground_box: stats.foreground_box,
        density: stats.density,
        aspect_ratio: stats.aspect_ratio,
      },
    });
  }

  const payload = {
    stage: 8,
    name: 'match_manifest',
    input_manifests: [
      'indexes/00_set_context.json',
      'indexes/08_part_segmentation_map.json',
    ],
    method: 'set_context_rgb_foreground_cutout_match_v1',
    entry_count: entries.length,
    candidate_pool_count: candidatePool.length,
    entries,
  };

  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, `${JSON.stringify(payload, null, 2)}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
