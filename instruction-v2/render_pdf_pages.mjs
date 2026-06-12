import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";


function argValue(name) {
  const idx = process.argv.indexOf(name);
  if (idx === -1 || idx + 1 >= process.argv.length) {
    return null;
  }
  return process.argv[idx + 1];
}


const pdfPath = argValue("--pdf");
const outDir = argValue("--out-dir");
const scale = Number(argValue("--scale") || "2");
const nodeModules = argValue("--node-modules");

if (!pdfPath || !outDir || !nodeModules) {
  console.error("Usage: render_pdf_pages.mjs --pdf <path> --out-dir <dir> --node-modules <dir> [--scale 2]");
  process.exit(2);
}

const requireFromBundle = createRequire(path.join(nodeModules, "package.json"));
const pdfjsUrl = pathToFileURL(requireFromBundle.resolve("pdfjs-dist/legacy/build/pdf.mjs")).href;
const canvasUrl = pathToFileURL(requireFromBundle.resolve("@napi-rs/canvas")).href;
const pdfjs = await import(pdfjsUrl);
const { createCanvas } = await import(canvasUrl);

fs.mkdirSync(outDir, { recursive: true });

const data = new Uint8Array(fs.readFileSync(pdfPath));
const doc = await pdfjs.getDocument({ data, disableWorker: true }).promise;
const pages = [];

for (let pageNumber = 1; pageNumber <= doc.numPages; pageNumber += 1) {
  const page = await doc.getPage(pageNumber);
  const viewport = page.getViewport({ scale });
  const width = Math.ceil(viewport.width);
  const height = Math.ceil(viewport.height);
  const canvas = createCanvas(width, height);
  const canvasContext = canvas.getContext("2d");

  await page.render({ canvasContext, viewport }).promise;

  const fileName = `page_${String(pageNumber).padStart(3, "0")}.png`;
  const imagePath = path.join(outDir, fileName);
  fs.writeFileSync(imagePath, canvas.toBuffer("image/png"));
  pages.push({ page: pageNumber, file_name: fileName, width, height });
}

console.log(JSON.stringify({ page_count: pages.length, pages }));
