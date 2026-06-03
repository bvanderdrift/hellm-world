/**
 * Charting for the `addy-new` correctness map.
 *
 * Reads a checkpoint's `correctness_grid.bin` and renders a downsampled heatmap
 * PNG (`correctness-heatmap.png`) so accuracy can be inspected visually. The
 * testing script (`correctness-map.ts`) calls `dumpImage` every 1000 tests; this
 * file can also be run on its own to (re)chart an already-saved grid.
 *
 * Usage: bun run models/addy-new/scripts/correctness-chart.ts <checkpointId>
 */

import { join } from "path";
import sharp from "sharp";

import {
  GRID_SIZE,
  MODEL_NAME,
  checkpointFolderPath,
  gridPath,
  loadGrid,
} from "./correctness-grid.ts";

const NULL = 0;
const TRUE = 2;

// Each plot pixel summarises a BOX x BOX square of grid cells.
const PLOT_SIZE = 1000; // 10_000 / 10
const BOX = GRID_SIZE / PLOT_SIZE; // 10

const escapeXml = (value: string) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

/** Convert HSL (h in [0,360), s,l in [0,1]) to RGB bytes. */
const hslToRgb = (
  h: number,
  s: number,
  l: number,
): [number, number, number] => {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;

  let r = 0;
  let g = 0;
  let bl = 0;
  if (h < 60) [r, g, bl] = [c, x, 0];
  else if (h < 120) [r, g, bl] = [x, c, 0];
  else if (h < 180) [r, g, bl] = [0, c, x];
  else if (h < 240) [r, g, bl] = [0, x, c];
  else if (h < 300) [r, g, bl] = [x, 0, c];
  else [r, g, bl] = [c, 0, x];

  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((bl + m) * 255),
  ];
};

/** Accuracy p in [0,1] -> colour. Red (0) -> yellow (0.5) -> green (1). */
const accuracyColor = (p: number): [number, number, number] =>
  hslToRgb(120 * p, 1, 0.5);

/**
 * Scan the full grid into a PLOT_SIZE x PLOT_SIZE raw RGB raster. Each pixel
 * averages the non-null cells in its BOX x BOX square; no data -> black.
 * Pixel (px, py): py corresponds to `a` (rows), px to `b` (columns).
 */
const buildRaster = (grid: Uint8Array): Buffer => {
  const raster = Buffer.alloc(PLOT_SIZE * PLOT_SIZE * 3);

  for (let py = 0; py < PLOT_SIZE; py++) {
    const aStart = py * BOX;
    for (let px = 0; px < PLOT_SIZE; px++) {
      const bStart = px * BOX;

      let data = 0;
      let trues = 0;
      for (let da = 0; da < BOX; da++) {
        const rowBase = (aStart + da) * GRID_SIZE + bStart;
        for (let db = 0; db < BOX; db++) {
          const value = grid[rowBase + db]!;
          if (value === NULL) continue;
          data++;
          if (value === TRUE) trues++;
        }
      }

      const offset = (py * PLOT_SIZE + px) * 3;
      if (data === 0) {
        // black, already zeroed
        continue;
      }
      const [r, g, b] = accuracyColor(trues / data);
      raster[offset] = r;
      raster[offset + 1] = g;
      raster[offset + 2] = b;
    }
  }

  return raster;
};

const margin = { top: 96, right: 40, bottom: 70, left: 80 };
const legendHeight = 18;

const buildSvg = async (
  grid: Uint8Array,
  checkpointId: number,
  testCount: number,
  correctCount: number,
): Promise<string> => {
  const raster = buildRaster(grid);
  const plotPng = await sharp(raster, {
    raw: { width: PLOT_SIZE, height: PLOT_SIZE, channels: 3 },
  })
    .png()
    .toBuffer();
  const plotDataUri = `data:image/png;base64,${plotPng.toString("base64")}`;

  const width = margin.left + PLOT_SIZE + margin.right;
  const height = margin.top + PLOT_SIZE + margin.bottom;

  // Axis ticks at every 1000 (0 .. 10000).
  const ticks = new Array(11).fill(0).map((_, i) => {
    const value = i * 1000;
    const along = (value / GRID_SIZE) * PLOT_SIZE;
    const x = margin.left + along;
    const y = margin.top + along;
    return `
    <line x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + PLOT_SIZE}" stroke="#ffffff22" />
    <text x="${x}" y="${margin.top + PLOT_SIZE + 20}" text-anchor="middle" class="axis-text">${value}</text>
    <line x1="${margin.left}" y1="${y}" x2="${margin.left + PLOT_SIZE}" y2="${y}" stroke="#ffffff22" />
    <text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" class="axis-text">${value}</text>`;
  });

  // Red -> yellow -> green legend.
  const legendStops = new Array(11).fill(0).map((_, i) => {
    const p = i / 10;
    const [r, g, b] = accuracyColor(p);
    return `<stop offset="${p * 100}%" stop-color="rgb(${r},${g},${b})" />`;
  });
  const legendWidth = 240;
  const legendX = margin.left;
  const legendY = 58;

  const accuracy = testCount > 0 ? (correctCount / testCount) * 100 : 0;
  const title = `${MODEL_NAME} checkpoint ${checkpointId} — correctness map`;
  const subtitle = `${testCount.toLocaleString()} tests · ${accuracy.toFixed(2)}% correct · each pixel = ${BOX}×${BOX} pairs (avg, black = untested)`;

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    .title { font: 700 24px system-ui, -apple-system, sans-serif; fill: #f9fafb; }
    .subtitle { font: 500 14px system-ui, -apple-system, sans-serif; fill: #9ca3af; }
    .axis-text { font: 11px system-ui, -apple-system, sans-serif; fill: #9ca3af; }
    .axis-label { font: 700 13px system-ui, -apple-system, sans-serif; fill: #d1d5db; }
  </style>
  <defs>
    <linearGradient id="legend" x1="0%" y1="0%" x2="100%" y2="0%">
      ${legendStops.join("")}
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="#111827" />
  <text x="${margin.left}" y="32" class="title">${escapeXml(title)}</text>
  <text x="${margin.left}" y="${legendY - 8}" class="subtitle">${escapeXml(subtitle)}</text>

  <rect x="${legendX}" y="${legendY}" width="${legendWidth}" height="${legendHeight}" fill="url(#legend)" rx="3" />
  <text x="${legendX}" y="${legendY + legendHeight + 14}" text-anchor="start" class="axis-text">0% (wrong)</text>
  <text x="${legendX + legendWidth}" y="${legendY + legendHeight + 14}" text-anchor="end" class="axis-text">100% (correct)</text>

  <image x="${margin.left}" y="${margin.top}" width="${PLOT_SIZE}" height="${PLOT_SIZE}" href="${plotDataUri}" style="image-rendering: pixelated" preserveAspectRatio="none" />
  <rect x="${margin.left}" y="${margin.top}" width="${PLOT_SIZE}" height="${PLOT_SIZE}" fill="none" stroke="#374151" />
  ${ticks.join("")}
  <text x="${margin.left + PLOT_SIZE / 2}" y="${height - 8}" text-anchor="middle" class="axis-label">b</text>
  <text transform="translate(20 ${margin.top + PLOT_SIZE / 2}) rotate(-90)" text-anchor="middle" class="axis-label">a</text>
</svg>`;
};

/** Render the grid to `correctness-heatmap.png` in the checkpoint folder. */
export const dumpImage = async (
  grid: Uint8Array,
  checkpointId: number,
  testCount: number,
  correctCount: number,
) => {
  const svg = await buildSvg(grid, checkpointId, testCount, correctCount);
  const outputPath = join(
    checkpointFolderPath(checkpointId),
    "correctness-heatmap.png",
  );
  await sharp(Buffer.from(svg)).png().toFile(outputPath);
  return outputPath;
};

const main = async () => {
  const checkpointArg = process.argv[2];
  if (checkpointArg === undefined || Number.isNaN(Number(checkpointArg))) {
    console.error(
      "Usage: bun run models/addy-new/scripts/correctness-chart.ts <checkpointId>",
    );
    process.exit(1);
  }
  const checkpointId = Number(checkpointArg);

  const { grid, testCount, correctCount } = loadGrid(checkpointId);
  if (testCount === 0) {
    console.error(
      `No grid data at ${gridPath(checkpointId)} — run the testing script first.`,
    );
    process.exit(1);
  }

  const outputPath = await dumpImage(
    grid,
    checkpointId,
    testCount,
    correctCount,
  );
  const accuracy = (correctCount / testCount) * 100;
  console.log(
    `${testCount.toLocaleString()} tests · ${accuracy.toFixed(2)}% correct · wrote ${outputPath}`,
  );
};

// Only run when invoked directly, not when imported by the testing script.
if (import.meta.main) {
  main();
}
