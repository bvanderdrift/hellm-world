/**
 * Charting for the `addy-new` correctness map.
 *
 * Reads a checkpoint's `correctness_grid.bin` and renders a downsampled
 * PNG (`correctness-<type>.png`) so accuracy can be inspected visually. Two
 * render types are supported:
 *   - "heatmap" (default): green = all correct, red = any mistake, black = untested.
 *   - "faultmap": white = any wrong answer, black = everything else.
 * The testing script (`correctness-map.ts`) calls `dumpImage` every 1000 tests;
 * this file can also be run on its own to (re)chart an already-saved grid.
 *
 * Usage: bun run models/addy-new/scripts/correctness-chart.ts <checkpointId> [type]
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

// Render modes:
// - "heatmap": green = all correct, red = any mistake, black = untested.
// - "faultmap": white = any wrong answer, black = everything else (correct or untested).
const RENDER_TYPES = ["heatmap", "faultmap"] as const;
export type RenderType = (typeof RENDER_TYPES)[number];

// Each plot pixel summarises a BOX x BOX square of grid cells.
const PLOT_SIZE = 1000; // 10_000 / 10
const BOX = GRID_SIZE / PLOT_SIZE; // 10

const escapeXml = (value: string) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

// Binary colouring: a box is green only if it has zero mistakes, red otherwise.
const CORRECT_COLOR: [number, number, number] = [0, 255, 0]; // green
const WRONG_COLOR: [number, number, number] = [255, 0, 0]; // red
const FAULT_COLOR: [number, number, number] = [255, 255, 255]; // white

/**
 * Scan the full grid into a PLOT_SIZE x PLOT_SIZE raw RGB raster. Each pixel
 * summarises the non-null cells in its BOX x BOX square; no data -> black.
 * Pixel (px, py): py corresponds to `a` (rows), px to `b` (columns).
 *
 * Colouring depends on `type`:
 * - "heatmap": green if every tested cell is correct, red if any mistake,
 *   black if untested.
 * - "faultmap": white if any wrong answer, black otherwise (correct/untested).
 */
const buildRaster = (grid: Uint8Array, type: RenderType): Buffer => {
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
      const hasMistake = data > 0 && trues < data;

      let color: [number, number, number] | null;
      if (type === "faultmap") {
        // White marks any wrong answer; everything else stays black.
        color = hasMistake ? FAULT_COLOR : null;
      } else {
        // Heatmap: untested -> black, all-correct -> green, any mistake -> red.
        color = data === 0 ? null : hasMistake ? WRONG_COLOR : CORRECT_COLOR;
      }

      if (color === null) continue; // black, already zeroed
      raster[offset] = color[0];
      raster[offset + 1] = color[1];
      raster[offset + 2] = color[2];
    }
  }

  return raster;
};

const margin = { top: 96, right: 40, bottom: 70, left: 80 };

const buildSvg = async (
  grid: Uint8Array,
  checkpointId: number,
  testCount: number,
  correctCount: number,
  type: RenderType,
): Promise<string> => {
  const raster = buildRaster(grid, type);
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

  const swatch = 18;
  const legendX = margin.left;
  const legendY = 58;
  const correctRgb = `rgb(${CORRECT_COLOR.join(",")})`;
  const wrongRgb = `rgb(${WRONG_COLOR.join(",")})`;

  const faultRgb = `rgb(${FAULT_COLOR.join(",")})`;

  const accuracy = testCount > 0 ? (correctCount / testCount) * 100 : 0;
  const mapLabel = type === "faultmap" ? "fault map" : "correctness map";
  const summary =
    type === "faultmap"
      ? `each pixel = ${BOX}×${BOX} pairs (white = any wrong answer, black = correct/untested)`
      : `each pixel = ${BOX}×${BOX} pairs (red = any mistake, black = untested)`;
  const title = `${MODEL_NAME} checkpoint ${checkpointId} — ${mapLabel}`;
  const subtitle = `${testCount.toLocaleString()} tests · ${accuracy.toFixed(2)}% correct · ${summary}`;

  const legend =
    type === "faultmap"
      ? `<rect x="${legendX}" y="${legendY}" width="${swatch}" height="${swatch}" fill="${faultRgb}" rx="3" />
  <text x="${legendX + swatch + 8}" y="${legendY + swatch - 4}" text-anchor="start" class="axis-text">≥1 wrong</text>`
      : `<rect x="${legendX}" y="${legendY}" width="${swatch}" height="${swatch}" fill="${correctRgb}" rx="3" />
  <text x="${legendX + swatch + 8}" y="${legendY + swatch - 4}" text-anchor="start" class="axis-text">all correct</text>
  <rect x="${legendX + 140}" y="${legendY}" width="${swatch}" height="${swatch}" fill="${wrongRgb}" rx="3" />
  <text x="${legendX + 140 + swatch + 8}" y="${legendY + swatch - 4}" text-anchor="start" class="axis-text">≥1 mistake</text>`;

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    .title { font: 700 24px system-ui, -apple-system, sans-serif; fill: #f9fafb; }
    .subtitle { font: 500 14px system-ui, -apple-system, sans-serif; fill: #9ca3af; }
    .axis-text { font: 11px system-ui, -apple-system, sans-serif; fill: #9ca3af; }
    .axis-label { font: 700 13px system-ui, -apple-system, sans-serif; fill: #d1d5db; }
  </style>
  <rect width="100%" height="100%" fill="#111827" />
  <text x="${margin.left}" y="32" class="title">${escapeXml(title)}</text>
  <text x="${margin.left}" y="${legendY - 8}" class="subtitle">${escapeXml(subtitle)}</text>

  ${legend}

  <image x="${margin.left}" y="${margin.top}" width="${PLOT_SIZE}" height="${PLOT_SIZE}" href="${plotDataUri}" style="image-rendering: pixelated" preserveAspectRatio="none" />
  <rect x="${margin.left}" y="${margin.top}" width="${PLOT_SIZE}" height="${PLOT_SIZE}" fill="none" stroke="#374151" />
  ${ticks.join("")}
  <text x="${margin.left + PLOT_SIZE / 2}" y="${height - 8}" text-anchor="middle" class="axis-label">b</text>
  <text transform="translate(20 ${margin.top + PLOT_SIZE / 2}) rotate(-90)" text-anchor="middle" class="axis-label">a</text>
</svg>`;
};

/**
 * Render the grid to `correctness-<type>.png` in the checkpoint folder.
 * Defaults to the "heatmap" rendering.
 */
export const dumpImage = async (
  grid: Uint8Array,
  checkpointId: number,
  testCount: number,
  correctCount: number,
  type: RenderType = "heatmap",
) => {
  const svg = await buildSvg(grid, checkpointId, testCount, correctCount, type);
  const outputPath = join(
    checkpointFolderPath(checkpointId),
    `correctness-${type}.png`,
  );
  await sharp(Buffer.from(svg)).png().toFile(outputPath);
  return outputPath;
};

const main = async () => {
  const checkpointArg = process.argv[2];
  if (checkpointArg === undefined || Number.isNaN(Number(checkpointArg))) {
    console.error(
      "Usage: bun run models/addy-new/scripts/correctness-chart.ts <checkpointId> [type]",
    );
    console.error(`  type: ${RENDER_TYPES.join(" | ")} (default: heatmap)`);
    process.exit(1);
  }
  const checkpointId = Number(checkpointArg);

  const typeArg = process.argv[3] ?? "heatmap";
  if (!RENDER_TYPES.includes(typeArg as RenderType)) {
    console.error(
      `Unknown type "${typeArg}" — expected one of: ${RENDER_TYPES.join(", ")}`,
    );
    process.exit(1);
  }
  const type = typeArg as RenderType;

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
    type,
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
