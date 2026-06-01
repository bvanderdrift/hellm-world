/**
 * Correctness map for the `addy-new` addition model.
 *
 * Given a checkpoint version number it loads that exact checkpoint, then forever
 * samples random pairs (a, b) with a, b < 10_000, runs the model on "a+b=" and
 * records whether the fully-correct answer is produced. Every 1000 tests it dumps
 * a downsampled heatmap PNG so accuracy can be inspected visually.
 *
 * This is an independent script: it only imports already-exported inference
 * helpers, it does not modify any inference/training code.
 *
 * Usage: bun run model/addy-new/scripts/correctness-map.ts <checkpointId>
 */

import { readFileSync } from "fs";
import { join } from "path";
import sharp from "sharp";

import { getModelFolderPath } from "../../../model/model-io.ts";
import { modelMetadataSchema, type Model } from "../../../model/model-types.ts";
import { tokenize } from "../../../shared/tokenizer.ts";
import { getRawVector } from "../../../shared/matrices.ts";
import { END_OF_SEQUENCE_TOKEN } from "../../../shared/const.ts";
import {
  getHighestValueIndex,
  llmForwardPassByTokens,
} from "../../../running/llm.ts";
import {
  getCheckpointTrainingState,
  unwrapFlatWeights,
} from "../../../model/model-checkpoint-io.ts";

const MODEL_NAME = "addy-new";

// Result grid is 10_000 x 10_000 = 100M cells.
const GRID_SIZE = 10_000;
const CELL_COUNT = GRID_SIZE * GRID_SIZE;

// Cell encoding (acts as the requested (boolean | null)[][]).
const NULL = 0;
const FALSE = 1;
const TRUE = 2;

// Each plot pixel summarises a BOX x BOX square of grid cells.
const PLOT_SIZE = 1000; // 10_000 / 10
const BOX = GRID_SIZE / PLOT_SIZE; // 10

const DUMP_EVERY = 1000;

const escapeXml = (value: string) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

/** Load a *specific* checkpoint into a Model, reusing exported IO helpers. */
const loadCheckpoint = (modelName: string, checkpointId: number): Model => {
  const modelFolderPath = getModelFolderPath(modelName);

  const metadata = modelMetadataSchema.parse(
    JSON.parse(
      readFileSync(join(modelFolderPath, "_metadata.json")).toString(),
    ),
  );

  const checkpointFile = `checkpoint_${checkpointId.toString().padStart(6, "0")}.bin`;
  const buffer = readFileSync(join(modelFolderPath, checkpointFile));
  const weights = unwrapFlatWeights(metadata, buffer);

  const history = getCheckpointTrainingState(modelFolderPath, checkpointId);

  return { ...metadata, ...weights, trainingState: history };
};

/** Greedy argmax of the next-token logits for a given token sequence. */
const nextToken = (tokens: string[], model: Model): string => {
  const { embeddings: unembeddedState } = llmForwardPassByTokens(
    tokens,
    model,
    false,
  );
  const logits = getRawVector(unembeddedState, unembeddedState.vectors - 1);
  const index = getHighestValueIndex(logits);
  const token = model.vocabulary[index];
  if (!token) {
    throw new Error(`Failed to find token at index ${index}`);
  }
  return token;
};

/**
 * Returns true iff the model produces exactly `expected` digits followed by
 * <EOS>. Stops generating as soon as a token diverges from the expected answer.
 */
const isCorrect = (a: number, b: number, model: Model): boolean => {
  const input = `${a}+${b}=`;
  const inputTokens = tokenize(input, model.vocabulary);
  const expected = String(a + b);

  const generated: string[] = [];

  // expected.length digit tokens, then one <EOS>. Small cap guards runaways.
  const maxTokens = expected.length + 1;

  for (let step = 0; step < maxTokens; step++) {
    const token = nextToken([...inputTokens, ...generated], model);

    if (step < expected.length) {
      // Still expecting a digit; it must match exactly.
      if (token !== expected[step]) {
        return false;
      }
    } else {
      // All digits matched; the sequence must now terminate.
      return token === END_OF_SEQUENCE_TOKEN;
    }

    generated.push(token);
  }

  return false;
};

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

const dumpImage = async (
  grid: Uint8Array,
  checkpointId: number,
  testCount: number,
  correctCount: number,
) => {
  const svg = await buildSvg(grid, checkpointId, testCount, correctCount);
  const outputPath = join(
    getModelFolderPath(MODEL_NAME),
    `correctness_${checkpointId}_${testCount}.png`,
  );
  await sharp(Buffer.from(svg)).png().toFile(outputPath);
  return outputPath;
};

const PROGRESS_WIDTH = 30;

/** In-place progress bar towards the next render, e.g. "[------        ]  37%". */
const writeProgress = (fraction: number) => {
  const filled = Math.round(fraction * PROGRESS_WIDTH);
  const bar = "-".repeat(filled) + " ".repeat(PROGRESS_WIDTH - filled);
  const pct = Math.round(fraction * 100)
    .toString()
    .padStart(3);
  process.stdout.write(`\r[${bar}] ${pct}%`);
};

const main = async () => {
  const checkpointArg = process.argv[2];
  if (checkpointArg === undefined || Number.isNaN(Number(checkpointArg))) {
    console.error(
      "Usage: bun run model/addy-new/scripts/correctness-map.ts <checkpointId>",
    );
    process.exit(1);
  }
  const checkpointId = Number(checkpointArg);

  console.log(`Loading ${MODEL_NAME} checkpoint ${checkpointId}...`);
  const model = loadCheckpoint(MODEL_NAME, checkpointId);

  const grid = new Uint8Array(CELL_COUNT); // all NULL (0)
  let testCount = 0;
  let correctCount = 0;

  console.log("Sampling forever — Ctrl-C to stop.\n");

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const a = Math.floor(Math.random() * GRID_SIZE);
    const b = Math.floor(Math.random() * GRID_SIZE);

    const correct = isCorrect(a, b, model);
    grid[a * GRID_SIZE + b] = correct ? TRUE : FALSE;

    testCount++;
    if (correct) correctCount++;

    const intoBatch = testCount % DUMP_EVERY;
    if (intoBatch === 0) {
      // Batch complete: finish the bar, render, then start a fresh bar at 0%.
      writeProgress(1);
      process.stdout.write("\n");

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
      process.stdout.write("\n");
    } else {
      writeProgress(intoBatch / DUMP_EVERY);
    }
  }
};

main();
