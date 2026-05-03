/**
 * THIS FILE IS AI-GENERATED
 */

import { mkdir, readFile } from "fs/promises";
import { dirname, join } from "path";
import sharp from "sharp";

type Checkpoint = {
  historyLosses: number[];
};

const escapeXml = (value: string) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

export const writeLossChart = async (
  modelName: string,
  checkpointName: string,
) => {
  const modelDirectory = join(import.meta.dirname, "..", "model", modelName);
  const checkpointPath = join(modelDirectory, checkpointName);
  const outputPath = join(
    modelDirectory,
    checkpointName.replace(".json", "_loss.png"),
  );
  const checkpoint = JSON.parse(
    await readFile(checkpointPath, "utf8"),
  ) as Checkpoint;
  const losses = checkpoint.historyLosses;

  if (!Array.isArray(losses) || losses.length === 0) {
    throw new Error(`Checkpoint ${checkpointPath} has no loss history`);
  }

  const width = 1400;
  const height = 820;
  const margin = {
    top: 72,
    right: 52,
    bottom: 86,
    left: 92,
  };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const maxLoss = Math.max(...losses);
  const minLoss = Math.min(...losses);
  const yMin = 0;
  const yMax = Math.ceil(maxLoss * 10) / 10;
  const xFor = (index: number) =>
    margin.left + (index / Math.max(losses.length - 1, 1)) * chartWidth;
  const yFor = (loss: number) =>
    margin.top + ((yMax - loss) / (yMax - yMin)) * chartHeight;
  const points = losses
    .map((loss, index) => `${xFor(index).toFixed(2)},${yFor(loss).toFixed(2)}`)
    .join(" ");
  const yTicks = new Array(6).fill(0).map((_, index) => {
    const value = yMin + ((yMax - yMin) * index) / 5;
    const y = yFor(value);

    return `
    <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#e5e7eb" />
    <text x="${margin.left - 14}" y="${y + 5}" text-anchor="end" class="axis-text">${value.toFixed(2)}</text>`;
  });
  const xTicks = new Array(7).fill(0).map((_, index) => {
    const step = Math.round(((losses.length - 1) * index) / 6);
    const x = xFor(step);

    return `
    <line x1="${x}" y1="${margin.top}" x2="${x}" y2="${height - margin.bottom}" stroke="#f3f4f6" />
    <text x="${x}" y="${height - margin.bottom + 34}" text-anchor="middle" class="axis-text">${step + 1}</text>`;
  });
  const title = `${modelName} ${checkpointName} loss history`;
  const subtitle = `${losses.length.toLocaleString("en-US")} steps - start ${losses[0]!.toFixed(4)} - min ${minLoss.toFixed(4)} - final ${losses[losses.length - 1]!.toFixed(4)}`;
  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    .title { font: 700 32px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }
    .subtitle { font: 500 18px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5563; }
    .axis-text { font: 14px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #6b7280; }
    .axis-label { font: 700 16px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="${margin.left}" y="38" class="title">${escapeXml(title)}</text>
  <text x="${margin.left}" y="66" class="subtitle">${escapeXml(subtitle)}</text>
  ${xTicks.join("")}
  ${yTicks.join("")}
  <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#111827" stroke-width="2" />
  <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#111827" stroke-width="2" />
  <polyline points="${points}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
  <circle cx="${xFor(losses.length - 1)}" cy="${yFor(losses[losses.length - 1]!)}" r="5" fill="#dc2626" />
  <text x="${width / 2}" y="${height - 24}" text-anchor="middle" class="axis-label">training step</text>
  <text transform="translate(26 ${height / 2}) rotate(-90)" text-anchor="middle" class="axis-label">average loss</text>
</svg>`;

  await mkdir(dirname(outputPath), { recursive: true });
  await sharp(Buffer.from(svg)).png().toFile(outputPath);

  return outputPath;
};
