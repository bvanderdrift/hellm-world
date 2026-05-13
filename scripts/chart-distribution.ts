/**
 * THIS FILE IS AI-GENERATED
 */

import { mkdir, readFile } from "fs/promises";
import { dirname, join } from "path";
import sharp from "sharp";

const escapeXml = (value: string) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

export const writeDistributionChart = async (
  modelName: string,
  trainingDataFileName: string,
) => {
  const modelDirectory = join(import.meta.dirname, "..", "model", modelName);
  const dataPath = join(modelDirectory, trainingDataFileName);
  const outputPath = join(
    modelDirectory,
    trainingDataFileName.replace(".txt", "_distribution.png"),
  );
  const raw = await readFile(dataPath, "utf8");
  const lines = raw.split("\n").filter((line) => line.length > 0);

  const buckets = new Map<number, number>();
  for (const line of lines) {
    const match = line.match(/=(\d+)<EOS>/);
    if (!match) continue;
    const digits = match[1]!.length;
    buckets.set(digits, (buckets.get(digits) ?? 0) + 1);
  }

  const sortedKeys = [...buckets.keys()].sort((a, b) => a - b);
  const counts = sortedKeys.map((key) => buckets.get(key)!);
  const labels = sortedKeys.map((key) => `${key}-digit`);
  const maxCount = Math.max(...counts);
  const totalExamples = counts.reduce((sum, c) => sum + c, 0);

  const width = 1400;
  const height = 820;
  const margin = { top: 72, right: 52, bottom: 86, left: 92 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const barGap = 24;
  const barWidth =
    (chartWidth - barGap * (sortedKeys.length - 1)) / sortedKeys.length;
  const yMax = Math.ceil(maxCount / 10000) * 10000;
  const yFor = (value: number) =>
    margin.top + ((yMax - value) / yMax) * chartHeight;

  const yTicks = new Array(6).fill(0).map((_, index) => {
    const value = (yMax * index) / 5;
    const y = yFor(value);

    return `
    <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#e5e7eb" />
    <text x="${margin.left - 14}" y="${y + 5}" text-anchor="end" class="axis-text">${value.toLocaleString("en-US")}</text>`;
  });

  const bars = sortedKeys.map((_, index) => {
    const x = margin.left + index * (barWidth + barGap);
    const barHeight = (counts[index]! / yMax) * chartHeight;
    const y = margin.top + chartHeight - barHeight;
    const pct = ((counts[index]! / totalExamples) * 100).toFixed(1);

    return `
    <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" fill="#2563eb" rx="4" />
    <text x="${x + barWidth / 2}" y="${y - 10}" text-anchor="middle" class="axis-text" fill="#111827" font-weight="600">${counts[index]!.toLocaleString("en-US")} (${pct}%)</text>
    <text x="${x + barWidth / 2}" y="${height - margin.bottom + 34}" text-anchor="middle" class="axis-text">${labels[index]}</text>`;
  });

  const title = `${modelName} training data distribution`;
  const subtitle = `${totalExamples.toLocaleString("en-US")} examples from ${trainingDataFileName}`;
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
  ${yTicks.join("")}
  ${bars.join("")}
  <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#111827" stroke-width="2" />
  <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#111827" stroke-width="2" />
  <text x="${width / 2}" y="${height - 24}" text-anchor="middle" class="axis-label">output digit count</text>
  <text transform="translate(26 ${height / 2}) rotate(-90)" text-anchor="middle" class="axis-label">number of examples</text>
</svg>`;

  await mkdir(dirname(outputPath), { recursive: true });
  await sharp(Buffer.from(svg)).png().toFile(outputPath);

  return outputPath;
};

const [modelName, dataFile] = process.argv.slice(2);
if (!modelName || !dataFile) {
  console.error("Usage: bun run scripts/chart-distribution.ts <model> <data-file.txt>");
  process.exit(1);
}
writeDistributionChart(modelName, dataFile).then((path) =>
  console.log(`Written: ${path}`),
);
