/**
 * THIS FILE IS AI-GENERATED
 */

import { mkdir, writeFile } from "fs/promises";
import { join } from "path";
import sharp from "sharp";
import { runLlm } from "../../../running/llm.ts";

const escapeXml = (value: string) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

type TestResult = {
  input: string;
  expected: number;
  output: string;
  correct: boolean;
};

type MagnitudeResult = {
  label: string;
  maxDigit: number;
  results: TestResult[];
};

const generatePairs = (max: number, count: number): [number, number][] => {
  const pairs: [number, number][] = [];
  for (let i = 0; i < count; i++) {
    const a = Math.floor(Math.random() * max);
    const b = Math.floor(Math.random() * max);
    pairs.push([a, b]);
  }
  return pairs;
};

const runTest = (a: number, b: number, modelName: string): TestResult => {
  const input = `${a}+${b}=`;
  const expected = a + b;

  const tokens: string[] = [];
  for (const token of runLlm(input, modelName)) {
    tokens.push(token);
  }
  const output = tokens.join("");

  return {
    input,
    expected,
    output,
    correct: output === String(expected),
  };
};

const magnitudes = [
  { label: "1-digit (0-9)", maxDigit: 10 },
  { label: "2-digit (0-99)", maxDigit: 100 },
  { label: "3-digit (0-999)", maxDigit: 1000 },
];

const writeResultsChart = async (
  modelName: string,
  allResults: MagnitudeResult[],
) => {
  const modelDirectory = join(import.meta.dirname, "..");
  const outputPath = join(modelDirectory, "test_accuracy.png");

  const labels = allResults.map((r) => r.label);
  const correctCounts = allResults.map(
    (r) => r.results.filter((t) => t.correct).length,
  );
  const totals = allResults.map((r) => r.results.length);
  const percentages = allResults.map(
    (_, i) => (correctCounts[i]! / totals[i]!) * 100,
  );

  const width = 1400;
  const height = 820;
  const margin = { top: 72, right: 52, bottom: 86, left: 92 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const barGap = 24;
  const barWidth =
    (chartWidth - barGap * (labels.length - 1)) / labels.length;
  const yMax = 100;
  const yFor = (value: number) =>
    margin.top + ((yMax - value) / yMax) * chartHeight;

  const yTicks = new Array(6).fill(0).map((_, index) => {
    const value = (yMax * index) / 5;
    const y = yFor(value);

    return `
    <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#e5e7eb" />
    <text x="${margin.left - 14}" y="${y + 5}" text-anchor="end" class="axis-text">${value}%</text>`;
  });

  const bars = labels.map((_, index) => {
    const x = margin.left + index * (barWidth + barGap);
    const pct = percentages[index]!;
    const barHeight = (pct / yMax) * chartHeight;
    const y = margin.top + chartHeight - barHeight;
    const fill = pct === 100 ? "#16a34a" : pct >= 50 ? "#eab308" : "#dc2626";

    return `
    <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${fill}" rx="4" />
    <text x="${x + barWidth / 2}" y="${y - 10}" text-anchor="middle" class="axis-text" fill="#111827" font-weight="600">${correctCounts[index]}/${totals[index]} (${pct.toFixed(0)}%)</text>
    <text x="${x + barWidth / 2}" y="${height - margin.bottom + 34}" text-anchor="middle" class="axis-text">${labels[index]}</text>`;
  });

  const title = `${modelName} addition accuracy by magnitude`;
  const subtitle = `${totals.reduce((a, b) => a + b, 0)} test cases`;
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
  <text x="${width / 2}" y="${height - 24}" text-anchor="middle" class="axis-label">magnitude</text>
  <text transform="translate(26 ${height / 2}) rotate(-90)" text-anchor="middle" class="axis-label">accuracy</text>
</svg>`;

  await mkdir(modelDirectory, { recursive: true });
  await sharp(Buffer.from(svg)).png().toFile(outputPath);

  return outputPath;
};

const modelName = process.argv[2];
if (!modelName) {
  console.error("Usage: bun run scripts/test-addy.ts <model>");
  process.exit(1);
}

const samplesPerMagnitude = 20;
const allResults: MagnitudeResult[] = [];

for (const mag of magnitudes) {
  const pairs = generatePairs(mag.maxDigit, samplesPerMagnitude);
  const results: TestResult[] = [];

  console.log(`\n--- ${mag.label} ---`);

  for (const [a, b] of pairs) {
    const result = runTest(a, b, modelName);
    const mark = result.correct ? "OK" : "FAIL";
    console.log(
      `  ${result.input} expected ${result.expected}, got "${result.output}" [${mark}]`,
    );
    results.push(result);
  }

  const correct = results.filter((r) => r.correct).length;
  console.log(`  Score: ${correct}/${results.length}`);

  allResults.push({ label: mag.label, maxDigit: mag.maxDigit, results });
}

const modelDirectory = join(import.meta.dirname, "..");
const jsonPath = join(modelDirectory, "test_results.json");
await writeFile(jsonPath, JSON.stringify(allResults, null, 2));
console.log(`\nResults saved to: ${jsonPath}`);

const chartPath = await writeResultsChart(modelName, allResults);
console.log(`Chart saved to: ${chartPath}`);
