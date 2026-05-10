import {
  createMatrixBuffer,
  extractMatrixBuffer,
  multiplyMatricesOnGPU,
} from "../shared/matrices-gpu.ts";
import { createMatrix, multiplyMatrices } from "../shared/matrices.ts";
import { gpuContext } from "../shared/gpu-context.ts";

type MatMul = (
  a: number[][],
  b: number[][],
) => number[][] | Promise<number[][]>;

type Stats = {
  mean: number;
  median: number;
  min: number;
  max: number;
  stddev: number;
};

const SIZES: Array<{ label: string; m: number; k: number; n: number }> = [
  { label: "tiny    64x64 * 64x64", m: 64, k: 64, n: 64 },
  { label: "small  128x128 * 128x128", m: 128, k: 128, n: 128 },
  { label: "med    256x256 * 256x256", m: 256, k: 256, n: 256 },
  { label: "large  512x512 * 512x512", m: 512, k: 512, n: 512 },
  { label: "tall   512x128 * 128x512", m: 512, k: 128, n: 512 },
];

const WARMUP_ITERS = 3;
const MEASURE_ITERS = 10;

const rand = () => Math.random() * 2 - 1;

const computeStats = (samples: number[]): Stats => {
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((s, v) => s + v, 0) / samples.length;
  const median = sorted[Math.floor(sorted.length / 2)]!;
  const min = sorted[0]!;
  const max = sorted[sorted.length - 1]!;
  const variance =
    samples.reduce((s, v) => s + (v - mean) ** 2, 0) / samples.length;
  const stddev = Math.sqrt(variance);
  return { mean, median, min, max, stddev };
};

const benchmark = async (
  fn: MatMul,
  a: number[][],
  b: number[][],
): Promise<Stats> => {
  for (let i = 0; i < WARMUP_ITERS; i++) await fn(a, b);

  const samples: number[] = [];
  for (let i = 0; i < MEASURE_ITERS; i++) {
    const start = performance.now();
    await fn(a, b);
    samples.push(performance.now() - start);
  }
  return computeStats(samples);
};

const benchmarkGPU = async (
  a: number[][],
  b: number[][],
): Promise<{ stats: Stats; lastResult: number[][] }> => {
  const m1 = createMatrixBuffer(a);
  const m2 = createMatrixBuffer(b);
  const mOut = createMatrixBuffer(
    createMatrix(a.length, b[0]!.length),
  );

  for (let i = 0; i < WARMUP_ITERS; i++) {
    multiplyMatricesOnGPU(m1, m2, mOut);
    await gpuContext.device.queue.onSubmittedWorkDone();
  }

  const samples: number[] = [];
  for (let i = 0; i < MEASURE_ITERS; i++) {
    const start = performance.now();
    multiplyMatricesOnGPU(m1, m2, mOut);
    await gpuContext.device.queue.onSubmittedWorkDone();
    samples.push(performance.now() - start);
  }
  const lastResult = await extractMatrixBuffer(mOut);
  return { stats: computeStats(samples), lastResult };
};

const matricesMatch = (
  m1: number[][],
  m2: number[][],
  tolerance = 1e-4,
): { ok: true } | { ok: false; reason: string } => {
  if (m1.length !== m2.length)
    return { ok: false, reason: `row count: ${m1.length} vs ${m2.length}` };
  for (let i = 0; i < m1.length; i++) {
    const r1 = m1[i]!;
    const r2 = m2[i]!;
    if (r1.length !== r2.length)
      return {
        ok: false,
        reason: `row ${i} length: ${r1.length} vs ${r2.length}`,
      };
    for (let j = 0; j < r1.length; j++) {
      const diff = Math.abs(r1[j]! - r2[j]!);

      if (Math.abs(r1[j]! - r2[j]!) > tolerance) {
        return {
          ok: false,
          reason: `cell [${i},${j}]: ${r1[j]} vs ${r2[j]} - diff ${diff}`,
        };
      }
    }
  }
  return { ok: true };
};

type Measure = (
  n: number,
) => Promise<{ baselineMedian: number; candidateMedian: number }>;

export const findCrossover = async (
  lo: number,
  hi: number,
  measure: Measure,
): Promise<{ crossoverN: number; parameterCount: number }> => {
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    const { baselineMedian, candidateMedian } = await measure(mid);
    if (candidateMedian < baselineMedian) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return { crossoverN: lo, parameterCount: 2 * lo * lo };
};

const fmtMs = (ms: number) => `${ms.toFixed(3)}ms`;

const printRow = (
  label: string,
  baseline: Stats,
  candidate: Stats,
  speedup: number,
) => {
  const speedupStr =
    speedup >= 1
      ? `${speedup.toFixed(2)}x faster`
      : `${(1 / speedup).toFixed(2)}x slower`;
  console.log(`\n  ${label}`);
  console.log(
    `    baseline   mean=${fmtMs(baseline.mean)} median=${fmtMs(baseline.median)} min=${fmtMs(baseline.min)} stddev=${fmtMs(baseline.stddev)}`,
  );
  console.log(
    `    candidate  mean=${fmtMs(candidate.mean)} median=${fmtMs(candidate.median)} min=${fmtMs(candidate.min)} stddev=${fmtMs(candidate.stddev)}`,
  );
  console.log(`    -> candidate is ${speedupStr} (median)`);
};

const main = async () => {
  console.log("matmul A/B benchmark");
  console.log(
    `  baseline = multiplyMatrices, candidate = multiplyMatricesOnGPU`,
  );
  console.log(`  warmup=${WARMUP_ITERS}, measure=${MEASURE_ITERS} iters\n`);

  let anyMismatch = false;

  for (const { label, m, k, n } of SIZES) {
    const a = createMatrix(m, k, rand);
    const b = createMatrix(k, n, rand);

    const r1 = await multiplyMatrices(a, b);
    const { lastResult: r2 } = await benchmarkGPU(a, b);
    const match = matricesMatch(r1, r2);
    if (!match.ok) {
      anyMismatch = true;
      console.log(`  [${label}] MISMATCH: ${match.reason}`);
      continue;
    }

    const baseline = await benchmark(multiplyMatrices, a, b);
    const { stats: candidate } = await benchmarkGPU(a, b);
    const speedup = baseline.median / candidate.median;
    printRow(label, baseline, candidate, speedup);
  }

  console.log("");
  if (anyMismatch) {
    console.log("  WARNING: at least one size produced mismatched outputs");
    process.exit(1);
  }

  console.log("\n--- crossover search (binary search on square dimension) ---");
  const measureReal: Measure = async (n) => {
    const a = createMatrix(n, n, rand);
    const b = createMatrix(n, n, rand);
    const baselineStats = await benchmark(multiplyMatrices, a, b);
    const { stats: candidateStats } = await benchmarkGPU(a, b);
    console.log(
      `  probe n=${n} (params=${2 * n * n}): baseline=${fmtMs(baselineStats.median)} candidate=${fmtMs(candidateStats.median)}`,
    );
    return {
      baselineMedian: baselineStats.median,
      candidateMedian: candidateStats.median,
    };
  };

  const { crossoverN, parameterCount } = await findCrossover(
    32,
    512,
    measureReal,
  );
  console.log(
    `\n  crossover at n=${crossoverN} (square ${crossoverN}x${crossoverN}), parameter count=${parameterCount}`,
  );

  console.log(
    "\n--- shape comparison (same ~param count, different shapes) ---",
  );
  const targetParams = parameterCount;
  const shapes: Array<{ label: string; m: number; k: number; n: number }> = [
    // m×k * k×n → params = m*k + k*n
    // For equal params with symmetric shapes: solve m*k + k*n = targetParams
    { label: "1×P * P×1 (dot product)", m: 1, k: targetParams / 2, n: 1 },
    { label: "4×K * K×4", m: 4, k: Math.round(targetParams / 8), n: 4 },
    { label: "16×K * K×16", m: 16, k: Math.round(targetParams / 32), n: 16 },
    { label: "64×K * K×64", m: 64, k: Math.round(targetParams / 128), n: 64 },
    {
      label: `${crossoverN}×${crossoverN} (square)`,
      m: crossoverN,
      k: crossoverN,
      n: crossoverN,
    },
    {
      label: "wide 1×K * K×256",
      m: 1,
      k: Math.round(targetParams / 257),
      n: 256,
    },
  ];

  console.log(`  target param count ≈ ${targetParams}\n`);
  for (const { label, m, k, n } of shapes) {
    const actualParams = m * k + k * n;
    const outputCells = m * n;
    const a = createMatrix(m, k, rand);
    const b = createMatrix(k, n, rand);
    try {
      const baselineStats = await benchmark(multiplyMatrices, a, b);
      const { stats: candidateStats } = await benchmarkGPU(a, b);
      const speedup = baselineStats.median / candidateStats.median;
      const winner = speedup >= 1 ? "GPU" : "CPU";
      console.log(
        `  ${label}  (${m}×${k} * ${k}×${n}, params=${actualParams}, output=${outputCells} cells)`,
      );
      console.log(
        `    baseline=${fmtMs(baselineStats.median)} candidate=${fmtMs(candidateStats.median)} → ${winner} wins (${speedup >= 1 ? speedup.toFixed(2) : (1 / speedup).toFixed(2)}x)`,
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.log(
        `  ${label}  (${m}×${k} * ${k}×${n}, params=${actualParams}, output=${outputCells} cells)`,
      );
      console.log(`    SKIPPED — GPU error: ${msg.split("\n")[0]}`);
    }
  }
};

if (import.meta.main) {
  await main();
}
