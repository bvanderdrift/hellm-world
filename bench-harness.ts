import { createMatrix, type Matrix } from "./shared/matrices.ts";
import {
  createMatrixBufferAndCopy,
  extractMatrixBuffer,
  type MatrixBuffer,
} from "./shared/matrices-gpu.ts";
import { gpuContext } from "./shared/gpu-context.ts";

export const WARMUP_ITERS = 3;
export const MEASURE_ITERS = 10;

export const rand = () => Math.random() * 2 - 1;

export type BenchSize = {
  label: string;
  vectors: number;
  dimensions: number;
};

export const BENCH_SIZES: BenchSize[] = [
  { label: "tiny", vectors: 8, dimensions: 32 },
  { label: "small", vectors: 32, dimensions: 128 },
  { label: "med", vectors: 128, dimensions: 256 },
  { label: "large", vectors: 256, dimensions: 512 },
  { label: "xlarge", vectors: 512, dimensions: 512 },
];

export type Stats = {
  mean: number;
  median: number;
  min: number;
  max: number;
  stddev: number;
};

export const computeStats = (samples: number[]): Stats => {
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

export const fmtMs = (ms: number) => `${ms.toFixed(3)}ms`;

export const benchmark = async (
  fn: () => void | Promise<void>,
): Promise<Stats> => {
  for (let i = 0; i < WARMUP_ITERS; i++) await fn();

  const samples: number[] = [];
  for (let i = 0; i < MEASURE_ITERS; i++) {
    const start = performance.now();
    await fn();
    samples.push(performance.now() - start);
  }
  return computeStats(samples);
};

export const printRow = (
  label: string,
  baselineLabel: string,
  baseline: Stats,
  candidateLabel: string,
  candidate: Stats,
  speedup: number,
) => {
  const speedupStr =
    speedup >= 1
      ? `${speedup.toFixed(2)}x faster`
      : `${(1 / speedup).toFixed(2)}x slower`;
  console.log(`\n  ${label}`);
  console.log(
    `    ${baselineLabel.padEnd(10)} mean=${fmtMs(baseline.mean)} median=${fmtMs(baseline.median)} min=${fmtMs(baseline.min)} stddev=${fmtMs(baseline.stddev)}`,
  );
  console.log(
    `    ${candidateLabel.padEnd(10)} mean=${fmtMs(candidate.mean)} median=${fmtMs(candidate.median)} min=${fmtMs(candidate.min)} stddev=${fmtMs(candidate.stddev)}`,
  );
  console.log(`    -> ${candidateLabel} is ${speedupStr} (median)`);
};

export const matricesMatch = (
  m1: Matrix,
  m2: Matrix,
  tolerance: number,
): { ok: true } | { ok: false; reason: string } => {
  if (m1.vectors !== m2.vectors)
    return {
      ok: false,
      reason: `vector count: ${m1.vectors} vs ${m2.vectors}`,
    };
  if (m1.dimensions !== m2.dimensions)
    return {
      ok: false,
      reason: `dimension count: ${m1.dimensions} vs ${m2.dimensions}`,
    };
  for (let i = 0; i < m1.values.length; i++) {
    const diff = Math.abs(m1.values[i]! - m2.values[i]!);
    if (diff > tolerance) {
      const row = Math.floor(i / m1.dimensions);
      const col = i % m1.dimensions;
      return {
        ok: false,
        reason: `cell [${row},${col}]: ${m1.values[i]} vs ${m2.values[i]} (diff ${diff})`,
      };
    }
  }
  return { ok: true };
};

export type BenchInput = { matrix: Matrix; buffer: MatrixBuffer };

export type Comparison<Ctx = undefined> = {
  name: string;
  baselineLabel?: string;
  candidateLabel?: string;
  tolerance?: number;
  setup?: (size: BenchSize) => Ctx;
  cpu: (input: BenchInput, ctx: Ctx) => Matrix;
  gpu: (
    input: BenchInput,
    ctx: Ctx,
  ) => Matrix | MatrixBuffer | Promise<Matrix | MatrixBuffer>;
};

const isMatrix = (out: Matrix | MatrixBuffer): out is Matrix => "values" in out;

const toMatrix = async (
  out: Matrix | MatrixBuffer,
  shape: Matrix,
): Promise<Matrix> => {
  if (isMatrix(out)) return out;
  await gpuContext.device.queue.onSubmittedWorkDone();
  const full = await extractMatrixBuffer(out);
  return {
    vectors: shape.vectors,
    dimensions: shape.dimensions,
    values: full.values.slice(0, shape.vectors * shape.dimensions),
  };
};

export const compareAcrossSizes = async <Ctx>(
  cmp: Comparison<Ctx>,
): Promise<boolean> => {
  const baselineLabel = cmp.baselineLabel ?? "CPU";
  const candidateLabel = cmp.candidateLabel ?? "GPU";
  const tolerance = cmp.tolerance ?? 1e-3;

  console.log(cmp.name);
  console.log(`  warmup=${WARMUP_ITERS}, measure=${MEASURE_ITERS} iters`);

  let allMatched = true;

  for (const size of BENCH_SIZES) {
    const matrix = createMatrix(size.vectors, size.dimensions, rand);
    const input: BenchInput = {
      matrix,
      buffer: createMatrixBufferAndCopy(matrix),
    };
    const ctx = (cmp.setup ? cmp.setup(size) : undefined) as Ctx;
    const label = `${size.label.padEnd(6)} ${size.vectors}x${size.dimensions}`;

    const cpuResult = cmp.cpu(input, ctx);
    const gpuResult = await toMatrix(await cmp.gpu(input, ctx), cpuResult);

    const match = matricesMatch(cpuResult, gpuResult, tolerance);
    if (!match.ok) {
      allMatched = false;
      console.log(`\n  [${label}] MISMATCH: ${match.reason}`);
      continue;
    }

    const cpuStats = await benchmark(() => {
      cmp.cpu(input, ctx);
    });
    const gpuStats = await benchmark(async () => {
      await cmp.gpu(input, ctx);
      await gpuContext.device.queue.onSubmittedWorkDone();
    });

    printRow(
      label,
      baselineLabel,
      cpuStats,
      candidateLabel,
      gpuStats,
      cpuStats.median / gpuStats.median,
    );
  }

  console.log("");
  if (!allMatched) {
    console.log("  WARNING: at least one size produced mismatched outputs");
  }
  return allMatched;
};
