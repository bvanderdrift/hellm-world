import {
  createMatrixBufferAndCopy,
  extractMatrixBuffer,
} from "../shared/matrices-gpu.ts";
import { createMatrix, type Matrix } from "../shared/matrices.ts";
import { normalize } from "../shared/normalize.ts";
import { normalizeOnGpu } from "../shared/normalize-gpu.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import {
  type Stats,
  benchmark,
  matricesMatch,
  printRow,
  WARMUP_ITERS,
  MEASURE_ITERS,
} from "./bench-harness.ts";

const SIZES: Array<{ label: string; vectors: number; dimensions: number }> = [
  { label: "tiny    16x64", vectors: 16, dimensions: 64 },
  { label: "small   64x256", vectors: 64, dimensions: 256 },
  { label: "med    256x512", vectors: 256, dimensions: 512 },
  { label: "large  512x1024", vectors: 512, dimensions: 1024 },
  { label: "tall  2048x128", vectors: 2048, dimensions: 128 },
  { label: "wide    16x2048", vectors: 16, dimensions: 2048 },
];

const rand = () => Math.random() * 2 - 1;

// Allocate + upload the buffer ONCE, outside the timed loop, so we measure the
// kernel dispatch only and not the (fixed-size, ~64MB) buffer alloc/upload.
// normalizeOnGpu mutates in place, so re-running drifts the values — but the
// kernel does identical work each call, so the timing stays representative.
const benchmarkGPU = async (
  input: Matrix,
): Promise<{ stats: Stats; lastResult: Matrix }> => {
  const buffer = createMatrixBufferAndCopy(input);

  const stats = await benchmark(async () => {
    normalizeOnGpu(buffer);
    await gpuContext.device.queue.onSubmittedWorkDone();
  });

  // Re-upload clean input so the returned result is a true single-pass
  // normalization for the correctness check.
  const verifyBuffer = createMatrixBufferAndCopy(input);
  normalizeOnGpu(verifyBuffer);
  await gpuContext.device.queue.onSubmittedWorkDone();
  const lastResult = await extractMatrixBuffer(verifyBuffer);

  return { stats, lastResult };
};

const main = async () => {
  console.log("normalize CPU vs GPU benchmark");
  console.log(`  baseline = normalize, candidate = normalizeOnGpu`);
  console.log(`  warmup=${WARMUP_ITERS}, measure=${MEASURE_ITERS} iters\n`);

  let anyMismatch = false;

  for (const { label, vectors, dimensions } of SIZES) {
    const input = createMatrix(vectors, dimensions, rand);

    const cpuResult = normalize(input);
    const { lastResult: gpuResult } = await benchmarkGPU(input);

    // GPU buffer is padded; compare only the used region by trimming.
    const gpuUsed: Matrix = {
      vectors,
      dimensions,
      values: gpuResult.values.slice(0, vectors * dimensions),
    };
    const match = matricesMatch(cpuResult, gpuUsed, 1e-4);
    if (!match.ok) {
      anyMismatch = true;
      console.log(`  [${label}] MISMATCH (GPU): ${match.reason}`);
      continue;
    }

    const baseline = await benchmark(() => {
      normalize(input);
    });
    const { stats: candidateGPU } = await benchmarkGPU(input);
    const speedupGPU = baseline.median / candidateGPU.median;
    printRow(label, "CPU", baseline, "GPU", candidateGPU, speedupGPU);
  }

  console.log("");
  if (anyMismatch) {
    console.log("  WARNING: at least one size produced mismatched outputs");
    process.exit(1);
  }
};

if (import.meta.main) {
  await main();
}
