import { createMatrix, createVector } from "../shared/matrices.ts";
import {
  createMatrixBuffer,
  extractMatrixBuffer,
  type MatrixBuffer,
} from "../shared/matrices-gpu.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import { getMultilayerPerceptronActivations } from "../transforming/mlp.ts";
import { getMultilayerPerceptronActivationsOnGPU } from "../transforming/mlp-gpu.ts";
import type { MultilayerPerceptronWeights } from "../model/model-types.ts";
import type { MultilayerPerceptronGPUBuffers } from "../model/model-gpu-helpers.ts";

const WARMUP_ITERS = 3;
const MEASURE_ITERS = 10;
const MLP_MULTIPLE = 4;

const rand = () => Math.random() * 2 - 1;

const SIZES: Array<{
  label: string;
  contextLength: number;
  dimensions: number;
}> = [
  { label: "small  C=5  D=64", contextLength: 5, dimensions: 64 },
  { label: "med    C=5  D=128", contextLength: 5, dimensions: 128 },
  { label: "large  C=5  D=256", contextLength: 5, dimensions: 256 },
  { label: "large  C=10 D=256", contextLength: 10, dimensions: 256 },
];

type Stats = {
  mean: number;
  median: number;
  min: number;
  max: number;
  stddev: number;
};

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

const fmtMs = (ms: number) => `${ms.toFixed(3)}ms`;

const createTestWeights = (
  dimensions: number,
): MultilayerPerceptronWeights => ({
  wUp: {
    weightsMatrix: createMatrix(dimensions, dimensions * MLP_MULTIPLE, rand),
    biasVector: createVector(dimensions * MLP_MULTIPLE, rand),
  },
  wDown: {
    weightsMatrix: createMatrix(dimensions * MLP_MULTIPLE, dimensions, rand),
    biasVector: createVector(dimensions, rand),
  },
});

const weightsToGPU = (
  weights: MultilayerPerceptronWeights,
): MultilayerPerceptronGPUBuffers => ({
  wUp: {
    weightsMatrix: createMatrixBuffer(weights.wUp.weightsMatrix),
    biasVector: createMatrixBuffer([weights.wUp.biasVector]),
  },
  wDown: {
    weightsMatrix: createMatrixBuffer(weights.wDown.weightsMatrix),
    biasVector: createMatrixBuffer([weights.wDown.biasVector]),
  },
});

const benchmarkCPU = (
  encoding: number[][],
  perceptron: MultilayerPerceptronWeights,
): Stats => {
  for (let i = 0; i < WARMUP_ITERS; i++) {
    getMultilayerPerceptronActivations(encoding, perceptron, MLP_MULTIPLE);
  }

  const samples: number[] = [];
  for (let i = 0; i < MEASURE_ITERS; i++) {
    const start = performance.now();
    getMultilayerPerceptronActivations(encoding, perceptron, MLP_MULTIPLE);
    samples.push(performance.now() - start);
  }
  return computeStats(samples);
};

const benchmarkGPU = async (
  encodingBuf: MatrixBuffer,
  uppedBuf: MatrixBuffer,
  outBuf: MatrixBuffer,
  perceptronBuf: MultilayerPerceptronGPUBuffers,
): Promise<Stats> => {
  for (let i = 0; i < WARMUP_ITERS; i++) {
    getMultilayerPerceptronActivationsOnGPU(
      encodingBuf,
      uppedBuf,
      outBuf,
      perceptronBuf,
    );
    await gpuContext.device.queue.onSubmittedWorkDone();
  }

  const samples: number[] = [];
  for (let i = 0; i < MEASURE_ITERS; i++) {
    const start = performance.now();
    getMultilayerPerceptronActivationsOnGPU(
      encodingBuf,
      uppedBuf,
      outBuf,
      perceptronBuf,
    );
    await gpuContext.device.queue.onSubmittedWorkDone();
    samples.push(performance.now() - start);
  }
  return computeStats(samples);
};

const matricesMatch = (
  m1: number[][],
  m2: number[][],
  tolerance = 1e-3,
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
      if (diff > tolerance) {
        return {
          ok: false,
          reason: `cell [${i},${j}]: ${r1[j]} vs ${r2[j]} (diff ${diff})`,
        };
      }
    }
  }
  return { ok: true };
};

const printRow = (
  label: string,
  cpu: Stats,
  gpu: Stats,
  speedup: number,
) => {
  const speedupStr =
    speedup >= 1
      ? `${speedup.toFixed(2)}x faster`
      : `${(1 / speedup).toFixed(2)}x slower`;
  console.log(`\n  ${label}`);
  console.log(
    `    CPU   mean=${fmtMs(cpu.mean)} median=${fmtMs(cpu.median)} min=${fmtMs(cpu.min)} stddev=${fmtMs(cpu.stddev)}`,
  );
  console.log(
    `    GPU   mean=${fmtMs(gpu.mean)} median=${fmtMs(gpu.median)} min=${fmtMs(gpu.min)} stddev=${fmtMs(gpu.stddev)}`,
  );
  console.log(`    -> GPU is ${speedupStr} (median)`);
};

const main = async () => {
  console.log("MLP CPU vs GPU benchmark");
  console.log(`  mlpMultiple=${MLP_MULTIPLE}`);
  console.log(`  warmup=${WARMUP_ITERS}, measure=${MEASURE_ITERS} iters\n`);

  let anyMismatch = false;

  for (const { label, contextLength, dimensions } of SIZES) {
    const encoding = createMatrix(contextLength, dimensions, rand);
    const weights = createTestWeights(dimensions);

    // Pre-allocate all GPU buffers (excluded from timing)
    const encodingBuf = createMatrixBuffer(encoding);
    const perceptronBuf = weightsToGPU(weights);
    const uppedBuf = createMatrixBuffer(
      createMatrix(contextLength, dimensions * MLP_MULTIPLE),
    );
    const outBuf = createMatrixBuffer(
      createMatrix(contextLength, dimensions),
    );

    // Correctness check: compare CPU output with GPU output
    const cpuResult = getMultilayerPerceptronActivations(
      encoding,
      weights,
      MLP_MULTIPLE,
    );
    getMultilayerPerceptronActivationsOnGPU(
      encodingBuf,
      uppedBuf,
      outBuf,
      perceptronBuf,
    );
    const gpuOutput = await extractMatrixBuffer(outBuf);

    const match = matricesMatch(cpuResult.downingOutput, gpuOutput);
    if (!match.ok) {
      anyMismatch = true;
      console.log(`  [${label}] MISMATCH: ${match.reason}`);
    }

    // Re-create encoding buffer (GPU output buffers get overwritten each call)
    const freshEncodingBuf = createMatrixBuffer(encoding);
    const freshUppedBuf = createMatrixBuffer(
      createMatrix(contextLength, dimensions * MLP_MULTIPLE),
    );
    const freshOutBuf = createMatrixBuffer(
      createMatrix(contextLength, dimensions),
    );

    const cpuStats = benchmarkCPU(encoding, weights);
    const gpuStats = await benchmarkGPU(
      freshEncodingBuf,
      freshUppedBuf,
      freshOutBuf,
      perceptronBuf,
    );
    const speedup = cpuStats.median / gpuStats.median;
    printRow(label, cpuStats, gpuStats, speedup);
  }

  console.log("");
  if (anyMismatch) {
    console.log(
      "  WARNING: at least one size produced mismatched outputs — check your GPU implementation",
    );
  }
};

if (import.meta.main) {
  await main();
}
