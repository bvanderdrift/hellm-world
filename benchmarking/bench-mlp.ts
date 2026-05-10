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
import {
  benchmark,
  matricesMatch,
  printRow,
  WARMUP_ITERS,
  MEASURE_ITERS,
} from "./bench-harness.ts";

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

const main = async () => {
  console.log("MLP CPU vs GPU benchmark");
  console.log(`  mlpMultiple=${MLP_MULTIPLE}`);
  console.log(`  warmup=${WARMUP_ITERS}, measure=${MEASURE_ITERS} iters\n`);

  let anyMismatch = false;

  for (const { label, contextLength, dimensions } of SIZES) {
    const encoding = createMatrix(contextLength, dimensions, rand);
    const weights = createTestWeights(dimensions);

    const encodingBuf = createMatrixBuffer(encoding);
    const perceptronBuf = weightsToGPU(weights);
    const uppedBuf = createMatrixBuffer(
      createMatrix(contextLength, dimensions * MLP_MULTIPLE),
    );
    const outBuf = createMatrixBuffer(
      createMatrix(contextLength, dimensions),
    );

    // Correctness check
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

    const match = matricesMatch(cpuResult.downingOutput, gpuOutput, 1e-3);
    if (!match.ok) {
      anyMismatch = true;
      console.log(`  [${label}] MISMATCH: ${match.reason}`);
    }

    // Fresh buffers for benchmark runs
    const freshEncodingBuf = createMatrixBuffer(encoding);
    const freshUppedBuf = createMatrixBuffer(
      createMatrix(contextLength, dimensions * MLP_MULTIPLE),
    );
    const freshOutBuf = createMatrixBuffer(
      createMatrix(contextLength, dimensions),
    );

    const cpuStats = await benchmark(() => {
      getMultilayerPerceptronActivations(encoding, weights, MLP_MULTIPLE);
    });
    const gpuStats = await benchmark(async () => {
      getMultilayerPerceptronActivationsOnGPU(
        freshEncodingBuf,
        freshUppedBuf,
        freshOutBuf,
        perceptronBuf,
      );
      await gpuContext.device.queue.onSubmittedWorkDone();
    });

    const speedup = cpuStats.median / gpuStats.median;
    printRow(label, "CPU", cpuStats, "GPU", gpuStats, speedup);
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
