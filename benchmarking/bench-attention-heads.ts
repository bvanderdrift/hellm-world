import {
  createMatrix,
  multiplyMatrices,
} from "../shared/matrices.ts";
import { divideToWhole } from "../shared/math.ts";
import { runSelfAttentionHead } from "../transforming/attention.ts";
import { runSelfAttentionMechanismOnGPU } from "../transforming/attention-gpu.ts";
import type { AttentionWeights } from "../model/model-types.ts";
import {
  benchmark,
  matricesMatch,
  printRow,
  WARMUP_ITERS,
  MEASURE_ITERS,
} from "./bench-harness.ts";

const rand = () => Math.random() * 2 - 1;

const SIZES: Array<{
  label: string;
  contextLength: number;
  hiddenDimensions: number;
  headsCount: number;
}> = [
  { label: "small  C=5   D=64   H=4", contextLength: 5, hiddenDimensions: 64, headsCount: 4 },
  { label: "med    C=10  D=128  H=4", contextLength: 10, hiddenDimensions: 128, headsCount: 4 },
  { label: "med    C=10  D=128  H=8", contextLength: 10, hiddenDimensions: 128, headsCount: 8 },
  { label: "large  C=20  D=256  H=8", contextLength: 20, hiddenDimensions: 256, headsCount: 8 },
  { label: "large  C=20  D=256  H=16", contextLength: 20, hiddenDimensions: 256, headsCount: 16 },
];

const createTestWeights = (hiddenDimensions: number): AttentionWeights => ({
  Q: createMatrix(hiddenDimensions, hiddenDimensions, rand),
  K: createMatrix(hiddenDimensions, hiddenDimensions, rand),
  V: createMatrix(hiddenDimensions, hiddenDimensions, rand),
  out: createMatrix(hiddenDimensions, hiddenDimensions, rand),
});

const main = async () => {
  console.log("Attention mechanism CPU vs GPU benchmark");
  console.log(`  warmup=${WARMUP_ITERS}, measure=${MEASURE_ITERS} iters\n`);

  let anyMismatch = false;

  for (const { label, contextLength, hiddenDimensions, headsCount } of SIZES) {
    const headDimensionsCount = divideToWhole(hiddenDimensions, headsCount);
    const input = createMatrix(contextLength, hiddenDimensions, rand);
    const weights = createTestWeights(hiddenDimensions);

    // Correctness check: CPU full mechanism
    const inputQ = multiplyMatrices(input, weights.Q);
    const inputK = multiplyMatrices(input, weights.K);
    const inputV = multiplyMatrices(input, weights.V);

    const cpuHeadResult = runSelfAttentionHead(inputQ, inputK, inputV, headsCount, headDimensionsCount);
    const cpuOutput = multiplyMatrices(cpuHeadResult.output, weights.out);

    const gpuResult = runSelfAttentionMechanismOnGPU(input, headsCount, weights);

    const match = matricesMatch(cpuOutput, gpuResult.output, 1e-3);
    if (!match.ok) {
      anyMismatch = true;
      console.log(`  [${label}] MISMATCH: ${match.reason}`);
    }

    // Benchmark CPU: full attention mechanism
    const cpuStats = await benchmark(() => {
      const qProj = multiplyMatrices(input, weights.Q);
      const kProj = multiplyMatrices(input, weights.K);
      const vProj = multiplyMatrices(input, weights.V);

      const headResult = runSelfAttentionHead(qProj, kProj, vProj, headsCount, headDimensionsCount);
      multiplyMatrices(headResult.output, weights.out);
    });

    // Benchmark GPU: single call for full mechanism
    const gpuStats = await benchmark(() => {
      runSelfAttentionMechanismOnGPU(input, headsCount, weights);
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
