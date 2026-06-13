import { describe, it } from "bun:test";
import { d } from "typegpu";
import { gpuContext } from "../../shared/gpu-context.ts";
import {
  createMatrixBuffer,
  extractMatrixBuffer,
} from "../../shared/matrices-gpu.ts";
import { createMatrix, type Matrix } from "../../shared/matrices.ts";
import {
  matrixFrom,
  expectMatrixCloseTo,
} from "../../testing/testing-utils.ts";
import { runSelfAttentionHead } from "./attention.ts";
import { calculateRelevancyOnGPU } from "./calculateRelevancyOnGPU.ts";

const runRelevancyGpu = async (
  inputQ: Matrix,
  inputK: Matrix,
  headsCount: number,
  headDimensionsCount: number,
): Promise<Matrix> => {
  const headDim = gpuContext
    .createBuffer(d.u32, headDimensionsCount)
    .$usage("uniform");
  const inputQBuffer = createMatrixBuffer(inputQ);
  const inputKBuffer = createMatrixBuffer(inputK);
  const out = createMatrixBuffer(
    createMatrix(inputQ.vectors, inputQ.vectors * headsCount),
  );

  const encoder = gpuContext.device.createCommandEncoder();
  calculateRelevancyOnGPU(
    headsCount,
    headDim,
    inputKBuffer,
    inputQBuffer,
    out,
    encoder,
  );
  gpuContext.device.queue.submit([encoder.finish()]);
  await gpuContext.device.queue.onSubmittedWorkDone();

  return extractMatrixBuffer(out);
};

describe("calculateRelevancyOnGPU", () => {
  it("scores a single head's query/key dot products", async () => {
    const inputQ = matrixFrom([
      [1, 0],
      [0, 1],
    ]);
    const inputK = matrixFrom([
      [1, 0],
      [0, 1],
    ]);
    const inputV = matrixFrom([
      [1, 10],
      [2, 20],
    ]);
    const headsCount = 1;
    const headDimensionsCount = 2;

    const { attentionRelevancyOutput } = runSelfAttentionHead(
      inputQ,
      inputK,
      inputV,
      headsCount,
      headDimensionsCount,
    );

    const actual = await runRelevancyGpu(
      inputQ,
      inputK,
      headsCount,
      headDimensionsCount,
    );

    expectMatrixCloseTo(actual, attentionRelevancyOutput, 4);
  });

  it("writes each head's scores into its own column block", async () => {
    const inputQ = matrixFrom([
      [0, 0],
      [0, 1],
    ]);
    const inputK = matrixFrom([
      [0, 0],
      [0, Math.log(2)],
    ]);
    const inputV = matrixFrom([
      [1, 10],
      [2, 20],
    ]);
    const headsCount = 2;
    const headDimensionsCount = 1;

    const { attentionRelevancyOutput } = runSelfAttentionHead(
      inputQ,
      inputK,
      inputV,
      headsCount,
      headDimensionsCount,
    );

    const actual = await runRelevancyGpu(
      inputQ,
      inputK,
      headsCount,
      headDimensionsCount,
    );

    expectMatrixCloseTo(actual, attentionRelevancyOutput, 4);
  });

  it("leaves future (masked) positions untouched", async () => {
    const inputQ = matrixFrom([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const inputK = matrixFrom([
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const inputV = matrixFrom([
      [1, 10],
      [2, 20],
      [3, 30],
    ]);
    const headsCount = 1;
    const headDimensionsCount = 2;

    const { attentionRelevancyOutput } = runSelfAttentionHead(
      inputQ,
      inputK,
      inputV,
      headsCount,
      headDimensionsCount,
    );

    const actual = await runRelevancyGpu(
      inputQ,
      inputK,
      headsCount,
      headDimensionsCount,
    );

    expectMatrixCloseTo(actual, attentionRelevancyOutput, 4);
  });

  it("matches the CPU reference for a larger random multi-head case", async () => {
    const vectors = 7;
    const dimensions = 8;
    const headsCount = 2;
    const headDimensionsCount = dimensions / headsCount;
    const rand = () => Math.random() * 2 - 1;
    const randomMatrix = (rows: number, cols: number): Matrix => ({
      vectors: rows,
      dimensions: cols,
      values: new Float32Array(Array.from({ length: rows * cols }, rand)),
    });

    const inputQ = randomMatrix(vectors, dimensions);
    const inputK = randomMatrix(vectors, dimensions);
    const inputV = randomMatrix(vectors, dimensions);

    const { attentionRelevancyOutput } = runSelfAttentionHead(
      inputQ,
      inputK,
      inputV,
      headsCount,
      headDimensionsCount,
    );

    const actual = await runRelevancyGpu(
      inputQ,
      inputK,
      headsCount,
      headDimensionsCount,
    );

    expectMatrixCloseTo(actual, attentionRelevancyOutput, 4);
  });
});
