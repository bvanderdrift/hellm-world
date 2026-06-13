import { describe, it } from "bun:test";
import { d } from "typegpu";
import { gpuContext } from "../shared/gpu-context.ts";
import {
  createMatrixBufferAndCopy,
  extractMatrixBuffer,
} from "../shared/matrices-gpu.ts";
import { createMatrix, type Matrix } from "../shared/matrices.ts";
import { matrixFrom, expectMatrixCloseTo } from "../testing/testing-utils.ts";
import { runSelfAttentionHead } from "./attention.ts";
import {
  applyAttentionValuesOnGPU,
  calculateRelevancyOnGPU,
} from "./attention-gpu.ts";

const runGpu = async (
  inputV: Matrix,
  matchingKeyProducts: Matrix,
  headDimensionsCount: number,
): Promise<Matrix> => {
  const headDim = gpuContext
    .createBuffer(d.u32, headDimensionsCount)
    .$usage("uniform");
  const inputVBuffer = createMatrixBufferAndCopy(inputV);
  const matchingKeyProductsBuffer = createMatrixBufferAndCopy(
    matchingKeyProducts,
  );
  const out = createMatrixBufferAndCopy(
    createMatrix(inputV.vectors, inputV.dimensions),
  );

  applyAttentionValuesOnGPU(
    headDim,
    inputVBuffer,
    matchingKeyProductsBuffer,
    out,
  );
  await gpuContext.device.queue.onSubmittedWorkDone();

  return extractMatrixBuffer(out);
};

describe("applyAttentionValuesOnGPU", () => {
  it("mixes a single head's values by the softmax weights", async () => {
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
    const headDimensionsCount = 2;

    const { softmaxOutput, output } = runSelfAttentionHead(
      inputQ,
      inputK,
      inputV,
      1,
      headDimensionsCount,
    );

    const actual = await runGpu(inputV, softmaxOutput, headDimensionsCount);

    expectMatrixCloseTo(actual, output, 4);
  });

  it("reads each head's own softmax block by offsetting on vectors", async () => {
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
    const headDimensionsCount = 1;

    const { softmaxOutput, output } = runSelfAttentionHead(
      inputQ,
      inputK,
      inputV,
      2,
      headDimensionsCount,
    );

    const actual = await runGpu(inputV, softmaxOutput, headDimensionsCount);

    expectMatrixCloseTo(actual, output, 4);
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

    const { softmaxOutput, output } = runSelfAttentionHead(
      inputQ,
      inputK,
      inputV,
      headsCount,
      headDimensionsCount,
    );

    const actual = await runGpu(inputV, softmaxOutput, headDimensionsCount);

    expectMatrixCloseTo(actual, output, 4);
  });
});

const runRelevancyGpu = async (
  inputQ: Matrix,
  inputK: Matrix,
  headsCount: number,
  headDimensionsCount: number,
): Promise<Matrix> => {
  const headDim = gpuContext
    .createBuffer(d.u32, headDimensionsCount)
    .$usage("uniform");
  const inputQBuffer = createMatrixBufferAndCopy(inputQ);
  const inputKBuffer = createMatrixBufferAndCopy(inputK);
  const out = createMatrixBufferAndCopy(
    createMatrix(inputQ.vectors, inputQ.vectors * headsCount),
  );

  calculateRelevancyOnGPU(headsCount, headDim, inputKBuffer, inputQBuffer, out);
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
