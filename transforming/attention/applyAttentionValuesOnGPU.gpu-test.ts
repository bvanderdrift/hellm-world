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
  restripeHeadBlocks,
} from "../../testing/testing-utils.ts";
import { runSelfAttentionHead } from "./attention.ts";
import { applyAttentionValuesOnGPU } from "./applyAttentionValuesOnGPU.ts";
import { MAX_CONTEXT } from "../../running/llm-shared.ts";

const runGpu = async (
  inputV: Matrix,
  matchingKeyProducts: Matrix,
  headDimensionsCount: number,
): Promise<Matrix> => {
  const headDim = gpuContext
    .createBuffer(d.u32, headDimensionsCount)
    .$usage("uniform");
  const headsCount = inputV.dimensions / headDimensionsCount;
  const inputVBuffer = createMatrixBuffer(inputV);
  const matchingKeyProductsBuffer = createMatrixBuffer(
    restripeHeadBlocks(
      matchingKeyProducts,
      headsCount,
      inputV.vectors,
      MAX_CONTEXT,
    ),
  );
  const out = createMatrixBuffer(
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
