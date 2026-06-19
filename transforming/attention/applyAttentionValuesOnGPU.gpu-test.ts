import { describe, it, expect } from "bun:test";
import { d } from "typegpu";
import { gpuContext } from "../../shared/gpu-context.ts";
import {
  createMatrixBuffer,
  extractMatrixBuffer,
} from "../../shared/matrices/matrices-gpu.ts";
import {
  createMatrix,
  getFlatIndex,
  type Matrix,
} from "../../shared/matrices/matrices.ts";
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

const rand = () => Math.random() * 2 - 1;

const randomMatrix = (rows: number, cols: number): Matrix => ({
  vectors: rows,
  dimensions: cols,
  values: new Float32Array(Array.from({ length: rows * cols }, rand)),
});

/**
 * Lay each batch's rows out in its own MAX_CONTEXT-sized row slot,
 * zero-padding the unused tail. Batch b lives in rows [b*MAX_CONTEXT, ...).
 */
const packBatches = (batches: Matrix[]): Matrix => {
  const dimensions = batches[0]!.dimensions;
  const packed = createMatrix(batches.length * MAX_CONTEXT, dimensions);

  batches.forEach((batch, batchIndex) => {
    const rowOffset = batchIndex * MAX_CONTEXT;
    for (let r = 0; r < batch.vectors; r++) {
      for (let c = 0; c < dimensions; c++) {
        packed.values[getFlatIndex(rowOffset + r, c, dimensions)] =
          batch.values[getFlatIndex(r, c, dimensions)]!;
      }
    }
  });

  return packed;
};

/** Compare only the real (non-padding) rows of each batch slot. */
const expectBatchRowsCloseTo = (
  packedActual: Matrix,
  perBatchExpected: Matrix[],
  precision = 4,
) => {
  perBatchExpected.forEach((expected, batchIndex) => {
    const rowOffset = batchIndex * MAX_CONTEXT;
    for (let r = 0; r < expected.vectors; r++) {
      for (let c = 0; c < expected.dimensions; c++) {
        const actualValue =
          packedActual.values[
            getFlatIndex(rowOffset + r, c, packedActual.dimensions)
          ];
        const expectedValue =
          expected.values[getFlatIndex(r, c, expected.dimensions)]!;
        expect(actualValue).toBeCloseTo(expectedValue, precision);
      }
    }
  });
};

const runApplyPacked = async (
  packedV: Matrix,
  packedMatchingKeyProducts: Matrix,
  headDimensionsCount: number,
): Promise<Matrix> => {
  const headDim = gpuContext
    .createBuffer(d.u32, headDimensionsCount)
    .$usage("uniform");
  const inputVBuffer = createMatrixBuffer(packedV);
  const matchingKeyProductsBuffer = createMatrixBuffer(
    packedMatchingKeyProducts,
  );
  const out = createMatrixBuffer(
    createMatrix(packedV.vectors, packedV.dimensions),
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

describe("applyAttentionValuesOnGPU — multi-batch", () => {
  it("mixes each batch's values within its own row slot without leaking", async () => {
    const dimensions = 8;
    const headsCount = 2;
    const headDimensionsCount = dimensions / headsCount;

    const lengths = [3, 6];
    const heads = lengths.map((length) => {
      const inputQ = randomMatrix(length, dimensions);
      const inputK = randomMatrix(length, dimensions);
      const inputV = randomMatrix(length, dimensions);
      return runSelfAttentionHead(
        inputQ,
        inputK,
        inputV,
        headsCount,
        headDimensionsCount,
      );
    });

    const packedV = packBatches(heads.map((head) => head.inputV));
    const packedMatchingKeyProducts = packBatches(
      heads.map((head) =>
        restripeHeadBlocks(
          head.softmaxOutput,
          headsCount,
          head.inputV.vectors,
          MAX_CONTEXT,
        ),
      ),
    );

    const actual = await runApplyPacked(
      packedV,
      packedMatchingKeyProducts,
      headDimensionsCount,
    );

    expectBatchRowsCloseTo(
      actual,
      heads.map((head) => head.output),
    );
  });
});
