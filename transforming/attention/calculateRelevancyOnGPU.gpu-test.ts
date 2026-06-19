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
import { calculateRelevancyOnGPU } from "./calculateRelevancyOnGPU.ts";
import { MAX_CONTEXT } from "../../running/llm-shared.ts";

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
    createMatrix(inputQ.vectors, MAX_CONTEXT * headsCount),
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

    expectMatrixCloseTo(
      actual,
      restripeHeadBlocks(
        attentionRelevancyOutput,
        headsCount,
        inputQ.vectors,
        MAX_CONTEXT,
      ),
      4,
    );
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

    expectMatrixCloseTo(
      actual,
      restripeHeadBlocks(
        attentionRelevancyOutput,
        headsCount,
        inputQ.vectors,
        MAX_CONTEXT,
      ),
      4,
    );
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

    expectMatrixCloseTo(
      actual,
      restripeHeadBlocks(
        attentionRelevancyOutput,
        headsCount,
        inputQ.vectors,
        MAX_CONTEXT,
      ),
      4,
    );
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

    expectMatrixCloseTo(
      actual,
      restripeHeadBlocks(
        attentionRelevancyOutput,
        headsCount,
        inputQ.vectors,
        MAX_CONTEXT,
      ),
      4,
    );
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

describe("calculateRelevancyOnGPU — multi-batch", () => {
  it("scores each batch's relevancy in its own row slot without leaking", async () => {
    const dimensions = 8;
    const headsCount = 2;
    const headDimensionsCount = dimensions / headsCount;

    const qBatches = [randomMatrix(3, dimensions), randomMatrix(6, dimensions)];
    const kBatches = [randomMatrix(3, dimensions), randomMatrix(6, dimensions)];

    const packedQ = packBatches(qBatches);
    const packedK = packBatches(kBatches);

    const actual = await runRelevancyGpu(
      packedQ,
      packedK,
      headsCount,
      headDimensionsCount,
    );

    const expected = qBatches.map((inputQ, batchIndex) => {
      const inputK = kBatches[batchIndex]!;
      const inputV = randomMatrix(inputQ.vectors, dimensions);
      const { attentionRelevancyOutput } = runSelfAttentionHead(
        inputQ,
        inputK,
        inputV,
        headsCount,
        headDimensionsCount,
      );
      return restripeHeadBlocks(
        attentionRelevancyOutput,
        headsCount,
        inputQ.vectors,
        MAX_CONTEXT,
      );
    });

    expectBatchRowsCloseTo(actual, expected);
  });
});
