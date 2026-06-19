import { describe, it, expect } from "bun:test";
import { d } from "typegpu";
import { gpuContext } from "../../shared/gpu-context.ts";
import {
  createMatrixBuffer,
  extractMatrixBuffer,
} from "../../shared/matrices-gpu.ts";
import {
  createMatrix,
  getFlatIndex,
  type Matrix,
} from "../../shared/matrices.ts";
import {
  matrixFrom,
  expectMatrixCloseTo,
} from "../../testing/testing-utils.ts";
import type { AttentionWeights } from "../../model/model-types.ts";
import type { AttentionGPUBuffers } from "../../model/model-gpu-helpers.ts";
import { runSelfAttentionMechanism } from "./attention.ts";
import { runSelfAttentionMechanismOnGPU } from "./attention-gpu.ts";
import { MAX_CONTEXT } from "../../running/llm-shared.ts";

const runGpu = async (
  input: Matrix,
  headsCount: number,
  attentionWeights: AttentionWeights,
): Promise<Matrix> => {
  const vectors = input.vectors;
  const hiddenDimensions = input.dimensions;
  const headDimensionsCount = hiddenDimensions / headsCount;

  const headDimensions = gpuContext
    .createBuffer(d.u32, headDimensionsCount)
    .$usage("uniform");

  const inputBuffer = createMatrixBuffer(input);
  const weights: AttentionGPUBuffers = {
    Q: createMatrixBuffer(attentionWeights.Q),
    K: createMatrixBuffer(attentionWeights.K),
    V: createMatrixBuffer(attentionWeights.V),
    out: createMatrixBuffer(attentionWeights.out),
  };

  const inputQ = createMatrixBuffer(createMatrix(vectors, hiddenDimensions));
  const inputK = createMatrixBuffer(createMatrix(vectors, hiddenDimensions));
  const inputV = createMatrixBuffer(createMatrix(vectors, hiddenDimensions));
  const attentionRelevancyOutput = createMatrixBuffer(
    createMatrix(vectors, MAX_CONTEXT * headsCount),
  );
  const matchingKeyProducts = createMatrixBuffer(
    createMatrix(vectors, MAX_CONTEXT * headsCount),
  );
  const output = createMatrixBuffer(createMatrix(vectors, hiddenDimensions));
  const attentionUpdate = createMatrixBuffer(
    createMatrix(vectors, hiddenDimensions),
  );

  runSelfAttentionMechanismOnGPU(
    inputBuffer,
    headsCount,
    headDimensions,
    weights,
    inputQ,
    inputK,
    inputV,
    attentionRelevancyOutput,
    matchingKeyProducts,
    output,
    attentionUpdate,
  );
  await gpuContext.device.queue.onSubmittedWorkDone();

  return extractMatrixBuffer(attentionUpdate);
};

describe("runSelfAttentionMechanismOnGPU", () => {
  it("matches the CPU mechanism for a single head with identity projections", async () => {
    const input = matrixFrom([
      [1, 10],
      [2, 20],
    ]);
    const identity = matrixFrom([
      [1, 0],
      [0, 1],
    ]);
    const attentionWeights: AttentionWeights = {
      Q: identity,
      K: identity,
      V: identity,
      out: identity,
    };
    const headsCount = 1;

    const expected = runSelfAttentionMechanism(
      input,
      headsCount,
      attentionWeights,
    );
    const actual = await runGpu(input, headsCount, attentionWeights);

    expectMatrixCloseTo(actual, expected.output, 4);
  });

  it("processes a contiguous sequence longer than MAX_CONTEXT as independent MAX_CONTEXT batches", async () => {
    const vectors = MAX_CONTEXT + 5;
    const hiddenDimensions = 8;
    const headsCount = 1;

    const input = randomMatrix(vectors, hiddenDimensions);
    const attentionWeights = randomWeights(hiddenDimensions);

    const expected = createMatrix(vectors, hiddenDimensions);
    for (let start = 0; start < vectors; start += MAX_CONTEXT) {
      const count = Math.min(MAX_CONTEXT, vectors - start);
      const batch: Matrix = {
        vectors: count,
        dimensions: hiddenDimensions,
        values: input.values.slice(
          start * hiddenDimensions,
          (start + count) * hiddenDimensions,
        ),
      };
      const batchOutput = runSelfAttentionMechanism(
        batch,
        headsCount,
        attentionWeights,
      ).output;
      expected.values.set(batchOutput.values, start * hiddenDimensions);
    }

    const actual = await runGpu(input, headsCount, attentionWeights);

    expectMatrixCloseTo(actual, expected, 4);
  });

  it("matches the CPU mechanism across the full pipeline for a random multi-head case", async () => {
    const vectors = 7;
    const hiddenDimensions = 8;
    const headsCount = 2;
    const rand = () => Math.random() * 2 - 1;
    const randomMatrix = (rows: number, cols: number): Matrix => ({
      vectors: rows,
      dimensions: cols,
      values: new Float32Array(Array.from({ length: rows * cols }, rand)),
    });

    const input = randomMatrix(vectors, hiddenDimensions);
    const attentionWeights: AttentionWeights = {
      Q: randomMatrix(hiddenDimensions, hiddenDimensions),
      K: randomMatrix(hiddenDimensions, hiddenDimensions),
      V: randomMatrix(hiddenDimensions, hiddenDimensions),
      out: randomMatrix(hiddenDimensions, hiddenDimensions),
    };

    const expected = runSelfAttentionMechanism(
      input,
      headsCount,
      attentionWeights,
    );
    const actual = await runGpu(input, headsCount, attentionWeights);

    expectMatrixCloseTo(actual, expected.output, 4);
  });
});

const rand = () => Math.random() * 2 - 1;

const randomMatrix = (rows: number, cols: number): Matrix => ({
  vectors: rows,
  dimensions: cols,
  values: new Float32Array(Array.from({ length: rows * cols }, rand)),
});

const randomWeights = (hiddenDimensions: number): AttentionWeights => ({
  Q: randomMatrix(hiddenDimensions, hiddenDimensions),
  K: randomMatrix(hiddenDimensions, hiddenDimensions),
  V: randomMatrix(hiddenDimensions, hiddenDimensions),
  out: randomMatrix(hiddenDimensions, hiddenDimensions),
});

/**
 * Lay each batch's tokens out in its own MAX_CONTEXT-sized row slot,
 * zero-padding the unused tail. Batch b lives in rows [b*MAX_CONTEXT, ...).
 */
const packBatches = (batches: Matrix[]): Matrix => {
  const hiddenDimensions = batches[0]!.dimensions;
  const packed = createMatrix(batches.length * MAX_CONTEXT, hiddenDimensions);

  batches.forEach((batch, batchIndex) => {
    const rowOffset = batchIndex * MAX_CONTEXT;
    for (let r = 0; r < batch.vectors; r++) {
      for (let c = 0; c < hiddenDimensions; c++) {
        packed.values[getFlatIndex(rowOffset + r, c, hiddenDimensions)] =
          batch.values[getFlatIndex(r, c, hiddenDimensions)]!;
      }
    }
  });

  return packed;
};

/** Compare only the real (non-padding) rows of each batch slot. */
const expectBatchOutputsCloseTo = (
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

describe("runSelfAttentionMechanismOnGPU — multi-batch", () => {
  it("matches per-batch CPU output for two single-head batches of different lengths", async () => {
    const hiddenDimensions = 4;
    const headsCount = 1;
    const weights = randomWeights(hiddenDimensions);

    const batchA = randomMatrix(3, hiddenDimensions);
    const batchB = randomMatrix(6, hiddenDimensions);

    const packed = packBatches([batchA, batchB]);

    const expectedA = runSelfAttentionMechanism(batchA, headsCount, weights)
      .output;
    const expectedB = runSelfAttentionMechanism(batchB, headsCount, weights)
      .output;

    const actual = await runGpu(packed, headsCount, weights);

    expectBatchOutputsCloseTo(actual, [expectedA, expectedB]);
  });

  it("matches per-batch CPU output for two multi-head batches of different lengths", async () => {
    const hiddenDimensions = 8;
    const headsCount = 2;
    const weights = randomWeights(hiddenDimensions);

    const batchA = randomMatrix(4, hiddenDimensions);
    const batchB = randomMatrix(7, hiddenDimensions);

    const packed = packBatches([batchA, batchB]);

    const expectedA = runSelfAttentionMechanism(batchA, headsCount, weights)
      .output;
    const expectedB = runSelfAttentionMechanism(batchB, headsCount, weights)
      .output;

    const actual = await runGpu(packed, headsCount, weights);

    expectBatchOutputsCloseTo(actual, [expectedA, expectedB]);
  });

  it("matches per-batch CPU output for three batches", async () => {
    const hiddenDimensions = 6;
    const headsCount = 3;
    const weights = randomWeights(hiddenDimensions);

    const batches = [
      randomMatrix(2, hiddenDimensions),
      randomMatrix(5, hiddenDimensions),
      randomMatrix(1, hiddenDimensions),
    ];

    const packed = packBatches(batches);

    const expected = batches.map(
      (batch) => runSelfAttentionMechanism(batch, headsCount, weights).output,
    );

    const actual = await runGpu(packed, headsCount, weights);

    expectBatchOutputsCloseTo(actual, expected);
  });

  it("does not leak one batch's tokens into another batch's attention", async () => {
    const hiddenDimensions = 4;
    const headsCount = 1;
    const weights = randomWeights(hiddenDimensions);

    const batchB = randomMatrix(5, hiddenDimensions);

    const firstA = randomMatrix(3, hiddenDimensions);
    const secondA = randomMatrix(3, hiddenDimensions);

    const actualFirst = await runGpu(
      packBatches([firstA, batchB]),
      headsCount,
      weights,
    );
    const actualSecond = await runGpu(
      packBatches([secondA, batchB]),
      headsCount,
      weights,
    );

    const rowOffset = 1 * MAX_CONTEXT;
    for (let r = 0; r < batchB.vectors; r++) {
      for (let c = 0; c < hiddenDimensions; c++) {
        const flat = getFlatIndex(rowOffset + r, c, actualFirst.dimensions);
        expect(actualFirst.values[flat]).toBeCloseTo(
          actualSecond.values[flat]!,
          5,
        );
      }
    }
  });
});
