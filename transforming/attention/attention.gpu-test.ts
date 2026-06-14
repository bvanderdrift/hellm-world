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
