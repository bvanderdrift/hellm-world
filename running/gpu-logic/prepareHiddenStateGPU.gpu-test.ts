import { describe, it, expect } from "bun:test";
import { d } from "typegpu";
import {
  createMatrix,
  getFlatIndex,
  type Matrix,
} from "../../shared/matrices.ts";
import {
  createMatrixBuffer,
  extractMatrixBuffer,
} from "../../shared/matrices-gpu.ts";
import { gpuContext } from "../../shared/gpu-context.ts";
import { expectMatrixCloseTo } from "../../testing/testing-utils.ts";
import { getPositionEncoding } from "../position-encoding.ts";
import { prepareHiddenState } from "./prepareHiddenStateGPU.ts";
import { MAX_CONTEXT } from "../llm-shared.ts";

// Inline CPU reference, mirroring the head of llmForwardPassByTokens:
//   hiddenState[token][j] = sqrt(dim) * embeddings[vocabIndex][j] + posEnc[token][j]
const cpuPrepareHiddenState = (
  embeddings: Matrix,
  vocabIndices: number[],
): Matrix => {
  const dim = embeddings.dimensions;
  const scalar = Math.sqrt(dim);
  const out = createMatrix(vocabIndices.length, dim);
  const posEnc = getPositionEncoding(vocabIndices.length, dim);

  for (let token = 0; token < vocabIndices.length; token++) {
    const vocabIndex = vocabIndices[token]!;
    for (let j = 0; j < dim; j++) {
      out.values[getFlatIndex(token, j, dim)] =
        scalar * embeddings.values[getFlatIndex(vocabIndex, j, dim)]! +
        posEnc.values[getFlatIndex(token, j, dim)]!;
    }
  }

  return out;
};

const makeTokenIndicesBuffer = (vocabIndices: number[]) =>
  gpuContext
    .createBuffer(d.arrayOf(d.f32, vocabIndices.length), vocabIndices)
    .$usage("storage");

const runGpu = async (
  embeddings: Matrix,
  vocabIndices: number[],
): Promise<Matrix> => {
  const embeddingsBuffer = createMatrixBuffer(embeddings);
  const hiddenState = createMatrixBuffer({
    vectors: vocabIndices.length,
    dimensions: embeddings.dimensions,
  });
  const tokenIndices = makeTokenIndicesBuffer(vocabIndices);

  prepareHiddenState(tokenIndices, hiddenState, embeddingsBuffer);
  await gpuContext.device.queue.onSubmittedWorkDone();

  return extractMatrixBuffer(hiddenState);
};

/**
 * Lay each batch's tokens into its own MAX_CONTEXT-sized row slot (batch b at
 * rows [b*MAX_CONTEXT, ...)), zero-padding the tail, then run the kernel over
 * the whole packed buffer — the same layout the batched forward pass uses.
 */
const runGpuBatched = async (
  embeddings: Matrix,
  batches: number[][],
): Promise<Matrix> => {
  const totalRows = batches.length * MAX_CONTEXT;
  const flatTokenIndices = new Array(totalRows).fill(0);
  batches.forEach((vocabIndices, batchIndex) => {
    vocabIndices.forEach((vocabIndex, tokenIndex) => {
      flatTokenIndices[batchIndex * MAX_CONTEXT + tokenIndex] = vocabIndex;
    });
  });

  const embeddingsBuffer = createMatrixBuffer(embeddings);
  const hiddenState = createMatrixBuffer({
    vectors: totalRows,
    dimensions: embeddings.dimensions,
  });
  const tokenIndices = makeTokenIndicesBuffer(flatTokenIndices);

  prepareHiddenState(tokenIndices, hiddenState, embeddingsBuffer);
  await gpuContext.device.queue.onSubmittedWorkDone();

  return extractMatrixBuffer(hiddenState);
};

/** Each batch's real rows must equal the standalone per-sequence CPU result. */
const expectBatchesMatchCpu = (
  actual: Matrix,
  embeddings: Matrix,
  batches: number[][],
  precision = 4,
) => {
  const dim = embeddings.dimensions;
  batches.forEach((vocabIndices, batchIndex) => {
    const expected = cpuPrepareHiddenState(embeddings, vocabIndices);
    const rowOffset = batchIndex * MAX_CONTEXT;
    for (let token = 0; token < vocabIndices.length; token++) {
      for (let j = 0; j < dim; j++) {
        const actualValue =
          actual.values[getFlatIndex(rowOffset + token, j, dim)];
        const expectedValue = expected.values[getFlatIndex(token, j, dim)]!;
        expect(actualValue).toBeCloseTo(expectedValue, precision);
      }
    }
  });
};

describe("prepareHiddenState (GPU)", () => {
  it("scales the embedding by sqrt(dim) and adds positional encoding", async () => {
    // 3 vocab entries, 4 hidden dimensions
    const embeddings: Matrix = {
      vectors: 3,
      dimensions: 4,
      values: new Float32Array([
        // vocab 0
        1, 2, 3, 4,
        // vocab 1
        -1, 0.5, 2, -3,
        // vocab 2
        0.25, -0.5, 10, -10,
      ]),
    };
    const vocabIndices = [2, 0]; // two tokens, picking vocab rows 2 then 0

    const actual = await runGpu(embeddings, vocabIndices);
    const expected = cpuPrepareHiddenState(embeddings, vocabIndices);

    expectMatrixCloseTo(actual, expected, 4);
  });

  it("matches the CPU reference for a larger random case", async () => {
    const vocabSize = 8;
    const dim = 16;
    const rand = () => Math.random() * 2 - 1;
    const embeddings = createMatrix(vocabSize, dim, rand);
    const vocabIndices = [5, 0, 7, 3, 1];

    const actual = await runGpu(embeddings, vocabIndices);
    const expected = cpuPrepareHiddenState(embeddings, vocabIndices);

    expectMatrixCloseTo(actual, expected, 4);
  });

  it("writes position 0 with zero positional offset (sin0/cos0 => +0/+1 baseline)", async () => {
    // single token => positional encoding row 0 = [0,1,0,1,...]
    const dim = 4;
    const embeddings: Matrix = {
      vectors: 1,
      dimensions: dim,
      values: new Float32Array([2, 2, 2, 2]),
    };
    const actual = await runGpu(embeddings, [0]);

    const scalar = Math.sqrt(dim); // 2
    // even dims get +0, odd dims get +1
    expect(actual.values[0]!).toBeCloseTo(scalar * 2 + 0, 4);
    expect(actual.values[1]!).toBeCloseTo(scalar * 2 + 1, 4);
    expect(actual.values[2]!).toBeCloseTo(scalar * 2 + 0, 4);
    expect(actual.values[3]!).toBeCloseTo(scalar * 2 + 1, 4);
  });

  it("encodes each batched sequence from its own slot's position 0, not the flattened row", async () => {
    const vocabSize = 8;
    const dim = 16;
    const rand = () => Math.random() * 2 - 1;
    const embeddings = createMatrix(vocabSize, dim, rand);

    // Different lengths so any position drift surfaces; each batch's positions
    // must restart at 0 within its slot regardless of the slot's row offset.
    const batches = [
      [5, 0, 7],
      [1, 2, 3, 4],
      [6],
    ];

    const actual = await runGpuBatched(embeddings, batches);

    expectBatchesMatchCpu(actual, embeddings, batches);
  });

  it("gives an identical encoding to the same token sitting at position 0 of every batch slot", async () => {
    const dim = 4;
    const embeddings = createMatrix(4, dim, () => Math.random() * 2 - 1);

    // Same token, position 0, in three different slots. If the kernel used the
    // flattened row (or the batch index) for position, these would diverge.
    const batches = [[2], [2], [2]];

    const actual = await runGpuBatched(embeddings, batches);

    for (let j = 0; j < dim; j++) {
      const first = actual.values[getFlatIndex(0 * MAX_CONTEXT, j, dim)]!;
      const second = actual.values[getFlatIndex(1 * MAX_CONTEXT, j, dim)]!;
      const third = actual.values[getFlatIndex(2 * MAX_CONTEXT, j, dim)]!;
      expect(second).toBeCloseTo(first, 4);
      expect(third).toBeCloseTo(first, 4);
    }
  });
});
