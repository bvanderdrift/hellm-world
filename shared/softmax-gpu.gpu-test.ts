import { describe, it } from "bun:test";
import { gpuContext } from "./gpu-context.ts";
import { createMatrixBuffer, extractMatrixBuffer } from "./matrices-gpu.ts";
import { createMatrix, type Matrix } from "./matrices.ts";
import { matrixFrom, expectMatrixCloseTo } from "../testing/testing-utils.ts";
import { runSelfAttentionHead } from "../transforming/attention/attention.ts";
import { softmaxOnGpu } from "./softmax-gpu.ts";

const runGpu = async (
  attentionRelevancyOutput: Matrix,
  headsCount: number,
): Promise<Matrix> => {
  const relevancyBuffer = createMatrixBuffer(attentionRelevancyOutput);
  const matchingKeyProducts = createMatrixBuffer(
    createMatrix(
      attentionRelevancyOutput.vectors,
      attentionRelevancyOutput.dimensions,
    ),
  );

  const encoder = gpuContext.device.createCommandEncoder();
  softmaxOnGpu(relevancyBuffer, matchingKeyProducts, encoder, headsCount);
  gpuContext.device.queue.submit([encoder.finish()]);
  await gpuContext.device.queue.onSubmittedWorkDone();

  return extractMatrixBuffer(matchingKeyProducts);
};

describe("softmaxOnGpu", () => {
  it("softmaxes a single head's causal relevancy row", async () => {
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

    const { attentionRelevancyOutput, softmaxOutput } = runSelfAttentionHead(
      inputQ,
      inputK,
      inputV,
      headsCount,
      headDimensionsCount,
    );

    const actual = await runGpu(attentionRelevancyOutput, headsCount);

    expectMatrixCloseTo(actual, softmaxOutput, 4);
  });

  it("includes the diagonal (self) position in each row's softmax", async () => {
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

    const { attentionRelevancyOutput, softmaxOutput } = runSelfAttentionHead(
      inputQ,
      inputK,
      inputV,
      headsCount,
      headDimensionsCount,
    );

    const actual = await runGpu(attentionRelevancyOutput, headsCount);

    expectMatrixCloseTo(actual, softmaxOutput, 4);
  });

  it("softmaxes each head's own column block independently", async () => {
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

    const { attentionRelevancyOutput, softmaxOutput } = runSelfAttentionHead(
      inputQ,
      inputK,
      inputV,
      headsCount,
      headDimensionsCount,
    );

    const actual = await runGpu(attentionRelevancyOutput, headsCount);

    expectMatrixCloseTo(actual, softmaxOutput, 4);
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

    const { attentionRelevancyOutput, softmaxOutput } = runSelfAttentionHead(
      inputQ,
      inputK,
      inputV,
      headsCount,
      headDimensionsCount,
    );

    const actual = await runGpu(attentionRelevancyOutput, headsCount);

    expectMatrixCloseTo(actual, softmaxOutput, 4);
  });
});
