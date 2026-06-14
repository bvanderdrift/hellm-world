import { describe, it } from "bun:test";
import { gpuContext } from "./gpu-context.ts";
import { createMatrixBuffer, extractMatrixBuffer } from "./matrices-gpu.ts";
import { createMatrix, type Matrix } from "./matrices.ts";
import { matrixFrom, expectMatrixCloseTo } from "../testing/testing-utils.ts";
import { softmaxOnGpu } from "./softmax-gpu.ts";

const softmaxRows = (input: Matrix): Matrix => {
  const output = createMatrix(input.vectors, input.dimensions);

  for (let row = 0; row < input.vectors; row++) {
    const start = row * input.dimensions;

    let biggest = -Infinity;
    for (let j = 0; j < input.dimensions; j++) {
      biggest = Math.max(biggest, input.values[start + j]!);
    }

    let summed = 0;
    for (let j = 0; j < input.dimensions; j++) {
      summed += Math.exp(input.values[start + j]! - biggest);
    }

    for (let j = 0; j < input.dimensions; j++) {
      output.values[start + j] = Math.exp(input.values[start + j]! - biggest) / summed;
    }
  }

  return output;
};

const runGpu = async (input: Matrix): Promise<Matrix> => {
  const inputBuffer = createMatrixBuffer(input);
  const outputBuffer = createMatrixBuffer(
    createMatrix(input.vectors, input.dimensions),
  );

  softmaxOnGpu(inputBuffer, outputBuffer);
  await gpuContext.device.queue.onSubmittedWorkDone();

  return extractMatrixBuffer(outputBuffer);
};

describe("softmaxOnGpu", () => {
  it("normalizes a single row over its full width", async () => {
    const input = matrixFrom([[1, 2, 3]]);

    const actual = await runGpu(input);

    expectMatrixCloseTo(actual, softmaxRows(input), 4);
  });

  it("turns uniform logits into a uniform distribution", async () => {
    const input = matrixFrom([[5, 5, 5, 5]]);

    const actual = await runGpu(input);

    expectMatrixCloseTo(actual, softmaxRows(input), 4);
  });

  it("softmaxes each row independently over the full width", async () => {
    const input = matrixFrom([
      [1, 2, 3],
      [0, 0, 0],
      [-1, 2, -3],
    ]);

    const actual = await runGpu(input);

    expectMatrixCloseTo(actual, softmaxRows(input), 4);
  });

  it("stays numerically stable for large logits", async () => {
    const input = matrixFrom([
      [1000, 1001, 1002],
      [-1000, -1001, -1002],
    ]);

    const actual = await runGpu(input);

    expectMatrixCloseTo(actual, softmaxRows(input), 4);
  });

  it("matches the CPU reference for a larger random case", async () => {
    const vectors = 7;
    const dimensions = 8;
    const rand = () => Math.random() * 2 - 1;

    const input: Matrix = {
      vectors,
      dimensions,
      values: new Float32Array(
        Array.from({ length: vectors * dimensions }, rand),
      ),
    };

    const actual = await runGpu(input);

    expectMatrixCloseTo(actual, softmaxRows(input), 4);
  });
});
