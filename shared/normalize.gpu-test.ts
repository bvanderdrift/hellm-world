import { describe, it, expect } from "bun:test";
import { gpuContext } from "./gpu-context.ts";
import { createMatrix, type Matrix } from "./matrices.ts";
import { createMatrixBuffer, extractMatrixBuffer } from "./matrices-gpu.ts";
import { normalize } from "./normalize.ts";
import { normalizeOnGpu } from "./normalize-gpu.ts";
import { expectMatrixCloseTo } from "../testing/testing-utils.ts";

// normalizeOnGpu mutates the buffer in place; the CPU `normalize` returns a new
// matrix. We use the CPU version as the oracle and compare the used region.
const runGpu = async (matrix: Matrix): Promise<Matrix> => {
  const buffer = createMatrixBuffer(matrix);
  const encoder = gpuContext.device.createCommandEncoder();
  normalizeOnGpu(buffer, encoder);
  gpuContext.device.queue.submit([encoder.finish()]);
  await gpuContext.device.queue.onSubmittedWorkDone();
  return extractMatrixBuffer(buffer);
};

describe("normalizeOnGpu", () => {
  it("matches the CPU normalize for a single vector", async () => {
    const matrix: Matrix = {
      vectors: 1,
      dimensions: 4,
      values: new Float32Array([1, 2, 3, 4]),
    };

    const gpu = await runGpu(matrix);
    expectMatrixCloseTo(gpu, normalize(matrix), 4);
  });

  it("normalizes each row independently", async () => {
    const matrix: Matrix = {
      vectors: 3,
      dimensions: 4,
      values: new Float32Array([
        // row 0
        1, 2, 3, 4,
        // row 1 (different scale)
        10, 20, 30, 40,
        // row 2 (negatives)
        -2, -1, 1, 2,
      ]),
    };

    const gpu = await runGpu(matrix);
    expectMatrixCloseTo(gpu, normalize(matrix), 4);
  });

  it("produces a zero-mean, unit-variance row (standard normal)", async () => {
    const matrix: Matrix = {
      vectors: 1,
      dimensions: 5,
      values: new Float32Array([2, 4, 4, 4, 6]),
    };

    const gpu = await runGpu(matrix);

    const row = Array.from(gpu.values.slice(0, matrix.dimensions));
    const mean = row.reduce((sum, value) => sum + value, 0) / row.length;
    const variance =
      row.reduce((sum, value) => sum + (value - mean) ** 2, 0) / row.length;

    expect(mean).toBeCloseTo(0, 4);
    expect(variance).toBeCloseTo(1, 4);
  });

  it("does not divide by zero when every value in a row is identical", async () => {
    const matrix: Matrix = {
      vectors: 1,
      dimensions: 4,
      values: new Float32Array([3, 3, 3, 3]),
    };

    const gpu = await runGpu(matrix);

    const row = Array.from(gpu.values.slice(0, matrix.dimensions));
    for (const value of row) {
      expect(Number.isNaN(value)).toBe(false);
      expect(value).toBeCloseTo(0, 4);
    }
  });

  it("matches CPU normalize for a larger random matrix", async () => {
    const rand = () => Math.random() * 2 - 1;
    const matrix = createMatrix(6, 16, rand);

    const gpu = await runGpu(matrix);
    expectMatrixCloseTo(gpu, normalize(matrix), 4);
  });
});
