import { describe, it, expect } from "bun:test";
import { gpuContext } from "./gpu-context.ts";
import { createMatrixBuffer, extractMatrixBuffer } from "./matrices/matrices-gpu.ts";
import type { Matrix } from "./matrices/matrices.ts";
import { reluOnGpu } from "./relu-gpu.ts";

// reluOnGpu is host-callable (runs its own pipeline), so we can test directly.
describe("reluOnGpu", () => {
  it("zeroes negatives and keeps non-negatives, in place", async () => {
    const matrix: Matrix = {
      vectors: 2,
      dimensions: 3,
      values: new Float32Array([-1, 2, -3, 4, 0, -0.5]),
    };
    const buffer = createMatrixBuffer(matrix);

    reluOnGpu(buffer);
    await gpuContext.device.queue.onSubmittedWorkDone();

    const result = await extractMatrixBuffer(buffer);
    const used = result.values.slice(0, matrix.vectors * matrix.dimensions);
    expect(Array.from(used)).toEqual([0, 2, 0, 4, 0, 0]);
  });

  it("activates every element when the matrix spans multiple workgroups", async () => {
    const vectors = 40;
    const dimensions = 48;
    const values = new Float32Array(
      Array.from({ length: vectors * dimensions }, (_, k) =>
        k % 2 === 0 ? k + 1 : -(k + 1),
      ),
    );
    const matrix: Matrix = { vectors, dimensions, values };
    const buffer = createMatrixBuffer(matrix);

    reluOnGpu(buffer);
    await gpuContext.device.queue.onSubmittedWorkDone();

    const result = await extractMatrixBuffer(buffer);
    const expected = Array.from(values, (v) => (v > 0 ? v : 0));
    expect(Array.from(result.values)).toEqual(expected);
  });
});
