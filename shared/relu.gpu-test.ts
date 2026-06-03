import { describe, it, expect } from "bun:test";
import { gpuContext } from "./gpu-context.ts";
import {
  createMatrixBufferAndCopy,
  extractMatrixBuffer,
} from "./matrices-gpu.ts";
import type { Matrix } from "./matrices.ts";
import { reluOnGpu } from "./relu-gpu.ts";

// reluOnGpu is host-callable (runs its own pipeline), so we can test directly.
describe("reluOnGpu", () => {
  it("zeroes negatives and keeps non-negatives, in place", async () => {
    const matrix: Matrix = {
      vectors: 2,
      dimensions: 3,
      values: new Float32Array([-1, 2, -3, 4, 0, -0.5]),
    };
    const buffer = createMatrixBufferAndCopy(matrix);

    reluOnGpu(buffer);
    await gpuContext.device.queue.onSubmittedWorkDone();

    const result = await extractMatrixBuffer(buffer);
    const used = result.values.slice(0, matrix.vectors * matrix.dimensions);
    expect(Array.from(used)).toEqual([0, 2, 0, 4, 0, 0]);
  });
});
