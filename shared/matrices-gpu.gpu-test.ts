import { describe, it, expect } from "bun:test";
import { d } from "typegpu";
import { gpuContext } from "./gpu-context.ts";
import {
  createMatrixBuffer,
  extractMatrixBuffer,
  multiplyMatricesOnGPU,
  applyScalarToMatrixOnGPU,
  addMatricesOnGPU,
  addVectorAcrossMatrixOnGPU,
} from "./matrices-gpu.ts";
import {
  type Matrix,
  multiplyMatrices,
  applyScalarToMatrix,
  addMatrices,
  addVectorAcrossMatrix,
} from "./matrices.ts";
import { matrixFrom, expectMatrixCloseTo } from "../testing/testing-utils.ts";

const randomMatrix = (vectors: number, dimensions: number): Matrix => ({
  vectors,
  dimensions,
  values: new Float32Array(
    Array.from({ length: vectors * dimensions }, () => Math.random() * 2 - 1),
  ),
});

const flush = async (): Promise<void> => {
  await gpuContext.device.queue.onSubmittedWorkDone();
};

const scalarUniform = (value: number) =>
  gpuContext.createBuffer(d.f32, value).$usage("uniform");

describe("createMatrixBuffer / extractMatrixBuffer", () => {
  it("round-trips a matrix through the GPU unchanged", async () => {
    const matrix = matrixFrom([
      [1, -2, 3],
      [4, 5, -6],
    ]);

    const buffer = createMatrixBuffer(matrix);
    const result = await extractMatrixBuffer(buffer);

    expectMatrixCloseTo(result, matrix, 5);
  });

  it("allocates an uninitialized buffer with the requested shape", async () => {
    const buffer = createMatrixBuffer({ vectors: 4, dimensions: 7 });
    const result = await extractMatrixBuffer(buffer);

    expect(result.vectors).toBe(4);
    expect(result.dimensions).toBe(7);
    expect(result.values.length).toBe(28);
  });
});

describe("multiplyMatricesOnGPU", () => {
  it("matches the CPU reference for a small fixed case", async () => {
    const m1 = matrixFrom([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const m2 = matrixFrom([
      [7, 8, 9],
      [10, 11, 12],
    ]);

    const out = createMatrixBuffer({ vectors: 3, dimensions: 3 });
    multiplyMatricesOnGPU(createMatrixBuffer(m1), createMatrixBuffer(m2), out);
    await flush();

    expectMatrixCloseTo(
      await extractMatrixBuffer(out),
      multiplyMatrices(m1, m2),
      4,
    );
  });

  it("matches the CPU reference when both output dims exceed one workgroup", async () => {
    const shared = 10;
    const m1 = randomMatrix(20, shared);
    const m2 = randomMatrix(shared, 24);

    const out = createMatrixBuffer({ vectors: 20, dimensions: 24 });
    multiplyMatricesOnGPU(createMatrixBuffer(m1), createMatrixBuffer(m2), out);
    await flush();

    expectMatrixCloseTo(
      await extractMatrixBuffer(out),
      multiplyMatrices(m1, m2),
      3,
    );
  });

  it("matches the CPU reference for dims that are not multiples of the tile", async () => {
    const shared = 7;
    const m1 = randomMatrix(33, shared);
    const m2 = randomMatrix(shared, 19);

    const out = createMatrixBuffer({ vectors: 33, dimensions: 19 });
    multiplyMatricesOnGPU(createMatrixBuffer(m1), createMatrixBuffer(m2), out);
    await flush();

    expectMatrixCloseTo(
      await extractMatrixBuffer(out),
      multiplyMatrices(m1, m2),
      3,
    );
  });
});

describe("applyScalarToMatrixOnGPU", () => {
  it("scales every value in place", async () => {
    const matrix = matrixFrom([
      [1, -2, 3],
      [4, 5, -6],
    ]);
    const buffer = createMatrixBuffer(matrix);

    applyScalarToMatrixOnGPU(scalarUniform(2.5), buffer);
    await flush();

    expectMatrixCloseTo(
      await extractMatrixBuffer(buffer),
      applyScalarToMatrix(2.5, matrix),
      4,
    );
  });

  it("scales correctly for dims that are not multiples of the tile", async () => {
    const matrix = randomMatrix(19, 23);
    const buffer = createMatrixBuffer(matrix);

    applyScalarToMatrixOnGPU(scalarUniform(-1.75), buffer);
    await flush();

    expectMatrixCloseTo(
      await extractMatrixBuffer(buffer),
      applyScalarToMatrix(-1.75, matrix),
      4,
    );
  });
});

describe("addMatricesOnGPU", () => {
  it("adds the second matrix into the first in place", async () => {
    const m1 = randomMatrix(5, 9);
    const m2 = randomMatrix(5, 9);
    const buffer = createMatrixBuffer(m1);

    addMatricesOnGPU(buffer, createMatrixBuffer(m2));
    await flush();

    expectMatrixCloseTo(
      await extractMatrixBuffer(buffer),
      addMatrices(m1, m2),
      4,
    );
  });

  it("adds correctly when the element count is not a multiple of the workgroup", async () => {
    const m1 = randomMatrix(17, 35);
    const m2 = randomMatrix(17, 35);
    const buffer = createMatrixBuffer(m1);

    addMatricesOnGPU(buffer, createMatrixBuffer(m2));
    await flush();

    expectMatrixCloseTo(
      await extractMatrixBuffer(buffer),
      addMatrices(m1, m2),
      4,
    );
  });
});

describe("addVectorAcrossMatrixOnGPU", () => {
  it("adds a row vector to every vector of the matrix", async () => {
    const matrix = randomMatrix(6, 8);
    const vector = randomMatrix(1, 8);
    const buffer = createMatrixBuffer(matrix);

    addVectorAcrossMatrixOnGPU(buffer, createMatrixBuffer(vector));
    await flush();

    expectMatrixCloseTo(
      await extractMatrixBuffer(buffer),
      addVectorAcrossMatrix(vector, matrix),
      4,
    );
  });

  it("adds correctly for dims that are not multiples of the tile", async () => {
    const matrix = randomMatrix(19, 23);
    const vector = randomMatrix(1, 23);
    const buffer = createMatrixBuffer(matrix);

    addVectorAcrossMatrixOnGPU(buffer, createMatrixBuffer(vector));
    await flush();

    expectMatrixCloseTo(
      await extractMatrixBuffer(buffer),
      addVectorAcrossMatrix(vector, matrix),
      4,
    );
  });
});

describe("running the same kernel twice with different sizes", () => {
  it("keeps independent sizes across consecutive dispatches", async () => {
    const big = randomMatrix(7, 32);
    const small = randomMatrix(3, 5);
    const bigVector = randomMatrix(1, 32);
    const smallVector = randomMatrix(1, 5);

    const bigBuffer = createMatrixBuffer(big);
    const smallBuffer = createMatrixBuffer(small);

    addVectorAcrossMatrixOnGPU(bigBuffer, createMatrixBuffer(bigVector));
    addVectorAcrossMatrixOnGPU(smallBuffer, createMatrixBuffer(smallVector));
    await flush();

    expectMatrixCloseTo(
      await extractMatrixBuffer(bigBuffer),
      addVectorAcrossMatrix(bigVector, big),
      4,
    );
    expectMatrixCloseTo(
      await extractMatrixBuffer(smallBuffer),
      addVectorAcrossMatrix(smallVector, small),
      4,
    );
  });
});
