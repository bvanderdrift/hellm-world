import { describe, expect, it } from "vitest";
import { embeddingsBackprop } from "./embeddingBackprop.ts";
import { createMatrix, getFlatIndex, type Matrix } from "../../shared/matrices.ts";

const matrixFrom = (rows: number[][]): Matrix => {
  const vectors = rows.length;
  const dimensions = rows[0]!.length;
  const m = createMatrix(vectors, dimensions);
  for (let i = 0; i < vectors; i++) {
    for (let j = 0; j < dimensions; j++) {
      m.values[getFlatIndex(i, j, dimensions)] = rows[i]![j]!;
    }
  }
  return m;
};

const expectMatrixCloseTo = (actual: Matrix, expected: number[][]) => {
  const exp = matrixFrom(expected);
  expect(actual.vectors).toBe(exp.vectors);
  expect(actual.dimensions).toBe(exp.dimensions);

  for (let i = 0; i < exp.vectors; i++) {
    for (let j = 0; j < exp.dimensions; j++) {
      const idx = getFlatIndex(i, j, exp.dimensions);
      expect(actual.values[idx]).toBeCloseTo(exp.values[idx]!, 10);
    }
  }
};

describe("embeddingsBackprop", () => {
  it("scales each output gradient by the embedding width square root", () => {
    const gradients = embeddingsBackprop(
      matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      matrixFrom([[1, 2, 3, 4]]),
      [1],
    );

    expectMatrixCloseTo(gradients, [
      [0, 0, 0, 0],
      [2, 4, 6, 8],
    ]);
  });

  it("sums scaled gradients when the same token appears multiple times", () => {
    const gradients = embeddingsBackprop(
      matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      matrixFrom([
        [1, 2, 3, 4],
        [10, 20, 30, 40],
      ]),
      [0, 0],
    );

    expectMatrixCloseTo(gradients, [
      [22, 44, 66, 88],
      [0, 0, 0, 0],
    ]);
  });

  it("keeps separate embedding rows independent", () => {
    const gradients = embeddingsBackprop(
      matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      matrixFrom([
        [1, 2, 3, 4],
        [10, 20, 30, 40],
        [100, 200, 300, 400],
      ]),
      [2, 0, 2],
    );

    expectMatrixCloseTo(gradients, [
      [20, 40, 60, 80],
      [0, 0, 0, 0],
      [202, 404, 606, 808],
    ]);
  });
});
