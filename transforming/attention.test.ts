import { describe, expect, it } from "vitest";
import {
  runSelfAttentionHead,
  runSelfAttentionMechanism,
} from "./attention.ts";
import type { AttentionWeights } from "../model/model-types.ts";
import { createMatrix, type Matrix, getFlatIndex } from "../shared/matrices.ts";

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

const input = matrixFrom([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
]);

describe("runSelfAttentionHead", () => {
  it("uses the value matrix as the payload that gets mixed across positions", () => {
    const output = runSelfAttentionHead(
      matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      matrixFrom([
        [1, 10, 100, 1000],
        [2, 20, 200, 2000],
      ]),
      1,
      4,
    );

    expectMatrixCloseTo(output.output, [
      [1, 10, 100, 1000],
      [1.5, 15, 150, 1500],
    ]);
  });

  it("uses query-key similarity to weight the visible values", () => {
    const output = runSelfAttentionHead(
      matrixFrom([[0], [Math.log(2)]]),
      matrixFrom([[1], [0]]),
      matrixFrom([[1], [2]]),
      1,
      1,
    );

    expectMatrixCloseTo(output.output, [[1], [4 / 3]]);
  });

  it("does not attend to future keys and values", () => {
    const output = runSelfAttentionHead(
      matrixFrom([[1], [0]]),
      matrixFrom([[0], [10]]),
      matrixFrom([[1], [5]]),
      1,
      1,
    );

    expectMatrixCloseTo(output.output, [[1], [3]]);
  });
});

describe("runSelfAttentionMechanism", () => {
  it("projects Q/K/V once, splits heads by feature columns, then applies one shared output projection", () => {
    const twoHeadAttention: AttentionWeights = {
      Q: matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      K: matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      V: matrixFrom([
        [1, 0, 10, 0],
        [3, 0, 30, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      out: matrixFrom([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
      ]),
    };

    const output = runSelfAttentionMechanism(input, 2, twoHeadAttention);

    expectMatrixCloseTo(output.output, [
      [1, 10, 0, 0],
      [2, 20, 0, 0],
    ]);
  });
});
