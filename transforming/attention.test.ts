import { describe, expect, it } from "vitest";
import {
  runSelfAttentionHead,
  runSelfAttentionMechanism,
} from "./attention.ts";
import type { AttentionWeights } from "../weights/types.ts";

const expectMatrixCloseTo = (actual: number[][], expected: number[][]) => {
  expect(actual).toHaveLength(expected.length);

  for (const [rowIndex, expectedRow] of expected.entries()) {
    const actualRow = actual[rowIndex];

    expect(actualRow).toHaveLength(expectedRow.length);

    for (const [columnIndex, expectedValue] of expectedRow.entries()) {
      expect(actualRow?.[columnIndex]).toBeCloseTo(expectedValue, 10);
    }
  }
};

const input = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
];

describe("runSelfAttentionHead", () => {
  it("uses the value matrix as the payload that gets mixed across positions", () => {
    const output = runSelfAttentionHead(
      [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      [
        [1, 10, 100, 1000],
        [2, 20, 200, 2000],
      ],
    );

    expectMatrixCloseTo(output, [
      [1, 10, 100, 1000],
      [1.5, 15, 150, 1500],
    ]);
  });

  it("uses query-key similarity to weight the visible values", () => {
    const output = runSelfAttentionHead(
      [[0], [Math.log(2)]],
      [[1], [0]],
      [[1], [2]],
    );

    expectMatrixCloseTo(output, [[1], [4 / 3]]);
  });

  it("does not attend to future keys and values", () => {
    const output = runSelfAttentionHead([[1], [0]], [[0], [10]], [[1], [5]]);

    expectMatrixCloseTo(output, [[1], [3]]);
  });
});

describe("runSelfAttentionMechanism", () => {
  it("projects Q/K/V once, splits heads by feature columns, then applies one shared output projection", () => {
    const twoHeadAttention: AttentionWeights = {
      Q: [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      K: [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      V: [
        [1, 0, 10, 0],
        [3, 0, 30, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      out: [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
      ],
    };

    const output = runSelfAttentionMechanism(input, 2, twoHeadAttention);

    expectMatrixCloseTo(output, [
      [1, 10, 0, 0],
      [2, 20, 0, 0],
    ]);
  });
});
