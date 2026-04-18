import { describe, expect, it } from "vitest";
import {
  runSelfAttentionHead,
  runSelfAttentionMechanism,
} from "./attention.ts";
import type {
  AttentionHeadWeights,
  AttentionWeights,
} from "../weights/weight-helpers.ts";

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

const identityLikeHead = (valueDown: number[], valueUp: number[][]) => ({
  Q: [[0], [0], [0], [0]],
  K: [[0], [0], [0], [0]],
  V: {
    down: valueDown.map((value) => [value]),
    up: valueUp,
  },
});

const zeroedHeadSpaceRow = [0];

const headThatProjectsValues = (): AttentionHeadWeights => ({
  Q: [
    zeroedHeadSpaceRow,
    zeroedHeadSpaceRow,
    zeroedHeadSpaceRow,
    zeroedHeadSpaceRow,
  ],
  K: [
    zeroedHeadSpaceRow,
    zeroedHeadSpaceRow,
    zeroedHeadSpaceRow,
    zeroedHeadSpaceRow,
  ],
  V: {
    down: [[1], [2], [0], [0]],
    up: [[1, 10, 100, 1000]],
  },
});

const headThatUsesQueryKeySimilarity = (): AttentionHeadWeights => ({
  Q: [[0], [Math.log(2)], [0], [0]],
  K: [[1], [0], [0], [0]],
  V: {
    down: [[1], [2], [0], [0]],
    up: [[1, 0, 0, 0]],
  },
});

const headWithStrongFutureMatch = (): AttentionHeadWeights => ({
  Q: [[1], [0], [0], [0]],
  K: [[0], [10], [0], [0]],
  V: {
    down: [[1], [5], [0], [0]],
    up: [[1, 0, 0, 0]],
  },
});

const headA: AttentionHeadWeights = identityLikeHead(
  [1, 3, 0, 0],
  [[1, 0, 0, 0]],
);

const headB: AttentionHeadWeights = identityLikeHead(
  [10, 30, 0, 0],
  [[0, 1, 0, 0]],
);

const twoHeadAttention: AttentionWeights = {
  heads: [headA, headB],
};

describe("runSelfAttentionHead", () => {
  it("uses the value projection as the payload that gets mixed across positions", () => {
    const output = runSelfAttentionHead(input, headThatProjectsValues());

    expectMatrixCloseTo(output, [
      [1, 10, 100, 1000],
      [1.5, 15, 150, 1500],
    ]);
  });

  it("uses query-key similarity to weight the visible values", () => {
    const output = runSelfAttentionHead(
      input,
      headThatUsesQueryKeySimilarity(),
    );

    expectMatrixCloseTo(output, [
      [1, 0, 0, 0],
      [4 / 3, 0, 0, 0],
    ]);
  });

  it("does not attend to future keys and values", () => {
    const output = runSelfAttentionHead(input, headWithStrongFutureMatch());

    expectMatrixCloseTo(output, [
      [1, 0, 0, 0],
      [3, 0, 0, 0],
    ]);
  });
});

describe("runSelfAttentionMechanism", () => {
  it("adds the model-space contribution from each head", () => {
    const output = runSelfAttentionMechanism(input, twoHeadAttention);

    expectMatrixCloseTo(output, [
      [1, 10, 0, 0],
      [2, 20, 0, 0],
    ]);
  });
});
