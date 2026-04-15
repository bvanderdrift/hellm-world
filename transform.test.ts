import { describe, expect, it } from "vitest";
import {
  runMultilayerPerceptronOnMatrix,
  runMultilayerPerceptronOnVector,
} from "./transform.ts";
import type { MultilayerPerceptronWeights } from "./weights.ts";

const twoDimPerceptron: MultilayerPerceptronWeights = {
  wUp: {
    weightsMatrix: [
      [1, 0, -1, 2, 0, 1, 0, -2],
      [0, 1, 2, -1, 3, 0, -2, 1],
    ],
    biasVector: [0, -1, 1, 0, -2, 2, 1, -3],
  },
  wDown: {
    weightsMatrix: [
      [1, 0],
      [2, 1],
      [0, 3],
      [-1, 2],
      [4, 0],
      [0, -2],
      [1, 1],
      [3, 0],
    ],
    biasVector: [0.5, -1.5],
  },
};

const threeDimPerceptron: MultilayerPerceptronWeights = {
  wUp: {
    weightsMatrix: [
      [1, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, -2, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0],
    ],
    biasVector: [0, 1, 0, -3, 0, 2, -1, 0, 1, 0, 2, -2],
  },
  wDown: {
    weightsMatrix: [
      [1, 0, 0],
      [2, 0, 0],
      [0, 0, 3],
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
      [0, 0, 1],
      [0, -1, 0],
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, -2],
      [0, 0, 1],
    ],
    biasVector: [0, -1, 2],
  },
};

describe("runMultilayerPerceptronOnVector", () => {
  it("supports hidden states wider than 1 feature", () => {
    expect(runMultilayerPerceptronOnVector([2, -1], twoDimPerceptron)).toEqual(
      [2.5, 2.5],
    );
  });

  it("handles a 3-feature input with a 12-neuron intermediate layer", () => {
    expect(
      runMultilayerPerceptronOnVector([1, 2, -1], threeDimPerceptron),
    ).toEqual([4, 1, -5]);
  });

  it("throws when wUp does not expand to 4x the input width", () => {
    expect(() =>
      runMultilayerPerceptronOnVector([2, -1], {
        ...twoDimPerceptron,
        wUp: {
          weightsMatrix: [
            [1, 0, -1, 2, 0, 1, 0],
            [0, 1, 2, -1, 3, 0, -2],
          ],
          biasVector: [0, -1, 1, 0, -2, 2, 1],
        },
      }),
    ).toThrow("m has unexpected vector depth 7, expected 8");
  });

  it("throws when wDown does not project back to the input width", () => {
    expect(() =>
      runMultilayerPerceptronOnVector([1, 2, -1], {
        ...threeDimPerceptron,
        wDown: {
          weightsMatrix: [
            [1, 0],
            [2, 0],
            [0, 3],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 1],
            [0, -1],
            [1, 0],
            [0, 2],
            [0, -2],
            [0, 1],
          ],
          biasVector: [0, -1],
        },
      }),
    ).toThrow("m has unexpected vector depth 2, expected 3");
  });
});

describe("runMultilayerPerceptronOnMatrix", () => {
  it("runs the same MLP independently on each row vector", () => {
    expect(
      runMultilayerPerceptronOnMatrix(
        [
          [2, -1],
          [-1, 3],
          [0, 0],
        ],
        twoDimPerceptron,
      ),
    ).toEqual([
      [2.5, 2.5],
      [37.5, 25.5],
      [1.5, -1.5],
    ]);
  });
});
