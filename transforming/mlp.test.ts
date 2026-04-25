import { describe, expect, it } from "vitest";
import {
  getMultilayerPerceptronUpdateMatrix,
  getMultilayerPerceptronUpdateVector,
} from "./mlp.ts";
import type { MultilayerPerceptronWeights } from "../model/types.ts";

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

describe("getMultilayerPerceptronUpdateVector", () => {
  it("supports hidden states wider than 1 feature", () => {
    expect(
      getMultilayerPerceptronUpdateVector([2, -1], twoDimPerceptron),
    ).toEqual([0.5, 3.5]);
  });

  it("handles a 3-feature input with a 12-neuron intermediate layer", () => {
    expect(
      getMultilayerPerceptronUpdateVector([1, 2, -1], threeDimPerceptron),
    ).toEqual([3, -1, -4]);
  });

  it("throws when wUp does not expand to 4x the input width", () => {
    expect(() =>
      getMultilayerPerceptronUpdateVector([2, -1], {
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
      getMultilayerPerceptronUpdateVector([1, 2, -1], {
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

  it("returns only the learned update, not the original residual input", () => {
    const input = [2, -1];

    expect(
      getMultilayerPerceptronUpdateVector(input, twoDimPerceptron),
    ).not.toEqual([4.5, 1.5]);
  });
});

describe("getMultilayerPerceptronUpdateMatrix", () => {
  it("runs the same MLP independently on each row vector", () => {
    expect(
      getMultilayerPerceptronUpdateMatrix(
        [
          [2, -1],
          [-1, 3],
          [0, 0],
        ],
        twoDimPerceptron,
      ),
    ).toEqual([
      [0.5, 3.5],
      [38.5, 22.5],
      [1.5, -1.5],
    ]);
  });

  it("returns per-row updates without adding them back to the original matrix", () => {
    const input = [
      [2, -1],
      [0, 0],
    ];

    expect(
      getMultilayerPerceptronUpdateMatrix(input, twoDimPerceptron),
    ).not.toEqual([
      [4.5, 1.5],
      [1.5, -1.5],
    ]);
  });
});
