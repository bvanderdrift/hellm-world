import { describe, expect, it } from "vitest";
import { getMultilayerPerceptronActivations } from "./mlp.ts";
import type { MultilayerPerceptronWeights } from "../model/model-types.ts";

const DEFAULT_MLP_MULTIPLE = 4;

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

describe("getMultilayerPerceptronUpdateMatrix", () => {
  it("supports hidden states wider than 1 feature", () => {
    expect(
      getMultilayerPerceptronActivations(
        [[2, -1]],
        twoDimPerceptron,
        DEFAULT_MLP_MULTIPLE,
      ).downingOutput,
    ).toEqual([[0.5, 3.5]]);
  });

  it("handles a 3-feature input with a 12-neuron intermediate layer", () => {
    expect(
      getMultilayerPerceptronActivations(
        [[1, 2, -1]],
        threeDimPerceptron,
        DEFAULT_MLP_MULTIPLE,
      ).downingOutput,
    ).toEqual([[3, -1, -4]]);
  });

  it("uses the provided MLP multiple when validating the expanded width", () => {
    expect(() =>
      getMultilayerPerceptronActivations([[2, -1]], twoDimPerceptron, 3),
    ).toThrow("m has unexpected vector depth 8, expected 6");
  });

  it("throws when wDown does not project back to the input width", () => {
    expect(() =>
      getMultilayerPerceptronActivations(
        [[1, 2, -1]],
        {
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
        },
        DEFAULT_MLP_MULTIPLE,
      ),
    ).toThrow("m has unexpected vector depth 2, expected 3");
  });

  it("returns only the learned update, not the original residual input", () => {
    const input = [[2, -1]];

    expect(
      getMultilayerPerceptronActivations(
        input,
        twoDimPerceptron,
        DEFAULT_MLP_MULTIPLE,
      ).downingOutput,
    ).not.toEqual([[4.5, 1.5]]);
  });

  it("runs the same MLP independently on each row vector", () => {
    expect(
      getMultilayerPerceptronActivations(
        [
          [2, -1],
          [-1, 3],
          [0, 0],
        ],
        twoDimPerceptron,
        DEFAULT_MLP_MULTIPLE,
      ).downingOutput,
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
      getMultilayerPerceptronActivations(
        input,
        twoDimPerceptron,
        DEFAULT_MLP_MULTIPLE,
      ).downingOutput,
    ).not.toEqual([
      [4.5, 1.5],
      [1.5, -1.5],
    ]);
  });

  it("throws for an empty input matrix", () => {
    expect(() =>
      getMultilayerPerceptronActivations(
        [],
        twoDimPerceptron,
        DEFAULT_MLP_MULTIPLE,
      ),
    ).toThrow();
  });
});
