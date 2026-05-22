import { describe, expect, it } from "vitest";
import { getMultilayerPerceptronActivations } from "./mlp.ts";
import type { MultilayerPerceptronWeights } from "../model/model-types.ts";
import { matrixFrom, expectMatrixCloseTo } from "../testing/testing-utils.ts";
import type { Matrix } from "../shared/matrices.ts";

const vectorFrom = (values: number[]): Matrix => matrixFrom([values]);

const twoDimPerceptron: MultilayerPerceptronWeights = {
  wUp: {
    weightsMatrix: matrixFrom([
      [1, 0, -1, 2, 0, 1, 0, -2],
      [0, 1, 2, -1, 3, 0, -2, 1],
    ]),
    biasVector: vectorFrom([0, -1, 1, 0, -2, 2, 1, -3]),
  },
  wDown: {
    weightsMatrix: matrixFrom([
      [1, 0],
      [2, 1],
      [0, 3],
      [-1, 2],
      [4, 0],
      [0, -2],
      [1, 1],
      [3, 0],
    ]),
    biasVector: vectorFrom([0.5, -1.5]),
  },
};

const threeDimPerceptron: MultilayerPerceptronWeights = {
  wUp: {
    weightsMatrix: matrixFrom([
      [1, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, -2, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0],
    ]),
    biasVector: vectorFrom([0, 1, 0, -3, 0, 2, -1, 0, 1, 0, 2, -2]),
  },
  wDown: {
    weightsMatrix: matrixFrom([
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
    ]),
    biasVector: vectorFrom([0, -1, 2]),
  },
};

describe("getMultilayerPerceptronUpdateMatrix", () => {
  it("supports hidden states wider than 1 feature", () => {
    expectMatrixCloseTo(
      getMultilayerPerceptronActivations(
        matrixFrom([[2, -1]]),
        twoDimPerceptron,
      ).downingOutput,
      matrixFrom([[0.5, 3.5]]),
    );
  });

  it("handles a 3-feature input with a 12-neuron intermediate layer", () => {
    expectMatrixCloseTo(
      getMultilayerPerceptronActivations(
        matrixFrom([[1, 2, -1]]),
        threeDimPerceptron,
      ).downingOutput,
      matrixFrom([[3, -1, -4]]),
    );
  });

  it("returns only the learned update, not the original residual input", () => {
    const input = matrixFrom([[2, -1]]);

    const result = getMultilayerPerceptronActivations(
      input,
      twoDimPerceptron,
    ).downingOutput;

    const notExpected = matrixFrom([[4.5, 1.5]]);
    const matches = result.values.every(
      (v: number, i: number) => Math.abs(v - notExpected.values[i]!) < 1e-10,
    );
    expect(matches).toBe(false);
  });

  it("runs the same MLP independently on each row vector", () => {
    expectMatrixCloseTo(
      getMultilayerPerceptronActivations(
        matrixFrom([
          [2, -1],
          [-1, 3],
          [0, 0],
        ]),
        twoDimPerceptron,
      ).downingOutput,
      matrixFrom([
        [0.5, 3.5],
        [38.5, 22.5],
        [1.5, -1.5],
      ]),
    );
  });

  it("returns per-row updates without adding them back to the original matrix", () => {
    const input = matrixFrom([
      [2, -1],
      [0, 0],
    ]);

    const result = getMultilayerPerceptronActivations(
      input,
      twoDimPerceptron,
    ).downingOutput;

    const notExpected = matrixFrom([
      [4.5, 1.5],
      [1.5, -1.5],
    ]);
    const matches = result.values.every(
      (v: number, i: number) => Math.abs(v - notExpected.values[i]!) < 1e-10,
    );
    expect(matches).toBe(false);
  });
});
