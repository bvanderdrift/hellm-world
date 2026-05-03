import { describe, expect, it } from "vitest";
import { backpropMlp, reluBackprop } from "./mlpBackprop.ts";

describe("reluBackprop", () => {
  it("passes gradients through positive pre-ReLU activations", () => {
    expect(reluBackprop([[1, 2]], [[10, 20]])).toEqual([[10, 20]]);
  });

  it("blocks gradients for negative pre-ReLU activations", () => {
    expect(reluBackprop([[-1, -2]], [[10, 20]])).toEqual([[0, 0]]);
  });

  it("blocks gradients for zero pre-ReLU activations", () => {
    expect(reluBackprop([[0]], [[10]])).toEqual([[0]]);
  });

  it("applies the ReLU mask element-by-element", () => {
    expect(
      reluBackprop(
        [
          [-1, 0, 1],
          [2, -3, 4],
        ],
        [
          [10, 20, 30],
          [40, 50, 60],
        ],
      ),
    ).toEqual([
      [0, 0, 30],
      [40, 0, 60],
    ]);
  });

  it("throws when output gradients do not match the input activation shape", () => {
    expect(() => reluBackprop([[1, 2]], [[10]])).toThrow(
      "unexpected vector depth",
    );
  });
});

describe("backpropMlp", () => {
  it("calculates MLP weight, bias, and input activation gradients", () => {
    const gradients = backpropMlp(
      {
        wUp: {
          weightsMatrix: [
            [1, 2, 3],
            [4, 5, 6],
          ],
          biasVector: [0, 0, 0],
        },
        wDown: {
          weightsMatrix: [
            [1, 2],
            [3, 4],
            [5, 6],
          ],
          biasVector: [0, 0],
        },
      },
      {
        normalizedInputToUpping: [
          [2, 3],
          [4, 5],
        ],
        uppingToNonLinear: [
          [1, -1, 2],
          [0, 3, 4],
        ],
        nonLinearToDowning: [
          [1, 0, 2],
          [0, 3, 4],
        ],
        downingOutput: [
          [0, 0],
          [0, 0],
        ],
      },
      [
        [7, 11],
        [13, 17],
      ],
    );

    expect(gradients).toEqual({
      inputActivationGradients: [
        [332, 722],
        [715, 1537],
      ],
      weightGradients: {
        wUp: {
          weightsMatrix: [
            [58, 428, 870],
            [87, 535, 1138],
          ],
          biasVector: [29, 107, 268],
        },
        wDown: {
          weightsMatrix: [
            [7, 11],
            [39, 51],
            [66, 90],
          ],
          biasVector: [20, 28],
        },
      },
    });
  });
});
