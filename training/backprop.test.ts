import { describe, expect, it } from "vitest";
import { multiplyMatrices, transpose } from "../shared/matrices.ts";
import { backpropMlp, matrixBackprop, reluBackprop } from "./backprop.ts";

describe("matrixBackprop", () => {
  it("returns one gradient per weight", () => {
    const weights = [
      [0, 0],
      [0, 0],
      [0, 0],
    ];

    const gradients = matrixBackprop(
      weights,
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      [
        [7, 8],
        [9, 10],
      ],
    );

    expect(gradients).toHaveLength(weights.length);
    expect(gradients[0]).toHaveLength(weights[0]!.length);
    expect(gradients[1]).toHaveLength(weights[1]!.length);
    expect(gradients[2]).toHaveLength(weights[2]!.length);
  });

  it("sums each input activation multiplied by the matching output gradient", () => {
    const gradients = matrixBackprop(
      [
        [0, 0],
        [0, 0],
      ],
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      [
        [10, 20],
        [30, 40],
        [50, 60],
      ],
    );

    expect(gradients).toEqual([
      [1 * 10 + 3 * 30 + 5 * 50, 1 * 20 + 3 * 40 + 5 * 60],
      [2 * 10 + 4 * 30 + 6 * 50, 2 * 20 + 4 * 40 + 6 * 60],
    ]);
  });

  it("matches transposed inputs multiplied by output gradients", () => {
    const inputActivations = [
      [2, -1, 0],
      [3, 4, 5],
    ];
    const outputGradients = [
      [7, 11],
      [13, 17],
    ];

    expect(
      matrixBackprop(
        [
          [0, 0],
          [0, 0],
          [0, 0],
        ],
        inputActivations,
        outputGradients,
      ),
    ).toEqual(multiplyMatrices(transpose(inputActivations), outputGradients));
  });
});

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
