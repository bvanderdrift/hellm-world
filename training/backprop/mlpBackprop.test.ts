import { describe, expect, it } from "vitest";
import { matrixFrom } from "../../testing/testing-utils.ts";
import { backpropMlp, reluBackprop } from "./mlpBackprop.ts";

describe("reluBackprop", () => {
  it("passes gradients through positive pre-ReLU activations", () => {
    expect(reluBackprop(matrixFrom([[1, 2]]), matrixFrom([[10, 20]]))).toEqual(matrixFrom([[10, 20]]));
  });

  it("blocks gradients for negative pre-ReLU activations", () => {
    expect(reluBackprop(matrixFrom([[-1, -2]]), matrixFrom([[10, 20]]))).toEqual(matrixFrom([[0, 0]]));
  });

  it("blocks gradients for zero pre-ReLU activations", () => {
    expect(reluBackprop(matrixFrom([[0]]), matrixFrom([[10]]))).toEqual(matrixFrom([[0]]));
  });

  it("applies the ReLU mask element-by-element", () => {
    expect(
      reluBackprop(
        matrixFrom([
          [-1, 0, 1],
          [2, -3, 4],
        ]),
        matrixFrom([
          [10, 20, 30],
          [40, 50, 60],
        ]),
      ),
    ).toEqual(
      matrixFrom([
        [0, 0, 30],
        [40, 0, 60],
      ]),
    );
  });
});

describe("backpropMlp", () => {
  it("calculates MLP weight, bias, and input activation gradients", () => {
    const gradients = backpropMlp(
      {
        wUp: {
          weightsMatrix: matrixFrom([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          biasVector: matrixFrom([[0, 0, 0]]),
        },
        wDown: {
          weightsMatrix: matrixFrom([
            [1, 2],
            [3, 4],
            [5, 6],
          ]),
          biasVector: matrixFrom([[0, 0]]),
        },
      },
      {
        normalizedInputToUpping: matrixFrom([
          [2, 3],
          [4, 5],
        ]),
        uppingToNonLinear: matrixFrom([
          [1, -1, 2],
          [0, 3, 4],
        ]),
        nonLinearToDowning: matrixFrom([
          [1, 0, 2],
          [0, 3, 4],
        ]),
        downingOutput: matrixFrom([
          [0, 0],
          [0, 0],
        ]),
      },
      matrixFrom([
        [7, 11],
        [13, 17],
      ]),
    );

    expect(gradients).toEqual({
      inputActivationGradients: matrixFrom([
        [332, 722],
        [715, 1537],
      ]),
      weightGradients: {
        wUp: {
          weightsMatrix: matrixFrom([
            [58, 428, 870],
            [87, 535, 1138],
          ]),
          biasVector: matrixFrom([[29, 107, 268]]),
        },
        wDown: {
          weightsMatrix: matrixFrom([
            [7, 11],
            [39, 51],
            [66, 90],
          ]),
          biasVector: matrixFrom([[20, 28]]),
        },
      },
    });
  });
});
