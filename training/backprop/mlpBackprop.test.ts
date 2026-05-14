import { describe, expect, it } from "vitest";
import { createMatrix, type Matrix } from "../../shared/matrices.ts";
import { backpropMlp, reluBackprop } from "./mlpBackprop.ts";

const m = (data: number[][]): Matrix => {
  const mat = createMatrix(data.length, data[0]!.length);
  mat.values.set(data.flat());
  return mat;
};

describe("reluBackprop", () => {
  it("passes gradients through positive pre-ReLU activations", () => {
    expect(reluBackprop(m([[1, 2]]), m([[10, 20]]))).toEqual(m([[10, 20]]));
  });

  it("blocks gradients for negative pre-ReLU activations", () => {
    expect(reluBackprop(m([[-1, -2]]), m([[10, 20]]))).toEqual(m([[0, 0]]));
  });

  it("blocks gradients for zero pre-ReLU activations", () => {
    expect(reluBackprop(m([[0]]), m([[10]]))).toEqual(m([[0]]));
  });

  it("applies the ReLU mask element-by-element", () => {
    expect(
      reluBackprop(
        m([
          [-1, 0, 1],
          [2, -3, 4],
        ]),
        m([
          [10, 20, 30],
          [40, 50, 60],
        ]),
      ),
    ).toEqual(
      m([
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
          weightsMatrix: m([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          biasVector: m([[0, 0, 0]]),
        },
        wDown: {
          weightsMatrix: m([
            [1, 2],
            [3, 4],
            [5, 6],
          ]),
          biasVector: m([[0, 0]]),
        },
      },
      {
        normalizedInputToUpping: m([
          [2, 3],
          [4, 5],
        ]),
        uppingToNonLinear: m([
          [1, -1, 2],
          [0, 3, 4],
        ]),
        nonLinearToDowning: m([
          [1, 0, 2],
          [0, 3, 4],
        ]),
        downingOutput: m([
          [0, 0],
          [0, 0],
        ]),
      },
      m([
        [7, 11],
        [13, 17],
      ]),
    );

    expect(gradients).toEqual({
      inputActivationGradients: m([
        [332, 722],
        [715, 1537],
      ]),
      weightGradients: {
        wUp: {
          weightsMatrix: m([
            [58, 428, 870],
            [87, 535, 1138],
          ]),
          biasVector: m([[29, 107, 268]]),
        },
        wDown: {
          weightsMatrix: m([
            [7, 11],
            [39, 51],
            [66, 90],
          ]),
          biasVector: m([[20, 28]]),
        },
      },
    });
  });
});
