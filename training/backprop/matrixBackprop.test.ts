import { describe, expect, it } from "vitest";
import {
  createMatrix,
  type Matrix,
  multiplyMatrices,
  transpose,
} from "../../shared/matrices.ts";
import { matrixBackprop } from "./matrixBackprop.ts";

const m = (data: number[][]): Matrix => {
  const mat = createMatrix(data.length, data[0]!.length);
  mat.values.set(data.flat());
  return mat;
};

describe("matrixBackprop", () => {
  it("returns one gradient per weight", () => {
    const weights = m([
      [0, 0],
      [0, 0],
      [0, 0],
    ]);

    const { weightGradients } = matrixBackprop(
      weights,
      m([
        [1, 2, 3],
        [4, 5, 6],
      ]),
      m([
        [7, 8],
        [9, 10],
      ]),
    );

    expect(weightGradients.vectors).toBe(weights.vectors);
    expect(weightGradients.dimensions).toBe(weights.dimensions);
  });

  it("sums each input activation multiplied by the matching output gradient", () => {
    const { weightGradients } = matrixBackprop(
      m([
        [0, 0],
        [0, 0],
      ]),
      m([
        [1, 2],
        [3, 4],
        [5, 6],
      ]),
      m([
        [10, 20],
        [30, 40],
        [50, 60],
      ]),
    );

    expect(weightGradients).toEqual(
      m([
        [1 * 10 + 3 * 30 + 5 * 50, 1 * 20 + 3 * 40 + 5 * 60],
        [2 * 10 + 4 * 30 + 6 * 50, 2 * 20 + 4 * 40 + 6 * 60],
      ]),
    );
  });

  it("matches transposed inputs multiplied by output gradients", () => {
    const inputActivations = m([
      [2, -1, 0],
      [3, 4, 5],
    ]);
    const outputGradients = m([
      [7, 11],
      [13, 17],
    ]);

    const { weightGradients } = matrixBackprop(
      m([
        [0, 0],
        [0, 0],
        [0, 0],
      ]),
      inputActivations,
      outputGradients,
    );

    expect(weightGradients).toEqual(
      multiplyMatrices(transpose(inputActivations), outputGradients),
    );
  });

  it("returns activation gradients by multiplying output gradients by transposed weights", () => {
    const weights = m([
      [2, 3],
      [5, 7],
      [11, 13],
    ]);
    const outputGradients = m([
      [17, 19],
      [23, 29],
    ]);

    const { activationGradients } = matrixBackprop(
      weights,
      m([
        [0, 0, 0],
        [0, 0, 0],
      ]),
      outputGradients,
    );

    expect(activationGradients).toEqual(
      multiplyMatrices(outputGradients, transpose(weights)),
    );
  });
});
