import { describe, expect, it } from "vitest";
import { multiplyMatrices, transpose } from "../../shared/matrices.ts";
import { matrixBackprop } from "./matrixBackprop.ts";

describe("matrixBackprop", () => {
  it("returns one gradient per weight", () => {
    const weights = [
      [0, 0],
      [0, 0],
      [0, 0],
    ];

    const { weightGradients } = matrixBackprop(
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

    expect(weightGradients).toHaveLength(weights.length);
    expect(weightGradients[0]).toHaveLength(weights[0]!.length);
    expect(weightGradients[1]).toHaveLength(weights[1]!.length);
    expect(weightGradients[2]).toHaveLength(weights[2]!.length);
  });

  it("sums each input activation multiplied by the matching output gradient", () => {
    const { weightGradients } = matrixBackprop(
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

    expect(weightGradients).toEqual([
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

    const { weightGradients } = matrixBackprop(
      [
        [0, 0],
        [0, 0],
        [0, 0],
      ],
      inputActivations,
      outputGradients,
    );

    expect(weightGradients).toEqual(
      multiplyMatrices(transpose(inputActivations), outputGradients),
    );
  });

  it("returns activation gradients by multiplying output gradients by transposed weights", () => {
    const weights = [
      [2, 3],
      [5, 7],
      [11, 13],
    ];
    const outputGradients = [
      [17, 19],
      [23, 29],
    ];

    const { activationGradients } = matrixBackprop(
      weights,
      [
        [0, 0, 0],
        [0, 0, 0],
      ],
      outputGradients,
    );

    expect(activationGradients).toEqual(
      multiplyMatrices(outputGradients, transpose(weights)),
    );
  });
});
