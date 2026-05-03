import { describe, expect, it } from "vitest";
import { multiplyMatrices, normalize, transpose } from "../shared/matrices.ts";
import {
  embeddingsBackprop,
  backpropMlp,
  backpropNormalize,
  matrixBackprop,
  reluBackprop,
} from "./backprop.ts";

describe("embeddingsBackprop", () => {
  it("scales each output gradient by the embedding width square root", () => {
    const gradients = embeddingsBackprop(
      [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      [[1, 2, 3, 4]],
      [1],
    );

    expect(gradients).toEqual([
      [0, 0, 0, 0],
      [2, 4, 6, 8],
    ]);
  });

  it("sums scaled gradients when the same token appears multiple times", () => {
    const gradients = embeddingsBackprop(
      [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      [
        [1, 2, 3, 4],
        [10, 20, 30, 40],
      ],
      [0, 0],
    );

    expect(gradients).toEqual([
      [22, 44, 66, 88],
      [0, 0, 0, 0],
    ]);
  });

  it("keeps separate embedding rows independent", () => {
    const gradients = embeddingsBackprop(
      [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      [
        [1, 2, 3, 4],
        [10, 20, 30, 40],
        [100, 200, 300, 400],
      ],
      [2, 0, 2],
    );

    expect(gradients).toEqual([
      [20, 40, 60, 80],
      [0, 0, 0, 0],
      [202, 404, 606, 808],
    ]);
  });
});

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

describe("backpropNormalize", () => {
  const normalizeObjective = (
    inputActivations: number[][],
    outputGradients: number[][],
  ) => {
    const normalized = normalize(inputActivations);

    return normalized.reduce(
      (total, vector, vectorIndex) =>
        total +
        vector.reduce(
          (vectorTotal, value, valueIndex) =>
            vectorTotal +
            value * outputGradients[vectorIndex]![valueIndex]!,
          0,
        ),
      0,
    );
  };

  it("matches finite differences for one normalized vector", () => {
    const inputActivations = [[1, 2, 4]];
    const outputGradients = [[0.3, -0.7, 1.2]];
    const epsilon = 0.000001;

    const gradients = backpropNormalize(outputGradients, inputActivations);

    for (const [valueIndex] of inputActivations[0]!.entries()) {
      const increasedInput = inputActivations.map((vector) => [...vector]);
      const decreasedInput = inputActivations.map((vector) => [...vector]);

      increasedInput[0]![valueIndex]! += epsilon;
      decreasedInput[0]![valueIndex]! -= epsilon;

      const numericalGradient =
        (normalizeObjective(increasedInput, outputGradients) -
          normalizeObjective(decreasedInput, outputGradients)) /
        (2 * epsilon);

      expect(gradients[0]![valueIndex]).toBeCloseTo(numericalGradient, 5);
    }
  });

  it("treats each vector independently", () => {
    const inputActivations = [
      [1, 2, 4],
      [10, 15, 30],
    ];
    const outputGradients = [
      [0.3, -0.7, 1.2],
      [-0.5, 0.8, 0.1],
    ];
    const epsilon = 0.000001;

    const gradients = backpropNormalize(outputGradients, inputActivations);

    for (const [vectorIndex, vector] of inputActivations.entries()) {
      for (const [valueIndex] of vector.entries()) {
        const increasedInput = inputActivations.map((row) => [...row]);
        const decreasedInput = inputActivations.map((row) => [...row]);

        increasedInput[vectorIndex]![valueIndex]! += epsilon;
        decreasedInput[vectorIndex]![valueIndex]! -= epsilon;

        const numericalGradient =
          (normalizeObjective(increasedInput, outputGradients) -
            normalizeObjective(decreasedInput, outputGradients)) /
          (2 * epsilon);

        expect(gradients[vectorIndex]![valueIndex]).toBeCloseTo(
          numericalGradient,
          5,
        );
      }
    }
  });

  it("keeps gradients finite for constant vectors", () => {
    const gradients = backpropNormalize(
      [[0.3, -0.7, 1.2]],
      [[3, 3, 3]],
    );

    for (const gradient of gradients[0]!) {
      expect(Number.isFinite(gradient)).toBe(true);
    }
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
