import { describe, expect, it } from "vitest";
import {
  normalize,
  getFlatIndex,
  type Matrix,
} from "../../shared/matrices.ts";
import { backpropNormalize } from "./normalizeBackprop.ts";
import { matrixFrom } from "../../testing/testing-utils.ts";
import { FINITE_DIFFERENCE_EPSILON, FINITE_DIFFERENCE_PRECISION } from "../../testing/constants.ts";

describe("backpropNormalize", () => {
  const normalizeObjective = (
    inputActivations: Matrix,
    outputGradients: Matrix,
  ) => {
    const normalized = normalize(inputActivations);

    let total = 0;
    for (let i = 0; i < normalized.vectors; i++) {
      for (let j = 0; j < normalized.dimensions; j++) {
        const idx = getFlatIndex(i, j, normalized.dimensions);
        total += normalized.values[idx]! * outputGradients.values[idx]!;
      }
    }
    return total;
  };

  it("matches finite differences for one normalized vector", () => {
    const inputActivations = matrixFrom([[1, 2, 4]]);
    const outputGradients = matrixFrom([[0.3, -0.7, 1.2]]);

    const gradients = backpropNormalize(outputGradients, inputActivations);

    for (let valueIndex = 0; valueIndex < inputActivations.dimensions; valueIndex++) {
      const increased = matrixFrom([[1, 2, 4]]);
      const decreased = matrixFrom([[1, 2, 4]]);

      increased.values[valueIndex]! += FINITE_DIFFERENCE_EPSILON;
      decreased.values[valueIndex]! -= FINITE_DIFFERENCE_EPSILON;

      const numericalGradient =
        (normalizeObjective(increased, outputGradients) -
          normalizeObjective(decreased, outputGradients)) /
        (2 * FINITE_DIFFERENCE_EPSILON);

      expect(
        gradients.values[getFlatIndex(0, valueIndex, gradients.dimensions)],
      ).toBeCloseTo(numericalGradient, FINITE_DIFFERENCE_PRECISION);
    }
  });

  it("treats each vector independently", () => {
    const inputActivations = matrixFrom([
      [1, 2, 4],
      [10, 15, 30],
    ]);
    const outputGradients = matrixFrom([
      [0.3, -0.7, 1.2],
      [-0.5, 0.8, 0.1],
    ]);
    const gradients = backpropNormalize(outputGradients, inputActivations);

    for (let vectorIndex = 0; vectorIndex < inputActivations.vectors; vectorIndex++) {
      for (let valueIndex = 0; valueIndex < inputActivations.dimensions; valueIndex++) {
        const increased = matrixFrom([
          [1, 2, 4],
          [10, 15, 30],
        ]);
        const decreased = matrixFrom([
          [1, 2, 4],
          [10, 15, 30],
        ]);

        const idx = getFlatIndex(vectorIndex, valueIndex, inputActivations.dimensions);
        increased.values[idx]! += FINITE_DIFFERENCE_EPSILON;
        decreased.values[idx]! -= FINITE_DIFFERENCE_EPSILON;

        const numericalGradient =
          (normalizeObjective(increased, outputGradients) -
            normalizeObjective(decreased, outputGradients)) /
          (2 * FINITE_DIFFERENCE_EPSILON);

        expect(
          gradients.values[getFlatIndex(vectorIndex, valueIndex, gradients.dimensions)],
        ).toBeCloseTo(numericalGradient, FINITE_DIFFERENCE_PRECISION);
      }
    }
  });

  it("keeps gradients finite for constant vectors", () => {
    const gradients = backpropNormalize(
      matrixFrom([[0.3, -0.7, 1.2]]),
      matrixFrom([[3, 3, 3]]),
    );

    for (let j = 0; j < gradients.dimensions; j++) {
      expect(Number.isFinite(gradients.values[j]!)).toBe(true);
    }
  });
});
