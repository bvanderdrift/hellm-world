import { describe, expect, it } from "vitest";
import { normalize } from "../../shared/matrices.ts";
import { backpropNormalize } from "./normalizeBackprop.ts";

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
            vectorTotal + value * outputGradients[vectorIndex]![valueIndex]!,
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
    const gradients = backpropNormalize([[0.3, -0.7, 1.2]], [[3, 3, 3]]);

    for (const gradient of gradients[0]!) {
      expect(Number.isFinite(gradient)).toBe(true);
    }
  });
});
