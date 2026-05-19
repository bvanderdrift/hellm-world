import { describe, it } from "vitest";
import type { AttentionWeights } from "../../model/model-types.ts";
import type { AttentionHeadActivations } from "../../model/activations-types.ts";
import {
  runSelfAttentionHead,
  runSelfAttentionMechanism,
} from "../../transforming/attention.ts";
import {
  attentionBackprop,
  attentionHeadBackprop,
} from "./attentionBackprop.ts";
import {
  createMatrix,
  getFlatIndex,
  type Matrix,
} from "../../shared/matrices.ts";
import {
  FINITE_DIFFERENCE_EPSILON,
  FINITE_DIFFERENCE_PRECISION,
} from "../../testing/constants.ts";
import {
  matrixFrom,
  expectMatrixCloseTo,
} from "../../testing/testing-utils.ts";

const perturbMatrix = (
  matrix: Matrix,
  perturbRow: number,
  perturbColumn: number,
  delta: number,
): Matrix => {
  const clone = createMatrix(matrix.vectors, matrix.dimensions);
  clone.values.set(matrix.values);
  clone.values[getFlatIndex(perturbRow, perturbColumn, matrix.dimensions)]! +=
    delta;
  return clone;
};

const finiteDifferenceMatrixEntry = (
  matrix: Matrix,
  rowIndex: number,
  columnIndex: number,
  objective: (perturbedMatrix: Matrix) => number,
) => {
  const increased = perturbMatrix(matrix, rowIndex, columnIndex, FINITE_DIFFERENCE_EPSILON);
  const decreased = perturbMatrix(matrix, rowIndex, columnIndex, -FINITE_DIFFERENCE_EPSILON);

  return (objective(increased) - objective(decreased)) / (2 * FINITE_DIFFERENCE_EPSILON);
};

const finiteDifferenceMatrix = (
  matrix: Matrix,
  objective: (perturbedMatrix: Matrix) => number,
): Matrix => {
  const result = createMatrix(matrix.vectors, matrix.dimensions);
  for (let i = 0; i < matrix.vectors; i++) {
    for (let j = 0; j < matrix.dimensions; j++) {
      result.values[getFlatIndex(i, j, matrix.dimensions)] =
        finiteDifferenceMatrixEntry(matrix, i, j, objective);
    }
  }
  return result;
};

const matrixObjective = (output: Matrix, outputGradients: Matrix) => {
  let total = 0;
  for (let i = 0; i < output.vectors; i++) {
    for (let j = 0; j < output.dimensions; j++) {
      const idx = getFlatIndex(i, j, output.dimensions);
      total += output.values[idx]! * outputGradients.values[idx]!;
    }
  }
  return total;
};

const attentionHeadObjective = (
  inputQ: Matrix,
  inputK: Matrix,
  inputV: Matrix,
  outputGradients: Matrix,
) =>
  matrixObjective(
    runSelfAttentionHead(inputQ, inputK, inputV, 1, inputQ.dimensions).output,
    outputGradients,
  );

const attentionObjective = (
  input: Matrix,
  headsCount: number,
  weights: AttentionWeights,
  outputGradients: Matrix,
) =>
  matrixObjective(
    runSelfAttentionMechanism(input, headsCount, weights).output,
    outputGradients,
  );

describe("attentionHeadBackprop", () => {
  it("matches finite differences for value gradients through the causal lookback window", () => {
    const inputQ = matrixFrom([
      [0.2, -0.5],
      [1.1, 0.3],
      [-0.7, 0.8],
    ]);
    const inputK = matrixFrom([
      [0.4, 0.1],
      [-0.3, 0.9],
      [0.2, -0.6],
    ]);
    const inputV = matrixFrom([
      [0.5, -0.2],
      [1.3, 0.7],
      [-0.4, 0.9],
    ]);
    const outputGradients = matrixFrom([
      [0.6, -0.1],
      [-0.2, 0.8],
      [1.1, -0.5],
    ]);

    const rawActivations = runSelfAttentionHead(inputQ, inputK, inputV, 1, 2);
    const activations: AttentionHeadActivations = {
      inputK: rawActivations.inputK,
      inputQ: rawActivations.inputQ,
      inputV: rawActivations.inputV,
      attentionRelevancyOutput: rawActivations.attentionRelevancyOutput[0]!,
      softmaxOutput: rawActivations.softmaxOutput[0]!,
      output: rawActivations.output,
    };
    const { inputVGradients } = attentionHeadBackprop(
      activations,
      outputGradients,
    );

    const numericalVGradients = finiteDifferenceMatrix(inputV, (perturbedV) =>
      attentionHeadObjective(inputQ, inputK, perturbedV, outputGradients),
    );

    expectMatrixCloseTo(inputVGradients, numericalVGradients, FINITE_DIFFERENCE_PRECISION);
  });

  it("matches finite differences for query and key gradients after value mixing", () => {
    const inputQ = matrixFrom([[0.4], [0.7]]);
    const inputK = matrixFrom([[-0.2], [0.9]]);
    const inputV = matrixFrom([[1.5], [-0.4]]);
    const outputGradients = matrixFrom([[0.3], [1.2]]);

    const rawActivations = runSelfAttentionHead(inputQ, inputK, inputV, 1, 1);
    const activations: AttentionHeadActivations = {
      inputK: rawActivations.inputK,
      inputQ: rawActivations.inputQ,
      inputV: rawActivations.inputV,
      attentionRelevancyOutput: rawActivations.attentionRelevancyOutput[0]!,
      softmaxOutput: rawActivations.softmaxOutput[0]!,
      output: rawActivations.output,
    };
    const { inputQGradients, inputKGradients } = attentionHeadBackprop(
      activations,
      outputGradients,
    );

    const numericalQGradients = finiteDifferenceMatrix(inputQ, (perturbedQ) =>
      attentionHeadObjective(perturbedQ, inputK, inputV, outputGradients),
    );
    const numericalKGradients = finiteDifferenceMatrix(inputK, (perturbedK) =>
      attentionHeadObjective(inputQ, perturbedK, inputV, outputGradients),
    );

    expectMatrixCloseTo(inputQGradients, numericalQGradients, FINITE_DIFFERENCE_PRECISION);
    expectMatrixCloseTo(inputKGradients, numericalKGradients, FINITE_DIFFERENCE_PRECISION);
  });
});

describe("attentionBackprop", () => {
  it("matches finite differences for weights and inputs across split heads", () => {
    const input = matrixFrom([
      [0.2, -0.4, 0.6, 1.1],
      [-0.3, 0.8, -0.5, 0.7],
      [1.0, -0.2, 0.3, -0.9],
    ]);
    const weights: AttentionWeights = {
      Q: matrixFrom([
        [0.1, -0.2, 0.3, 0.4],
        [-0.5, 0.2, 0.1, -0.3],
        [0.4, 0.7, -0.6, 0.2],
        [0.3, -0.1, 0.5, -0.4],
      ]),
      K: matrixFrom([
        [-0.2, 0.5, 0.4, -0.1],
        [0.6, -0.3, 0.2, 0.1],
        [0.1, 0.4, -0.5, 0.7],
        [-0.4, 0.3, 0.6, -0.2],
      ]),
      V: matrixFrom([
        [0.7, -0.1, 0.2, -0.6],
        [-0.3, 0.8, -0.4, 0.5],
        [0.2, 0.6, 0.1, -0.7],
        [-0.5, 0.4, 0.9, 0.3],
      ]),
      out: matrixFrom([
        [0.3, -0.6, 0.2, 0.1],
        [-0.4, 0.7, -0.5, 0.2],
        [0.6, 0.1, -0.3, 0.8],
        [0.2, -0.5, 0.4, -0.1],
      ]),
    };
    const outputGradients = matrixFrom([
      [0.5, -0.7, 0.2, 0.9],
      [-0.4, 0.3, 0.8, -0.6],
      [0.1, 0.6, -0.2, 0.4],
    ]);

    const activations = runSelfAttentionMechanism(input, 2, weights);
    const { inputGradients, weightGradients } = attentionBackprop(
      weights,
      outputGradients,
      activations,
    );

    const numericalOutGradients = finiteDifferenceMatrix(
      weights.out,
      (perturbedOut) =>
        attentionObjective(
          input,
          2,
          { ...weights, out: perturbedOut },
          outputGradients,
        ),
    );
    const numericalVGradients = finiteDifferenceMatrix(
      weights.V,
      (perturbedV) =>
        attentionObjective(
          input,
          2,
          { ...weights, V: perturbedV },
          outputGradients,
        ),
    );
    const numericalQGradients = finiteDifferenceMatrix(
      weights.Q,
      (perturbedQ) =>
        attentionObjective(
          input,
          2,
          { ...weights, Q: perturbedQ },
          outputGradients,
        ),
    );
    const numericalKGradients = finiteDifferenceMatrix(
      weights.K,
      (perturbedK) =>
        attentionObjective(
          input,
          2,
          { ...weights, K: perturbedK },
          outputGradients,
        ),
    );
    const numericalInputGradients = finiteDifferenceMatrix(
      input,
      (perturbedInput) =>
        attentionObjective(perturbedInput, 2, weights, outputGradients),
    );

    const multiHeadPrecision = FINITE_DIFFERENCE_PRECISION - 1;
    expectMatrixCloseTo(inputGradients, numericalInputGradients, multiHeadPrecision);
    expectMatrixCloseTo(weightGradients.out, numericalOutGradients, multiHeadPrecision);
    expectMatrixCloseTo(weightGradients.V, numericalVGradients, multiHeadPrecision);
    expectMatrixCloseTo(weightGradients.Q, numericalQGradients, multiHeadPrecision);
    expectMatrixCloseTo(weightGradients.K, numericalKGradients, multiHeadPrecision);
  });
});
