import { describe, expect, it } from "vitest";
import type { AttentionWeights } from "../../model/model-types.ts";
import {
  runSelfAttentionHead,
  runSelfAttentionMechanism,
} from "../../transforming/attention.ts";
import { attentionBackprop, attentionHeadBackprop } from "./attentionBackprop.ts";

const epsilon = 0.000001;

const expectMatrixCloseTo = (
  actual: number[][],
  expected: number[][],
  precision = 5,
) => {
  expect(actual).toHaveLength(expected.length);

  for (const [rowIndex, expectedRow] of expected.entries()) {
    const actualRow = actual[rowIndex];

    expect(actualRow).toHaveLength(expectedRow.length);

    for (const [columnIndex, expectedValue] of expectedRow.entries()) {
      expect(actualRow?.[columnIndex]).toBeCloseTo(expectedValue, precision);
    }
  }
};

const perturbMatrix = (
  matrix: number[][],
  perturbRow: number,
  perturbColumn: number,
  delta: number,
) =>
  matrix.map((row, rowIndex) =>
    row.map((value, columnIndex) =>
      rowIndex === perturbRow && columnIndex === perturbColumn
        ? value + delta
        : value,
    ),
  );

const finiteDifferenceMatrixEntry = (
  matrix: number[][],
  rowIndex: number,
  columnIndex: number,
  objective: (perturbedMatrix: number[][]) => number,
) => {
  const increased = perturbMatrix(matrix, rowIndex, columnIndex, epsilon);
  const decreased = perturbMatrix(matrix, rowIndex, columnIndex, -epsilon);

  return (objective(increased) - objective(decreased)) / (2 * epsilon);
};

const finiteDifferenceMatrix = (
  matrix: number[][],
  objective: (perturbedMatrix: number[][]) => number,
) =>
  matrix.map((row, rowIndex) =>
    row.map((_, columnIndex) =>
      finiteDifferenceMatrixEntry(matrix, rowIndex, columnIndex, objective),
    ),
  );

const matrixObjective = (output: number[][], outputGradients: number[][]) =>
  output.reduce(
    (matrixSum, outputVector, vectorIndex) =>
      matrixSum +
      outputVector.reduce(
        (vectorSum, outputValue, dimensionIndex) =>
          vectorSum +
          outputValue * outputGradients[vectorIndex]![dimensionIndex]!,
        0,
      ),
    0,
  );

const attentionHeadObjective = (
  inputQ: number[][],
  inputK: number[][],
  inputV: number[][],
  outputGradients: number[][],
) =>
  matrixObjective(
    runSelfAttentionHead(inputQ, inputK, inputV).output,
    outputGradients,
  );

const attentionObjective = (
  input: number[][],
  headsCount: number,
  weights: AttentionWeights,
  outputGradients: number[][],
) =>
  matrixObjective(
    runSelfAttentionMechanism(input, headsCount, weights).output,
    outputGradients,
  );

describe("attentionHeadBackprop", () => {
  it("matches finite differences for value gradients through the causal lookback window", () => {
    const inputQ = [
      [0.2, -0.5],
      [1.1, 0.3],
      [-0.7, 0.8],
    ];
    const inputK = [
      [0.4, 0.1],
      [-0.3, 0.9],
      [0.2, -0.6],
    ];
    const inputV = [
      [0.5, -0.2],
      [1.3, 0.7],
      [-0.4, 0.9],
    ];
    const outputGradients = [
      [0.6, -0.1],
      [-0.2, 0.8],
      [1.1, -0.5],
    ];

    const activations = runSelfAttentionHead(inputQ, inputK, inputV);
    const { inputVGradients } = attentionHeadBackprop(
      activations,
      outputGradients,
    );

    const numericalVGradients = finiteDifferenceMatrix(inputV, (perturbedV) =>
      attentionHeadObjective(inputQ, inputK, perturbedV, outputGradients),
    );

    expectMatrixCloseTo(inputVGradients, numericalVGradients);
  });

  it("matches finite differences for query and key gradients after value mixing", () => {
    const inputQ = [[0.4], [0.7]];
    const inputK = [[-0.2], [0.9]];
    const inputV = [[1.5], [-0.4]];
    const outputGradients = [[0.3], [1.2]];

    const activations = runSelfAttentionHead(inputQ, inputK, inputV);
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

    expectMatrixCloseTo(inputQGradients, numericalQGradients);
    expectMatrixCloseTo(inputKGradients, numericalKGradients);
  });
});

describe("attentionBackprop", () => {
  it("matches finite differences for output and value weights across split heads", () => {
    const input = [
      [0.2, -0.4, 0.6, 1.1],
      [-0.3, 0.8, -0.5, 0.7],
      [1.0, -0.2, 0.3, -0.9],
    ];
    const weights: AttentionWeights = {
      Q: [
        [0.1, -0.2, 0.3, 0.4],
        [-0.5, 0.2, 0.1, -0.3],
        [0.4, 0.7, -0.6, 0.2],
        [0.3, -0.1, 0.5, -0.4],
      ],
      K: [
        [-0.2, 0.5, 0.4, -0.1],
        [0.6, -0.3, 0.2, 0.1],
        [0.1, 0.4, -0.5, 0.7],
        [-0.4, 0.3, 0.6, -0.2],
      ],
      V: [
        [0.7, -0.1, 0.2, -0.6],
        [-0.3, 0.8, -0.4, 0.5],
        [0.2, 0.6, 0.1, -0.7],
        [-0.5, 0.4, 0.9, 0.3],
      ],
      out: [
        [0.3, -0.6, 0.2, 0.1],
        [-0.4, 0.7, -0.5, 0.2],
        [0.6, 0.1, -0.3, 0.8],
        [0.2, -0.5, 0.4, -0.1],
      ],
    };
    const outputGradients = [
      [0.5, -0.7, 0.2, 0.9],
      [-0.4, 0.3, 0.8, -0.6],
      [0.1, 0.6, -0.2, 0.4],
    ];

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

    expect(inputGradients).toHaveLength(input.length);
    expect(inputGradients[0]).toHaveLength(input[0]!.length);
    expectMatrixCloseTo(weightGradients.out, numericalOutGradients);
    expectMatrixCloseTo(weightGradients.V, numericalVGradients);
  });
});
