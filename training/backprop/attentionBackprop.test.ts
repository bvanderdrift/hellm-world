import { describe, it, expect } from "vitest";
import type { AttentionWeights } from "../../model/model-types.ts";
import type { AttentionHeadActivations } from "../../model/activations-types.ts";
import {
  runSelfAttentionHead,
  runSelfAttentionMechanism,
} from "../../transforming/attention.ts";
import {
  attentionBackprop,
  attentionHeadsBackprop,
  getInputVGradients,
  getSoftmaxOutputGradients,
  getSoftmaxInputGradients,
  getQKInputGradients,
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

describe("attentionHeadsBackprop", () => {
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

    // For a single head the flattened activation already matches the
    // AttentionHeadActivations shape, so it can be passed straight through.
    const activations = runSelfAttentionHead(inputQ, inputK, inputV, 1, 2);
    const { inputVGradients } = attentionHeadsBackprop(
      activations,
      outputGradients,
      1,
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

    const activations = runSelfAttentionHead(inputQ, inputK, inputV, 1, 1);
    const { inputQGradients, inputKGradients } = attentionHeadsBackprop(
      activations,
      outputGradients,
      1,
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
      2,
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

// ---------------------------------------------------------------------------
// Unit tests for the individual helpers attentionHeadsBackprop is built from.
// Each uses headDimensionality != contextLength so the flattened per-head
// indexing (which has been the source of every bug here) is actually exercised.
// ---------------------------------------------------------------------------

const emptyMatrix = matrixFrom([[0]]);

// Builds a full AttentionHeadActivations, filling unused fields with dummies so
// each test only has to specify the matrices the function under test reads.
const headActivationsWith = (
  fields: Partial<AttentionHeadActivations>,
): AttentionHeadActivations => ({
  inputK: emptyMatrix,
  inputV: emptyMatrix,
  inputQ: emptyMatrix,
  attentionRelevancyOutput: emptyMatrix,
  softmaxOutput: emptyMatrix,
  output: emptyMatrix,
  ...fields,
});

const expectArrayCloseTo = (
  actual: Float32Array,
  expected: number[],
  precision = 5,
) => {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) =>
    expect(actual[index]).toBeCloseTo(value, precision),
  );
};

describe("getInputVGradients", () => {
  it("weights each value token by its head's softmax score for the query", () => {
    // contextLength 3, headsCount 2, headDim 2, hidden 4; query is vectorIndex 1
    // (sees keys 0 and 1). dL/dV[key, headCol] = softmax(query attends key) * dL/dOutput.
    const outputGradients = matrixFrom([
      [0, 0, 0, 0],
      [10, 20, 30, 40],
      [0, 0, 0, 0],
    ]);
    // softmaxOutput row 1, laid out [head0: keys 0,1,2 | head1: keys 0,1,2].
    const softmaxOutput = matrixFrom([
      [0, 0, 0, 0, 0, 0],
      [0.25, 0.75, 0, 0.5, 0.5, 0],
      [0, 0, 0, 0, 0, 0],
    ]);

    const result = getInputVGradients(
      outputGradients,
      headActivationsWith({ softmaxOutput }),
      1,
      2,
    );

    // key 0: head0 0.25*[10,20]=[2.5,5], head1 0.5*[30,40]=[15,20]
    // key 1: head0 0.75*[10,20]=[7.5,15], head1 0.5*[30,40]=[15,20]
    expectMatrixCloseTo(
      result,
      matrixFrom([
        [2.5, 5, 15, 20],
        [7.5, 15, 15, 20],
        [0, 0, 0, 0],
      ]),
    );
  });
});

describe("getSoftmaxOutputGradients", () => {
  it("dots the query's per-head output gradient with each visible value token", () => {
    // contextLength 3, headsCount 2, headDim 2, hidden 4; query vectorIndex 1.
    const outputGradients = matrixFrom([
      [0, 0, 0, 0],
      [10, 20, 30, 40],
      [0, 0, 0, 0],
    ]);
    const inputV = matrixFrom([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [0, 0, 0, 0],
    ]);

    // softmaxLength = headsCount * (vectorIndex + 1) = 4, headDimensionality = 2.
    const result = getSoftmaxOutputGradients(
      outputGradients,
      headActivationsWith({ inputV }),
      2,
      4,
      1,
      2,
    );

    // head0: [10,20]·[1,2]=50, [10,20]·[5,6]=170; head1: [30,40]·[3,4]=250, [30,40]·[7,8]=530
    expectArrayCloseTo(result, [50, 170, 250, 530]);
  });
});

describe("getSoftmaxInputGradients", () => {
  it("backprops through each head's own softmax slice and places results by head", () => {
    // Uniform logits (all zero) per head -> softmax [0.5,0.5], so for upstream
    // gradient [g0,g1] the result is [0.25(g0-g1), -0.25(g0-g1)].
    // vectorIndex 1, headsCount 2, contextLength 3, softmaxLength 4.
    const attentionRelevancyOutput = matrixFrom([
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
    ]);
    const softmaxOutputGradients = new Float32Array([4, 0, 0, 8]);

    const result = getSoftmaxInputGradients(
      softmaxOutputGradients,
      headActivationsWith({ attentionRelevancyOutput }),
      2,
      4,
      1,
      3,
    );

    // head0 grad [4,0] -> [1,-1]; head1 grad [0,8] -> [-2,2]
    expectArrayCloseTo(result, [1, -1, -2, 2]);
  });
});

describe("getQKInputGradients", () => {
  // contextLength 3, headsCount 2, headDim 2, hidden 4; query vectorIndex 0
  // (sees only key 0), so vectorIndex+1 (1) != contextLength (3).
  // relevancy_h = Q_h · K_h, so dQ_h = dr_h @ K_h and dK_h[l] = dr_h[l] * Q_h.
  const outputGradients = matrixFrom([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ]);
  const inputK = matrixFrom([
    [1, 2, 3, 4],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ]);
  const inputQ = matrixFrom([
    [5, 6, 7, 8],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ]);
  const relevancyVectorGradients = matrixFrom([[2, 3]]); // head0 dr=[2], head1 dr=[3]

  it("computes per-head query gradients", () => {
    const { inputQGradientsMatrix } = getQKInputGradients(
      outputGradients,
      relevancyVectorGradients,
      headActivationsWith({ inputK, inputQ }),
      0,
      2,
    );

    // head0: 2*[1,2]=[2,4]; head1: 3*[3,4]=[9,12]
    expectMatrixCloseTo(inputQGradientsMatrix, matrixFrom([[2, 4, 9, 12]]));
  });

  it("scatters per-head key gradients to the right token rows and head columns", () => {
    const { inputKGradients } = getQKInputGradients(
      outputGradients,
      relevancyVectorGradients,
      headActivationsWith({ inputK, inputQ }),
      0,
      2,
    );

    // Only key 0 contributes. head0: dr 2 * Q[5,6]=[10,12]; head1: dr 3 * Q[7,8]=[21,24].
    expectMatrixCloseTo(
      inputKGradients,
      matrixFrom([
        [10, 12, 21, 24],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
    );
  });
});
