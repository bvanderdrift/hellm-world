import {
  addVectorsInMatrix,
  getMatrixSize,
  multiplyMatrices,
  transpose,
  validateSize,
} from "../shared/matrices.ts";
import type {
  Model,
  MultilayerPerceptronWeights,
} from "../model/model-types.ts";
import { makeZeroVersion } from "../model/model-helpers.ts";
import { calculateLoss } from "./calculateLoss.ts";
import type {
  Activations,
  MultilayerPerceptronActivations,
} from "../model/activations-types.ts";
import { sum } from "../shared/math.ts";

export const backprop = (
  inputTokens: string[],
  correctOutputToken: string,
  weights: Model,
  activations: Activations,
): {
  loss: number;
  gradients: Model;
} => {
  const outputLogits =
    activations.unembeddingsOutputLogits[inputTokens.length - 1];

  if (!outputLogits) {
    throw new Error(
      `Couldn't find output logits in activations. Activations vector count: ${activations.unembeddingsOutputLogits.length}, inputTokensLength: ${inputTokens.length}`,
    );
  }

  validateSize([outputLogits], 1, weights.vocabulary.length);

  return {
    loss: calculateLoss(outputLogits, correctOutputToken, weights.vocabulary),
    gradients: makeZeroVersion(weights),
  };
};

export const backpropMlp = (
  weights: MultilayerPerceptronWeights,
  activations: MultilayerPerceptronActivations,
  /** C x D matrix */
  outputGradients: number[][],
): {
  inputActivationGradients: number[][];
  weightGradients: MultilayerPerceptronWeights;
} => {
  // dL/db = dL/dMLP * dMLP/db = outputGradients * 1
  const downBiasGradient = addVectorsInMatrix(outputGradients);

  const downWeightsGradients = matrixBackprop(
    weights.wDown.weightsMatrix,
    activations.nonLinearToDowning,
    outputGradients,
  );

  // dL/da = dl/dMLP * dMLP/da = outputGradients * wDownT.
  // Simply because of a function `a * x + b`, if we take the derivative for `a` we simply have `x` which in this case is `w_ij` so we just transpose the weight matrix
  const downInputActivationsGradients = multiplyMatrices(
    outputGradients,
    transpose(weights.wDown.weightsMatrix),
  );

  const upOutputGradients = reluBackprop(
    activations.uppingToNonLinear,
    downInputActivationsGradients,
  );

  const upBiasGradient = addVectorsInMatrix(upOutputGradients);

  const upWeightsGradients = matrixBackprop(
    weights.wUp.weightsMatrix,
    activations.normalizedInputToUpping,
    upOutputGradients,
  );

  const inputActivationGradients = multiplyMatrices(
    upOutputGradients,
    transpose(weights.wUp.weightsMatrix),
  );

  return {
    inputActivationGradients,
    weightGradients: {
      wUp: {
        biasVector: upBiasGradient,
        weightsMatrix: upWeightsGradients,
      },
      wDown: {
        biasVector: downBiasGradient,
        weightsMatrix: downWeightsGradients,
      },
    },
  };
};

export const matrixBackprop = (
  weights: number[][],
  inputActivations: number[][],
  outputGradients: number[][],
) => {
  const inputsByDimension = transpose(inputActivations);

  return weights.map((incomingDimensionVector, incomingDimension) =>
    incomingDimensionVector.map((_, outgoingDimension) => {
      return sum(
        inputsByDimension[incomingDimension]!.map(
          (activation, tokenIndex) =>
            activation * outputGradients[tokenIndex]![outgoingDimension]!,
        ),
      );
    }),
  );
};

export const reluBackprop = (
  inputActivations: number[][],
  outputGradients: number[][],
) => {
  const inputSize = getMatrixSize(inputActivations);
  validateSize(
    outputGradients,
    inputSize.vectorCount,
    inputSize.dimensionsCount,
  );

  return outputGradients.map((gradientVector, vectorIndex) =>
    gradientVector.map((gradient, dimensionIndex) => {
      const inputActivation = inputActivations[vectorIndex]![dimensionIndex]!;

      return inputActivation > 0 ? gradient : 0;
    }),
  );
};
