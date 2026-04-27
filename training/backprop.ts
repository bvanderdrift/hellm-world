import {
  addVectorsInMatrix,
  getMatrixSize,
  operateOnMatrices,
  operateOnMatrix,
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
