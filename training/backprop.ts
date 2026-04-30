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
  TransformerWeights,
  Weights,
} from "../model/model-types.ts";
import { makeZeroVersion } from "../model/model-helpers.ts";
import { calculateLoss } from "./calculateLoss.ts";
import type {
  Activations,
  MultilayerPerceptronActivations,
  TransformerActivations,
} from "../model/activations-types.ts";
import { calculateStandardDeviation, softmax, sum } from "../shared/math.ts";

export const backprop = (
  inputTokens: string[],
  weights: Model,
  activations: Activations,
  correctTokenIndex: number,
): {
  loss: number;
  gradients: Weights;
} => {
  const outputLogits =
    activations.unembeddingsOutputLogits[inputTokens.length - 1];

  if (!outputLogits) {
    throw new Error(
      `Couldn't find output logits in activations. Activations vector count: ${activations.unembeddingsOutputLogits.length}, inputTokensLength: ${inputTokens.length}`,
    );
  }

  const outputProbabilities = softmax(outputLogits);

  const unembeddingsInputActivationsGradients = probabilityOutputBackprop(
    activations.unembeddingsOutputLogits,
    outputProbabilities,
    inputTokens.length,
    correctTokenIndex,
  );

  const {
    weightGradients: unembeddingWeightGradients,
    activationGradients: unembeddingInputActivationGradients,
  } = matrixBackprop(
    weights.unembeddings,
    activations.normalizerToUnembeddings,
    unembeddingsInputActivationsGradients,
  );

  const preUnembeddingNormalizationGradients = backpropNormalize(
    unembeddingInputActivationGradients,
    activations.transformersToNormalizer,
  );

  const {
    transformerGradients,
    inputActivationGradients: transformerInputActivationGradients,
  } = transformersBackprop(
    preUnembeddingNormalizationGradients,
    weights.transformers,
    activations.transformerActivations,
  );

  /**
   * No need to backprop for positional encoding
   * Since the transformer input is h_i = z_i + a_i where z_i is output of embedding matrix, and a_i is output of positional encoding
   * we know dL/dz_i = dL/h_i * 1 + 0 since d(z_i)/dz_i = 1 and d(a_i)/dz_i = 0
   *
   * So dL/dz_i is just the direct dL/h_i so just the transformer input gradients.
   *
   * We don't care about dL/da_i (which is also dL/h_i) since a_i a non-trainable algorithmic output
   */

  // TODO: embeddings gradient. Embeddings aren't just a matrix multiplication, but rather a token lookup table
  const embeddingWeightsGradients: number[][] = [];

  return {
    loss: calculateLoss(outputLogits, correctTokenIndex, weights.vocabulary),
    gradients: {
      unembeddings: unembeddingWeightGradients,
      transformers: transformerGradients,
      embeddings: embeddingWeightsGradients,
    },
  };
};

export const probabilityOutputBackprop = (
  unembeddingsOutputLogits: number[][],
  outputProbabilitiesVector: number[],
  contextLength: number,
  correctTokenIndex: number,
) =>
  unembeddingsOutputLogits.map((outputVector, inputTokenIndex) => {
    if (inputTokenIndex !== contextLength - 1) {
      // We're not training on these, so no gradients
      return new Array(outputVector.length).fill(0);
    }

    return outputVector.map((_, vocabIndex) => {
      const isCorrectToken = vocabIndex === correctTokenIndex;
      const actualProbability = outputProbabilitiesVector[vocabIndex]!;
      const wantedProbability = isCorrectToken ? 1 : 0;

      /**
       * This seems to be too easy and not like it's a derivative
       * But it is. This is dL/dz_i
       *
       * Check this out:
       *
       * L = CEL_i = -log(p_i)
       * p_i = exp(z_i) / sum_j exp(z_j)
       * So L = -log(exp(z_i) / sum_j exp(z_j)) = -z_i + log(sum_j exp(z_j))
       * So d(-z_i)/dz_i = -1 and d(log(n))/dn = 1 / n and d(log(sum_j exp(z_j)))/dz_i = (1 / (sum_j exp(z_j)) * exp(z_i) = exp(z_i) / (sum_j exp(z_j) = p_i
       * Wowow; so dL/dz_i = d(-z_i)/dz_i + that sum part = -1 + p_i = p_i - 1 for i = k and 0 + p_j for non-i values (since d(z_j)/dz_i = 0)
       */
      return actualProbability - wantedProbability;
    });
  });

export const transformersBackprop = (
  outputGradients: number[][],
  weights: TransformerWeights[],
  activations: TransformerActivations[],
): {
  transformerGradients: TransformerWeights[];
  inputActivationGradients: number[][];
} => {
  if (weights.length !== activations.length) {
    throw new Error(
      `Transformer weights count ${weights.length} does not equal transformers activations count ${activations.length}`,
    );
  }

  // Spread b/c reverse mutates in-place
  const reversedActivations = [...activations].reverse();

  let lastOutputGradients = outputGradients;
  let transformerGradients: TransformerWeights[] = [];

  for (let index = 0; index < reversedActivations.length; index++) {
    const transformerActivations = reversedActivations[index]!;
    const transformerWeights = weights[weights.length - index - 1]!;

    const updateActivationsGradients: number[][] = [
      /** TODO */
    ];

    const { inputActivationGradients, weightGradients } = backpropMlp(
      transformerWeights.multilayerPerceptron,
      transformerActivations.mlp,
      updateActivationsGradients,
    );

    const preNormalizationInputGradients = backpropNormalize(
      inputActivationGradients,
      transformerActivations.mlp.normalizedInputToUpping,
    );

    // TODO: Attention Backprop
  }

  return {
    inputActivationGradients: lastOutputGradients,
    transformerGradients,
  };
};

export const backpropNormalize = (
  outputGradients: number[][],
  inputActivations: number[][],
): number[][] => {
  return inputActivations.map((vector) =>
    vector.map((value) => {
      // TODO implement norm derivative in respect to value (h_vi)
      return 0;
    }),
  );
};

/**
 * mean = sum(h_0...j) / j
 * dmean/dh_i = 1 / j
 */
const meanDerivative = (values: number[]) => {
  return 1 / values.length;
};

/**
 * std(h_v0...j) = Math.sqrt(mean(Math.pow(h_vj - mean(h_v0...j)))
 * = Math.sqrt(variance(h_v0...j))
 * and variance(h_v0...j) = 1/j * sum(Math.pow(h_vj - mean(h_v0...j)))
 * dstd(h_v0...j)/dh_vi = d/dvariance * dvariance/h_vj
 *
 * d/dvariance = (1/ (2 * std))
 *
 * dvariance/dh_vi = 1/j * 2 * (h_vi - mean(h_v0...j))
 *
 * So dstd/dh_i = (1/ (2 * std)) * 2/j * (h_vi - mean(h_v0...j))
 *  = 1 / (j * std) * (h_vi - mean)
 *  = (h_vh - mean) / (j * std)
 */
const standardDeviationDerivative = (
  values: number[],
  derivativeIndex: number,
) => {
  const { average, standardDeviation } = calculateStandardDeviation(values);

  const derivative = values[derivativeIndex]!;

  return (
    (derivative - average) /
    (values.length *
      (standardDeviation +
        // To stabalize againt 0-division
        Number.EPSILON))
  );
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

  const {
    weightGradients: downWeightsGradients,
    activationGradients: downInputActivationsGradients,
  } = matrixBackprop(
    weights.wDown.weightsMatrix,
    activations.nonLinearToDowning,
    outputGradients,
  );

  const upOutputGradients = reluBackprop(
    activations.uppingToNonLinear,
    downInputActivationsGradients,
  );

  const upBiasGradient = addVectorsInMatrix(upOutputGradients);

  const {
    weightGradients: upWeightsGradients,
    activationGradients: inputActivationGradients,
  } = matrixBackprop(
    weights.wUp.weightsMatrix,
    activations.normalizedInputToUpping,
    upOutputGradients,
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

  const weightGradients = weights.map(
    (incomingDimensionVector, incomingDimension) =>
      incomingDimensionVector.map((_, outgoingDimension) => {
        return sum(
          inputsByDimension[incomingDimension]!.map(
            (activation, tokenIndex) =>
              activation * outputGradients[tokenIndex]![outgoingDimension]!,
          ),
        );
      }),
  );

  // dL/da = dl/dMLP * dMLP/da = outputGradients * wDownT.
  // Simply because of a function `a * x + b`, if we take the derivative for `a` we simply have `x` which in this case is `w_ij` so we just transpose the weight matrix
  const activationGradients = multiplyMatrices(
    outputGradients,
    transpose(weights),
  );

  return {
    weightGradients,
    activationGradients,
  };
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
