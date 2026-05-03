import {
  addVectors,
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

  const unembeddingsOutputActivationsGradients = probabilityOutputBackprop(
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
    unembeddingsOutputActivationsGradients,
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

  return {
    loss: calculateLoss(outputLogits, correctTokenIndex, weights.vocabulary),
    gradients: {
      unembeddings: unembeddingWeightGradients,
      transformers: transformerGradients,
      embeddings: embeddingsBackprop(
        weights.embeddings,
        transformerInputActivationGradients,
        activations.inputPositionToVocabPosition,
      ),
    },
  };
};

export const embeddingsBackprop = (
  embeddingWeights: number[][],
  outputGradients: number[][],
  inputPositionToVocabPosition: number[],
) => {
  /**
   * Since embeddings are direct inputs to transformer inputs; the transformer input gradients are ALMOST the gradients for the embeddings.
   * Only thing to take into account is the fact that one token might be fetched multiple times from the embeddings lookup table
   * So we need to sum the gradients if it's multiple ones
   */
  const embeddingWeightsGradients = embeddingWeights.map((tokenEmbedding) =>
    new Array(tokenEmbedding.length).fill(0),
  );

  /**
   * z_i = Math.sqrt(j) * e_i + p_i
   *    where e_i is token embedding and p_i is positional encoding
   *
   * dL/e_i = dL/dz_i * dz_i/de_i
   *
   * dz_i/de_i = Math.sqrt(j)
   *
   * dL/e_i = dL/dz_i * Math.sqrt(j)
   *
   * We don't care about dL/dp_i (which is also dL/dz_i) since p_i a non-trainable algorithmic output
   */
  outputGradients.forEach((inputGradientsVector, inputTokenIndex) => {
    const vocabIndex = inputPositionToVocabPosition[inputTokenIndex]!;

    const currentInputGradients = embeddingWeightsGradients[vocabIndex]!;

    const newInputGradients = currentInputGradients.map(
      (partialInputGradient, dimensionIndex) => {
        const z_i = inputGradientsVector[dimensionIndex]!;
        return (
          partialInputGradient + z_i * Math.sqrt(inputGradientsVector.length)
        );
      },
    );

    embeddingWeightsGradients[vocabIndex] = newInputGradients;
  });

  return embeddingWeightsGradients;
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

/**
 * Normalize function is f(h_vi) where h_vi is input activation of vector v at index i
 *
 * f(h_vi) = (h_vi - mean(h_v0...j)) / std(h_v0...j)
 * = (1 / std(h_v0...j)) * (h_vi) - (1 / std(h_v0...j)) * mean(h_v0...j)
 * = std^-1 * h_vi - std^-1 * mean
 *
 * So if k is the input index we are taking the derivative for:
 *
 * df/dh_k = d(std^-1 * h_vi)/dh_k - d(std^-1 * mean)/dh_k
 *
 * d(std^-1 * h_vi)/dh_k = d(std^-1)/dh_k * h_vi + std^-1 * dh_vi/dh_k
 * where dh_vi/dh_k = 1 if i === k, otherwise 0
 *
 * and
 *
 * d(std^-1 * mean)/dh_k = d(std^-1)/dh_k * mean + std^-1 * dmean/dh_k
 *  = d(std^-1)/dh_k * mean + std^-1 * 1/j
 *  = d(std^-1)/dh_k * mean + std^-1/j
 *  = d(std^-1)/dh_k * mean + 1 / (j * std)
 *
 * and
 *
 * d(std^-1)/dh_k = d(std^-1)/dstd * dstd/dh_k
 *  = -std^-2 * dstd/dh_k
 *
 * The full backprop step needs every y_i that depends on h_k:
 *
 * dL/dh_k = sum_i(dL/dy_i * dy_i/dh_k)
 */
export const backpropNormalize = (
  outputGradients: number[][],
  inputActivations: number[][],
): number[][] => {
  return inputActivations.map((vector, vectorIndex) =>
    vector.map((_, valueIndex) => {
      const vectorOutputGradients = outputGradients[vectorIndex]!;

      const { average, standardDeviation } = calculateStandardDeviation(vector);

      // To prevent divide-by-0
      const safeStandardDeviation = standardDeviation + Number.EPSILON;

      const vectorInputGradients = vectorOutputGradients.map(
        (outputGradient, gradientIndex) => {
          const dStdDh =
            -(safeStandardDeviation ** -2) *
            standardDeviationDerivative(vector, valueIndex);

          const dhiDhk = valueIndex === gradientIndex ? 1 : 0;

          const hi = vector[gradientIndex]!;

          const firstBranch = dStdDh * hi + dhiDhk / safeStandardDeviation;

          const secondBranch =
            dStdDh * average + 1 / (vector.length * safeStandardDeviation);

          const dyiDh = firstBranch - secondBranch;

          return outputGradient * dyiDh;
        },
      );

      return sum(vectorInputGradients);
    }),
  );
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
