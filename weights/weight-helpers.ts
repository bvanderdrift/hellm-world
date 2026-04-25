import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { divideToWhole } from "../shared/math.ts";
import {
  operateOnMatrices,
  operateOnVectors,
  validateSize,
} from "../shared/matrices.ts";
import type { TransformerWeights, Weights } from "./types.ts";

export const extractHiddenDimensionSize = (weights: Weights) => {
  const embeddingsArray = Object.values(weights.embeddings);
  return embeddingsArray[0]!.length;
};

export const validateWeights = (weights: Weights) => {
  if (weights.vocabulary.length === 0) {
    throw new Error("Provided vocabulary cannot be empty");
  }

  const hiddenDimensionsSize = extractHiddenDimensionSize(weights);

  if (weights.headsCount <= 0) {
    throw new Error("headsCount must be a positive integer");
  }

  divideToWhole(hiddenDimensionsSize, weights.headsCount);

  const tokensDeduped = new Set(weights.vocabulary);

  const duplicateCount = weights.vocabulary.length - tokensDeduped.size;

  if (duplicateCount > 0) {
    throw new Error(`Provided weights have ${duplicateCount} duplicate tokens`);
  }

  if (!tokensDeduped.has(END_OF_SEQUENCE_TOKEN)) {
    throw new Error(
      `Model embeddings are missing special end-of-sequence token "${END_OF_SEQUENCE_TOKEN}"`,
    );
  }

  validateSize(
    weights.embeddings,
    weights.vocabulary.length,
    hiddenDimensionsSize,
  );
  validateSize(
    weights.unembeddings,
    hiddenDimensionsSize,
    weights.vocabulary.length,
  );

  for (const transformer of weights.transformers) {
    validateSize(
      transformer.attention.Q,
      hiddenDimensionsSize,
      hiddenDimensionsSize,
    );
    validateSize(
      transformer.attention.K,
      hiddenDimensionsSize,
      hiddenDimensionsSize,
    );
    validateSize(
      transformer.attention.V,
      hiddenDimensionsSize,
      hiddenDimensionsSize,
    );

    validateSize(
      transformer.attention.out,
      hiddenDimensionsSize,
      hiddenDimensionsSize,
    );

    // MLP validation
    validateSize(
      transformer.multilayerPerceptron.wUp.weightsMatrix,
      hiddenDimensionsSize,
      hiddenDimensionsSize * 4,
    );
    validateSize(
      [transformer.multilayerPerceptron.wUp.biasVector],
      1,
      hiddenDimensionsSize * 4,
    );

    validateSize(
      transformer.multilayerPerceptron.wDown.weightsMatrix,
      hiddenDimensionsSize * 4,
      hiddenDimensionsSize,
    );
    validateSize(
      [transformer.multilayerPerceptron.wDown.biasVector],
      1,
      hiddenDimensionsSize,
    );
  }
};

export const findTokenIndex = (vocabulary: string[], token: string) => {
  const tokenIndex = vocabulary.indexOf(token);

  if (tokenIndex === -1) {
    throw new Error(`Failed to find token ${token} in vocabulary`);
  }

  return tokenIndex;
};

export const validateSameWeightShape = (
  weights1: Weights,
  weights2: Weights,
) => {
  const allTokensExactMatch = weights1.vocabulary.every(
    (token1, tokenIndex) => token1 === weights2.vocabulary[tokenIndex],
  );

  if (!allTokensExactMatch) {
    throw new Error(`Vocabularies between weights don't match`);
  }

  validateSize(
    weights1.embeddings,
    weights2.embeddings.length,
    weights2.embeddings[0]!.length,
  );

  if (weights1.transformers.length !== weights2.transformers.length) {
    throw new Error(
      `Weights1 has different transformers count ${weights1.transformers.length} than Weights2 (${weights2.transformers.length})`,
    );
  }

  if (weights1.headsCount !== weights2.headsCount) {
    throw new Error(
      `Weights1 has different head count ${weights1.headsCount} than Weights2 (${weights2.headsCount})`,
    );
  }
};

export const operateSingleWeights = (
  weights: Weights,
  operation: (value: number) => number,
) => {
  // Hacky solution hihi
  return operateCombinedWeights(weights, weights, (v1) => operation(v1));
};

export const operateCombinedWeights = (
  weights1: Weights,
  weights2: Weights,
  operation: (v1: number, w2: number) => number,
): Weights => {
  validateSameWeightShape(weights1, weights2);

  return {
    ...weights1,
    embeddings: operateOnMatrices(
      weights1.embeddings,
      weights2.embeddings,
      operation,
    ),
    unembeddings: operateOnMatrices(
      weights1.unembeddings,
      weights2.unembeddings,
      operation,
    ),
    transformers: weights1.transformers.map(
      (transformerWeights1, index): TransformerWeights => {
        const transformerWeights2 = weights2.transformers[index]!; // Type-safe b/c of shape check above

        return {
          attention: {
            Q: operateOnMatrices(
              transformerWeights1.attention.Q,
              transformerWeights2.attention.Q,
              operation,
            ),
            K: operateOnMatrices(
              transformerWeights1.attention.K,
              transformerWeights2.attention.K,
              operation,
            ),
            V: operateOnMatrices(
              transformerWeights1.attention.V,
              transformerWeights2.attention.V,
              operation,
            ),
            out: operateOnMatrices(
              transformerWeights1.attention.out,
              transformerWeights2.attention.out,
              operation,
            ),
          },
          multilayerPerceptron: {
            wDown: {
              weightsMatrix: operateOnMatrices(
                transformerWeights1.multilayerPerceptron.wDown.weightsMatrix,
                transformerWeights2.multilayerPerceptron.wDown.weightsMatrix,
                operation,
              ),
              biasVector: operateOnVectors(
                transformerWeights1.multilayerPerceptron.wDown.biasVector,
                transformerWeights2.multilayerPerceptron.wDown.biasVector,
                operation,
              ),
            },
            wUp: {
              weightsMatrix: operateOnMatrices(
                transformerWeights1.multilayerPerceptron.wUp.weightsMatrix,
                transformerWeights2.multilayerPerceptron.wUp.weightsMatrix,
                operation,
              ),
              biasVector: operateOnVectors(
                transformerWeights1.multilayerPerceptron.wUp.biasVector,
                transformerWeights2.multilayerPerceptron.wUp.biasVector,
                operation,
              ),
            },
          },
        };
      },
    ),
  };
};

export const makeZeroVersion = (weights: Weights) =>
  operateSingleWeights(weights, () => 0);
