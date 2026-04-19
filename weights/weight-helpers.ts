import { divideToWhole } from "../math.ts";
import { validateSize } from "../matrices.ts";
import type { Weights } from "./types.ts";

export const extractDimensionSizes = (weights: Weights) => {
  const embeddingsArray = Object.values(weights.embeddings);
  return {
    hiddenDimensionsSize: embeddingsArray[0]!.length,
    vocabSize: embeddingsArray.length,
  };
};

export const validateSizing = (weights: Weights) => {
  const { hiddenDimensionsSize, vocabSize } = extractDimensionSizes(weights);

  divideToWhole(hiddenDimensionsSize, weights.headsCount);

  for (const [token, vector] of Object.entries(weights.embeddings)) {
    if (vector.length !== hiddenDimensionsSize) {
      throw new Error(
        `Token ${token} has unexpected vector length ${vector.length} vs base length ${hiddenDimensionsSize}`,
      );
    }
  }

  const tokensDeduped = new Set(weights.tokens);
  const duplicateCount = weights.tokens.length - tokensDeduped.size;

  if (duplicateCount > 0) {
    throw new Error(`Provided weights have ${duplicateCount} duplicate tokens`);
  }

  validateSize(weights.unembeddings, hiddenDimensionsSize, vocabSize);

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
