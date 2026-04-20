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

export const validateWeights = (weights: Weights) => {
  if (weights.vocabulary.length === 0) {
    throw new Error("Provided vocabulary cannot be empty");
  }

  const { hiddenDimensionsSize, vocabSize } = extractDimensionSizes(weights);

  if (weights.headsCount <= 0) {
    throw new Error("headsCount must be a positive integer");
  }

  divideToWhole(hiddenDimensionsSize, weights.headsCount);

  const tokensDeduped = new Set(weights.vocabulary);

  const duplicateCount = weights.vocabulary.length - tokensDeduped.size;

  if (duplicateCount > 0) {
    throw new Error(`Provided weights have ${duplicateCount} duplicate tokens`);
  }

  const embeddingsEntries = Object.entries(weights.embeddings);

  if (embeddingsEntries.length !== tokensDeduped.size) {
    throw new Error(
      `Provided embeddings has unexpected vocabulary size ${embeddingsEntries.length}, expected ${tokensDeduped.size}`,
    );
  }

  for (const [token, vector] of embeddingsEntries) {
    if (!tokensDeduped.has(token)) {
      throw new Error(
        `Unknown embedding token ${token}. Does not occur in vocabulary of model.`,
      );
    }

    if (vector.length !== hiddenDimensionsSize) {
      throw new Error(
        `Token ${token} has unexpected vector length ${vector.length} vs base length ${hiddenDimensionsSize}`,
      );
    }
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
