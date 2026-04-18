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

  for (const [token, vector] of Object.entries(weights.embeddings)) {
    if (vector.length !== hiddenDimensionsSize) {
      throw new Error(
        `Token ${token} has unexpected vector length ${vector.length} vs base length ${hiddenDimensionsSize}`,
      );
    }
  }

  const tokensDeduped = new Set(...weights.tokens);
  const duplicateCount = weights.tokens.length - tokensDeduped.size;

  if(duplicateCount > 0){
    throw new Error(`Provided weights have ${duplicateCount} duplicate tokens`)
  }

  validateSize(weights.unembeddings, hiddenDimensionsSize, vocabSize);

  for (const transformer of weights.transformers) {
    const attentionDimension = divideToWhole(
      hiddenDimensionsSize,
      transformer.attention.heads.length,
    );

    for (const attentionHead of transformer.attention.heads) {
      validateSize(attentionHead.Q, hiddenDimensionsSize, attentionDimension);
      validateSize(attentionHead.K, hiddenDimensionsSize, attentionDimension);
      validateSize(
        attentionHead.V.up,
        attentionDimension,
        hiddenDimensionsSize,
      );
      validateSize(
        attentionHead.V.down,
        hiddenDimensionsSize,
        attentionDimension,
      );
    }

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
