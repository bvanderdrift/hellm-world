import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { divideToWhole } from "../shared/math.ts";
import { validateSize } from "../shared/matrices.ts";
import { extractHiddenDimensionSize } from "./model-helpers.ts";
import type { Model } from "./model-types.ts";

export const validateModel = (model: Model) => {
  if (model.vocabulary.length === 0) {
    throw new Error("Provided vocabulary cannot be empty");
  }

  const hiddenDimensionsSize = extractHiddenDimensionSize(model);

  if (model.headsCount <= 0) {
    throw new Error("headsCount must be a positive integer");
  }

  divideToWhole(hiddenDimensionsSize, model.headsCount);

  const tokensDeduped = new Set(model.vocabulary);

  const duplicateCount = model.vocabulary.length - tokensDeduped.size;

  if (duplicateCount > 0) {
    throw new Error(`Provided weights have ${duplicateCount} duplicate tokens`);
  }

  if (!tokensDeduped.has(END_OF_SEQUENCE_TOKEN)) {
    throw new Error(
      `Model embeddings are missing special end-of-sequence token "${END_OF_SEQUENCE_TOKEN}"`,
    );
  }

  validateSize(model.embeddings, model.vocabulary.length, hiddenDimensionsSize);
  validateSize(
    model.unembeddings,
    hiddenDimensionsSize,
    model.vocabulary.length,
  );

  for (const transformer of model.transformers) {
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
      hiddenDimensionsSize * model.mlpMultiple,
    );
    validateSize(
      [transformer.multilayerPerceptron.wUp.biasVector],
      1,
      hiddenDimensionsSize * model.mlpMultiple,
    );

    validateSize(
      transformer.multilayerPerceptron.wDown.weightsMatrix,
      hiddenDimensionsSize * model.mlpMultiple,
      hiddenDimensionsSize,
    );
    validateSize(
      [transformer.multilayerPerceptron.wDown.biasVector],
      1,
      hiddenDimensionsSize,
    );
  }
};

export const validateSameWeightShape = (weights1: Model, weights2: Model) => {
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
