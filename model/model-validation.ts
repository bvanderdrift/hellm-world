import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { divideToWhole } from "../shared/math.ts";
import { type Matrix } from "../shared/matrices.ts";
import { extractHiddenDimensionSize } from "./model-helpers.ts";
import type { Model, Weights } from "./model-types.ts";

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

  validateSize(model.embeddings);
  validateSize(model.unembeddings);

  for (const transformer of model.transformers) {
    validateSize(transformer.attention.Q);
    validateSize(transformer.attention.K);
    validateSize(transformer.attention.V);

    validateSize(transformer.attention.out);

    // MLP validation
    validateSize(transformer.multilayerPerceptron.wUp.weightsMatrix);
    validateSize(transformer.multilayerPerceptron.wUp.biasVector);

    validateSize(transformer.multilayerPerceptron.wDown.weightsMatrix);
    validateSize(transformer.multilayerPerceptron.wDown.biasVector);
  }
};

export const validateSameModelShape = (model1: Model, model2: Model) => {
  const allTokensExactMatch = model1.vocabulary.every(
    (token1, tokenIndex) => token1 === model2.vocabulary[tokenIndex],
  );

  if (!allTokensExactMatch) {
    throw new Error(`Vocabularies between weights don't match`);
  }

  if (model1.headsCount !== model2.headsCount) {
    throw new Error(
      `Model 1 has different head count ${model1.headsCount} than Model 2 (${model2.headsCount})`,
    );
  }

  validateSameWeightsShape(model1, model2);
};

export const validateSameWeightsShape = (
  weights1: Weights,
  weights2: Weights,
) => {
  validateSize(weights1.embeddings);

  if (weights1.transformers.length !== weights2.transformers.length) {
    throw new Error(
      `Weights1 has different transformers count ${weights1.transformers.length} than Weights2 (${weights2.transformers.length})`,
    );
  }
};

const validateSize = (matrix: Matrix) => {
  const expectedSize = matrix.vectors * matrix.dimensions;

  if (expectedSize !== matrix.values.length) {
    throw new Error(
      `m has unexpected parameter count ${matrix.values.length}, expected ${expectedSize}`,
    );
  }
};
