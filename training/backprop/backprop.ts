import type { Model, Weights } from "../../model/model-types.ts";
import type { Activations } from "../../model/activations-types.ts";
import { embeddingsBackprop } from "./embeddingBackprop.ts";
import { probabilityOutputBackprop } from "./probabilityOutputBackprop.ts";
import { matrixBackprop } from "./matrixBackprop.ts";
import { backpropNormalize } from "./normalizeBackprop.ts";
import { transformersBackprop } from "./transformersBackprop.ts";

export const backprop = (
  weights: Model,
  activations: Activations,
  /** -1 is mask aka ignore this token */
  correctTokenIndices: number[],
  outputProbabilities: {
    probabilities: number[];
    loss: number;
  }[],
): Weights => {
  const unembeddingsOutputActivationsGradients = probabilityOutputBackprop(
    activations.unembeddingsOutputLogits,
    outputProbabilities.map((o) => o.probabilities),
    correctTokenIndices,
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
    unembeddings: unembeddingWeightGradients,
    transformers: transformerGradients,
    embeddings: embeddingsBackprop(
      weights.embeddings,
      transformerInputActivationGradients,
      activations.inputPositionToVocabPosition,
    ),
  };
};
