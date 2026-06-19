import type { Model, Weights } from "../../model/model-types.ts";
import type { Activations } from "../../model/activations-types.ts";
import { embeddingsBackprop } from "./embeddingBackprop.ts";
import { probabilityOutputBackprop } from "./probabilityOutputBackprop.ts";
import { matrixBackprop } from "./matrixBackprop.ts";
import { backpropNormalize } from "./normalizeBackprop.ts";
import { transformersBackprop } from "./transformersBackprop.ts";
import type { Matrix } from "../../shared/matrices/matrices.ts";

export const backprop = (
  model: Model,
  activations: Activations,
  /** -1 is mask aka ignore this token */
  correctTokenIndices: number[],
  outputProbabilities: Matrix,
): Weights => {
  const unembeddingsOutputActivationsGradients = probabilityOutputBackprop(
    activations.unembeddingsOutputLogits,
    outputProbabilities,
    correctTokenIndices,
  );

  const {
    weightGradients: unembeddingWeightGradients,
    activationGradients: unembeddingInputActivationGradients,
  } = matrixBackprop(
    model.unembeddings,
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
    model.transformers,
    activations.transformerActivations,
    model.counts.attentionHeads,
  );

  return {
    unembeddings: unembeddingWeightGradients,
    transformers: transformerGradients,
    embeddings: embeddingsBackprop(
      model.embeddings,
      transformerInputActivationGradients,
      activations.inputPositionToVocabPosition,
    ),
  };
};
