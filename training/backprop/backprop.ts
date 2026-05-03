import type { Model, Weights } from "../../model/model-types.ts";
import { calculateLoss } from "../calculateLoss.ts";
import type { Activations } from "../../model/activations-types.ts";
import { softmax, sum } from "../../shared/math.ts";
import { embeddingsBackprop } from "./embeddingBackprop.ts";
import { probabilityOutputBackprop } from "./probabilityOutputBackprop.ts";
import { matrixBackprop } from "./matrixBackprop.ts";
import { backpropNormalize } from "./normalizeBackprop.ts";
import { transformersBackprop } from "./transformersBackprop.ts";

export const backprop = (
  weights: Model,
  activations: Activations,
  correctTokenIndices: number[],
): {
  loss: number;
  gradients: Weights;
} => {
  const outputProbabilities = activations.unembeddingsOutputLogits.map(
    (outputLogits) => softmax(outputLogits),
  );

  const unembeddingsOutputActivationsGradients = probabilityOutputBackprop(
    activations.unembeddingsOutputLogits,
    outputProbabilities,
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
    loss: sum(
      activations.unembeddingsOutputLogits.map((outputLogits, tokenIndex) =>
        calculateLoss(
          outputLogits,
          correctTokenIndices[tokenIndex]!,
          weights.vocabulary,
        ),
      ),
    ),
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
