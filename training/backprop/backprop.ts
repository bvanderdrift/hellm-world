import type { Model, Weights } from "../../model/model-types.ts";
import { calculateLoss } from "../calculateLoss.ts";
import type { Activations } from "../../model/activations-types.ts";
import { softmax } from "../../shared/math.ts";
import { embeddingsBackprop } from "./embeddingBackprop.ts";
import { probabilityOutputBackprop } from "./probabilityOutputBackprop.ts";
import { matrixBackprop } from "./matrixBackprop.ts";
import { backpropNormalize } from "./normalizeBackprop.ts";
import { transformersBackprop } from "./transformersBackprop.ts";

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
