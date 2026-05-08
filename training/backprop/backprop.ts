import type { Model, Weights } from "../../model/model-types.ts";
import type { Activations } from "../../model/activations-types.ts";
import { safeSumExponatedLogits, softmax, sum } from "../../shared/math.ts";
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
): {
  loss: number;
  gradients: Weights;
} => {
  const outputProbabilities = activations.unembeddingsOutputLogits.map(
    (outputLogits, contextIndex) => {
      const correctTokenIndex = correctTokenIndices[contextIndex]!;

      /**
       * This is essentially just a more elaborate softmax, so we can deal with problems around JS floating points when we get to really small numbers
       *
       * Problem:
       * Normal Cross-entropy loss function is -Math.log(p_k)
       * p_k is calculated from logits.
       * A really small logit vs the others is valid.
       * But, Math.exp(small_logit) can be 0, very quickly. For example Math.exp(-1000) = 0 not e^-1000
       * No problem is probability space, but problematic in the loss function
       * Since we do want to calculate the gradient if the logit of the correct output token is small like -1000.
       * But if we just use probabilities; the Math.exp problem would've turned it into 0, and we would have calculated cost as Infinity
       *
       * Solution:
       * We do softmax but without the normalizing (so not make the vector sum to 1)
       * This way the `correctTokenLogit` never has to be exponentiated for the loss calculation and can be used directly
       */
      const { summed, safeLogits, biggestLogit } =
        safeSumExponatedLogits(outputLogits);

      const correctTokenLogit = outputLogits[correctTokenIndex]!;
      const correctTokenLogitAdjusted = correctTokenLogit - biggestLogit;

      const baseAdjusted = Math.log(summed);

      const probability = safeLogits.map((logit) => Math.exp(logit) / summed);

      return {
        probability,
        loss: baseAdjusted - correctTokenLogitAdjusted,
      };
    },
  );

  const unembeddingsOutputActivationsGradients = probabilityOutputBackprop(
    activations.unembeddingsOutputLogits,
    outputProbabilities.map((o) => o.probability),
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
    loss: sum(outputProbabilities.map((o) => o.loss)),
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
