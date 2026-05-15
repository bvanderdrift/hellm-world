import type { Model } from "../model/model-types.ts";
import { llmForwardPassByTokens } from "../running/llm.ts";
import { safeSumExponatedLogits } from "../shared/math.ts";
import type { TrainingExample } from "./doSingleTrainingPass.ts";

export const getSequenceLoss = (
  { sequence, maskBeforeIndex }: TrainingExample,
  model: Model,
) => {
  const { activations } = llmForwardPassByTokens(sequence, model, true);

  if (!activations) {
    throw new Error(`No activations returned during LLM Forward pass`);
  }

  const correctTokenIndices = sequence.map((_, index) => {
    if (maskBeforeIndex !== null && index < maskBeforeIndex) {
      return -1; //Mask
    }

    const correctToken = sequence[index + 1]!;

    return model.vocabulary.indexOf(correctToken);
  });

  const unmaskedTokenCount = correctTokenIndices.filter((i) => i !== -1).length;

  const outputProbabilities = activations.unembeddingsOutputLogits.map(
    (outputLogits, contextIndex) => {
      const correctTokenIndex = correctTokenIndices[contextIndex]!;

      if (correctTokenIndex === -1) {
        return {
          probabilities: outputLogits.map((_) => 0),
          loss: 0,
        };
      }

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
        probabilities: probability,
        loss: baseAdjusted - correctTokenLogitAdjusted,
      };
    },
  );

  return {
    activations,
    correctTokenIndices,
    unmaskedTokenCount,
    outputProbabilities,
  };
};
