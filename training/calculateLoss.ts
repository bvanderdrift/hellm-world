import { safeSumExponatedLogits, sum } from "../shared/math.ts";
import { validateSize } from "../shared/matrices.ts";

/**
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
export const calculateLoss = (
  outputLogits: number[],
  correctTokenIndex: number,
  vocabulary: string[],
) => {
  validateSize([outputLogits], 1, vocabulary.length);

  const { summed, biggestLogit } = safeSumExponatedLogits(outputLogits);

  const correctTokenLogit = outputLogits[correctTokenIndex]!;
  const correctTokenLogitAdjusted = correctTokenLogit - biggestLogit;

  const baseAdjusted = Math.log(summed);

  return baseAdjusted - correctTokenLogitAdjusted;
};
