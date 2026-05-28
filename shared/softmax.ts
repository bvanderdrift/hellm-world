import { sum } from "./math.ts";

export const safeSumExponatedLogits = (logits: Float32Array) => {
  const biggestLogit = Math.max(...logits);
  // To prevent overflows. Logits are still related the same since they all subtract the same value
  const safeLogits = logits.map((logit) => logit - biggestLogit);
  const exponatedLogits = safeLogits.map((l) => Math.exp(l));

  return {
    safeLogits,
    exponatedLogits,
    summed: sum(exponatedLogits),
    biggestLogit,
  };
};

/**
 * s_i = e^l_i / sum(e^l_j)
 */
export const softmax = (logits: Float32Array) => {
  const { safeLogits, summed } = safeSumExponatedLogits(logits);

  return safeLogits.map((logit) => Math.exp(logit) / summed);
};
