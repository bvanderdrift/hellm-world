import { validateSize } from "../shared/matrices.ts";
import type { Activations, Model } from "../model/model-types.ts";
import { makeZeroVersion } from "../model/model-helpers.ts";
import { calculateLoss } from "./calculateLoss.ts";

export const backprop = (
  inputTokensLength: number,
  correctOutputToken: string,
  weights: Model,
  activations: Activations,
): {
  loss: number;
  gradients: Model;
} => {
  const outputLogits = activations.outputLogits[inputTokensLength - 1];

  if (!outputLogits) {
    throw new Error(
      `Couldn't find output logits in activations. Activations vector count: ${activations.outputLogits.length}, inputTokensLength: ${inputTokensLength}`,
    );
  }

  validateSize([outputLogits], 1, weights.vocabulary.length);

  return {
    loss: calculateLoss(outputLogits, correctOutputToken, weights.vocabulary),
    gradients: makeZeroVersion(weights),
  };
};
