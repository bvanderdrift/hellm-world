import { validateSize } from "../shared/matrices.ts";
import type { Model } from "../model/model-types.ts";
import { makeZeroVersion } from "../model/model-helpers.ts";
import { calculateLoss } from "./calculateLoss.ts";
import type { Activations } from "../model/activations-types.ts";

export const backprop = (
  inputTokens: string[],
  correctOutputToken: string,
  weights: Model,
  activations: Activations,
): {
  loss: number;
  gradients: Model;
} => {
  const outputLogits =
    activations.unembeddingsOutputLogits[inputTokens.length - 1];

  if (!outputLogits) {
    throw new Error(
      `Couldn't find output logits in activations. Activations vector count: ${activations.unembeddingsOutputLogits.length}, inputTokensLength: ${inputTokens.length}`,
    );
  }

  validateSize([outputLogits], 1, weights.vocabulary.length);

  return {
    loss: calculateLoss(outputLogits, correctOutputToken, weights.vocabulary),
    gradients: makeZeroVersion(weights),
  };
};
