import {
  operateCombinedWeights,
  makeZeroVersion,
  operateSingleWeights,
} from "../model/model-helpers.ts";
import type { Model, Weights } from "../model/model-types.ts";
import { sum } from "../shared/math.ts";
import { backprop } from "./backprop/backprop.ts";
import { getSequenceLoss } from "./getSequenceLoss.ts";

const TRAINING_ALPHA = 0.003;

export type TrainingExample = {
  sequence: string[];
  maskBeforeIndex: number | null;
};

export const doSingleTrainingPass = async (
  model: Model,
  trainingData: TrainingExample[],
): Promise<{
  averageLoss: number;
  adjustedWeights: Weights;
}> => {
  const summedLossWithGradients = await trainingData.reduce(
    async (accP, example, index) => {
      const acc = await accP;

      // This is needed to give the node 'macrotask' queue to resolve. If we just run synchronous awaits here it will never get around to running that.
      await new Promise((resolve) => setTimeout(resolve, 0));

      const start = Date.now();
      const {
        activations,
        correctTokenIndices,
        outputProbabilities,
        outputLosses,
        unmaskedTokenCount,
      } = getSequenceLoss(example, model);

      const gradients = backprop(
        model,
        activations,
        correctTokenIndices,
        outputProbabilities,
      );
      const duration = Date.now() - start;
      console.log(
        `${(index + 1).toString().padStart(3, "0")}/${trainingData.length} - Duration: ${duration}ms`,
      );

      const summedLoss = sum(outputLosses);

      return {
        loss: acc.loss + summedLoss,
        // The full sequence won't be trained against (there's nothing to predict) so we remove 1 testcase per sequence
        flatTrainingSize: acc.flatTrainingSize + unmaskedTokenCount,
        gradients: operateCombinedWeights(
          acc.gradients,
          gradients,
          (v1, v2) => v1 + v2,
        ),
      };
    },
    Promise.resolve({
      loss: 0,
      flatTrainingSize: 0,
      gradients: makeZeroVersion(model),
    }),
  );

  const averageLoss =
    summedLossWithGradients.loss / summedLossWithGradients.flatTrainingSize;
  const averageGradient = operateSingleWeights(
    summedLossWithGradients.gradients,
    (v1) => v1 / summedLossWithGradients.flatTrainingSize,
  );

  return {
    averageLoss,
    adjustedWeights: operateCombinedWeights(
      model,
      averageGradient,
      // Subtraction since we need to go DOWNHILL
      (v1, v2) => v1 - TRAINING_ALPHA * v2,
    ),
  };
};
