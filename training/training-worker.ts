import {
  operateCombinedWeights,
  makeZeroVersion,
  operateSingleWeights,
} from "../model/model-helpers.ts";
import type { Model, Weights } from "../model/model-types.ts";
import { llmForwardPassByTokens } from "../running/llm.ts";
import { backprop } from "./backprop/backprop.ts";

const TRAINING_ALPHA = 0.01;

// prevents TS errors
declare var self: Worker;

export type InputMessagePayload = {
  model: Model;
  trainingData: string[][];
};

export type OutputMessagePayload = {
  averageLoss: number;
  adjustedWeights: Weights;
};

self.onmessage = async (event: MessageEvent<InputMessagePayload>) => {
  const output: OutputMessagePayload = await doSingleTrainingPass(
    event.data.model,
    event.data.trainingData,
  );

  self.postMessage(output);
};

export const doSingleTrainingPass = async (
  model: Model,
  trainingData: string[][],
): Promise<{
  averageLoss: number;
  adjustedWeights: Weights;
}> => {
  const summedLossWithGradients = await trainingData.reduce(
    async (accP, sequence, index) => {
      const acc = await accP;
      const { activations } = llmForwardPassByTokens(sequence, model, true);

      if (!activations) {
        throw new Error(`No activations returned during LLM Forward pass`);
      }

      const correctTokenIndices = sequence.map((_, index) => {
        const correctToken = sequence[index + 1]!;

        return model.vocabulary.indexOf(correctToken);
      });

      const unmaskedTokenCount = correctTokenIndices.filter(
        (i) => i !== -1,
      ).length;

      const start = Date.now();
      const backpropResults = backprop(model, activations, correctTokenIndices);
      const duration = Date.now() - start;
      console.log(
        `${(index + 1).toString().padStart(3, "0")}/${trainingData.length} - Duration: ${duration}ms`,
      );

      return {
        loss: acc.loss + backpropResults.loss,
        // The full sequence won't be trained against (there's nothing to predict) so we remove 1 testcase per sequence
        flatTrainingSize: acc.flatTrainingSize + unmaskedTokenCount,
        gradients: operateCombinedWeights(
          acc.gradients,
          backpropResults.gradients,
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
