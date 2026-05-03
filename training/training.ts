import { llmForwardPassByTokens } from "../running/llm.ts";
import { sum } from "../shared/math.ts";
import type { Model, Weights } from "../model/model-types.ts";
import {
  makeZeroVersion,
  operateCombinedWeights,
  operateSingleWeights,
} from "../model/model-helpers.ts";
import {
  getLatestCheckpointModel,
  writeNewCheckpoint,
} from "../model/model-io.ts";
import { backprop } from "./backprop/backprop.ts";
import { prepareTrainingData } from "./prepareTrainingData.ts";

const TRAINING_ALPHA = 0.1;

export const doTrainingLoopAndStoreCheckpoint = (
  modelName: string,
  steps: number,
) => {
  const { historyLosses, model: modelLoaded } =
    getLatestCheckpointModel(modelName);

  let model = modelLoaded;

  if (steps <= 0) {
    throw new Error(`steps has to be a positive integer, received: ${steps}`);
  }

  const trainingData = prepareTrainingData(modelName, model.vocabulary);

  for (let index = 0; index < steps; index++) {
    const { averageLoss, adjustedWeights } = doSingleTrainingPass(
      model,
      trainingData,
    );

    const indexPadded = index.toString().padStart(steps.toString().length, "0");

    console.log(
      `(${indexPadded}/${steps}) Training pass done - average loss: ${averageLoss}`,
    );
    historyLosses.push(averageLoss);
    model = {
      ...model,
      ...adjustedWeights,
    };
  }

  writeNewCheckpoint(modelName, {
    historyLosses,
    weights: model,
  });

  console.log(`✅ Succesfully ran training loop for model ${modelName}`);
};

export const doSingleTrainingPass = (
  model: Model,
  trainingData: string[][],
): {
  averageLoss: number;
  adjustedWeights: Weights;
} => {
  const flatTrainingSize = sum(
    trainingData.map(
      (sequence) =>
        // The full sequence won't be trained against (there's nothing to predict) so we remove 1 testcase per sequence
        sequence.length - 1,
    ),
  );

  if (flatTrainingSize === 0) {
    throw new Error(`No training data present`);
  }

  const summedLossWithGradients = trainingData.reduce(
    (acc, sequence) => {
      const inputTokens = sequence.slice(0, sequence.length - 1);
      const { activations } = llmForwardPassByTokens(inputTokens, model, true);

      if (!activations) {
        throw new Error(`No activations returned during LLM Forward pass`);
      }

      const correctTokenIndices = inputTokens.map((_, index) => {
        const correctToken = sequence[index + 1]!;

        return model.vocabulary.indexOf(correctToken);
      });

      const backpropResults = backprop(model, activations, correctTokenIndices);

      return {
        loss: acc.loss + backpropResults.loss,
        gradients: operateCombinedWeights(
          acc.gradients,
          backpropResults.gradients,
          (v1, v2) => v1 + v2,
        ),
      };
    },
    { loss: 0, gradients: makeZeroVersion(model) },
  );

  const averageLoss = summedLossWithGradients.loss / flatTrainingSize;
  const averageGradient = operateSingleWeights(
    summedLossWithGradients.gradients,
    (v1) => v1 / flatTrainingSize,
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
