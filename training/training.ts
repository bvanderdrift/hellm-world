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

const TRAINING_ALPHA = 0.01;

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

    console.log(`Training pass done - average loss: ${averageLoss}`);
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
      const { activations } = llmForwardPassByTokens(sequence, model, true);

      if (!activations) {
        throw new Error(`No activations returned during LLM Forward pass`);
      }

      const summedLossWithGradientsWithinSequence = sequence.reduce(
        (acc, _, index) => {
          if (index === sequence.length - 1) {
            // We're at last, nothing more to predict so stop here
            return acc;
          }

          const correctToken = sequence[index + 1]!;
          const correctTokenIndex = model.vocabulary.indexOf(correctToken);

          if (correctTokenIndex === -1) {
            throw new Error(
              `Failed to find token ${correctToken} in model vocab`,
            );
          }

          const inputTokens = sequence.slice(0, index + 1);

          const backpropResults = backprop(
            inputTokens,
            model,
            activations,
            correctTokenIndex,
          );

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

      return {
        loss: acc.loss + summedLossWithGradientsWithinSequence.loss,
        gradients: operateCombinedWeights(
          acc.gradients,
          summedLossWithGradientsWithinSequence.gradients,
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
