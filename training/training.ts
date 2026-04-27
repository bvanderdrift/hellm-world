import { llmForwardPassByTokens } from "../running/llm.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { sum } from "../shared/math.ts";
import type { Model } from "../model/model-types.ts";
import {
  makeZeroVersion,
  operateCombinedWeights,
  operateSingleWeights,
} from "../model/model-helpers.ts";
import {
  getLatestCheckpointModel,
  writeNewCheckpoint,
} from "../model/model-io.ts";
import { backprop } from "./backprop.ts";
import { prepareTrainingData } from "./prepareTrainingData.ts";

const TRAINING_ALPHA = 0.0001;

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
    model = adjustedWeights;
  }

  writeNewCheckpoint(modelName, {
    historyLosses,
    weights: model,
  });

  console.log(`✅ Succesfully ran training loop for model ${modelName}`);
};

export const doSingleTrainingPass = (
  weights: Model,
  trainingData: string[][],
): {
  averageLoss: number;
  adjustedWeights: Model;
} => {
  const flatTrainingSize = sum(
    // Add one for the EOS special token
    trainingData.map((sequence) => sequence.length + 1),
  );

  if (flatTrainingSize === 0) {
    throw new Error(`No training data present`);
  }

  const summedLossWithGradients = trainingData.reduce(
    (acc, sequence) => {
      const embeddings = llmForwardPassByTokens(sequence, weights);

      const summedLossWithGradientsWithinSequence = sequence.reduce(
        (acc, _, index) => {
          const inputTokens = sequence.slice(0, index + 1);
          const expectedOutput = sequence[index + 1] ?? END_OF_SEQUENCE_TOKEN;

          const backpropResults = backprop(
            inputTokens.length,
            expectedOutput,
            weights,
            {
              outputLogits: embeddings,
            },
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
        { loss: 0, gradients: makeZeroVersion(weights) },
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
    { loss: 0, gradients: makeZeroVersion(weights) },
  );

  const averageLoss = summedLossWithGradients.loss / flatTrainingSize;
  const averageGradient = operateSingleWeights(
    summedLossWithGradients.gradients,
    (v1) => v1 / flatTrainingSize,
  );

  return {
    averageLoss,
    adjustedWeights: operateCombinedWeights(
      weights,
      averageGradient,
      // Subtraction since we need to go DOWNHILL
      (v1, v2) => v1 - TRAINING_ALPHA * v2,
    ),
  };
};
