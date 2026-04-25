import { llmForwardPassByTokens } from "../running/llm.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { sum } from "../shared/math.ts";
import type { Model } from "../weights/types.ts";
import {
  makeZeroVersion,
  operateCombinedWeights,
  operateSingleWeights,
} from "../weights/weight-helpers.ts";
import {
  getLatestCheckpointWeights,
  writeNewCheckpoint,
} from "../weights/weight-io.ts";
import { backprop } from "./backprop.ts";
import { prepareTrainingData } from "./prepareTrainingData.ts";

const TRAINING_ALPHA = 0.0001;

export const doTrainingLoopAndStoreCheckpoint = (
  model: string,
  steps: number,
) => {
  let weights = getLatestCheckpointWeights(model);

  if (steps <= 0) {
    throw new Error(`steps has to be a positive integer, received: ${steps}`);
  }

  const trainingData = prepareTrainingData(model, weights.vocabulary);

  for (let index = 0; index < steps; index++) {
    const { averageLoss, adjustedWeights } = doSingleTrainingPass(
      weights,
      trainingData,
    );

    console.log(`Training pass done - average loss: ${averageLoss}`);
    weights = adjustedWeights;
  }

  writeNewCheckpoint(model, weights);

  console.log(`✅ Succesfully ran training loop for model ${model}`);
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
