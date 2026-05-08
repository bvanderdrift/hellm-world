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

const TRAINING_ALPHA = 0.03;
const MAX_TRAINING_DATA_PER_PASS = 100;

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

  const trainingData = shuffleArray(
    prepareTrainingData(modelName, model.vocabulary),
  );

  const startTime = Date.now();

  for (let index = 0; index < steps; index++) {
    const offset =
      Math.random() * (trainingData.length - MAX_TRAINING_DATA_PER_PASS);
    const trainingDataToWorkWith = trainingData.slice(
      offset,
      offset + MAX_TRAINING_DATA_PER_PASS,
    );

    const { averageLoss, adjustedWeights } = doSingleTrainingPass(
      model,
      trainingDataToWorkWith,
    );

    const indexPadded = (index + 1)
      .toString()
      .padStart(steps.toString().length, "0");

    const totalDuration = Date.now() - startTime;
    const avgDuration = totalDuration / (index + 1);

    console.log(
      `(${indexPadded}/${steps}) Training pass done - average loss: ${averageLoss} - avg duration: ${Math.round(avgDuration)} ms`,
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
  const summedLossWithGradients = trainingData.reduce(
    (acc, sequence) => {
      const { activations } = llmForwardPassByTokens(sequence, model, true);

      if (!activations) {
        throw new Error(`No activations returned during LLM Forward pass`);
      }

      const correctTokenIndices = sequence.map((_, index) => {
        const correctToken = sequence[index + 1]!;

        return model.vocabulary.indexOf(correctToken);
      });

      const backpropResults = backprop(model, activations, correctTokenIndices);

      return {
        loss: acc.loss + backpropResults.loss,
        // The full sequence won't be trained against (there's nothing to predict) so we remove 1 testcase per sequence
        flatTrainingSize: acc.flatTrainingSize + sequence.length - 1,
        gradients: operateCombinedWeights(
          acc.gradients,
          backpropResults.gradients,
          (v1, v2) => v1 + v2,
        ),
      };
    },
    { loss: 0, flatTrainingSize: 0, gradients: makeZeroVersion(model) },
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

const shuffleArray = <T>(values: T[]): T[] =>
  values
    .map((value) => ({ value, sort: Math.random() }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ value }) => value);
