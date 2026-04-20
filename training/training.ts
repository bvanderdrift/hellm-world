import type { Weights } from "../weights/types.ts";
import {
  getLatestCheckpointWeights,
  writeNewCheckpoint,
} from "../weights/weight-io.ts";
import { prepareTrainingData } from "./prepareTrainingData.ts";

export const doSingleTrainingPass = (
  weights: Weights,
  trainingData: string[][],
): Weights => {
  return weights;
};

export const doTrainingLoopAndStoreCheckpoint = (
  model: string,
  steps: number,
) => {
  let weights = getLatestCheckpointWeights(model);

  if (steps <= 0) {
    new Error(`steps has to be a positive integer, received: ${steps}`);
  }

  const trainingData = prepareTrainingData(model, weights.vocabulary);

  for (let index = 0; index < steps; index++) {
    weights = doSingleTrainingPass(weights, trainingData);
  }

  writeNewCheckpoint(model, weights);

  console.log(`✅ Succesfully ran training loop for model ${model}`);
};
