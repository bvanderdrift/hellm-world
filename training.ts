import type { Weights } from "./weights/types.ts";
import {
  getLatestCheckpointWeights,
  writeNewCheckpoint,
} from "./weights/weight-io.ts";

export const doSingleTrainingPass = (weights: Weights): Weights => {
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

  for (let index = 0; index < steps; index++) {
    weights = doSingleTrainingPass(weights);
  }

  writeNewCheckpoint(model, weights);

  console.log(`✅ Succesfully ran training loop for model ${model}`);
};
