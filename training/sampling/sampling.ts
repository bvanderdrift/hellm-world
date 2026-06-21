import type { ModelTrainingState } from "../../model/model-types.ts";
import type { TrainingExample } from "../doSingleTrainingPass.ts";
import {
  computeSamplingWeights,
  sampleIndices,
} from "./loss-weighted-sampling.ts";

const MAX_TRAINING_DATA_PER_PASS = 100;

export const sampleBatch = (
  trainingState: ModelTrainingState,
  trainingData: TrainingExample[],
) => {
  if (trainingState.samplerState.type === "uniform") {
    const offset = Math.floor(
      Math.random() * (trainingData.length - MAX_TRAINING_DATA_PER_PASS),
    );
    return trainingData
      .slice(offset, offset + MAX_TRAINING_DATA_PER_PASS)
      .map((dataPoint, index) => ({
        originalIndex: offset + index,
        trainingData: dataPoint,
      }));
  }

  const weights = computeSamplingWeights(
    trainingState.samplerState.lossRecord,
    trainingData.length,
  );
  const indices = sampleIndices(weights, MAX_TRAINING_DATA_PER_PASS);

  return indices.map((pickedIndex) => ({
    originalIndex: pickedIndex,
    trainingData: trainingData[pickedIndex]!,
  }));
};
