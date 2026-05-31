import {
  getModelFolderPath,
  writeCheckpoint,
  writeHistory,
} from "../model/model-io.ts";
import type {
  Model,
  ModelTrainingHistory,
  Weights,
} from "../model/model-types.ts";
import type { TrainingExample } from "./doSingleTrainingPass.ts";
import {
  computeSamplingWeights,
  sampleIndices,
} from "./sampling/loss-weighted-sampling.ts";
import {
  MAX_TRAINING_DATA_PER_PASS,
  type SamplerState,
} from "./sampling/sampling.ts";
import type { EndDefinition } from "./training.ts";

const STORE_INTERVAL = 500;

export const createStateStore = (
  endDefinition: EndDefinition | null,
  modelName: string,
  incomingModel: Model,
  initialSamplingState: SamplerState,
) => {
  const startTime = Date.now();
  let index = 0;
  let modelUnderTraining = incomingModel;
  const history = incomingModel.history;
  const samplingState = initialSamplingState;

  const getPercentComplete = (def: EndDefinition) => {
    if (def.type === "steps") {
      return index / def.count;
    }

    const timeLapsed = Date.now() - startTime;

    const minutesLapsed = timeLapsed / (1000 * 60);

    return minutesLapsed / def.count;
  };

  const getState = () => {
    if (!endDefinition) {
      return {
        model: modelUnderTraining,
        history: history,
        startTime,
        isDone: false,
        percentDone: null,
        stepsInThisRun: index,
      };
    }

    const percentDone = getPercentComplete(endDefinition);

    return {
      model: modelUnderTraining,
      history: history,
      startTime,
      isDone: percentDone >= 1,
      percentDone,
      stepsInThisRun: index,
    };
  };

  const writeNewCheckpointAndHistory = () => {
    writeHistory(getModelFolderPath(modelName), history);
    writeCheckpoint(
      modelName,
      history.trainingLosses.length,
      modelUnderTraining,
    );
  };

  return {
    getState,
    updateModelWithNewWeights: (
      weights: Weights,
      latestLoss: number,
      averageValidationLoss: number | null,
    ) => {
      history.trainingLosses.push(latestLoss);

      if (averageValidationLoss !== null) {
        history.validationLosses.push({
          loss: averageValidationLoss,
          stepIndex: history.trainingLosses.length,
        });
      }

      index++;
      modelUnderTraining = {
        ...modelUnderTraining,
        ...weights,
      };

      if (index % STORE_INTERVAL === 0) {
        // Auto-safe
        writeNewCheckpointAndHistory();
      }
    },
    writeNewCheckpoint: writeNewCheckpointAndHistory,
    sampleBatch: (trainingData: TrainingExample[]) => {
      if (samplingState.type === "uniform") {
        const offset =
          Math.random() * (trainingData.length - MAX_TRAINING_DATA_PER_PASS);
        return trainingData
          .slice(offset, offset + MAX_TRAINING_DATA_PER_PASS)
          .map((dataPoint, index) => ({
            originalIndex: index,
            trainingData: dataPoint,
          }));
      }

      const weights = computeSamplingWeights(
        samplingState.lossRecord,
        trainingData.length,
      );
      const indices = sampleIndices(weights, MAX_TRAINING_DATA_PER_PASS);

      return indices.map((pickedIndex) => ({
        originalIndex: pickedIndex,
        trainingData: trainingData[pickedIndex]!,
      }));
    },
  };
};

export type StateStore = ReturnType<typeof createStateStore>;
