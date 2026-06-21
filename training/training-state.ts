import {
  writeCheckpoint,
  writeTrainingState,
} from "../model/model-checkpoint-io.ts";
import { getModelFolderPath } from "../model/model-io.ts";
import type { Model, Weights } from "../model/model-types.ts";
import type { TrainingExample } from "./doSingleTrainingPass.ts";
import {
  computeSamplingWeights,
  sampleIndices,
} from "./sampling/loss-weighted-sampling.ts";

const MAX_TRAINING_DATA_PER_PASS = 100;

const STORE_INTERVAL = 500;

export const createStateStore = (modelName: string, incomingModel: Model) => {
  const startTime = Date.now();
  let index = 0;
  let modelUnderTraining = incomingModel;
  const trainingState = incomingModel.trainingState;

  const getState = () => {
    return {
      model: modelUnderTraining,
      history: trainingState,
      startTime,
      isDone: false,
      percentDone: null,
      stepsInThisRun: index,
    };
  };

  const writeNewCheckpointAndHistory = () => {
    writeTrainingState(getModelFolderPath(modelName), trainingState);
    writeCheckpoint(
      modelName,
      trainingState.trainingLosses.length,
      modelUnderTraining,
    );
  };

  return {
    getState,
    updateModelWithNewWeights: (
      weights: Weights,
      losses: {
        trainingDataIndex: number;
        loss: number;
      }[],
      averageValidationLoss: number | null,
    ) => {
      const summedLosses = losses.reduce((a, b) => a + b.loss, 0);
      const latestAverageLoss = summedLosses / losses.length;

      trainingState.trainingLosses.push(latestAverageLoss);

      if (averageValidationLoss !== null) {
        trainingState.validationLosses.push({
          loss: averageValidationLoss,
          stepIndex: trainingState.trainingLosses.length,
        });
      }

      if (trainingState.samplerState.type === "loss-weighted") {
        for (let index = 0; index < losses.length; index++) {
          const { loss, trainingDataIndex } = losses[index]!;

          trainingState.samplerState.lossRecord[trainingDataIndex] = loss;
        }
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
    },
  };
};

export type StateStore = ReturnType<typeof createStateStore>;
