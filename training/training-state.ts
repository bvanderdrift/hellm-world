import {
  writeCheckpoint,
  writeTrainingState,
} from "../model/model-checkpoint-io.ts";
import { getModelFolderPath } from "../model/model-io.ts";
import type { Model, Weights } from "../model/model-types.ts";

const STORE_INTERVAL = 500;

export const createStateStore = (modelName: string, incomingModel: Model) => {
  const startTime = Date.now();
  let index = 0;
  let modelUnderTraining = incomingModel;
  const trainingState = incomingModel.trainingState;

  const getState = () => {
    return {
      model: modelUnderTraining,
      trainingState,
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
  };
};

export type StateStore = ReturnType<typeof createStateStore>;
