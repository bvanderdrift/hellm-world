import {
  getModelFolderPath,
  writeHistory,
  writeNewCheckpoint,
} from "../model/model-io.ts";
import type {
  Model,
  ModelTrainingHistory,
  Weights,
} from "../model/model-types.ts";
import type { EndDefinition } from "./training.ts";

const STORE_INTERVAL = 500;

export const createStateStore = (
  endDefinition: EndDefinition | null,
  modelName: string,
  incomingModel: Model,
  history: ModelTrainingHistory,
) => {
  const startTime = Date.now();
  let index = 0;
  let modelUnderTraining = incomingModel;

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
        history,
        startTime,
        isDone: false,
        percentDone: null,
        stepsInThisRun: index,
      };
    }

    const percentDone = getPercentComplete(endDefinition);

    return {
      model: modelUnderTraining,
      history,
      startTime,
      isDone: percentDone >= 1,
      percentDone,
      stepsInThisRun: index,
    };
  };

  const writeNewCheckpointAndHistory = () => {
    writeHistory(getModelFolderPath(modelName), history);
    writeNewCheckpoint(modelName, modelUnderTraining);
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
  };
};

export type StateStore = ReturnType<typeof createStateStore>;
