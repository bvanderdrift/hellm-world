import { writeNewCheckpoint } from "../model/model-io.ts";
import type {
  Model,
  ModelTrainingHistory,
  Weights,
} from "../model/model-types.ts";
import type { EndDefinition } from "./training.ts";

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
    const currentStepIndex = index;

    if (!endDefinition) {
      return {
        model: modelUnderTraining,
        history,
        currentStepIndex,
        startTime,
        isDone: false,
        percentDone: null,
      };
    }

    const percentDone = getPercentComplete(endDefinition);

    return {
      model: modelUnderTraining,
      history,
      currentStepIndex,
      startTime,
      isDone: percentDone >= 1,
      percentDone,
    };
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
    },
    writeNewCheckpoint: () => {
      writeNewCheckpoint(modelName, {
        history,
        weights: modelUnderTraining,
      });
    },
  };
};

export type StateStore = ReturnType<typeof createStateStore>;
