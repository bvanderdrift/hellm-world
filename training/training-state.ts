import type { ModelTrainingState } from "../model/model-types.ts";

const STORE_INTERVAL = 500;

export const createStateStore = (
  onSave: () => Promise<void> | void,
  trainingState: ModelTrainingState,
) => {
  const startTime = Date.now();
  let index = 0;

  const getState = () => {
    return {
      trainingState,
      startTime,
      stepsInThisRun: index,
    };
  };

  return {
    getState,
    notifyCycleComplete: async (
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

      if (index % STORE_INTERVAL === 0) {
        // Auto-safe
        await onSave();
      }
    },
  };
};

export type StateStore = ReturnType<typeof createStateStore>;
