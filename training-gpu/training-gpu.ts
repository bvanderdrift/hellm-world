import type { Model, Weights } from "../model/model-types.ts";
import {
  makeZeroVersion,
  operateCombinedWeights,
} from "../model/model-helpers.ts";
import { getModelFolderPath, readRawTrainingData } from "../model/model-io.ts";
import {
  extractWeightsFromGpu,
  loadWeightsIntoGpu,
  type WeightGPUBuffers,
} from "../model-gpu/model-weights-gpu.ts";
import {
  doSingleTrainingPass,
  type TrainingExample,
} from "../training/doSingleTrainingPass.ts";
import { startKeyboardListening } from "../training/keyboard-listener.ts";
import { prepareExampleData } from "../training/prepareExampleData.ts";
import { sampleBatch } from "../training/sampling/sampling.ts";
import {
  logSingleStepProgress,
  logStateProgress,
  runNaNGuard,
} from "../training/training.ts";
import {
  writeCheckpoint,
  writeTrainingState,
} from "../model/model-checkpoint-io.ts";
import { createStateStore } from "../training/training-state.ts";
import { addWeights } from "./addWeights.ts";

export const runTrainingCycleGPU = async (model: Model) => {
  const weightBuffers = await loadWeightsIntoGpu(model.counts, model);

  const onSave = async () => {
    const weights = await extractWeightsFromGpu(weightBuffers);
    writeTrainingState(getModelFolderPath(model.name), model.trainingState);
    writeCheckpoint(
      model.name,
      model.trainingState.trainingLosses.length,
      weights,
    );
  };

  const stateStore = createStateStore(onSave, model.trainingState);

  const trainingData = prepareExampleData(
    readRawTrainingData(model.name),
    model.vocabulary,
    model.trainingMaskSeparator ?? null,
  );

  let state = stateStore.getState();

  startKeyboardListening({
    onSave,
  });

  while (true) {
    state = stateStore.getState();

    const trainingDataToWorkWith = sampleBatch(
      state.trainingState,
      trainingData,
    );

    const onStepComplete = (durationMs: number) =>
      logSingleStepProgress(durationMs, trainingDataToWorkWith.length);

    const pickedTrainingData = trainingDataToWorkWith.map(
      ({ trainingData }) => trainingData,
    );

    const { weightAdjustments: adjustedWeights, losses } =
      await doSingleTrainingPass(model, pickedTrainingData, onStepComplete);

    // Newline to finish the progress writing
    console.log("");

    const weightAdjustmentsGPU = loadWeightsIntoGpu(
      model.counts,
      adjustedWeights,
    );

    addWeights(weightBuffers, weightAdjustmentsGPU);

    runNaNGuard(losses, pickedTrainingData);

    stateStore.notifyCycleComplete(
      losses.map((loss, index) => ({
        loss,
        trainingDataIndex: trainingDataToWorkWith[index]!.originalIndex,
      })),
      null, // For now no validation loss implementation on GPU
    );

    logStateProgress(stateStore);
  }
};
