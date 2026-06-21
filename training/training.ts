import type { Model, Weights } from "../model/model-types.ts";
import {
  makeZeroVersion,
  operateCombinedWeights,
  operateSingleWeights,
} from "../model/model-helpers.ts";
import { readRawTrainingData } from "../model/model-io.ts";
import { prepareExampleData } from "./prepareExampleData.ts";
import type {
  InputMessagePayload,
  OutputMessagePayload,
  ResultsMessagePayload,
} from "./workers/training-worker.ts";
import {
  doSingleTrainingPass,
  type TrainingExample,
} from "./doSingleTrainingPass.ts";
import { createStateStore, type StateStore } from "./training-state.ts";
import { startKeyboardListening } from "./keyboard-listener.ts";
import { runValidationCheck } from "./validation.ts";
import { getWorkers, terminateWorkers } from "./workers/worker-mangement.ts";
import { cpus } from "os";
import { splitAcrossWorkers } from "./workers/batching.ts";
import { getLatestCheckpointModel } from "../model/model-checkpoint-io.ts";
import { sampleBatch } from "./sampling/sampling.ts";

const VALIDATION_INTERVAL = 20;

export const runTrainingCycle = async (
  modelName: string,
  workersCount: number,
) => {
  const modelLoaded = getLatestCheckpointModel(modelName);

  console.log(
    "Going to run training indefinitly. S to save checkpoint, Ctrl+C to exit",
  );

  const stateStore = createStateStore(modelName, modelLoaded);

  const trainingData = prepareExampleData(
    readRawTrainingData(modelName),
    modelLoaded.vocabulary,
    modelLoaded.trainingMaskSeparator ?? null,
  );

  let state = stateStore.getState();

  startKeyboardListening({ onSave: () => stateStore.writeNewCheckpoint() });

  while (true) {
    state = stateStore.getState();

    const trainingDataToWorkWith = sampleBatch(
      state.model.trainingState,
      trainingData,
    );

    const shouldRunValidation =
      state.trainingState.trainingLosses.length % VALIDATION_INTERVAL === 0;

    let averageValidationLoss: number | null = null;

    if (shouldRunValidation) {
      const prefix = `(step ${state.trainingState.trainingLosses.length})`;
      console.log(`${prefix} - Starting validation test`);
      averageValidationLoss = await runValidationCheck(modelName, state.model);

      console.log(`${prefix} - validation loss: ${averageValidationLoss}`);
    }

    const pickedTrainingData = trainingDataToWorkWith.map(
      ({ trainingData }) => trainingData,
    );

    const { losses, adjustedWeights } = await runTrainingPasses(
      state.model,
      pickedTrainingData,
      workersCount,
    );

    runNaNGuard(losses, pickedTrainingData);

    stateStore.updateModelWithNewWeights(
      adjustedWeights,
      losses.map((loss, index) => ({
        loss,
        trainingDataIndex: trainingDataToWorkWith[index]!.originalIndex,
      })),
      averageValidationLoss,
    );

    logStateProgress(stateStore);
  }

  stateStore.writeNewCheckpoint();

  terminateWorkers();

  console.log(`✅ Succesfully ran training loop for model ${modelName}`);
};

export const runNaNGuard = (
  losses: number[],
  dataPoints: TrainingExample[],
) => {
  const firstNaN = losses.findIndex((l) => !Number.isFinite(l));
  if (firstNaN === -1) {
    return;
  }

  const trainingData = dataPoints[firstNaN]!;
  console.error(`Training example: ${JSON.stringify(trainingData)}`);
  console.error(`All losses: ${losses}`);
  throw new Error("NaN detected");
};

export const logStateProgress = (store: StateStore) => {
  const { stepsInThisRun, startTime, trainingState } = store.getState();

  const lastLoss =
    trainingState.trainingLosses[trainingState.trainingLosses.length - 1]!;

  const currentStep = trainingState.trainingLosses.length + 1;

  const currentStepPadded = currentStep
    .toString()
    .padStart(currentStep.toString().length, "0");

  const totalDuration = Date.now() - startTime;
  const avgDuration = totalDuration / stepsInThisRun;

  const stepFormatted = `(step ${currentStepPadded}) - `;

  console.log(
    `${stepFormatted}Training pass done - average loss: ${lastLoss} - avg duration: ${Math.round(avgDuration)} ms`,
  );
};

let stepsCompleted = 0;

export const logSingleStepProgress = (
  durationMs: number,
  stepsExpected: number,
) => {
  stepsCompleted++;
  const localCycleStepsCompleted = stepsCompleted % stepsExpected;

  const progressPct = localCycleStepsCompleted / stepsExpected;
  const progressBarComplete = Math.floor(progressPct * PROGRESS_BAR_LENGTH);

  const progressBar =
    "[" +
    "-".repeat(progressBarComplete) +
    " ".repeat(PROGRESS_BAR_LENGTH - progressBarComplete) +
    "]";

  process.stdout.write(
    `\r${progressBar} (${(progressPct * 100).toFixed(0)}%) - Duration: ${durationMs}ms`,
  );
};

const cpuCount = cpus().length;
const PROGRESS_BAR_LENGTH = 20;

const runTrainingPasses = async (
  model: Model,
  trainingData: TrainingExample[],
  workersCount: number,
): Promise<{
  losses: number[];
  adjustedWeights: Weights;
}> => {
  const effectiveCpuCount = Math.min(workersCount, cpuCount);

  const onStepComplete = (durationMs: number) =>
    logSingleStepProgress(durationMs, trainingData.length);

  if (effectiveCpuCount === 1) {
    return doSingleTrainingPass(model, trainingData, onStepComplete);
  }

  const workers = getWorkers(effectiveCpuCount);

  const splitData = splitAcrossWorkers(trainingData, workers.length);

  const results = await Promise.all(
    splitData.map(async (singleBatch, workerIndex) => {
      return runTrainingWorker(
        model,
        singleBatch,
        workers[workerIndex]!,
        onStepComplete,
      );
    }),
  );

  // Newline to finish the progress writing
  console.log("");

  let losses: number[] = [];
  let summedWeightAdjustements: Weights = makeZeroVersion(model);

  for (let index = 0; index < results.length; index++) {
    const result = results[index]!;
    const batchSize = splitData[index]!.length;

    losses.push(...result.losses);
    summedWeightAdjustements = operateCombinedWeights(
      summedWeightAdjustements,
      result.adjustedWeights,
      (v1, v2) => v1 + v2 * batchSize,
    );
  }

  return {
    adjustedWeights: operateSingleWeights(
      summedWeightAdjustements,
      (v) => v / trainingData.length,
    ),
    losses,
  };
};

const runTrainingWorker = async (
  model: Model,
  trainingData: TrainingExample[],
  trainingWorker: Worker,
  onStepComplete: (durationMs: number) => void,
): Promise<ResultsMessagePayload> => {
  return new Promise((resolve, reject) => {
    trainingWorker.onerror = (e) => reject(e);
    trainingWorker.onmessageerror = (e) => reject(e);
    trainingWorker.onmessage = (event: MessageEvent<OutputMessagePayload>) => {
      if (event.data.type === "results") {
        resolve(event.data);
        return;
      }

      onStepComplete(event.data.durationMs);
    };

    const input: InputMessagePayload = {
      model,
      trainingData,
    };

    trainingWorker.postMessage(input);
  });
};
