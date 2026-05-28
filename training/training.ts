import type { Model, Weights } from "../model/model-types.ts";
import {
  makeZeroVersion,
  operateCombinedWeights,
  operateSingleWeights,
} from "../model/model-helpers.ts";
import {
  getLatestCheckpointModel,
  getModelFolderPath,
  getModelHistory,
  readRawTrainingData,
} from "../model/model-io.ts";
import { prepareExampleData } from "./prepareExampleData.ts";
import { cpus } from "os";
import type {
  InputMessagePayload,
  OutputMessagePayload,
} from "./training-worker.ts";
import {
  doSingleTrainingPass,
  type TrainingExample,
} from "./doSingleTrainingPass.ts";
import { createStateStore, type StateStore } from "./training-state.ts";
import { startKeyboardListening } from "./keyboard-listener.ts";
import { runValidationCheck } from "./validation.ts";
import {
  computeSamplingWeights,
  createLossRecord,
  sampleIndices,
} from "./loss-weighted-sampling.ts";

const MAX_TRAINING_DATA_PER_PASS = 100;

const VALIDATION_INTERVAL = 20;

export type EndDefinition = { type: "minutes" | "steps"; count: number };

export const doTrainingLoopAndStoreCheckpoint = async (
  modelName: string,
  endDefinition: EndDefinition | null,
  workersCount: number,
) => {
  const modelFolderPath = getModelFolderPath(modelName);
  const history = getModelHistory(modelFolderPath);
  const modelLoaded = getLatestCheckpointModel(modelName);

  if (endDefinition && endDefinition.count <= 0) {
    throw new Error(
      `End count has to be a positive integer, received: ${endDefinition.count}`,
    );
  }

  if (!endDefinition) {
    console.log(
      "Going to run training indefinitly. S to save checkpoint, Ctrl+C to exit",
    );
  } else if (endDefinition.type === "steps") {
    console.log(`Going to run training for ${endDefinition.count} steps`);
  } else {
    console.log(
      `Going to run training for ${endDefinition.count} minutes (~${(endDefinition.count / 60).toFixed(1)} hours)`,
    );
  }

  const stateStore = createStateStore(
    endDefinition,
    modelName,
    modelLoaded,
    history,
  );

  const trainingData = prepareExampleData(
    readRawTrainingData(modelName),
    modelLoaded.vocabulary,
    modelLoaded.trainingMaskSeparator ?? null,
  );
  const lossRecord = createLossRecord();

  let state = stateStore.getState();

  startKeyboardListening(stateStore);

  while (!(state = stateStore.getState()).isDone) {
    const weights = computeSamplingWeights(lossRecord, trainingData.length);
    const indices = sampleIndices(weights, MAX_TRAINING_DATA_PER_PASS);
    const trainingDataToWorkWith = indices.map((pickedIndex) => ({
      originalIndex: pickedIndex,
      trainingData: trainingData[pickedIndex]!,
    }));

    const shouldRunValidation =
      state.history.trainingLosses.length % VALIDATION_INTERVAL === 0;

    let averageValidationLoss: number | null = null;

    if (shouldRunValidation) {
      console.log(`Starting validation test`);
      averageValidationLoss = await runValidationCheck(modelName, state.model);
    }

    const { losses, adjustedWeights } = await runTrainingPasses(
      state.model,
      trainingDataToWorkWith.map(({ trainingData }) => trainingData),
      workersCount,
    );

    let summedLoss = 0;
    for (let index = 0; index < losses.length; index++) {
      const loss = losses[index]!;
      summedLoss += loss;
      const { originalIndex } = trainingDataToWorkWith[index]!;

      lossRecord.set(originalIndex, loss);
    }

    stateStore.updateModelWithNewWeights(
      adjustedWeights,
      summedLoss / losses.length,
      averageValidationLoss,
    );

    logStateProgress(stateStore, averageValidationLoss);
  }

  stateStore.writeNewCheckpoint();

  terminateWorkers();

  console.log(`✅ Succesfully ran training loop for model ${modelName}`);
};

const logStateProgress = (
  store: StateStore,
  newValidationLossAverage: number | null,
) => {
  const {
    currentStepIndex: index,
    percentDone: newPercentDone,
    startTime,
    history,
  } = store.getState();

  const lastLoss = history.trainingLosses[history.trainingLosses.length - 1]!;

  const indexPadded = index.toString().padStart(index.toString().length, "0");

  const totalDuration = Date.now() - startTime;
  const avgDuration = totalDuration / index;

  const percentFormatted =
    newPercentDone === null
      ? ""
      : `(${(newPercentDone * 100).toFixed(2)}% complete) `;

  const stepFormatted = `(step ${indexPadded}) - `;

  console.log(
    `${stepFormatted}${percentFormatted}Training pass done - average loss: ${lastLoss} - avg duration: ${Math.round(avgDuration)} ms`,
  );
  if (newValidationLossAverage !== null) {
    console.log(`${stepFormatted}validation loss: ${newValidationLossAverage}`);
  }
};

const cpuCount = cpus().length;

let cachedWorkers: Worker[] | null = null;

export const terminateWorkers = () => {
  cachedWorkers?.forEach((worker) => worker.terminate());
};

export const getWorkers = (count: number) => {
  if (cachedWorkers && cachedWorkers.length !== count) {
    console.log(`Workers were initialized at different size`);
  }

  cachedWorkers =
    cachedWorkers ??
    new Array(count)
      .fill(0)
      .map(() => new Worker("./training/training-worker.ts"));

  return cachedWorkers;
};

const runTrainingPasses = async (
  model: Model,
  trainingData: TrainingExample[],
  workersCount: number,
): Promise<{
  losses: number[];
  adjustedWeights: Weights;
}> => {
  const effectiveCpuCount = Math.min(workersCount, cpuCount);

  if (effectiveCpuCount === 1) {
    return doSingleTrainingPass(model, trainingData);
  }

  const workers = getWorkers(effectiveCpuCount);

  const batchSize = Math.ceil(trainingData.length / workers.length);

  const splitData = workers.map((_, cpuIndex) => {
    return trainingData.slice(cpuIndex * batchSize, (cpuIndex + 1) * batchSize);
  });

  const results = await Promise.all(
    splitData.map(async (singleBatch, workerIndex) => {
      return runTrainingWorker(model, singleBatch, workers[workerIndex]!);
    }),
  );

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
): Promise<OutputMessagePayload> => {
  return new Promise((resolve, reject) => {
    trainingWorker.onerror = (e) => reject(e);
    trainingWorker.onmessageerror = (e) => reject(e);
    trainingWorker.onmessage = (event: MessageEvent<OutputMessagePayload>) => {
      resolve(event.data);
    };

    const input: InputMessagePayload = {
      model,
      trainingData,
    };

    trainingWorker.postMessage(input);
  });
};
