import type { Model, Weights } from "../model/model-types.ts";
import {
  makeZeroVersion,
  operateCombinedWeights,
  operateSingleWeights,
} from "../model/model-helpers.ts";
import {
  getLatestCheckpointModel,
  writeNewCheckpoint,
} from "../model/model-io.ts";
import { prepareTrainingData } from "./prepareTrainingData.ts";
import { cpus } from "os";
import type {
  InputMessagePayload,
  OutputMessagePayload,
} from "./training-worker.ts";
import { doSingleTrainingPass } from "./doSingleTrainingPass.ts";

const MAX_TRAINING_DATA_PER_PASS = 100;

export const doTrainingLoopAndStoreCheckpoint = async (
  modelName: string,
  steps: number,
) => {
  const { historyLosses, model: modelLoaded } =
    getLatestCheckpointModel(modelName);

  let model = modelLoaded;

  if (steps <= 0) {
    throw new Error(`steps has to be a positive integer, received: ${steps}`);
  }

  const trainingData = shuffleArray(
    prepareTrainingData(modelName, model.vocabulary),
  );

  const startTime = Date.now();

  for (let index = 0; index < steps; index++) {
    const offset =
      Math.random() * (trainingData.length - MAX_TRAINING_DATA_PER_PASS);
    const trainingDataToWorkWith = trainingData.slice(
      offset,
      offset + MAX_TRAINING_DATA_PER_PASS,
    );

    const { averageLoss, adjustedWeights } = await runTrainingPasses(
      model,
      trainingDataToWorkWith,
    );

    const indexPadded = (index + 1)
      .toString()
      .padStart(steps.toString().length, "0");

    const totalDuration = Date.now() - startTime;
    const avgDuration = totalDuration / (index + 1);

    console.log(
      `(${indexPadded}/${steps}) Training pass done - average loss: ${averageLoss} - avg duration: ${Math.round(avgDuration)} ms`,
    );
    historyLosses.push(averageLoss);
    model = {
      ...model,
      ...adjustedWeights,
    };
  }

  terminateWorkers();

  writeNewCheckpoint(modelName, {
    historyLosses,
    weights: model,
  });

  console.log(`✅ Succesfully ran training loop for model ${modelName}`);
};

const cpuCount = cpus().length;
const MULTITHREAD_FLAG = false;
const effectiveCpuCount = MULTITHREAD_FLAG ? 1 : cpuCount;

const workers = new Array(effectiveCpuCount)
  .fill(0)
  .map(() => new Worker("./training/training-worker.ts"));

export const terminateWorkers = () => {
  workers.forEach((worker) => worker.terminate());
};

const runTrainingPasses = async (
  model: Model,
  trainingData: string[][],
): Promise<{
  averageLoss: number;
  adjustedWeights: Weights;
}> => {
  if (effectiveCpuCount === 1) {
    return doSingleTrainingPass(model, trainingData);
  }

  const batchSize = Math.ceil(trainingData.length / workers.length);

  const splitData = workers.map((_, cpuIndex) => {
    return trainingData.slice(cpuIndex * batchSize, (cpuIndex + 1) * batchSize);
  });

  const results = await Promise.all(
    splitData.map(async (singleBatch, workerIndex) => {
      return runTrainingWorker(model, singleBatch, workers[workerIndex]!);
    }),
  );

  let summedLoss = 0;
  let summedWeightAdjustements: Weights = makeZeroVersion(model);

  for (let index = 0; index < results.length; index++) {
    const result = results[index]!;
    const batchSize = splitData[index]!.length;

    summedLoss += result.averageLoss * batchSize;
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
    averageLoss: summedLoss / trainingData.length,
  };
};

const runTrainingWorker = async (
  model: Model,
  trainingData: string[][],
  trainingWorker: Worker,
): Promise<{
  averageLoss: number;
  adjustedWeights: Weights;
}> => {
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

const shuffleArray = <T>(values: T[]): T[] =>
  values
    .map((value) => ({ value, sort: Math.random() }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ value }) => value);
