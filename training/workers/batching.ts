import type { TrainingExample } from "../doSingleTrainingPass.ts";

export const splitAcrossWorkers = (
  trainingData: TrainingExample[],
  workerCount: number,
) => {
  const base = Math.floor(trainingData.length / workerCount);
  const remainder = trainingData.length % workerCount;

  const splitData = new Array<TrainingExample[]>(workerCount);

  let splicedCount = 0;

  for (let workerIndex = 0; workerIndex < workerCount; workerIndex++) {
    const size = base + (workerIndex < remainder ? 1 : 0);

    splitData[workerIndex] = trainingData.slice(
      splicedCount,
      splicedCount + size,
    );

    splicedCount += size;
  }

  return splitData;
};
