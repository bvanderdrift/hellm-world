
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
      .map(() => new Worker("./training/workers/training-worker.ts"));

  return cachedWorkers;
};