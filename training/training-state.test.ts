import { describe, expect, it } from "vitest";
import type {
  LossRecord,
  Model,
  SamplerState,
  Weights,
} from "../model/model-types.ts";
import type { TrainingExample } from "./doSingleTrainingPass.ts";
import { createStateStore } from "./training-state.ts";

// createStateStore only ever reads `trainingState` and spreads in `weights`, so
// a structurally-minimal model is enough for these unit tests. The filesystem is
// never touched as long as we stay under STORE_INTERVAL (500) calls and never
// call writeNewCheckpoint().
const makeModel = (samplerState: SamplerState): Model =>
  ({
    vocabulary: [],
    counts: {
      transformers: 1,
      attentionHeads: 1,
      hiddenDimensions: 1,
      mlpMultiple: 1,
    },
    embeddings: { vectors: 0, dimensions: 0, values: new Float32Array() },
    unembeddings: { vectors: 0, dimensions: 0, values: new Float32Array() },
    transformers: [],
    trainingState: {
      trainingLosses: [],
      validationLosses: [],
      samplerState,
    },
  }) as unknown as Model;

const NO_WEIGHTS = {} as Weights;

const makeData = (count: number): TrainingExample[] =>
  Array.from({ length: count }, (_, i) => ({
    sequence: [String(i)],
    maskBeforeIndex: null,
  }));

describe("createStateStore - updateModelWithNewWeights", () => {
  it("records the mean of the batch losses as the step's training loss", () => {
    const store = createStateStore("test", makeModel({ type: "uniform" }));

    store.updateModelWithNewWeights(
      NO_WEIGHTS,
      [
        { trainingDataIndex: 0, loss: 0.2 },
        { trainingDataIndex: 1, loss: 0.4 },
        { trainingDataIndex: 2, loss: 0.6 },
      ],
      null,
    );

    // mean(0.2, 0.4, 0.6) = 0.4
    expect(store.getState().history.trainingLosses).toHaveLength(1);
    expect(store.getState().history.trainingLosses[0]).toBeCloseTo(0.4, 10);
  });

  it("appends one training loss per call and advances stepsInThisRun", () => {
    const store = createStateStore("test", makeModel({ type: "uniform" }));

    store.updateModelWithNewWeights(
      NO_WEIGHTS,
      [{ trainingDataIndex: 0, loss: 1 }],
      null,
    );
    store.updateModelWithNewWeights(
      NO_WEIGHTS,
      [{ trainingDataIndex: 0, loss: 2 }],
      null,
    );

    const { history, stepsInThisRun } = store.getState();
    expect(history.trainingLosses).toEqual([1, 2]);
    expect(stepsInThisRun).toBe(2);
  });

  it("records validation loss tagged with the current step index", () => {
    const store = createStateStore("test", makeModel({ type: "uniform" }));

    store.updateModelWithNewWeights(
      NO_WEIGHTS,
      [{ trainingDataIndex: 0, loss: 1 }],
      null,
    );
    store.updateModelWithNewWeights(
      NO_WEIGHTS,
      [{ trainingDataIndex: 0, loss: 1 }],
      0.5,
    );

    expect(store.getState().history.validationLosses).toEqual([
      { loss: 0.5, stepIndex: 2 },
    ]);
  });

  it("does not record a validation loss when none is provided", () => {
    const store = createStateStore("test", makeModel({ type: "uniform" }));

    store.updateModelWithNewWeights(
      NO_WEIGHTS,
      [{ trainingDataIndex: 0, loss: 1 }],
      null,
    );

    expect(store.getState().history.validationLosses).toEqual([]);
  });

  it("writes each example's loss into the loss-weighted sampler record by global index", () => {
    const lossRecord: LossRecord = {};
    const store = createStateStore("test",
      makeModel({ type: "loss-weighted", lossRecord }),
    );

    store.updateModelWithNewWeights(
      NO_WEIGHTS,
      [
        { trainingDataIndex: 5, loss: 1.5 },
        { trainingDataIndex: 42, loss: 0.25 },
      ],
      null,
    );

    expect(lossRecord[5]).toBe(1.5);
    expect(lossRecord[42]).toBe(0.25);
    expect(Object.keys(lossRecord)).toHaveLength(2);
  });

  it("overwrites a previously recorded loss for a re-sampled example", () => {
    const lossRecord: LossRecord = { 7: 9.9 };
    const store = createStateStore("test",
      makeModel({ type: "loss-weighted", lossRecord }),
    );

    store.updateModelWithNewWeights(
      NO_WEIGHTS,
      [{ trainingDataIndex: 7, loss: 0.1 }],
      null,
    );

    expect(lossRecord[7]).toBe(0.1);
  });

  it("does not throw or record losses when sampling uniformly", () => {
    const store = createStateStore("test", makeModel({ type: "uniform" }));

    expect(() =>
      store.updateModelWithNewWeights(
        NO_WEIGHTS,
        [{ trainingDataIndex: 0, loss: 1 }],
        null,
      ),
    ).not.toThrow();
  });
});

describe("createStateStore - sampleBatch (uniform)", () => {
  it("returns a full contiguous window of MAX_TRAINING_DATA_PER_PASS examples", () => {
    const store = createStateStore("test", makeModel({ type: "uniform" }));
    const data = makeData(120);

    const batch = store.sampleBatch(data);

    expect(batch).toHaveLength(100);

    // The window is a contiguous slice: consecutive sequence values.
    const first = Number(batch[0]!.trainingData.sequence[0]);
    for (let i = 0; i < batch.length; i++) {
      expect(Number(batch[i]!.trainingData.sequence[0])).toBe(first + i);
    }
  });
});

describe("createStateStore - sampleBatch (loss-weighted)", () => {
  it("returns MAX_TRAINING_DATA_PER_PASS distinct in-range indices", () => {
    const lossRecord: LossRecord = {};
    const store = createStateStore("test",
      makeModel({ type: "loss-weighted", lossRecord }),
    );
    const data = makeData(120);

    const batch = store.sampleBatch(data);
    const indices = batch.map((b) => b.originalIndex);

    expect(batch).toHaveLength(100);
    expect(new Set(indices).size).toBe(100);
    for (const i of indices) {
      expect(i).toBeGreaterThanOrEqual(0);
      expect(i).toBeLessThan(120);
    }
  });

  it("returns the example that lives at each sampled global index", () => {
    const store = createStateStore("test",
      makeModel({ type: "loss-weighted", lossRecord: {} }),
    );
    const data = makeData(120);

    const batch = store.sampleBatch(data);

    for (const { originalIndex, trainingData } of batch) {
      expect(Number(trainingData.sequence[0])).toBe(originalIndex);
    }
  });

  it("strongly prefers a high-loss example over low-loss ones", () => {
    const lossRecord: LossRecord = {};
    for (let i = 0; i < 120; i++) {
      lossRecord[i] = i === 0 ? 1000 : 0.001;
    }
    const store = createStateStore("test",
      makeModel({ type: "loss-weighted", lossRecord }),
    );
    const data = makeData(120);

    const batch = store.sampleBatch(data);
    const indices = batch.map((b) => b.originalIndex);

    // Index 0 carries ~all the weight; over a 100-of-120 draw it is
    // effectively certain to be included.
    expect(indices).toContain(0);
  });
});

// "Weird violations": the originalIndex returned must be the *global* dataset
// index of the example, usable as a lossRecord key. The uniform branch is the
// trap here — it slices a window, and it's easy to return the window-local
// index (0..99) or a fractional one (a non-floored random offset).
describe("createStateStore - sampleBatch index integrity", () => {
  // 120 > 100 so the window offset is non-zero (and can be fractional),
  // exposing both the local-index and fractional-offset bugs.
  const DATASET_SIZE = 120;

  it("uniform: originalIndex equals the example's true global index", () => {
    const store = createStateStore("test", makeModel({ type: "uniform" }));
    const data = makeData(DATASET_SIZE);

    const batch = store.sampleBatch(data);

    for (const { originalIndex, trainingData } of batch) {
      // makeData encodes the global index as the sequence's only token, so a
      // correct originalIndex must match it exactly.
      expect(originalIndex).toBe(Number(trainingData.sequence[0]));
    }
  });

  it("uniform: every originalIndex is an integer", () => {
    const store = createStateStore("test", makeModel({ type: "uniform" }));
    const data = makeData(DATASET_SIZE);

    const batch = store.sampleBatch(data);

    for (const { originalIndex } of batch) {
      expect(Number.isInteger(originalIndex)).toBe(true);
    }
  });

  it("loss-weighted: throws when the dataset is smaller than one batch", () => {
    const store = createStateStore("test",
      makeModel({ type: "loss-weighted", lossRecord: {} }),
    );
    // Only 50 examples but a batch wants MAX_TRAINING_DATA_PER_PASS (100).
    const data = makeData(50);

    expect(() => store.sampleBatch(data)).toThrow();
  });
});
