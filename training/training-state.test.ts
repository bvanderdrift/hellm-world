import { describe, expect, it } from "vitest";
import type {
  LossRecord,
  Model,
  SamplerState,
  Weights,
} from "../model/model-types.ts";
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
    expect(store.getState().trainingState.trainingLosses).toHaveLength(1);
    expect(store.getState().trainingState.trainingLosses[0]).toBeCloseTo(
      0.4,
      10,
    );
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

    const { trainingState: history, stepsInThisRun } = store.getState();
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

    expect(store.getState().trainingState.validationLosses).toEqual([
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

    expect(store.getState().trainingState.validationLosses).toEqual([]);
  });

  it("writes each example's loss into the loss-weighted sampler record by global index", () => {
    const lossRecord: LossRecord = {};
    const store = createStateStore(
      "test",
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
    const store = createStateStore(
      "test",
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

