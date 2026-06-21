import { describe, expect, it } from "vitest";
import type {
  LossRecord,
  ModelTrainingState,
  SamplerState,
} from "../../model/model-types.ts";
import type { TrainingExample } from "../doSingleTrainingPass.ts";
import { sampleBatch } from "./sampling.ts";

const makeTrainingState = (samplerState: SamplerState): ModelTrainingState => ({
  trainingLosses: [],
  validationLosses: [],
  samplerState,
});

const makeData = (count: number): TrainingExample[] =>
  Array.from({ length: count }, (_, i) => ({
    sequence: [String(i)],
    maskBeforeIndex: null,
  }));

describe("sampleBatch (uniform)", () => {
  it("returns a full contiguous window of MAX_TRAINING_DATA_PER_PASS examples", () => {
    const data = makeData(120);

    const batch = sampleBatch(makeTrainingState({ type: "uniform" }), data);

    expect(batch).toHaveLength(100);

    // The window is a contiguous slice: consecutive sequence values.
    const first = Number(batch[0]!.trainingData.sequence[0]);
    for (let i = 0; i < batch.length; i++) {
      expect(Number(batch[i]!.trainingData.sequence[0])).toBe(first + i);
    }
  });
});

describe("sampleBatch (loss-weighted)", () => {
  it("returns MAX_TRAINING_DATA_PER_PASS distinct in-range indices", () => {
    const lossRecord: LossRecord = {};
    const data = makeData(120);

    const batch = sampleBatch(
      makeTrainingState({ type: "loss-weighted", lossRecord }),
      data,
    );
    const indices = batch.map((b) => b.originalIndex);

    expect(batch).toHaveLength(100);
    expect(new Set(indices).size).toBe(100);
    for (const i of indices) {
      expect(i).toBeGreaterThanOrEqual(0);
      expect(i).toBeLessThan(120);
    }
  });

  it("returns the example that lives at each sampled global index", () => {
    const data = makeData(120);

    const batch = sampleBatch(
      makeTrainingState({ type: "loss-weighted", lossRecord: {} }),
      data,
    );

    for (const { originalIndex, trainingData } of batch) {
      expect(Number(trainingData.sequence[0])).toBe(originalIndex);
    }
  });

  it("strongly prefers a high-loss example over low-loss ones", () => {
    const lossRecord: LossRecord = {};
    for (let i = 0; i < 120; i++) {
      lossRecord[i] = i === 0 ? 1000 : 0.001;
    }
    const data = makeData(120);

    const batch = sampleBatch(
      makeTrainingState({ type: "loss-weighted", lossRecord }),
      data,
    );
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
describe("sampleBatch index integrity", () => {
  // 120 > 100 so the window offset is non-zero (and can be fractional),
  // exposing both the local-index and fractional-offset bugs.
  const DATASET_SIZE = 120;

  it("uniform: originalIndex equals the example's true global index", () => {
    const data = makeData(DATASET_SIZE);

    const batch = sampleBatch(makeTrainingState({ type: "uniform" }), data);

    for (const { originalIndex, trainingData } of batch) {
      // makeData encodes the global index as the sequence's only token, so a
      // correct originalIndex must match it exactly.
      expect(originalIndex).toBe(Number(trainingData.sequence[0]));
    }
  });

  it("uniform: every originalIndex is an integer", () => {
    const data = makeData(DATASET_SIZE);

    const batch = sampleBatch(makeTrainingState({ type: "uniform" }), data);

    for (const { originalIndex } of batch) {
      expect(Number.isInteger(originalIndex)).toBe(true);
    }
  });

  it("loss-weighted: throws when the dataset is smaller than one batch", () => {
    // Only 50 examples but a batch wants MAX_TRAINING_DATA_PER_PASS (100).
    const data = makeData(50);

    expect(() =>
      sampleBatch(
        makeTrainingState({ type: "loss-weighted", lossRecord: {} }),
        data,
      ),
    ).toThrow();
  });
});
