import { describe, expect, it } from "vitest";
import {
  createLossRecord,
  computeSamplingWeights,
  sampleIndices,
} from "./loss-weighted-sampling.ts";

describe("createLossRecord", () => {
  it("starts with no recorded losses", () => {
    const record = createLossRecord();
    expect(record.size).toBe(0);
  });
});

describe("computeSamplingWeights", () => {
  it("returns a weight for every example", () => {
    const record = createLossRecord();
    const weights = computeSamplingWeights(record, 5);
    expect(weights).toHaveLength(5);
  });

  it("returns uniform weights when no losses are recorded", () => {
    const record = createLossRecord();
    const weights = computeSamplingWeights(record, 4);
    expect(weights[0]).toBe(weights[1]);
    expect(weights[1]).toBe(weights[2]);
    expect(weights[2]).toBe(weights[3]);
  });

  it("returns weights proportional to loss", () => {
    const record = createLossRecord();
    record.set(0, 8.0);
    record.set(1, 2.0);
    const weights = computeSamplingWeights(record, 2);
    // Ratio should be approximately 8:2 = 4:1 (epsilon makes it approximate)
    expect(weights[0]! / weights[1]!).toBeCloseTo(4.0, 1);
  });

  it("gives unseen examples the maximum observed loss", () => {
    const record = createLossRecord();
    record.set(0, 3.0);
    record.set(1, 7.0);
    // index 2 has never been seen — should get the max observed loss (7.0)
    const weights = computeSamplingWeights(record, 3);
    expect(weights[2]).toBeCloseTo(weights[1]!, 5);
  });

  it("adds epsilon so zero-loss examples still have non-zero weight", () => {
    const record = createLossRecord();
    record.set(0, 5.0);
    record.set(1, 0.0);
    const weights = computeSamplingWeights(record, 2);
    expect(weights[1]).toBeGreaterThan(0);
  });

  it("all weights are positive", () => {
    const record = createLossRecord();
    record.set(0, 2.0);
    record.set(1, 0.0);
    record.set(2, 5.0);
    const weights = computeSamplingWeights(record, 3);
    for (const w of weights) {
      expect(w).toBeGreaterThan(0);
    }
  });
});

describe("sampleIndices", () => {
  it("returns exactly the requested number of indices", () => {
    const weights = [1, 1, 1, 1, 1];
    const indices = sampleIndices(weights, 3);
    expect(indices).toHaveLength(3);
  });

  it("never returns duplicate indices", () => {
    const weights = [1, 1, 1, 1, 1];
    const indices = sampleIndices(weights, 4);
    const unique = new Set(indices);
    expect(unique.size).toBe(indices.length);
  });

  it("returns all indices when count equals the number of weights", () => {
    const weights = [1, 2, 3];
    const indices = sampleIndices(weights, 3);
    expect(indices.sort((a, b) => a - b)).toEqual([0, 1, 2]);
  });

  it("throws when count exceeds the number of weights", () => {
    const weights = [1, 1];
    expect(() => sampleIndices(weights, 10)).toThrow();
  });

  it("returns indices within the valid range", () => {
    const weights = [1, 2, 3, 4, 5];
    const indices = sampleIndices(weights, 3);
    for (const idx of indices) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(weights.length);
    }
  });

  it("favors higher-weighted indices over many trials", () => {
    // Index 0 has weight 100, rest have weight 1 → P(0) ≈ 0.92
    const weights = [100, 1, 1, 1, 1, 1, 1, 1, 1, 1];

    let timesIndex0Picked = 0;
    const trials = 200;
    for (let i = 0; i < trials; i++) {
      const indices = sampleIndices(weights, 1);
      if (indices[0] === 0) timesIndex0Picked++;
    }

    expect(timesIndex0Picked / trials).toBeGreaterThan(0.5);
  });
});
