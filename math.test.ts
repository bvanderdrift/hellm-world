import { describe, expect, it } from "vitest";
import {
  calculateStandardDeviation,
  dotProduct,
  mean,
  relu,
  softmax,
  sum,
} from "./math.ts";

describe("sum", () => {
  it("adds numbers together", () => {
    const values = [1, 6, -3, 0, 2.5, -3.7];

    expect(sum(values)).toBe(2.8);
  });
});

describe("dotProduct", () => {
  it("multiplies matching vector entries and sums them", () => {
    expect(dotProduct([1, 2, 3], [4, 5, 6])).toBe(32);
  });

  it("throws when vector sizes do not match", () => {
    expect(() => dotProduct([1, 2], [3])).toThrow("not overlapping");
  });
});

describe("softmax", () => {
  it("returns probabilities that sum to 1", () => {
    const probabilities = softmax([2, 1, 0]);
    const sum = probabilities.reduce((total, value) => total + value, 0);

    expect(sum).toBeCloseTo(1, 10);
  });

  it("keeps the biggest logit as the biggest probability for large numbers", () => {
    const probabilities = softmax([1000, 999, 998]);

    expect(probabilities[0]).toBeGreaterThan(probabilities[1] ?? -Infinity);
    expect(probabilities[1]).toBeGreaterThan(probabilities[2] ?? -Infinity);
  });
});

describe("relu", () => {
  it("limits at or above 0", () => {
    const output = relu([-1, 5, -33.2, 0, 12]);

    expect(output).toEqual([0, 5, 0, 0, 12]);
  });
});

describe("mean", () => {
  it("averages out a set of numbers", () => {
    const values = [-1, 5, -33.2, 0, 12];

    expect(mean(values)).toBeCloseTo(-3.44);
  });
});

describe("calculateStandardDeviation", () => {
  it("returns the average and population standard deviation", () => {
    const result = calculateStandardDeviation([2, 4, 4, 4, 5, 5, 7, 9]);

    expect(result.average).toBe(5);
    expect(result.standardDeviation).toBe(2);
  });

  it("returns zero when every value is identical", () => {
    const result = calculateStandardDeviation([3, 3, 3, 3]);

    expect(result.average).toBe(3);
    expect(result.standardDeviation).toBe(0);
  });
});
