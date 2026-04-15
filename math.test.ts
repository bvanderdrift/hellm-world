import { describe, expect, it } from "vitest";
import { dotProduct, softmax } from "./math.ts";

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
