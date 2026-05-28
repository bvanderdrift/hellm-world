import { describe, expect, it } from "vitest";
import { softmax } from "./softmax.ts";
import { TESTING_PRECISION } from "../testing/constants.ts";

describe("softmax", () => {
  it("returns probabilities that sum to 1", () => {
    const probabilities = softmax(new Float32Array([2, 1, 0]));
    let total = 0;
    for (let i = 0; i < probabilities.length; i++) {
      total += probabilities[i]!;
    }

    expect(total).toBeCloseTo(1, TESTING_PRECISION);
  });

  it("keeps the biggest logit as the biggest probability for large numbers", () => {
    const probabilities = softmax(new Float32Array([1000, 999, 998]));

    expect(probabilities[0]).toBeGreaterThan(probabilities[1] ?? -Infinity);
    expect(probabilities[1]).toBeGreaterThan(probabilities[2] ?? -Infinity);
  });
});
