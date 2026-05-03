import { describe, expect, it } from "vitest";
import { calculateLoss } from "./calculateLoss.ts";

describe("calculateLoss", () => {
  it("returns log vocabulary size when logits produce a uniform distribution", () => {
    const loss = calculateLoss([0, 0, 0], 1, [
      "hello",
      "world",
      "beer",
    ]);

    expect(loss).toBeCloseTo(Math.log(3), 10);
  });

  it("uses the correct token index, not the highest-logit token", () => {
    const loss = calculateLoss([0, 2, 0], 0, [
      "hello",
      "world",
      "beer",
    ]);

    expect(loss).toBeCloseTo(Math.log(Math.exp(0) + Math.exp(2) + Math.exp(0)));
  });

  it("returns a lower loss when the correct token has a higher logit", () => {
    const vocabulary = ["hello", "world"];

    const confidentLoss = calculateLoss([8, 0], 0, vocabulary);
    const uncertainLoss = calculateLoss([0, 0], 0, vocabulary);

    expect(confidentLoss).toBeLessThan(uncertainLoss);
  });

  it("stays finite when the correct token probability is extremely small", () => {
    const loss = calculateLoss([0, -1000], 1, ["hello", "world"]);

    expect(loss).toBeCloseTo(1000, 10);
  });

  it("stays finite when logits are too large to exponentiate directly", () => {
    const loss = calculateLoss([1000, 999], 0, ["hello", "world"]);

    expect(loss).toBeCloseTo(Math.log(1 + Math.exp(-1)), 10);
  });

  it("throws when the logits length does not match the vocabulary length", () => {
    expect(() => calculateLoss([1], 0, ["hello", "world"])).toThrow(
      "unexpected vector depth",
    );
  });
});
