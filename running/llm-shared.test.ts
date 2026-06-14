import { describe, expect, it } from "vitest";
import { getHighestValueIndex, pickToken } from "./llm-shared.ts";

const vocabulary = ["hello", "world", " ", "beer", "!"];

describe("getHighestValueIndex", () => {
  it("should get highest value", () => {
    const foundIndex = getHighestValueIndex(
      new Float32Array([3, -5, -22.4, 33.2, 9]),
    );

    expect(foundIndex).toBe(3);
  });

  it("keeps the first index when multiple values are tied for highest", () => {
    const foundIndex = getHighestValueIndex(new Float32Array([7, 7, 2]));

    expect(foundIndex).toBe(0);
  });
});

describe("pickToken", () => {
  it("returns the token behind the highest logit", () => {
    expect(pickToken(new Float32Array([0, 5, 1, -3, 2]), vocabulary)).toBe(
      "world",
    );
  });
});
