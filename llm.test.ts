import { describe, expect, it } from "vitest";
import { getHighestValueIndex } from "./llm.ts";

describe("getHighestValueIndex", () => {
  it("should get highest value", () => {
    const foundIndex = getHighestValueIndex([3, -5, -22.4, 33.2, 9]);

    expect(foundIndex).toBe(3);
  });
});
