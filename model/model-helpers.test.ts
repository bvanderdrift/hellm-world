import { describe, expect, it } from "vitest";
import { findTokenIndex, operateCombinedWeights } from "./model-helpers.ts";
import { createConstantModel } from "../testing/model-fixtures.ts";

const validModel = createConstantModel();

describe("findTokenIndex", () => {
  it("returns the vocabulary index for a known token", () => {
    expect(findTokenIndex(validModel.vocabulary, "beer")).toBe(2);
  });

  it("throws when the token is not in the vocabulary", () => {
    expect(() => findTokenIndex(validModel.vocabulary, "ghost")).toThrow(
      "Failed to find token ghost in vocabulary",
    );
  });
});

describe("operateWeights", () => {
  it("applies the operation across embeddings, unembeddings, attention, and MLP weights", () => {
    const operatedWeights = operateCombinedWeights(
      createConstantModel(2),
      createConstantModel(3),
      (value1, value2) => value1 + value2,
    );

    expect(operatedWeights).toEqual(createConstantModel(5));
  });
});
