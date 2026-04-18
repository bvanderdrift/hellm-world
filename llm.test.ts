import { describe, expect, it } from "vitest";
import { getHighestValueIndex } from "./llm.ts";
import { multiplyMatrices, validateSize } from "./matrices.ts";
import { tokenize } from "./tokenizer.ts";
import { toyWeights } from "./weights/toy_weights/toyWeights.ts";

describe("getHighestValueIndex", () => {
  it("should get highest value", () => {
    const foundIndex = getHighestValueIndex([3, -5, -22.4, 33.2, 9]);

    expect(foundIndex).toBe(3);
  });

  it("keeps the first index when multiple values are tied for highest", () => {
    const foundIndex = getHighestValueIndex([7, 7, 2]);

    expect(foundIndex).toBe(0);
  });
});

describe("llm pipeline contracts", () => {
  it("embeds each input token into the hidden dimension", () => {
    const inputTokens = tokenize("hello world beer");
    const embeddedState = inputTokens.map(
      (token) => toyWeights.embeddings[token],
    );

    validateSize(
      embeddedState,
      inputTokens.length,
      toyWeights.hiddenDimensionsSize,
    );
  });

  it("keeps one hidden-state row per context position after the hidden projection", () => {
    const inputTokens = tokenize("hello world beer");
    const embeddedState = inputTokens.map(
      (token) => toyWeights.embeddings[token],
    );
    const unembeddedState = multiplyMatrices(
      embeddedState,
      toyWeights.unembeddings,
    );

    validateSize(unembeddedState, inputTokens.length, toyWeights.vocabSize);
  });

  it("projects the hidden state to one vocab-sized logit vector per position", () => {
    const inputTokens = tokenize("hello world beer");
    const embeddedState = inputTokens.map(
      (token) => toyWeights.embeddings[token],
    );
    const logitsByPosition = multiplyMatrices(
      embeddedState,
      toyWeights.unembeddings,
    );

    validateSize(logitsByPosition, inputTokens.length, toyWeights.vocabSize);
  });
});
