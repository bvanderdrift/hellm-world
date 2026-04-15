import { describe, expect, it } from "vitest";
import { getHighestValueIndex } from "./llm.ts";
import { multiplyMatrices, validateSize } from "./matrices.ts";
import {
  embeddings,
  HIDDEN_DIMENSIONS_SIZE,
  outMatrix,
  unembeddingsMatrix,
  VOCAB_SIZE,
} from "./weights.ts";
import { tokenize } from "./tokenizer.ts";

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
    const embeddedState = inputTokens.map((token) => embeddings[token]);

    validateSize(embeddedState, inputTokens.length, HIDDEN_DIMENSIONS_SIZE);
  });

  it("keeps one hidden-state row per context position after the hidden projection", () => {
    const inputTokens = tokenize("hello world beer");
    const embeddedState = inputTokens.map((token) => embeddings[token]);
    const unembeddedState = multiplyMatrices(embeddedState, unembeddingsMatrix);

    validateSize(unembeddedState, inputTokens.length, HIDDEN_DIMENSIONS_SIZE);
  });

  it("projects the hidden state to one vocab-sized logit vector per position", () => {
    const inputTokens = tokenize("hello world beer");
    const embeddedState = inputTokens.map((token) => embeddings[token]);
    const unembeddedState = multiplyMatrices(embeddedState, unembeddingsMatrix);
    const logitsByPosition = multiplyMatrices(unembeddedState, outMatrix);

    validateSize(logitsByPosition, inputTokens.length, VOCAB_SIZE);
  });
});
