import { describe, expect, it } from "vitest";
import {
  decodeLogits,
  getHighestValueIndex,
  llmForwardPass,
  runLlm,
} from "./llm.ts";
import { multiplyMatrices, validateSize } from "./matrices.ts";
import { tokenize } from "./tokenizer.ts";
import { toyWeights, type Token } from "./weights/toy_weights/toyWeights.ts";
import type { Weights } from "./weights/types.ts";

const getStartState = (input: string) => {
  const inputTokens = tokenize(input, toyWeights.tokens);

  return inputTokens.map((token) =>
    toyWeights.embeddings[token].map(
      (value) => value * Math.sqrt(toyWeights.hiddenDimensionsSize),
    ),
  );
};

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

describe("decodeLogits", () => {
  it("returns the token behind the highest logit", () => {
    expect(decodeLogits([0, 5, 1, -3, 2, 4], toyWeights.tokens)).toBe("world");
  });
});

describe("llmForwardPass", () => {
  it("returns one vocab-sized logit vector per input position", () => {
    const startState = getStartState("hello world beer");

    const logitsByPosition = llmForwardPass(startState, toyWeights);

    validateSize(logitsByPosition, startState.length, toyWeights.vocabSize);
  });
});

describe("runLlm", () => {
  it("decodes the last-position logits from the forward pass", () => {
    const input = "hello world beer";
    const logitsByPosition = llmForwardPass(getStartState(input), toyWeights);
    const lastLogits = logitsByPosition[logitsByPosition.length - 1];

    if (!lastLogits) {
      throw new Error(`Expected the forward pass to return at least one row`);
    }

    expect(runLlm(input, toyWeights)).toBe(
      decodeLogits(lastLogits, toyWeights.tokens),
    );
  });
});

describe("llm pipeline contracts", () => {
  it("embeds each input token into the hidden dimension", () => {
    const inputTokens = tokenize("hello world beer", toyWeights.tokens);
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
    const inputTokens = tokenize("hello world beer", toyWeights.tokens);
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
    const inputTokens = tokenize("hello world beer", toyWeights.tokens);
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
