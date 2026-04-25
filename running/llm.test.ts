import { afterEach, describe, expect, it, vi } from "vitest";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import {
  getHighestValueIndex,
  llmForwardPass,
  pickToken,
  runLlm,
} from "./llm.ts";
import { multiplyMatrices, validateSize } from "../shared/matrices.ts";
import { tokenize } from "../shared/tokenizer.ts";
import {
  extractHiddenDimensionSize,
  findTokenIndex,
} from "../weights/weight-helpers.ts";
import * as weightReading from "../weights/weight-io.ts";
import type { Model } from "../weights/types.ts";

const MODEL_NAME = "toy_model";
const toyWeights = weightReading.getLatestCheckpointWeights(MODEL_NAME);
const hiddenDimensionsSize = extractHiddenDimensionSize(toyWeights);
const vocabSize = toyWeights.vocabulary.length;

const getEmbedding = (weights: Model, token: string) => {
  const tokenIndex = findTokenIndex(weights.vocabulary, token);

  return weights.embeddings[tokenIndex]!;
};

const getStartState = (input: string) => {
  const inputTokens = tokenize(input, toyWeights.vocabulary);

  return inputTokens.map((token) =>
    getEmbedding(toyWeights, token).map(
      (value) => value * Math.sqrt(hiddenDimensionsSize),
    ),
  );
};

afterEach(() => {
  vi.restoreAllMocks();
});

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

describe("pickToken", () => {
  it("returns the token behind the highest logit", () => {
    expect(pickToken([0, 5, 1, -3, 2, 4], toyWeights.vocabulary)).toBe("world");
  });
});

describe("llmForwardPass", () => {
  it("returns one vocab-sized logit vector per input position", () => {
    const startState = getStartState("hello world beer");

    const logitsByPosition = llmForwardPass(startState, toyWeights);

    validateSize(logitsByPosition, startState.length, vocabSize);
  });
});

describe("runLlm", () => {
  it("stops generation when the model predicts EOS and does not include it in the output", () => {
    const eosStoppingWeights: Model = {
      vocabulary: ["hello", END_OF_SEQUENCE_TOKEN],
      headsCount: 1,
      embeddings: [
        [0, 0],
        [0, 0],
      ],
      unembeddings: [
        [0, -1],
        [0, 1],
      ],
      transformers: [],
    };

    vi.spyOn(weightReading, "getLatestCheckpointWeights").mockReturnValue(
      eosStoppingWeights,
    );

    expect(runLlm("hello", MODEL_NAME)).toBe("");
  });
});

describe("llm pipeline contracts", () => {
  it("embeds each input token into the hidden dimension", () => {
    const inputTokens = tokenize("hello world beer", toyWeights.vocabulary);
    const embeddedState = inputTokens.map((token) =>
      getEmbedding(toyWeights, token),
    );

    validateSize(embeddedState, inputTokens.length, hiddenDimensionsSize);
  });

  it("keeps one hidden-state row per context position after the hidden projection", () => {
    const inputTokens = tokenize("hello world beer", toyWeights.vocabulary);
    const embeddedState = inputTokens.map((token) =>
      getEmbedding(toyWeights, token),
    );
    const unembeddedState = multiplyMatrices(
      embeddedState,
      toyWeights.unembeddings,
    );

    validateSize(unembeddedState, inputTokens.length, vocabSize);
  });

  it("projects the hidden state to one vocab-sized logit vector per position", () => {
    const inputTokens = tokenize("hello world beer", toyWeights.vocabulary);
    const embeddedState = inputTokens.map((token) =>
      getEmbedding(toyWeights, token),
    );
    const logitsByPosition = multiplyMatrices(
      embeddedState,
      toyWeights.unembeddings,
    );

    validateSize(logitsByPosition, inputTokens.length, vocabSize);
  });
});

describe("weights validation contract", () => {
  it("fails fast when loaded weights contain an invalid embedding row, even if that token is unused", () => {
    const malformedWeights: Model = {
      vocabulary: [...toyWeights.vocabulary],
      headsCount: toyWeights.headsCount,
      embeddings: [
        [1, 1, 1, 1],
        [1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
      ],
      unembeddings: toyWeights.unembeddings,
      transformers: toyWeights.transformers,
    };

    vi.spyOn(weightReading, "getLatestCheckpointWeights").mockReturnValue(
      malformedWeights,
    );

    expect(() => runLlm("hello", MODEL_NAME)).toThrow(
      "Vector at index 1 has unexpected depth 3 (expected 4)",
    );
  });
});
