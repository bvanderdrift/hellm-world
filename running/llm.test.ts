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
} from "../model/model-helpers.ts";
import * as weightReading from "../model/model-io.ts";
import type { Model } from "../model/types.ts";

const MODEL_NAME = "toy_model";
const { model: toyModel } = weightReading.getLatestCheckpointModel(MODEL_NAME);
const hiddenDimensionsSize = extractHiddenDimensionSize(toyModel);
const vocabSize = toyModel.vocabulary.length;

const getEmbedding = (weights: Model, token: string) => {
  const tokenIndex = findTokenIndex(weights.vocabulary, token);

  return weights.embeddings[tokenIndex]!;
};

const getStartState = (input: string) => {
  const inputTokens = tokenize(input, toyModel.vocabulary);

  return inputTokens.map((token) =>
    getEmbedding(toyModel, token).map(
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
    expect(pickToken([0, 5, 1, -3, 2, 4], toyModel.vocabulary)).toBe("world");
  });
});

describe("llmForwardPass", () => {
  it("returns one vocab-sized logit vector per input position", () => {
    const startState = getStartState("hello world beer");

    const logitsByPosition = llmForwardPass(startState, toyModel);

    validateSize(logitsByPosition, startState.length, vocabSize);
  });
});

describe("runLlm", () => {
  it("stops generation when the model predicts EOS and does not include it in the output", () => {
    const eosStoppingWeights: { historyLosses: number[]; model: Model } = {
      historyLosses: [3],
      model: {
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
      },
    };

    vi.spyOn(weightReading, "getLatestCheckpointModel").mockReturnValue(
      eosStoppingWeights,
    );

    expect(runLlm("hello", MODEL_NAME)).toBe("");
  });
});

describe("llm pipeline contracts", () => {
  it("embeds each input token into the hidden dimension", () => {
    const inputTokens = tokenize("hello world beer", toyModel.vocabulary);
    const embeddedState = inputTokens.map((token) =>
      getEmbedding(toyModel, token),
    );

    validateSize(embeddedState, inputTokens.length, hiddenDimensionsSize);
  });

  it("keeps one hidden-state row per context position after the hidden projection", () => {
    const inputTokens = tokenize("hello world beer", toyModel.vocabulary);
    const embeddedState = inputTokens.map((token) =>
      getEmbedding(toyModel, token),
    );
    const unembeddedState = multiplyMatrices(
      embeddedState,
      toyModel.unembeddings,
    );

    validateSize(unembeddedState, inputTokens.length, vocabSize);
  });

  it("projects the hidden state to one vocab-sized logit vector per position", () => {
    const inputTokens = tokenize("hello world beer", toyModel.vocabulary);
    const embeddedState = inputTokens.map((token) =>
      getEmbedding(toyModel, token),
    );
    const logitsByPosition = multiplyMatrices(
      embeddedState,
      toyModel.unembeddings,
    );

    validateSize(logitsByPosition, inputTokens.length, vocabSize);
  });
});

describe("weights validation contract", () => {
  it("fails fast when loaded weights contain an invalid embedding row, even if that token is unused", () => {
    const malformedWeights: { historyLosses: number[]; model: Model } = {
      historyLosses: [3],
      model: {
        vocabulary: [...toyModel.vocabulary],
        headsCount: toyModel.headsCount,
        embeddings: [
          [1, 1, 1, 1],
          [1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
        ],
        unembeddings: toyModel.unembeddings,
        transformers: toyModel.transformers,
      },
    };

    vi.spyOn(weightReading, "getLatestCheckpointModel").mockReturnValue(
      malformedWeights,
    );

    expect(() => runLlm("hello", MODEL_NAME)).toThrow(
      "Vector at index 1 has unexpected depth 3 (expected 4)",
    );
  });
});
