import { afterEach, describe, expect, it, vi } from "vitest";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import {
  getHighestValueIndex,
  llmForwardPassByTokens,
  pickToken,
  runLlm,
} from "./llm.ts";
import {
  addMatrices,
  multiplyMatrices,
  normalize,
  validateSize,
} from "../shared/matrices.ts";
import { tokenize } from "../shared/tokenizer.ts";
import {
  extractHiddenDimensionSize,
  findTokenIndex,
} from "../model/model-helpers.ts";
import * as weightReading from "../model/model-io.ts";
import type { Model } from "../model/model-types.ts";
import { getPositionEncoding } from "./position-encoding.ts";

const MODEL_NAME = "toy_model";
const { model: toyModel } = weightReading.getLatestCheckpointModel(MODEL_NAME);
const hiddenDimensionsSize = extractHiddenDimensionSize(toyModel);
const vocabSize = toyModel.vocabulary.length;

const getEmbedding = (weights: Model, token: string) => {
  const tokenIndex = findTokenIndex(weights.vocabulary, token);

  return weights.embeddings[tokenIndex]!;
};

const getStartState = (inputTokens: string[], weights: Model) => {
  const hiddenSize = extractHiddenDimensionSize(weights);

  return inputTokens.map((token) =>
    getEmbedding(weights, token).map((value) => value * Math.sqrt(hiddenSize)),
  );
};

const expectMatrixCloseTo = (actual: number[][], expected: number[][]) => {
  expect(actual).toHaveLength(expected.length);

  for (const [rowIndex, expectedRow] of expected.entries()) {
    const actualRow = actual[rowIndex];

    expect(actualRow).toHaveLength(expectedRow.length);

    for (const [columnIndex, expectedValue] of expectedRow.entries()) {
      expect(actualRow?.[columnIndex]).toBeCloseTo(expectedValue, 10);
    }
  }
};

const attentionOnlyModel: Model = {
  vocabulary: ["hello", "world", "beer"],
  headsCount: 1,
  mlpMultiple: 1,
  embeddings: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
  unembeddings: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
  transformers: [
    {
      attention: {
        Q: [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ],
        K: [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ],
        V: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        out: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
          ],
          biasVector: [0, 0, 0],
        },
        wDown: {
          weightsMatrix: [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
          ],
          biasVector: [0, 0, 0],
        },
      },
    },
  ],
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

describe("llmForwardPassByTokens", () => {
  it("returns one vocab-sized logit vector per input position", () => {
    const inputTokens = tokenize("hello world beer", toyModel.vocabulary);

    const { embeddings: logitsByPosition } = llmForwardPassByTokens(
      inputTokens,
      toyModel,
      false,
    );

    validateSize(logitsByPosition, inputTokens.length, vocabSize);
  });

  it("returns the activation operands needed to backprop through one transformer", () => {
    const inputTokens = ["hello", "world"];
    const expectedStartState = getStartState(inputTokens, attentionOnlyModel);

    const { activations } = llmForwardPassByTokens(
      inputTokens,
      attentionOnlyModel,
      true,
    );

    expect(activations).not.toBeNull();

    const transformerActivations = activations!.transformerActivations[0]!;
    const attentionActivations = transformerActivations.attention;
    const headActivations = attentionActivations.heads[0]!;

    expect(activations!.tokensToPosition).toEqual(expectedStartState);
    expect(activations!.positionToTransformers).toEqual(
      getPositionEncoding(2, 3),
    );

    expectMatrixCloseTo(
      transformerActivations.transformerInput,
      addMatrices(expectedStartState, activations!.positionToTransformers),
    );

    validateSize(attentionActivations.normalizedInput, 2, 3);
    validateSize(attentionActivations.output, 2, 3);
    expect(attentionActivations.heads).toHaveLength(1);

    validateSize(headActivations.inputQ, 2, 3);
    validateSize(headActivations.inputK, 2, 3);
    validateSize(headActivations.inputV, 2, 3);
    expect(headActivations.attentionRelevancyOutput).toHaveLength(2);
    expect(headActivations.attentionRelevancyOutput[0]).toHaveLength(1);
    expect(headActivations.attentionRelevancyOutput[1]).toHaveLength(2);
    expect(headActivations.softmaxOutput).toHaveLength(2);
    expect(headActivations.softmaxOutput[0]).toHaveLength(1);
    expect(headActivations.softmaxOutput[1]).toHaveLength(2);
    validateSize(headActivations.output, 2, 3);
    expect(headActivations.lookbackUpdateVectors).toHaveLength(2);
    validateSize(headActivations.lookbackUpdateVectors[0]!, 1, 3);
    validateSize(headActivations.lookbackUpdateVectors[1]!, 2, 3);

    validateSize(transformerActivations.mlp.normalizedInputToUpping, 2, 3);
    validateSize(transformerActivations.mlp.uppingToNonLinear, 2, 3);
    validateSize(transformerActivations.mlp.nonLinearToDowning, 2, 3);
    validateSize(transformerActivations.mlp.downingOutput, 2, 3);

    validateSize(activations!.normalizerToUnembeddings, 2, 3);
    validateSize(activations!.unembeddingsOutputLogits, 2, 3);
  });

  it("unembeds the transformer state after both attention and MLP residual updates", () => {
    const inputTokens = ["hello", "world"];

    const { embeddings: logitsByPosition, activations } = llmForwardPassByTokens(
      inputTokens,
      attentionOnlyModel,
      true,
    );

    const transformerActivations = activations!.transformerActivations[0]!;
    const expectedFinalTransformerState = addMatrices(
      addMatrices(
        transformerActivations.transformerInput,
        transformerActivations.attention.output,
      ),
      transformerActivations.mlp.downingOutput,
    );
    const expectedLogits = multiplyMatrices(
      normalize(expectedFinalTransformerState),
      attentionOnlyModel.unembeddings,
    );

    expectMatrixCloseTo(logitsByPosition, expectedLogits);
    expectMatrixCloseTo(activations!.unembeddingsOutputLogits, expectedLogits);
  });
});

describe("runLlm", () => {
  it("stops generation when the model predicts EOS and does not include it in the output", () => {
    const eosStoppingWeights: { historyLosses: number[]; model: Model } = {
      historyLosses: [3],
      model: {
        vocabulary: ["hello", END_OF_SEQUENCE_TOKEN],
        headsCount: 1,
        mlpMultiple: 1,
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

    expect(Array.from(runLlm("hello", MODEL_NAME))).toEqual([]);
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
        mlpMultiple: toyModel.mlpMultiple,
        embeddings: [
          [1, 1, 1, 1],
          [1, 1, 1],
          [1, 1, 1, 1],
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

    expect(() => Array.from(runLlm("hello", MODEL_NAME))).toThrow(
      "Vector at index 1 has unexpected depth 3 (expected 4)",
    );
  });
});
