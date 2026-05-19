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
} from "../shared/matrices.ts";
import { tokenize } from "../shared/tokenizer.ts";
import { extractHiddenDimensionSize } from "../model/model-helpers.ts";
import * as weightReading from "../model/model-io.ts";
import type { Model, ModelCheckpoint } from "../model/model-types.ts";
import { getPositionEncoding } from "./position-encoding.ts";
import { matrixFrom, expectMatrixCloseTo } from "../testing/testing-utils.ts";

const MODEL_NAME = "timmy";

const testModel: Model = {
  vocabulary: ["hello", "world", " ", "beer", "!", END_OF_SEQUENCE_TOKEN],
  headsCount: 1,
  mlpMultiple: 1,
  embeddings: matrixFrom([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5, 1.6],
    [1.7, 1.8, 1.9, 2.0],
    [2.1, 2.2, 2.3, 2.4],
  ]),
  unembeddings: matrixFrom([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    [1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
  ]),
  transformers: [],
};
const hiddenDimensionsSize = extractHiddenDimensionSize(testModel);
const vocabSize = testModel.vocabulary.length;

const attentionOnlyModel: Model = {
  vocabulary: ["hello", "world", "beer"],
  headsCount: 1,
  mlpMultiple: 1,
  embeddings: matrixFrom([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]),
  unembeddings: matrixFrom([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]),
  transformers: [
    {
      attention: {
        Q: matrixFrom([
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ]),
        K: matrixFrom([
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ]),
        V: matrixFrom([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ]),
        out: matrixFrom([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ]),
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: matrixFrom([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
          ]),
          biasVector: matrixFrom([[0, 0, 0]]),
        },
        wDown: {
          weightsMatrix: matrixFrom([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
          ]),
          biasVector: matrixFrom([[0, 0, 0]]),
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
    expect(
      pickToken(new Float32Array([0, 5, 1, -3, 2, 4]), testModel.vocabulary),
    ).toBe("world");
  });
});

describe("llmForwardPassByTokens", () => {
  it("returns one vocab-sized logit vector per input position", () => {
    const inputTokens = tokenize("hello world beer", testModel.vocabulary);

    const { embeddings: logitsByPosition } = llmForwardPassByTokens(
      inputTokens,
      testModel,
      false,
    );

    expect(logitsByPosition.vectors).toBe(inputTokens.length);
    expect(logitsByPosition.dimensions).toBe(vocabSize);
  });

  it("returns the activation operands needed to backprop through one transformer", () => {
    const inputTokens = ["hello", "world"];

    const { activations } = llmForwardPassByTokens(
      inputTokens,
      attentionOnlyModel,
      true,
    );

    expect(activations).not.toBeNull();

    const transformerActivations = activations!.transformerActivations[0]!;
    const attentionActivations = transformerActivations.attention;
    const headActivations = attentionActivations.heads[0]!;

    expect(activations!.positionToTransformers).toEqual(
      getPositionEncoding(2, 3),
    );

    expectMatrixCloseTo(
      transformerActivations.transformerInput,
      addMatrices(
        activations!.tokensToPosition,
        activations!.positionToTransformers,
      ),
    );

    expect(attentionActivations.normalizedInput.vectors).toBe(2);
    expect(attentionActivations.normalizedInput.dimensions).toBe(3);
    expect(attentionActivations.output.vectors).toBe(2);
    expect(attentionActivations.output.dimensions).toBe(3);
    expect(attentionActivations.heads).toHaveLength(1);

    expect(headActivations.inputQ.vectors).toBe(2);
    expect(headActivations.inputQ.dimensions).toBe(3);
    expect(headActivations.inputK.vectors).toBe(2);
    expect(headActivations.inputK.dimensions).toBe(3);
    expect(headActivations.inputV.vectors).toBe(2);
    expect(headActivations.inputV.dimensions).toBe(3);
    expect(headActivations.output.vectors).toBe(2);
    expect(headActivations.output.dimensions).toBe(3);

    expect(transformerActivations.mlp.normalizedInputToUpping.vectors).toBe(2);
    expect(transformerActivations.mlp.normalizedInputToUpping.dimensions).toBe(
      3,
    );
    expect(transformerActivations.mlp.uppingToNonLinear.vectors).toBe(2);
    expect(transformerActivations.mlp.uppingToNonLinear.dimensions).toBe(3);
    expect(transformerActivations.mlp.nonLinearToDowning.vectors).toBe(2);
    expect(transformerActivations.mlp.nonLinearToDowning.dimensions).toBe(3);
    expect(transformerActivations.mlp.downingOutput.vectors).toBe(2);
    expect(transformerActivations.mlp.downingOutput.dimensions).toBe(3);

    expect(activations!.normalizerToUnembeddings.vectors).toBe(2);
    expect(activations!.normalizerToUnembeddings.dimensions).toBe(3);
    expect(activations!.unembeddingsOutputLogits.vectors).toBe(2);
    expect(activations!.unembeddingsOutputLogits.dimensions).toBe(3);
  });

  it("unembeds the transformer state after both attention and MLP residual updates", () => {
    const inputTokens = ["hello", "world"];

    const { embeddings: logitsByPosition, activations } =
      llmForwardPassByTokens(inputTokens, attentionOnlyModel, true);

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
    const eosStoppingWeights: Omit<ModelCheckpoint, "weights"> & {
      model: Model;
    } = {
      history: { trainingLosses: [3], validationLosses: [] },
      model: {
        vocabulary: ["hello", END_OF_SEQUENCE_TOKEN],
        headsCount: 1,
        mlpMultiple: 1,
        embeddings: matrixFrom([
          [0, 0],
          [0, 0],
        ]),
        unembeddings: matrixFrom([
          [0, -1],
          [0, 1],
        ]),
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
    expect(testModel.embeddings.vectors).toBe(testModel.vocabulary.length);
    expect(testModel.embeddings.dimensions).toBe(hiddenDimensionsSize);
  });

  it("keeps one hidden-state row per context position after the hidden projection", () => {
    const unembeddedState = multiplyMatrices(
      testModel.embeddings,
      testModel.unembeddings,
    );

    expect(unembeddedState.vectors).toBe(testModel.vocabulary.length);
    expect(unembeddedState.dimensions).toBe(vocabSize);
  });

  it("projects the hidden state to one vocab-sized logit vector per position", () => {
    const logitsByPosition = multiplyMatrices(
      testModel.embeddings,
      testModel.unembeddings,
    );

    expect(logitsByPosition.vectors).toBe(testModel.vocabulary.length);
    expect(logitsByPosition.dimensions).toBe(vocabSize);
  });
});

describe("weights validation contract", () => {
  it("fails fast when loaded weights have wrong total parameter count", () => {
    const malformedWeights: Omit<ModelCheckpoint, "weights"> & {
      model: Model;
    } = {
      history: { trainingLosses: [3], validationLosses: [] },
      model: {
        vocabulary: [...testModel.vocabulary],
        headsCount: testModel.headsCount,
        mlpMultiple: testModel.mlpMultiple,
        embeddings: {
          vectors: 6,
          dimensions: 4,
          values: new Float32Array(23),
        },
        unembeddings: testModel.unembeddings,
        transformers: testModel.transformers,
      },
    };

    vi.spyOn(weightReading, "getLatestCheckpointModel").mockReturnValue(
      malformedWeights,
    );

    expect(() => Array.from(runLlm("hello", MODEL_NAME))).toThrow();
  });
});
