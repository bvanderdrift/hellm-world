import { describe, expect, it } from "vitest";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { createMatrix, type Matrix } from "../shared/matrices.ts";
import { validateModel, validateSameModelShape } from "./model-validation.ts";
import type { Model } from "./model-types.ts";

const m = (rows: number, columns: number, value = 1): Matrix => {
  const mat = createMatrix(rows, columns, () => value);
  return mat;
};

const HIDDEN_DIMENSION_SIZE = 4;
const DEFAULT_MLP_MULTIPLE = 4;
const DEFAULT_MLP_DIMENSION_SIZE = HIDDEN_DIMENSION_SIZE * DEFAULT_MLP_MULTIPLE;
const SMALLER_MLP_MULTIPLE = 2;
const SMALLER_MLP_DIMENSION_SIZE = HIDDEN_DIMENSION_SIZE * SMALLER_MLP_MULTIPLE;

const validModel: Model = {
  vocabulary: ["hello", "world", "beer", END_OF_SEQUENCE_TOKEN],
  headsCount: 2,
  mlpMultiple: DEFAULT_MLP_MULTIPLE,
  embeddings: m(4, HIDDEN_DIMENSION_SIZE),
  unembeddings: m(HIDDEN_DIMENSION_SIZE, 4),
  transformers: [
    {
      attention: {
        Q: m(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE),
        K: m(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE),
        V: m(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE),
        out: m(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE),
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: m(
            HIDDEN_DIMENSION_SIZE,
            DEFAULT_MLP_DIMENSION_SIZE,
          ),
          biasVector: m(1, DEFAULT_MLP_DIMENSION_SIZE),
        },
        wDown: {
          weightsMatrix: m(
            DEFAULT_MLP_DIMENSION_SIZE,
            HIDDEN_DIMENSION_SIZE,
          ),
          biasVector: m(1, HIDDEN_DIMENSION_SIZE),
        },
      },
    },
  ],
};

const createModel = (overrides: Partial<Model> = {}): Model => ({
  ...structuredClone(validModel),
  ...overrides,
});

describe("validateModel", () => {
  it("accepts a self-consistent checkpoint", () => {
    expect(() => validateModel(validModel)).not.toThrow();
  });

  it("rejects a headsCount that does not evenly divide the hidden width", () => {
    const malformedWeights = createModel({
      headsCount: 3,
    });

    expect(() => validateModel(malformedWeights)).toThrow(
      "Can't perfectly divide the nominator 4 by denominator (3)",
    );
  });

  it("rejects duplicate tokens in the checkpoint vocabulary", () => {
    const duplicateTokenWeights = createModel({
      vocabulary: ["hello", "world", "hello", END_OF_SEQUENCE_TOKEN],
    });

    expect(() => validateModel(duplicateTokenWeights)).toThrow(
      "Provided weights have 1 duplicate tokens",
    );
  });

  it("rejects vocabularies that are missing the EOS token", () => {
    const malformedWeights = createModel({
      vocabulary: ["hello", "world", "beer"],
      embeddings: m(3, 4),
      unembeddings: m(4, 3),
    });

    expect(() => validateModel(malformedWeights)).toThrow(
      `Model embeddings are missing special end-of-sequence token "${END_OF_SEQUENCE_TOKEN}"`,
    );
  });

  it("rejects checkpoints with empty vocabulary using a helpful error", () => {
    const malformedWeights = createModel({
      vocabulary: [],
      embeddings: m(0, 0),
      unembeddings: m(0, 0),
    });

    expect(() => validateModel(malformedWeights)).toThrow(
      "Provided vocabulary cannot be empty",
    );
  });

  it("rejects a negative headsCount", () => {
    const malformedWeights = createModel({
      headsCount: -2,
    });

    expect(() => validateModel(malformedWeights)).toThrow(
      "headsCount must be a positive integer",
    );
  });

  it("detects a corrupted matrix where values length mismatches dimensions", () => {
    const model = createModel();
    model.embeddings.values = new Float32Array(5);

    expect(() => validateModel(model)).toThrow(
      "m has unexpected parameter count 5, expected 16",
    );
  });

  it("uses mlpMultiple from metadata when validating MLP shapes", () => {
    const modelWithSmallerMlp = createModel({
      mlpMultiple: SMALLER_MLP_MULTIPLE,
      transformers: [
        {
          ...structuredClone(validModel.transformers[0]!),
          multilayerPerceptron: {
            wUp: {
              weightsMatrix: m(
                HIDDEN_DIMENSION_SIZE,
                SMALLER_MLP_DIMENSION_SIZE,
              ),
              biasVector: m(1, SMALLER_MLP_DIMENSION_SIZE),
            },
            wDown: {
              weightsMatrix: m(
                SMALLER_MLP_DIMENSION_SIZE,
                HIDDEN_DIMENSION_SIZE,
              ),
              biasVector: m(1, HIDDEN_DIMENSION_SIZE),
            },
          },
        },
      ],
    });

    expect(() => validateModel(modelWithSmallerMlp)).not.toThrow();
  });
});

describe("validateSameWeightShape", () => {
  it("accepts weights with the same shape", () => {
    expect(() =>
      validateSameModelShape(createModel(), createModel()),
    ).not.toThrow();
  });

  it("rejects different transformer counts", () => {
    const weightsWithExtraTransformer = createModel({
      transformers: [
        ...structuredClone(validModel.transformers),
        structuredClone(validModel.transformers[0]!),
      ],
    });

    expect(() =>
      validateSameModelShape(weightsWithExtraTransformer, createModel()),
    ).toThrow("Weights1 has different transformers count 2 than Weights2 (1)");
  });

  it("rejects models with the same weight shape but a different vocabulary order", () => {
    const modelWithReorderedVocabulary = createModel({
      vocabulary: ["world", "hello", "beer", END_OF_SEQUENCE_TOKEN],
    });

    expect(() =>
      validateSameModelShape(modelWithReorderedVocabulary, createModel()),
    ).toThrow("Vocabularies between weights don't match");
  });

  it("rejects different head counts", () => {
    const weightsWithDifferentHeadCount = createModel({
      headsCount: 4,
    });

    expect(() =>
      validateSameModelShape(weightsWithDifferentHeadCount, createModel()),
    ).toThrow("Model 1 has different head count 4 than Model 2 (2)");
  });
});
