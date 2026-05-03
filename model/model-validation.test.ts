import { describe, expect, it } from "vitest";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { validateModel, validateSameModelShape } from "./model-validation.ts";
import type { Model } from "./model-types.ts";

const vector = (length: number, value = 1) => new Array(length).fill(value);

const matrix = (rows: number, columns: number, value = 1) =>
  new Array(rows).fill(null).map(() => vector(columns, value));

const HIDDEN_DIMENSION_SIZE = 4;
const DEFAULT_MLP_MULTIPLE = 4;
const DEFAULT_MLP_DIMENSION_SIZE = HIDDEN_DIMENSION_SIZE * DEFAULT_MLP_MULTIPLE;
const SMALLER_MLP_MULTIPLE = 2;
const SMALLER_MLP_DIMENSION_SIZE = HIDDEN_DIMENSION_SIZE * SMALLER_MLP_MULTIPLE;

const validModel: Model = {
  vocabulary: ["hello", "world", "beer", END_OF_SEQUENCE_TOKEN],
  headsCount: 2,
  mlpMultiple: DEFAULT_MLP_MULTIPLE,
  embeddings: matrix(4, HIDDEN_DIMENSION_SIZE),
  unembeddings: matrix(HIDDEN_DIMENSION_SIZE, 4),
  transformers: [
    {
      attention: {
        Q: matrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE),
        K: matrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE),
        V: matrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE),
        out: matrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE),
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: matrix(
            HIDDEN_DIMENSION_SIZE,
            DEFAULT_MLP_DIMENSION_SIZE,
          ),
          biasVector: vector(DEFAULT_MLP_DIMENSION_SIZE),
        },
        wDown: {
          weightsMatrix: matrix(
            DEFAULT_MLP_DIMENSION_SIZE,
            HIDDEN_DIMENSION_SIZE,
          ),
          biasVector: vector(HIDDEN_DIMENSION_SIZE),
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
      embeddings: matrix(3, 4),
      unembeddings: matrix(4, 3),
    });

    expect(() => validateModel(malformedWeights)).toThrow(
      `Model embeddings are missing special end-of-sequence token "${END_OF_SEQUENCE_TOKEN}"`,
    );
  });

  it("rejects embeddings that do not cover the full vocabulary", () => {
    const malformedWeights = createModel({
      embeddings: matrix(3, 4),
    });

    expect(() => validateModel(malformedWeights)).toThrow(
      "matrix vector count (3) doesn't match expected vector count 4",
    );
  });

  it("rejects embeddings with too many rows for the vocabulary", () => {
    const malformedWeights = createModel({
      embeddings: matrix(5, 4),
    });

    expect(() => validateModel(malformedWeights)).toThrow(
      "matrix vector count (5) doesn't match expected vector count 4",
    );
  });

  it("throws when any embedding row has the wrong width", () => {
    const malformedWeights = createModel({
      embeddings: [
        [1, 1, 1, 1],
        [1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
      ],
    });

    expect(() => validateModel(malformedWeights)).toThrow(
      "Vector at index 1 has unexpected depth 3 (expected 4)",
    );
  });

  it("rejects checkpoints with empty vocabulary using a helpful error", () => {
    const malformedWeights = createModel({
      vocabulary: [],
      embeddings: [],
      unembeddings: [],
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

  it("uses mlpMultiple from metadata when validating MLP shapes", () => {
    const modelWithSmallerMlp = createModel({
      mlpMultiple: SMALLER_MLP_MULTIPLE,
      transformers: [
        {
          ...structuredClone(validModel.transformers[0]!),
          multilayerPerceptron: {
            wUp: {
              weightsMatrix: matrix(
                HIDDEN_DIMENSION_SIZE,
                SMALLER_MLP_DIMENSION_SIZE,
              ),
              biasVector: vector(SMALLER_MLP_DIMENSION_SIZE),
            },
            wDown: {
              weightsMatrix: matrix(
                SMALLER_MLP_DIMENSION_SIZE,
                HIDDEN_DIMENSION_SIZE,
              ),
              biasVector: vector(HIDDEN_DIMENSION_SIZE),
            },
          },
        },
      ],
    });

    expect(() => validateModel(modelWithSmallerMlp)).not.toThrow();
  });

  it("rejects MLP shapes that do not match mlpMultiple metadata", () => {
    const malformedWeights = createModel({
      mlpMultiple: SMALLER_MLP_MULTIPLE,
    });

    expect(() => validateModel(malformedWeights)).toThrow(
      `m has unexpected vector depth ${DEFAULT_MLP_DIMENSION_SIZE}, expected ${SMALLER_MLP_DIMENSION_SIZE}`,
    );
  });
});

describe("validateSameWeightShape", () => {
  it("accepts weights with the same shape", () => {
    expect(() =>
      validateSameModelShape(createModel(), createModel()),
    ).not.toThrow();
  });

  it("rejects embeddings with different vocabulary sizes", () => {
    const weightsWithMoreEmbeddings = createModel({
      embeddings: matrix(5, 4),
    });

    expect(() =>
      validateSameModelShape(weightsWithMoreEmbeddings, createModel()),
    ).toThrow("matrix vector count (5) doesn't match expected vector count 4");
  });

  it("rejects embeddings with different hidden widths", () => {
    const weightsWithWiderEmbeddings = createModel({
      embeddings: matrix(4, 5),
    });

    expect(() =>
      validateSameModelShape(weightsWithWiderEmbeddings, createModel()),
    ).toThrow("m has unexpected vector depth 5, expected 4");
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
