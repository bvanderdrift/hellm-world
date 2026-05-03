import { describe, expect, it } from "vitest";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import {
  extractHiddenDimensionSize,
  findTokenIndex,
  operateCombinedWeights,
} from "./model-helpers.ts";
import type { Model } from "./model-types.ts";

const vector = (length: number, value = 1) => new Array(length).fill(value);

const matrix = (rows: number, columns: number, value = 1) =>
  new Array(rows).fill(null).map(() => vector(columns, value));

const HIDDEN_DIMENSION_SIZE = 4;
const DEFAULT_MLP_MULTIPLE = 4;
const DEFAULT_MLP_DIMENSION_SIZE = HIDDEN_DIMENSION_SIZE * DEFAULT_MLP_MULTIPLE;
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

const createModelWithValue = (value: number): Model => ({
  ...validModel,
  embeddings: matrix(4, HIDDEN_DIMENSION_SIZE, value),
  unembeddings: matrix(HIDDEN_DIMENSION_SIZE, 4, value),
  transformers: [
    {
      attention: {
        Q: matrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
        K: matrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
        V: matrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
        out: matrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: matrix(
            HIDDEN_DIMENSION_SIZE,
            DEFAULT_MLP_DIMENSION_SIZE,
            value,
          ),
          biasVector: vector(DEFAULT_MLP_DIMENSION_SIZE, value),
        },
        wDown: {
          weightsMatrix: matrix(
            DEFAULT_MLP_DIMENSION_SIZE,
            HIDDEN_DIMENSION_SIZE,
            value,
          ),
          biasVector: vector(HIDDEN_DIMENSION_SIZE, value),
        },
      },
    },
  ],
});

describe("extractHiddenDimensionSize", () => {
  it("derives the hidden width and vocab size from embeddings", () => {
    expect(extractHiddenDimensionSize(validModel)).toEqual(4);
  });
});

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
      createModelWithValue(2),
      createModelWithValue(3),
      (value1, value2) => value1 + value2,
    );

    expect(operatedWeights).toEqual(createModelWithValue(5));
  });
});
