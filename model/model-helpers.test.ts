import { describe, expect, it } from "vitest";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { createMatrix, type Matrix } from "../shared/matrices.ts";
import {
  extractHiddenDimensionSize,
  findTokenIndex,
  operateCombinedWeights,
} from "./model-helpers.ts";
import type { Model } from "./model-types.ts";

const m = (rows: number, columns: number, value = 1): Matrix =>
  createMatrix(rows, columns, () => value);

const HIDDEN_DIMENSION_SIZE = 4;
const DEFAULT_MLP_MULTIPLE = 4;
const DEFAULT_MLP_DIMENSION_SIZE = HIDDEN_DIMENSION_SIZE * DEFAULT_MLP_MULTIPLE;
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

const createModelWithValue = (value: number): Model => ({
  ...validModel,
  embeddings: m(4, HIDDEN_DIMENSION_SIZE, value),
  unembeddings: m(HIDDEN_DIMENSION_SIZE, 4, value),
  transformers: [
    {
      attention: {
        Q: m(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
        K: m(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
        V: m(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
        out: m(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: m(
            HIDDEN_DIMENSION_SIZE,
            DEFAULT_MLP_DIMENSION_SIZE,
            value,
          ),
          biasVector: m(1, DEFAULT_MLP_DIMENSION_SIZE, value),
        },
        wDown: {
          weightsMatrix: m(
            DEFAULT_MLP_DIMENSION_SIZE,
            HIDDEN_DIMENSION_SIZE,
            value,
          ),
          biasVector: m(1, HIDDEN_DIMENSION_SIZE, value),
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
