import { describe, expect, it } from "vitest";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import {
  extractHiddenDimensionSize,
  findTokenIndex,
  operateCombinedWeights,
  validateSameWeightShape,
  validateWeights,
} from "./weight-helpers.ts";
import type { Model } from "./types.ts";

const vector = (length: number, value = 1) => new Array(length).fill(value);

const matrix = (rows: number, columns: number, value = 1) =>
  new Array(rows).fill(null).map(() => vector(columns, value));

const validWeights: Model = {
  vocabulary: ["hello", "world", "beer", END_OF_SEQUENCE_TOKEN],
  headsCount: 2,
  embeddings: matrix(4, 4),
  unembeddings: matrix(4, 4),
  transformers: [
    {
      attention: {
        Q: matrix(4, 4),
        K: matrix(4, 4),
        V: matrix(4, 4),
        out: matrix(4, 4),
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: matrix(4, 16),
          biasVector: vector(16),
        },
        wDown: {
          weightsMatrix: matrix(16, 4),
          biasVector: vector(4),
        },
      },
    },
  ],
};

const createWeights = (overrides: Partial<Model> = {}): Model => ({
  ...structuredClone(validWeights),
  ...overrides,
});

const createWeightsWithValue = (value: number): Model => ({
  ...validWeights,
  embeddings: matrix(4, 4, value),
  unembeddings: matrix(4, 4, value),
  transformers: [
    {
      attention: {
        Q: matrix(4, 4, value),
        K: matrix(4, 4, value),
        V: matrix(4, 4, value),
        out: matrix(4, 4, value),
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: matrix(4, 16, value),
          biasVector: vector(16, value),
        },
        wDown: {
          weightsMatrix: matrix(16, 4, value),
          biasVector: vector(4, value),
        },
      },
    },
  ],
});

describe("extractHiddenDimensionSize", () => {
  it("derives the hidden width and vocab size from embeddings", () => {
    expect(extractHiddenDimensionSize(validWeights)).toEqual(4);
  });
});

describe("findTokenIndex", () => {
  it("returns the vocabulary index for a known token", () => {
    expect(findTokenIndex(validWeights.vocabulary, "beer")).toBe(2);
  });

  it("throws when the token is not in the vocabulary", () => {
    expect(() => findTokenIndex(validWeights.vocabulary, "ghost")).toThrow(
      "Failed to find token ghost in vocabulary",
    );
  });
});

describe("validateWeights", () => {
  it("accepts a self-consistent checkpoint", () => {
    expect(() => validateWeights(validWeights)).not.toThrow();
  });

  it("rejects a headsCount that does not evenly divide the hidden width", () => {
    const malformedWeights = createWeights({
      headsCount: 3,
    });

    expect(() => validateWeights(malformedWeights)).toThrow(
      "Can't perfectly divide the nominator by denominator (3)",
    );
  });

  it("rejects duplicate tokens in the checkpoint vocabulary", () => {
    const duplicateTokenWeights = createWeights({
      vocabulary: ["hello", "world", "hello", END_OF_SEQUENCE_TOKEN],
    });

    expect(() => validateWeights(duplicateTokenWeights)).toThrow(
      "Provided weights have 1 duplicate tokens",
    );
  });

  it("rejects vocabularies that are missing the EOS token", () => {
    const malformedWeights = createWeights({
      vocabulary: ["hello", "world", "beer"],
      embeddings: matrix(3, 4),
      unembeddings: matrix(4, 3),
    });

    expect(() => validateWeights(malformedWeights)).toThrow(
      `Model embeddings are missing special end-of-sequence token "${END_OF_SEQUENCE_TOKEN}"`,
    );
  });

  it("rejects embeddings that do not cover the full vocabulary", () => {
    const malformedWeights = createWeights({
      embeddings: matrix(3, 4),
    });

    expect(() => validateWeights(malformedWeights)).toThrow(
      "matrix vector count (3) doesn't match expected vector count 4",
    );
  });

  it("rejects embeddings with too many rows for the vocabulary", () => {
    const malformedWeights = createWeights({
      embeddings: matrix(5, 4),
    });

    expect(() => validateWeights(malformedWeights)).toThrow(
      "matrix vector count (5) doesn't match expected vector count 4",
    );
  });

  it("throws when any embedding row has the wrong width", () => {
    const malformedWeights = createWeights({
      embeddings: [
        [1, 1, 1, 1],
        [1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
      ],
    });

    expect(() => validateWeights(malformedWeights)).toThrow(
      "Vector at index 1 has unexpected depth 3 (expected 4)",
    );
  });

  it("rejects checkpoints with empty vocabulary using a helpful error", () => {
    const malformedWeights = createWeights({
      vocabulary: [],
      embeddings: [],
      unembeddings: [],
    });

    expect(() => validateWeights(malformedWeights)).toThrow(
      "Provided vocabulary cannot be empty",
    );
  });

  it("rejects a negative headsCount", () => {
    const malformedWeights = createWeights({
      headsCount: -2,
    });

    expect(() => validateWeights(malformedWeights)).toThrow(
      "headsCount must be a positive integer",
    );
  });
});

describe("validateSameWeightShape", () => {
  it("accepts weights with the same shape", () => {
    expect(() =>
      validateSameWeightShape(createWeights(), createWeights()),
    ).not.toThrow();
  });

  it("rejects embeddings with different vocabulary sizes", () => {
    const weightsWithMoreEmbeddings = createWeights({
      embeddings: matrix(5, 4),
    });

    expect(() =>
      validateSameWeightShape(weightsWithMoreEmbeddings, createWeights()),
    ).toThrow("matrix vector count (5) doesn't match expected vector count 4");
  });

  it("rejects embeddings with different hidden widths", () => {
    const weightsWithWiderEmbeddings = createWeights({
      embeddings: matrix(4, 5),
    });

    expect(() =>
      validateSameWeightShape(weightsWithWiderEmbeddings, createWeights()),
    ).toThrow("m has unexpected vector depth 5, expected 4");
  });

  it("rejects different transformer counts", () => {
    const weightsWithExtraTransformer = createWeights({
      transformers: [
        ...structuredClone(validWeights.transformers),
        structuredClone(validWeights.transformers[0]!),
      ],
    });

    expect(() =>
      validateSameWeightShape(weightsWithExtraTransformer, createWeights()),
    ).toThrow("Weights1 has different transformers count 2 than Weights2 (1)");
  });

  it("rejects different head counts", () => {
    const weightsWithDifferentHeadCount = createWeights({
      headsCount: 4,
    });

    expect(() =>
      validateSameWeightShape(weightsWithDifferentHeadCount, createWeights()),
    ).toThrow("Weights1 has different head count 4 than Weights2 (2)");
  });
});

describe("operateWeights", () => {
  it("applies the operation across embeddings, unembeddings, attention, and MLP weights", () => {
    const operatedWeights = operateCombinedWeights(
      createWeightsWithValue(2),
      createWeightsWithValue(3),
      (value1, value2) => value1 + value2,
    );

    expect(operatedWeights).toEqual(createWeightsWithValue(5));
  });

  it("rejects weights with the same shape but a different vocabulary order", () => {
    const weightsWithReorderedVocabulary = createWeights({
      vocabulary: ["world", "hello", "beer", END_OF_SEQUENCE_TOKEN],
    });

    expect(() =>
      operateCombinedWeights(
        weightsWithReorderedVocabulary,
        createWeights(),
        (value1, value2) => value1 + value2,
      ),
    ).toThrow();
  });
});
