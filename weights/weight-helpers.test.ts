import { describe, expect, it } from "vitest";
import { extractDimensionSizes, validateWeights } from "./weight-helpers.ts";
import type { Weights } from "./types.ts";

const vector = (length: number, value = 1) => new Array(length).fill(value);

const matrix = (rows: number, columns: number, value = 1) =>
  new Array(rows).fill(null).map(() => vector(columns, value));

const validWeights: Weights = {
  vocabulary: ["hello", "world", "beer"],
  headsCount: 2,
  embeddings: {
    hello: [1, 1, 1, 1],
    world: [1, 1, 1, 1],
    beer: [1, 1, 1, 1],
  },
  unembeddings: matrix(4, 3),
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

const cloneEmbeddings = () => structuredClone(validWeights.embeddings);

const createWeights = (overrides: Partial<Weights> = {}): Weights => ({
  ...structuredClone(validWeights),
  ...overrides,
});

describe("extractDimensionSizes", () => {
  it("derives the hidden width and vocab size from embeddings", () => {
    expect(extractDimensionSizes(validWeights)).toEqual({
      hiddenDimensionsSize: 4,
      vocabSize: 3,
    });
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
      vocabulary: ["hello", "world", "hello"],
    });

    expect(() => validateWeights(duplicateTokenWeights)).toThrow(
      "Provided weights have 1 duplicate tokens",
    );
  });

  it("rejects embeddings that do not cover the full vocabulary", () => {
    const { beer: _, ...embeddingsMissingBeer } = cloneEmbeddings();
    const malformedWeights = createWeights({
      embeddings: embeddingsMissingBeer,
    });

    expect(() => validateWeights(malformedWeights)).toThrow(
      "Provided embeddings has unexpected vocabulary size 2, expected 3",
    );
  });

  it("rejects embeddings whose keys do not exactly match the vocabulary", () => {
    const { beer: _, ...embeddingsWithoutBeer } = cloneEmbeddings();
    const malformedWeights = createWeights({
      embeddings: {
        ...embeddingsWithoutBeer,
        ghost: [1, 1, 1, 1],
      },
    });

    expect(() => validateWeights(malformedWeights)).toThrow(
      "Unknown embedding token ghost. Does not occur in vocabulary of model.",
    );
  });

  it("throws when any embedding row has the wrong width", () => {
    const malformedWeights = createWeights({
      embeddings: {
        ...cloneEmbeddings(),
        world: [1, 1, 1],
      },
    });

    expect(() => validateWeights(malformedWeights)).toThrow(
      "Token world has unexpected vector length 3 vs base length 4",
    );
  });

  it("rejects checkpoints with empty vocabulary using a helpful error", () => {
    const malformedWeights = createWeights({
      vocabulary: [],
      embeddings: {},
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
