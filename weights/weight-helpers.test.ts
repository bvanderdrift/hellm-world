import { describe, expect, it } from "vitest";
import { extractDimensionSizes, validateSizing } from "./weight-helpers.ts";
import { toyWeights, type Token } from "./toy_weights/toyWeights.ts";
import type { Weights } from "./types.ts";

const cloneEmbeddings = (): Record<Token, number[]> =>
  Object.fromEntries(
    Object.entries(toyWeights.embeddings).map(([token, vector]) => [
      token,
      [...vector],
    ]),
  ) as Record<Token, number[]>;

const createWeights = (
  overrides: Partial<Weights<Token>> = {},
): Weights<Token> => ({
  tokens: [...toyWeights.tokens],
  headsCount: toyWeights.headsCount,
  embeddings: cloneEmbeddings(),
  unembeddings: toyWeights.unembeddings.map((row) => [...row]),
  transformers: toyWeights.transformers,
  ...overrides,
});

describe("extractDimensionSizes", () => {
  it("derives the hidden width and vocab size from embeddings", () => {
    expect(extractDimensionSizes(toyWeights)).toEqual({
      hiddenDimensionsSize: 4,
      vocabSize: 6,
    });
  });
});

describe("validateSizing", () => {
  it("accepts a self-consistent checkpoint with unique multi-character tokens", () => {
    expect(() => validateSizing(toyWeights)).not.toThrow();
  });

  it("rejects a headsCount that does not evenly divide the hidden width", () => {
    const malformedWeights = createWeights({
      headsCount: 3,
    });

    expect(() => validateSizing(malformedWeights)).toThrow(
      "Can't perfectly divide the nominator by denominator (3)",
    );
  });

  it("rejects duplicate tokens in the checkpoint vocabulary", () => {
    const duplicateTokenWeights = createWeights({
      tokens: ["hello", "world", "my", "name", "is", "hello"],
    });

    expect(() => validateSizing(duplicateTokenWeights)).toThrow(
      "Provided weights have 1 duplicate tokens",
    );
  });

  it("throws when any embedding row has the wrong width", () => {
    const malformedWeights = createWeights({
      embeddings: {
        ...cloneEmbeddings(),
        world: [1, 1, 1],
      },
    });

    expect(() => validateSizing(malformedWeights)).toThrow(
      "Token world has unexpected vector length 3 vs base length 4",
    );
  });
});
