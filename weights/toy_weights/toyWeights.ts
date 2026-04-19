import type { Weights } from "../types.ts";
import { extractDimensionSizes } from "../weight-helpers.ts";

export const tokens = ["hello", "world", "my", "name", "is", "beer"] as const;
export type Token = (typeof tokens)[number];

const toyWeightsBare: Weights<Token> = {
  tokens: [...tokens],
  headsCount: 2,
  embeddings: {
    hello: [1, 1, 1, 1],
    world: [1, 1, 1, 1],
    my: [1, 1, 1, 1],
    name: [1, 1, 1, 1],
    is: [1, 1, 1, 1],
    beer: [1, 1, 1, 1],
  },
  unembeddings: [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
  ],
  transformers: [
    {
      attention: {
        Q: [
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
        ],
        K: [
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
        ],
        V: [
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
        ],
        out: [
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
        ],
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          ],
          biasVector: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        wDown: {
          weightsMatrix: [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
          ],
          biasVector: [1, 1, 1, 1],
        },
      },
    },
  ],
};

export const toyWeights: Weights<Token> & {
  hiddenDimensionsSize: number;
  vocabSize: number;
} = {
  ...toyWeightsBare,
  ...extractDimensionSizes(toyWeightsBare),
};
