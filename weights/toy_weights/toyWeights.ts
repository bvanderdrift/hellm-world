import type { Token } from "../../tokenizer.ts";
import type { Weights } from "../types.ts";
import { extractDimensionSizes } from "../weight-helpers.ts";

const toyWeightsBare: Weights<Token> = {
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
        heads: [
          {
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
            V: {
              up: [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
              ],
              down: [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
              ],
            },
          },
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
