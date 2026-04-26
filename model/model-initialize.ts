import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { randomNormalDistribution } from "../shared/math.ts";
import { createMatrix, createVector } from "../shared/matrices.ts";
import { validateModel } from "./model-helpers.ts";
import type { AttentionWeights, Model, TransformerWeights } from "./types.ts";

export const decodeVocab = (input: string) => {
  return input.split(",").filter((token) => !!token);
};

export const initializeModel = ({
  headsCount,
  hiddenDimensionCount,
  transformerCount,
  vocabulary: vocabularyWithoutEosMaybe,
}: {
  headsCount: number;
  hiddenDimensionCount: number;
  transformerCount: number;
  vocabulary: string[];
}): Model => {
  const mlpMultiple = 4;

  const vocabulary = Array.from(
    new Set([...vocabularyWithoutEosMaybe, END_OF_SEQUENCE_TOKEN]),
  );

  const weightRandomNumberGenerator = () => randomNormalDistribution(0, 0.02);

  const model: Model = {
    vocabulary,
    headsCount,
    mlpMultiple,
    transformers: new Array(transformerCount)
      .fill(0)
      .map((_): TransformerWeights => {
        return {
          attention: {
            Q: createMatrix(
              hiddenDimensionCount,
              hiddenDimensionCount,
              weightRandomNumberGenerator,
            ),
            K: createMatrix(
              hiddenDimensionCount,
              hiddenDimensionCount,
              weightRandomNumberGenerator,
            ),
            V: createMatrix(
              hiddenDimensionCount,
              hiddenDimensionCount,
              weightRandomNumberGenerator,
            ),
            out: createMatrix(
              hiddenDimensionCount,
              hiddenDimensionCount,
              weightRandomNumberGenerator,
            ),
          },
          multilayerPerceptron: {
            wUp: {
              weightsMatrix: createMatrix(
                hiddenDimensionCount,
                hiddenDimensionCount * mlpMultiple,
                weightRandomNumberGenerator,
              ),
              biasVector: createVector(hiddenDimensionCount * mlpMultiple),
            },
            wDown: {
              weightsMatrix: createMatrix(
                hiddenDimensionCount * mlpMultiple,
                hiddenDimensionCount,
                weightRandomNumberGenerator,
              ),
              biasVector: createVector(hiddenDimensionCount),
            },
          },
        };
      }),
    embeddings: createMatrix(
      vocabulary.length,
      hiddenDimensionCount,
      weightRandomNumberGenerator,
    ),
    unembeddings: createMatrix(
      hiddenDimensionCount,
      vocabulary.length,
      weightRandomNumberGenerator,
    ),
  };

  validateModel(model);

  return model;
};
