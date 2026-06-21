import { createMatrix, type Matrix } from "../shared/matrices/matrices.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import type { Model, ModelTrainingState } from "../model/model-types.ts";

export const HIDDEN_DIMENSION_SIZE = 4;
export const DEFAULT_MLP_MULTIPLE = 4;
export const DEFAULT_MLP_DIMENSION_SIZE =
  HIDDEN_DIMENSION_SIZE * DEFAULT_MLP_MULTIPLE;

export const emptyTrainingState: ModelTrainingState = {
  trainingLosses: [],
  validationLosses: [],
  samplerState: { type: "uniform" },
};

export const constantMatrix = (
  rows: number,
  columns: number,
  value = 1,
): Matrix => createMatrix(rows, columns, () => value);

export const createConstantModel = (value = 1): Model => ({
  name: "test-model",
  vocabulary: ["hello", "world", "beer", END_OF_SEQUENCE_TOKEN],
  trainingState: emptyTrainingState,
  counts: {
    attentionHeads: 2,
    mlpMultiple: DEFAULT_MLP_MULTIPLE,
    transformers: 1,
    hiddenDimensions: HIDDEN_DIMENSION_SIZE,
  },
  embeddings: constantMatrix(4, HIDDEN_DIMENSION_SIZE, value),
  unembeddings: constantMatrix(HIDDEN_DIMENSION_SIZE, 4, value),
  transformers: [
    {
      attention: {
        Q: constantMatrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
        K: constantMatrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
        V: constantMatrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
        out: constantMatrix(HIDDEN_DIMENSION_SIZE, HIDDEN_DIMENSION_SIZE, value),
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: constantMatrix(
            HIDDEN_DIMENSION_SIZE,
            DEFAULT_MLP_DIMENSION_SIZE,
            value,
          ),
          biasVector: constantMatrix(1, DEFAULT_MLP_DIMENSION_SIZE, value),
        },
        wDown: {
          weightsMatrix: constantMatrix(
            DEFAULT_MLP_DIMENSION_SIZE,
            HIDDEN_DIMENSION_SIZE,
            value,
          ),
          biasVector: constantMatrix(1, HIDDEN_DIMENSION_SIZE, value),
        },
      },
    },
  ],
});
