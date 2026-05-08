import {
  createMatrixBuffer,
  type MatrixBuffer,
} from "../shared/matrices-gpu.ts";
import type { Weights } from "./model-types.ts";

export interface AttentionGPUBuffers {
  Q: MatrixBuffer;
  K: MatrixBuffer;
  V: MatrixBuffer;
  out: MatrixBuffer;
}

export interface MultilayerPerceptronGPUBuffers {
  wUp: {
    weightsMatrix: MatrixBuffer;
    biasVector: MatrixBuffer;
  };
  wDown: {
    weightsMatrix: MatrixBuffer;
    biasVector: MatrixBuffer;
  };
}

export interface TransformerGPUBuffers {
  attention: AttentionGPUBuffers;
  multilayerPerceptron: MultilayerPerceptronGPUBuffers;
}

export type WeightGPUBuffers = {
  embeddings: MatrixBuffer; // T x D
  unembeddings: MatrixBuffer; // D x T
  transformers: TransformerGPUBuffers[];
};

export const loadWeightsIntoGpu = (weight: Weights): WeightGPUBuffers => {
  return {
    embeddings: createMatrixBuffer(weight.embeddings),
    unembeddings: createMatrixBuffer(weight.unembeddings),
    transformers: weight.transformers.map(
      (t): TransformerGPUBuffers => ({
        attention: {
          K: createMatrixBuffer(t.attention.K),
          V: createMatrixBuffer(t.attention.V),
          Q: createMatrixBuffer(t.attention.Q),
          out: createMatrixBuffer(t.attention.out),
        },
        multilayerPerceptron: {
          wDown: {
            weightsMatrix: createMatrixBuffer(
              t.multilayerPerceptron.wDown.weightsMatrix,
            ),
            biasVector: createMatrixBuffer([
              t.multilayerPerceptron.wDown.biasVector,
            ]),
          },
          wUp: {
            weightsMatrix: createMatrixBuffer(
              t.multilayerPerceptron.wUp.weightsMatrix,
            ),
            biasVector: createMatrixBuffer([
              t.multilayerPerceptron.wUp.biasVector,
            ]),
          },
        },
      }),
    ),
  };
};
