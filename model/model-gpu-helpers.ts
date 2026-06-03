import {
  createMatrixBufferAndCopy,
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
    embeddings: createMatrixBufferAndCopy(weight.embeddings),
    unembeddings: createMatrixBufferAndCopy(weight.unembeddings),
    transformers: weight.transformers.map(
      (t): TransformerGPUBuffers => ({
        attention: {
          K: createMatrixBufferAndCopy(t.attention.K),
          V: createMatrixBufferAndCopy(t.attention.V),
          Q: createMatrixBufferAndCopy(t.attention.Q),
          out: createMatrixBufferAndCopy(t.attention.out),
        },
        multilayerPerceptron: {
          wDown: {
            weightsMatrix: createMatrixBufferAndCopy(
              t.multilayerPerceptron.wDown.weightsMatrix,
            ),
            biasVector: createMatrixBufferAndCopy(
              t.multilayerPerceptron.wDown.biasVector,
            ),
          },
          wUp: {
            weightsMatrix: createMatrixBufferAndCopy(
              t.multilayerPerceptron.wUp.weightsMatrix,
            ),
            biasVector: createMatrixBufferAndCopy(
              t.multilayerPerceptron.wUp.biasVector,
            ),
          },
        },
      }),
    ),
  };
};
