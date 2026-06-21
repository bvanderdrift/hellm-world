import {
  d,
  type StorageFlag,
  type TgpuBuffer,
  type UniformFlag,
} from "typegpu";
import { gpuContext } from "../shared/gpu-context.ts";
import { divideToWhole } from "../shared/math.ts";
import {
  createMatrixBuffer,
  extractMatrixBuffer,
  type MatrixBuffer,
} from "../shared/matrices/matrices-gpu.ts";
import type {
  Model,
  TransformerWeights,
  Weights,
} from "../model/model-types.ts";
import type { WgslArray } from "typegpu/data";
import { MAX_CONTEXT } from "../running/llm-shared.ts";

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
  headDimensionsCountBuffer: TgpuBuffer<d.U32> & UniformFlag;
  embeddings: MatrixBuffer; // T x D
  unembeddings: MatrixBuffer; // D x T
  transformers: TransformerGPUBuffers[];
};

export const loadWeightsIntoGpu = (
  counts: Model["counts"],
  weights: Weights,
): WeightGPUBuffers => {
  const headDimensionsCount = divideToWhole(
    counts.hiddenDimensions,
    counts.attentionHeads,
  );

  const headDimensionsCountBuffer = gpuContext
    .createBuffer(d.u32, headDimensionsCount)
    .$usage("uniform");

  return {
    headDimensionsCountBuffer,
    embeddings: createMatrixBuffer(weights.embeddings),
    unembeddings: createMatrixBuffer(weights.unembeddings),
    transformers: weights.transformers.map(
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
            biasVector: createMatrixBuffer(
              t.multilayerPerceptron.wDown.biasVector,
            ),
          },
          wUp: {
            weightsMatrix: createMatrixBuffer(
              t.multilayerPerceptron.wUp.weightsMatrix,
            ),
            biasVector: createMatrixBuffer(
              t.multilayerPerceptron.wUp.biasVector,
            ),
          },
        },
      }),
    ),
  };
};

export const extractWeightsFromGpu = async (
  weights: WeightGPUBuffers,
): Promise<Weights> => {
  return {
    embeddings: await extractMatrixBuffer(weights.embeddings),
    transformers: await Promise.all(
      weights.transformers.map(
        async (transformerBuffers): Promise<TransformerWeights> => ({
          attention: {
            K: await extractMatrixBuffer(transformerBuffers.attention.K),
            V: await extractMatrixBuffer(transformerBuffers.attention.V),
            Q: await extractMatrixBuffer(transformerBuffers.attention.Q),
            out: await extractMatrixBuffer(transformerBuffers.attention.out),
          },
          multilayerPerceptron: {
            wUp: {
              weightsMatrix: await extractMatrixBuffer(
                transformerBuffers.multilayerPerceptron.wUp.weightsMatrix,
              ),
              biasVector: await extractMatrixBuffer(
                transformerBuffers.multilayerPerceptron.wUp.biasVector,
              ),
            },
            wDown: {
              weightsMatrix: await extractMatrixBuffer(
                transformerBuffers.multilayerPerceptron.wDown.weightsMatrix,
              ),
              biasVector: await extractMatrixBuffer(
                transformerBuffers.multilayerPerceptron.wDown.biasVector,
              ),
            },
          },
        }),
      ),
    ),
    unembeddings: await extractMatrixBuffer(weights.unembeddings),
  };
};
