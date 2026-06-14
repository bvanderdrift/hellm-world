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
  type MatrixBuffer,
} from "../shared/matrices-gpu.ts";
import type { Model, Weights } from "./model-types.ts";
import type { WgslArray } from "typegpu/data";

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

export const loadWeightsIntoGpu = (model: Model): WeightGPUBuffers => {
  const headDimensionsCount = divideToWhole(
    model.counts.hiddenDimensions,
    model.counts.attentionHeads,
  );

  const headDimensionsCountBuffer = gpuContext
    .createBuffer(d.u32, headDimensionsCount)
    .$usage("uniform");

  return {
    headDimensionsCountBuffer,
    embeddings: createMatrixBuffer(model.embeddings),
    unembeddings: createMatrixBuffer(model.unembeddings),
    transformers: model.transformers.map(
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

export type InferenceBuffers = {
  hiddenState: MatrixBuffer;
  attentionInputBuffer: MatrixBuffer;
  attentionUpdateBuffer: MatrixBuffer;
  attentionInputKBuffer: MatrixBuffer;
  attentionInputVBuffer: MatrixBuffer;
  attentionInputQBuffer: MatrixBuffer;
  attentionOutBuffer: MatrixBuffer;
  attentionRelevancyOutput: MatrixBuffer;
  matchingKeyProducts: MatrixBuffer;
  unembeddedStateBuffer: MatrixBuffer;
  probabilitiesBuffer: MatrixBuffer;
  mlpInputBuffer: MatrixBuffer;
  uppedMlpBuffer: MatrixBuffer;
  outMlpBuffer: MatrixBuffer;
  postTransformersBuffer: MatrixBuffer;
};

export const allocateInferenceBuffers = (
  contextSize: number,
  model: Model,
): InferenceBuffers => {
  const hiddenDimensionsSize = model.counts.hiddenDimensions;

  const hiddenState = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize,
  });

  const attentionInputBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize,
  });
  const attentionUpdateBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize,
  });

  const attentionInputKBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize,
  });
  const attentionInputVBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize,
  });
  const attentionInputQBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize,
  });
  const attentionOutBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize,
  });
  const attentionRelevancyOutput = createMatrixBuffer({
    vectors: contextSize,
    dimensions: contextSize * model.counts.attentionHeads,
  });
  const matchingKeyProducts = createMatrixBuffer({
    vectors: contextSize,
    dimensions: contextSize * model.counts.attentionHeads,
  });
  const unembeddedStateBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: model.vocabulary.length,
  });
  const probabilitiesBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: model.vocabulary.length,
  });

  const mlpInputBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize,
  });
  const uppedMlpBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize * model.counts.mlpMultiple,
  });
  const outMlpBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize,
  });

  const postTransformersBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize,
  });

  return {
    hiddenState,
    attentionInputBuffer,
    attentionUpdateBuffer,
    attentionInputKBuffer,
    attentionInputVBuffer,
    attentionInputQBuffer,
    attentionOutBuffer,
    attentionRelevancyOutput,
    matchingKeyProducts,
    unembeddedStateBuffer,
    probabilitiesBuffer,
    mlpInputBuffer,
    uppedMlpBuffer,
    outMlpBuffer,
    postTransformersBuffer,
  };
};
