import type { Model } from "../model/model-types.ts";
import { MAX_CONTEXT } from "../running/llm-shared.ts";
import {
  createMatrixBuffer,
  type MatrixBuffer,
} from "../shared/matrices/matrices-gpu.ts";

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
  batchCount: number,
  model: Model,
): InferenceBuffers => {
  const hiddenDimensionsSize = model.counts.hiddenDimensions;
  const multiBatchContextSize = batchCount * contextSize;

  const hiddenState = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: hiddenDimensionsSize,
  });

  const attentionInputBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: hiddenDimensionsSize,
  });
  const attentionUpdateBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: hiddenDimensionsSize,
  });

  const attentionInputKBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: hiddenDimensionsSize,
  });
  const attentionInputVBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: hiddenDimensionsSize,
  });
  const attentionInputQBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: hiddenDimensionsSize,
  });
  const attentionOutBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: hiddenDimensionsSize,
  });
  const attentionRelevancyOutput = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: MAX_CONTEXT * model.counts.attentionHeads,
  });
  const matchingKeyProducts = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: MAX_CONTEXT * model.counts.attentionHeads,
  });
  const unembeddedStateBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: model.vocabulary.length,
  });
  const probabilitiesBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: model.vocabulary.length,
  });

  const mlpInputBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: hiddenDimensionsSize,
  });
  const uppedMlpBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: hiddenDimensionsSize * model.counts.mlpMultiple,
  });
  const outMlpBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
    dimensions: hiddenDimensionsSize,
  });

  const postTransformersBuffer = createMatrixBuffer({
    vectors: multiBatchContextSize,
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
