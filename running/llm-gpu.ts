import { d } from "typegpu";
import type { TransformerActivations } from "../model/activations-types.ts";
import {
  loadWeightsIntoGpu,
  type WeightGPUBuffers,
} from "../model/model-gpu-helpers.ts";
import { findTokenIndex } from "../model/model-helpers.ts";
import type { Model } from "../model/model-types.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import {
  createMatrixBuffer,
  addMatricesOnGPU,
  extractMatrixBuffer,
  multiplyMatricesOnGPU,
} from "../shared/matrices-gpu.ts";
import { getRawVector, type Matrix } from "../shared/matrices.ts";
import { getMultilayerPerceptronActivationsOnGPU } from "../transforming/mlp-gpu.ts";
import { prepareHiddenState } from "./gpu-logic/prepareHiddenStateGPU.ts";
import { normalizeOnGpu } from "../shared/normalize-gpu.ts";
import { runSelfAttentionMechanismOnGPU } from "../transforming/attention/attention-gpu.ts";
import { divideToWhole } from "../shared/math.ts";
import { MAX_CONTEXT, pickToken } from "./llm-shared.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { softmaxOnGpu } from "../shared/softmax-gpu.ts";
import { forwardPassOnGPU } from "./llm-gpu-forward-pass.ts";

export const runLlmOnGPU = async function* (
  inputTokens: string[],
  model: Model,
) {
  let outputTokens: string[] = [];

  const weightBuffers = loadWeightsIntoGpu(model);

  for (let index = 0; index < MAX_CONTEXT; index++) {
    const nextInput = [...inputTokens, ...outputTokens];

    const probabilities = await llmForwardPassByTokensOnGPU(
      nextInput,
      model,
      weightBuffers,
      false,
    );

    const nextToken = pickToken(
      getRawVector(probabilities, probabilities.vectors - 1),
      model.vocabulary,
    );

    if (nextToken === END_OF_SEQUENCE_TOKEN) {
      break;
    }

    outputTokens.push(nextToken);

    yield nextToken;
  }
};

export const llmForwardPassByTokensOnGPU = async (
  input: string[],
  model: Model,
  weightBuffers: WeightGPUBuffers,
  withActivations: boolean,
): Promise<Matrix> => {
  const hiddenDimensionsSize = model.counts.hiddenDimensions;
  const contextSize = input.length;

  /** middle-state needed for backprop */
  const inputPositionToVocabPosition = input.map((token) => {
    return findTokenIndex(model.vocabulary, token);
  });

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

  const inputPositionToVocabPositionGPUBuffer = gpuContext
    .createBuffer(
      d.arrayOf(d.f32, inputPositionToVocabPosition.length),
      inputPositionToVocabPosition,
    )
    .$usage("storage");

  forwardPassOnGPU({
    weightBuffers,
    model,
    withActivations,
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
    inputPositionToVocabPositionGPUBuffer,
  });

  const probabilities = await extractMatrixBuffer(probabilitiesBuffer);

  hiddenState.buffer.destroy();

  attentionInputBuffer.buffer.destroy();
  attentionUpdateBuffer.buffer.destroy();
  attentionInputKBuffer.buffer.destroy();
  attentionInputVBuffer.buffer.destroy();
  attentionInputQBuffer.buffer.destroy();
  attentionOutBuffer.buffer.destroy();
  attentionRelevancyOutput.buffer.destroy();
  matchingKeyProducts.buffer.destroy();

  mlpInputBuffer.buffer.destroy();
  uppedMlpBuffer.buffer.destroy();
  outMlpBuffer.buffer.destroy();

  postTransformersBuffer.buffer.destroy();

  inputPositionToVocabPositionGPUBuffer.buffer.destroy();
  unembeddedStateBuffer.buffer.destroy();

  return probabilities;
};
