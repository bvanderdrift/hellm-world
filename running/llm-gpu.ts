import { d } from "typegpu";
import {
  allocateInferenceBuffers,
  loadWeightsIntoGpu,
} from "../model/model-gpu-helpers.ts";
import { findTokenIndex } from "../model/model-helpers.ts";
import type { Model } from "../model/model-types.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import { extractMatrixBuffer } from "../shared/matrices-gpu.ts";
import { getRawVector } from "../shared/matrices.ts";
import { MAX_CONTEXT, pickToken } from "./llm-shared.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { forwardPassOnGPU } from "./llm-gpu-forward-pass.ts";

export const runLlmOnGPU = async function* (
  inputTokens: string[],
  model: Model,
) {
  let outputTokens: string[] = [];

  const weightBuffers = loadWeightsIntoGpu(model);

  const inferenceBuffers = allocateInferenceBuffers(MAX_CONTEXT, model);

  /** middle-state needed for backprop */
  const inputPositionToVocabPosition = inputTokens.map((token) => {
    return findTokenIndex(model.vocabulary, token);
  });

  const inputPositionToVocabPositionGPUBuffer = gpuContext
    .createBuffer(d.arrayOf(d.f32, MAX_CONTEXT), inputPositionToVocabPosition)
    .$usage("storage");

  for (let index = 0; index < MAX_CONTEXT; index++) {
    const nextInput = [...inputTokens, ...outputTokens];

    forwardPassOnGPU({
      weightBuffers,
      model,
      withActivations: false,
      inferenceBuffers,
      inputPositionToVocabPositionGPUBuffer,
    });

    const probabilities = await extractMatrixBuffer(
      inferenceBuffers.probabilitiesBuffer,
    );

    const nextToken = pickToken(
      getRawVector(probabilities, nextInput.length - 1),
      model.vocabulary,
    );

    if (nextToken === END_OF_SEQUENCE_TOKEN) {
      break;
    }

    inputPositionToVocabPositionGPUBuffer.patch({
      [nextInput.length]: findTokenIndex(model.vocabulary, nextToken),
    });

    outputTokens.push(nextToken);

    yield nextToken;
  }

  inferenceBuffers.hiddenState.buffer.destroy();

  inferenceBuffers.attentionInputBuffer.buffer.destroy();
  inferenceBuffers.attentionUpdateBuffer.buffer.destroy();
  inferenceBuffers.attentionInputKBuffer.buffer.destroy();
  inferenceBuffers.attentionInputVBuffer.buffer.destroy();
  inferenceBuffers.attentionInputQBuffer.buffer.destroy();
  inferenceBuffers.attentionOutBuffer.buffer.destroy();
  inferenceBuffers.attentionRelevancyOutput.buffer.destroy();
  inferenceBuffers.matchingKeyProducts.buffer.destroy();

  inferenceBuffers.mlpInputBuffer.buffer.destroy();
  inferenceBuffers.uppedMlpBuffer.buffer.destroy();
  inferenceBuffers.outMlpBuffer.buffer.destroy();

  inferenceBuffers.postTransformersBuffer.buffer.destroy();

  inputPositionToVocabPositionGPUBuffer.buffer.destroy();
  inferenceBuffers.unembeddedStateBuffer.buffer.destroy();
};
