import { d } from "typegpu";
import { loadWeightsIntoGpu } from "../model-gpu/model-weights-gpu.ts";
import { findTokenIndex } from "../model/model-helpers.ts";
import type { Model } from "../model/model-types.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import { extractMatrixBuffer } from "../shared/matrices/matrices-gpu.ts";
import { getRawVector } from "../shared/matrices/matrices.ts";
import { MAX_CONTEXT, pickToken } from "./llm-shared.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { forwardPassOnGPU } from "./llm-gpu-forward-pass.ts";
import { allocateInferenceBuffers } from "../model-gpu/model-activations.gpu.ts";

export const runLlmOnGPU = async function* (batches: string[][], model: Model) {
  const weightBuffers = loadWeightsIntoGpu(model);

  const inferenceBuffers = allocateInferenceBuffers(
    MAX_CONTEXT,
    batches.length,
    model,
  );

  const ended = batches.map(() => false);

  /** middle-state needed for backprop */
  const inputPositionToVocabPosition = batches.flatMap((inputTokens) =>
    new Array(MAX_CONTEXT).fill(0).map((_, tokenIndex) => {
      const token = inputTokens[tokenIndex];

      if (!token) {
        return 0;
      }

      return findTokenIndex(model.vocabulary, token);
    }),
  );

  const inputPositionToVocabPositionGPUBuffer = gpuContext
    .createBuffer(
      d.arrayOf(d.f32, batches.length * MAX_CONTEXT),
      inputPositionToVocabPosition,
    )
    .$usage("storage");

  for (let index = 0; index < MAX_CONTEXT; index++) {
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

    const nextTokens = batches.map((inputTokens, batchIndex) => {
      const nextToken = pickToken(
        getRawVector(
          probabilities,
          batchIndex * MAX_CONTEXT + inputTokens.length - 1,
        ),
        model.vocabulary,
      );

      if (nextToken === END_OF_SEQUENCE_TOKEN) {
        ended[batchIndex] = true;
      }

      return nextToken;
    });

    if (ended.every((state) => state === true)) {
      break;
    }

    batches = batches.map((inputTokens, batchIndex) => {
      inputTokens.push(nextTokens[batchIndex]!);

      return inputTokens;
    });

    inputPositionToVocabPositionGPUBuffer.patch(
      Object.fromEntries(
        batches.map((inputTokens, batchIndex) => [
          batchIndex * MAX_CONTEXT + inputTokens.length - 1,
          findTokenIndex(model.vocabulary, nextTokens[batchIndex]!),
        ]),
      ),
    );

    yield nextTokens;
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
