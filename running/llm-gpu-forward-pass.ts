import {
  addMatricesOnGPU,
  multiplyMatricesOnGPU,
} from "../shared/matrices-gpu.ts";
import type { TransformerActivations } from "../model/activations-types.ts";
import { normalizeOnGpu } from "../shared/normalize-gpu.ts";
import { softmaxOnGpu } from "../shared/softmax-gpu.ts";
import { runSelfAttentionMechanismOnGPU } from "../transforming/attention/attention-gpu.ts";
import { getMultilayerPerceptronActivationsOnGPU } from "../transforming/mlp-gpu.ts";
import { prepareHiddenState } from "./gpu-logic/prepareHiddenStateGPU.ts";
import type {
  InferenceBuffers,
  WeightGPUBuffers,
} from "../model/model-gpu-helpers.ts";
import type { Model } from "../model/model-types.ts";
import type { StorageFlag, TgpuBuffer } from "typegpu";
import type { F32, WgslArray } from "typegpu/data";

export const forwardPassOnGPU = ({
  weightBuffers,
  model,
  withActivations,
  inferenceBuffers: {
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
  },
  inputPositionToVocabPositionGPUBuffer,
}: ForwardPassInput) => {
  prepareHiddenState(
    inputPositionToVocabPositionGPUBuffer,
    hiddenState,
    weightBuffers.embeddings,
  );

  const transformerActivations: TransformerActivations[] = [];

  for (const transformerBuffers of weightBuffers.transformers) {
    normalizeOnGpu(hiddenState, attentionInputBuffer);

    runSelfAttentionMechanismOnGPU(
      attentionInputBuffer,
      model.counts.attentionHeads,
      weightBuffers.headDimensionsCountBuffer,
      transformerBuffers.attention,
      attentionInputQBuffer,
      attentionInputKBuffer,
      attentionInputVBuffer,
      attentionRelevancyOutput,
      matchingKeyProducts,
      attentionOutBuffer,
      attentionUpdateBuffer,
    );

    addMatricesOnGPU(hiddenState, attentionUpdateBuffer);

    normalizeOnGpu(hiddenState, mlpInputBuffer);

    getMultilayerPerceptronActivationsOnGPU(
      // Normalize input only, don't normalize the intermediateState iself
      // Reason: of this block outputs 0 for a feature, we keep x + 0 = x. But if we normalize the root variable we get norm(x) + 0 = norm(x) so a transform has still happened even if the block said not to
      mlpInputBuffer,
      uppedMlpBuffer,
      outMlpBuffer,
      transformerBuffers.multilayerPerceptron,
    );

    // Apply updated knowledge
    addMatricesOnGPU(hiddenState, outMlpBuffer);

    transformerActivations.push({
      attention: null as unknown,
      mlp: null as unknown,
      transformerInput: null as unknown,
    } as any);
  }

  normalizeOnGpu(hiddenState, postTransformersBuffer);

  const missingTransformerActivationsCount =
    model.transformers.length - transformerActivations.length;

  if (withActivations && missingTransformerActivationsCount > 0) {
    // One sanity check, the rest is available either way
    throw new Error(
      `Missing ${missingTransformerActivationsCount} transformer activations`,
    );
  }

  multiplyMatricesOnGPU(
    postTransformersBuffer,
    weightBuffers.unembeddings,
    unembeddedStateBuffer,
  );

  softmaxOnGpu(unembeddedStateBuffer, probabilitiesBuffer);
};

// moved it down here to not bloat top of file
type ForwardPassInput = {
  model: Model;
  weightBuffers: WeightGPUBuffers;
  withActivations: boolean;
  inferenceBuffers: InferenceBuffers;
  inputPositionToVocabPositionGPUBuffer: TgpuBuffer<WgslArray<F32>> &
    StorageFlag;
};
