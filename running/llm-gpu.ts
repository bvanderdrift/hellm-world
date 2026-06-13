import { d } from "typegpu";
import type {
  Activations,
  TransformerActivations,
} from "../model/activations-types.ts";
import { type WeightGPUBuffers } from "../model/model-gpu-helpers.ts";
import { findTokenIndex } from "../model/model-helpers.ts";
import type { Model } from "../model/model-types.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import {
  createMatrixBuffer,
  addMatricesOnGPU,
  extractMatrixBuffer,
  multiplyMatricesOnGPU,
} from "../shared/matrices-gpu.ts";
import { createMatrix, type Matrix } from "../shared/matrices.ts";
import { getMultilayerPerceptronActivationsOnGPU } from "../transforming/mlp-gpu.ts";
import { prepareHiddenState } from "./gpu-logic/prepareHiddenStateGPU.ts";
import { normalizeOnGpu } from "../shared/normalize-gpu.ts";
import { runSelfAttentionMechanismOnGPU } from "../transforming/attention/attention-gpu.ts";
import { divideToWhole } from "../shared/math.ts";

export const llmForwardPassByTokensOnGPU = async (
  input: string[],
  model: Model,
  weightBuffers: WeightGPUBuffers,
  withActivations: boolean,
): Promise<{
  embeddings: Matrix;
  activations: Activations | null;
}> => {
  const hiddenDimensionsSize = model.counts.hiddenDimensions;
  const contextSize = input.length;

  /** middle-state needed for backprop */
  const inputPositionToVocabPosition = input.map((token) => {
    return findTokenIndex(model.vocabulary, token);
  });

  const hiddenState = createMatrixBuffer({
    vectors: inputPositionToVocabPosition.length,
    dimensions: model.counts.hiddenDimensions,
  });

  const uppedMlpBuffer = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize * model.counts.mlpMultiple,
  });

  const outMlpBuffer = createMatrixBuffer({
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
    dimensions: hiddenDimensionsSize,
  });
  const matchingKeyProducts = createMatrixBuffer({
    vectors: contextSize,
    dimensions: hiddenDimensionsSize,
  });

  const inputPositionToVocabPositionGPUBuffer = gpuContext
    .createBuffer(
      d.arrayOf(d.f32, inputPositionToVocabPosition.length),
      inputPositionToVocabPosition,
    )
    .$usage("storage");

  prepareHiddenState(
    inputPositionToVocabPositionGPUBuffer,
    hiddenState,
    weightBuffers.embeddings,
  );

  const transformerActivations: TransformerActivations[] = [];

  const headDimensionsCount = divideToWhole(
    hiddenDimensionsSize,
    model.counts.attentionHeads,
  );

  const headDimensionsCountBuffer = gpuContext
    .createBuffer(d.u32, headDimensionsCount)
    .$usage("uniform");

  const contextLengthBuffer = gpuContext
    .createBuffer(d.u32, contextSize)
    .$usage("uniform");

  for (const transformerIndex in model.transformers) {
    const transformerBuffers = weightBuffers.transformers[transformerIndex]!;

    normalizeOnGpu(hiddenState);

    runSelfAttentionMechanismOnGPU(
      hiddenState,
      model.counts.attentionHeads,
      contextLengthBuffer,
      headDimensionsCountBuffer,
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

    normalizeOnGpu(hiddenState);

    getMultilayerPerceptronActivationsOnGPU(
      // Normalize input only, don't normalize the intermediateState iself
      // Reason: of this block outputs 0 for a feature, we keep x + 0 = x. But if we normalize the root variable we get norm(x) + 0 = norm(x) so a transform has still happened even if the block said not to
      hiddenState,
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

  normalizeOnGpu(hiddenState);

  const unembeddedStateBuffer = createMatrixBuffer(
    createMatrix(contextSize, model.vocabulary.length),
  );

  multiplyMatricesOnGPU(
    hiddenState,
    weightBuffers.unembeddings,
    unembeddedStateBuffer,
  );

  const missingTransformerActivationsCount =
    model.transformers.length - transformerActivations.length;

  if (withActivations && missingTransformerActivationsCount > 0) {
    // One sanity check, the rest is available either way
    throw new Error(
      `Missing ${missingTransformerActivationsCount} transformer activations`,
    );
  }

  const embeddings = await extractMatrixBuffer(unembeddedStateBuffer);

  hiddenState.buffer.destroy();
  uppedMlpBuffer.buffer.destroy();
  outMlpBuffer.buffer.destroy();
  attentionUpdateBuffer.buffer.destroy();
  attentionInputKBuffer.buffer.destroy();
  attentionInputVBuffer.buffer.destroy();
  attentionInputQBuffer.buffer.destroy();
  attentionOutBuffer.buffer.destroy();
  attentionRelevancyOutput.buffer.destroy();
  matchingKeyProducts.buffer.destroy();
  inputPositionToVocabPositionGPUBuffer.buffer.destroy();
  headDimensionsCountBuffer.buffer.destroy();
  contextLengthBuffer.buffer.destroy();
  unembeddedStateBuffer.buffer.destroy();

  return {
    embeddings,
    activations: null,
  };
};
