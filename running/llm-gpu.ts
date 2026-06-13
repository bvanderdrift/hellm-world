import { d } from "typegpu";
import type {
  Activations,
  TransformerActivations,
} from "../model/activations-types.ts";
import { loadWeightsIntoGpu } from "../model/model-gpu-helpers.ts";
import { findTokenIndex } from "../model/model-helpers.ts";
import type { Model } from "../model/model-types.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import {
  createMatrixBufferAndCopy,
  addMatricesOnGPU,
  extractMatrixBuffer,
  createMatrixBuffer,
  multiplyMatricesOnGPU,
} from "../shared/matrices-gpu.ts";
import { createMatrix, type Matrix } from "../shared/matrices.ts";
import { getMultilayerPerceptronActivationsOnGPU } from "../transforming/mlp-gpu.ts";
import { prepareHiddenState } from "./gpu-logic/prepareHiddenStateGPU.ts";
import { normalizeOnGpu } from "../shared/normalize-gpu.ts";
import { runSelfAttentionMechanismOnGPU } from "../transforming/attention-gpu.ts";
import { divideToWhole } from "../shared/math.ts";

export const llmForwardPassByTokensOnGPU = async (
  input: string[],
  model: Model,
  withActivations: boolean,
): Promise<{
  embeddings: Matrix;
  activations: Activations | null;
}> => {
  const hiddenDimensionsSize = model.counts.hiddenDimensions;
  const contextSize = input.length;

  const weightBuffers = loadWeightsIntoGpu(model);

  /** middle-state needed for backprop */
  const inputPositionToVocabPosition = input.map((token) => {
    return findTokenIndex(model.vocabulary, token);
  });

  const hiddenState = createMatrixBuffer(
    inputPositionToVocabPosition.length,
    model.counts.hiddenDimensions,
  );

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

  const uppedMlpBuffer = createMatrixBufferAndCopy(
    createMatrix(contextSize, hiddenDimensionsSize * model.counts.mlpMultiple),
  );

  const outMlpBuffer = createMatrixBufferAndCopy(
    createMatrix(contextSize, hiddenDimensionsSize),
  );

  const attentionUpdateBuffer = createMatrixBufferAndCopy(
    createMatrix(contextSize, hiddenDimensionsSize),
  );

  const attentionInputKBuffer = createMatrixBufferAndCopy(
    createMatrix(contextSize, hiddenDimensionsSize),
  );
  const attentionInputVBuffer = createMatrixBufferAndCopy(
    createMatrix(contextSize, hiddenDimensionsSize),
  );
  const attentionInputQBuffer = createMatrixBufferAndCopy(
    createMatrix(contextSize, hiddenDimensionsSize),
  );
  const attentionOutBuffer = createMatrixBufferAndCopy(
    createMatrix(contextSize, hiddenDimensionsSize),
  );
  const attentionRelevancyOutput = createMatrixBufferAndCopy(
    createMatrix(contextSize, hiddenDimensionsSize),
  );
  const matchingKeyProducts = createMatrixBufferAndCopy(
    createMatrix(contextSize, hiddenDimensionsSize),
  );

  const headDimensionsCount = divideToWhole(
    hiddenDimensionsSize,
    model.counts.attentionHeads,
  );

  const headDimensionsCountBuffer = gpuContext
    .createBuffer(d.u32, headDimensionsCount)
    .$usage("uniform");

  for (const transformerIndex in model.transformers) {
    const transformerBuffers = weightBuffers.transformers[transformerIndex]!;

    normalizeOnGpu(hiddenState);

    runSelfAttentionMechanismOnGPU(
      hiddenState,
      model.counts.attentionHeads,
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

  const unembeddedStateBuffer = createMatrixBufferAndCopy(
    createMatrix(contextSize, hiddenDimensionsSize),
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

  return {
    embeddings: await extractMatrixBuffer(unembeddedStateBuffer),
    activations: null,
  };
};
