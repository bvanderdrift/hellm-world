import tgpu, { d, type StorageFlag, type TgpuBuffer } from "typegpu";
import type {
  Activations,
  TransformerActivations,
} from "../model/activations-types.ts";
import { loadWeightsIntoGpu } from "../model/model-gpu-helpers.ts";
import { findTokenIndex } from "../model/model-helpers.ts";
import type { Model } from "../model/model-types.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import {
  type MatrixBuffer,
  createMatrixBufferAndCopy,
  applyScalarToMatrixOnGPU,
  addMatricesOnGPU,
  extractMatrixBuffer,
  createMatrixBuffer,
  matrixBufferDefinition,
  getFlatIndexOnGPU,
} from "../shared/matrices-gpu.ts";
import {
  createMatrix,
  getFlatIndex,
  multiplyMatrices,
  normalize,
  type Matrix,
} from "../shared/matrices.ts";
import { runSelfAttentionMechanism } from "../transforming/attention.ts";
import { getMultilayerPerceptronActivationsOnGPU } from "../transforming/mlp-gpu.ts";
import { getPositionEncodingOnGPU } from "./position-encoding-gpu.ts";
import type { F32, WgslArray } from "typegpu/data";
import { sqrt } from "typegpu/std";
import { prepareHiddenState } from "./gpu-logic/prepareHiddenStateGPU.ts";

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

  for (const transformerIndex in model.transformers) {
    const transformer = model.transformers[transformerIndex]!;
    const transformerBuffers = weightBuffers.transformers[transformerIndex]!;

    const transformerInputState = await extractMatrixBuffer(hiddenState);

    const attentionInputEmbeddings = normalize(transformerInputState);

    const attentionActivations = runSelfAttentionMechanism(
      // Normalize input only, don't normalize the intermediateState iself
      // Reason: of this block outputs 0 for a feature, we keep x + 0 = x. But if we normalize the root variable we get norm(x) + 0 = norm(x) so a transform has still happened even if the block said not to
      attentionInputEmbeddings,
      model.counts.attentionHeads,
      transformer.attention,
    );

    await addMatricesOnGPU(
      intermediateState,
      createMatrixBufferAndCopy(attentionActivations.output),
    );

    const embeddingsWithAttentionUpdates =
      await extractMatrixBuffer(intermediateState);

    const mlpInputEmbeddings = normalize(embeddingsWithAttentionUpdates);

    intermediateState.buffer.patch({
      values: mlpInputEmbeddings.values,
    });

    getMultilayerPerceptronActivationsOnGPU(
      // Normalize input only, don't normalize the intermediateState iself
      // Reason: of this block outputs 0 for a feature, we keep x + 0 = x. But if we normalize the root variable we get norm(x) + 0 = norm(x) so a transform has still happened even if the block said not to
      intermediateState,
      uppedMlpBuffer,
      outMlpBuffer,
      transformerBuffers.multilayerPerceptron,
    );

    // Apply updated knowledge
    addMatricesOnGPU(intermediateState, outMlpBuffer);

    transformerActivations.push({
      attention: null as unknown,
      mlp: null as unknown,
      transformerInput: null as unknown,
    } as any);
  }

  const postTransformerState = await extractMatrixBuffer(intermediateState);

  const normalizedTransformersOutput = normalize(postTransformerState);

  const unembeddedState = multiplyMatrices(
    normalizedTransformersOutput,
    model.unembeddings,
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
    embeddings: unembeddedState,
    activations: withActivations
      ? ({
          inputPositionToVocabPosition,
          tokensToPosition: null as any,
          positionToTransformers: positionalEncoding,
          transformerActivations,
          transformersToNormalizer: null as any,
          normalizerToUnembeddings: normalizedTransformersOutput,
          unembeddingsOutputLogits: unembeddedState,
        } as any)
      : null,
  };
};
