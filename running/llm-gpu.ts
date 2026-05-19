import { d } from "typegpu";
import type {
  Activations,
  TransformerActivations,
} from "../model/activations-types.ts";
import { loadWeightsIntoGpu } from "../model/model-gpu-helpers.ts";
import {
  extractHiddenDimensionSize,
  findTokenIndex,
} from "../model/model-helpers.ts";
import type { Model } from "../model/model-types.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import {
  type MatrixBuffer,
  createMatrixBuffer,
  applyScalarToMatrixOnGPU,
  addMatricesOnGPU,
  extractMatrixBuffer,
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

export const llmForwardPassByTokensOnGPU = async (
  input: string[],
  model: Model,
  withActivations: boolean,
): Promise<{
  embeddings: Matrix;
  activations: Activations | null;
}> => {
  const hiddenDimensionsSize = extractHiddenDimensionSize(model);
  const contextSize = input.length;

  const weightBuffers = loadWeightsIntoGpu(model);

  /** middle-state needed for backprop */
  const inputPositionToVocabPosition = input.map((token) => {
    return findTokenIndex(model.vocabulary, token);
  });

  const startStateInCPU = createMatrix(input.length, hiddenDimensionsSize);

  for (let inputIndex = 0; inputIndex < input.length; inputIndex++) {
    const token = input[inputIndex]!;
    const vocabIndex = findTokenIndex(model.vocabulary, token);

    for (let j = 0; j < hiddenDimensionsSize; j++) {
      startStateInCPU.values[
        getFlatIndex(inputIndex, j, hiddenDimensionsSize)
      ] =
        model.embeddings.values[
          getFlatIndex(vocabIndex, j, hiddenDimensionsSize)
        ]!;
    }
  }

  const intermediateState: MatrixBuffer = createMatrixBuffer(startStateInCPU);

  await applyScalarToMatrixOnGPU(
    gpuContext.createUniform(d.f32, Math.sqrt(hiddenDimensionsSize)).buffer,
    intermediateState,
  );

  const positionalEncoding = getPositionEncodingOnGPU(
    contextSize,
    hiddenDimensionsSize,
  );

  await addMatricesOnGPU(intermediateState, positionalEncoding);

  const transformerActivations: TransformerActivations[] = [];

  const uppedMlpBuffer = createMatrixBuffer(
    createMatrix(contextSize, hiddenDimensionsSize * model.mlpMultiple),
  );

  const outMlpBuffer = createMatrixBuffer(
    createMatrix(contextSize, hiddenDimensionsSize),
  );

  for (const transformerIndex in model.transformers) {
    const transformer = model.transformers[transformerIndex]!;
    const transformerBuffers = weightBuffers.transformers[transformerIndex]!;

    const transformerInputState = await extractMatrixBuffer(intermediateState);

    const attentionInputEmbeddings = normalize(transformerInputState);

    const attentionActivations = runSelfAttentionMechanism(
      // Normalize input only, don't normalize the intermediateState iself
      // Reason: of this block outputs 0 for a feature, we keep x + 0 = x. But if we normalize the root variable we get norm(x) + 0 = norm(x) so a transform has still happened even if the block said not to
      attentionInputEmbeddings,
      model.headsCount,
      transformer.attention,
    );

    await addMatricesOnGPU(
      intermediateState,
      createMatrixBuffer(attentionActivations.output),
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
