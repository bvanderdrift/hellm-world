import tgpu, { d, type TgpuBuffer, type StorageFlag } from "typegpu";
import type { WgslArray, F32 } from "typegpu/data";
import { mod, sqrt } from "typegpu/std";
import { gpuContext } from "../../shared/gpu-context.ts";
import {
  matrixBufferDefinition,
  getFlatIndexOnGPU,
  type MatrixBuffer,
} from "../../shared/matrices-gpu.ts";
import { getPositionEncodingOnGPU } from "../position-encoding-gpu.ts";
import { MAX_CONTEXT } from "../llm-shared.ts";

const prepareHiddenStateParamsLayout = tgpu.bindGroupLayout({
  inputTokenIndices: {
    storage: d.arrayOf(d.f32),
    access: "readonly",
  },
  embeddings: {
    storage: matrixBufferDefinition,
    access: "readonly",
  },
  hiddenState: {
    storage: matrixBufferDefinition,
    access: "mutable",
  },
});

const prepareHiddenStateKernel = (
  contextIndex: number,
  dimensionsIndex: number,
) => {
  "use gpu";

  const batchContextIndex = mod(contextIndex, MAX_CONTEXT);

  const vocabIndex =
    prepareHiddenStateParamsLayout.$.inputTokenIndices[contextIndex]!;

  const hiddenDimensionsSize =
    prepareHiddenStateParamsLayout.$.embeddings.dimensions;

  prepareHiddenStateParamsLayout.$.hiddenState.dimensions =
    hiddenDimensionsSize;
  prepareHiddenStateParamsLayout.$.hiddenState.vectors =
    prepareHiddenStateParamsLayout.$.inputTokenIndices.length;

  const positionalEncoding = getPositionEncodingOnGPU(
    batchContextIndex,
    dimensionsIndex,
    hiddenDimensionsSize,
  );

  const scaledEmbedding =
    sqrt(hiddenDimensionsSize) *
    prepareHiddenStateParamsLayout.$.embeddings.values[
      getFlatIndexOnGPU(vocabIndex, dimensionsIndex, hiddenDimensionsSize)
    ]!;

  prepareHiddenStateParamsLayout.$.hiddenState.values[
    getFlatIndexOnGPU(contextIndex, dimensionsIndex, hiddenDimensionsSize)
  ] = scaledEmbedding + positionalEncoding;
};

const prepareHiddenStateRunner = gpuContext.createGuardedComputePipeline(
  prepareHiddenStateKernel,
);

export const prepareHiddenState = (
  inputTokenIndices: TgpuBuffer<WgslArray<F32>> & StorageFlag,
  hiddenState: MatrixBuffer,
  embeddings: MatrixBuffer,
) => {
  const bindGroup = gpuContext.createBindGroup(prepareHiddenStateParamsLayout, {
    inputTokenIndices: inputTokenIndices.buffer,
    embeddings: embeddings.buffer,
    hiddenState: hiddenState.buffer,
  });

  prepareHiddenStateRunner
    .with(bindGroup)
    .dispatchThreads(hiddenState.vectors, hiddenState.dimensions);
};
