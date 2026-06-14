import tgpu, { d, type TgpuBuffer, type UniformFlag } from "typegpu";
import { floor, mod, sqrt } from "typegpu/std";
import { gpuContext } from "../../shared/gpu-context.ts";
import {
  matrixBufferDefinition,
  getFlatIndexOnGPU,
  type MatrixBuffer,
} from "../../shared/matrices-gpu.ts";
import { MAX_CONTEXT } from "../../running/llm-shared.ts";

const calculateRelevancyParams = tgpu.bindGroupLayout({
  inputK: { storage: matrixBufferDefinition, access: "readonly" },
  inputQ: { storage: matrixBufferDefinition, access: "readonly" },
  attentionRelevancyOutput: {
    storage: matrixBufferDefinition,
    access: "mutable",
  },
  headDimensionsCount: { uniform: d.u32 },
});

const calculateRelevancyKernel = gpuContext.createGuardedComputePipeline(
  (vectorIndex: number, headIndex: number, lookbackIndex: number) => {
    "use gpu";

    const batchVectorIndex = mod(vectorIndex, MAX_CONTEXT);
    const batchCount = floor(vectorIndex / MAX_CONTEXT);

    if (lookbackIndex > batchVectorIndex) {
      // masked
      return;
    }

    const headDimensionsCount = calculateRelevancyParams.$.headDimensionsCount;
    const headOffset = headIndex * headDimensionsCount;
    const inputK = calculateRelevancyParams.$.inputK;
    const inputQ = calculateRelevancyParams.$.inputQ;
    const attentionRelevancyOutput =
      calculateRelevancyParams.$.attentionRelevancyOutput;

    let summed = d.f32(0);

    for (let k = d.u32(0); k < headDimensionsCount; k++) {
      summed +=
        inputQ.values[
          getFlatIndexOnGPU(vectorIndex, headOffset + k, inputQ.dimensions)
        ]! *
        inputK.values[
          getFlatIndexOnGPU(
            batchCount * MAX_CONTEXT + lookbackIndex,
            headOffset + k,
            inputK.dimensions,
          )
        ]!;
    }

    const headRelevancyOffset = getFlatIndexOnGPU(
      vectorIndex,
      headIndex * MAX_CONTEXT,
      attentionRelevancyOutput.dimensions,
    );

    attentionRelevancyOutput.values[headRelevancyOffset + lookbackIndex] =
      summed / sqrt(headDimensionsCount);
  },
);

export const calculateRelevancyOnGPU = (
  headCount: number,
  headDimensionsCount: TgpuBuffer<d.U32> & UniformFlag,
  inputK: MatrixBuffer,
  inputQ: MatrixBuffer,
  attentionRelevancyOutput: MatrixBuffer,
) => {
  const params = gpuContext.createBindGroup(calculateRelevancyParams, {
    headDimensionsCount,
    inputK: inputK.buffer,
    inputQ: inputQ.buffer,
    attentionRelevancyOutput: attentionRelevancyOutput.buffer,
  });

  calculateRelevancyKernel
    .with(params)
    .dispatchThreads(inputK.vectors, headCount, MAX_CONTEXT);
};
