import tgpu, { d, type TgpuBuffer, type UniformFlag } from "typegpu";
import {
  getFlatIndexOnGPU,
  matrixBufferDefinition,
  type MatrixBuffer,
} from "../../shared/matrices-gpu.ts";
import { gpuContext } from "../../shared/gpu-context.ts";
import { builtin } from "typegpu/data";
import { exp, max } from "typegpu/std";

const softmaxAttentionHeadsParams = tgpu.bindGroupLayout({
  contextLength: {
    uniform: d.u32,
  },
  attentionRelevancyOutput: {
    storage: matrixBufferDefinition,
    access: "readonly",
  },
  matchingKeyProducts: { storage: matrixBufferDefinition, access: "mutable" },
});

const WORKGROUP_SIZE = 1;

const softmaxAttentionHeadsKernel = tgpu.computeFn({
  in: {
    localId: builtin.localInvocationId,
    groupId: builtin.workgroupId,
  },
  workgroupSize: [WORKGROUP_SIZE],
})((input) => {
  const headIndex = input.groupId.x;
  const vectorIndex = input.groupId.y;

  const contextLength = softmaxAttentionHeadsParams.$.contextLength;
  const attentionRelevancyOutput =
    softmaxAttentionHeadsParams.$.attentionRelevancyOutput;
  const matchingKeyProducts = softmaxAttentionHeadsParams.$.matchingKeyProducts;

  const startIndexToSet = getFlatIndexOnGPU(
    vectorIndex,
    headIndex * contextLength,
    attentionRelevancyOutput.dimensions,
  );

  let biggest = attentionRelevancyOutput.values[startIndexToSet]!;

  for (
    let lookbackIndex = 1;
    lookbackIndex < vectorIndex + 1;
    lookbackIndex++
  ) {
    const logit =
      attentionRelevancyOutput.values[startIndexToSet + lookbackIndex]!;
    biggest = max(biggest, logit);
  }

  let summed = d.f32(0);

  for (
    let lookbackIndex = 0;
    lookbackIndex < vectorIndex + 1;
    lookbackIndex++
  ) {
    const logit =
      attentionRelevancyOutput.values[startIndexToSet + lookbackIndex]!;
    const safeLogit = logit - biggest;
    const exponated = exp(safeLogit);

    summed += exponated;
  }

  for (
    let lookbackIndex = 0;
    lookbackIndex < vectorIndex + 1;
    lookbackIndex++
  ) {
    const logit =
      attentionRelevancyOutput.values[startIndexToSet + lookbackIndex]!;
    const safeLogit = logit - biggest;
    const exponated = exp(safeLogit);

    matchingKeyProducts.values[startIndexToSet + lookbackIndex] =
      exponated / summed;
  }
});

const softmaxAttentionHeadsPipeline = gpuContext.createComputePipeline({
  compute: softmaxAttentionHeadsKernel,
});

export const softmaxAttentionHeadsOnGPU = (
  headsCount: number,
  contextLength: TgpuBuffer<d.U32> & UniformFlag,
  attentionRelevancyOutput: MatrixBuffer,
  matchingKeyProducts: MatrixBuffer,
  encoder: GPUCommandEncoder,
) => {
  const params = gpuContext.createBindGroup(softmaxAttentionHeadsParams, {
    contextLength,
    attentionRelevancyOutput: attentionRelevancyOutput.buffer,
    matchingKeyProducts: matchingKeyProducts.buffer,
  });

  softmaxAttentionHeadsPipeline
    .with(encoder)
    .with(params)
    .dispatchWorkgroups(headsCount, attentionRelevancyOutput.vectors);
};
