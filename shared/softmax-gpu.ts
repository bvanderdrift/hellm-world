import tgpu, { d } from "typegpu";
import {
  getFlatIndexOnGPU,
  matrixBufferDefinition,
  type MatrixBuffer,
} from "./matrices-gpu.ts";
import { gpuContext } from "./gpu-context.ts";
import { builtin } from "typegpu/data";
import { exp, max } from "typegpu/std";

const softmaxParamsLayout = tgpu.bindGroupLayout({
  logits: {
    storage: matrixBufferDefinition,
    access: "readonly",
  },
  output: { storage: matrixBufferDefinition, access: "mutable" },
});

const WORKGROUP_SIZE = 1;

const softmaxGpuKernel = tgpu.computeFn({
  in: {
    localId: builtin.localInvocationId,
    groupId: builtin.workgroupId,
  },
  workgroupSize: [WORKGROUP_SIZE],
})((input) => {
  const headIndex = input.groupId.x;
  const vectorIndex = input.groupId.y;

  const logits = softmaxParamsLayout.$.logits;
  const contextLength = logits.vectors;
  const output = softmaxParamsLayout.$.output;

  const startIndexToSet = getFlatIndexOnGPU(
    vectorIndex,
    headIndex * contextLength,
    logits.dimensions,
  );

  let biggest = logits.values[startIndexToSet]!;

  for (
    let lookbackIndex = 1;
    lookbackIndex < vectorIndex + 1;
    lookbackIndex++
  ) {
    const logit = logits.values[startIndexToSet + lookbackIndex]!;
    biggest = max(biggest, logit);
  }

  let summed = d.f32(0);

  for (
    let lookbackIndex = 0;
    lookbackIndex < vectorIndex + 1;
    lookbackIndex++
  ) {
    const logit = logits.values[startIndexToSet + lookbackIndex]!;
    const safeLogit = logit - biggest;
    const exponated = exp(safeLogit);

    summed += exponated;
  }

  for (
    let lookbackIndex = 0;
    lookbackIndex < vectorIndex + 1;
    lookbackIndex++
  ) {
    const logit = logits.values[startIndexToSet + lookbackIndex]!;
    const safeLogit = logit - biggest;
    const exponated = exp(safeLogit);

    output.values[startIndexToSet + lookbackIndex] = exponated / summed;
  }
});

const softmaxGpuPipeline = gpuContext.createComputePipeline({
  compute: softmaxGpuKernel,
});

export const softmaxOnGpu = (
  logits: MatrixBuffer,
  output: MatrixBuffer,
  encoder: GPUCommandEncoder,
  headsCount: number = 1,
) => {
  const params = gpuContext.createBindGroup(softmaxParamsLayout, {
    logits: logits.buffer,
    output: output.buffer,
  });

  softmaxGpuPipeline
    .with(encoder)
    .with(params)
    .dispatchWorkgroups(headsCount, logits.vectors);
};
