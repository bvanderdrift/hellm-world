import tgpu, { d } from "typegpu";
import {
  getFlatIndexOnGPU,
  matrixBufferDefinition,
  type MatrixBuffer,
} from "./matrices/matrices-gpu.ts";
import { gpuContext } from "./gpu-context.ts";
import { builtin } from "typegpu/data";
import { exp, max } from "typegpu/std";

export const softmaxParamsLayout = tgpu.bindGroupLayout({
  logits: {
    storage: matrixBufferDefinition,
    access: "readonly",
  },
  output: { storage: matrixBufferDefinition, access: "mutable" },
});

export const softmaxGpuFn = tgpu.fn([d.u32, d.u32])((offset, length) => {
  const logits = softmaxParamsLayout.$.logits;

  const output = softmaxParamsLayout.$.output;

  let biggest = logits.values[offset]!;

  for (let lookbackIndex = 1; lookbackIndex < length; lookbackIndex++) {
    const logit = logits.values[offset + lookbackIndex]!;
    biggest = max(biggest, logit);
  }

  let summed = d.f32(0);

  for (let lookbackIndex = 0; lookbackIndex < length; lookbackIndex++) {
    const logit = logits.values[offset + lookbackIndex]!;
    const safeLogit = logit - biggest;
    const exponated = exp(safeLogit);

    summed += exponated;
  }

  for (let lookbackIndex = 0; lookbackIndex < length; lookbackIndex++) {
    const logit = logits.values[offset + lookbackIndex]!;
    const safeLogit = logit - biggest;
    const exponated = exp(safeLogit);

    output.values[offset + lookbackIndex] = exponated / summed;
  }
});

const WORKGROUP_SIZE = 1;

const softmaxGpuKernel = tgpu.computeFn({
  in: {
    localId: builtin.localInvocationId,
    groupId: builtin.workgroupId,
  },
  workgroupSize: [WORKGROUP_SIZE],
})((input) => {
  const vectorIndex = input.groupId.x;

  const logits = softmaxParamsLayout.$.logits;

  const startIndexToSet = getFlatIndexOnGPU(vectorIndex, 0, logits.dimensions);

  softmaxGpuFn(startIndexToSet, logits.dimensions);
});

const softmaxGpuPipeline = gpuContext.createComputePipeline({
  compute: softmaxGpuKernel,
});

export const softmaxOnGpu = (logits: MatrixBuffer, output: MatrixBuffer) => {
  const params = gpuContext.createBindGroup(softmaxParamsLayout, {
    logits: logits.buffer,
    output: output.buffer,
  });

  softmaxGpuPipeline.with(params).dispatchWorkgroups(logits.vectors);
};
