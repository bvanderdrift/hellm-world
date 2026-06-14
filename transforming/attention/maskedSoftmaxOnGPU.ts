import tgpu from "typegpu";
import { builtin } from "typegpu/data";
import { softmaxGpuFn, softmaxParamsLayout } from "../../shared/softmax-gpu.ts";
import {
  getFlatIndexOnGPU,
  type MatrixBuffer,
} from "../../shared/matrices-gpu.ts";
import { gpuContext } from "../../shared/gpu-context.ts";

const WORKGROUP_SIZE = 1;

const maskedSoftmaxOnGpuKernel = tgpu.computeFn({
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

  const startIndexToSet = getFlatIndexOnGPU(
    vectorIndex,
    headIndex * contextLength,
    logits.dimensions,
  );

  softmaxGpuFn(startIndexToSet, vectorIndex + 1);
});

const maskedSoftmaxOnGpuPipeline = gpuContext.createComputePipeline({
  compute: maskedSoftmaxOnGpuKernel,
});

export const maskedSoftmaxOnGpu = (
  logits: MatrixBuffer,
  output: MatrixBuffer,
  headsCount: number = 1,
) => {
  const params = gpuContext.createBindGroup(softmaxParamsLayout, {
    logits: logits.buffer,
    output: output.buffer,
  });

  maskedSoftmaxOnGpuPipeline
    .with(params)
    .dispatchWorkgroups(headsCount, logits.vectors);
};
