import { gpuContext } from "./gpu-context.ts";
import { singleMatrixParamsLayout, type MatrixBuffer } from "./matrices-gpu.ts";

const reluKernel = gpuContext.createGuardedComputePipeline((i: number) => {
  "use gpu";
  const m = singleMatrixParamsLayout.$.m;

  const currentValue = m.values[i]!;

  if (currentValue > 0) {
    m.values[i]! = currentValue;
  } else {
    m.values[i]! = 0;
  }
});

export const reluOnGpu = (matrix: MatrixBuffer) => {
  const params = gpuContext.createBindGroup(singleMatrixParamsLayout, {
    m: matrix.buffer,
  });

  reluKernel.with(params).dispatchThreads(matrix.vectors * matrix.dimensions);
};
