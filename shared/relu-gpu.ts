import tgpu from "typegpu";
import { gpuContext } from "./gpu-context.ts";
import { singleMatrixParamsLayout, type MatrixBuffer } from "./matrices-gpu.ts";
import { builtin } from "typegpu/data";

const reluKernel = tgpu.computeFn({
  in: {
    localId: builtin.localInvocationId,
    numWorkgroups: builtin.numWorkgroups,
  },
  workgroupSize: [16, 16],
})((input) => {
  "use gpu";

  const i1 = input.localId.x;
  const i2 = input.localId.y;

  const i = i1 * input.numWorkgroups.x + i2;

  const m = singleMatrixParamsLayout.$.m;

  if (i >= m.vectors * m.dimensions) {
    return;
  }

  const currentValue = m.values[i]!;

  if (currentValue > 0) {
    m.values[i]! = currentValue;
  } else {
    m.values[i]! = 0;
  }
});

const reluPipeline = gpuContext.createComputePipeline({
  compute: reluKernel,
});

export const reluOnGpu = (matrix: MatrixBuffer) => {
  const params = gpuContext.createBindGroup(singleMatrixParamsLayout, {
    m: matrix.buffer,
  });

  reluPipeline
    .with(params)
    .dispatchWorkgroups(
      Math.ceil(matrix.vectors / 16),
      Math.ceil(matrix.dimensions / 16),
    );
};
