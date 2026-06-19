import tgpu from "typegpu";
import { gpuContext } from "./gpu-context.ts";
import {
  getFlatIndexOnGPU,
  singleMatrixParamsLayout,
  type MatrixBuffer,
} from "./matrices-gpu.ts";
import { builtin } from "typegpu/data";
import { getFlatIndex } from "./matrices.ts";

const reluKernel = tgpu.computeFn({
  in: {
    globalId: builtin.globalInvocationId,
    localId: builtin.localInvocationId,
    numWorkgroups: builtin.numWorkgroups,
  },
  workgroupSize: [16, 16],
})((input) => {
  "use gpu";

  const m = singleMatrixParamsLayout.$.m;

  const i = getFlatIndexOnGPU(input.globalId.x, input.globalId.y, m.dimensions);

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
