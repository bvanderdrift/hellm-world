import tgpu, { d } from "typegpu";
import { matrixBufferDefinition, type MatrixBuffer } from "./matrices-gpu.ts";
import { gpuContext } from "./gpu-context.ts";
import { sqrt } from "typegpu/std";

const normalizeParamsLayout = tgpu.bindGroupLayout({
  hiddenState: {
    storage: matrixBufferDefinition,
    access: "mutable",
  },
});

const normalizeVectorKernel = (vectorIndex: number) => {
  "use gpu";

  const hiddenState = normalizeParamsLayout.$.hiddenState;

  const offset = vectorIndex * hiddenState.dimensions;
  const length = hiddenState.dimensions;

  // standard deviation logic inline in kernel so we don't mess with the shifting window of the hidden state too much

  let summedValues = d.f32(0);
  let summedSquares = d.f32(0);

  for (let index = offset; index < offset + length; index++) {
    summedValues += hiddenState.values[index]!;
    summedSquares += hiddenState.values[index]! ** 2;
  }

  const average = summedValues / d.f32(length);
  /**
   * summedSquareDeltas = sum((x_i - avg)^2)
   * = sum(x_i^2 - 2*x_i*avg + avg^2)
   * = sum(x_i^2) - 2*avg*sum(x_i) - n * avg^2
   *
   * this allows us to do one loop instead of two
   */
  const summedSquareDeltas =
    summedSquares - 2 * average * summedValues + length * average ** 2;

  const averageSquareDeltas = summedSquareDeltas / d.f32(length);

  const standardDeviation = sqrt(averageSquareDeltas);

  for (let j = 0; j < length; j++) {
    const valueIndex = offset + j;

    hiddenState.values[valueIndex] =
      (hiddenState.values[valueIndex]! - average) /
      (standardDeviation +
        // to prevent 0-divisions
        Number.EPSILON);
  }
};

const normalizeVectorRunner = gpuContext.createGuardedComputePipeline(
  normalizeVectorKernel,
);

export const normalizeOnGpu = (hiddenState: MatrixBuffer) => {
  const bindGroup = gpuContext.createBindGroup(normalizeParamsLayout, {
    hiddenState: hiddenState.buffer,
  });

  normalizeVectorRunner.with(bindGroup).dispatchThreads(hiddenState.vectors);
};
