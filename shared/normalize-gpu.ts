import tgpu, { d } from "typegpu";
import { matrixBufferDefinition, type MatrixBuffer } from "./matrices-gpu.ts";
import { gpuContext } from "./gpu-context.ts";
import { sqrt, workgroupBarrier } from "typegpu/std";
import { builtin } from "typegpu/data";

const normalizeParamsLayout = tgpu.bindGroupLayout({
  hiddenState: {
    storage: matrixBufferDefinition,
    access: "mutable",
  },
});

const WORKGROUP_SIZE = 64;
const sharedSummedValues = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));
const sharedSummedSquares = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));

const normalizeKernel = tgpu.computeFn({
  in: {
    localId: builtin.localInvocationId,
    groupId: builtin.workgroupId,
  },
  workgroupSize: [WORKGROUP_SIZE],
})((input) => {
  const t = input.localId.x;
  const vectorIndex = input.groupId.x;

  const hiddenState = normalizeParamsLayout.$.hiddenState;

  const offset = vectorIndex * hiddenState.dimensions;
  const length = hiddenState.dimensions;

  // standard deviation logic inline in kernel so we don't mess with the shifting window of the hidden state too much

  let partialSummedValues = d.f32(0);
  let partialSummedSquares = d.f32(0);

  for (
    let index = offset + t;
    index < offset + length;
    index += WORKGROUP_SIZE
  ) {
    partialSummedValues += hiddenState.values[index]!;
    partialSummedSquares += hiddenState.values[index]! ** 2;
  }

  sharedSummedValues.$[t] = partialSummedValues;
  sharedSummedSquares.$[t] = partialSummedSquares;

  workgroupBarrier();

  /**
   * Tree folding summation
   * With WORKGROUP SIZE = 8, will sum
   * - 4 sets of neighbours
   * - 2 sets of neighbours
   * - 1 set of neighbours
   *
   * So only 3 steps, which is log_2(8)
   */
  for (let stride = d.u32(WORKGROUP_SIZE / 2); stride > 0; stride /= 2) {
    if (t < stride) {
      sharedSummedValues.$[t] =
        sharedSummedValues.$[t] + sharedSummedValues.$[t + stride]!;

      sharedSummedSquares.$[t] =
        sharedSummedSquares.$[t] + sharedSummedSquares.$[t + stride]!;
    }
    workgroupBarrier();
  }

  const summedValues = sharedSummedValues.$[0]!;
  const summedSquares = sharedSummedSquares.$[0]!;

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

  for (let j = t; j < length; j += WORKGROUP_SIZE) {
    const valueIndex = offset + j;

    hiddenState.values[valueIndex] =
      (hiddenState.values[valueIndex]! - average) /
      (standardDeviation +
        // to prevent 0-divisions
        Number.EPSILON);
  }
});

const normalizeVectorRunner = gpuContext.createComputePipeline({
  compute: normalizeKernel,
});

export const normalizeOnGpu = (
  hiddenState: MatrixBuffer,
  encoder: GPUCommandEncoder,
) => {
  const bindGroup = gpuContext.createBindGroup(normalizeParamsLayout, {
    hiddenState: hiddenState.buffer,
  });

  normalizeVectorRunner
    .with(encoder)
    .with(bindGroup)
    .dispatchWorkgroups(hiddenState.vectors);
};
