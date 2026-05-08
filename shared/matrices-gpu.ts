import tgpu, { d } from "typegpu";
import { setupGlobals } from "bun-webgpu";
setupGlobals();

const root = await tgpu.init();

const SINGLE_DIMENSION_SIZE = 4_000;

const inputType = d.struct({
  m1: d.arrayOf(d.f32, SINGLE_DIMENSION_SIZE * SINGLE_DIMENSION_SIZE),
  m2: d.arrayOf(d.f32, SINGLE_DIMENSION_SIZE * SINGLE_DIMENSION_SIZE),
  dimensions: d.struct({
    m1Vectors: d.i32,
    m1Depth: d.i32,
    m2Vectors: d.i32,
    m2Depth: d.i32,
  }),
});

const outputType = d.arrayOf(d.f32);

const zeroMatrix = new Array(
  SINGLE_DIMENSION_SIZE * SINGLE_DIMENSION_SIZE,
).fill(0);

const paramsLayout = tgpu.bindGroupLayout({
  input: {
    storage: inputType,
    /** @default 'readonly' */
    access: "readonly",
  },
  output: {
    storage: outputType,
    /** @default 'readonly' */
    access: "mutable",
  },
});

const inputBuffer = root
  .createBuffer(inputType, {
    dimensions: {
      m1Vectors: 0,
      m1Depth: 0,
      m2Vectors: 0,
      m2Depth: 0,
    },
    m1: zeroMatrix,
    m2: zeroMatrix,
  })
  .$usage("storage");

const outputBuffers = new Array(7).fill(0).map((_, i) => {
  // 5 -> 32
  // 5 + 7 -> 4096
  const size = 2 ** (i + 5);
  return root
    .createBuffer(outputType(size * size), new Array(size * size).fill(0))
    .$usage("storage");
});

const dotProductOnGPU = (i: number, j: number) => {
  "use gpu";

  const m1 = paramsLayout.$.input.m1;
  const m2 = paramsLayout.$.input.m2;

  const m1Depth = paramsLayout.$.input.dimensions.m1Depth;
  const m2Depth = paramsLayout.$.input.dimensions.m2Depth;

  let summed = d.f32(0);

  for (let k = 0; k < m1Depth; k++) {
    summed += m1[i * m1Depth + k]! * m2[k * m2Depth + j]!;
  }

  paramsLayout.$.output[i * m2Depth + j]! = summed;
};

const dotProductRunner = root.createGuardedComputePipeline(dotProductOnGPU);

export const multiplyMatricesOnGPU = async (
  m1: number[][],
  m2: number[][],
): Promise<{ durationGpu: number; m3: number[][] }> => {
  const m1Depth = m1[0]!.length;
  const m2Depth = m2[0]!.length;

  const biggestSizePower = Math.ceil(
    Math.log2(Math.max(m1.length, m1Depth, m2Depth)),
  );

  const bufferSizePower = Math.max(Math.min(biggestSizePower, 12), 5);

  const outputBuffer = outputBuffers[bufferSizePower - 5]!;

  inputBuffer.patch({
    m1: new Float32Array(m1.flat()),
    m2: new Float32Array(m2.flat()),
    dimensions: {
      m1Vectors: m1.length,
      m1Depth,
      m2Vectors: m2.length,
      m2Depth,
    },
  });

  const startGpu = performance.now();
  const params = root.createBindGroup(paramsLayout, {
    input: inputBuffer,
    output: outputBuffer,
  });

  dotProductRunner.with(params).dispatchThreads(m1.length, m2Depth);
  const durationGpu = performance.now() - startGpu;

  const m3Flat = await outputBuffer.read();

  const m3: number[][] = new Array(m1.length).fill([]).map((_, i) => {
    return m3Flat.slice(i * m2Depth, (i + 1) * m2Depth);
  });

  return { m3, durationGpu };
};
