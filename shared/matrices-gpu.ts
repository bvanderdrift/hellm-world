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

/** 
 * Wrote this, benched it but not worth it for now.
 * 
 * On my machine:
 * 
 *   tiny    64x64 * 64x64
    baseline   mean=0.327ms median=0.327ms min=0.313ms stddev=0.009ms
    candidate  mean=1.146ms median=1.044ms min=0.794ms stddev=0.361ms
    -> candidate is 3.19x slower (median)

  small  128x128 * 128x128
    baseline   mean=2.882ms median=2.786ms min=2.771ms stddev=0.197ms
    candidate  mean=1.583ms median=1.423ms min=1.345ms stddev=0.429ms
    -> candidate is 1.96x faster (median)

  med    256x256 * 256x256
    baseline   mean=22.210ms median=21.941ms min=20.864ms stddev=1.086ms
    candidate  mean=5.278ms median=5.396ms min=4.394ms stddev=0.794ms
    -> candidate is 4.07x faster (median)

  large  512x512 * 512x512
    baseline   mean=198.152ms median=186.214ms min=167.922ms stddev=49.233ms
    candidate  mean=20.899ms median=19.797ms min=17.404ms stddev=3.114ms
    -> candidate is 9.41x faster (median)

  tall   512x128 * 128x512
    baseline   mean=45.227ms median=43.348ms min=42.001ms stddev=5.657ms
    candidate  mean=11.205ms median=8.837ms min=8.021ms stddev=5.379ms
    -> candidate is 4.91x faster (median)

 * So with 256 dimensionslity but with only 10 tokens input we don't cross the threshold
 * Since with a m x k @ k x n multiplication you can parallize m * n dotproduct which at 10 * 256 = 2.5k 
 * In the 64 x 64 test we were more than 3x and that's at 4.1k dotproducts
 * So not worth it
 * 
 * Only in the future if I truly would move entire transformers to the GPU not having to pass around memory arrays it would be better
 */
export const multiplyMatricesOnGPU = async (
  m1: number[][],
  m2: number[][],
): Promise<number[][]> => {
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

  const params = root.createBindGroup(paramsLayout, {
    input: inputBuffer,
    output: outputBuffer,
  });

  dotProductRunner.with(params).dispatchThreads(m1.length, m2Depth);

  const m3Flat = await outputBuffer.read();

  const m3: number[][] = new Array(m1.length).fill([]).map((_, i) => {
    return m3Flat.slice(i * m2Depth, (i + 1) * m2Depth);
  });

  return m3;
};
