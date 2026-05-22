import tgpu, {
  d,
  type StorageFlag,
  type TgpuBuffer,
  type UniformFlag,
} from "typegpu";
import { gpuContext } from "./gpu-context.ts";
import type { Matrix } from "./matrices.ts";

const SINGLE_DIMENSION_SIZE = 4_000;

export const matrixBufferDefinition = d.struct({
  values: d.arrayOf(d.f32, SINGLE_DIMENSION_SIZE ** 2),
  vectors: d.u32,
  dimensions: d.u32,
});

export type MatrixBuffer = {
  buffer: TgpuBuffer<typeof matrixBufferDefinition> & StorageFlag;
  vectors: number;
  dimensions: number;
};

const multiplyMatrixParamsLayout = tgpu.bindGroupLayout({
  m1: {
    storage: matrixBufferDefinition,
    access: "readonly",
  },
  m2: {
    storage: matrixBufferDefinition,
    access: "readonly",
  },
  mOut: {
    storage: matrixBufferDefinition,
    access: "mutable",
  },
});

const getFlatIndex = (i: number, j: number, dimensions: number) => {
  "use gpu";
  return i * dimensions + j;
};

const dotProductOnGPU = (i: number, j: number) => {
  "use gpu";

  const m1 = multiplyMatrixParamsLayout.$.m1;
  const m2 = multiplyMatrixParamsLayout.$.m2;
  const mOut = multiplyMatrixParamsLayout.$.mOut;

  let summed = d.f32(0);

  for (let k = 0; k < m1.dimensions; k++) {
    summed +=
      m1.values[getFlatIndex(i, k, m1.dimensions)]! *
      m2.values[getFlatIndex(k, j, m2.dimensions)]!;
  }

  mOut.vectors = m1.vectors;
  mOut.dimensions = m2.dimensions;

  mOut.values[getFlatIndex(i, j, mOut.dimensions)]! = summed;
};

const dotProductRunner =
  gpuContext.createGuardedComputePipeline(dotProductOnGPU);

export const multiplyMatricesOnGPU = (
  m1: MatrixBuffer,
  m2: MatrixBuffer,
  mOut: MatrixBuffer,
) => {
  const params = gpuContext.createBindGroup(multiplyMatrixParamsLayout, {
    m1: m1.buffer,
    m2: m2.buffer,
    mOut: mOut.buffer,
  });

  dotProductRunner.with(params).dispatchThreads(m1.vectors, m2.dimensions);
};

export const createMatrixBuffer = (m: Matrix): MatrixBuffer => {
  const embeddingsBuffer = gpuContext
    .createBuffer(matrixBufferDefinition, m)
    .$usage("storage");

  return {
    buffer: embeddingsBuffer,
    vectors: m.vectors,
    dimensions: m.dimensions,
  };
};

export const extractMatrixBuffer = async (m: MatrixBuffer): Promise<Matrix> => {
  const onCpu = await m.buffer.read();

  return {
    ...onCpu,
    values: new Float32Array(onCpu.values),
  };
};

const applyScalarParamsLayout = tgpu.bindGroupLayout({
  scalar: {
    uniform: d.f32,
  },
  matrix: {
    storage: matrixBufferDefinition,
    access: "mutable",
  },
});

export const applyScalarToMatrixOnGPU = (
  scalar: TgpuBuffer<d.F32> & UniformFlag,
  matrix: MatrixBuffer,
) => {
  const params = gpuContext.createBindGroup(applyScalarParamsLayout, {
    scalar,
    matrix: matrix.buffer,
  });

  gpuContext
    .createGuardedComputePipeline((i: number, j: number) => {
      "use gpu";
      const scalar = applyScalarParamsLayout.$.scalar;
      const matrix = applyScalarParamsLayout.$.matrix;
      const flatIndex = getFlatIndex(i, j, matrix.dimensions);

      matrix.values[flatIndex]! *= scalar;
    })
    .with(params)
    .dispatchThreads(matrix.vectors, matrix.dimensions);
};

const addMatricesParamsLayout = tgpu.bindGroupLayout({
  m1WillMutate: {
    storage: matrixBufferDefinition,
    access: "mutable",
  },
  m2: {
    storage: matrixBufferDefinition,
    access: "readonly",
  },
});

export const addMatricesOnGPU = (
  m1WillMutate: MatrixBuffer,
  m2: MatrixBuffer,
) => {
  const params = gpuContext.createBindGroup(addMatricesParamsLayout, {
    m1WillMutate: m1WillMutate.buffer,
    m2: m2.buffer,
  });

  gpuContext
    .createGuardedComputePipeline((i: number) => {
      "use gpu";
      const m1 = addMatricesParamsLayout.$.m1WillMutate;
      const m2 = addMatricesParamsLayout.$.m2;

      m1.values[i]! = m1.values[i]! + m2.values[i]!;
    })
    .with(params)
    .dispatchThreads(m1WillMutate.vectors * m1WillMutate.dimensions);
};

export const addVectorAcrossMatrixOnGPU = (
  m1WillMutate: MatrixBuffer,
  vector: MatrixBuffer,
) => {
  const params = gpuContext.createBindGroup(addMatricesParamsLayout, {
    m1WillMutate: m1WillMutate.buffer,
    m2: vector.buffer,
  });

  gpuContext
    .createGuardedComputePipeline((i: number, j: number) => {
      "use gpu";
      const m1 = addMatricesParamsLayout.$.m1WillMutate;
      const m2 = addMatricesParamsLayout.$.m2;
      const flatIndex = getFlatIndex(i, j, m1.dimensions);

      m1.values[flatIndex]! = m1.values[flatIndex]! + m2.values[j]!;
    })
    .with(params)
    .dispatchThreads(m1WillMutate.vectors, m1WillMutate.dimensions);
};

export const singleMatrixParamsLayout = tgpu.bindGroupLayout({
  m: {
    storage: matrixBufferDefinition,
    access: "mutable",
  },
});
