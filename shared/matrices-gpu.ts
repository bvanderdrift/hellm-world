import tgpu, {
  d,
  type StorageFlag,
  type TgpuBuffer,
  type UniformFlag,
} from "typegpu";
import { gpuContext } from "./gpu-context.ts";
import { type Matrix } from "./matrices.ts";
import { builtin } from "typegpu/data";
import { add, mul, workgroupBarrier } from "typegpu/std";

const createMatrixBufferDefintionInstance = (
  vectors: number,
  dimensions: number,
) =>
  d.struct({
    vectors: d.u32,
    dimensions: d.u32,
    values: d.arrayOf(d.f32, vectors * dimensions),
  });

/** 0-size is special marker meaning 'dynamically set on creation' */
export const matrixBufferDefinition = createMatrixBufferDefintionInstance(0, 0);

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

export const getFlatIndexOnGPU = (i: number, j: number, dimensions: number) => {
  "use gpu";
  return i * dimensions + j;
};

const singleThreadDimensions = 4;
const singleThreadValueCount = singleThreadDimensions ** 2;
const outputTileWorkgroupSize = 8;

const singleTileDimensions = singleThreadDimensions * outputTileWorkgroupSize;
const singleTileValueCount = singleTileDimensions ** 2;

const tileMemoryM1 = tgpu.workgroupVar(d.arrayOf(d.f32, singleTileValueCount));

const tileMemoryM2 = tgpu.workgroupVar(d.arrayOf(d.f32, singleTileValueCount));

export const dotProductKernel = tgpu.computeFn({
  in: {
    workgroupIndex: builtin.workgroupId,
    innerTileIndex: builtin.localInvocationId,
  },
  workgroupSize: [outputTileWorkgroupSize, outputTileWorkgroupSize],
})((input) => {
  "use gpu";

  const m1 = multiplyMatrixParamsLayout.$.m1;
  const m2 = multiplyMatrixParamsLayout.$.m2;
  const mOut = multiplyMatrixParamsLayout.$.mOut;

  const innerStartX = input.innerTileIndex.x * singleThreadDimensions;
  const innerStartY = input.innerTileIndex.y * singleThreadDimensions;

  const outputStartX =
    input.workgroupIndex.x * singleTileDimensions + innerStartX;
  const outputStartY =
    input.workgroupIndex.y * singleTileDimensions + innerStartY;

  // Hardcoded to match 16 elements. Watch out though that this doesn't get unsynced
  let sums1 = d.vec4f();
  let sums2 = d.vec4f();
  let sums3 = d.vec4f();
  let sums4 = d.vec4f();

  for (
    let tileStride = 0;
    tileStride < m1.dimensions;
    tileStride += singleTileDimensions
  ) {
    for (let xIndex = 0; xIndex < singleThreadDimensions; xIndex++) {
      for (let yIndex = 0; yIndex < singleThreadDimensions; yIndex++) {
        const tileMemoryXIndex = innerStartX + xIndex;
        const tileMemoryYIndex = innerStartY + yIndex;

        const tileMemoryIndex = getFlatIndexOnGPU(
          tileMemoryXIndex,
          tileMemoryYIndex,
          singleTileDimensions,
        );

        const m1XIndex = outputStartX + xIndex;
        const m1YIndex = tileStride + tileMemoryYIndex;

        if (m1XIndex >= m1.vectors || m1YIndex >= m1.dimensions) {
          tileMemoryM1.$[tileMemoryIndex] = 0;
        } else {
          tileMemoryM1.$[tileMemoryIndex] =
            m1.values[getFlatIndexOnGPU(m1XIndex, m1YIndex, m1.dimensions)]!;
        }

        const m2XIndex = tileStride + tileMemoryXIndex;
        const m2YIndex = outputStartY + yIndex;

        if (m2XIndex >= m2.vectors || m2YIndex >= m2.dimensions) {
          tileMemoryM2.$[tileMemoryIndex] = 0;
        } else {
          tileMemoryM2.$[tileMemoryIndex] =
            m2.values[getFlatIndexOnGPU(m2XIndex, m2YIndex, m2.dimensions)]!;
        }
      }
    }

    workgroupBarrier();
    for (let k = 0; k < singleTileDimensions; k++) {
      const m1LocalBase = getFlatIndexOnGPU(
        innerStartX,
        k,
        singleTileDimensions,
      );
      // Get values at `k` for 4 different vectors
      const m1LocalValues = d.vec4f(
        tileMemoryM1.$[m1LocalBase]!,
        tileMemoryM1.$[m1LocalBase + 1 * singleTileDimensions]!,
        tileMemoryM1.$[m1LocalBase + 2 * singleTileDimensions]!,
        tileMemoryM1.$[m1LocalBase + 3 * singleTileDimensions]!,
      );

      const m2LocalBase = getFlatIndexOnGPU(
        k,
        innerStartY,
        singleTileDimensions,
      );
      // Get values at `k` for 4 different dimensions
      const m2LocalValue = d.vec4f(
        tileMemoryM2.$[m2LocalBase]!,
        tileMemoryM2.$[m2LocalBase + 1]!,
        tileMemoryM2.$[m2LocalBase + 2]!,
        tileMemoryM2.$[m2LocalBase + 3]!,
      );

      sums1 = add(sums1, mul(m1LocalValues.x, m2LocalValue));
      sums2 = add(sums2, mul(m1LocalValues.y, m2LocalValue));
      sums3 = add(sums3, mul(m1LocalValues.z, m2LocalValue));
      sums4 = add(sums4, mul(m1LocalValues.w, m2LocalValue));
    }

    workgroupBarrier();
  }

  const base1 = getFlatIndexOnGPU(outputStartX, outputStartY, mOut.dimensions);
  const base2 = getFlatIndexOnGPU(
    outputStartX + 1,
    outputStartY,
    mOut.dimensions,
  );
  const base3 = getFlatIndexOnGPU(
    outputStartX + 2,
    outputStartY,
    mOut.dimensions,
  );
  const base4 = getFlatIndexOnGPU(
    outputStartX + 3,
    outputStartY,
    mOut.dimensions,
  );

  mOut.values[base1] = sums1.x;
  mOut.values[base1 + 1] = sums1.y;
  mOut.values[base1 + 2] = sums1.z;
  mOut.values[base1 + 3] = sums1.w;

  mOut.values[base2] = sums2.x;
  mOut.values[base2 + 1] = sums2.y;
  mOut.values[base2 + 2] = sums2.z;
  mOut.values[base2 + 3] = sums2.w;

  mOut.values[base3] = sums3.x;
  mOut.values[base3 + 1] = sums3.y;
  mOut.values[base3 + 2] = sums3.z;
  mOut.values[base3 + 3] = sums3.w;

  mOut.values[base4] = sums4.x;
  mOut.values[base4 + 1] = sums4.y;
  mOut.values[base4 + 2] = sums4.z;
  mOut.values[base4 + 3] = sums4.w;
});

const dotProductRunner = gpuContext.createComputePipeline({
  compute: dotProductKernel,
});

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

  dotProductRunner
    .with(params)
    .dispatchWorkgroups(
      Math.ceil(m1.vectors / singleTileDimensions),
      Math.ceil(m2.dimensions / singleTileDimensions),
    );
};

export const createMatrixBuffer = (
  m: Matrix | Omit<Matrix, "values">,
): MatrixBuffer => {
  const init = "values" in m ? m : undefined;
  const buffer = gpuContext
    .createBuffer(
      createMatrixBufferDefintionInstance(m.vectors, m.dimensions),
      init,
    )
    .$usage("storage");

  if (!("values" in m)) {
    buffer.patch({
      vectors: m.vectors,
      dimensions: m.dimensions,
    });
  }

  return {
    buffer,
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

const applyScalarToMatrixKernel = gpuContext.createGuardedComputePipeline(
  (i: number, j: number) => {
    "use gpu";
    const scalar = applyScalarParamsLayout.$.scalar;
    const matrix = applyScalarParamsLayout.$.matrix;
    const flatIndex = getFlatIndexOnGPU(i, j, matrix.dimensions);

    matrix.values[flatIndex]! *= scalar;
  },
);

export const applyScalarToMatrixOnGPU = (
  scalar: TgpuBuffer<d.F32> & UniformFlag,
  matrix: MatrixBuffer,
) => {
  const params = gpuContext.createBindGroup(applyScalarParamsLayout, {
    scalar,
    matrix: matrix.buffer,
  });

  applyScalarToMatrixKernel
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

const addMatricesKernel = tgpu.computeFn({
  workgroupSize: [256],
  in: {
    globalId: builtin.globalInvocationId,
  },
})((input) => {
  const i = input.globalId.x;

  const m1 = addMatricesParamsLayout.$.m1WillMutate;

  if (i >= m1.vectors * m1.dimensions) {
    return;
  }

  const m2 = addMatricesParamsLayout.$.m2;

  m1.values[i]! = m1.values[i]! + m2.values[i]!;
});

const addMatricesRunner = gpuContext.createComputePipeline({
  compute: addMatricesKernel,
});

export const addMatricesOnGPU = (
  m1WillMutate: MatrixBuffer,
  m2: MatrixBuffer,
) => {
  const params = gpuContext.createBindGroup(addMatricesParamsLayout, {
    m1WillMutate: m1WillMutate.buffer,
    m2: m2.buffer,
  });

  addMatricesRunner
    .with(params)
    .dispatchWorkgroups(
      Math.ceil((m1WillMutate.vectors * m1WillMutate.dimensions) / 256),
    );
};

const addVectorAcrossMatrixKernel = tgpu.computeFn({
  in: {
    globalId: builtin.globalInvocationId,
  },
  workgroupSize: [16, 16],
})((input) => {
  const i = input.globalId.x;
  const j = input.globalId.y;

  const m1 = addMatricesParamsLayout.$.m1WillMutate;
  const m2 = addMatricesParamsLayout.$.m2;

  if (i >= m1.vectors || j >= m1.dimensions) {
    return;
  }

  const flatIndex = getFlatIndexOnGPU(i, j, m1.dimensions);

  m1.values[flatIndex]! = m1.values[flatIndex]! + m2.values[j]!;
});

const addVectorAcrossMatrixRunner = gpuContext.createComputePipeline({
  compute: addVectorAcrossMatrixKernel,
});

export const addVectorAcrossMatrixOnGPU = (
  m1WillMutate: MatrixBuffer,
  vector: MatrixBuffer,
) => {
  const params = gpuContext.createBindGroup(addMatricesParamsLayout, {
    m1WillMutate: m1WillMutate.buffer,
    m2: vector.buffer,
  });

  addVectorAcrossMatrixRunner
    .with(params)
    .dispatchWorkgroups(
      Math.ceil(m1WillMutate.vectors / 16),
      Math.ceil(m1WillMutate.dimensions / 16),
    );
};

export const singleMatrixParamsLayout = tgpu.bindGroupLayout({
  m: {
    storage: matrixBufferDefinition,
    access: "mutable",
  },
});
