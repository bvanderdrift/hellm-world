import tgpu, { d, type TgpuBuffer, type UniformFlag } from "typegpu";
import { floor, mod } from "typegpu/std";
import { gpuContext } from "../../shared/gpu-context.ts";
import {
  matrixBufferDefinition,
  getFlatIndexOnGPU,
  type MatrixBuffer,
} from "../../shared/matrices-gpu.ts";
import { MAX_CONTEXT } from "../../running/llm-shared.ts";

const applyWeightedValuesParams = tgpu.bindGroupLayout({
  inputV: { storage: matrixBufferDefinition, access: "readonly" },
  matchingKeyProducts: { storage: matrixBufferDefinition, access: "readonly" },
  output: { storage: matrixBufferDefinition, access: "mutable" },
  headDimensionsCount: { uniform: d.u32 },
});

const applyValuesKernel = gpuContext.createGuardedComputePipeline(
  (vectorIndex: number, dimensionIndex: number) => {
    "use gpu";

    const batchVectorIndex = mod(vectorIndex, MAX_CONTEXT);
    const batchCount = floor(vectorIndex / MAX_CONTEXT);

    const headDimensionsCount = applyWeightedValuesParams.$.headDimensionsCount;
    const inputV = applyWeightedValuesParams.$.inputV;
    const output = applyWeightedValuesParams.$.output;
    const matchingKeyProducts = applyWeightedValuesParams.$.matchingKeyProducts;

    const h = floor(dimensionIndex / headDimensionsCount);
    const offset = h * MAX_CONTEXT;
    const outputIndex = getFlatIndexOnGPU(
      vectorIndex,
      dimensionIndex,
      output.dimensions,
    );

    let sum = d.f32(0);

    for (let lookback = d.u32(0); lookback < batchVectorIndex + 1; lookback++) {
      const lookbackTokenWeight =
        matchingKeyProducts.values[
          getFlatIndexOnGPU(
            vectorIndex,
            offset + lookback,
            matchingKeyProducts.dimensions,
          )
        ]!;

      const lookbackTokenValue =
        inputV.values[
          getFlatIndexOnGPU(
            batchCount * MAX_CONTEXT + lookback,
            dimensionIndex,
            inputV.dimensions,
          )
        ]!;

      sum += lookbackTokenWeight * lookbackTokenValue;
    }

    output.values[outputIndex] = sum;
  },
);

export const applyAttentionValuesOnGPU = (
  headDimensionsCount: TgpuBuffer<d.U32> & UniformFlag,
  inputV: MatrixBuffer,
  matchingKeyProducts: MatrixBuffer,
  output: MatrixBuffer,
) => {
  const params = gpuContext.createBindGroup(applyWeightedValuesParams, {
    headDimensionsCount,
    inputV: inputV.buffer,
    matchingKeyProducts: matchingKeyProducts.buffer,
    output: output.buffer,
  });

  applyValuesKernel
    .with(params)
    .dispatchThreads(output.vectors, output.dimensions);
};
