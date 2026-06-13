import { divideToWhole } from "../shared/math.ts";
import { getFlatIndex } from "../shared/matrices.ts";
import { softmax } from "../shared/softmax.ts";
import {
  getFlatIndexOnGPU,
  matrixBufferDefinition,
  multiplyMatricesOnGPU,
  type MatrixBuffer,
} from "../shared/matrices-gpu.ts";
import type { AttentionGPUBuffers } from "../model/model-gpu-helpers.ts";
import tgpu, { d, type TgpuBuffer, type UniformFlag } from "typegpu";
import { gpuContext } from "../shared/gpu-context.ts";
import { floor } from "typegpu/std";

export const runSelfAttentionMechanismOnGPU = (
  input: MatrixBuffer,
  headsCount: number,
  headDimensionsCount: TgpuBuffer<d.U32> & UniformFlag,
  attentionWeights: AttentionGPUBuffers,
  inputQ: MatrixBuffer,
  inputK: MatrixBuffer,
  inputV: MatrixBuffer,
  attentionRelevancyOutput: MatrixBuffer,
  matchingKeyProducts: MatrixBuffer,
  output: MatrixBuffer,
  attentionUpdate: MatrixBuffer,
) => {
  multiplyMatricesOnGPU(input, attentionWeights.Q, inputQ);
  multiplyMatricesOnGPU(input, attentionWeights.K, inputK);
  multiplyMatricesOnGPU(input, attentionWeights.V, inputV);
  runSelfAttentionHeadsOnGPU(
    inputQ,
    inputK,
    inputV,
    headsCount,
    headDimensionsCount,
    attentionRelevancyOutput,
    matchingKeyProducts,
    output,
  );

  multiplyMatricesOnGPU(output, attentionWeights.out, attentionUpdate);
};

export const runSelfAttentionHeadsOnGPU = (
  inputQ: MatrixBuffer,
  inputK: MatrixBuffer,
  inputV: MatrixBuffer,
  headCount: number,
  headDimensionsCount: TgpuBuffer<d.U32> & UniformFlag,
  attentionRelevancyOutput: MatrixBuffer,
  matchingKeyProducts: MatrixBuffer,
  output: MatrixBuffer,
) => {
  for (let h = 0; h < headCount; h++) {
    const offset = h * headDimensionsCount;

    for (let i = 0; i < inputQ.vectors; i++) {
      const relevancyLogits = new Float32Array(i + 1).fill(0);

      for (let l = 0; l < relevancyLogits.length; l++) {
        let summed = 0;

        for (let k = 0; k < headDimensionsCount; k++) {
          summed +=
            inputQ.values[getFlatIndex(i, k + offset, inputQ.dimensions)]! *
            inputK.values[getFlatIndex(l, k + offset, inputK.dimensions)]!;
        }

        relevancyLogits[l]! = summed / Math.sqrt(headDimensionsCount);
      }

      const relevancy = softmax(relevancyLogits);

      const startIndexToSet = getFlatIndex(
        i,
        h * inputQ.vectors,
        attentionRelevancyOutput.dimensions,
      );
      attentionRelevancyOutput.values.set(relevancyLogits, startIndexToSet);
      matchingKeyProducts.values.set(relevancy, startIndexToSet);
    }
  }

  applyAttentionValuesOnGPU(
    headDimensionsCount,
    inputV,
    matchingKeyProducts,
    output,
  );
};

const applyWeightedValuesParams = tgpu.bindGroupLayout({
  inputV: { storage: matrixBufferDefinition, access: "readonly" },
  matchingKeyProducts: { storage: matrixBufferDefinition, access: "readonly" },
  output: { storage: matrixBufferDefinition, access: "mutable" },
  headDimensionsCount: { uniform: d.u32 },
});

const applyValuesKernel = gpuContext.createGuardedComputePipeline(
  (vectorIndex: number, dimensionIndex: number) => {
    "use gpu";

    const headDimensionsCount = applyWeightedValuesParams.$.headDimensionsCount;
    const inputV = applyWeightedValuesParams.$.inputV;
    const output = applyWeightedValuesParams.$.output;
    const matchingKeyProducts = applyWeightedValuesParams.$.matchingKeyProducts;

    const h = floor(dimensionIndex / headDimensionsCount);
    const offset = h * output.vectors;
    const outputIndex = getFlatIndexOnGPU(
      vectorIndex,
      dimensionIndex,
      output.dimensions,
    );

    let sum = d.f32(0);

    for (let lookback = d.u32(0); lookback < vectorIndex + 1; lookback++) {
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
          getFlatIndexOnGPU(lookback, dimensionIndex, inputV.dimensions)
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
