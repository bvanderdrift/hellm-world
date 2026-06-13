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
import { floor, sqrt } from "typegpu/std";

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
      const startIndexToSet = getFlatIndex(
        i,
        h * inputQ.vectors,
        attentionRelevancyOutput.dimensions,
      );

      for (let l = 0; l < i + 1; l++) {
        let summed = 0;

        for (let k = 0; k < headDimensionsCount; k++) {
          summed +=
            inputQ.values[getFlatIndex(i, k + offset, inputQ.dimensions)]! *
            inputK.values[getFlatIndex(l, k + offset, inputK.dimensions)]!;
        }

        attentionRelevancyOutput.values[startIndexToSet + l] =
          summed / Math.sqrt(headDimensionsCount);
      }

      const relevancy = softmax(
        attentionRelevancyOutput.values.slice(
          startIndexToSet,
          startIndexToSet + i + 1,
        ),
      );

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

const calculateRelevancyParams = tgpu.bindGroupLayout({
  inputK: { storage: matrixBufferDefinition, access: "readonly" },
  inputQ: { storage: matrixBufferDefinition, access: "readonly" },
  attentionRelevancyOutput: {
    storage: matrixBufferDefinition,
    access: "mutable",
  },
  headDimensionsCount: { uniform: d.u32 },
});

const calculateRelevancyKernel = gpuContext.createGuardedComputePipeline(
  (vectorIndex: number, headIndex: number, lookbackIndex: number) => {
    "use gpu";

    if (lookbackIndex > vectorIndex) {
      // masked
      return;
    }

    const headDimensionsCount = calculateRelevancyParams.$.headDimensionsCount;
    const headOffset = headIndex * headDimensionsCount;
    const inputK = calculateRelevancyParams.$.inputK;
    const inputQ = calculateRelevancyParams.$.inputQ;
    const attentionRelevancyOutput =
      calculateRelevancyParams.$.attentionRelevancyOutput;

    let summed = d.f32(0);

    for (let k = d.u32(0); k < headDimensionsCount; k++) {
      summed +=
        inputQ.values[
          getFlatIndexOnGPU(vectorIndex, headOffset + k, inputQ.dimensions)
        ]! *
        inputK.values[
          getFlatIndexOnGPU(lookbackIndex, headOffset + k, inputK.dimensions)
        ]!;
    }

    const headRelevancyOffset = getFlatIndexOnGPU(
      vectorIndex,
      headIndex * inputK.vectors,
      attentionRelevancyOutput.dimensions,
    );

    attentionRelevancyOutput.values[headRelevancyOffset + lookbackIndex] =
      summed / sqrt(headDimensionsCount);
  },
);

export const calculateRelevancyOnGPU = (
  headCount: number,
  headDimensionsCount: TgpuBuffer<d.U32> & UniformFlag,
  inputK: MatrixBuffer,
  inputQ: MatrixBuffer,
  attentionRelevancyOutput: MatrixBuffer,
) => {
  const params = gpuContext.createBindGroup(calculateRelevancyParams, {
    headDimensionsCount,
    inputK: inputK.buffer,
    inputQ: inputQ.buffer,
    attentionRelevancyOutput: attentionRelevancyOutput.buffer,
  });

  calculateRelevancyKernel
    .with(params)
    .dispatchThreads(inputK.vectors, headCount, inputK.vectors);
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
