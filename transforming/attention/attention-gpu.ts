import { getFlatIndex } from "../../shared/matrices.ts";
import { softmax } from "../../shared/softmax.ts";
import {
  multiplyMatricesOnGPU,
  type MatrixBuffer,
} from "../../shared/matrices-gpu.ts";
import type { AttentionGPUBuffers } from "../../model/model-gpu-helpers.ts";
import { d, type TgpuBuffer, type UniformFlag } from "typegpu";
import { applyAttentionValuesOnGPU } from "./applyAttentionValuesOnGPU.ts";
import { calculateRelevancyOnGPU } from "./calculateRelevancyOnGPU.ts";

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
  calculateRelevancyOnGPU(
    headCount,
    headDimensionsCount,
    inputK,
    inputQ,
    attentionRelevancyOutput,
  );

  for (let h = 0; h < headCount; h++) {
    for (let i = 0; i < inputQ.vectors; i++) {
      const startIndexToSet = getFlatIndex(
        i,
        h * inputQ.vectors,
        attentionRelevancyOutput.dimensions,
      );

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
