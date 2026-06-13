import {
  multiplyMatricesOnGPU,
  type MatrixBuffer,
} from "../../shared/matrices-gpu.ts";
import type { AttentionGPUBuffers } from "../../model/model-gpu-helpers.ts";
import { d, type TgpuBuffer, type UniformFlag } from "typegpu";
import { applyAttentionValuesOnGPU } from "./applyAttentionValuesOnGPU.ts";
import { calculateRelevancyOnGPU } from "./calculateRelevancyOnGPU.ts";
import { softmaxAttentionHeadsOnGPU } from "./softmaxAttentionHeadsOnGPU.ts";

export const runSelfAttentionMechanismOnGPU = (
  input: MatrixBuffer,
  headsCount: number,
  contextLength: TgpuBuffer<d.U32> & UniformFlag,
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
    contextLength,
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
  contextLength: TgpuBuffer<d.U32> & UniformFlag,
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

  softmaxAttentionHeadsOnGPU(
    headCount,
    contextLength,
    attentionRelevancyOutput,
    matchingKeyProducts,
  );

  applyAttentionValuesOnGPU(
    headDimensionsCount,
    inputV,
    matchingKeyProducts,
    output,
  );
};
