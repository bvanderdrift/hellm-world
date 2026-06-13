import { getFlatIndex } from "../../shared/matrices.ts";
import { softmax } from "../../shared/softmax.ts";
import {
  multiplyMatricesOnGPU,
  type MatrixBuffer,
} from "../../shared/matrices-gpu.ts";
import type { AttentionGPUBuffers } from "../../model/model-gpu-helpers.ts";
import { d, type TgpuBuffer, type UniformFlag } from "typegpu";
import { applyAttentionValuesOnGPU } from "./applyAttentionValuesOnGPU.ts";

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
