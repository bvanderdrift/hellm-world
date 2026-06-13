import { divideToWhole } from "../shared/math.ts";
import {
  multiplyMatrices,
  sliceRows,
  createMatrix,
  type Matrix,
  getFlatIndex,
} from "../shared/matrices.ts";
import type { AttentionWeights } from "../model/model-types.ts";
import type {
  AttentionActivations,
  AttentionHeadActivations,
} from "../model/activations-types.ts";
import { softmax } from "../shared/softmax.ts";
import {
  matrixBufferDefinition,
  multiplyMatricesOnGPU,
  type MatrixBuffer,
} from "../shared/matrices-gpu.ts";
import type { AttentionGPUBuffers } from "../model/model-gpu-helpers.ts";
import tgpu from "typegpu";

export const runSelfAttentionMechanismOnGPU = (
  input: MatrixBuffer,
  headsCount: number,
  attentionWeights: AttentionGPUBuffers,
  inputQ: MatrixBuffer,
  inputK: MatrixBuffer,
  inputV: MatrixBuffer,
  out: MatrixBuffer,
  attentionUpdate: MatrixBuffer,
) => {
  const hiddenDimensionsCount = input.dimensions;

  multiplyMatricesOnGPU(input, attentionWeights.Q, inputQ);
  multiplyMatricesOnGPU(input, attentionWeights.K, inputK);
  multiplyMatricesOnGPU(input, attentionWeights.V, inputV);

  const headDimensionsCount = divideToWhole(hiddenDimensionsCount, headsCount);

  runSelfAttentionHeadOnGPU(
    inputQ,
    inputK,
    inputV,
    headsCount,
    headDimensionsCount,
    out,
  );

  multiplyMatricesOnGPU(out, attentionWeights.out, attentionUpdate);
};

export const runSelfAttentionHeadOnGPU = (
  inputQ: MatrixBuffer,
  inputK: MatrixBuffer,
  inputV: MatrixBuffer,
  headCount: number,
  headDimensionsCount: number,
  headsOut: MatrixBuffer,
) => {
  const attentionRelevancyOutput = new Array(headCount)
    .fill(0)
    .map((_) => createMatrix(inputQ.vectors, inputQ.vectors));
  const matchingKeyProducts = createMatrix(
    headCount * inputQ.vectors,
    inputQ.vectors,
  );

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

      const startIndexToSet = getFlatIndex(i, 0, inputQ.vectors);
      attentionRelevancyOutput[h]!.values.set(relevancyLogits, startIndexToSet);
      matchingKeyProducts[h]!.values.set(relevancy, startIndexToSet);
    }
  }

  for (let i = 0; i < headsOut.vectors; i++) {
    for (let j = 0; j < headsOut.dimensions; j++) {
      const h = Math.floor(j / headDimensionsCount);
      const outputIndex = getFlatIndex(i, j, headsOut.dimensions);

      for (let l = 0; l < i + 1; l++) {
        output.values[outputIndex]! +=
          matchingKeyProducts[h]!.values[getFlatIndex(i, l, output.vectors)]! *
          inputV.values[getFlatIndex(l, j, inputV.dimensions)]!;
      }
    }
  }
};
