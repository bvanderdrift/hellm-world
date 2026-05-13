import { divideToWhole } from "../shared/math.ts";
import {
  multiplyMatrices,
  validateSize,
  sliceRows,
  createMatrix,
  createVector,
} from "../shared/matrices.ts";
import type { AttentionWeights } from "../model/model-types.ts";
import type {
  AttentionActivations,
  AttentionHeadActivations,
} from "../model/activations-types.ts";
import {
  matrixBufferDefinition,
  multiplyMatricesOnGPU,
  type MatrixBuffer,
} from "../shared/matrices-gpu.ts";
import type { AttentionGPUBuffers } from "../model/model-gpu-helpers.ts";
import tgpu, { d } from "typegpu";
import { softmaxOnGpu } from "../shared/math-gpu.ts";

export const runSelfAttentionMechanismOnGPU = (
  input: number[][],
  headsCount: number,
  attentionWeights: AttentionWeights,
): AttentionActivations => {
  const contextLength = input.length;
  const hiddenDimensionsCount = input[0]?.length ?? -1;

  const inputQ = multiplyMatrices(input, attentionWeights.Q);
  const inputK = multiplyMatrices(input, attentionWeights.K);
  const inputV = multiplyMatrices(input, attentionWeights.V);

  const headDimensionsCount = divideToWhole(hiddenDimensionsCount, headsCount);

  const headActivations = runSelfAttentionHeadOnGPU(
    inputQ,
    inputK,
    inputV,
    headsCount,
    headDimensionsCount,
  );

  const attentionUpdate = multiplyMatrices(
    headActivations.output,
    attentionWeights.out,
  );

  validateSize(attentionUpdate, contextLength, hiddenDimensionsCount);

  return {
    normalizedInput: input,
    heads: new Array(headsCount)
      .fill(0)
      .map((_, h): AttentionHeadActivations => {
        return {
          attentionRelevancyOutput:
            headActivations.attentionRelevancyOutput[h]!,
          inputK: sliceRows(
            headActivations.inputK,
            h * headDimensionsCount,
            (h + 1) * headDimensionsCount,
          ),
          inputQ: sliceRows(
            headActivations.inputQ,
            h * headDimensionsCount,
            (h + 1) * headDimensionsCount,
          ),
          inputV: sliceRows(
            headActivations.inputV,
            h * headDimensionsCount,
            (h + 1) * headDimensionsCount,
          ),
          output: sliceRows(
            headActivations.output,
            h * headDimensionsCount,
            (h + 1) * headDimensionsCount,
          ),
          softmaxOutput: headActivations.softmaxOutput[h]!,
        };
      }),
    outMatrixInputActivations: headActivations.output,
    output: attentionUpdate,
  };
};

export const runSelfAttentionMechanismOnGPUBackup = (
  hiddenDimensionsCount: number,
  headsCount: number,
  attentionWeights: AttentionGPUBuffers,
  buffers: {
    input: MatrixBuffer;
    q: MatrixBuffer;
    k: MatrixBuffer;
    v: MatrixBuffer;
    headsOutputBuffer: MatrixBuffer;
    output: MatrixBuffer;
  },
) => {
  multiplyMatricesOnGPU(buffers.input, attentionWeights.Q, buffers.q);
  multiplyMatricesOnGPU(buffers.input, attentionWeights.K, buffers.k);
  multiplyMatricesOnGPU(buffers.input, attentionWeights.V, buffers.v);

  // TODO: Run attention heads

  multiplyMatricesOnGPU(
    buffers.headsOutputBuffer,
    attentionWeights.out,
    buffers.output,
  );
};

const paramsLayout = tgpu.bindGroupLayout({
  q: {
    storage: matrixBufferDefinition,
    access: "readonly",
  },
  k: {
    storage: matrixBufferDefinition,
    access: "readonly",
  },
  v: {
    storage: matrixBufferDefinition,
    access: "readonly",
  },
  output: {
    storage: matrixBufferDefinition,
    access: "readonly",
  },
  dimensionsPerHead: {
    uniform: d.u32,
  },
});

export const runSelfAttentionHeadOnGPU = (
  inputQ: number[][],
  inputK: number[][],
  inputV: number[][],
  headCount: number,
  headDimensionsCount: number,
) => {
  const attentionRelevancyOutput = new Array(headCount)
    .fill(0)
    .map((_) => createMatrix(inputQ.length, inputQ.length));
  const matchingKeyProducts = new Array(headCount)
    .fill(0)
    .map((_) => createMatrix(inputQ.length, inputQ.length));
  const output = createMatrix(inputQ.length, inputQ[0]!.length);

  for (let h = 0; h < headCount; h++) {
    const offset = h * headDimensionsCount;

    for (let i = 0; i < inputQ.length; i++) {
      const relevancyLogits = createVector(i + 1);

      for (let l = 0; l < relevancyLogits.length; l++) {
        let summed = 0;

        for (let k = 0; k < headDimensionsCount; k++) {
          summed += inputQ[i]![k + offset]! * inputK[l]![k + offset]!;
        }

        relevancyLogits[l]! = summed / Math.sqrt(headDimensionsCount);
      }

      const relevancy = softmaxOnGpu(relevancyLogits);

      for (let l = 0; l < relevancy.length; l++) {
        attentionRelevancyOutput[h]![i]! = relevancyLogits;
        matchingKeyProducts[h]![i]! = relevancy;
      }
    }
  }

  for (let i = 0; i < output.length; i++) {
    for (let j = 0; j < output[0]!.length; j++) {
      const h = Math.floor(j / headDimensionsCount);

      for (let l = 0; l < i + 1; l++) {
        output[i]![j]! += matchingKeyProducts[h]![i]![l]! * inputV[l]![j]!;
      }
    }
  }

  return {
    inputK,
    inputQ,
    inputV,
    attentionRelevancyOutput,
    softmaxOutput: matchingKeyProducts,
    output,
  };
};
