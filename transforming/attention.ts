import { divideToWhole, softmax } from "../shared/math.ts";
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

export const runSelfAttentionMechanism = (
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

  const headActivations = runSelfAttentionHead(
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
export const runSelfAttentionHead = (
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

      const relevancy = softmax(relevancyLogits);

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
