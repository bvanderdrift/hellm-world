import { divideToWhole, softmax } from "../shared/math.ts";
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

export const runSelfAttentionMechanism = (
  input: Matrix,
  headsCount: number,
  attentionWeights: AttentionWeights,
): AttentionActivations => {
  const hiddenDimensionsCount = input.dimensions;

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
  inputQ: Matrix,
  inputK: Matrix,
  inputV: Matrix,
  headCount: number,
  headDimensionsCount: number,
) => {
  const attentionRelevancyOutput = new Array(headCount)
    .fill(0)
    .map((_) => createMatrix(inputQ.vectors, inputQ.vectors));
  const matchingKeyProducts = new Array(headCount)
    .fill(0)
    .map((_) => createMatrix(inputQ.vectors, inputQ.vectors));
  const output = createMatrix(inputQ.vectors, inputQ.dimensions);

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

  for (let i = 0; i < output.vectors; i++) {
    for (let j = 0; j < output.dimensions; j++) {
      const h = Math.floor(j / headDimensionsCount);
      const outputIndex = getFlatIndex(i, j, output.dimensions);

      for (let l = 0; l < i + 1; l++) {
        output.values[outputIndex]! +=
          matchingKeyProducts[h]!.values[getFlatIndex(i, l, output.vectors)]! *
          inputV.values[getFlatIndex(l, j, inputV.dimensions)]!;
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
