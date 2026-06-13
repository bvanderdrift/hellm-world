import { divideToWhole } from "../shared/math.ts";
import {
  multiplyMatrices,
  sliceRows,
  createMatrix,
  type Matrix,
  getFlatIndex,
} from "../shared/matrices.ts";
import type { AttentionWeights } from "../model/model-types.ts";
import type { AttentionActivations } from "../model/activations-types.ts";
import { softmax } from "../shared/softmax.ts";

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

  const headsActivations = runSelfAttentionHead(
    inputQ,
    inputK,
    inputV,
    headsCount,
    headDimensionsCount,
  );

  const attentionUpdate = multiplyMatrices(
    headsActivations.output,
    attentionWeights.out,
  );

  return {
    normalizedInput: input,
    headsActivations,
    outMatrixInputActivations: headsActivations.output,
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
  const attentionRelevancyOutput = createMatrix(
    inputQ.vectors,
    inputQ.vectors * headCount,
  );
  const matchingKeyProducts = createMatrix(
    inputQ.vectors,
    inputQ.vectors * headCount,
  );
  const output = createMatrix(inputQ.vectors, inputQ.dimensions);

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

  for (let i = 0; i < output.vectors; i++) {
    for (let j = 0; j < output.dimensions; j++) {
      const h = Math.floor(j / headDimensionsCount);
      const offset = h * inputQ.vectors;
      const outputIndex = getFlatIndex(i, j, output.dimensions);

      for (let l = 0; l < i + 1; l++) {
        output.values[outputIndex]! +=
          matchingKeyProducts.values[
            getFlatIndex(i, offset + l, matchingKeyProducts.dimensions)
          ]! * inputV.values[getFlatIndex(l, j, inputV.dimensions)]!;
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
