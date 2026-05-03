import { divideToWhole, softmax } from "../shared/math.ts";
import {
  addVectorsInMatrix,
  applyScalarToVector,
  transpose,
  multiplyMatrices,
  multiplyMatrixWithVector,
  validateSize,
  concatenateMatricesVertically,
  sliceRows,
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

  const sliceForHead = (matrix: number[][], headIndex: number) =>
    sliceRows(
      matrix,
      headIndex * headDimensionsCount,
      (headIndex + 1) * headDimensionsCount,
    );

  const headActivations = new Array(headsCount).fill([]).map((_, headIndex) => {
    return runSelfAttentionHead(
      sliceForHead(inputQ, headIndex),
      sliceForHead(inputK, headIndex),
      sliceForHead(inputV, headIndex),
    );
  });

  const headsConcatenated = concatenateMatricesVertically(
    headActivations.map((h) => h.output),
  );

  const attentionUpdate = multiplyMatrices(
    headsConcatenated,
    attentionWeights.out,
  );

  validateSize(attentionUpdate, contextLength, hiddenDimensionsCount);

  return {
    normalizedInput: input,
    heads: headActivations,
    outMatrixInputActivations: headsConcatenated,
    output: attentionUpdate,
  };
};

export const runSelfAttentionHead = (
  inputHeadQ: number[][],
  inputHeadK: number[][],
  inputHeadV: number[][],
): AttentionHeadActivations => {
  const contextLength = inputHeadQ.length;
  const headDimensionCount = inputHeadQ[0]?.length ?? -1;

  const headActivations = inputHeadQ.reduce<AttentionHeadActivations>(
    (partialActivations, vectorQ, index): AttentionHeadActivations => {
      const lookbackKeys = inputHeadK.slice(0, index + 1); // inclusive
      const lookbackValues = inputHeadV.slice(0, index + 1); // inclusive

      const relevancyVector = multiplyMatrixWithVector(
        vectorQ,
        transpose(lookbackKeys),
      );

      const matchingKeyProducts = relevancyVector.map(
        (value) => value / Math.sqrt(headDimensionCount),
      );

      const matchingKeyDistribution = softmax(matchingKeyProducts);

      const vectorUpdatePayload = matchingKeyDistribution.map(
        (scalar, index) => {
          const valueVector = lookbackValues[index]!;

          return applyScalarToVector(scalar, valueVector);
        },
      );

      const updateVector = addVectorsInMatrix(vectorUpdatePayload);

      return {
        ...partialActivations,
        attentionRelevancyOutput: [
          ...partialActivations.attentionRelevancyOutput,
          matchingKeyProducts,
        ],
        softmaxOutput: [
          ...partialActivations.softmaxOutput,
          matchingKeyDistribution,
        ],
        lookbackUpdateVectors: [
          ...partialActivations.lookbackUpdateVectors,
          vectorUpdatePayload,
        ],
        output: [...partialActivations.output, updateVector],
      };
    },
    {
      inputK: inputHeadK,
      inputV: inputHeadV,
      inputQ: inputHeadQ,
      attentionRelevancyOutput: [],
      softmaxOutput: [],
      lookbackUpdateVectors: [],
      output: [],
    } satisfies AttentionHeadActivations,
  );

  validateSize(headActivations.output, contextLength, headDimensionCount);

  return headActivations;
};
