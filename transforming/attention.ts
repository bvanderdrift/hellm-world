import { divideToWhole, softmax } from "../shared/math.ts";
import {
  addVectorsInMatrix,
  applyScalarToVector,
  transpose,
  multiplyMatrices,
  multiplyMatrixWithVector,
  validateSize,
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

  const sliceRows = (matrix: number[][], headIndex: number) =>
    matrix.map((vector) =>
      vector.slice(
        headIndex * headDimensionsCount,
        (headIndex + 1) * headDimensionsCount,
      ),
    );

  const headActivations = new Array(headsCount).fill([]).map((_, headIndex) => {
    return runSelfAttentionHead(
      sliceRows(inputQ, headIndex),
      sliceRows(inputK, headIndex),
      sliceRows(inputV, headIndex),
    );
  });

  const headsConcatenated = headActivations.reduce(
    (partial, head) =>
      partial.map((vector, vectorIndex) => [
        ...vector,
        ...head.output[vectorIndex]!,
      ]),
    new Array(contextLength).fill([]),
  );

  const attentionUpdate = multiplyMatrices(
    headsConcatenated,
    attentionWeights.out,
  );

  validateSize(attentionUpdate, contextLength, hiddenDimensionsCount);

  return {
    normalizedInput: input,
    inputK,
    inputV,
    inputQ,
    heads: headActivations,
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

      const matchingKeyProductPadded = [
        ...matchingKeyProducts,
        // We now did a lookback only. We need to pad the key matching array with -Infinity for the full length of input to match length
        ...new Array<number>(contextLength - (index + 1)).fill(-Infinity),
      ];

      if (matchingKeyProductPadded.length !== contextLength) {
        throw new Error(
          `Expected key matching vector to be of dimension ${contextLength}, received ${matchingKeyProductPadded.length}`,
        );
      }

      const matchingKeyDistribution = softmax(matchingKeyProductPadded);

      const vectorUpdatePayload = matchingKeyDistribution.map(
        (scalar, index) => {
          const value = lookbackValues[index];

          if (!value) {
            // attempting to look-forward - return 0-vector
            return new Array<number>(headDimensionCount).fill(0);
          }

          return applyScalarToVector(scalar, value);
        },
      );

      const updateVector = addVectorsInMatrix(vectorUpdatePayload);

      return {
        attentionRelevancyOutput: [
          ...partialActivations.attentionRelevancyOutput,
          matchingKeyProductPadded,
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
      attentionRelevancyOutput: [],
      softmaxOutput: [],
      lookbackUpdateVectors: [],
      output: [],
    } satisfies AttentionHeadActivations,
  );

  validateSize(headActivations.output, contextLength, headDimensionCount);

  return headActivations;
};
