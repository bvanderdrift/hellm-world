import { divideToWhole, dotProduct, softmax } from "../shared/math.ts";
import {
  addVectorsInMatrix,
  applyScalarToVector,
  multiplyMatrices,
  validateSize,
} from "../shared/matrices.ts";
import type { AttentionWeights } from "../model/types.ts";

export const runSelfAttentionMechanism = (
  input: number[][],
  headsCount: number,
  attentionWeights: AttentionWeights,
) => {
  const contextLength = input.length;
  const hiddenDimensionsCount = input[0]?.length ?? -1;

  const inputQ = multiplyMatrices(input, attentionWeights.Q);
  const inputK = multiplyMatrices(input, attentionWeights.K);
  const inputV = multiplyMatrices(input, attentionWeights.V);

  const headDimensionsCount = divideToWhole(hiddenDimensionsCount, headsCount);

  const headOutputs = new Array(headsCount).fill([]).map((_, headIndex) => {
    return runSelfAttentionHead(
      inputQ.map((vector) =>
        vector.slice(
          headIndex * headDimensionsCount,
          (headIndex + 1) * headDimensionsCount,
        ),
      ),
      inputK.map((vector) =>
        vector.slice(
          headIndex * headDimensionsCount,
          (headIndex + 1) * headDimensionsCount,
        ),
      ),
      inputV.map((vector) =>
        vector.slice(
          headIndex * headDimensionsCount,
          (headIndex + 1) * headDimensionsCount,
        ),
      ),
    );
  });

  const headsConcatenated = headOutputs.reduce(
    (partial, head) =>
      partial.map((vector, vectorIndex) => [...vector, ...head[vectorIndex]!]),
    new Array(contextLength).fill([]),
  );

  const attentionUpdate = multiplyMatrices(
    headsConcatenated,
    attentionWeights.out,
  );

  validateSize(attentionUpdate, contextLength, hiddenDimensionsCount);

  return attentionUpdate;
};

export const runSelfAttentionHead = (
  inputHeadQ: number[][],
  inputHeadK: number[][],
  inputHeadV: number[][],
) => {
  const keyValues: {
    key: number[];
    value: number[];
  }[] = [];

  const contextLength = inputHeadQ.length;
  const headDimensionCount = inputHeadQ[0]?.length ?? -1;

  const attentionMatrix = inputHeadQ.map((vectorQ, index) => {
    const vectorK = inputHeadK[index]!;
    const vectorV = inputHeadV[index]!;

    // Do it BEFORE the dotproducts so it self-matches
    keyValues.push({
      key: vectorK,
      value: vectorV,
    });

    const matchingKeyProducts = keyValues.map(
      ({ key }) => dotProduct(vectorQ, key) / Math.sqrt(headDimensionCount),
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

    const vectorUpdatePayload = matchingKeyDistribution.map((scalar, index) => {
      const entry = keyValues[index];

      if (!entry) {
        // attempting to look-forward - return 0-vector
        return new Array<number>(headDimensionCount).fill(0);
      }

      return applyScalarToVector(scalar, entry.value);
    });

    return addVectorsInMatrix(vectorUpdatePayload);
  });

  validateSize(attentionMatrix, contextLength, headDimensionCount);

  return attentionMatrix;
};
