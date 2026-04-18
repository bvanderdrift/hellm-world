import { dotProduct, softmax } from "../math.ts";
import {
  addMatrices,
  addVectorsInMatrix,
  applyScalarToVector,
  multiplyMatrixWithVector,
  validateSize,
} from "../matrices.ts";
import {
  ATTENTION_DIMENSIONS,
  HIDDEN_DIMENSIONS_SIZE,
  type AttentionHeadWeights,
  type AttentionWeights,
} from "../weights.ts";

export const runSelfAttentionMechanism = (
  input: number[][],
  attentionWeights: AttentionWeights,
) => {
  const contextLength = input.length;
  const hiddenDimensionsCount = input[0]?.length ?? -1;

  const headOutputs = attentionWeights.heads.reduce(
    (summedActivation, head) => {
      const headOutput = runSelfAttentionHead(input, head);

      validateSize(headOutput, contextLength, hiddenDimensionsCount);

      return addMatrices(summedActivation, headOutput);
    },
    new Array(input.length).fill(new Array(hiddenDimensionsCount).fill(0)),
  );

  validateSize(headOutputs, contextLength, hiddenDimensionsCount);

  return headOutputs;
};

export const runSelfAttentionHead = (
  input: number[][],
  attentionHeadWeights: AttentionHeadWeights,
) => {
  const keyValues: {
    key: number[];
    value: number[];
  }[] = [];

  const contextLength = input.length;
  const hiddenDimensionCount = input[0]?.length ?? -1;
  const headDimensionCount = attentionHeadWeights.Q[0]?.length ?? -1;

  const attentionMatrix = input.map((vector, index) => {
    const vectorQ = multiplyMatrixWithVector(attentionHeadWeights.Q, vector);
    const vectorK = multiplyMatrixWithVector(attentionHeadWeights.K, vector);

    const vectorVDownsized = multiplyMatrixWithVector(
      attentionHeadWeights.V.down,
      vector,
    );
    const vectorV = multiplyMatrixWithVector(
      attentionHeadWeights.V.up,
      vectorVDownsized,
    );

    validateSize([vectorV], 1, hiddenDimensionCount);

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
        return new Array<number>(hiddenDimensionCount).fill(0);
      }

      return applyScalarToVector(scalar, entry.value);
    });

    return addVectorsInMatrix(vectorUpdatePayload);
  });

  validateSize(attentionMatrix, contextLength, hiddenDimensionCount);

  return attentionMatrix;
};
