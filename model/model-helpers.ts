import {
  getMatrixParameterCount,
  operateOnMatrices,
  operateOnVectors,
} from "../shared/matrices.ts";
import { validateSameWeightShape } from "./model-validation.ts";
import type { TransformerWeights, Model } from "./model-types.ts";

export const extractHiddenDimensionSize = (model: Model) => {
  const embeddingsArray = Object.values(model.embeddings);
  return embeddingsArray[0]!.length;
};

export const findTokenIndex = (vocabulary: string[], token: string) => {
  const tokenIndex = vocabulary.indexOf(token);

  if (tokenIndex === -1) {
    throw new Error(`Failed to find token ${token} in vocabulary`);
  }

  return tokenIndex;
};

export const operateSingleWeights = (
  weights: Model,
  operation: (value: number) => number,
) => {
  // Hacky solution hihi
  return operateCombinedWeights(weights, weights, (v1) => operation(v1));
};

export const operateCombinedWeights = (
  weights1: Model,
  weights2: Model,
  operation: (v1: number, w2: number) => number,
): Model => {
  validateSameWeightShape(weights1, weights2);

  return {
    ...weights1,
    embeddings: operateOnMatrices(
      weights1.embeddings,
      weights2.embeddings,
      operation,
    ),
    unembeddings: operateOnMatrices(
      weights1.unembeddings,
      weights2.unembeddings,
      operation,
    ),
    transformers: weights1.transformers.map(
      (transformerWeights1, index): TransformerWeights => {
        const transformerWeights2 = weights2.transformers[index]!; // Type-safe b/c of shape check above

        return {
          attention: {
            Q: operateOnMatrices(
              transformerWeights1.attention.Q,
              transformerWeights2.attention.Q,
              operation,
            ),
            K: operateOnMatrices(
              transformerWeights1.attention.K,
              transformerWeights2.attention.K,
              operation,
            ),
            V: operateOnMatrices(
              transformerWeights1.attention.V,
              transformerWeights2.attention.V,
              operation,
            ),
            out: operateOnMatrices(
              transformerWeights1.attention.out,
              transformerWeights2.attention.out,
              operation,
            ),
          },
          multilayerPerceptron: {
            wDown: {
              weightsMatrix: operateOnMatrices(
                transformerWeights1.multilayerPerceptron.wDown.weightsMatrix,
                transformerWeights2.multilayerPerceptron.wDown.weightsMatrix,
                operation,
              ),
              biasVector: operateOnVectors(
                transformerWeights1.multilayerPerceptron.wDown.biasVector,
                transformerWeights2.multilayerPerceptron.wDown.biasVector,
                operation,
              ),
            },
            wUp: {
              weightsMatrix: operateOnMatrices(
                transformerWeights1.multilayerPerceptron.wUp.weightsMatrix,
                transformerWeights2.multilayerPerceptron.wUp.weightsMatrix,
                operation,
              ),
              biasVector: operateOnVectors(
                transformerWeights1.multilayerPerceptron.wUp.biasVector,
                transformerWeights2.multilayerPerceptron.wUp.biasVector,
                operation,
              ),
            },
          },
        };
      },
    ),
  };
};

export const makeZeroVersion = (weights: Model) =>
  operateSingleWeights(weights, () => 0);

export const getModelParameterCount = (model: Model) => {
  const transformersParameterCount = model.transformers.reduce(
    (sum, transformer) => {
      // All 4 should be same size, but let's just calculate each seperately just to be sure
      const kSize = getMatrixParameterCount(transformer.attention.K);
      const vSize = getMatrixParameterCount(transformer.attention.V);
      const qSize = getMatrixParameterCount(transformer.attention.Q);
      const outSize = getMatrixParameterCount(transformer.attention.out);

      const mlpUpSize = getMatrixParameterCount(
        transformer.multilayerPerceptron.wUp.weightsMatrix,
      );
      const mlpUpBiasSize = getMatrixParameterCount([
        transformer.multilayerPerceptron.wUp.biasVector,
      ]);

      const mlpDownSize = getMatrixParameterCount(
        transformer.multilayerPerceptron.wDown.weightsMatrix,
      );
      const mlpDownBiasSize = getMatrixParameterCount([
        transformer.multilayerPerceptron.wDown.biasVector,
      ]);

      const transformerParamterType =
        kSize +
        vSize +
        qSize +
        outSize +
        mlpUpSize +
        mlpUpBiasSize +
        mlpDownSize +
        mlpDownBiasSize;

      return sum + transformerParamterType;
    },
    0,
  );

  return (
    getMatrixParameterCount(model.embeddings) +
    transformersParameterCount +
    getMatrixParameterCount(model.unembeddings)
  );
};
