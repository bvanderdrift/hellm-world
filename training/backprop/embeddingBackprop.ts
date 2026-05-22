import {
  createMatrix,
  getFlatIndex,
  type Matrix,
} from "../../shared/matrices.ts";

export const embeddingsBackprop = (
  embeddingWeights: Matrix,
  outputGradients: Matrix,
  inputPositionToVocabPosition: number[],
) => {
  /**
   * Since embeddings are direct inputs to transformer inputs; the transformer input gradients are ALMOST the gradients for the embeddings.
   * Only thing to take into account is the fact that one token might be fetched multiple times from the embeddings lookup table
   * So we need to sum the gradients if it's multiple ones
   */
  const embeddingWeightsGradients = createMatrix(
    embeddingWeights.vectors,
    embeddingWeights.dimensions,
  );

  /**
   * z_i = Math.sqrt(j) * e_i + p_i
   *    where e_i is token embedding and p_i is positional encoding
   *
   * dL/e_i = dL/dz_i * dz_i/de_i
   *
   * dz_i/de_i = Math.sqrt(j)
   *
   * dL/e_i = dL/dz_i * Math.sqrt(j)
   *
   * We don't care about dL/dp_i (which is also dL/dz_i) since p_i a non-trainable algorithmic output
   */
  for (
    let inputTokenIndex = 0;
    inputTokenIndex < outputGradients.vectors;
    inputTokenIndex++
  ) {
    const vocabIndex = inputPositionToVocabPosition[inputTokenIndex]!;

    for (
      let dimensionIndex = 0;
      dimensionIndex < outputGradients.dimensions;
      dimensionIndex++
    ) {
      const flatEmbeddingIndex = getFlatIndex(
        vocabIndex,
        dimensionIndex,
        embeddingWeightsGradients.dimensions,
      );

      const z_i =
        outputGradients.values[
          getFlatIndex(
            inputTokenIndex,
            dimensionIndex,
            outputGradients.dimensions,
          )
        ]!;

      embeddingWeightsGradients.values[flatEmbeddingIndex]! +=
        z_i * Math.sqrt(outputGradients.dimensions);
    }
  }

  return embeddingWeightsGradients;
};
