export const embeddingsBackprop = (
  embeddingWeights: number[][],
  outputGradients: number[][],
  inputPositionToVocabPosition: number[],
) => {
  /**
   * Since embeddings are direct inputs to transformer inputs; the transformer input gradients are ALMOST the gradients for the embeddings.
   * Only thing to take into account is the fact that one token might be fetched multiple times from the embeddings lookup table
   * So we need to sum the gradients if it's multiple ones
   */
  const embeddingWeightsGradients = embeddingWeights.map((tokenEmbedding) =>
    new Array(tokenEmbedding.length).fill(0),
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
  outputGradients.forEach((inputGradientsVector, inputTokenIndex) => {
    const vocabIndex = inputPositionToVocabPosition[inputTokenIndex]!;

    const currentInputGradients = embeddingWeightsGradients[vocabIndex]!;

    const newInputGradients = currentInputGradients.map(
      (partialInputGradient, dimensionIndex) => {
        const z_i = inputGradientsVector[dimensionIndex]!;
        return (
          partialInputGradient + z_i * Math.sqrt(inputGradientsVector.length)
        );
      },
    );

    embeddingWeightsGradients[vocabIndex] = newInputGradients;
  });

  return embeddingWeightsGradients;
};
