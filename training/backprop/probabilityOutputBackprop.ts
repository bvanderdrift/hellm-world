export const probabilityOutputBackprop = (
  unembeddingsOutputLogits: number[][],
  outputProbabilitiesVector: number[],
  contextLength: number,
  correctTokenIndex: number,
) =>
  unembeddingsOutputLogits.map((outputVector, inputTokenIndex) => {
    if (inputTokenIndex !== contextLength - 1) {
      // We're not training on these, so no gradients
      return new Array(outputVector.length).fill(0);
    }

    return outputVector.map((_, vocabIndex) => {
      const isCorrectToken = vocabIndex === correctTokenIndex;
      const actualProbability = outputProbabilitiesVector[vocabIndex]!;
      const wantedProbability = isCorrectToken ? 1 : 0;

      /**
       * This seems to be too easy and not like it's a derivative
       * But it is. This is dL/dz_i
       *
       * Check this out:
       *
       * L = CEL_i = -log(p_i)
       * p_i = exp(z_i) / sum_j exp(z_j)
       * So L = -log(exp(z_i) / sum_j exp(z_j)) = -z_i + log(sum_j exp(z_j))
       * So d(-z_i)/dz_i = -1 and d(log(n))/dn = 1 / n and d(log(sum_j exp(z_j)))/dz_i = (1 / (sum_j exp(z_j)) * exp(z_i) = exp(z_i) / (sum_j exp(z_j) = p_i
       * Wowow; so dL/dz_i = d(-z_i)/dz_i + that sum part = -1 + p_i = p_i - 1 for i = k and 0 + p_j for non-i values (since d(z_j)/dz_i = 0)
       */
      return actualProbability - wantedProbability;
    });
  });
