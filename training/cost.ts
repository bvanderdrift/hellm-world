import { generateProbabilities } from "../running/llm.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { addVectorsInMatrix } from "../shared/matrices.ts";
import type { Weights } from "../weights/types.ts";

export const getAvgCostVector = (
  trainingData: string[][],
  weights: Weights,
) => {
  const costVectors = trainingData.flatMap((sequence) =>
    sequence.map((_, i) => {
      const trainingInput = sequence.slice(0, i + 1);
      const trainingOutput = sequence[i + 1] ?? END_OF_SEQUENCE_TOKEN;

      let seenTokens = 0;

      const expectedOutputProbabilities = weights.vocabulary.map((token) => {
        if (token === trainingOutput) {
          seenTokens++;
          return 1;
        }

        return 0;
      });

      if (seenTokens !== 1) {
        throw new Error(
          `Saw output token ${seenTokens} times, expected to see it 1 time`,
        );
      }

      const actualOutputProbabilities = generateProbabilities(
        trainingInput,
        weights,
      );

      if (
        actualOutputProbabilities.length !== expectedOutputProbabilities.length
      ) {
        throw new Error(
          `Expected probability vector length (${expectedOutputProbabilities.length}) does not match generated logit length (${actualOutputProbabilities.length})`,
        );
      }

      return actualOutputProbabilities.map((actualProbability, index) => {
        const delta = actualProbability - expectedOutputProbabilities[index]!;
        return Math.pow(delta, 2);
      });
    }),
  );

  return addVectorsInMatrix(costVectors).map(
    (totalCost) => totalCost / costVectors.length,
  );
};
