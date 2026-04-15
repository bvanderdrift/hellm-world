import {
  embeddings,
  HIDDEN_DIMENSIONS_SIZE,
  transformers,
  unembeddingsMatrix,
} from "./weights.ts";
import { softmax } from "./math.ts";
import { multiplyMatrices, validateSize } from "./matrices.ts";
import { tokenize, tokens } from "./tokenizer.ts";
import { runMultilayerPerceptronOnMatrix } from "./mlp.ts";

export const runLlm = (input: string) => {
  const inputTokens = tokenize(input);

  const embeddedState = inputTokens.map((t) => embeddings[t]);

  const CONTEXT_SIZE = inputTokens.length;

  validateSize(embeddedState, CONTEXT_SIZE, HIDDEN_DIMENSIONS_SIZE);

  let intermediateState = embeddedState;

  for (const transformer of transformers) {
    // TODO attention

    intermediateState = runMultilayerPerceptronOnMatrix(
      intermediateState,
      transformer.multilayerPerceptron,
    );
  }

  const unembeddedState = multiplyMatrices(
    intermediateState,
    unembeddingsMatrix,
  );

  // Last vector is probability logits
  const logits = unembeddedState[CONTEXT_SIZE - 1];

  if (!logits) {
    throw new Error(`Logits array is undefined`);
  }

  const probabilities = softmax(logits);

  const nextTokenIndex = getHighestValueIndex(probabilities);

  const nextToken = tokens[nextTokenIndex];

  if (!nextToken) {
    throw new Error(`Failed to find token at index ${nextTokenIndex}`);
  }

  return nextToken;
};

export const getHighestValueIndex = (values: number[]) => {
  return values.reduce(
    (tracker, value, index) => {
      if (value > tracker.value) {
        return {
          index,
          value,
        };
      }

      return tracker;
    },
    {
      index: 0,
      value: -Infinity,
    },
  ).index;
};
