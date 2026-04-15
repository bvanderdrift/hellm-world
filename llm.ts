import {
  embeddings,
  HIDDEN_DIMENSIONS_SIZE,
  outMatrix,
  unembeddingsMatrix,
} from "./weights.ts";
import { softmax } from "./math.ts";
import { multiplyMatrices, validateSize } from "./matrices.ts";
import { tokenize } from "./tokenizer.ts";

export const runLlm = (input: string) => {
  const inputTokens = tokenize(input);

  const embeddedState = inputTokens.map((t) => embeddings[t]);

  const CONTEXT_SIZE = inputTokens.length;

  validateSize(embeddedState, CONTEXT_SIZE, HIDDEN_DIMENSIONS_SIZE);

  // TODO the actual shizzle

  const unembeddedState = multiplyMatrices(embeddedState, unembeddingsMatrix);

  validateSize(unembeddedState, CONTEXT_SIZE, HIDDEN_DIMENSIONS_SIZE);

  const logits = multiplyMatrices(unembeddedState, outMatrix)[0];

  if (!logits) {
    throw new Error(`Logits array is undefined`);
  }

  const probabilities = softmax(logits);

  const nextTokenIndex = getHighestValueIndex(probabilities);

  const nextToken = inputTokens[nextTokenIndex];

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
