import {
  embeddings,
  HIDDEN_DIMENSIONS_SIZE,
  transformers,
  unembeddingsMatrix,
} from "./weights.ts";
import { softmax } from "./math.ts";
import { addMatrices, multiplyMatrices, validateSize } from "./matrices.ts";
import { tokenize, tokens } from "./tokenizer.ts";
import { getMultilayerPerceptronUpdateMatrix } from "./mlp.ts";
import { getPositionEncoding } from "./position-encoding.ts";

export const runLlm = (input: string) => {
  const inputTokens = tokenize(input);

  let intermediateState = inputTokens.map((t) => embeddings[t]);

  const CONTEXT_SIZE = inputTokens.length;

  validateSize(intermediateState, CONTEXT_SIZE, HIDDEN_DIMENSIONS_SIZE);

  const positionalEncoding = getPositionEncoding(
    CONTEXT_SIZE,
    HIDDEN_DIMENSIONS_SIZE,
  );

  intermediateState = addMatrices(intermediateState, positionalEncoding);

  validateSize(intermediateState, CONTEXT_SIZE, HIDDEN_DIMENSIONS_SIZE);

  for (const transformer of transformers) {
    // TODO: normalization

    // TODO: attention processing

    // TODO: residual connection

    // TODO: normalization

    const mlpUpdateMatrix = getMultilayerPerceptronUpdateMatrix(
      intermediateState,
      transformer.multilayerPerceptron,
    );

    // Apply updated knowledge
    intermediateState = addMatrices(intermediateState, mlpUpdateMatrix);
  }

  // TODO: Final normalization

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
