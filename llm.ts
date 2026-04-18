import { softmax } from "./math.ts";
import {
  addMatrices,
  multiplyMatrices,
  normalize,
  validateSize,
} from "./matrices.ts";
import { tokenize, tokens, type Token } from "./tokenizer.ts";
import { getMultilayerPerceptronUpdateMatrix } from "./transforming/mlp.ts";
import { getPositionEncoding } from "./position-encoding.ts";
import { runSelfAttentionMechanism } from "./transforming/attention.ts";
import type { Weights } from "./weights/types.ts";
import { extractDimensionSizes } from "./weights/weight-helpers.ts";

export const runLlm = (input: string, weights: Weights<Token>) => {
  const { hiddenDimensionsSize, vocabSize } = extractDimensionSizes(weights);

  const inputTokens = tokenize(input);

  let intermediateState = inputTokens.map((t) =>
    weights.embeddings[t].map((v) => v * Math.sqrt(hiddenDimensionsSize)),
  );

  const contextSize = inputTokens.length;

  validateSize(intermediateState, contextSize, hiddenDimensionsSize);

  const positionalEncoding = getPositionEncoding(
    contextSize,
    hiddenDimensionsSize,
  );

  intermediateState = addMatrices(intermediateState, positionalEncoding);

  validateSize(intermediateState, contextSize, hiddenDimensionsSize);

  for (const transformer of weights.transformers) {
    const attentionUpdateMatrix = runSelfAttentionMechanism(
      // Normalize input only, don't normalize the intermediateState iself
      // Reason: of this block outputs 0 for a feature, we keep x + 0 = x. But if we normalize the root variable we get norm(x) + 0 = norm(x) so a transform has still happened even if the block said not to
      normalize(intermediateState),
      transformer.attention,
    );

    intermediateState = addMatrices(intermediateState, attentionUpdateMatrix);

    const mlpUpdateMatrix = getMultilayerPerceptronUpdateMatrix(
      // Normalize input only, don't normalize the intermediateState iself
      // Reason: of this block outputs 0 for a feature, we keep x + 0 = x. But if we normalize the root variable we get norm(x) + 0 = norm(x) so a transform has still happened even if the block said not to
      normalize(intermediateState),
      transformer.multilayerPerceptron,
    );

    // Apply updated knowledge
    intermediateState = addMatrices(intermediateState, mlpUpdateMatrix);
  }

  const unembeddedState = multiplyMatrices(
    normalize(intermediateState),
    weights.unembeddings,
  );

  validateSize(unembeddedState, contextSize, vocabSize);

  // Last vector is probability logits
  const logits = unembeddedState[contextSize - 1];

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
