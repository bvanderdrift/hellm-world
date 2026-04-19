import { softmax } from "./math.ts";
import {
  addMatrices,
  multiplyMatrices,
  normalize,
  validateSize,
} from "./matrices.ts";
import { tokenize } from "./tokenizer.ts";
import { getMultilayerPerceptronUpdateMatrix } from "./transforming/mlp.ts";
import { getPositionEncoding } from "./position-encoding.ts";
import { runSelfAttentionMechanism } from "./transforming/attention.ts";
import type { Weights } from "./weights/types.ts";
import {
  extractDimensionSizes,
  validateSizing,
} from "./weights/weight-helpers.ts";

export const runLlm = <T extends string>(
  input: string,
  weights: Weights<T>,
) => {
  validateSizing(weights);

  const { hiddenDimensionsSize } = extractDimensionSizes(weights);

  const inputTokens = tokenize(input, weights.tokens);

  const startState = inputTokens.map((t) =>
    weights.embeddings[t].map((v) => v * Math.sqrt(hiddenDimensionsSize)),
  );

  const unembeddedState = llmForwardPass(startState, weights);

  // Last vector is probability logits
  const logits = unembeddedState[unembeddedState.length - 1];

  if (!logits) {
    throw new Error(`Logits array is undefined`);
  }

  return decodeLogits(logits, weights.tokens);
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

export const llmForwardPass = (startState: number[][], weights: Weights) => {
  const contextSize = startState.length;

  const { hiddenDimensionsSize, vocabSize } = extractDimensionSizes(weights);

  validateSize(startState, contextSize, hiddenDimensionsSize);

  const positionalEncoding = getPositionEncoding(
    contextSize,
    hiddenDimensionsSize,
  );

  let intermediateState = addMatrices(startState, positionalEncoding);

  validateSize(intermediateState, contextSize, hiddenDimensionsSize);

  for (const transformer of weights.transformers) {
    const attentionUpdateMatrix = runSelfAttentionMechanism(
      // Normalize input only, don't normalize the intermediateState iself
      // Reason: of this block outputs 0 for a feature, we keep x + 0 = x. But if we normalize the root variable we get norm(x) + 0 = norm(x) so a transform has still happened even if the block said not to
      normalize(intermediateState),
      weights.headsCount,
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

  return unembeddedState;
};

export const decodeLogits = (logits: number[], tokens: string[]) => {
  const probabilities = softmax(logits);

  const nextTokenIndex = getHighestValueIndex(probabilities);

  const nextToken = tokens[nextTokenIndex];

  if (!nextToken) {
    throw new Error(`Failed to find token at index ${nextTokenIndex}`);
  }

  return nextToken;
};
