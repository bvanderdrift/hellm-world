import { softmax } from "../shared/math.ts";
import {
  addMatrices,
  multiplyMatrices,
  normalize,
  validateSize,
} from "../shared/matrices.ts";
import { tokenize } from "../shared/tokenizer.ts";
import { getMultilayerPerceptronUpdateMatrix } from "../transforming/mlp.ts";
import { getPositionEncoding } from "./position-encoding.ts";
import { runSelfAttentionMechanism } from "../transforming/attention.ts";
import type { Weights } from "../weights/types.ts";
import {
  extractHiddenDimensionSize,
  findTokenIndex,
  validateWeights,
} from "../weights/weight-helpers.ts";
import { getLatestCheckpointWeights } from "../weights/weight-io.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";

const contextTimeout = 100;

export const runLlm = (input: string, model: string) => {
  let outputTokens: string[] = [];

  const weights = getLatestCheckpointWeights(model);

  validateWeights(weights);

  const inputTokens = tokenize(input, weights.vocabulary);

  for (let index = 0; index < contextTimeout; index++) {
    const probabilities = generateProbabilities(
      [...inputTokens, ...outputTokens],
      weights,
    );

    const nextToken = pickToken(probabilities, weights.vocabulary);

    if (nextToken === END_OF_SEQUENCE_TOKEN) {
      break;
    }

    outputTokens.push(nextToken);
  }

  return outputTokens.join(" ");
};

export const llmForwardPassByTokens = (input: string[], weights: Weights) => {
  const hiddenDimensionsSize = extractHiddenDimensionSize(weights);

  const startState = input.map((token) => {
    const tokenIndex = findTokenIndex(weights.vocabulary, token);

    return weights.embeddings[tokenIndex]!.map(
      (v) => v * Math.sqrt(hiddenDimensionsSize),
    );
  });

  return llmForwardPass(startState, weights);
};

export const generateLogits = (input: string[], weights: Weights) => {
  const unembeddedState = llmForwardPassByTokens(input, weights);

  // Last vector is probability logits
  const logits = unembeddedState[unembeddedState.length - 1];

  if (!logits) {
    throw new Error(`Logits array is undefined`);
  }

  return logits;
};

export const generateProbabilities = (input: string[], weights: Weights) => {
  const logits = generateLogits(input, weights);

  return softmax(logits);
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

  const hiddenDimensionsSize = extractHiddenDimensionSize(weights);

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

  validateSize(unembeddedState, contextSize, weights.vocabulary.length);

  return unembeddedState;
};

export const pickToken = (probabilities: number[], vocabulary: string[]) => {
  const nextTokenIndex = getHighestValueIndex(probabilities);

  const nextToken = vocabulary[nextTokenIndex];

  if (!nextToken) {
    throw new Error(`Failed to find token at index ${nextTokenIndex}`);
  }

  return nextToken;
};
