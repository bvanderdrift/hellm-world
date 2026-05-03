import { softmax } from "../shared/math.ts";
import {
  addMatrices,
  applyScalarToVector,
  multiplyMatrices,
  normalize,
  validateSize,
} from "../shared/matrices.ts";
import { tokenize } from "../shared/tokenizer.ts";
import { getMultilayerPerceptronActivations as getMultilayerPerceptronActivations } from "../transforming/mlp.ts";
import { getPositionEncoding } from "./position-encoding.ts";
import { runSelfAttentionMechanism } from "../transforming/attention.ts";
import type { Model } from "../model/model-types.ts";
import {
  extractHiddenDimensionSize,
  findTokenIndex,
} from "../model/model-helpers.ts";
import { getLatestCheckpointModel } from "../model/model-io.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { validateModel } from "../model/model-validation.ts";
import type {
  Activations,
  TransformerActivations,
} from "../model/activations-types.ts";

const contextTimeout = 100;

export const runLlm = function* (input: string, modelName: string) {
  let outputTokens: string[] = [];

  const { model } = getLatestCheckpointModel(modelName);

  validateModel(model);

  const inputTokens = tokenize(input, model.vocabulary);

  for (let index = 0; index < contextTimeout; index++) {
    const nextInput = [...inputTokens, ...outputTokens];
    const logits = generateLogits(nextInput, model);

    const probabilities = softmax(logits);

    const nextToken = pickToken(probabilities, model.vocabulary);

    if (nextToken === END_OF_SEQUENCE_TOKEN) {
      break;
    }

    outputTokens.push(nextToken);

    yield nextToken;
  }
};

const generateLogits = (input: string[], weights: Model) => {
  const { embeddings: unembeddedState } = llmForwardPassByTokens(
    input,
    weights,
    false,
  );

  // Last vector is probability logits
  const logits = unembeddedState[unembeddedState.length - 1];

  if (!logits) {
    throw new Error(`Logits array is undefined`);
  }

  return logits;
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

export const llmForwardPassByTokens = (
  input: string[],
  weights: Model,
  withActivations: boolean,
): {
  embeddings: number[][];
  activations: Activations | null;
} => {
  const hiddenDimensionsSize = extractHiddenDimensionSize(weights);

  /** middle-state needed for backprop */
  const inputPositionToVocabPosition = input.map((token) => {
    return findTokenIndex(weights.vocabulary, token);
  });

  const startState = inputPositionToVocabPosition.map((vocabIndex) => {
    const tokenEmbedding = weights.embeddings[vocabIndex]!;

    return applyScalarToVector(Math.sqrt(hiddenDimensionsSize), tokenEmbedding);
  });

  const contextSize = startState.length;

  validateSize(startState, contextSize, hiddenDimensionsSize);

  const positionalEncoding = getPositionEncoding(
    contextSize,
    hiddenDimensionsSize,
  );

  const embeddingsPositionallyEncoded = addMatrices(
    startState,
    positionalEncoding,
  );

  validateSize(
    embeddingsPositionallyEncoded,
    contextSize,
    hiddenDimensionsSize,
  );

  const transformerActivations: TransformerActivations[] = [];

  const embeddingsAfterTransformers = weights.transformers.reduce(
    (inputEmbeddings, transformer) => {
      const attentionInputEmbeddings = normalize(inputEmbeddings);

      const attentionActivations = runSelfAttentionMechanism(
        // Normalize input only, don't normalize the intermediateState iself
        // Reason: of this block outputs 0 for a feature, we keep x + 0 = x. But if we normalize the root variable we get norm(x) + 0 = norm(x) so a transform has still happened even if the block said not to
        attentionInputEmbeddings,
        weights.headsCount,
        transformer.attention,
      );

      const embeddingsWithAttentionUpdates = addMatrices(
        inputEmbeddings,
        attentionActivations.output,
      );

      const mlpInputEmbeddings = normalize(embeddingsWithAttentionUpdates);

      const mlpActivations = getMultilayerPerceptronActivations(
        // Normalize input only, don't normalize the intermediateState iself
        // Reason: of this block outputs 0 for a feature, we keep x + 0 = x. But if we normalize the root variable we get norm(x) + 0 = norm(x) so a transform has still happened even if the block said not to
        mlpInputEmbeddings,
        transformer.multilayerPerceptron,
        weights.mlpMultiple,
      );

      if (withActivations) {
        // We only store this when the flag is true; so we don't overflow memory during inference
        transformerActivations.push({
          transformerInput: inputEmbeddings,
          attention: attentionActivations,
          mlp: mlpActivations,
        });
      }

      // Apply updated knowledge
      return addMatrices(
        embeddingsWithAttentionUpdates,
        mlpActivations.downingOutput,
      );
    },
    embeddingsPositionallyEncoded,
  );

  const normalizedTransformersOutput = normalize(embeddingsAfterTransformers);

  const unembeddedState = multiplyMatrices(
    normalizedTransformersOutput,
    weights.unembeddings,
  );

  validateSize(unembeddedState, contextSize, weights.vocabulary.length);

  const missingTransformerActivationsCount =
    weights.transformers.length - transformerActivations.length;

  if (withActivations && missingTransformerActivationsCount > 0) {
    // One sanity check, the rest is available either way
    throw new Error(
      `Missing ${missingTransformerActivationsCount} transformer activations`,
    );
  }

  return {
    embeddings: unembeddedState,
    activations: withActivations
      ? {
          inputPositionToVocabPosition,
          tokensToPosition: startState,
          positionToTransformers: positionalEncoding,
          transformerActivations,
          transformersToNormalizer: embeddingsAfterTransformers,
          normalizerToUnembeddings: normalizedTransformersOutput,
          unembeddingsOutputLogits: unembeddedState,
        }
      : null,
  };
};

export const pickToken = (probabilities: number[], vocabulary: string[]) => {
  const nextTokenIndex = getHighestValueIndex(probabilities);

  const nextToken = vocabulary[nextTokenIndex];

  if (!nextToken) {
    throw new Error(`Failed to find token at index ${nextTokenIndex}`);
  }

  return nextToken;
};
