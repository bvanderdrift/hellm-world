import { describe, expect, it } from "vitest";
import { findTokenIndex } from "../model/model-helpers.ts";
import type {
  Model,
  TransformerWeights,
  Weights,
} from "../model/model-types.ts";
import { llmForwardPassByTokens } from "../running/llm.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { softmax } from "../shared/math.ts";
import { backprop } from "./backprop/backprop.ts";
import { doSingleTrainingPass } from "./doSingleTrainingPass.ts";
import { getSequenceLoss } from "./getSequenceLoss.ts";
import { matrixFrom } from "../testing/testing-utils.ts";
import { getRawVector } from "../shared/matrices.ts";

const zeroTransformer: TransformerWeights = {
  attention: {
    Q: matrixFrom([
      [0, 0],
      [0, 0],
    ]),
    K: matrixFrom([
      [0, 0],
      [0, 0],
    ]),
    V: matrixFrom([
      [0, 0],
      [0, 0],
    ]),
    out: matrixFrom([
      [0, 0],
      [0, 0],
    ]),
  },
  multilayerPerceptron: {
    wUp: {
      weightsMatrix: matrixFrom([
        [0, 0],
        [0, 0],
      ]),
      biasVector: matrixFrom([[0, 0]]),
    },
    wDown: {
      weightsMatrix: matrixFrom([
        [0, 0],
        [0, 0],
      ]),
      biasVector: matrixFrom([[0, 0]]),
    },
  },
};

const model: Model = {
  vocabulary: ["prompt", "answer", END_OF_SEQUENCE_TOKEN],
  headsCount: 2,
  mlpMultiple: 1,
  embeddings: matrixFrom([
    [0.25, -0.75],
    [0.5, -0.5],
    [0.75, 0.25],
  ]),
  unembeddings: matrixFrom([
    [0, 0, 0],
    [0, 0, 0],
  ]),
  transformers: [zeroTransformer],
};

const flattenWeights = (weights: Weights): number[] => [
  ...weights.embeddings.values,
  ...weights.unembeddings.values,
  ...weights.transformers.flatMap((transformer) => [
    ...transformer.attention.Q.values,
    ...transformer.attention.K.values,
    ...transformer.attention.V.values,
    ...transformer.attention.out.values,
    ...transformer.multilayerPerceptron.wUp.weightsMatrix.values,
    ...transformer.multilayerPerceptron.wUp.biasVector.values,
    ...transformer.multilayerPerceptron.wDown.weightsMatrix.values,
    ...transformer.multilayerPerceptron.wDown.biasVector.values,
  ]),
];

const expectWeightsToBeFinite = (weights: Weights) => {
  for (const value of flattenWeights(weights)) {
    expect(Number.isFinite(value)).toBe(true);
  }
};

const expectAnyWeightChanged = (before: Weights, after: Weights) => {
  const beforeValues = flattenWeights(before);
  const afterValues = flattenWeights(after);

  expect(afterValues).toHaveLength(beforeValues.length);
  expect(
    afterValues.some((afterValue, index) => afterValue !== beforeValues[index]),
  ).toBe(true);
};

const lossForNextToken = (
  currentModel: Model,
  inputTokens: string[],
  targetToken: string,
) => {
  const { embeddings: logitsByPosition } = llmForwardPassByTokens(
    inputTokens,
    currentModel,
    false,
  );
  const finalLogits = getRawVector(
    logitsByPosition,
    logitsByPosition.vectors - 1,
  );

  if (!finalLogits) {
    throw new Error("Expected the forward pass to return final logits");
  }

  return -Math.log(
    softmax(finalLogits)[findTokenIndex(currentModel.vocabulary, targetToken)]!,
  );
};

describe("training/backprop integration readiness", () => {
  it("keeps gradients and adjusted weights finite, then nudges the trained next-token objective downhill", async () => {
    const prompt = "prompt";
    const target = "answer";
    const promptOnlyInput = [prompt];
    const trainingSequence = [prompt, target];

    const {
      activations,
      correctTokenIndices,
      outputProbabilities,
      outputLosses,
    } = getSequenceLoss(
      { sequence: trainingSequence, maskBeforeIndex: null },
      model,
    );

    const loss = outputLosses.reduce((sum, loss) => sum + loss, 0);
    const gradients = backprop(
      model,
      activations,
      correctTokenIndices,
      outputProbabilities,
    );

    expect(Number.isFinite(loss)).toBe(true);
    expectWeightsToBeFinite(gradients);
    expect(flattenWeights(gradients).some((value) => value !== 0)).toBe(true);

    const beforeTargetLoss = lossForNextToken(model, promptOnlyInput, target);
    const { averageLoss, adjustedWeights } = await doSingleTrainingPass(model, [
      { sequence: trainingSequence, maskBeforeIndex: null },
    ]);
    const trainedModel: Model = { ...model, ...adjustedWeights };
    const afterTargetLoss = lossForNextToken(
      trainedModel,
      promptOnlyInput,
      target,
    );

    expect(Number.isFinite(averageLoss)).toBe(true);
    expectWeightsToBeFinite(adjustedWeights);
    expectAnyWeightChanged(model, adjustedWeights);
    expect(afterTargetLoss).toBeLessThan(beforeTargetLoss);
  });
});
