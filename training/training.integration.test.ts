import { describe, expect, it } from "vitest";
import { findTokenIndex } from "../model/model-helpers.ts";
import type {
  Model,
  TransformerWeights,
  Weights,
} from "../model/model-types.ts";
import { llmForwardPassByTokens } from "../running/llm.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { calculateLoss } from "./calculateLoss.ts";
import { backprop } from "./backprop/backprop.ts";
import { doSingleTrainingPass } from "./training.ts";

const zeroTransformer: TransformerWeights = {
  attention: {
    Q: [
      [0, 0],
      [0, 0],
    ],
    K: [
      [0, 0],
      [0, 0],
    ],
    V: [
      [0, 0],
      [0, 0],
    ],
    out: [
      [0, 0],
      [0, 0],
    ],
  },
  multilayerPerceptron: {
    wUp: {
      weightsMatrix: [
        [0, 0],
        [0, 0],
      ],
      biasVector: [0, 0],
    },
    wDown: {
      weightsMatrix: [
        [0, 0],
        [0, 0],
      ],
      biasVector: [0, 0],
    },
  },
};

const model: Model = {
  vocabulary: ["prompt", "answer", END_OF_SEQUENCE_TOKEN],
  headsCount: 2,
  mlpMultiple: 1,
  embeddings: [
    [0.25, -0.75],
    [0.5, -0.5],
    [0.75, 0.25],
  ],
  unembeddings: [
    [0, 0, 0],
    [0, 0, 0],
  ],
  transformers: [zeroTransformer],
};

const flattenWeights = (weights: Weights): number[] => [
  ...weights.embeddings.flat(),
  ...weights.unembeddings.flat(),
  ...weights.transformers.flatMap((transformer) => [
    ...transformer.attention.Q.flat(),
    ...transformer.attention.K.flat(),
    ...transformer.attention.V.flat(),
    ...transformer.attention.out.flat(),
    ...transformer.multilayerPerceptron.wUp.weightsMatrix.flat(),
    ...transformer.multilayerPerceptron.wUp.biasVector,
    ...transformer.multilayerPerceptron.wDown.weightsMatrix.flat(),
    ...transformer.multilayerPerceptron.wDown.biasVector,
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
  const finalLogits = logitsByPosition[logitsByPosition.length - 1];

  if (!finalLogits) {
    throw new Error("Expected the forward pass to return final logits");
  }

  return calculateLoss(
    finalLogits,
    findTokenIndex(currentModel.vocabulary, targetToken),
    currentModel.vocabulary,
  );
};

describe("training/backprop integration readiness", () => {
  it("keeps gradients and adjusted weights finite, then nudges the trained next-token objective downhill", () => {
    const prompt = "prompt";
    const target = "answer";
    const targetIndex = findTokenIndex(model.vocabulary, target);
    const promptOnlyInput = [prompt];
    const trainingSequence = [prompt, target];

    const { activations } = llmForwardPassByTokens(
      promptOnlyInput,
      model,
      true,
    );

    if (!activations) {
      throw new Error("Expected activations for backprop integration check");
    }

    const { loss, gradients } = backprop(model, activations, [targetIndex]);

    expect(Number.isFinite(loss)).toBe(true);
    expectWeightsToBeFinite(gradients);
    expect(flattenWeights(gradients).some((value) => value !== 0)).toBe(true);

    const beforeTargetLoss = lossForNextToken(model, promptOnlyInput, target);
    const { averageLoss, adjustedWeights } = doSingleTrainingPass(model, [
      trainingSequence,
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
