import { describe, expect, it } from "vitest";
import type { Activations } from "../../model/activations-types.ts";
import type { Model } from "../../model/model-types.ts";
import { softmax } from "../../shared/math.ts";
import { calculateLoss } from "../calculateLoss.ts";
import { backprop } from "./backprop.ts";

const expectMatrixToBeCloseTo = (actual: number[][], expected: number[][]) => {
  expect(actual).toHaveLength(expected.length);

  for (const [rowIndex, expectedRow] of expected.entries()) {
    expect(actual[rowIndex]).toHaveLength(expectedRow.length);

    for (const [columnIndex, expectedValue] of expectedRow.entries()) {
      expect(actual[rowIndex]![columnIndex]).toBeCloseTo(expectedValue, 10);
    }
  }
};

describe("backprop", () => {
  it("uses only the final position for loss and unembedding gradients", () => {
    const model: Model = {
      vocabulary: ["alpha", "beta", "gamma", "delta"],
      headsCount: 1,
      mlpMultiple: 1,
      embeddings: [
        [0.5, -0.25, 0.75],
        [0.1, 0.2, -0.3],
        [0.4, 0.5, 0.6],
        [-0.7, 0.8, -0.9],
      ],
      unembeddings: [
        [0.2, -0.4, 0.6, -0.8],
        [1, -1.2, 1.4, -1.6],
        [-0.3, 0.5, -0.7, 0.9],
      ],
      transformers: [],
    };
    const activations: Activations = {
      inputPositionToVocabPosition: [0, 1],
      tokensToPosition: [
        [0, 0, 0],
        [0, 0, 0],
      ],
      positionToTransformers: [
        [0, 0, 0],
        [0, 0, 0],
      ],
      transformerActivations: [],
      transformersToNormalizer: [
        [100, -100, 50],
        [2, -1, 4],
      ],
      normalizerToUnembeddings: [
        [99, 88, 77],
        [0.25, -0.5, 1.25],
      ],
      unembeddingsOutputLogits: [
        [8, -4, 3, 7],
        [0.7, -1.1, 2.2, -0.4],
      ],
    };
    const correctTokenIndex = 3;

    const { loss, gradients } = backprop(
      ["alpha", "beta"],
      model,
      activations,
      correctTokenIndex,
    );

    const trainedPositionProbabilities = softmax(
      activations.unembeddingsOutputLogits[1]!,
    );
    const outputGradients = trainedPositionProbabilities.map(
      (probability, vocabIndex) =>
        probability - (vocabIndex === correctTokenIndex ? 1 : 0),
    );
    const trainedActivation = activations.normalizerToUnembeddings[1]!;
    const expectedUnembeddingGradients = model.unembeddings.map(
      (weightsForIncomingDimension, incomingDimension) =>
        weightsForIncomingDimension.map(
          (_, outgoingDimension) =>
            trainedActivation[incomingDimension]! *
            outputGradients[outgoingDimension]!,
        ),
    );

    expect(loss).toBeCloseTo(
      calculateLoss(
        activations.unembeddingsOutputLogits[1]!,
        correctTokenIndex,
        model.vocabulary,
      ),
      10,
    );
    expectMatrixToBeCloseTo(
      gradients.unembeddings,
      expectedUnembeddingGradients,
    );

    expect(gradients.transformers).toEqual([]);
    expect(gradients.embeddings).toHaveLength(model.embeddings.length);
    for (const [rowIndex, row] of model.embeddings.entries()) {
      expect(gradients.embeddings[rowIndex]).toHaveLength(row.length);
    }

    expect(gradients.embeddings[0]).toEqual([0, 0, 0]);
    expect(gradients.embeddings[2]).toEqual([0, 0, 0]);
    expect(gradients.embeddings[3]).toEqual([0, 0, 0]);
    expect(
      gradients.embeddings[1]!.some((value) => Math.abs(value) > 1e-12),
    ).toBe(true);
  });
});
