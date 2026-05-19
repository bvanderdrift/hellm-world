import { describe, expect, it } from "vitest";
import type { Activations } from "../../model/activations-types.ts";
import type { Model } from "../../model/model-types.ts";
import { safeSumExponatedLogits, softmax, sum } from "../../shared/math.ts";
import type { Matrix } from "../../shared/matrices.ts";
import {
  createMatrix,
  getFlatIndex,
  getRawVector,
} from "../../shared/matrices.ts";
import {
  expectMatrixCloseTo,
  matrixFrom,
} from "../../testing/testing-utils.ts";
import { backprop } from "./backprop.ts";

describe("backprop", () => {
  it("uses every trained position for loss and unembedding gradients", () => {
    const model: Model = {
      vocabulary: ["alpha", "beta", "gamma", "delta"],
      headsCount: 1,
      mlpMultiple: 1,
      embeddings: matrixFrom([
        [0.5, -0.25, 0.75],
        [0.1, 0.2, -0.3],
        [0.4, 0.5, 0.6],
        [-0.7, 0.8, -0.9],
      ]),
      unembeddings: matrixFrom([
        [0.2, -0.4, 0.6, -0.8],
        [1, -1.2, 1.4, -1.6],
        [-0.3, 0.5, -0.7, 0.9],
      ]),
      transformers: [],
    };
    const activations: Activations = {
      inputPositionToVocabPosition: [0, 1],
      tokensToPosition: matrixFrom([
        [0, 0, 0],
        [0, 0, 0],
      ]),
      positionToTransformers: matrixFrom([
        [0, 0, 0],
        [0, 0, 0],
      ]),
      transformerActivations: [],
      transformersToNormalizer: matrixFrom([
        [100, -100, 50],
        [2, -1, 4],
      ]),
      normalizerToUnembeddings: matrixFrom([
        [99, 88, 77],
        [0.25, -0.5, 1.25],
      ]),
      unembeddingsOutputLogits: matrixFrom([
        [8, -4, 3, 7],
        [0.7, -1.1, 2.2, -0.4],
      ]),
    };
    const correctTokenIndices = [1, 3];

    const tokenCount = activations.unembeddingsOutputLogits.vectors;
    const vocabSize = activations.unembeddingsOutputLogits.dimensions;
    const outputProbabilities: Matrix = createMatrix(tokenCount, vocabSize);
    const outputLosses: number[] = [];
    for (let t = 0; t < tokenCount; t++) {
      const logits = getRawVector(activations.unembeddingsOutputLogits, t);
      const probs = softmax(logits);
      outputProbabilities.values.set(probs, t * vocabSize);
      outputLosses.push(-Math.log(probs[correctTokenIndices[t]!]!));
    }

    const gradients = backprop(
      model,
      activations,
      correctTokenIndices,
      outputProbabilities,
    );

    const outputGradients: number[][] = [];
    for (let t = 0; t < tokenCount; t++) {
      const logits = getRawVector(activations.unembeddingsOutputLogits, t);
      const probs = softmax(logits);
      outputGradients.push(
        Array.from(probs).map(
          (p, v) => p - (v === correctTokenIndices[t] ? 1 : 0),
        ),
      );
    }

    const inDims = model.unembeddings.vectors;
    const outDims = model.unembeddings.dimensions;
    const expectedUnembeddingGradients = createMatrix(inDims, outDims);
    for (let i = 0; i < inDims; i++) {
      for (let j = 0; j < outDims; j++) {
        let acc = 0;
        for (let t = 0; t < tokenCount; t++) {
          acc +=
            activations.normalizerToUnembeddings.values[
              getFlatIndex(t, i, inDims)
            ]! * outputGradients[t]![j]!;
        }
        expectedUnembeddingGradients.values[getFlatIndex(i, j, outDims)] = acc;
      }
    }

    const losses = new Float32Array(tokenCount);
    for (let t = 0; t < tokenCount; t++) {
      const logits = getRawVector(activations.unembeddingsOutputLogits, t);
      const probs = softmax(logits);
      losses[t] = -Math.log(probs[correctTokenIndices[t]!]!);
    }
    const expectedLoss = sum(losses);

    const actualLoss = sum(new Float32Array(outputLosses));
    expect(actualLoss).toBeCloseTo(expectedLoss, 5);
    expectMatrixCloseTo(
      gradients.unembeddings,
      expectedUnembeddingGradients,
    );

    expect(gradients.transformers).toEqual([]);
    expect(gradients.embeddings.vectors).toBe(model.embeddings.vectors);
    expect(gradients.embeddings.dimensions).toBe(model.embeddings.dimensions);

    const embRow2 = getRawVector(gradients.embeddings, 2);
    const embRow3 = getRawVector(gradients.embeddings, 3);
    expect(Array.from(embRow2)).toEqual([0, 0, 0]);
    expect(Array.from(embRow3)).toEqual([0, 0, 0]);

    const embRow0 = getRawVector(gradients.embeddings, 0);
    const embRow1 = getRawVector(gradients.embeddings, 1);
    expect(embRow0.some((value) => Math.abs(value) > 1e-6)).toBe(true);
    expect(embRow1.some((value) => Math.abs(value) > 1e-6)).toBe(true);
  });

  it("stays finite when the correct token logit is far below the dominant logit", () => {
    const model: Model = {
      vocabulary: ["dominant", "tiny"],
      headsCount: 1,
      mlpMultiple: 1,
      embeddings: matrixFrom([
        [1, 0],
        [0, 1],
      ]),
      unembeddings: matrixFrom([
        [1, 0],
        [0, 1],
      ]),
      transformers: [],
    };
    const activations: Activations = {
      inputPositionToVocabPosition: [0],
      tokensToPosition: matrixFrom([[1, 0]]),
      positionToTransformers: matrixFrom([[1, 0]]),
      transformerActivations: [],
      transformersToNormalizer: matrixFrom([[1, 0]]),
      normalizerToUnembeddings: matrixFrom([[1, 0]]),
      unembeddingsOutputLogits: matrixFrom([[0, -1000]]),
    };

    const correctTokenIndices = [1];
    const tokenCount = activations.unembeddingsOutputLogits.vectors;
    const vocabSize = activations.unembeddingsOutputLogits.dimensions;
    const outputProbabilities: Matrix = createMatrix(tokenCount, vocabSize);
    const outputLosses: number[] = [];
    for (let t = 0; t < tokenCount; t++) {
      const logits = getRawVector(activations.unembeddingsOutputLogits, t);
      const correctTokenIndex = correctTokenIndices[t]!;
      const { summed, safeLogits, biggestLogit } =
        safeSumExponatedLogits(logits);
      const probabilities = Array.from(safeLogits).map(
        (logit) => Math.exp(logit) / summed,
      );
      outputProbabilities.values.set(probabilities, t * vocabSize);
      const correctTokenLogitAdjusted =
        logits[correctTokenIndex]! - biggestLogit;
      outputLosses.push(Math.log(summed) - correctTokenLogitAdjusted);
    }

    const loss = sum(new Float32Array(outputLosses));

    backprop(model, activations, correctTokenIndices, outputProbabilities);

    expect(Number.isFinite(loss)).toBe(true);
    expect(loss).toBeCloseTo(1000, 0);
  });
});
