import { describe, it, expect } from "bun:test";
import { initializeModel } from "../model/model-initialize.ts";
import {
  loadWeightsIntoGpu,
  extractWeightsFromGpu,
} from "./model-weights-gpu.ts";
import { expectMatrixCloseTo } from "../testing/testing-utils.ts";
import type { Model, TransformerWeights } from "../model/model-types.ts";

const testModel = (): Model =>
  initializeModel({
    name: "test-model",
    headsCount: 2,
    hiddenDimensionCount: 8,
    transformerCount: 3,
    vocabulary: ["a", "b", "c", "d"],
  });

const expectTransformerCloseTo = (
  actual: TransformerWeights,
  expected: TransformerWeights,
) => {
  expectMatrixCloseTo(actual.attention.Q, expected.attention.Q, 5);
  expectMatrixCloseTo(actual.attention.K, expected.attention.K, 5);
  expectMatrixCloseTo(actual.attention.V, expected.attention.V, 5);
  expectMatrixCloseTo(actual.attention.out, expected.attention.out, 5);

  expectMatrixCloseTo(
    actual.multilayerPerceptron.wUp.weightsMatrix,
    expected.multilayerPerceptron.wUp.weightsMatrix,
    5,
  );
  expectMatrixCloseTo(
    actual.multilayerPerceptron.wUp.biasVector,
    expected.multilayerPerceptron.wUp.biasVector,
    5,
  );
  expectMatrixCloseTo(
    actual.multilayerPerceptron.wDown.weightsMatrix,
    expected.multilayerPerceptron.wDown.weightsMatrix,
    5,
  );
  expectMatrixCloseTo(
    actual.multilayerPerceptron.wDown.biasVector,
    expected.multilayerPerceptron.wDown.biasVector,
    5,
  );
};

describe("loadWeightsIntoGpu / extractWeightsFromGpu", () => {
  it("round-trips the embeddings and unembeddings unchanged", async () => {
    const model = testModel();

    const buffers = loadWeightsIntoGpu(model.counts, model);
    const result = await extractWeightsFromGpu(buffers);

    expectMatrixCloseTo(result.embeddings, model.embeddings, 5);
    expectMatrixCloseTo(result.unembeddings, model.unembeddings, 5);
  });

  it("round-trips every transformer's weights unchanged", async () => {
    const model = testModel();

    const buffers = loadWeightsIntoGpu(model.counts, model);
    const result = await extractWeightsFromGpu(buffers);

    expect(result.transformers.length).toBe(model.transformers.length);

    for (let i = 0; i < model.transformers.length; i++) {
      expectTransformerCloseTo(result.transformers[i]!, model.transformers[i]!);
    }
  });
});
