import { describe, it, expect } from "bun:test";
import { initializeModel } from "../model/model-initialize.ts";
import {
  loadWeightsIntoGpu,
  extractWeightsFromGpu,
} from "../model-gpu/model-weights-gpu.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import { addMatrices } from "../shared/matrices/matrices.ts";
import { expectMatrixCloseTo } from "../testing/testing-utils.ts";
import type { Model, TransformerWeights, Weights } from "../model/model-types.ts";
import { addWeights } from "./addWeights.ts";

const testModel = (name: string): Model =>
  initializeModel({
    name,
    headsCount: 2,
    hiddenDimensionCount: 8,
    transformerCount: 3,
    vocabulary: ["a", "b", "c", "d"],
  });

const flush = async (): Promise<void> => {
  await gpuContext.device.queue.onSubmittedWorkDone();
};

const expectTransformerIsSum = (
  actual: TransformerWeights,
  base: TransformerWeights,
  adjustments: TransformerWeights,
) => {
  expectMatrixCloseTo(
    actual.attention.Q,
    addMatrices(base.attention.Q, adjustments.attention.Q),
    4,
  );
  expectMatrixCloseTo(
    actual.attention.K,
    addMatrices(base.attention.K, adjustments.attention.K),
    4,
  );
  expectMatrixCloseTo(
    actual.attention.V,
    addMatrices(base.attention.V, adjustments.attention.V),
    4,
  );
  expectMatrixCloseTo(
    actual.attention.out,
    addMatrices(base.attention.out, adjustments.attention.out),
    4,
  );

  expectMatrixCloseTo(
    actual.multilayerPerceptron.wUp.weightsMatrix,
    addMatrices(
      base.multilayerPerceptron.wUp.weightsMatrix,
      adjustments.multilayerPerceptron.wUp.weightsMatrix,
    ),
    4,
  );
  expectMatrixCloseTo(
    actual.multilayerPerceptron.wUp.biasVector,
    addMatrices(
      base.multilayerPerceptron.wUp.biasVector,
      adjustments.multilayerPerceptron.wUp.biasVector,
    ),
    4,
  );
  expectMatrixCloseTo(
    actual.multilayerPerceptron.wDown.weightsMatrix,
    addMatrices(
      base.multilayerPerceptron.wDown.weightsMatrix,
      adjustments.multilayerPerceptron.wDown.weightsMatrix,
    ),
    4,
  );
  expectMatrixCloseTo(
    actual.multilayerPerceptron.wDown.biasVector,
    addMatrices(
      base.multilayerPerceptron.wDown.biasVector,
      adjustments.multilayerPerceptron.wDown.biasVector,
    ),
    4,
  );
};

const expectIsSum = (
  actual: Weights,
  base: Weights,
  adjustments: Weights,
) => {
  expectMatrixCloseTo(
    actual.embeddings,
    addMatrices(base.embeddings, adjustments.embeddings),
    4,
  );
  expectMatrixCloseTo(
    actual.unembeddings,
    addMatrices(base.unembeddings, adjustments.unembeddings),
    4,
  );

  expect(actual.transformers.length).toBe(base.transformers.length);
  for (let i = 0; i < base.transformers.length; i++) {
    expectTransformerIsSum(
      actual.transformers[i]!,
      base.transformers[i]!,
      adjustments.transformers[i]!,
    );
  }
};

describe("addWeights", () => {
  it("adds the adjustments into the base weights in place", async () => {
    const baseBuffers = loadWeightsIntoGpu(testModel("base"));
    const adjustmentBuffers = loadWeightsIntoGpu(testModel("adjustments"));

    const base = await extractWeightsFromGpu(baseBuffers);
    const adjustments = await extractWeightsFromGpu(adjustmentBuffers);

    addWeights(baseBuffers, adjustmentBuffers);
    await flush();

    const result = await extractWeightsFromGpu(baseBuffers);

    expectIsSum(result, base, adjustments);
  });

  it("leaves the adjustment buffers untouched", async () => {
    const baseBuffers = loadWeightsIntoGpu(testModel("base"));
    const adjustmentBuffers = loadWeightsIntoGpu(testModel("adjustments"));

    const adjustmentsBefore = await extractWeightsFromGpu(adjustmentBuffers);

    addWeights(baseBuffers, adjustmentBuffers);
    await flush();

    const adjustmentsAfter = await extractWeightsFromGpu(adjustmentBuffers);

    expectMatrixCloseTo(
      adjustmentsAfter.embeddings,
      adjustmentsBefore.embeddings,
      5,
    );
    expectMatrixCloseTo(
      adjustmentsAfter.unembeddings,
      adjustmentsBefore.unembeddings,
      5,
    );
    for (let i = 0; i < adjustmentsBefore.transformers.length; i++) {
      const before = adjustmentsBefore.transformers[i]!;
      const after = adjustmentsAfter.transformers[i]!;
      expectMatrixCloseTo(after.attention.Q, before.attention.Q, 5);
      expectMatrixCloseTo(after.attention.K, before.attention.K, 5);
      expectMatrixCloseTo(after.attention.V, before.attention.V, 5);
      expectMatrixCloseTo(after.attention.out, before.attention.out, 5);
    }
  });
});
