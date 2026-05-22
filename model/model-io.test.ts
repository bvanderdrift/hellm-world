import { describe, expect, it } from "vitest";
import type { ModelMetadata, Weights } from "./model-types.ts";
import { flattenWeights, unwrapFlatWeights } from "./model-io.ts";
import { matrixFrom, expectMatrixCloseTo } from "../testing/testing-utils.ts";

const metadata: ModelMetadata = {
  vocabulary: ["a", "b", "c"],
  counts: {
    transformers: 1,
    attentionHeads: 1,
    hiddenDimensions: 2,
    mlpMultiple: 2,
  },
};

const weights: Weights = {
  embeddings: matrixFrom([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
  ]),
  transformers: [
    {
      attention: {
        K: matrixFrom([
          [1.0, 1.1],
          [1.2, 1.3],
        ]),
        V: matrixFrom([
          [2.0, 2.1],
          [2.2, 2.3],
        ]),
        Q: matrixFrom([
          [3.0, 3.1],
          [3.2, 3.3],
        ]),
        out: matrixFrom([
          [4.0, 4.1],
          [4.2, 4.3],
        ]),
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: matrixFrom([
            [5.0, 5.1, 5.2, 5.3],
            [5.4, 5.5, 5.6, 5.7],
          ]),
          biasVector: matrixFrom([[6.0, 6.1, 6.2, 6.3]]),
        },
        wDown: {
          weightsMatrix: matrixFrom([
            [7.0, 7.1],
            [7.2, 7.3],
            [7.4, 7.5],
            [7.6, 7.7],
          ]),
          biasVector: matrixFrom([[8.0, 8.1]]),
        },
      },
    },
  ],
  unembeddings: matrixFrom([
    [9.0, 9.1, 9.2],
    [9.3, 9.4, 9.5],
  ]),
};

describe("flattenWeights / unwrapFlatWeights roundtrip", () => {
  it("recovers the original weights after flatten then unwrap", () => {
    const flat = flattenWeights(weights);
    const buffer = Buffer.from(flat.buffer);
    const recovered = unwrapFlatWeights(metadata, buffer);

    expectMatrixCloseTo(recovered.embeddings, weights.embeddings);
    expectMatrixCloseTo(recovered.unembeddings, weights.unembeddings);

    expect(recovered.transformers).toHaveLength(1);
    const t = recovered.transformers[0]!;
    const orig = weights.transformers[0]!;

    expectMatrixCloseTo(t.attention.K, orig.attention.K);
    expectMatrixCloseTo(t.attention.V, orig.attention.V);
    expectMatrixCloseTo(t.attention.Q, orig.attention.Q);
    expectMatrixCloseTo(t.attention.out, orig.attention.out);

    expectMatrixCloseTo(
      t.multilayerPerceptron.wUp.weightsMatrix,
      orig.multilayerPerceptron.wUp.weightsMatrix,
    );
    expectMatrixCloseTo(
      t.multilayerPerceptron.wUp.biasVector,
      orig.multilayerPerceptron.wUp.biasVector,
    );
    expectMatrixCloseTo(
      t.multilayerPerceptron.wDown.weightsMatrix,
      orig.multilayerPerceptron.wDown.weightsMatrix,
    );
    expectMatrixCloseTo(
      t.multilayerPerceptron.wDown.biasVector,
      orig.multilayerPerceptron.wDown.biasVector,
    );
  });

  it("preserves matrix shapes through the roundtrip", () => {
    const flat = flattenWeights(weights);
    const buffer = Buffer.from(flat.buffer);
    const recovered = unwrapFlatWeights(metadata, buffer);

    expect(recovered.embeddings.vectors).toBe(3);
    expect(recovered.embeddings.dimensions).toBe(2);

    expect(recovered.unembeddings.vectors).toBe(2);
    expect(recovered.unembeddings.dimensions).toBe(3);

    const t = recovered.transformers[0]!;
    expect(t.multilayerPerceptron.wUp.weightsMatrix.vectors).toBe(2);
    expect(t.multilayerPerceptron.wUp.weightsMatrix.dimensions).toBe(4);
    expect(t.multilayerPerceptron.wDown.weightsMatrix.vectors).toBe(4);
    expect(t.multilayerPerceptron.wDown.weightsMatrix.dimensions).toBe(2);
  });
});
