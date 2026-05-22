import type { ModelCheckpoint } from "../model/model-types.ts";
import { matrixFrom } from "../testing/testing-utils.ts";

const migrateOldCheckpoint = (old: ModelCheckpoint): ModelCheckpoint => ({
  history: old.history,
  weights: {
    embeddings: matrixFrom(old.weights.embeddings as unknown as number[][]),
    transformers: old.weights.transformers.map((t) => ({
      multilayerPerceptron: {
        wDown: {
          weightsMatrix: matrixFrom(
            t.multilayerPerceptron.wDown.weightsMatrix as unknown as number[][],
          ),
          biasVector: matrixFrom([
            t.multilayerPerceptron.wDown.biasVector as unknown as number[],
          ]),
        },
        wUp: {
          weightsMatrix: matrixFrom(
            t.multilayerPerceptron.wUp.weightsMatrix as unknown as number[][],
          ),
          biasVector: matrixFrom([
            t.multilayerPerceptron.wUp.biasVector as unknown as number[],
          ]),
        },
      },
      attention: {
        K: matrixFrom(t.attention.K as unknown as number[][]),
        V: matrixFrom(t.attention.V as unknown as number[][]),
        Q: matrixFrom(t.attention.Q as unknown as number[][]),
        out: matrixFrom(t.attention.out as unknown as number[][]),
      },
    })),
    unembeddings: matrixFrom(old.weights.unembeddings as unknown as number[][]),
  },
});