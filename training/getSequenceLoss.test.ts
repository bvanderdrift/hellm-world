import { describe, expect, it } from "vitest";
import type { Model } from "../model/model-types.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { getSequenceLoss } from "./getSequenceLoss.ts";
import { matrixFrom } from "../testing/testing-utils.ts";
import { emptyTrainingState as emptyHistory } from "../testing/model-fixtures.ts";

const model: Model = {
  name: "test-model",
  vocabulary: ["hello", "world", END_OF_SEQUENCE_TOKEN],
  trainingState: emptyHistory,
  counts: {
    attentionHeads: 1,
    mlpMultiple: 1,
    transformers: 0,
    hiddenDimensions: 1,
  },
  embeddings: matrixFrom([[0.5], [-0.3], [0.1]]),
  unembeddings: matrixFrom([[0.2, -0.4, 0.6]]),
  transformers: [],
};

describe("getSequenceLoss", () => {
  it("includes a loss entry for every position including non-predictive ones", () => {
    const { outputLosses } = getSequenceLoss(
      {
        sequence: ["hello", "world", END_OF_SEQUENCE_TOKEN],
        maskBeforeIndex: null,
      },
      model,
    );

    expect(outputLosses).toHaveLength(3);
  });

  it("sets loss to 0 for the last position which has no next token", () => {
    const { outputLosses } = getSequenceLoss(
      {
        sequence: ["hello", "world", END_OF_SEQUENCE_TOKEN],
        maskBeforeIndex: null,
      },
      model,
    );

    expect(outputLosses[2]).toBe(0);
  });

  it("sets loss to 0 for masked positions", () => {
    const { outputLosses } = getSequenceLoss(
      {
        sequence: ["hello", "world", END_OF_SEQUENCE_TOKEN],
        maskBeforeIndex: 1,
      },
      model,
    );

    expect(outputLosses[0]).toBe(0);
    expect(outputLosses[1]).toBeGreaterThan(0);
  });

  it("returns unmaskedTokenCount excluding masked and non-predictive positions", () => {
    const { unmaskedTokenCount } = getSequenceLoss(
      {
        sequence: ["hello", "world", END_OF_SEQUENCE_TOKEN],
        maskBeforeIndex: null,
      },
      model,
    );

    expect(unmaskedTokenCount).toBe(2);
  });
});
