import { describe, expect, it } from "vitest";
import type { Model } from "../model/model-types.ts";
import { findTokenIndex } from "../model/model-helpers.ts";
import { llmForwardPassByTokens } from "../running/llm.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { calculateLoss } from "./calculateLoss.ts";
import { doSingleTrainingPass } from "./training.ts";

describe("doSingleTrainingPass", () => {
  it("averages loss over predictions, not raw token count", () => {
    const model: Model = {
      vocabulary: ["hello", "world", END_OF_SEQUENCE_TOKEN],
      headsCount: 1,
      mlpMultiple: 1,
      embeddings: [[0], [0], [0]],
      unembeddings: [[0, 0, 0]],
      transformers: [],
    };

    const { averageLoss } = doSingleTrainingPass(model, [
      ["hello", "world", END_OF_SEQUENCE_TOKEN],
    ]);

    expect(averageLoss).toBeCloseTo(Math.log(model.vocabulary.length), 10);
  });

  it("uses each context position to predict the following token", () => {
    const model: Model = {
      vocabulary: ["alpha", "beta", END_OF_SEQUENCE_TOKEN],
      headsCount: 1,
      mlpMultiple: 1,
      embeddings: [
        [1.5, -0.25],
        [-0.5, 1.25],
        [0.75, 0.5],
      ],
      unembeddings: [
        [1.2, -0.7, 0.3],
        [-0.4, 0.9, -1.1],
      ],
      transformers: [],
    };
    const sequence = ["alpha", "beta", END_OF_SEQUENCE_TOKEN];

    const { embeddings: logitsByPosition } = llmForwardPassByTokens(
      sequence,
      model,
      false,
    );

    const expectedAverageLoss =
      (calculateLoss(
        logitsByPosition[0]!,
        findTokenIndex(model.vocabulary, "beta"),
        model.vocabulary,
      ) +
        calculateLoss(
          logitsByPosition[1]!,
          findTokenIndex(model.vocabulary, END_OF_SEQUENCE_TOKEN),
          model.vocabulary,
        )) /
      2;

    const { averageLoss } = doSingleTrainingPass(model, [sequence]);

    expect(averageLoss).toBeCloseTo(expectedAverageLoss, 10);
  });

  it("does not update the target token embedding when it is not in the context", () => {
    const model: Model = {
      vocabulary: ["alpha", "beta"],
      headsCount: 1,
      mlpMultiple: 1,
      embeddings: [
        [1.5, -0.25],
        [-0.5, 1.25],
      ],
      unembeddings: [
        [1.2, -0.7],
        [-0.4, 0.9],
      ],
      transformers: [],
    };
    const betaIndex = findTokenIndex(model.vocabulary, "beta");

    const { adjustedWeights } = doSingleTrainingPass(model, [
      ["alpha", "beta"],
    ]);

    expect(adjustedWeights.embeddings[betaIndex]).toEqual(
      model.embeddings[betaIndex],
    );
  });
});
