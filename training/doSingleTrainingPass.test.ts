import { describe, expect, it } from "vitest";
import type { Model } from "../model/model-types.ts";
import { findTokenIndex } from "../model/model-helpers.ts";
import { llmForwardPassByTokens } from "../running/llm.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { softmax } from "../shared/math.ts";
import { getRawVector } from "../shared/matrices.ts";
import { doSingleTrainingPass } from "./doSingleTrainingPass.ts";
import { matrixFrom } from "../testing/testing-utils.ts";

describe("doSingleTrainingPass", () => {
  it("averages loss over predictions, not raw token count", async () => {
    const model: Model = {
      vocabulary: ["hello", "world", END_OF_SEQUENCE_TOKEN],
      headsCount: 1,
      mlpMultiple: 1,
      embeddings: matrixFrom([[0], [0], [0]]),
      unembeddings: matrixFrom([[0, 0, 0]]),
      transformers: [],
    };

    const { averageLoss } = await doSingleTrainingPass(model, [
      {
        sequence: ["hello", "world", END_OF_SEQUENCE_TOKEN],
        maskBeforeIndex: null,
      },
    ]);

    expect(averageLoss).toBeCloseTo(Math.log(model.vocabulary.length), 5);
  });

  it("uses each context position to predict the following token", async () => {
    const model: Model = {
      vocabulary: ["alpha", "beta", END_OF_SEQUENCE_TOKEN],
      headsCount: 1,
      mlpMultiple: 1,
      embeddings: matrixFrom([
        [1.5, -0.25],
        [-0.5, 1.25],
        [0.75, 0.5],
      ]),
      unembeddings: matrixFrom([
        [1.2, -0.7, 0.3],
        [-0.4, 0.9, -1.1],
      ]),
      transformers: [],
    };
    const sequence = ["alpha", "beta", END_OF_SEQUENCE_TOKEN];

    const { embeddings: logitsByPosition } = llmForwardPassByTokens(
      sequence,
      model,
      false,
    );

    const logitsRow0 = getRawVector(logitsByPosition, 0);
    const logitsRow1 = getRawVector(logitsByPosition, 1);

    const expectedAverageLoss =
      (-Math.log(
        softmax(logitsRow0)[findTokenIndex(model.vocabulary, "beta")]!,
      ) +
        -Math.log(
          softmax(logitsRow1)[
            findTokenIndex(model.vocabulary, END_OF_SEQUENCE_TOKEN)
          ]!,
        )) /
      2;

    const { averageLoss } = await doSingleTrainingPass(model, [
      { sequence, maskBeforeIndex: null },
    ]);

    expect(averageLoss).toBeCloseTo(expectedAverageLoss, 5);
  });

  it("does not update the target token embedding when it is not in the context", async () => {
    const model: Model = {
      vocabulary: ["alpha", "beta"],
      headsCount: 1,
      mlpMultiple: 1,
      embeddings: matrixFrom([
        [1.5, -0.25],
        [-0.5, 1.25],
      ]),
      unembeddings: matrixFrom([
        [1.2, -0.7],
        [-0.4, 0.9],
      ]),
      transformers: [],
    };
    const betaIndex = findTokenIndex(model.vocabulary, "beta");

    const { adjustedWeights } = await doSingleTrainingPass(model, [
      { sequence: ["alpha", "beta"], maskBeforeIndex: null },
    ]);

    expect(getRawVector(adjustedWeights.embeddings, betaIndex)).toEqual(
      getRawVector(model.embeddings, betaIndex),
    );
  });
});
