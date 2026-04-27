import { describe, expect, it } from "vitest";
import type { Model } from "../model/model-types.ts";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
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
});
