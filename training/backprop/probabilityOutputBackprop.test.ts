import { describe, expect, it } from "vitest";
import { softmax } from "../../shared/math.ts";
import { calculateLoss } from "../calculateLoss.ts";
import { probabilityOutputBackprop } from "./probabilityOutputBackprop.ts";

describe("probabilityOutputBackprop", () => {
  it("places cross-entropy logit gradients on every trained context position", () => {
    const logits = [
      [9, 8, 7, 6],
      [3, 2, 1, 0],
      [0.7, -1.1, 2.2, -0.4],
    ];
    const outputProbabilities = logits.map((outputLogits) =>
      softmax(outputLogits),
    );
    const correctTokenIndices = [1, 2, 3];

    const gradients = probabilityOutputBackprop(
      logits,
      outputProbabilities,
      correctTokenIndices,
    );

    expect(gradients[0]).toEqual(
      outputProbabilities[0]!.map(
        (probability, vocabIndex) =>
          probability - (vocabIndex === correctTokenIndices[0] ? 1 : 0),
      ),
    );
    expect(gradients[1]).toEqual(
      outputProbabilities[1]!.map(
        (probability, vocabIndex) =>
          probability - (vocabIndex === correctTokenIndices[1] ? 1 : 0),
      ),
    );
    expect(gradients[2]).toEqual(
      outputProbabilities[2]!.map(
        (probability, vocabIndex) =>
          probability - (vocabIndex === correctTokenIndices[2] ? 1 : 0),
      ),
    );
  });

  it("treats the correct token index as a vocabulary index", () => {
    const logits = [
      [0, 0, 0, 0],
      [0.7, -1.1, 2.2, -0.4],
    ];
    const outputProbabilities = logits.map((outputLogits) =>
      softmax(outputLogits),
    );
    const correctTokenIndex = 3;

    const gradients = probabilityOutputBackprop(
      logits,
      outputProbabilities,
      [0, correctTokenIndex],
    );

    expect(gradients[1]![1]).toBeCloseTo(outputProbabilities[1]![1]!, 10);
    expect(gradients[1]![3]).toBeCloseTo(
      outputProbabilities[1]![3]! - 1,
      10,
    );
  });

  it("matches finite differences of calculateLoss for trained-position logits", () => {
    const logits = [
      [10, -10, 5],
      [1.2, -0.7, 0.3],
    ];
    const vocabulary = ["alpha", "beta", "gamma"];
    const correctTokenIndex = 2;
    const outputProbabilities = logits.map((outputLogits) =>
      softmax(outputLogits),
    );
    const epsilon = 0.000001;

    const gradients = probabilityOutputBackprop(
      logits,
      outputProbabilities,
      [0, correctTokenIndex],
    );

    for (const [vocabIndex] of logits[1]!.entries()) {
      const increasedLogits = [...logits[1]!];
      const decreasedLogits = [...logits[1]!];

      increasedLogits[vocabIndex]! += epsilon;
      decreasedLogits[vocabIndex]! -= epsilon;

      const numericalGradient =
        (calculateLoss(increasedLogits, correctTokenIndex, vocabulary) -
          calculateLoss(decreasedLogits, correctTokenIndex, vocabulary)) /
        (2 * epsilon);

      expect(gradients[1]![vocabIndex]).toBeCloseTo(numericalGradient, 5);
    }
  });
});
